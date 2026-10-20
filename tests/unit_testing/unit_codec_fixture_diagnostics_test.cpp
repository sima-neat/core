#include "../perf/codec_fixture_diagnostics.h"

#include <iostream>
#include <unistd.h>

namespace diagnostics = sima_codec_diagnostics;

namespace {
void require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}

struct Frame {
  std::vector<std::uint8_t> bytes;
  std::string caps;
};

// Independent copy of the accepted length-prefixed AU serialization. Diagnostics
// may describe caps, but must not normalize them or change any fingerprint byte.
std::string fingerprint(const std::vector<Frame>& frames) {
  auto sum = diagnostics::checksum();
  const auto update = [&](const void* data, std::size_t size) {
    const auto length = GUINT64_TO_BE(static_cast<guint64>(size));
    g_checksum_update(sum.get(), reinterpret_cast<const guchar*>(&length), sizeof(length));
    g_checksum_update(sum.get(), static_cast<const guchar*>(data), size);
  };
  for (const auto& frame : frames) {
    update(frame.caps.data(), frame.caps.size());
    update(frame.bytes.data(), frame.bytes.size());
  }
  return g_checksum_get_string(sum.get());
}

struct TemporaryFile {
  std::string path;
  TemporaryFile() {
    gchar* name = nullptr;
    const int fd = g_file_open_tmp("codec-diagnostics-XXXXXX", &name, nullptr);
    require(fd >= 0, "cannot create diagnostic test fixture");
    path = name;
    g_free(name);
    const auto written = write(fd, "abc", 3);
    const int closed = close(fd);
    require(written == 3 && closed == 0, "cannot write diagnostic test fixture");
  }
  ~TemporaryFile() {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
  }
};
} // namespace

int main(int argc, char** argv) {
  try {
    gst_init(&argc, &argv);
    TemporaryFile fixture;
    const auto source = diagnostics::file_identity(fixture.path);
    require(source.at("size") == 3, "file size changed");
    require(source.at("sha256") ==
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "file SHA-256 differs from independent abc oracle");

    std::vector<Frame> frames = {
        {{'a', 'b', 'c'},
         "video/x-h264, width=(int)1280, height=(int)720, framerate=(fraction)30/1, "
         "alignment=(string)au"},
        {{0, 1, 255}, "video/x-h264,stream-format=(string)byte-stream"}};
    // The second sample represents the existing empty-emitted-caps fallback.
    const std::vector<std::string> emitted = {frames[0].caps, ""};
    const auto before = fingerprint(frames);
    require(before == "1bac8040fe386266457830e8e590470c01b455ceda424f5fe58139c9855d48c2",
            "length-prefixed AU fingerprint differs from independent oracle");
    const auto record = diagnostics::access_units(frames, emitted);
    require(fingerprint(frames) == before, "diagnostics changed the input fingerprint");
    require(record.size() == 2 && record[0]["size"] == 3, "incorrect AU boundaries");
    require(record[0]["sha256"] == source["sha256"], "incorrect AU payload hash");
    require(record[0]["emitted_caps"] == emitted[0], "raw caps were normalized");
    require(record[0]["effective_caps"] == frames[0].caps, "effective caps changed");
    require(record[0]["caps_fields"][0]["fields"]["width"]["value"] == "1280",
            "structural caps diagnostics lack width");
    require(record[0]["caps_fields"][0]["fields"]["framerate"]["value"] == "30/1",
            "structural caps diagnostics lack framerate");
    require(record[1]["emitted_caps"] == "" && record[1]["caps_fields"].is_null() &&
                record[1]["effective_caps"] == frames[1].caps,
            "diagnostics hid the existing fallback");
    frames[0].bytes[0] ^= 1;
    require(fingerprint(frames) != before, "payload mutation negative control did not fail");
    frames[0].bytes[0] ^= 1;
    frames[0].caps += ' ';
    require(fingerprint(frames) != before, "caps mutation negative control did not fail");
    bool rejected = false;
    try {
      (void)diagnostics::access_units(frames, std::vector<std::string>{});
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected, "diagnostics accepted mismatched AU/caps counts");

    // Use an actual element/provider, not a registry-only factory lookup. No
    // codec device, model, or accelerator is needed for this diagnostic contract.
    std::unique_ptr<GstElement, decltype(&gst_object_unref)> element(
        gst_element_factory_make("identity", nullptr), gst_object_unref);
    const auto provider = diagnostics::parser_identity(element.get());
    require(provider.at("factory") == "identity", "not the actual element factory");
    require(!provider.at("plugin_version").get<std::string>().empty(), "missing provider version");
    if (!provider.at("file").is_null())
      require(provider.at("file") ==
                  diagnostics::file_identity(provider.at("file").at("path").get<std::string>()),
              "provider file identity mismatch");
    std::cout << "[OK] codec fixture diagnostics preserve exact input bytes and caps\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}

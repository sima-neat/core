#include "PeripheralCatalog.h"
#include "PeripheralCatalogManager.h"
#include "pipeline/ErrorCodes.h"
#include "providers/AlsaProvider.h"
#include "test_main.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using simaai::neat::peripherals_internal::AlsaDiscoveryRoots;
using simaai::neat::peripherals_internal::discover_alsa_capture_peripherals;
using simaai::neat::peripherals_internal::PeripheralCatalog;
using simaai::neat::peripherals_internal::PeripheralCatalogManager;
using simaai::neat::peripherals_internal::PeripheralRecord;

class TempDirectory {
public:
  TempDirectory()
      : path_(fs::temp_directory_path() /
              ("neat_alsa_discovery_" +
               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
    fs::create_directories(path_);
  }

  ~TempDirectory() {
    std::error_code error;
    fs::remove_all(path_, error);
  }

  const fs::path& path() const {
    return path_;
  }

private:
  fs::path path_;
};

void write_file(const fs::path& path, const std::string& contents) {
  fs::create_directories(path.parent_path());
  std::ofstream output(path);
  require(static_cast<bool>(output), "failed to create ALSA fixture " + path.string());
  output << contents;
  require(static_cast<bool>(output), "failed to write ALSA fixture " + path.string());
}

struct Fixture {
  explicit Fixture(const fs::path& path)
      : root(path), roots{root / "proc/asound", root / "sys", root / "dev"} {
    fs::create_directories(roots.proc_asound);
    fs::create_directories(roots.sys / "class/sound");
    fs::create_directories(roots.dev / "snd/by-path");
    fs::create_directories(roots.dev / "snd/by-id");
  }

  void add_usb_card(unsigned card_index, std::string card_id, std::string bus_path,
                    std::string interface, bool optional_strings = true) {
    const fs::path card = roots.proc_asound / ("card" + std::to_string(card_index));
    write_file(card / "id", card_id + "\n");

    const fs::path usb = roots.sys / "devices/pci0000:00/usb1" / bus_path;
    const fs::path usb_interface = usb / interface;
    fs::create_directories(usb_interface);
    write_file(usb / "idVendor", "b58e\n");
    write_file(usb / "idProduct", "0005\n");
    if (optional_strings) {
      write_file(usb / "manufacturer", "Blue\n");
      write_file(usb / "product", "Yeti Nano\n");
      write_file(usb / "serial", "SERIAL-" + bus_path + "\n");
    }

    const fs::path class_card = roots.sys / "class/sound" / ("card" + std::to_string(card_index));
    fs::create_directories(class_card);
    fs::create_directory_symlink(usb_interface, class_card / "device");

    const std::string control = "controlC" + std::to_string(card_index);
    write_file(roots.dev / "snd" / control, "");
    fs::create_symlink("../" + control,
                       roots.dev / "snd/by-path" / ("pci-0000:00:14.0-usb-0:" + bus_path + ":1.0"));
    fs::create_symlink("../" + control,
                       roots.dev / "snd/by-id" / ("usb-Blue_Yeti_Nano-" + bus_path));
  }

  void add_platform_card(unsigned card_index, std::string card_id) {
    const fs::path card = roots.proc_asound / ("card" + std::to_string(card_index));
    write_file(card / "id", card_id + "\n");
    const fs::path device = roots.sys / "devices/platform/audio-codec";
    fs::create_directories(device);
    const fs::path class_card = roots.sys / "class/sound" / ("card" + std::to_string(card_index));
    fs::create_directories(class_card);
    fs::create_directory_symlink(device, class_card / "device");
  }

  void set_usb_attributes_visible(std::string_view bus_path, bool visible) {
    const fs::path usb = roots.sys / "devices/pci0000:00/usb1" / bus_path;
    for (const auto* attribute : {"idVendor", "idProduct"}) {
      const fs::path shown = usb / attribute;
      const fs::path hidden = usb / (std::string(attribute) + ".hidden");
      fs::rename(visible ? hidden : shown, visible ? shown : hidden);
    }
  }

  void set_card_device_visible(unsigned card_index, bool visible) {
    const fs::path card = roots.sys / "class/sound" / ("card" + std::to_string(card_index));
    fs::rename(card / (visible ? "device.hidden" : "device"),
               card / (visible ? "device" : "device.hidden"));
  }

  void add_capture(unsigned card_index, unsigned device, unsigned subdevices, unsigned available,
                   std::string pcm_name, std::string stream = {}) {
    const fs::path card = roots.proc_asound / ("card" + std::to_string(card_index));
    write_file(card / ("pcm" + std::to_string(device) + "c") / "info",
               "card: " + std::to_string(card_index) + "\n" + "device: " + std::to_string(device) +
                   "\n" + "name: " + pcm_name +
                   "\nsubdevices_count: " + std::to_string(subdevices) +
                   "\nsubdevices_avail: " + std::to_string(available) + "\n");
    write_file(roots.dev / "snd" /
                   ("pcmC" + std::to_string(card_index) + "D" + std::to_string(device) + "c"),
               "");
    if (!stream.empty())
      write_file(card / ("stream" + std::to_string(device)), stream);
  }

  void add_playback(unsigned card_index, unsigned device) {
    write_file(roots.proc_asound / ("card" + std::to_string(card_index)) /
                   ("pcm" + std::to_string(device) + "p") / "info",
               "stream: PLAYBACK\n");
  }

  fs::path root;
  AlsaDiscoveryRoots roots;
};

const nlohmann::json* find_mode(const nlohmann::json& modes, std::string_view format,
                                unsigned channels) {
  for (const auto& mode : modes) {
    if (mode.value("format", "") == format && mode.value("channels", 0U) == channels)
      return &mode;
  }
  return nullptr;
}

const PeripheralRecord& find_record(const std::vector<PeripheralRecord>& records,
                                    std::string_view stable_key) {
  for (const auto& record : records) {
    if (record.details["identity"].value("stable_key", "") == stable_key)
      return record;
  }
  throw std::runtime_error("missing ALSA fixture record " + std::string(stable_key));
}

const PeripheralRecord& find_capture(const std::vector<PeripheralRecord>& records,
                                     std::string_view selector) {
  for (const auto& record : records) {
    if (record.details["capture_target"].value("selector", "") == selector)
      return record;
  }
  throw std::runtime_error("missing ALSA fixture capture " + std::string(selector));
}

PeripheralRecord camera_record() {
  return {
      .id = "camera:test-camera",
      .type = "camera",
      .provider = "test.camera",
      .details = {{"camera_name", "test-camera"}, {"backend", "libcamera"}},
  };
}

} // namespace

RUN_TEST(
    "unit_alsa_discovery_test", ([] {
      TempDirectory temp;
      Fixture fixture(temp.path() / "complete");
      write_file(fixture.roots.proc_asound / "cards",
                 " 0 [Codec         ]: platform - Board Codec\n"
                 "                      Board Codec\n"
                 " 2 [Nano          ]: USB-Audio - Yeti Nano\n"
                 "                      Blue Yeti Nano at usb-1-3.2\n"
                 " 7 [Nano_1        ]: USB-Audio - Yeti Nano\n"
                 "                      Blue Yeti Nano at usb-1-3.3\n"
                 " 8 [Output        ]: USB-Audio - Playback Only\n");
      fixture.add_platform_card(0, "Codec");
      fixture.add_usb_card(2, "Nano", "1-3.2", "1-3.2:1.0");
      fixture.add_usb_card(7, "Nano_1", "1-3.3", "1-3.3:1.0", false);
      fixture.add_usb_card(8, "Output", "1-3.4", "1-3.4:1.0");

      fixture.add_capture(2, 0, 2, 1, "USB Audio",
                          "Blue Yeti Nano : USB Audio\n\n"
                          "Playback:\n"
                          " Status: Stop\n"
                          " Interface 2\n"
                          " Altset 1\n"
                          " Format: S16_LE\n"
                          " Channels: 2\n"
                          " Rates: 48000\n"
                          " Bits: 16\n\n"
                          "Capture:\n"
                          " Status: Stop\n"
                          " Interface 3\n"
                          " Altset 1\n"
                          " Format: S16_LE\n"
                          " Channels: 1\n"
                          " Rates: 16000, 48000, 16000\n"
                          " Bits: 16\n"
                          " Channel map: MONO\n"
                          " Interface 3\n"
                          " Altset 2\n"
                          " Format: S24_3LE\n"
                          " Channels: 2\n"
                          " Rates: 32000 - 96000 (continuous)\n"
                          " Bits: 24\n"
                          " Channel map: FL FR\n"
                          " Interface 4\n"
                          " Altset 3\n"
                          " Format: S32_LE\n"
                          " Channels: 4\n"
                          " Rates: 48000, 96000\n"
                          " Bits: 32\n");
      fixture.add_capture(
          2, 1, 1, 1, "USB Audio #1",
          "Blue Yeti Nano : USB Audio #1\n\nCapture:\n Status: Stop\n Interface 5\n"
          " Altset 1\n Format: S24_3LE\n Channels: 2\n Rates: 44100, 48000\n Bits: 24\n");
      fixture.add_capture(7, 0, 1, 0, "USB Audio",
                          "Blue Yeti Nano : USB Audio\n\nCapture:\n Status: Stop\n Interface 3\n"
                          " Altset 1\n Format: S16_LE\n Channels: 2\n Rates: 48000\n Bits: 16\n");
      fixture.add_capture(0, 0, 1, 1, "On-board capture");
      fixture.add_playback(8, 0);

      const auto records = discover_alsa_capture_peripherals(fixture.roots);
      require(records.size() == 4, "every capture PCM and no playback-only PCM must be discovered");

      const auto& first =
          find_record(records, "sysfs:devices/pci0000:00/usb1/1-3.2/1-3.2:1.0:pcm0c");
      require(first.id == "microphone:alsa:3ff3d77bf791d455" && first.type == "microphone" &&
                  first.provider == "daemon.microphone.alsa",
              "ALSA captures must use the microphone provider envelope");
      require(
          first.details["capture_target"]["selector"] == "plughw:CARD=Nano,DEV=0" &&
              first.details["identity"]["pcm_node"] == "/dev/snd/pcmC2D0c" &&
              first.details["identity"]["by_path"].get<std::string>().starts_with(
                  "/dev/snd/by-path/") &&
              first.details["identity"]["by_id"].get<std::string>().starts_with("/dev/snd/by-id/"),
          "the record must expose a snapshot-bound ALSA selector and stable links");
      require(first.details["availability"]["state"] == "available" &&
                  first.details["availability"]["subdevices"] == 2 &&
                  first.details["availability"]["subdevices_available"] == 1,
              "partly occupied multi-subdevice captures must remain available");

      const auto& modes = first.details["modes"];
      const auto* mono = find_mode(modes, "S16_LE", 1);
      const auto* stereo = find_mode(modes, "S24_3LE", 2);
      const auto* multichannel = find_mode(modes, "S32_LE", 4);
      require(mono && (*mono)["rates_hz"] == nlohmann::json::array({16000, 48000}) &&
                  (*mono)["channel_map"] == nlohmann::json::array({"MONO"}),
              "mono discrete rates and channel maps must be preserved and deduplicated");
      require(stereo && (*stereo)["sample_bits"] == 24 &&
                  (*stereo)["rate_range_hz"]["min"] == 32000 &&
                  (*stereo)["rate_range_hz"]["max"] == 96000,
              "stereo 24-bit continuous rate ranges must be preserved");
      require(multichannel && (*multichannel)["sample_bits"] == 32 &&
                  (*multichannel)["altset"] == 3 && (*multichannel)["interface"] == 4,
              "32-bit multichannel interfaces and altsets must remain distinct");

      const auto& second_capture =
          find_record(records, "sysfs:devices/pci0000:00/usb1/1-3.2/1-3.2:1.0:pcm1c");
      require(second_capture.id != first.id && second_capture.details["modes"].size() == 1,
              "several capture interfaces on one composite device must remain distinct");
      const auto& identical =
          find_record(records, "sysfs:devices/pci0000:00/usb1/1-3.3/1-3.3:1.0:pcm0c");
      require(identical.id != first.id && identical.details["availability"]["state"] == "in_use",
              "identical USB devices on different topology paths must not collide");
      const auto& sparse_usb = identical.details["identity"]["usb"];
      require(!sparse_usb.contains("manufacturer") && !sparse_usb.contains("product") &&
                  !sparse_usb.contains("serial"),
              "missing optional USB strings must be omitted");

      fixture.set_usb_attributes_visible("1-3.2", false);
      const auto partial_sysfs_records = discover_alsa_capture_peripherals(fixture.roots);
      const auto& partial_sysfs = find_capture(partial_sysfs_records, "plughw:CARD=Nano,DEV=0");
      require(partial_sysfs.id == first.id,
              "temporary USB attribute read failures must not change stable identity");
      fixture.set_usb_attributes_visible("1-3.2", true);

      const auto& platform = find_record(records, "sysfs:devices/platform/audio-codec:pcm0c");
      require(platform.details["connection"] == "platform" && platform.details["modes"].empty() &&
                  platform.details.contains("issues"),
              "non-USB captures must remain visible with explicit unknown capabilities");

      TempDirectory renumbered_temp;
      Fixture renumbered(renumbered_temp.path());
      write_file(renumbered.roots.proc_asound / "cards",
                 " 4 [Nano          ]: USB-Audio - Yeti Nano\n");
      renumbered.add_usb_card(4, "Nano", "1-3.2", "1-3.2:1.0");
      renumbered.add_capture(4, 0, 1, 1, "USB Audio",
                             "Yeti Nano : USB Audio\n\nCapture:\n Interface 3\n Altset 1\n Format: "
                             "S16_LE\n Channels: 1\n Rates: 48000\n Bits: 16\n");
      const auto renumbered_records = discover_alsa_capture_peripherals(renumbered.roots);
      require(renumbered_records.size() == 1 && renumbered_records[0].id == first.id,
              "stable microphone identity must survive ALSA card renumbering");
      require(renumbered_records[0].details["capture_target"]["selector"] ==
                      "plughw:CARD=Nano,DEV=0" &&
                  renumbered_records[0].details["identity"]["card_index"] == 4,
              "routing metadata must reflect the current card without entering the stable ID");

      TempDirectory missing_temp;
      const AlsaDiscoveryRoots missing_roots{missing_temp.path() / "missing-proc",
                                             missing_temp.path() / "sys",
                                             missing_temp.path() / "dev"};
      require_neat_error([&] { (void)discover_alsa_capture_peripherals(missing_roots); },
                         simaai::neat::error_codes::kIoOpen, "ALSA discovery could not read",
                         "ALSA procfs is mounted");

      PeripheralCatalog coexistence("alsa-coexistence", 8);
      PeripheralCatalogManager manager(
          coexistence,
          {
              {"test.camera", {"media"}, [] { return std::vector{camera_record()}; }},
              {"daemon.microphone.alsa",
               {"sound"},
               [&] { return discover_alsa_capture_peripherals(fixture.roots); }},
          },
          0);
      manager.initial_scan();
      const auto complete_snapshot = coexistence.catalog_json();
      require(complete_snapshot["devices"].size() == 5,
              "the ALSA provider must coexist with other provider records");

      fixture.set_card_device_visible(2, false);
      manager.initial_scan();
      const auto removal_race_snapshot = coexistence.catalog_json();
      require(removal_race_snapshot["state"] == "degraded" &&
                  removal_race_snapshot["devices"] == complete_snapshot["devices"] &&
                  removal_race_snapshot["issues"].size() == 1 &&
                  removal_race_snapshot["issues"][0]["provider"] == "daemon.microphone.alsa" &&
                  removal_race_snapshot["issues"][0]["retained_last_good"] == true,
              "a partial sysfs removal race must retain the last stable microphone identities");
      fixture.set_card_device_visible(2, true);

      PeripheralCatalog degraded("alsa-degraded", 8);
      PeripheralCatalogManager degraded_manager(
          degraded,
          {
              {"test.camera", {"media"}, [] { return std::vector{camera_record()}; }},
              {"daemon.microphone.alsa",
               {"sound"},
               [&] { return discover_alsa_capture_peripherals(missing_roots); }},
          },
          0);
      degraded_manager.initial_scan();
      const auto degraded_snapshot = degraded.catalog_json();
      require(degraded_snapshot["state"] == "degraded" &&
                  degraded_snapshot["devices"].size() == 1 &&
                  degraded_snapshot["devices"][0]["type"] == "camera" &&
                  degraded_snapshot["issues"].size() == 1 &&
                  degraded_snapshot["issues"][0]["provider"] == "daemon.microphone.alsa" &&
                  degraded_snapshot["issues"][0]["code"] == simaai::neat::error_codes::kIoOpen,
              "an unreadable ALSA provider must report an actionable isolated issue without "
              "hiding cameras");
    }));

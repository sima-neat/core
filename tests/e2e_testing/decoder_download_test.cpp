#include "asset_utils.h"
#include "dmabuf_test_utils.h"
#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "test_utils.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr const char* kVideoUrl =
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4";
constexpr int kExpectedWidth = 640;
constexpr int kExpectedHeight = 360;
constexpr int kDefaultMinDecodedFrames = 180;

int64_t now_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

std::string trim(const std::string& s) {
  size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])))
    b++;
  size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
    e--;
  return s.substr(b, e - b);
}

bool parse_int64(const std::string& s, int64_t& out) {
  if (s.empty())
    return false;
  char* end = nullptr;
  errno = 0;
  long long v = std::strtoll(s.c_str(), &end, 10);
  if (errno != 0 || end == s.c_str() || *end != '\0')
    return false;
  out = static_cast<int64_t>(v);
  return true;
}

std::string env_or(const char* name, const char* def_value) {
  const char* val = std::getenv(name);
  if (val && *val) {
    return std::string(val);
  }
  return std::string(def_value ? def_value : "");
}

struct DecoderDmaEvidence {
  std::mutex mutex;
  std::vector<sima_test::DmaBufSpan> published;
  std::vector<sima_test::DmaBufBackingIdentity> held;
  std::size_t publications_while_held = 0;
  std::atomic<bool> failed{false};
};

GstPadProbeReturn record_decoder_dma(GstPad*, GstPadProbeInfo* info, gpointer data) noexcept {
  auto& evidence = *static_cast<DecoderDmaEvidence*>(data);
  try {
    const auto span = sima_test::dmabuf_span(gst_pad_probe_info_get_buffer(info));
    std::lock_guard<std::mutex> lock(evidence.mutex);
    if (std::find(evidence.held.begin(), evidence.held.end(), span.backing) !=
        evidence.held.end()) {
      evidence.failed.store(true);
    }
    if (!evidence.held.empty()) {
      ++evidence.publications_while_held;
    }
    evidence.published.push_back(span);
  } catch (...) {
    evidence.failed.store(true);
  }
  return GST_PAD_PROBE_OK;
}

void stop_pipeline(GstElement* pipeline) {
  if (pipeline) {
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
  }
}

using SampleRef = std::shared_ptr<GstSample>;

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();

    const std::string decoder_element = env_or("SIMA_DECODER_ELEMENT", "neatdecoder");
    if (decoder_element != "neatdecoder") {
      throw std::runtime_error("Invalid SIMA_DECODER_ELEMENT (expected neatdecoder)");
    }

    if (!simaai::neat::element_exists("qtdemux") || !simaai::neat::element_exists("h264parse") ||
        !simaai::neat::element_exists("appsink") ||
        !simaai::neat::element_exists(decoder_element.c_str())) {
      throw std::runtime_error(
          "Missing required GStreamer elements (qtdemux/h264parse/appsink/decoder)");
    }

    const std::filesystem::path dst =
        std::filesystem::temp_directory_path() / "sima_bbb_360_1mb.mp4";
    if (!sima_test::download_file(kVideoUrl, dst)) {
      throw std::runtime_error("Failed to download test video");
    }

    const std::string pipeline_desc =
        "filesrc location=" + dst.string() +
        " ! qtdemux name=demux demux.video_0 "
        "! h264parse "
        "! video/x-h264,parsed=true,stream-format=(string)byte-stream,alignment=(string)au "
        "! " +
        decoder_element +
        " name=decoder sima-allocator-type=2 "
        "! appsink name=mysink emit-signals=false sync=false max-buffers=0 drop=false";

    DecoderDmaEvidence evidence;
    GError* err = nullptr;
    GstElement* pipeline = gst_parse_launch(pipeline_desc.c_str(), &err);
    if (!pipeline) {
      std::string msg = "gst_parse_launch failed";
      if (err && err->message)
        msg = err->message;
      if (err)
        g_error_free(err);
      throw std::runtime_error(msg);
    }

    std::unique_ptr<GstElement, decltype(&stop_pipeline)> pipeline_guard(pipeline, stop_pipeline);
    if (err) {
      const std::string detail = err->message ? err->message : "partial pipeline";
      g_error_free(err);
      throw std::runtime_error("pipeline parse failed: " + detail);
    }
    GstElement* decoder = gst_bin_get_by_name(GST_BIN(pipeline), "decoder");
    require(decoder != nullptr, "decoder not found");
    std::shared_ptr<GstElement> decoder_guard(decoder, [](GstElement* p) { gst_object_unref(p); });
    gboolean direct = FALSE;
    g_object_get(decoder, "zero-copy-output", &direct, nullptr);
    require(direct == TRUE, "decoder default must use direct DMA-BUF output");
    GstPad* decoder_src = gst_element_get_static_pad(decoder, "src");
    require(decoder_src != nullptr, "decoder source pad not found");
    std::shared_ptr<GstPad> pad_guard(decoder_src, [](GstPad* p) { gst_object_unref(p); });
    const gulong probe = gst_pad_add_probe(decoder_src, GST_PAD_PROBE_TYPE_BUFFER,
                                           record_decoder_dma, &evidence, nullptr);
    require(probe != 0, "decoder DMA identity probe was not installed");

    GstElement* appsink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
    require(appsink != nullptr, "appsink not found");
    std::shared_ptr<GstElement> sink_guard(appsink, [](GstElement* p) { gst_object_unref(p); });

    GstBus* bus = gst_element_get_bus(pipeline);
    require(bus != nullptr, "pipeline bus not found");
    std::shared_ptr<GstBus> bus_guard(bus, [](GstBus* p) { gst_object_unref(p); });

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    const int64_t start_ms = now_ms();
    const int64_t timeout_ms = 30000;
    int64_t frames = 0;
    bool eos = false;
    bool timed_out = false;
    bool checked_caps = false;
    std::string bus_error;
    std::vector<SampleRef> held_samples;
    std::vector<sima_test::DmaBufBackingIdentity> unique_backings;
    std::vector<sima_test::DmaBufBackingIdentity> released_backings;
    bool observed_reuse = false;
    bool observed_released_reuse = false;

    const auto inspect_sample = [&](GstSample* sample) {
      const auto span = sima_test::dmabuf_span(gst_sample_get_buffer(sample));
      {
        std::lock_guard<std::mutex> lock(evidence.mutex);
        require(static_cast<std::size_t>(frames) < evidence.published.size(),
                "appsink received an unobserved decoder frame");
        require(evidence.published[frames] == span,
                "decoder publication and appsink changed DMA-BUF identity or span");
        if (frames < 2) {
          held_samples.emplace_back(gst_sample_ref(sample), gst_sample_unref);
          evidence.held.push_back(span.backing);
        }
        if (frames == 64) {
          released_backings = evidence.held;
          evidence.held.clear();
        }
      }
      if (frames == 64) {
        held_samples.clear();
      }
      if (std::find(unique_backings.begin(), unique_backings.end(), span.backing) ==
          unique_backings.end()) {
        unique_backings.push_back(span.backing);
      } else {
        observed_reuse = true;
      }
      if (std::find(released_backings.begin(), released_backings.end(), span.backing) !=
          released_backings.end()) {
        observed_released_reuse = true;
      }
      ++frames;
    };

    while (!eos) {
      GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), 200 * GST_MSECOND);
      if (sample) {
        SampleRef sample_guard(sample, gst_sample_unref);
        inspect_sample(sample);
        if (!checked_caps) {
          GstCaps* caps = gst_sample_get_caps(sample);
          require(caps != nullptr, "appsink sample missing caps");
          GstStructure* s = gst_caps_get_structure(caps, 0);
          int out_w = 0;
          int out_h = 0;
          if (!gst_structure_get_int(s, "width", &out_w) ||
              !gst_structure_get_int(s, "height", &out_h)) {
            throw std::runtime_error("appsink caps missing width/height");
          }
          if (out_w != kExpectedWidth || out_h != kExpectedHeight) {
            std::ostringstream oss;
            oss << "decoded caps mismatch: got " << out_w << "x" << out_h << " expected "
                << kExpectedWidth << "x" << kExpectedHeight;
            throw std::runtime_error(oss.str());
          }
          checked_caps = true;
        }
      }

      if (!sample && gst_app_sink_is_eos(GST_APP_SINK(appsink))) {
        eos = true;
      }

      while (GstMessage* msg = gst_bus_pop(bus)) {
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
          GError* gerr = nullptr;
          gchar* dbg = nullptr;
          gst_message_parse_error(msg, &gerr, &dbg);
          if (gerr && gerr->message) {
            bus_error = gerr->message;
          } else {
            bus_error = "unknown gst error";
          }
          if (gerr)
            g_error_free(gerr);
          if (dbg)
            g_free(dbg);
          gst_message_unref(msg);
          eos = true;
          break;
        }
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS) {
          gst_message_unref(msg);
          eos = true;
          break;
        }
        gst_message_unref(msg);
      }

      if (now_ms() - start_ms > timeout_ms) {
        timed_out = true;
        break;
      }
    }

    if (bus_error.empty()) {
      while (GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), 0)) {
        SampleRef sample_guard(sample, gst_sample_unref);
        inspect_sample(sample);
      }
    }

    held_samples.clear();
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_pad_remove_probe(decoder_src, probe);
    gint pool_buffers = 0;
    g_object_get(decoder, "num-buffers", &pool_buffers, nullptr);

    if (!bus_error.empty()) {
      throw std::runtime_error("GStreamer error: " + bus_error);
    }
    if (timed_out && !eos) {
      throw std::runtime_error("Timeout waiting for EOS");
    }

    require(!evidence.failed.load(),
            "decoder DMA probe failed or reissued a still-held allocation");
    require(evidence.publications_while_held > 0,
            "decoder did not progress while the app retained earlier DMA-BUF frames");
    require(observed_reuse && observed_released_reuse,
            "decoder did not recycle its existing DMA-BUF allocations after reader release");
    require(pool_buffers > 0 && unique_backings.size() <= static_cast<std::size_t>(pool_buffers),
            "decoder output allocation identities exceeded its fixed pool budget");
    require(frames > 0, "No frames decoded");
    int64_t min_frames = kDefaultMinDecodedFrames;
    if (const char* env = std::getenv("SIMA_DECODER_MIN_FRAMES")) {
      int64_t parsed = 0;
      if (parse_int64(trim(env), parsed) && parsed > 0) {
        min_frames = parsed;
      }
    }
    require(checked_caps, "did not observe decoded caps");
    require(frames >= min_frames, "Decoded frame count mismatch: got " + std::to_string(frames) +
                                      " expected >= " + std::to_string(min_frames));

    std::cout << "[OK] decoder_download_test passed (" << frames << " frames, "
              << unique_backings.size()
              << " DMA-BUF allocations, retained-reader reuse verified)\n";
    return 0;
  } catch (const std::runtime_error& e) {
    return fail_test(e.what());
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

#include "asset_utils.h"
#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "test_utils.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

namespace {

constexpr const char* kVideoUrl =
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4";
constexpr int kExpectedWidth = 640;
constexpr int kExpectedHeight = 360;

int64_t now_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();

    if (!simaai::neat::element_exists("qtdemux") || !simaai::neat::element_exists("h264parse") ||
        !simaai::neat::element_exists("appsink") || !simaai::neat::element_exists("neatdecoder")) {
      throw std::runtime_error(
          "Missing required GStreamer elements (qtdemux/h264parse/appsink/neatdecoder)");
    }

    const std::filesystem::path dst =
        std::filesystem::temp_directory_path() / "sima_bbb_360_1mb.mp4";
    if (!sima_test::download_file(kVideoUrl, dst)) {
      throw std::runtime_error("Failed to download test video");
    }

    // Intentionally mismatched target to force drop-on-resolution-change.
    const int wrong_w = kExpectedWidth + 16;
    const int wrong_h = kExpectedHeight + 16;

    std::ostringstream caps;
    caps << "video/x-h264,parsed=true,stream-format=(string)byte-stream,"
         << "alignment=(string)au,width=(int)" << kExpectedWidth << ",height=(int)"
         << kExpectedHeight;

    std::ostringstream pipeline_desc;
    pipeline_desc << "filesrc location=" << dst.string() << " ! qtdemux name=demux demux.video_0 "
                  << "! h264parse ! " << caps.str()
                  << " ! neatdecoder drop-on-resolution-change=true "
                  << "dec-width=" << wrong_w << " dec-height=" << wrong_h
                  << " sima-allocator-type=2 "
                  << "! appsink name=mysink emit-signals=false sync=false max-buffers=0 drop=false";

    GError* err = nullptr;
    GstElement* pipeline = gst_parse_launch(pipeline_desc.str().c_str(), &err);
    if (!pipeline) {
      std::string msg = "gst_parse_launch failed";
      if (err && err->message)
        msg = err->message;
      if (err)
        g_error_free(err);
      throw std::runtime_error(msg);
    }

    GstElement* appsink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
    require(appsink != nullptr, "appsink not found");

    GstBus* bus = gst_element_get_bus(pipeline);
    require(bus != nullptr, "pipeline bus not found");

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    const int64_t start_ms = now_ms();
    const int64_t timeout_ms = 15000;
    int64_t frames = 0;
    bool eos = false;
    std::string bus_error;

    while (!eos) {
      GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), 200 * GST_MSECOND);
      if (sample) {
        frames++;
        gst_sample_unref(sample);
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

      if ((now_ms() - start_ms) > timeout_ms) {
        break;
      }
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(bus);
    gst_object_unref(appsink);
    gst_object_unref(pipeline);

    if (!bus_error.empty()) {
      throw std::runtime_error("pipeline error: " + bus_error);
    }
    if (frames != 0) {
      std::ostringstream oss;
      oss << "expected 0 frames with drop-on-resolution-change, got " << frames;
      throw std::runtime_error(oss.str());
    }

    if (eos) {
      std::cout << "[OK] decoder_drop_on_resolution_change_test passed (0 frames, EOS)\n";
    } else {
      std::cout << "[OK] decoder_drop_on_resolution_change_test passed (0 frames, timeout)\n";
    }
    return 0;
  } catch (const std::runtime_error& e) {
    return fail_test(e.what());
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

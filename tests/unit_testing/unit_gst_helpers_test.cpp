#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"

#include "test_utils.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

class EnvVarGuard {
public:
  EnvVarGuard(const char* key, const char* value) : key_(key) {
    if (const char* current = std::getenv(key_)) {
      had_value_ = true;
      old_value_ = current;
    }
    ::setenv(key_, value, 1);
  }

  ~EnvVarGuard() {
    if (had_value_) {
      ::setenv(key_, old_value_.c_str(), 1);
    } else {
      ::unsetenv(key_);
    }
  }

private:
  const char* key_;
  bool had_value_ = false;
  std::string old_value_;
};

struct BlockingProbe {
  std::mutex mu;
  std::condition_variable cv;
  bool entered = false;
  bool released = false;
};

GstPadProbeReturn block_buffer(GstPad*, GstPadProbeInfo*, gpointer user_data) {
  auto* probe = static_cast<BlockingProbe*>(user_data);
  std::unique_lock<std::mutex> lock(probe->mu);
  probe->entered = true;
  probe->cv.notify_all();
  probe->cv.wait(lock, [&] { return probe->released; });
  return GST_PAD_PROBE_REMOVE;
}

void require_bounded_synchronous_teardown() {
  EnvVarGuard teardown_timeout("SIMA_GST_TEARDOWN_TIMEOUT_MS", "100");
  EnvVarGuard synchronous_teardown("SIMA_GST_TEARDOWN_DEFER_NO_FLUSH", "0");

  GError* error = nullptr;
  GstElement* pipeline = gst_parse_launch(
      "videotestsrc is-live=true ! identity name=blocker ! fakesink sync=false", &error);
  require(error == nullptr && pipeline != nullptr, "failed to create blocking teardown pipeline");

  GstElement* identity = gst_bin_get_by_name(GST_BIN(pipeline), "blocker");
  require(identity != nullptr, "blocking teardown identity is missing");
  GstPad* src_pad = gst_element_get_static_pad(identity, "src");
  require(src_pad != nullptr, "blocking teardown source pad is missing");

  BlockingProbe probe;
  gst_pad_add_probe(src_pad, GST_PAD_PROBE_TYPE_BUFFER, block_buffer, &probe, nullptr);
  require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
          "failed to start blocking teardown pipeline");
  {
    std::unique_lock<std::mutex> lock(probe.mu);
    require(probe.cv.wait_for(lock, std::chrono::seconds(2), [&] { return probe.entered; }),
            "blocking teardown probe was not reached");
  }

  std::thread release_probe([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    {
      std::lock_guard<std::mutex> lock(probe.mu);
      probe.released = true;
    }
    probe.cv.notify_all();
  });

  const auto started_at = std::chrono::steady_clock::now();
  simaai::neat::pipeline_internal::stop_and_unref_no_flush(pipeline, true);
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - started_at)
                              .count();

  release_probe.join();
  gst_object_unref(src_pad);
  gst_object_unref(identity);

  require(pipeline == nullptr, "teardown must transfer pipeline ownership");
  require(elapsed_ms < 400, "synchronous teardown exceeded its configured timeout");
}

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();
    simaai::neat::gst_init_once();

    require(simaai::neat::element_exists("identity"), "identity element missing");
    simaai::neat::require_element("identity", "unit_gst_helpers_test");
    require(simaai::neat::element_property_exists("videotestsrc", "num-buffers"),
            "lazy-loaded videotestsrc property was not discovered");
    require(simaai::neat::element_property_exists("videotestsrc", "num-buffers"),
            "element property discovery changed across repeated calls");
    require(!simaai::neat::element_property_exists("videotestsrc", "not-a-real-property"),
            "element property discovery accepted an unknown property");

    const char* desc =
        "videotestsrc num-buffers=1 ! video/x-raw,format=NV12,width=16,height=16,framerate=30/1 ! "
        "appsink name=mysink emit-signals=false sync=false max-buffers=1 drop=true";
    GError* err = nullptr;
    GstElement* pipeline = gst_parse_launch(desc, &err);
    require(pipeline != nullptr, "gst_parse_launch failed");
    if (err)
      g_error_free(err);

    GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
    require(sink != nullptr, "appsink not found");

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 2 * GST_SECOND);
    require(sample != nullptr, "failed to pull sample");

    simaai::neat::pipeline_internal::SampleHolder holder(sample);
    std::string map_err;
    require(simaai::neat::pipeline_internal::map_video_frame_read(holder, map_err),
            "map_video_frame_read failed: " + map_err);

    GstCaps* caps = gst_sample_get_caps(sample);
    auto caps_str = simaai::neat::pipeline_internal::gst_caps_to_string_safe(caps);
    require(!caps_str.empty(), "caps string empty");

    gst_sample_unref(sample);
    gst_object_unref(sink);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    require_bounded_synchronous_teardown();

    std::cout << "[OK] unit_gst_helpers_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

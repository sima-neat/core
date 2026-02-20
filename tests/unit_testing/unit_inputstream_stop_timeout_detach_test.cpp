#include "gst/GstInit.h"
#include "internal/InputStream.h"
#include "nodes/io/Input.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

namespace {

class EnvVarGuard {
public:
  EnvVarGuard(const char* key, const char* value) : key_(key), had_(false) {
    const char* cur = std::getenv(key_);
    if (cur && *cur) {
      had_ = true;
      old_ = cur;
    }
    ::setenv(key_, value, 1);
  }

  ~EnvVarGuard() {
    if (had_) {
      ::setenv(key_, old_.c_str(), 1);
    } else {
      ::unsetenv(key_);
    }
  }

private:
  const char* key_;
  bool had_;
  std::string old_;
};

simaai::neat::SampleSpec make_placeholder_spec() {
  simaai::neat::SampleSpec spec;
  spec.kind = simaai::neat::SampleMediaKind::Encoded;
  spec.caps_string = "application/octet-stream";
  spec.caps_key = simaai::neat::capkey_from_spec(spec);
  return spec;
}

bool wait_for_flag(const std::atomic<bool>& flag, int timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (flag.load(std::memory_order_acquire))
      return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return flag.load(std::memory_order_acquire);
}

} // namespace

RUN_TEST("unit_inputstream_stop_timeout_detach_test", ([] {
           using namespace simaai::neat;

           gst_init_once();

           EnvVarGuard stop_timeout("SIMA_INPUTSTREAM_STOP_TIMEOUT_MS", "0");
           EnvVarGuard stop_flush("SIMA_INPUTSTREAM_STOP_FLUSH", "0");
           EnvVarGuard stop_unblock("SIMA_INPUTSTREAM_STOP_UNBLOCK", "1");

           GError* err = nullptr;
           GstElement* pipeline =
               gst_parse_launch("videotestsrc is-live=true pattern=black ! "
                                "video/x-raw,format=RGB,width=64,height=48,framerate=30/1 ! "
                                "appsink name=mysink sync=false max-buffers=1 drop=true",
                                &err);
           if (err) {
             const std::string msg = err->message ? err->message : "gst_parse_launch failed";
             g_error_free(err);
             throw std::runtime_error(msg);
           }
           require(pipeline != nullptr, "expected valid pipeline in InputStream stop timeout test");

           GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
           if (!sink) {
             gst_object_unref(pipeline);
             throw std::runtime_error("appsink 'mysink' not found");
           }

           if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
             gst_object_unref(sink);
             gst_object_unref(pipeline);
             throw std::runtime_error("failed to set test pipeline to PLAYING");
           }

           InputOptions src_opt;
           InputStreamOptions opt;
           opt.timeout_ms = 500;
           opt.worker_poll_ms = 5;
           opt.appsink_drop = true;
           opt.appsink_max_buffers = 1;

           InputStream stream = InputStream::create(
               pipeline, nullptr, sink, make_placeholder_spec(), src_opt, opt, {}, nullptr);
           struct StreamGuard {
             InputStream* stream = nullptr;
             ~StreamGuard() {
               if (!stream)
                 return;
               try {
                 stream->close();
               } catch (...) {
               }
             }
           } guard{&stream};

           std::mutex mu;
           std::condition_variable cv;
           std::atomic<bool> callback_entered{false};
           std::atomic<bool> callback_done{false};
           bool release_callback = false;

           stream.start([&](Sample) {
             callback_entered.store(true, std::memory_order_release);
             cv.notify_all();
             std::unique_lock<std::mutex> lock(mu);
             cv.wait(lock, [&] { return release_callback; });
             callback_done.store(true, std::memory_order_release);
           });

           require(wait_for_flag(callback_entered, 3000),
                   "InputStream callback did not enter before stop()");

           const auto t0 = std::chrono::steady_clock::now();
           stream.stop();
           const auto t1 = std::chrono::steady_clock::now();

           const int stop_ms = static_cast<int>(
               std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
           require(stop_ms < 2000,
                   "InputStream::stop should return quickly on timeout detach path");

           {
             std::lock_guard<std::mutex> lock(mu);
             release_callback = true;
           }
           cv.notify_all();

           require(wait_for_flag(callback_done, 3000),
                   "InputStream callback did not finish after release");

           stream.stop(); // idempotent
           stream.close();
           guard.stream = nullptr;
         }));

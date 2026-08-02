#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "pipeline/runtime/RunCore.h"
#include "runtime_test_utils.h"
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

bool is_expected_teardown_error(const std::string& msg) {
  return msg.find("stream is stopping") != std::string::npos ||
         msg.find("EOS has been reached") != std::string::npos ||
         msg.find("gst_app_src_end_of_stream failed (flow=-2:flushing)") != std::string::npos ||
         msg.find("stream closed") != std::string::npos;
}

struct BlockingProbe {
  std::mutex mu;
  std::condition_variable cv;
  bool entered = false;
  bool released = false;
};

GstPadProbeReturn block_pipeline_output(GstPad*, GstPadProbeInfo*, gpointer user_data) {
  auto* probe = static_cast<BlockingProbe*>(user_data);
  std::unique_lock<std::mutex> lock(probe->mu);
  probe->entered = true;
  probe->cv.notify_all();
  probe->cv.wait(lock, [&] { return probe->released; });
  return GST_PAD_PROBE_REMOVE;
}

GstElement* find_appsink(GstElement* pipeline) {
  GstIterator* iterator = gst_bin_iterate_sinks(GST_BIN(pipeline));
  require(iterator != nullptr, "run lifecycle teardown: pipeline has no sink iterator");

  GstElement* appsink = nullptr;
  GValue item = G_VALUE_INIT;
  while (!appsink) {
    const GstIteratorResult result = gst_iterator_next(iterator, &item);
    if (result == GST_ITERATOR_OK) {
      auto* candidate = GST_ELEMENT(g_value_get_object(&item));
      if (GST_IS_APP_SINK(candidate)) {
        appsink = GST_ELEMENT(gst_object_ref(candidate));
      }
      g_value_reset(&item);
      continue;
    }
    if (result == GST_ITERATOR_RESYNC) {
      gst_iterator_resync(iterator);
      continue;
    }
    break;
  }
  g_value_unset(&item);
  gst_iterator_free(iterator);
  return appsink;
}

void detached_stream_stop_retains_runtime_until_cleanup() {
  using namespace simaai::neat;

  EnvVarGuard stream_stop_timeout("SIMA_PIPELINE_STREAM_STOP_TIMEOUT_MS", "100");
  EnvVarGuard input_stop_timeout("SIMA_PIPELINE_INPUT_THREAD_STOP_TIMEOUT_MS", "100");
  EnvVarGuard stop_flush("SIMA_INPUTSTREAM_STOP_FLUSH", "0");
  EnvVarGuard synchronous_teardown("SIMA_GST_TEARDOWN_DEFER_NO_FLUSH", "0");

  const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x4B);
  Run run = sima_test::make_async_rgb_run(seed, 8, 8);
  auto core = run_internal::core(run);
  GstElement* pipeline = core->pipeline.stream.pipeline_handle();
  require(pipeline != nullptr, "run lifecycle teardown: missing pipeline");
  GstElement* appsink = find_appsink(pipeline);
  require(appsink != nullptr, "run lifecycle teardown: missing appsink");
  GstPad* sink_pad = gst_element_get_static_pad(appsink, "sink");
  require(sink_pad != nullptr, "run lifecycle teardown: missing appsink sink pad");

  BlockingProbe probe;
  gst_pad_add_probe(sink_pad, GST_PAD_PROBE_TYPE_BUFFER, block_pipeline_output, &probe, nullptr);
  require(run.try_push(TensorList{seed}), "run lifecycle teardown: push failed");
  {
    std::unique_lock<std::mutex> lock(probe.mu);
    require(probe.cv.wait_for(lock, std::chrono::seconds(2), [&] { return probe.entered; }),
            "run lifecycle teardown: blocking probe was not reached");
  }

  std::weak_ptr<const runtime::RunCore> weak_core = core;
  core.reset();
  const auto close_started_at = std::chrono::steady_clock::now();
  run.close();
  const int close_ms = sima_test::elapsed_ms(close_started_at, std::chrono::steady_clock::now());
  auto retained_core = weak_core.lock();
  const bool stop_was_detached =
      retained_core && retained_core->stream_stop_detached.load(std::memory_order_acquire);
  retained_core.reset();

  {
    std::lock_guard<std::mutex> lock(probe.mu);
    probe.released = true;
  }
  probe.cv.notify_all();

  const auto cleanup_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!weak_core.expired() && std::chrono::steady_clock::now() < cleanup_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  gst_object_unref(sink_pad);
  gst_object_unref(appsink);

  require(close_ms < 1000, "Run::close must remain bounded when stream teardown blocks");
  require(stop_was_detached,
          "detached stream teardown must retain its RunCore until the blocked work exits");
  require(weak_core.expired(), "retained RunCore was not released after teardown completed");
}

} // namespace

RUN_TEST("unit_run_lifecycle_teardown_test", ([] {
           using namespace simaai::neat;

           EnvVarGuard input_stop_1("SIMA_PIPELINE_INPUT_THREAD_STOP_TIMEOUT_MS", "200");

           const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x3A);
           Run run = sima_test::make_async_rgb_run(seed, 128, 128);

           std::atomic<bool> keep_running{true};
           std::atomic<bool> teardown_started{false};
           std::atomic<int> pushes{0};
           std::atomic<int> pulls{0};
           std::mutex err_mu;
           std::string producer_err;
           std::string consumer_err;

           std::thread producer([&] {
             try {
               while (keep_running.load(std::memory_order_relaxed)) {
                 if (run.try_push(TensorList{seed})) {
                   pushes.fetch_add(1, std::memory_order_relaxed);
                 } else {
                   std::this_thread::sleep_for(std::chrono::milliseconds(1));
                 }
               }
             } catch (const std::exception& e) {
               if (teardown_started.load(std::memory_order_acquire) &&
                   is_expected_teardown_error(e.what())) {
                 return;
               }
               std::lock_guard<std::mutex> lock(err_mu);
               producer_err = e.what();
             }
           });

           std::thread consumer([&] {
             try {
               while (keep_running.load(std::memory_order_relaxed)) {
                 auto out = run.pull(20);
                 if (out.has_value()) {
                   pulls.fetch_add(1, std::memory_order_relaxed);
                 }
               }
             } catch (const std::exception& e) {
               if (teardown_started.load(std::memory_order_acquire) &&
                   is_expected_teardown_error(e.what())) {
                 return;
               }
               std::lock_guard<std::mutex> lock(err_mu);
               consumer_err = e.what();
             }
           });

           std::this_thread::sleep_for(std::chrono::milliseconds(200));

           const auto t0 = std::chrono::steady_clock::now();
           teardown_started.store(true, std::memory_order_release);
           run.close_input();
           run.stop();
           run.stop(); // idempotent stop under prior teardown
           const auto t1 = std::chrono::steady_clock::now();

           keep_running.store(false, std::memory_order_relaxed);
           producer.join();
           consumer.join();

           {
             std::lock_guard<std::mutex> lock(err_mu);
             if (!producer_err.empty()) {
               throw std::runtime_error("run lifecycle producer failed: " + producer_err);
             }
             if (!consumer_err.empty()) {
               throw std::runtime_error("run lifecycle consumer failed: " + consumer_err);
             }
           }

           const int stop_ms = sima_test::elapsed_ms(t0, t1);
           require(stop_ms < 5000, "Run::stop teardown exceeded expected bound");

           require(!run.try_push(TensorList{seed}), "Run::try_push should fail after stop()");
           (void)pushes.load(std::memory_order_relaxed);
           (void)pulls.load(std::memory_order_relaxed);

           detached_stream_stop_retains_runtime_until_cleanup();
         }));

#include "gst/GstInit.h"
#include "internal/InputStream.h"
#include "nodes/io/Input.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "test_main.h"

#include <gst/gst.h>
#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <functional>
#include <string>
#include <thread>

namespace {
using namespace simaai::neat;
using namespace std::chrono_literals;

bool wait_until(const std::function<bool()>& predicate, int timeout_ms = 2000) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (!predicate() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  return predicate();
}

struct BlockingGate {
  std::mutex mutex;
  std::condition_variable changed;
  bool blocked = false;
  bool entered = false;
  GstPad* pad = nullptr;
  gulong probe = 0;

  void install(GstElement* element) {
    pad = gst_element_get_static_pad(element, "sink");
    probe = gst_pad_add_probe(
        pad, GST_PAD_PROBE_TYPE_BUFFER,
        +[](GstPad*, GstPadProbeInfo*, gpointer data) -> GstPadProbeReturn {
          auto& gate = *static_cast<BlockingGate*>(data);
          std::unique_lock<std::mutex> lock(gate.mutex);
          gate.entered = true;
          gate.changed.notify_all();
          gate.changed.wait(lock, [&] { return !gate.blocked; });
          return GST_PAD_PROBE_OK;
        },
        this, nullptr);
  }

  void block() {
    std::lock_guard<std::mutex> lock(mutex);
    entered = false;
    blocked = true;
  }

  void release() {
    std::lock_guard<std::mutex> lock(mutex);
    blocked = false;
    changed.notify_all();
  }

  void wait_for_entry() {
    std::unique_lock<std::mutex> lock(mutex);
    require(changed.wait_for(lock, 1s, [&] { return entered; }), "Gate was not reached");
  }

  ~BlockingGate() {
    release();
    if (pad) {
      gst_pad_remove_probe(pad, probe);
      gst_object_unref(pad);
    }
  }
};

struct Fixture {
  std::atomic<int> outputs{0};
  std::atomic<int> arrivals{0};
  GstPad* arrival_pad = nullptr;
  gulong arrival_probe = 0;
  BlockingGate blocker;
  InputStream stream;
  GstElement* gate = nullptr;
  cv::Mat image{48, 64, CV_8UC3, cv::Scalar(10, 20, 30)};

  explicit Fixture(bool drop = false, const std::string& middle = "",
                   const char* block_name = "gate") {
    GError* error = nullptr;
    const std::string description =
        "appsrc name=mysrc is-live=true format=time block=true max-bytes=0 "
        "caps=video/x-raw,format=RGB,width=64,height=48,framerate=0/1 ! " +
        middle +
        "identity name=gate ! appsink name=mysink sync=false async=false "
        "max-buffers=4 drop=false";
    GstElement* pipeline = gst_parse_launch(description.c_str(), &error);
    if (error) {
      const std::string message = error->message;
      g_error_free(error);
      throw std::runtime_error(message);
    }
    require(pipeline != nullptr, "Failed to build idle test pipeline");
    auto* source = gst_bin_get_by_name(GST_BIN(pipeline), "mysrc");
    auto* sink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
    gate = gst_bin_get_by_name(GST_BIN(pipeline), block_name);
    if (drop) {
      g_object_set(sink, "drop", TRUE, "max-buffers", 1U, nullptr);
      arrival_pad = gst_element_get_static_pad(sink, "sink");
      arrival_probe = gst_pad_add_probe(
          arrival_pad, GST_PAD_PROBE_TYPE_BUFFER,
          +[](GstPad*, GstPadProbeInfo*, gpointer data) -> GstPadProbeReturn {
            static_cast<std::atomic<int>*>(data)->fetch_add(1);
            return GST_PAD_PROBE_OK;
          },
          &arrivals, nullptr);
    }
    InputOptions input;
    input.format = FormatTag::RGB;
    input.width = 64;
    input.height = 48;
    input.depth = 3;
    input.memory_policy = InputMemoryPolicy::SystemMemory;
    input.pool_max_buffers = 64;
    const auto tensor = tensor_from_cv_mat(image, input, "idle test");
    const auto spec = derive_tensor_spec_or_throw(tensor, input, "idle test");
    InputStreamOptions options;
    options.timeout_ms = 200;
    options.worker_poll_ms = 2;
    options.appsink_max_buffers = drop ? 1 : 4;
    options.appsink_drop = drop;
    require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
            "Failed to start idle test pipeline");
    stream = InputStream::create(pipeline, source, sink, spec, input, options, {}, nullptr);
    blocker.install(gate);
  }

  ~Fixture() {
    blocker.release();
    if (arrival_pad) {
      gst_pad_remove_probe(arrival_pad, arrival_probe);
      gst_object_unref(arrival_pad);
    }
    stream.close();
    if (gate) {
      gst_object_unref(gate);
    }
  }

  void start() {
    stream.start([this](Sample) { outputs.fetch_add(1); });
  }

  void send_and_wait() {
    const int expected = outputs.load() + 1;
    require(stream.try_push(image), "Input rejected after idle");
    require(wait_until([&] { return outputs.load() == expected; }),
            "Expected exactly one output after input");
    require(stream.last_error().empty(), stream.last_error());
  }

  void idle() {
    std::this_thread::sleep_for(450ms); // More than twice the worker timeout.
    require(stream.last_error().empty(), "Idle graph failed: " + stream.last_error());
    require(stream.running(), "Idle worker stopped");
  }
};
} // namespace

RUN_TEST("unit_inputstream_idle_timeout_test", ([] {
           gst_init_once();
           {
             std::cerr << "case: idle and fast outputs\n";
             Fixture test;
             test.start();
             test.idle(); // No input has ever been submitted.
             for (int i = 0; i < 3; ++i) {
               test.send_and_wait();
               test.idle(); // Completed work must disarm the watchdog.
             }
             // Exercise fast completion racing the return from appsrc's push call.
             for (int i = 0; i < 500; ++i) {
               test.send_and_wait();
             }
             test.idle();
             test.send_and_wait();
           }
           {
             std::cerr << "case: synchronous pull retirement\n";
             Fixture test;
             test.stream.push(test.image);
             (void)test.stream.pull(1000); // The seeded/preflight sync path also retires work.
             test.stream.push(test.image);
             test.stream.pull_and_discard(1000);
             const auto empty_pull_started = std::chrono::steady_clock::now();
             bool empty_pull_failed = false;
             try {
               (void)test.stream.pull(20);
             } catch (const std::runtime_error&) {
               empty_pull_failed = true;
             }
             require(empty_pull_failed, "Explicit empty pull must still time out");
             require(std::chrono::steady_clock::now() - empty_pull_started < 1s,
                     "Explicit pull wait became unbounded");
             test.start();
             test.idle();
             test.send_and_wait();
             test.idle();
           }
           {
             std::cerr << "case: dropping appsink\n";
             Fixture test(true);
             // The polling appsink may replace completed output before a pull.
             // Such results must not leave phantom outstanding work behind.
             for (int i = 0; i < 2; ++i)
               test.stream.push(test.image);
             require(wait_until([&] { return test.arrivals.load() == 2; }),
                     "Burst did not reach the dropping appsink");
             test.stream.pull_and_discard(1000);
             test.start();
             test.idle();
             test.send_and_wait();
             test.idle();
           }
           for (const char* policy : {"no", "downstream", "upstream"}) {
             std::cerr << "case: queue policy=" << policy << '\n';
             Fixture test(false, std::string("queue max-size-buffers=1 max-size-bytes=0 ") +
                                     "max-size-time=0 leaky=" + policy + " ! ");
             test.start();
             test.blocker.block();
             test.stream.push(test.image);
             test.blocker.wait_for_entry();
             for (int i = 1; i < 20; ++i)
               test.stream.push(test.image);
             std::this_thread::sleep_for(30ms);
             test.blocker.release();
             require(wait_until([&] { return test.outputs.load() >= 2; }),
                     "Queue survivors failed to arrive");
             test.idle();
             const int survivors = test.outputs.load();
             require(policy == std::string("no") ? survivors == 20 : survivors < 20,
                     "Queue fixture did not exercise the requested overflow policy");
             test.send_and_wait();
             test.idle();
           }
           {
             std::cerr << "case: queue admission after idle before worker startup\n";
             Fixture test(false, "queue leaky=downstream max-size-buffers=4 "
                                 "min-threshold-buffers=2 ! ");
             std::this_thread::sleep_for(450ms);
             test.stream.push(test.image);
             std::this_thread::sleep_for(30ms);
             test.start();
             std::this_thread::sleep_for(50ms);
             require(test.stream.last_error().empty(), "New queue work inherited an idle deadline");
             test.stream.push(test.image);
             require(wait_until([&] { return test.outputs.load() >= 1; }),
                     "Queue did not release work after its threshold was met");
           }
           {
             std::cerr << "case: valve discards are not stalled work\n";
             Fixture test(false, "valve name=discard ! ");
             std::shared_ptr<GstElement> valve(
                 gst_bin_get_by_name(GST_BIN(test.stream.pipeline_handle()), "discard"),
                 gst_object_unref);
             test.start();
             test.send_and_wait();
             g_object_set(valve.get(), "drop", TRUE, nullptr);
             for (int i = 0; i < 20; ++i)
               test.stream.push(test.image);
             test.idle();
             require(test.outputs.load() == 1, "Valve failed to discard the burst");
             g_object_set(valve.get(), "drop", FALSE, nullptr);
             test.send_and_wait();
             test.idle();
           }
           for (const std::string middle : {"", "queue leaky=downstream max-size-buffers=1 ! ",
                                            "queue leaky=upstream max-size-buffers=1 ! "}) {
             std::cerr << "case: stalled output after " << middle << '\n';
             Fixture test(false, middle);
             test.start();
             test.send_and_wait();
             test.blocker.block();
             test.stream.push(test.image);
             test.blocker.wait_for_entry(); // Accepted work really is blocked, not discarded.
             const auto started = std::chrono::steady_clock::now();
             // Keep feeding beyond the deadline. The old last-push rule could
             // hide this stall forever as long as more input kept arriving.
             while (test.stream.last_error().empty() &&
                    std::chrono::steady_clock::now() - started < 700ms) {
               std::this_thread::sleep_for(20ms);
               if (!test.stream.last_error().empty())
                 break;
               try {
                 (void)test.stream.try_push(test.image);
               } catch (const std::runtime_error&) {
                 require(!test.stream.last_error().empty(),
                         "Push failed without a recorded stream error");
               }
             }
             require(!test.stream.last_error().empty(),
                     "Continued input indefinitely postponed the stalled-output timeout");
             require(test.stream.last_error().find("runtime.output_timeout") != std::string::npos,
                     "Expected structured output timeout");
             require(std::chrono::steady_clock::now() - started < 1s,
                     "Stall detection exceeded its bounded deadline");
           }
           {
             std::cerr << "case: stall before dropping queue\n";
             Fixture test(false,
                          "identity name=early ! queue leaky=downstream "
                          "max-size-buffers=1 ! ",
                          "early");
             test.start();
             test.send_and_wait();
             test.blocker.block();
             test.stream.push(test.image);
             test.blocker.wait_for_entry();
             require(wait_until([&] { return !test.stream.last_error().empty(); }, 1000),
                     "Empty dropping queue masked an upstream stall");
             require(test.stream.last_error().find("runtime.output_timeout") != std::string::npos,
                     "Expected upstream output timeout");
           }
         }));

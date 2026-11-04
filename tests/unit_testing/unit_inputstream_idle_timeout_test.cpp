#include "gst/GstInit.h"
#include "internal/InputStream.h"
#include "nodes/io/Input.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "test_main.h"

#include <gst/gst.h>
#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
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

struct Fixture {
  std::atomic<int> outputs{0};
  std::atomic<int> arrivals{0};
  GstPad* arrival_pad = nullptr;
  gulong arrival_probe = 0;
  InputStream stream;
  GstElement* gate = nullptr;
  cv::Mat image{48, 64, CV_8UC3, cv::Scalar(10, 20, 30)};

  explicit Fixture(bool drop = false) {
    GError* error = nullptr;
    GstElement* pipeline =
        gst_parse_launch("appsrc name=mysrc is-live=true format=time block=true "
                         "caps=video/x-raw,format=RGB,width=64,height=48,framerate=0/1 ! "
                         "valve name=gate ! appsink name=mysink sync=false async=false "
                         "max-buffers=4 drop=false",
                         &error);
    if (error) {
      const std::string message = error->message;
      g_error_free(error);
      throw std::runtime_error(message);
    }
    require(pipeline != nullptr, "Failed to build idle test pipeline");
    auto* source = gst_bin_get_by_name(GST_BIN(pipeline), "mysrc");
    auto* sink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
    gate = gst_bin_get_by_name(GST_BIN(pipeline), "gate");
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
  }

  ~Fixture() {
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
           {
             std::cerr << "case: stalled output\n";
             Fixture test;
             test.start();
             test.send_and_wait();
             g_object_set(test.gate, "drop", TRUE, nullptr);
             test.stream.push(test.image); // Accepted input, but deliberately no output.
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
         }));

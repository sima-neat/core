#include "nodes/common/Caps.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "test_main.h"
#include "test_utils.h"

#include <chrono>
#include <string>

namespace {

using namespace simaai::neat;

// A caps request that the converter in front of the capsfilter cannot satisfy. The failure happens
// in a streaming thread behind a Neat async queue, so no bus error is posted while the input is
// open.
constexpr const char* kUnsatisfiableCaps =
    "videoconvert ! capsfilter caps=\"video/x-raw,format=RGB,interlace-mode=interleaved\"";

constexpr int kPullTimeoutMs = 5000;
constexpr int kMaxErrorLatencyMs = 4000;
constexpr int kMaxBuildMs = 8000;

Run build_failing_run(const Tensor& frame) {
  Graph graph("open_input_caps_error");
  graph.add(nodes::Input("in"));
  graph.add(nodes::Custom(kUnsatisfiableCaps));
  graph.add(nodes::Output());
  Run run = graph.build(TensorList{frame});
  require(run.push(TensorList{frame}), "push of the first frame should be accepted");
  return run;
}

long elapsed_since_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                               start)
      .count();
}

// The first reported error stays the Run's error: pulling again and closing the input must not turn
// it into a sample, a timeout or a different code.
void check_first_error_is_kept(Run& run, const std::string& first_message) {
  Sample sample;
  PullError err;
  require(run.pull(kPullTimeoutMs, sample, &err) == PullStatus::Error,
          "status pull should report Error once the failure was raised");
  require(err.code == error_codes::kMediaCaps, "status pull error_code was '" + err.code + "'");

  run.close_input();

  std::string code;
  try {
    run.pull(kPullTimeoutMs);
    require(false, "pull() after close_input() should keep raising the first error");
  } catch (const NeatError& e) {
    code = e.report().error_code;
  }
  require(code == error_codes::kMediaCaps,
          "pull() after close_input() raised error_code '" + code + "'");
  require(run.last_error() == first_message, "last_error() changed after close_input()");
}

// An infinite wait must end with the error instead of blocking for as long as the input is open.
void check_infinite_pull_raises(const Tensor& frame) {
  Run run = build_failing_run(frame);
  const auto start = std::chrono::steady_clock::now();
  std::string code;
  try {
    run.pull(-1);
    require(false, "pull(-1) returned instead of raising the caps negotiation error");
  } catch (const NeatError& e) {
    code = e.report().error_code;
  }
  require(code == error_codes::kMediaCaps, "pull(-1) raised error_code '" + code + "'");
  require(elapsed_since_ms(start) < kMaxErrorLatencyMs, "pull(-1) raised too late");
  run.close();
}

// With an unnamed Input the build preflight pulls the seed itself; it must report the caps failure
// instead of waiting out its own timeout.
void check_unnamed_input_build_fails_fast(const Tensor& frame) {
  Graph graph("open_input_caps_error_unnamed");
  graph.add(nodes::Input());
  graph.add(nodes::Custom(kUnsatisfiableCaps));
  graph.add(nodes::Output());

  const auto start = std::chrono::steady_clock::now();
  std::string code;
  try {
    Run run = graph.build(TensorList{frame});
    run.close();
    require(false, "build() accepted a graph whose caps cannot be negotiated");
  } catch (const NeatError& e) {
    code = e.report().error_code;
  }
  require(code == error_codes::kMediaCaps, "build() raised error_code '" + code + "'");
  require(elapsed_since_ms(start) < kMaxBuildMs,
          "build() reported the failure only after its own timeout");
}

} // namespace

RUN_TEST("open_input_caps_error_regression_test", ([] {
           const Tensor frame = make_color_tensor(320, 240, ImageSpec::PixelFormat::RGB, 0x40);
           Run run = build_failing_run(frame);

           // The input stays open on purpose: a negotiation failure must reach the caller anyway.
           std::string raised;
           std::string code;
           const auto start = std::chrono::steady_clock::now();
           try {
             const auto sample = run.pull(kPullTimeoutMs);
             require(false, std::string("pull() with the input open returned ") +
                                (sample.has_value() ? "a sample" : "no sample") +
                                " instead of raising the caps negotiation error; last_error='" +
                                run.last_error() + "'");
           } catch (const NeatError& e) {
             raised = e.what();
             code = e.report().error_code;
           }
           const auto elapsed_ms = elapsed_since_ms(start);

           require(code == error_codes::kMediaCaps,
                   "pull() raised error_code '" + code + "', expected misconfig.media_caps");
           require(elapsed_ms < kMaxErrorLatencyMs,
                   "pull() raised only after " + std::to_string(elapsed_ms) + " ms");
           require(!run.last_error().empty(), "last_error() is empty after pull() raised");
           require(run.last_error() == raised, "last_error() differs from the raised message");

           check_first_error_is_kept(run, raised);
           run.close();

           check_infinite_pull_raises(frame);
           check_unnamed_input_build_fails_fast(frame);
         }));

// Dataflow twin of unit_run_api_matrix_test (H0 C4, plan D12), on the public API only: every Run
// here requests RunBackend::Dataflow explicitly, and no internal header is used.
//
// The GStreamer test stays GStreamer-only: it posts a GStreamer bus error into the running
// pipeline (pipeline_handle()) and injects EOS into it, and its other blocks need elements the
// Dataflow backend does not take before H1 (custom() sources and identity, Graph::run's sync
// mode) or GStreamer buffer holders. Kept here, with the GStreamer test's case ids:
// - closed_*: the default (empty) Run;
// - push_close_*: application-driven EOS is normal closure with runtime.pull, not source-ended;
// - active_*: pull timeouts and their code, push/pull/run round trips, try_push of a Sample copy,
//   a CPU image Sample snapshot independent of later writes to the input, try_push backpressure
//   that is non-blocking, non-fatal and recovers, and refusal after close_input and after stop;
// - active_holder_zero_timeout_poll_*: pull(0) is a benign poll (no error) and a pull_samples
//   drain leaves no latched error;
// - active_*holder*: the holder push is a structured error on Dataflow;
// - format_depth_mismatch_*, empty_input_build: input validation at push and at build.
// Known, documented differences (c3.md section 4 row 5): the Dataflow backend carries no Sample
// attributes, dts_ns or duration_ns (no caps, no meta map; APPLICATION_MODEL 2.1). The GStreamer
// test's "attributes preserved" and "dts/duration preserved" assertions are checked here as "not
// carried" (empty / -1), explicitly, rather than skipped.

#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <thread>

namespace {

using namespace simaai::neat;

std::string run_api_case(const char* id, const std::string& msg) {
  return std::string("RUN_API_CASE=") + id + " " + msg;
}

bool throws_with(const std::function<void()>& fn, const std::string& needle) {
  try {
    fn();
  } catch (const std::exception& e) {
    return needle.empty() || std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

InputOptions rgb_input() {
  InputOptions src_opt;
  src_opt.payload_type = PayloadType::Image;
  src_opt.format = FormatTag::RGB;
  src_opt.memory_policy = InputMemoryPolicy::SystemMemory;
  src_opt.max_width = 96;
  src_opt.max_height = 96;
  src_opt.max_depth = 3;
  return src_opt;
}

RunOptions dataflow(int queue_depth) {
  RunOptions run_opt;
  run_opt.backend = RunBackend::Dataflow;
  run_opt.queue_depth = queue_depth;
  run_opt.overflow_policy = OverflowPolicy::Block;
  return run_opt;
}

// sima_test::make_async_rgb_run (the GStreamer test's), on Dataflow: Output EveryFrame(128).
Run make_async_rgb_run(const Tensor& seed, int queue_depth) {
  Graph graph;
  graph.add(nodes::Input(rgb_input()));
  graph.add(nodes::Output(OutputOptions::EveryFrame(128)));
  return graph.build(TensorList{seed}, dataflow(queue_depth));
}

Run make_async_rgb_run_with_copy_input(const Tensor& seed, int queue_depth) {
  Graph graph;
  graph.add(nodes::Input(rgb_input()));
  graph.add(nodes::Output(OutputOptions::EveryFrame(128)));
  RunOptions run_opt = dataflow(queue_depth);
  run_opt.advanced.copy_input = true;
  return graph.build(TensorList{seed}, run_opt);
}

void closed_default_run(const Tensor& seed) {
  Run closed;
  require(throws_with([&] { (void)closed.try_push(TensorList{seed}); }, "stream is closed"),
          run_api_case("closed_try_push_tensor", "Run::try_push should reject a default Run"));
  require(throws_with([&] { (void)closed.try_push_holder({}); }, "stream is closed"),
          run_api_case("closed_try_push_holder", "try_push_holder should reject a default Run"));
  Sample tmp;
  PullError err;
  require(closed.pull(10, tmp, &err) == PullStatus::Closed,
          run_api_case("closed_pull_status", "Run::pull closed status mismatch"));
  require(err.code == error_codes::kRuntimePull,
          run_api_case("closed_pull_error_code", "closed pull should set runtime.pull"));
  require_contains(err.message, "closed",
                   run_api_case("closed_pull_error_msg", "closed message needs context"));
  require(closed.pull_tensors(10).empty(),
          run_api_case("closed_pull_tensors", "pull_tensors on a closed Run is empty"));
}

void push_close(const Tensor& seed) {
  Run push_run = make_async_rgb_run(seed, 4);
  require(push_run.push(TensorList{seed}),
          run_api_case("push_close_input", "test input should be accepted"));
  Sample output;
  PullError err;
  require(push_run.pull(1000, output, &err) == PullStatus::Ok,
          run_api_case("push_close_output", "accepted input should produce output"));
  push_run.close_input();
  require(push_run.pull(1000, output, &err) == PullStatus::Closed,
          run_api_case("push_close_status", "closed push input should drain normally"));
  require(
      err.code == error_codes::kRuntimePull,
      run_api_case("push_close_code", "application-driven EOS must not use the source-ended code"));
  push_run.close();
}

void active_matrix(const Tensor& seed) {
  Run run = make_async_rgb_run(seed, 32);

  require(run.pull_tensors(120).empty(),
          run_api_case("active_pull_tensors_timeout", "pull_tensors should time out empty"));
  Sample tmp;
  PullError err;
  require(run.pull(120, tmp, &err) == PullStatus::Timeout,
          run_api_case("active_pull_timeout_status", "Run::pull timeout status mismatch"));
  require(err.code == error_codes::kRuntimePull,
          run_api_case("active_pull_timeout_error_code", "timeout should set runtime.pull"));
  require_contains(
      err.message, "runtime.pull",
      run_api_case("active_pull_timeout_error_msg", "timeout message should include runtime.pull"));
  require(run.pull_tensors(120).empty(),
          run_api_case("active_pull_tensors_timeout_repeat", "pull_tensors stays empty"));

  require(run.push(TensorList{seed}), run_api_case("active_push_tensor", "push should succeed"));
  require(run.pull_tensors(1000).size() == 1,
          run_api_case("active_pull_tensors_hit", "pull_tensors should return one payload"));
  require(run.push(TensorList{seed}),
          run_api_case("active_push_tensor_for_miss", "push should succeed again"));
  require(run.pull_tensors(1000).size() == 1,
          run_api_case("active_pull_tensors_second_hit", "pull_tensors returns one again"));
  {
    const TensorList single_input = {seed};
    require(run.push(single_input),
            run_api_case("active_push_tensor_list", "push(TensorList) should succeed"));
    const TensorList list_out = run.pull_tensors(1000);
    require(list_out.size() == 1 && list_out.front().shape == seed.shape,
            run_api_case("active_pull_tensors_shape", "pull_tensors keeps the shape"));
    const TensorList roundtrip = run.run(single_input, 1000);
    require(roundtrip.size() == 1 && roundtrip.front().dtype == seed.dtype,
            run_api_case("active_run_tensor_list", "run(TensorList) returns one tensor"));
  }

  Sample copy_msg;
  copy_msg.kind = SampleKind::Tensor;
  copy_msg.tensor = seed;
  copy_msg.payload_type = PayloadType::Image;
  copy_msg.format = "RGB";
  copy_msg.attributes = {{"image-channel", "color"}, {"image-index", "41"}};
  require(run.try_push(Sample{copy_msg}),
          run_api_case("active_try_push_sample_copy", "try_push(Sample copy) should succeed"));
  const auto copy_out = run.pull(1000);
  require(copy_out.has_value() && !tensors_from_sample(*copy_out, true).empty(),
          run_api_case("active_try_push_sample_copy_output", "a tensor output is expected"));
  // Documented difference (c3.md section 4 row 5): no Sample attributes on Dataflow.
  require(copy_out->attributes.empty(),
          run_api_case("active_try_push_sample_copy_attributes",
                       "Dataflow carries no Sample attributes (documented)"));

  Sample image_msg = copy_msg;
  image_msg.kind = SampleKind::TensorSet;
  image_msg.tensor.reset();
  image_msg.tensors = {make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x37)};
  image_msg.pts_ns = 123456789;
  image_msg.dts_ns = 123450000;
  image_msg.duration_ns = 33333333;
  require(run.push(image_msg), "CPU image Sample push failed");
  {
    auto input = image_msg.tensors.front().storage->map(MapMode::Write);
    std::memset(input.data, 0x91, input.size_bytes);
  }
  const auto image_out = run.pull(1000);
  require(image_out.has_value(), "CPU image Sample output missing");
  require(image_out->pts_ns == image_msg.pts_ns, "CPU image snapshot keeps pts_ns");
  // Documented difference (c3.md section 4 row 5): dts_ns, duration_ns, attributes not carried.
  require(image_out->dts_ns == -1 && image_out->duration_ns == -1 && image_out->attributes.empty(),
          "Dataflow carries no dts_ns, duration_ns or attributes (documented)");
  const auto image_tensors = tensors_from_sample(*image_out, true);
  require(image_tensors.size() == 1, "CPU image snapshot output missing tensor");
  const auto pixels = image_tensors.front().to_cv_mat_copy(ImageSpec::PixelFormat::RGB);
  require(cv::norm(pixels, cv::Mat(pixels.size(), pixels.type(), cv::Scalar::all(0x37)),
                   cv::NORM_INF) == 0,
          "CPU image snapshot must remain independent of subsequent input writes");

  // try_push backpressure: the link (depth 128) fills; the refusal is fast, non-fatal, recovers.
  int attempts = 0;
  int max_call_ms = 0;
  bool saw_backpressure = false;
  for (int i = 0; i < 4096 && !saw_backpressure; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = run.try_push(TensorList{seed});
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0)
            .count();
    ++attempts;
    max_call_ms = std::max(max_call_ms, static_cast<int>(ms));
    saw_backpressure = !ok;
  }
  require(saw_backpressure, run_api_case("active_backpressure_signal",
                                         "try_push should eventually report backpressure"));
  require(attempts == 129, run_api_case("active_backpressure_depth",
                                        "the 129th try_push meets the full link of 128, got " +
                                            std::to_string(attempts)));
  require(max_call_ms < 50, run_api_case("active_backpressure_non_blocking",
                                         "try_push should remain non-blocking under pressure"));
  require(run.last_error().empty(), run_api_case("active_backpressure_nonfatal_error",
                                                 "backpressure must not latch an error"));
  require(run.running(), run_api_case("active_backpressure_nonfatal_running",
                                      "backpressure must leave the Run active"));
  bool recovered = false;
  const auto recovery_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < recovery_deadline) {
    Sample drained;
    PullError drain_err;
    (void)run.pull(10, drained, &drain_err);
    if (run.try_push(TensorList{seed})) {
      recovered = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  require(recovered, run_api_case("active_backpressure_recovery",
                                  "the Run accepts input after transient backpressure"));
  require(run.last_error().empty(),
          run_api_case("active_backpressure_recovery_error", "a recovered Run stays error-free"));

  run.close_input();
  require(!run.try_push(TensorList{seed}),
          run_api_case("after_close_input_try_push", "try_push fails after close_input"));
  run.stop();
  require(!run.try_push(TensorList{seed}),
          run_api_case("after_stop_try_push", "try_push fails after stop"));
  run.close();
}

// The GStreamer test's holder block, minus the holder pushes: the zero-timeout poll contract.
void zero_timeout_poll(const Tensor& seed) {
  Run run = make_async_rgb_run_with_copy_input(seed, 1);
  require(run.push(TensorList{seed}),
          run_api_case("active_holder_seed_push", "push(TensorList) should be accepted"));
  const Sample first_samples = run.pull_samples(1000);
  require(!tensors_from_sample(first_samples, true).empty(),
          run_api_case("active_holder_seed_pull", "pull_samples should return one sample"));
  require(run.last_error().empty(),
          run_api_case("active_holder_seed_pull_last_error",
                       "a pull_samples drain poll should not leave a timeout error"));
  Sample drain_probe;
  PullError drain_err;
  require(run.pull(0, drain_probe, &drain_err) == PullStatus::Timeout,
          run_api_case("active_holder_zero_timeout_poll_status",
                       "pull(0) reports Timeout when nothing is queued"));
  require(drain_err.code.empty() && drain_err.message.empty(),
          run_api_case("active_holder_zero_timeout_poll_error",
                       "pull(0) is a benign non-blocking poll; got code='" + drain_err.code +
                           "' message='" + drain_err.message + "'"));
  // The holder pushes of the GStreamer block: a structured error on this backend.
  const auto holder = std::make_shared<int>(1);
  require_neat_error([&] { (void)run.try_push_holder(holder); }, error_codes::kPipelineShape,
                     "holders");
  require_neat_error([&] { (void)run.push_holder(holder); }, error_codes::kPipelineShape,
                     "holders");
  run.stop();
  require(throws_with([&] { (void)run.try_push_holder(holder); }, "") ||
              !run.try_push(TensorList{seed}),
          run_api_case("holder_stale_after_stop", "no push is accepted after stop"));
  run.close();
}

void format_depth_mismatch() {
  Graph graph;
  graph.add(nodes::Input(rgb_input()));
  graph.add(nodes::Output(OutputOptions::EveryFrame(32)));
  const cv::Mat rgb_seed(48, 64, CV_8UC3, cv::Scalar(20, 40, 60));
  Run run = graph.build(std::vector<cv::Mat>{rgb_seed}, dataflow(8));
  bool threw = false;
  std::string msg;
  try {
    const cv::Mat gray(48, 64, CV_8UC1, cv::Scalar(70));
    (void)run.run(std::vector<cv::Mat>{gray}, 300);
  } catch (const std::exception& e) {
    threw = true;
    msg = e.what();
  }
  run.stop();
  require(threw, run_api_case("format_depth_mismatch_throw",
                              "Run should throw for an incompatible frame depth"));
  require(!msg.empty(), run_api_case("format_depth_mismatch_error_msg",
                                     "the mismatch needs non-empty diagnostics"));
  run.close();
}

void empty_input_build() {
  Graph graph;
  graph.add(nodes::Input(rgb_input()));
  graph.add(nodes::Output(OutputOptions::EveryFrame(32)));
  require(throws_with(
              [&] {
                const cv::Mat empty;
                (void)graph.build(std::vector<cv::Mat>{empty}, dataflow(8));
              },
              "empty image input at index 0"),
          run_api_case("empty_input_build", "Graph::build should reject an empty cv::Mat"));
}

} // namespace

RUN_TEST("unit_dataflow_twin_api_matrix_test", ([] {
           const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x58);
           closed_default_run(seed);
           push_close(seed);
           active_matrix(seed);
           zero_timeout_poll(seed);
           format_depth_mismatch();
           empty_input_build();
         }));

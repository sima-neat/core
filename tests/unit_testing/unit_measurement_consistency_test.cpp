#include "nodes/common/Caps.h"
#include "nodes/common/Output.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "pipeline/Run.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr int kScoredTrials = 3;
constexpr int kInitialWarmupSamples = 250;
constexpr int kInterTrialWarmupSamples = 50;
constexpr int kCalibrationSamples = 250;
constexpr int kMeasuredSamples = 400;
constexpr int kTimeoutMs = 5000;

simaai::neat::Tensor make_rgb_input(int frame) {
  return make_color_tensor(640, 480, simaai::neat::ImageSpec::PixelFormat::RGB,
                           static_cast<std::uint8_t>(0x30 + (frame & 0x7f)));
}

simaai::neat::Sample make_keyed_sample(const simaai::neat::Tensor& input, int frame, int trial) {
  simaai::neat::Sample sample = simaai::neat::sample_from_tensors(simaai::neat::TensorList{input});
  sample.frame_id = frame;
  sample.stream_id = "measurement-consistency-" + std::to_string(trial);
  return sample;
}

simaai::neat::Run make_single_flight_run(const simaai::neat::Tensor& seed) {
  using namespace simaai::neat;

  Graph graph;
  InputOptions src_opt;
  src_opt.payload_type = PayloadType::Image;
  src_opt.format = FormatTag::RGB;
  src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
  src_opt.max_width = 640;
  src_opt.max_height = 480;
  src_opt.max_depth = 3;
  graph.add(nodes::Input(src_opt));
  // Force a real RGB->BGR conversion at VGA size so node-latency accounting has measurable work.
  graph.add(nodes::VideoConvert());
  graph.add(nodes::CapsRaw("BGR", 640, 480));
  graph.add(nodes::Output(OutputOptions::EveryFrame(64)));

  RunOptions run_opt;
  run_opt.queue_depth = 8;
  run_opt.overflow_policy = OverflowPolicy::Block;
  run_opt.output_memory = OutputMemory::Owned;
  // Inputs are pre-created and kept alive for the whole measurement, so avoid measuring a large
  // host copy in Run::push(). The consistency check is about runtime measurement math, not memcpy.
  run_opt.advanced.copy_input = false;
  return graph.build(TensorList{seed}, run_opt);
}

simaai::neat::MeasureOptions make_measure_options() {
  simaai::neat::MeasureOptions opt;
  opt.duration_ms = 1;
  opt.warmup_ms = 0;
  opt.timeout_ms = kTimeoutMs;
  opt.include_plugin_latency = false;
  opt.include_edge_latency = false;
  opt.include_message_latency = false;
  opt.include_power = false;
  opt.logical_batch_size = 1;
  opt.title = "single-flight consistency";
  return opt;
}

void require_output(const std::optional<simaai::neat::Sample>& out, const std::string& where) {
  require(out.has_value(), where + ": pull timed out");
  require(!simaai::neat::tensors_from_sample(*out, true).empty(),
          where + ": output should contain tensors");
}

struct TrialStats {
  simaai::neat::MeasureReport report;
  double period_ms = 0.0;
  double node_sum_avg_ms = 0.0;
  double avg_push_call_ms = 0.0;
  double avg_pull_call_ms = 0.0;
  std::size_t node_rows_with_samples = 0;
};

void run_unmeasured_samples(simaai::neat::Run& run, const simaai::neat::Tensor& input, int count,
                            int trial, int frame_base, const std::string& where) {
  std::vector<simaai::neat::Sample> samples;
  samples.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    samples.push_back(make_keyed_sample(input, frame_base + i, trial));
  }
  for (int i = 0; i < count; ++i) {
    require(run.push(simaai::neat::Sample{samples[static_cast<std::size_t>(i)]}),
            where + ": push should succeed");
    require_output(run.pull(kTimeoutMs), where);
  }
}

TrialStats run_single_flight_trial(simaai::neat::Run& run, const simaai::neat::Tensor& input,
                                   int trial, int sample_count = kMeasuredSamples) {
  using namespace simaai::neat;

  std::vector<Sample> samples;
  samples.reserve(static_cast<std::size_t>(sample_count));
  const int frame_base = 1'000'000 + trial * 100'000;
  for (int i = 0; i < sample_count; ++i) {
    samples.push_back(make_keyed_sample(input, frame_base + i, trial));
  }

  double total_push_ms = 0.0;
  double total_pull_ms = 0.0;
  MeasureScope scope = run.start_measurement(make_measure_options());
  for (int i = 0; i < sample_count; ++i) {
    const auto push_begin = std::chrono::steady_clock::now();
    require(run.push(Sample{samples[static_cast<std::size_t>(i)]}),
            "measured single-flight push should succeed");
    const auto push_end = std::chrono::steady_clock::now();
    const auto pull_begin = push_end;
    require_output(run.pull(kTimeoutMs), "measured single-flight");
    const auto pull_end = std::chrono::steady_clock::now();
    total_push_ms += std::chrono::duration<double, std::milli>(push_end - push_begin).count();
    total_pull_ms += std::chrono::duration<double, std::milli>(pull_end - pull_begin).count();
  }
  TrialStats stats;
  stats.report = scope.stop();

  stats.period_ms = stats.report.throughput_batches_per_s > 0.0
                        ? 1000.0 / stats.report.throughput_batches_per_s
                        : 0.0;
  stats.avg_push_call_ms = total_push_ms / static_cast<double>(sample_count);
  stats.avg_pull_call_ms = total_pull_ms / static_cast<double>(sample_count);
  for (const auto& node : stats.report.node_metrics) {
    if (node.latency.samples == 0 || node.latency.avg_ms <= 0.0) {
      continue;
    }
    stats.node_sum_avg_ms += node.latency.avg_ms;
    ++stats.node_rows_with_samples;
  }

  std::cout << "[measurement_consistency] trial=" << trial << " outputs=" << stats.report.outputs
            << " e2e_count=" << stats.report.end_to_end.count
            << " e2e_avg_ms=" << stats.report.end_to_end.avg_ms
            << " e2e_p50_ms=" << stats.report.end_to_end.p50_ms
            << " frame_gap_count=" << stats.report.frame_gap.count
            << " frame_gap_avg_ms=" << stats.report.frame_gap.avg_ms
            << " tput_batches_s=" << stats.report.throughput_batches_per_s
            << " inverse_tput_ms=" << stats.period_ms << " abs_period_minus_e2e_ms="
            << std::abs(stats.period_ms - stats.report.end_to_end.avg_ms)
            << " avg_push_call_ms=" << stats.avg_push_call_ms
            << " avg_pull_call_ms=" << stats.avg_pull_call_ms
            << " node_sum_avg_ms=" << stats.node_sum_avg_ms
            << " node_rows=" << stats.node_rows_with_samples << "\n";
  return stats;
}

void require_single_flight_consistency(const TrialStats& stats, int trial) {
  const auto& report = stats.report;
  const std::string prefix = "trial " + std::to_string(trial) + ": ";

  require(report.outputs == static_cast<std::size_t>(kMeasuredSamples),
          prefix + "measured outputs should equal pushed samples");
  require(report.counters.outputs_pulled == static_cast<std::uint64_t>(kMeasuredSamples),
          prefix + "outputs_pulled should equal pushed samples");
  require(report.end_to_end.count == static_cast<std::size_t>(kMeasuredSamples),
          prefix + "one e2e latency sample should be collected per output");
  require(report.frame_gap.count == static_cast<std::size_t>(kMeasuredSamples - 1),
          prefix + "one frame-gap sample should be collected between each pair of outputs");
  require(report.graph_sample_timing_unkeyed == 0,
          prefix + "keyed measured samples should not be reported as unkeyed");
  require(report.graph_sample_timing_misses == 0,
          prefix + "keyed measured samples should not miss graph timing correlation");

  require(std::isfinite(report.elapsed_s) && report.elapsed_s > 0.0,
          prefix + "elapsed_s should be finite and positive");
  require(std::isfinite(report.end_to_end.avg_ms) && report.end_to_end.avg_ms > 0.0,
          prefix + "e2e latency should be finite and positive");
  require(std::isfinite(report.throughput_batches_per_s) && report.throughput_batches_per_s > 0.0,
          prefix + "throughput should be finite and positive");
  require(std::isfinite(stats.period_ms) && stats.period_ms > 0.0,
          prefix + "inverse throughput period should be finite and positive");
  require(std::isfinite(report.frame_gap.avg_ms) && report.frame_gap.avg_ms > 0.0,
          prefix + "frame-gap average should be finite and positive");

  // In this test the app does exactly one push followed by one pull at a time.  Inputs are
  // pre-created outside the observer window, so the only material app-side cost inside the window
  // is the public push/pull API overhead. That overhead is expected to be tiny relative to the
  // conversion work, so one output period (1 / throughput) should be within 1 ms of the measured
  // push->output latency.
  const double latency_ms = report.end_to_end.avg_ms;
  const double period_delta_ms = std::abs(stats.period_ms - latency_ms);
  require(period_delta_ms <= 1.0,
          prefix +
              "single-flight inverse throughput period should be within 1 ms of e2e "
              "latency after warmup and input pre-creation: period_ms=" +
              std::to_string(stats.period_ms) + " latency_ms=" + std::to_string(latency_ms) +
              " delta_ms=" + std::to_string(period_delta_ms));
  const double frame_gap_delta_ms = std::abs(report.frame_gap.avg_ms - stats.period_ms);
  require(frame_gap_delta_ms <= 1.0,
          prefix +
              "single-window inverse throughput period should be within 1 ms of measured output "
              "frame gaps: period_ms=" +
              std::to_string(stats.period_ms) +
              " frame_gap_avg_ms=" + std::to_string(report.frame_gap.avg_ms) +
              " delta_ms=" + std::to_string(frame_gap_delta_ms));
  require(stats.avg_push_call_ms <= 0.50,
          prefix + "average public push call should stay small; avg_push_call_ms=" +
              std::to_string(stats.avg_push_call_ms));

  require(stats.node_rows_with_samples > 0,
          prefix + "measured report should contain node latency rows with samples");
  require(std::isfinite(stats.node_sum_avg_ms) && stats.node_sum_avg_ms > 0.0,
          prefix + "sum of node average latencies should be finite and positive");
  // The graph forces a measurable RGB->BGR conversion, so summed node residency should track the
  // observed app-visible e2e latency closely enough to catch missing node accounting or unit
  // mistakes. It is still not bit-exact critical-path timing because app-visible e2e includes
  // queueing and push/pull overhead around the element.
  require(stats.node_sum_avg_ms + 0.50 >= latency_ms * 0.45,
          prefix + "sum of node latencies is too small relative to e2e latency: node_sum=" +
              std::to_string(stats.node_sum_avg_ms) + " latency_ms=" + std::to_string(latency_ms));
  require(stats.node_sum_avg_ms <= latency_ms * 1.75 + 1.0,
          prefix + "sum of node latencies is too large relative to e2e latency: node_sum=" +
              std::to_string(stats.node_sum_avg_ms) + " latency_ms=" + std::to_string(latency_ms));
}

} // namespace

RUN_TEST("unit_measurement_consistency_test", ([] {
           using namespace simaai::neat;

           const Tensor input = make_rgb_input(0);
           Run run = make_single_flight_run(input);

           // Keep this test about measurement math, not graph construction or first-use runtime
           // effects. CI machines can change CPU frequency/thermal state during the first few
           // hundred VGA conversions, which made the first rebuilt pipeline run faster than later
           // rebuilt runs. Warm and calibrate one persistent single-flight graph first, then score
           // repeated measured windows on that already-steady graph.
           run_unmeasured_samples(run, input, kInitialWarmupSamples, -100, 0,
                                  "initial single-flight warmup");
           const TrialStats calibration =
               run_single_flight_trial(run, input, -1, kCalibrationSamples);
           std::cout << "[measurement_consistency] calibration_outputs="
                     << calibration.report.outputs
                     << " calibration_e2e_avg_ms=" << calibration.report.end_to_end.avg_ms
                     << " calibration_inverse_tput_ms=" << calibration.period_ms << "\n";

           std::vector<TrialStats> trials;
           trials.reserve(kScoredTrials);
           for (int trial = 0; trial < kScoredTrials; ++trial) {
             run_unmeasured_samples(run, input, kInterTrialWarmupSamples, 100 + trial,
                                    2'000'000 + trial * 10'000, "inter-trial single-flight warmup");
             trials.push_back(run_single_flight_trial(run, input, trial));
             require_single_flight_consistency(trials.back(), trial);
           }
           run.close();

           // Cross-window latency can legitimately move on loaded CI because this test runs real
           // VGA conversion work. The contract we care about is now checked inside each measured
           // window above: e2e latency, output gaps, and throughput must agree for the same run.
           std::vector<double> latencies;
           latencies.reserve(trials.size());
           for (const auto& trial : trials) {
             latencies.push_back(trial.report.end_to_end.avg_ms);
           }
           const auto [min_it, max_it] = std::minmax_element(latencies.begin(), latencies.end());
           if (min_it != latencies.end() && max_it != latencies.end()) {
             std::cout << "[measurement_consistency] cross_window_latency_range_ms="
                       << (*max_it - *min_it) << " min_avg_ms=" << *min_it
                       << " max_avg_ms=" << *max_it << "\n";
           }
         }));

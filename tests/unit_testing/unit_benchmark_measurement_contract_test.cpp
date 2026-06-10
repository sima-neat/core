#include "asset_utils.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "pipeline/Run.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kSamples = 4;
constexpr int kTimeoutMs = 5000;

std::string read_text(const std::filesystem::path& path) {
  std::ifstream in(path);
  require(in.is_open(), "failed to open " + path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string join_failures(const std::vector<std::string>& failures) {
  std::ostringstream out;
  out << failures.size() << " benchmark measurement contract check(s) failed:";
  for (const auto& failure : failures) {
    out << "\n  - " << failure;
  }
  return out.str();
}

simaai::neat::Run make_benchmark_style_rgb_run(const simaai::neat::Tensor& seed) {
  using namespace simaai::neat;

  Graph graph;
  InputOptions src_opt;
  src_opt.payload_type = PayloadType::Image;
  src_opt.format = FormatTag::RGB;
  src_opt.use_simaai_pool = false;
  src_opt.max_width = 16;
  src_opt.max_height = 16;
  src_opt.max_depth = 3;
  graph.add(nodes::Input(src_opt));
  graph.add(nodes::Output(OutputOptions::EveryFrame(32)));

  RunOptions run_opt;
  run_opt.queue_depth = 8;
  run_opt.overflow_policy = OverflowPolicy::Block;
  run_opt.output_memory = OutputMemory::Owned;
  run_opt.advanced.copy_input = true;
  return graph.build(TensorList{seed}, run_opt);
}

simaai::neat::MeasureOptions make_fast_measure_options(int logical_batch_size = 1) {
  simaai::neat::MeasureOptions opt;
  opt.duration_ms = 1;
  opt.warmup_ms = 0;
  opt.timeout_ms = kTimeoutMs;
  opt.include_plugin_latency = false;
  opt.include_edge_latency = false;
  opt.include_message_latency = false;
  opt.include_power = false;
  opt.logical_batch_size = logical_batch_size;
  return opt;
}

simaai::neat::Tensor make_rgb_input(int frame) {
  return make_color_tensor(16, 16, simaai::neat::ImageSpec::PixelFormat::RGB,
                           static_cast<std::uint8_t>(0x40 + frame));
}

simaai::neat::Sample make_keyed_sample(int frame) {
  simaai::neat::Sample sample =
      simaai::neat::sample_from_tensors(simaai::neat::TensorList{make_rgb_input(frame)});
  sample.frame_id = frame;
  sample.stream_id = "benchmark-contract";
  return sample;
}

void require_measured_output(const std::optional<simaai::neat::Sample>& out,
                             const std::string& where) {
  require(out.has_value(), where + ": pull timed out");
  require(!simaai::neat::tensors_from_sample(*out, true).empty(),
          where + ": output should contain tensors");
}

void benchmark_style_tensorlist_push_must_collect_latency_for_every_output() {
  using namespace simaai::neat;

  const Tensor seed = make_rgb_input(0);
  Run run = make_benchmark_style_rgb_run(seed);
  MeasureScope scope = run.start_measurement(make_fast_measure_options());
  for (int i = 0; i < kSamples; ++i) {
    require(run.push(TensorList{make_rgb_input(i)}),
            "benchmark-style TensorList push should succeed");
    require_measured_output(run.pull(kTimeoutMs), "benchmark-style TensorList");
  }
  const MeasureReport report = scope.stop();
  run.close();

  require(report.counters.outputs_pulled == static_cast<std::uint64_t>(kSamples),
          "benchmark-style measured window should pull every output; got " +
              std::to_string(report.counters.outputs_pulled));
  require(report.end_to_end.count == static_cast<std::size_t>(kSamples),
          "benchmark-style TensorList pushes must produce one end-to-end latency sample per "
          "output; got " +
              std::to_string(report.end_to_end.count));
  require(report.graph_sample_timing_unkeyed == 0,
          "benchmark-style TensorList pushes must be stamped with a sample identity before graph "
          "entry; unkeyed=" +
              std::to_string(report.graph_sample_timing_unkeyed));
  require(report.graph_sample_timing_misses == 0,
          "benchmark-style TensorList pushes must not miss graph entry/output correlation; "
          "misses=" +
              std::to_string(report.graph_sample_timing_misses));
  require(report.latency_samples_collected,
          "benchmark-style measured window should mark latency samples collected");
  require(report.end_to_end.avg_ms > 0.0 && std::isfinite(report.end_to_end.avg_ms),
          "benchmark-style average latency should be finite and positive");
}

void keyed_sample_push_is_a_control_that_collects_exact_latency_samples() {
  using namespace simaai::neat;

  const Tensor seed = make_rgb_input(0);
  Run run = make_benchmark_style_rgb_run(seed);
  MeasureScope scope = run.start_measurement(make_fast_measure_options());
  for (int i = 0; i < kSamples; ++i) {
    require(run.push(Sample{make_keyed_sample(i)}), "keyed Sample push should succeed");
    require_measured_output(run.pull(kTimeoutMs), "keyed Sample");
  }
  const MeasureReport report = scope.stop();
  run.close();

  require(report.counters.outputs_pulled == static_cast<std::uint64_t>(kSamples),
          "keyed measured window should pull every output");
  require(report.end_to_end.count == static_cast<std::size_t>(kSamples),
          "keyed Sample control should collect exactly one latency sample per output; got " +
              std::to_string(report.end_to_end.count));
  require(report.graph_sample_timing_unkeyed == 0,
          "keyed Sample control should not produce unkeyed graph timing entries");
  require(report.graph_sample_timing_misses == 0,
          "keyed Sample control should not produce graph timing misses");
}

void logical_batch_size_controls_inference_throughput() {
  using namespace simaai::neat;

  constexpr int kLogicalBatch = 4;
  const Tensor seed = make_rgb_input(0);
  Run run = make_benchmark_style_rgb_run(seed);
  MeasureScope scope = run.start_measurement(make_fast_measure_options(kLogicalBatch));
  for (int i = 0; i < kSamples; ++i) {
    require(run.push(Sample{make_keyed_sample(i)}), "batched keyed Sample push should succeed");
    require_measured_output(run.pull(kTimeoutMs), "batched keyed Sample");
  }
  const MeasureReport report = scope.stop();
  run.close();

  require(report.options.logical_batch_size == kLogicalBatch,
          "MeasureReport should preserve positive logical_batch_size");
  require(report.throughput_batches_per_s > 0.0,
          "batch throughput should be positive after measured outputs");
  const double expected =
      report.throughput_batches_per_s * static_cast<double>(report.options.logical_batch_size);
  require(std::abs(report.throughput_inferences_per_s - expected) <=
              std::max(1e-9, expected * 1e-9),
          "logical inference throughput must equal batch throughput multiplied by "
          "logical_batch_size");
}

void graph_sample_timing_counters_are_measurement_window_deltas() {
  using namespace simaai::neat;

  const Tensor seed = make_rgb_input(0);
  Run run = make_benchmark_style_rgb_run(seed);

  // Deliberately create run-lifetime graph timing pollution before the measured window.  This
  // TensorList push has no public frame_id yet when graph entry is recorded, so it increments the
  // lifetime unkeyed/miss counters.  A later clean measured window must not inherit those counts.
  require(run.push(TensorList{make_rgb_input(100)}),
          "pre-measure unkeyed TensorList push should succeed");
  require_measured_output(run.pull(kTimeoutMs), "pre-measure unkeyed TensorList");

  MeasureScope scope = run.start_measurement(make_fast_measure_options());
  for (int i = 0; i < kSamples; ++i) {
    require(run.push(Sample{make_keyed_sample(200 + i)}),
            "measured keyed Sample push should succeed");
    require_measured_output(run.pull(kTimeoutMs), "measured keyed counter-delta Sample");
  }
  const MeasureReport report = scope.stop();
  run.close();

  require(report.counters.outputs_pulled == static_cast<std::uint64_t>(kSamples),
          "counter-delta measured window should pull every output");
  require(report.end_to_end.count == static_cast<std::size_t>(kSamples),
          "counter-delta measured window should collect one e2e latency sample per output; got " +
              std::to_string(report.end_to_end.count));
  require(report.graph_sample_timing_unkeyed == 0,
          "graph_sample_timing_unkeyed should be a measured-window delta, not a run-lifetime "
          "counter; got " +
              std::to_string(report.graph_sample_timing_unkeyed));
  require(report.graph_sample_timing_misses == 0,
          "graph_sample_timing_misses should be a measured-window delta, not a run-lifetime "
          "counter; got " +
              std::to_string(report.graph_sample_timing_misses));
}

void model_benchmark_wrapper_must_publish_measured_logical_fps_and_real_latency() {
  const std::filesystem::path root = sima_test::test_source_root();
  const std::filesystem::path model_cpp_path = root / "src/model/Model.cpp";
  if (!std::filesystem::exists(root / "CMakeLists.txt") ||
      !std::filesystem::exists(model_cpp_path)) {
    std::cout << "[INFO] complete source tree unavailable at " << root
              << "; skipping Model::benchmark source contract guard\n";
    return;
  }

  const std::string model_cpp = read_text(model_cpp_path);
  const std::size_t options_pos = model_cpp.find("MeasureOptions make_benchmark_measure_options");
  const std::size_t benchmark_pos = model_cpp.find("BenchmarkReport Model::benchmark");
  require(benchmark_pos != std::string::npos,
          "source guard could not locate Model::benchmark in Model.cpp");
  const std::size_t region_begin =
      options_pos == std::string::npos ? benchmark_pos : std::min(options_pos, benchmark_pos);
  const std::string benchmark_region = model_cpp.substr(region_begin);

  std::vector<std::string> failures;
  if (benchmark_region.find("report.fps = measured.throughput_batches_per_s") !=
      std::string::npos) {
    failures.push_back(
        "Model::benchmark must not publish batches/s as BenchmarkReport::fps; publish logical "
        "inferences/s instead");
  }
  if (benchmark_region.find("report.fps = measured.throughput_inferences_per_s") ==
      std::string::npos) {
    failures.push_back("Model::benchmark should assign BenchmarkReport::fps from "
                       "MeasureReport::throughput_inferences_per_s");
  }
  if (benchmark_region.find("logical_batch_size") == std::string::npos) {
    failures.push_back(
        "Model::benchmark should populate MeasureOptions::logical_batch_size from the compiled "
        "model/input contract before measuring throughput");
  }
  if (benchmark_region.find("measured.end_to_end.count") == std::string::npos &&
      benchmark_region.find("measured.latency_samples_collected") == std::string::npos) {
    failures.push_back(
        "Model::benchmark should validate that MeasureReport contains end-to-end latency "
        "samples before publishing BenchmarkReport::latency_ms");
  }
  require(failures.empty(), join_failures(failures));
}

} // namespace

RUN_TEST("unit_benchmark_measurement_contract_test", ([] {
           std::vector<std::string> failures;
           const auto run_case = [&](std::string name, const std::function<void()>& fn) {
             try {
               fn();
             } catch (const std::exception& e) {
               failures.push_back(std::move(name) + ": " + e.what());
             }
           };

           run_case("benchmark_style_tensorlist_latency",
                    benchmark_style_tensorlist_push_must_collect_latency_for_every_output);
           run_case("keyed_sample_latency_control",
                    keyed_sample_push_is_a_control_that_collects_exact_latency_samples);
           run_case("logical_batch_throughput", logical_batch_size_controls_inference_throughput);
           run_case("graph_sample_timing_counter_deltas",
                    graph_sample_timing_counters_are_measurement_window_deltas);
           run_case("model_benchmark_wrapper_contract",
                    model_benchmark_wrapper_must_publish_measured_logical_fps_and_real_latency);

           require(failures.empty(), join_failures(failures));
         }));

#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "graph/Graph.h"
#include "graph/GraphBuild.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"
#include "model/internal/ModelInternal.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "pipeline/Run.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kSamples = 4;
constexpr int kTimeoutMs = 5000;

class PassThroughStage final : public simaai::neat::graph::StageExecutor {
public:
  void set_ports(const simaai::neat::graph::StagePorts& ports) override {
    out_port_ = ports.only_output();
  }

  void on_input(simaai::neat::graph::StageMsg&& msg,
                std::vector<simaai::neat::graph::StageOutMsg>& out) override {
    out.push_back(
        simaai::neat::graph::StageOutMsg{.out_port = out_port_, .sample = std::move(msg.sample)});
  }

private:
  simaai::neat::graph::PortId out_port_ = simaai::neat::graph::kInvalidPort;
};

std::shared_ptr<simaai::neat::graph::Node> make_graph_pass_node(const std::string& label) {
  using simaai::neat::graph::PortDesc;
  using simaai::neat::graph::nodes::StageNode;
  StageNode::StageExecutorFactory factory = [] { return std::make_unique<PassThroughStage>(); };
  std::vector<PortDesc> inputs = {PortDesc{.name = "in", .spec = simaai::neat::OutputSpec{}}};
  std::vector<PortDesc> outputs = {PortDesc{.name = "out", .spec = simaai::neat::OutputSpec{}}};
  return std::make_shared<StageNode>("PassThrough", std::move(factory), std::move(inputs),
                                     std::move(outputs), label);
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

simaai::neat::MeasureReport make_complete_measure_report(int samples, int logical_batch_size) {
  simaai::neat::MeasureReport report;
  report.elapsed_s = 2.0;
  report.outputs = samples;
  report.options.logical_batch_size = logical_batch_size;
  report.counters.inputs_pushed = static_cast<std::uint64_t>(samples);
  report.counters.outputs_ready = static_cast<std::uint64_t>(samples);
  report.counters.outputs_pulled = static_cast<std::uint64_t>(samples);
  report.latency_samples_collected = true;
  report.end_to_end.count = static_cast<std::size_t>(samples);
  report.end_to_end.avg_ms = 3.25;
  report.end_to_end.p50_ms = 3.0;
  report.end_to_end.p90_ms = 3.5;
  report.end_to_end.p95_ms = 3.75;
  report.end_to_end.p99_ms = 4.0;
  report.end_to_end.max_ms = 4.25;
  report.throughput_batches_per_s = 10.0;
  report.throughput_inferences_per_s =
      report.throughput_batches_per_s * static_cast<double>(logical_batch_size);
  return report;
}

bool throws_benchmark_projection_error(const simaai::neat::MeasureReport& latency,
                                       const simaai::neat::MeasureReport& throughput,
                                       int expected_samples, const std::string& needle) {
  try {
    (void)simaai::neat::internal::build_benchmark_report_from_measurements(latency, throughput,
                                                                           expected_samples);
  } catch (const std::exception& e) {
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
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

  // Deliberately create run-lifetime activity before the measured window.  A later measured window
  // must report measured-window deltas, not inherit run-lifetime counters.
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

void internal_graph_run_measurement_collects_e2e_latency_and_throughput() {
  using namespace simaai::neat;

  graph::Graph g;
  const graph::NodeId pass = g.add(make_graph_pass_node("measured-pass"));

  graph::GraphRunOptions run_opt;
  run_opt.edge_queue = 8;
  graph::GraphRun run = graph::build(std::move(g), run_opt);
  graph::GraphRun::Output out = run.output(pass);

  MeasureScope scope = run.start_measurement(make_fast_measure_options());
  for (int i = 0; i < kSamples; ++i) {
    require(run.push(pass, Sample{make_keyed_sample(300 + i)}),
            "GraphRun measured push should succeed");
    require_measured_output(out.pull(kTimeoutMs), "GraphRun measured pull");
  }
  const MeasureReport report = scope.stop();
  run.stop();

  require(report.counters.inputs_pushed == static_cast<std::uint64_t>(kSamples),
          "GraphRun measured window should count pushed inputs; got " +
              std::to_string(report.counters.inputs_pushed));
  require(report.counters.outputs_pulled == static_cast<std::uint64_t>(kSamples),
          "GraphRun measured window should count pulled outputs; got " +
              std::to_string(report.counters.outputs_pulled));
  require(report.end_to_end.count == static_cast<std::size_t>(kSamples),
          "GraphRun start_measurement should collect one e2e sample per output; got " +
              std::to_string(report.end_to_end.count));
  require(report.graph_sample_timing_unkeyed == 0,
          "GraphRun measured keyed samples should not be unkeyed");
  require(report.graph_sample_timing_misses == 0,
          "GraphRun measured keyed samples should not miss correlation");
  require(report.throughput_batches_per_s > 0.0, "GraphRun measured throughput should be positive");
  require(report.plugin_latency_status == "off" && report.plugin_latency_source == "none",
          "GraphRun default measurement must keep plugin latency off");
}

void model_benchmark_projection_uses_same_window_reports() {
  using namespace simaai::neat;

  MeasureReport latency = make_complete_measure_report(kSamples, /*logical_batch_size=*/4);
  latency.end_to_end.avg_ms = 7.5;
  latency.throughput_batches_per_s = 1.0;
  latency.throughput_inferences_per_s = 4.0;

  MeasureReport throughput = make_complete_measure_report(kSamples, /*logical_batch_size=*/4);
  throughput.throughput_batches_per_s = 12.5;
  throughput.throughput_inferences_per_s = 50.0;
  throughput.end_to_end.avg_ms = 99.0;
  throughput.power.enabled = true;
  throughput.power.samples = 3;
  throughput.power.total_avg_watts = 8.25;
  throughput.power.energy_joules = 1.5;

  const BenchmarkReport report =
      internal::build_benchmark_report_from_measurements(latency, throughput, kSamples);

  require(report.latency_ms == 7.5,
          "Model::benchmark projection must publish latency from latency measurement window");
  require(report.fps == 50.0,
          "Model::benchmark projection must publish logical inferences/s from throughput window");
  require(report.avg_power_watts == 8.25,
          "Model::benchmark projection should preserve measured throughput-window power");
  require(report.energy_joules == 1.5,
          "Model::benchmark projection should preserve measured throughput-window energy");
}

void model_benchmark_projection_rejects_unreliable_correlation() {
  using namespace simaai::neat;

  MeasureReport latency = make_complete_measure_report(kSamples, /*logical_batch_size=*/2);
  MeasureReport throughput = make_complete_measure_report(kSamples, /*logical_batch_size=*/2);
  latency.graph_sample_timing_misses = 1;
  require(throws_benchmark_projection_error(latency, throughput, kSamples, "unreliable"),
          "Model::benchmark projection must reject latency windows with correlation misses");

  latency = make_complete_measure_report(kSamples, /*logical_batch_size=*/2);
  throughput.graph_sample_timing_unkeyed = 1;
  require(throws_benchmark_projection_error(latency, throughput, kSamples, "unreliable"),
          "Model::benchmark projection must reject throughput windows with unkeyed samples");
}

void model_benchmark_projection_rejects_bad_batch_arithmetic() {
  using namespace simaai::neat;

  MeasureReport latency = make_complete_measure_report(kSamples, /*logical_batch_size=*/4);
  MeasureReport throughput = make_complete_measure_report(kSamples, /*logical_batch_size=*/4);
  throughput.throughput_inferences_per_s = throughput.throughput_batches_per_s;
  require(throws_benchmark_projection_error(latency, throughput, kSamples, "logical batch size"),
          "Model::benchmark projection must reject batches/s reported as logical inferences/s");
}

void measurement_boolean_flags_make_profiling_explicit() {
  using namespace simaai::neat;

  const MeasureOptions default_opt;
  require(!default_opt.include_plugin_latency,
          "default MeasureOptions must not enable plugin profiling");
  require(!default_opt.include_edge_latency,
          "default MeasureOptions must not enable graph queue/edge diagnostics");
  require(!default_opt.include_message_latency,
          "default MeasureOptions must not enable message tracing");
  require(!default_opt.include_power,
          "default MeasureOptions must not start measurement-local power monitoring");

  const Tensor seed = make_rgb_input(0);
  Run run = make_benchmark_style_rgb_run(seed);
  MeasureScope scope = run.start_measurement(/*include_plugin_latency=*/false);
  require(run.push(TensorList{make_rgb_input(1)}), "e2e-only measured push should succeed");
  require_measured_output(run.pull(kTimeoutMs), "e2e-only measured pull");
  const MeasureReport report = scope.stop();
  run.close();
  require(report.plugin_latency_status == "off" && report.plugin_latency_source == "none",
          "e2e-only measurement must report plugin latency off/none");
  require(report.plugin_latency.empty() && report.plugin_latency_unattributed.empty(),
          "e2e-only measurement must not collect plugin latency rows");
  require(report.message_latency_status == "off" && report.message_latency_source == "none",
          "e2e-only measurement must report message tracing off/none");

  MeasureOptions plugin_opt;
  plugin_opt.include_plugin_latency = true;
  require(plugin_opt.include_plugin_latency,
          "plugin latency must be a simple explicit boolean in MeasureOptions");
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
           run_case("internal_graph_run_measurement",
                    internal_graph_run_measurement_collects_e2e_latency_and_throughput);
           run_case("model_benchmark_projection",
                    model_benchmark_projection_uses_same_window_reports);
           run_case("model_benchmark_projection_rejects_unreliable_correlation",
                    model_benchmark_projection_rejects_unreliable_correlation);
           run_case("model_benchmark_projection_rejects_bad_batch_arithmetic",
                    model_benchmark_projection_rejects_bad_batch_arithmetic);
           run_case("measurement_boolean_flags_make_profiling_explicit",
                    measurement_boolean_flags_make_profiling_explicit);

           require(failures.empty(), join_failures(failures));
         }));

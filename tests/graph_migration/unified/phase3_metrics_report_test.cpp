#include "graph_migration/common/phase3_graph_test_utils.h"
#include "nodes/common/Output.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/io/Input.h"
#include "pipeline/GraphMetrics.h"
#include "pipeline/runtime/RunCore.h"
#include "test_main.h"
#include "test_utils.h"

RUN_TEST("graph_migration_phase3_metrics_report_test", [] {
  simaai::neat::Graph input;
  input.add(simaai::neat::nodes::Input());
  simaai::neat::Graph output;
  output.add(simaai::neat::nodes::Output());
  simaai::neat::Graph app;
  app.connect(input, output);

  simaai::neat::RunOptions opt;
  opt.enable_metrics = true;
  opt.output_memory = simaai::neat::OutputMemory::Owned;
  simaai::neat::Run run = app.build(opt);

  require(run.push(simaai::neat::Sample{graph_phase3_test::make_tensor_sample(7, "metrics")}),
          "metrics Graph push failed");
  auto out = run.pull(5000);
  require(out.has_value(), "metrics Graph pull produced no sample");
  graph_phase3_test::require_sample_tensor_output(*out, "metrics Graph output");

  const simaai::neat::RuntimeMetrics metrics = run.metrics();
  require(!metrics.source_kind.empty(), "Run::metrics source_kind should be populated");
  require(metrics.counters.inputs_enqueued > 0 || metrics.counters.outputs_pulled > 0 ||
              !metrics.groups.empty(),
          "Run::metrics should include counters or metric groups after push/pull");

  const std::string metrics_report = run.metrics_report();
  require(!metrics_report.empty(), "Run::metrics_report should be non-empty");
  const std::string report = run.report();
  require(!report.empty(), "Run::report should be non-empty");
  require(report.find("pipeline_state_") == std::string::npos,
          "public report must not expose split pipeline_state_ internals");
  require(report.find("graph_state_") == std::string::npos,
          "public report must not expose split graph_state_ internals");
  run.stop();

  {
    simaai::neat::Graph power_input;
    power_input.add(simaai::neat::nodes::Input());
    simaai::neat::Graph power_output;
    power_output.add(simaai::neat::nodes::Output());
    simaai::neat::Graph power_app;
    power_app.connect(power_input, power_output);

    simaai::neat::RunOptions power_opt;
    power_opt.enable_metrics = true;
    power_opt.power_monitor.enabled = true;
    power_opt.power_monitor.profile = simaai::neat::PowerMonitorProfile::Custom;
    power_opt.power_monitor.sample_interval_ms = 10000;
    power_opt.power_monitor.rails.push_back(simaai::neat::PowerRailConfig{
        .name = "unit-test-invalid-rail",
        .i2c_bus = 99,
        .i2c_addr = 0x4f,
        .page = 0,
        .vout_exponent = -8,
        .iout_exponent = -6,
        .pout_exponent = -5,
    });

    simaai::neat::Run power_run = power_app.build(power_opt);
    const simaai::neat::PowerSummary power = power_run.power_summary();
    require(power.enabled, "RunOptions::power_monitor should start one graph-level monitor");
    require(power.rails.size() == 1U,
            "graph-level power monitor should retain caller-provided custom rail config");
    require(power.rails.front().config.name == "unit-test-invalid-rail",
            "power summary should expose the configured custom rail");
    simaai::neat::MeasureOptions measure_opt;
    measure_opt.duration_ms = 1;
    measure_opt.include_plugin_latency = false;
    measure_opt.include_power = true;
    simaai::neat::MeasureScope measure = power_run.start_measurement(measure_opt);
    const simaai::neat::MeasureReport measured = measure.stop();
    require(measured.power.enabled,
            "measurement report should use a measurement-local graph power monitor");
    require(measured.power.rails.size() == 1U,
            "measurement-local power monitor should retain caller-provided custom rail config");
    power_run.close();
  }

  {
    simaai::neat::runtime::PipelineSegmentPlan segment;
    segment.id = 7;
    segment.node_ids.push_back(42);
    segment.nodes.push_back(simaai::neat::nodes::VideoConvert());
    simaai::neat::runtime::Provenance provenance;
    provenance.runtime_node = 42;
    provenance.segment_id = segment.id;
    segment.provenance.push_back(provenance);

    const auto mapping =
        simaai::neat::runtime::make_materialized_node_attribution(segment, true, true);
    require(mapping.size() == 3U,
            "synthetic materialized mapping should include injected input, real node, output");
    require(mapping[0].role ==
                    simaai::neat::runtime::MaterializedNodeAttribution::Role::InjectedInput &&
                mapping[0].runtime_node == simaai::neat::graph::kInvalidNode,
            "injected input must stay explicitly unattributed");
    require(mapping[1].role ==
                    simaai::neat::runtime::MaterializedNodeAttribution::Role::SegmentNode &&
                mapping[1].segment_node_index == 0U && mapping[1].runtime_node == 42,
            "real materialized node must retain its runtime-node attribution");
    require(mapping[2].role ==
                    simaai::neat::runtime::MaterializedNodeAttribution::Role::InjectedOutput &&
                mapping[2].runtime_node == simaai::neat::graph::kInvalidNode,
            "injected output must stay explicitly unattributed");
  }

  {
    simaai::neat::Graph timing_graph;
    simaai::neat::InputOptions src_opt;
    src_opt.payload_type = simaai::neat::PayloadType::Image;
    src_opt.format = simaai::neat::FormatTag::RGB;
    src_opt.use_simaai_pool = false;
    src_opt.max_width = 16;
    src_opt.max_height = 16;
    src_opt.max_depth = 3;
    timing_graph.add(simaai::neat::nodes::Input(src_opt));
    timing_graph.add(simaai::neat::nodes::VideoConvert());
    timing_graph.add(simaai::neat::nodes::Output(simaai::neat::OutputOptions::EveryFrame(4)));

    simaai::neat::RunOptions timing_opt;
    timing_opt.enable_metrics = true;
    timing_opt.output_memory = simaai::neat::OutputMemory::Owned;
    timing_opt.advanced.copy_input = true;
    simaai::neat::Tensor seed = graph_phase3_test::make_rgb_tensor(16, 16, 0x43);
    simaai::neat::Run timing_run = timing_graph.build(simaai::neat::TensorList{seed},
                                                      simaai::neat::RunMode::Async, timing_opt);
    const simaai::neat::RunDiagSnapshot timing = timing_run.diag_snapshot();
    require(!timing.element_timings.empty(),
            "RunOptions::enable_metrics should attach element timing probes without "
            "SIMA_GST_ELEMENT_TIMINGS");
    const simaai::neat::GraphMetricsReport graph_metrics =
        simaai::neat::build_graph_metrics_report_run_lifetime(timing_run);
    require(graph_metrics.aggregation == "run_lifetime",
            "graph metrics should label run-lifetime aggregation");
    require(graph_metrics.latency_semantics == "sum_element_residency",
            "graph metrics should label node latency semantics");
    require(!graph_metrics.node_metrics.empty(),
            "graph metrics builder should produce node metric rows from node reports");
    bool saw_element_metric = false;
    for (const auto& node : graph_metrics.node_metrics) {
      saw_element_metric = saw_element_metric || !node.elements.empty();
    }
    require(saw_element_metric, "graph metrics should retain per-element latency summaries");

    simaai::neat::MeasureOptions node_measure_opt;
    node_measure_opt.duration_ms = 1;
    node_measure_opt.include_plugin_latency = false;
    node_measure_opt.include_power = false;
    simaai::neat::MeasureScope node_measure = timing_run.start_measurement(node_measure_opt);
    require(timing_run.push(simaai::neat::TensorList{seed}),
            "timing Graph measured-window push failed");
    auto measured_sample = timing_run.pull(5000);
    require(measured_sample.has_value(), "timing Graph measured-window pull produced no sample");
    const simaai::neat::MeasureReport measured_report = node_measure.stop();
    require(!measured_report.node_metrics.empty(),
            "measured report should include measured-window node latency deltas");
    bool saw_measured_minmax_unavailable = false;
    for (const auto& node : measured_report.node_metrics) {
      saw_measured_minmax_unavailable =
          saw_measured_minmax_unavailable || !node.latency.min_max_available;
    }
    require(saw_measured_minmax_unavailable,
            "measured-window node deltas should mark cumulative min/max unavailable");
    timing_run.close();
  }
});

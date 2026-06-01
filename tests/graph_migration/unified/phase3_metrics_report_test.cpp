#include "graph_migration/common/phase3_graph_test_utils.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
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
});

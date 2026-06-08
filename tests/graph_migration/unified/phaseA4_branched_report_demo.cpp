#include "graph_migration/common/phase3_graph_test_utils.h"
#include "graphs/Fragments.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/RunExport.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

simaai::neat::Graph input_graph(const std::string& name) {
  simaai::neat::Graph g(name);
  g.add(simaai::neat::nodes::Input(name));
  return g;
}

simaai::neat::Graph output_graph(const std::string& name) {
  simaai::neat::Graph g(name);
  g.add(simaai::neat::nodes::Output(name));
  return g;
}

fs::path output_json_path() {
  if (const char* raw = std::getenv("BRANCHED_GRAPH_METRICS_JSON"); raw && *raw) {
    return fs::path(raw);
  }
  return fs::path("/workspace/tmp/metrics_outputs/branched/graph_metrics.json");
}

int frame_count() {
  if (const char* raw = std::getenv("BRANCHED_GRAPH_METRICS_FRAMES"); raw && *raw) {
    return std::max(1, std::atoi(raw));
  }
  return 25;
}

void push_one(simaai::neat::Run& run, const std::string& input, int frame,
              const std::string& stream) {
  require(run.push(input, simaai::neat::Sample{graph_phase3_test::make_tensor_sample(frame, stream,
                                                                                     64, 48)}),
          "push failed on " + input + ": " + run.last_error());
}

void pull_one(simaai::neat::Run& run, const std::string& output) {
  auto sample = run.pull(output, 5000);
  require(sample.has_value(), "pull timed out on " + output + ": " + run.last_error());
}

void run_branched_report_demo() {
  using namespace simaai::neat;

  const std::vector<std::string> inputs = {"camera_left", "camera_right", "metadata"};
  const std::vector<std::string> outputs = {"left_preview", "left_model", "right_preview",
                                            "right_model",  "meta_debug", "meta_archive"};

  Graph app("branched_report_demo");

  Graph left_source = input_graph("camera_left");
  Graph left_branch = graphs::Branch("camera_left", {"left_preview", "left_model"});
  Graph left_preview = output_graph("left_preview");
  Graph left_model = output_graph("left_model");
  app.connect(left_source, left_branch);
  app.connect(left_branch, left_preview);
  app.connect(left_branch, left_model);

  Graph right_source = input_graph("camera_right");
  Graph right_branch = graphs::Branch("camera_right", {"right_preview", "right_model"});
  Graph right_preview = output_graph("right_preview");
  Graph right_model = output_graph("right_model");
  app.connect(right_source, right_branch);
  app.connect(right_branch, right_preview);
  app.connect(right_branch, right_model);

  Graph meta_source = input_graph("metadata");
  Graph meta_branch = graphs::Branch("metadata", {"meta_debug", "meta_archive"});
  Graph meta_debug = output_graph("meta_debug");
  Graph meta_archive = output_graph("meta_archive");
  app.connect(meta_source, meta_branch);
  app.connect(meta_branch, meta_debug);
  app.connect(meta_branch, meta_archive);

  RunOptions run_opt;
  run_opt.enable_metrics = true;
  Run run = app.build(run_opt);

  std::cout << "BRANCHED_DEMO_INPUTS=";
  for (const auto& input : run.input_names()) {
    std::cout << input << ",";
  }
  std::cout << "\nBRANCHED_DEMO_OUTPUTS=";
  for (const auto& output : run.output_names()) {
    std::cout << output << ",";
  }
  std::cout << "\n";

  // Warm up with a valid non-negative frame id so timestamp conversion remains valid.
  for (const auto& input : inputs) {
    push_one(run, input, 0, input + "_warmup");
  }
  for (const auto& output : outputs) {
    pull_one(run, output);
  }

  MeasureOptions measure_opt;
  measure_opt.title = "Branched public graph metrics demo";
  measure_opt.model = "synthetic three-way Branch fanout demo";
  measure_opt.input = "camera_left,camera_right,metadata";
  measure_opt.placement =
      "camera_left -> Branch(left_preview,left_model); camera_right -> "
      "Branch(right_preview,right_model); metadata -> Branch(meta_debug,meta_archive)";
  measure_opt.logical_batch_size = 1;
  measure_opt.include_plugin_latency = false;
  measure_opt.include_power = false;
  measure_opt.include_edge_latency = true;
  measure_opt.timeout_ms = 5000;
  measure_opt.warmup_ms = 0;

  auto scope = run.start_measurement(measure_opt);
  const int frames = frame_count();
  for (int frame = 1; frame <= frames; ++frame) {
    for (const auto& input : inputs) {
      push_one(run, input, frame, input + "_stream");
    }
    for (const auto& output : outputs) {
      pull_one(run, output);
    }
  }
  MeasureReport report = scope.stop();
  std::cout << report.to_text();

  RunExportOptions export_opt;
  export_opt.label = "branched-report-demo";
  export_opt.include_metrics = true;
  export_opt.include_node_metrics = true;
  export_opt.include_empty_node_metrics = true;
  export_opt.include_plugin_metrics = false;
  export_opt.include_power = false;
  export_opt.metadata.emplace_back("app", "graph_migration_phaseA4_branched_report_demo");
  export_opt.metadata.emplace_back("purpose", "branch visualization sanity check");

  const fs::path json_path = output_json_path();
  if (!json_path.parent_path().empty()) {
    fs::create_directories(json_path.parent_path());
  }
  std::string err;
  require(save_run_json(run, report, json_path.string(), export_opt, &err),
          "save_run_json failed: " + err);
  std::cout << "BRANCHED_DEMO_JSON=" << json_path << "\n";
  run.close();
}

} // namespace

RUN_TEST("graph_migration_phaseA4_branched_report_demo", [] { run_branched_report_demo(); });

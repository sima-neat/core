// Three diagnostic commands: Graph::validate, Run::stats, Run::report / diagnostics_summary.
//
// Usage:
//   tutorial_011_diagnose_a_pipeline

#include "neat.h"

#include <opencv2/core.hpp>

#include <iostream>
#include <stdexcept>

int main() {
  try {
    cv::Mat rgb(96, 128, CV_8UC3, cv::Scalar(22, 44, 66));
    if (!rgb.isContinuous())
      rgb = rgb.clone();

    simaai::neat::Graph graph;
    simaai::neat::InputOptions in;
    in.format = "RGB";
    in.width = rgb.cols;
    in.height = rgb.rows;
    in.depth = rgb.channels();
    graph.add(simaai::neat::nodes::Input(in));
    graph.add(simaai::neat::nodes::Output());

    // CORE LOGIC
    // STEP validate-graph
    // validate() checks the Graph before build() and prints any caps problems.
    auto report = graph.validate();
    std::cout << "validate.error_code=" << report.error_code << "\n";
    // END STEP

    // STEP run-with-metrics
    // Run the Graph with metrics enabled so stats() has data.
    simaai::neat::RunOptions run_opt;
    run_opt.enable_metrics = true;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;
    auto run = graph.build(std::vector<cv::Mat>{rgb}, simaai::neat::RunMode::Sync, run_opt);
    simaai::neat::TensorList out = run.run(std::vector<cv::Mat>{rgb}, /*timeout_ms=*/1000);
    if (out.empty())
      throw std::runtime_error("missing output tensor");
    // END STEP

    // STEP read-diagnostics
    // Post-run diagnostics: counters, per-element report, and a summary string.
    auto stats = run.stats();
    std::cout << "stats.inputs_enqueued=" << stats.inputs_enqueued
              << " outputs_pulled=" << stats.outputs_pulled << "\n";
    std::cout << "report.size=" << run.report().size() << "\n";
    std::cout << "diagnostics_summary=" << run.diagnostics_summary() << "\n";
    // END STEP
    // END CORE LOGIC

    std::cout << "[OK] 011_diagnose_a_pipeline\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

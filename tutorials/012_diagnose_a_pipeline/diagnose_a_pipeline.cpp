// Two diagnostic commands: Graph::validate and Run::start_measurement.
//
// Usage:
//   tutorial_012_diagnose_a_pipeline

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

    // STEP run-with-measurement
    // Build a reusable runner and measure the caller-owned workload.
    simaai::neat::RunOptions run_opt;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;
    auto run = graph.build(std::vector<cv::Mat>{rgb}, run_opt);
    simaai::neat::MeasureOptions measure_opt;
    measure_opt.title = "tutorial 011 diagnosis";
    auto scope = run.start_measurement(measure_opt);
    simaai::neat::TensorList out = run.run(std::vector<cv::Mat>{rgb}, /*timeout_ms=*/1000);
    if (out.empty())
      throw std::runtime_error("missing output tensor");
    const simaai::neat::MeasureReport measured = scope.stop();
    // END STEP

    // STEP read-diagnostics
    // Post-run diagnostics come from the measurement report.
    std::cout << "measure.inputs_enqueued=" << measured.counters.inputs_enqueued
              << " outputs_pulled=" << measured.counters.outputs_pulled << "\n";
    std::cout << "measure.text_size=" << measured.to_text().size() << "\n";
    // END STEP
    // END CORE LOGIC

    std::cout << "[OK] 012_diagnose_a_pipeline\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

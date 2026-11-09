// Tune async Graph throughput via RunOptions and MeasureReport.
//
// Usage:
//   tutorial_016_tune_throughput_and_queues [--iters 32] [--queue 4] [--drop block|latest|incoming]

#include "neat.h"

#include <opencv2/core.hpp>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string value;
  if (!get_arg(argc, argv, key, value))
    return def;
  return std::stoi(value);
}

simaai::neat::OverflowPolicy parse_drop_policy(int argc, char** argv) {
  std::string mode;
  if (!get_arg(argc, argv, "--drop", mode))
    return simaai::neat::OverflowPolicy::Block;
  if (mode == "latest")
    return simaai::neat::OverflowPolicy::KeepLatest;
  if (mode == "incoming")
    return simaai::neat::OverflowPolicy::DropIncoming;
  return simaai::neat::OverflowPolicy::Block;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const int iters = parse_int_arg(argc, argv, "--iters", 32);
    const int queue_depth = parse_int_arg(argc, argv, "--queue", 4);

    cv::Mat rgb(120, 160, CV_8UC3, cv::Scalar(70, 20, 200));
    if (!rgb.isContinuous())
      rgb = rgb.clone();

    simaai::neat::Graph graph;
    simaai::neat::InputOptions in;
    in.format = "RGB";
    in.width = rgb.cols;
    in.height = rgb.rows;
    in.depth = rgb.channels();
    in.is_live = true;
    graph.add(simaai::neat::nodes::Input(in));
    graph.add(simaai::neat::nodes::Output());

    // The frame as a Tensor: RGB pixels in EV74 memory, pushed in a TensorList.
    const simaai::neat::Tensor frame = simaai::neat::Tensor::from_cv_mat(
        rgb, simaai::neat::ImageSpec::PixelFormat::RGB, simaai::neat::TensorMemory::EV74);

    // CORE LOGIC
    // RunOptions controls how the async runner buffers and drops frames.
    // STEP configure-run-options
    simaai::neat::RunOptions opt;
    opt.queue_depth = queue_depth;
    opt.overflow_policy = parse_drop_policy(argc, argv);
    opt.output_memory = simaai::neat::OutputMemory::Owned;

    auto run = graph.build(simaai::neat::TensorList{frame}, opt);
    // END STEP

    // STEP push-workload
    // try_push never blocks; pair it with close_input + drain pull loop.
    simaai::neat::MeasureOptions measure_opt;
    measure_opt.title = "tutorial 016 throughput";
    auto scope = run.start_measurement(measure_opt);
    for (int i = 0; i < iters; ++i)
      (void)run.try_push(simaai::neat::TensorList{frame});
    run.close_input();

    // Drain until Timeout or Closed; an Error is a failure.
    int pulled = 0;
    auto out = run.pull(std::chrono::milliseconds(1000));
    while (out) {
      ++pulled;
      out = run.pull(std::chrono::milliseconds(1000));
    }
    if (out.status() == simaai::neat::PullStatus::Error)
      throw std::runtime_error("pull failed: " + out.error().message);
    const auto measured = scope.stop();
    if (measured.counters.inputs_enqueued <= 0 || pulled <= 0)
      throw std::runtime_error("throughput run produced no measured inputs/outputs");
    // END STEP
    // END CORE LOGIC

    // STEP read-measurement
    std::cout << "inputs_enqueued=" << measured.counters.inputs_enqueued << "\n";
    std::cout << "inputs_dropped=" << measured.counters.inputs_dropped << "\n";
    std::cout << "outputs_pulled=" << pulled << "\n";
    std::cout << "avg_latency_ms=" << measured.end_to_end.avg_ms << "\n";
    std::cout << "avg_push_us=" << measured.input.avg_push_us << "\n";
    std::cout << "renegotiations=" << measured.input.renegotiations << "\n";
    // END STEP
    std::cout << "[OK] 016_tune_throughput_and_queues\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

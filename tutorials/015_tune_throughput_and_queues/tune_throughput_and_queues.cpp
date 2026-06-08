// Tune async Graph throughput via RunOptions: queue_depth, overflow_policy, metrics.
//
// Usage:
//   tutorial_015_tune_throughput_and_queues [--iters 32] [--queue 4] [--drop block|latest|incoming]

#include "neat.h"

#include <opencv2/core.hpp>

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

    // CORE LOGIC
    // RunOptions controls how the async runner buffers and drops frames.
    // STEP configure-run-options
    simaai::neat::RunOptions opt;
    opt.queue_depth = queue_depth;
    opt.overflow_policy = parse_drop_policy(argc, argv);
    opt.output_memory = simaai::neat::OutputMemory::Owned;
    opt.enable_metrics = true;

    auto run = graph.build(std::vector<cv::Mat>{rgb}, simaai::neat::RunMode::Async, opt);
    // END STEP

    // STEP push-workload
    // try_push never blocks; pair it with close_input + drain pull loop.
    for (int i = 0; i < iters; ++i)
      (void)run.try_push(std::vector<cv::Mat>{rgb});
    run.close_input();

    int pulled = 0;
    while (run.pull(/*timeout_ms=*/1000).has_value())
      ++pulled;
    // END STEP

    // STEP read-metrics
    const auto stats = run.stats();
    const auto input_stats = run.input_stats();
    // END STEP
    // END CORE LOGIC

    std::cout << "inputs_enqueued=" << stats.inputs_enqueued << "\n";
    std::cout << "inputs_dropped=" << stats.inputs_dropped << "\n";
    std::cout << "outputs_pulled=" << pulled << "\n";
    std::cout << "avg_latency_ms=" << stats.avg_latency_ms << "\n";
    std::cout << "avg_push_us=" << input_stats.avg_push_us << "\n";
    std::cout << "renegotiations=" << input_stats.renegotiations << "\n";
    std::cout << "[OK] 015_tune_throughput_and_queues\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

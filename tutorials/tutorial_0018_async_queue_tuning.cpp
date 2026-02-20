// tutorial_0018_async_queue_tuning.cpp
// Story: tuning async queues and drop policies.
// What you learn:
// - How queue depth affects drops and latency.
// - How OverflowPolicy changes behavior under pressure.
// - How to inspect Run stats.

#include "neat/session.h"
#include "neat/nodes.h"

#include "tutorial_common.h"

#include <opencv2/core.hpp>

#include <iostream>
#include <string>

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--iters <n>] [--queue <n>] [--drop <policy>]\n";
  sima_tutorial::print_common_flags(std::cout);
  std::cout << "  --iters <n>          Frames to push (default 32)\n";
  std::cout << "  --queue <n>          Queue depth (default 4)\n";
  std::cout << "  --drop <policy>      block|newest|oldest (default block)\n";
}

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string val;
  if (!sima_tutorial::get_arg(argc, argv, key, val))
    return def;
  try {
    return std::stoi(val);
  } catch (...) {
    return def;
  }
}

simaai::neat::OverflowPolicy parse_drop_policy(int argc, char** argv) {
  std::string policy;
  if (!sima_tutorial::get_arg(argc, argv, "--drop", policy)) {
    return simaai::neat::OverflowPolicy::Block;
  }
  if (policy == "newest")
    return simaai::neat::OverflowPolicy::DropIncoming;
  if (policy == "oldest")
    return simaai::neat::OverflowPolicy::KeepLatest;
  return simaai::neat::OverflowPolicy::Block;
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (sima_tutorial::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const int iters = parse_int_arg(argc, argv, "--iters", 32);
    const int q = parse_int_arg(argc, argv, "--queue", 4);
    const simaai::neat::OverflowPolicy overflow_policy = parse_drop_policy(argc, argv);

    const int w = 160;
    const int h = 120;
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(90, 10, 200));
    if (!img.isContinuous())
      img = img.clone();

    simaai::neat::Session p;
    simaai::neat::InputOptions in;
    in.format = "RGB";
    in.width = w;
    in.height = h;
    in.depth = 3;
    in.is_live = true;
    p.add(simaai::neat::nodes::Input(in));
    p.add(simaai::neat::nodes::Output());

    if (sima_tutorial::wants_print_gst(argc, argv)) {
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    simaai::neat::RunOptions run_opt;
    run_opt.queue_depth = q;
    run_opt.overflow_policy = overflow_policy;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;

    auto run = p.build(img, simaai::neat::RunMode::Async, run_opt);

    for (int i = 0; i < iters; ++i) {
      (void)run.try_push(img);
    }
    run.close_input();

    int outputs = 0;
    while (true) {
      auto sample = run.pull(/*timeout_ms=*/2000);
      if (!sample.has_value())
        break;
      outputs += 1;
    }

    const auto stats = run.stats();
    std::cout << "Inputs enqueued: " << stats.inputs_enqueued << "\n";
    std::cout << "Inputs dropped:  " << stats.inputs_dropped << "\n";
    std::cout << "Outputs pulled:  " << outputs << "\n";
    std::cout << "Avg latency ms:  " << stats.avg_latency_ms << "\n";

    std::cout << "[OK] tutorial_0018 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

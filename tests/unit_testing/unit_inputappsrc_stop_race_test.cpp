#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/runtime/RunCore.h"
#include "pipeline/runtime/RunInternal.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <chrono>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace simaai::neat;

cv::Mat make_rgb_mat(int w, int h, const cv::Scalar& color) {
  cv::Mat img(h, w, CV_8UC3, color);
  if (!img.isContinuous())
    img = img.clone();
  return img;
}

std::unique_lock<std::mutex> park_input_worker(Run& run, runtime::RunCore& core,
                                               const cv::Mat& valid) {
  // Existing timing lock gates the worker immediately after it dequeues input.
  std::unique_lock gate(core.latency_mu);
  require(run.try_push(std::vector<cv::Mat>{valid}), "gate input was rejected");
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  bool dequeued = false;
  while (std::chrono::steady_clock::now() < deadline) {
    {
      std::lock_guard lock(core.pipeline.in_mu);
      dequeued = core.pipeline.in_queue.empty();
    }
    if (dequeued)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  require(dequeued, "input worker did not reach the lifecycle gate");
  return gate;
}

void reject_then_stop_pending_work() {
  constexpr int w = 64;
  constexpr int h = 48;
  const cv::Mat valid = make_rgb_mat(w, h, cv::Scalar(10, 20, 30));
  const cv::Mat larger = make_rgb_mat(w * 2, h * 2, cv::Scalar(20, 30, 40));
  Graph graph;
  InputOptions input;
  input.format = FormatTag::RGB;
  input.width = w;
  input.height = h;
  input.depth = 3;
  input.memory_policy = InputMemoryPolicy::SystemMemory;
  input.caps_override = "video/x-raw,format=RGB,width=" + std::to_string(w) +
                        ",height=" + std::to_string(h) + ",framerate=30/1";
  graph.add(nodes::Input(input));
  graph.add(nodes::Output(OutputOptions::EveryFrame(8)));

  RunOptions options;
  options.output_memory = OutputMemory::Owned;
  options.queue_depth = 1;
  options.overflow_policy = OverflowPolicy::KeepLatest;
  auto run = graph.build(std::vector<cv::Mat>{valid}, options);
  auto core = std::const_pointer_cast<runtime::RunCore>(run_internal::core(run));
  require(core != nullptr, "missing Run core");
  require(!run.run(std::vector<cv::Mat>{valid}, 2000).empty(), "warm output missing");

  {
    auto gate = park_input_worker(run, *core, valid);
    require(run.try_push(std::vector<cv::Mat>{valid}), "valid pending input was rejected");
    const auto enqueued = core->inputs_enqueued.load();
    const auto dropped = core->inputs_dropped.load();
    bool rejected = false;
    try {
      (void)run.push(std::vector<cv::Mat>{larger});
    } catch (const NeatError& error) {
      require(error.report().error_code == error_codes::kInputCapacity,
              "oversized input must fail with the structured capacity error");
      rejected = true;
    }
    require(rejected, "oversized input was accepted");
    require(core->inputs_enqueued.load() == enqueued && core->inputs_dropped.load() == dropped,
            "invalid input must not enqueue or evict valid pending work");
    require(!core->stop_requested.load(), "capacity rejection poisoned the Run");
    {
      std::lock_guard lock(core->pipeline.in_mu);
      require(core->pipeline.in_queue.size() == 1, "capacity rejection lost pending work");
    }
    gate.unlock();
  }
  require(!run.pull_tensors(2000).empty(), "gate output missing after rejected input");
  require(!run.pull_tensors(2000).empty(), "pending output missing after rejected input");
  require(!run.run(std::vector<cv::Mat>{valid}, 2000).empty(),
          "the same Run did not recover after rejected input");

  auto gate = park_input_worker(run, *core, valid);
  require(run.try_push(std::vector<cv::Mat>{valid}), "stop fixture needs valid pending work");
  {
    std::lock_guard lock(core->pipeline.in_mu);
    require(core->pipeline.in_queue.size() == 1, "stop fixture has no pending work");
  }
  run.close_input();
  auto stopped = std::async(std::launch::async, [&] { run.stop(); });
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (!core->stop_requested.load() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  const bool stopping = core->stop_requested.load();
  // Release before get(), assertions or any path which might destroy/join the future.
  gate.unlock();
  stopped.get();
  require(stopping, "stop did not begin while valid work was pending");
  require(core->pipeline.input_thread_done.load(), "stop returned before the worker finished");
  {
    std::lock_guard lock(core->pipeline.in_mu);
    require(core->pipeline.in_queue.empty(), "stop retained pending work");
  }
  require(!run.try_push(std::vector<cv::Mat>{valid}), "closed Run accepted input");
  run.close();
}
} // namespace

RUN_TEST("unit_inputappsrc_stop_race_test", ([] {
           for (int i = 0; i < 5; ++i)
             reject_then_stop_pending_work();
         }));

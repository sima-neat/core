#include "pipeline/Graph.h"
#include "nodes/common/Caps.h"
#include "nodes/common/Output.h"
#include "nodes/common/Queue.h"
#include "nodes/common/VideoRate.h"
#include "nodes/io/Input.h"
#include "test_main.h"

#include <opencv2/core.hpp>

#include <chrono>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace simaai::neat;
using namespace std::chrono_literals;

void check_resume(Run& run, const cv::Mat& image) {
  std::this_thread::sleep_for(450ms);
  require(run.last_error().empty(), "Drained Run failed while idle: " + run.last_error());
  require(run.running(), "Drained Run stopped");
  require(run.push(std::vector<cv::Mat>{image}), "Run rejected input after idle");
  Sample sample;
  PullError error;
  require(run.pull(1000, sample, &error) == PullStatus::Ok,
          "Run failed to produce output after idle: " + run.last_error());
  const auto tensors = tensors_from_sample(sample);
  require(tensors.size() == 1, "Expected one image");
  auto pixels = tensors.front().map(MapMode::Read);
  require(pixels.data && pixels.size_bytes >= 3, "Output image is missing");
  require(static_cast<const unsigned char*>(pixels.data)[0] == image.at<cv::Vec3b>(0, 0)[0],
          "Resume returned an old or incorrect image");
}

void exercise(OverflowPolicy policy, bool video_rate) {
  cv::Mat seed(48, 64, CV_8UC3, cv::Scalar(10, 20, 30));
  InputOptions input;
  input.format = FormatTag::RGB;
  input.width = seed.cols;
  input.height = seed.rows;
  input.depth = 3;
  input.fps_n = video_rate ? 100 : 0;
  input.memory_policy = InputMemoryPolicy::SystemMemory;
  input.max_bytes = 0;
  input.pool_max_buffers = 64;
  Graph graph;
  graph.add(nodes::Input(input));
  if (video_rate) {
    graph.add(nodes::VideoRate());
    graph.add(nodes::CapsRaw("RGB", 64, 48, 10));
  } else {
    QueueOptions queue;
    queue.max_buffers = 1;
    queue.overflow_policy = policy;
    graph.add(nodes::Queue(queue));
  }
  graph.custom("identity sleep-time=5000");
  graph.add(nodes::Output(OutputOptions::EveryFrame(64)));
  RunOptions options;
  options.queue_depth = 64;
  options.input_timeout_ms = 200;
  options.startup_preflight = false;
  options.output_memory = OutputMemory::Owned;
  auto run = graph.build(std::vector<cv::Mat>{seed}, options);
  check_resume(run, seed);

  for (int cycle = 0; cycle < 3; ++cycle) {
    for (int i = 0; i < 30; ++i)
      require(run.push(std::vector<cv::Mat>{seed}), "Public Run rejected the burst");
    int survivors = 0;
    for (;;) {
      Sample sample;
      PullError error;
      const auto status = run.pull(75, sample, &error);
      if (status == PullStatus::Timeout)
        break;
      require(status == PullStatus::Ok, "Burst failed: " + run.last_error());
      ++survivors;
    }
    require(video_rate || survivors > 0, "No surviving output from the queue burst");
    require(!video_rate && policy == OverflowPolicy::Block ? survivors == 30 : survivors < 30,
            "Fixture did not exercise the requested drop policy");
    cv::Mat resumed(48, 64, CV_8UC3, cv::Scalar(70 + cycle, 80, 90));
    check_resume(run, resumed);
  }
  const auto closing = std::chrono::steady_clock::now();
  run.close();
  require(std::chrono::steady_clock::now() - closing < 2s, "Public Run close was not bounded");
}
} // namespace

RUN_TEST("unit_public_run_drop_idle_test", ([] {
           for (const auto policy :
                {simaai::neat::OverflowPolicy::Block, simaai::neat::OverflowPolicy::KeepLatest,
                 simaai::neat::OverflowPolicy::DropIncoming}) {
             std::cerr << "case: public Queue policy=" << static_cast<int>(policy) << '\n';
             exercise(policy, false);
           }
           std::cerr << "case: public VideoRate\n";
           exercise(simaai::neat::OverflowPolicy::Block, true);
         }));

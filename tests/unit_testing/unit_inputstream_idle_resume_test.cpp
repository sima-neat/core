#include "pipeline/Graph.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "test_main.h"

#include <opencv2/core.hpp>

#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

namespace {
using namespace simaai::neat;
using namespace std::chrono_literals;
const cv::Mat image(24, 32, CV_8UC3, cv::Scalar(10, 20, 30));

Run build(bool preflight, const char* stage = nullptr) {
  InputOptions input;
  input.format = FormatTag::RGB;
  input.width = input.max_width = image.cols;
  input.height = input.max_height = image.rows;
  input.depth = input.max_depth = 3;
  input.memory_policy = InputMemoryPolicy::SystemMemory;
  Graph graph;
  graph.add(nodes::Input(input));
  if (stage)
    graph.custom(stage);
  graph.add(nodes::Output(OutputOptions::EveryFrame(8)));
  RunOptions options;
  options.input_timeout_ms = 200;
  options.output_memory = OutputMemory::Owned;
  options.startup_preflight = preflight;
  return graph.build(std::vector<cv::Mat>{image}, options);
}

void expect_output(Run& run, const cv::Mat& input) {
  Sample sample;
  const auto status = run.pull(1000, sample);
  require(status == PullStatus::Ok, "Expected output: " + run.last_error());
  const auto tensors = tensors_from_sample(sample);
  require(tensors.size() == 1, "Expected one output tensor");
  auto pixels = tensors.front().map(MapMode::Read);
  const auto bytes = input.total() * input.elemSize();
  require(pixels.data && pixels.size_bytes >= bytes &&
              std::memcmp(pixels.data, input.data, bytes) == 0,
          "Output changed");
}

void exchange(Run& run, unsigned char value) {
  const cv::Mat input(image.rows, image.cols, image.type(), cv::Scalar(value, 20, 30));
  require(run.push(std::vector<cv::Mat>{input}), "Input rejected");
  expect_output(run, input);
}

void close(Run& run) {
  const auto start = std::chrono::steady_clock::now();
  run.close();
  require(std::chrono::steady_clock::now() - start < 2s, "Close was not bounded");
}

} // namespace

RUN_TEST("unit_inputstream_idle_resume_test", ([] {
           for (bool preflight : {false, true}) {
             std::cerr << "case: idle/resume, preflight=" << preflight << '\n';
             auto run = build(preflight);
             std::this_thread::sleep_for(450ms);
             require(run.running() && run.last_error().empty(), "Run failed while idle");
             exchange(run, 40);
             std::this_thread::sleep_for(450ms);
             require(run.running() && run.last_error().empty(), "Completed work failed while idle");
             exchange(run, 80);
             close(run);
           }
           {
             std::cerr << "case: delayed output after pull timeout\n";
             auto run = build(/*preflight=*/false, "identity sleep-time=500000");
             const cv::Mat first(image.rows, image.cols, image.type(), cv::Scalar(40, 20, 30));
             require(run.push(std::vector<cv::Mat>{first}), "Delayed input rejected");
             Sample sample;
             require(run.pull(50, sample) == PullStatus::Timeout,
                     "Delayed input should time out the first pull");
             require(run.running() && run.last_error().empty(), "Pull timeout failed the Run");
             expect_output(run, first);
             exchange(run, 80);
             require(run.pull(0, sample) == PullStatus::Timeout, "Unexpected duplicate output");
             close(run);
           }
           {
             std::cerr << "case: intentional discard\n";
             auto run = build(/*preflight=*/false, "valve drop=true");
             require(run.push(std::vector<cv::Mat>{image}), "Discard input rejected");
             Sample sample;
             require(run.pull(450, sample) == PullStatus::Timeout,
                     "Discarded input should time out only the pull");
             require(run.running() && run.last_error().empty(), "Discarded input failed the Run");
             require(run.push(std::vector<cv::Mat>{image}),
                     "Run rejected input after pull timeout");
             close(run);
           }
         }));

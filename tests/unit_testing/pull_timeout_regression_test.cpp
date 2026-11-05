#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "test_main.h"
#include "test_utils.h"

#include <chrono>
#include <sstream>
#include <string>
#include <thread>

namespace {

int elapsed_ms(std::chrono::steady_clock::time_point start,
               std::chrono::steady_clock::time_point end) {
  return static_cast<int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
}

void require_timeout_window(const char* op_name, int measured_ms, int timeout_ms) {
  const int lower_bound = timeout_ms - 40;
  const int upper_bound = timeout_ms + 700;
  if (measured_ms < lower_bound || measured_ms > upper_bound) {
    std::ostringstream oss;
    oss << op_name << " timeout outside expected window: measured=" << measured_ms
        << "ms, expected=[" << lower_bound << "," << upper_bound << "] for timeout=" << timeout_ms
        << "ms";
    throw std::runtime_error(oss.str());
  }
}

} // namespace

RUN_TEST("pull_timeout_regression_test", [] {
  using namespace simaai::neat;

  Graph graph;
  InputOptions input_opt;
  input_opt.payload_type = simaai::neat::PayloadType::Image;
  input_opt.format = simaai::neat::FormatTag::RGB;
  input_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
  input_opt.max_width = 96;
  input_opt.max_height = 96;
  input_opt.max_depth = 3;
  graph.add(nodes::Input(input_opt));
  graph.add(nodes::Output(OutputOptions::Latest()));

  RunOptions run_opt;
  run_opt.queue_depth = 8;

  const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x33);
  Run run = graph.build(TensorList{seed}, run_opt);

  const int timeout_ms = 120;
  {
    const auto t0 = std::chrono::steady_clock::now();
    auto out = run.pull(timeout_ms);
    const auto t1 = std::chrono::steady_clock::now();
    require(!out.has_value(), "pull() should timeout with no pending output");
    require_timeout_window("pull()", elapsed_ms(t0, t1), timeout_ms);
  }
  {
    const auto t0 = std::chrono::steady_clock::now();
    TensorList out = run.pull_tensors(timeout_ms);
    const auto t1 = std::chrono::steady_clock::now();
    require(out.empty(), "pull_tensors() should timeout with no pending output");
    require_timeout_window("pull_tensors()", elapsed_ms(t0, t1), timeout_ms);
  }

  for (int i = 0; i < 5; ++i) {
    Tensor input =
        make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, static_cast<uint8_t>(0x40 + i));
    const auto t0 = std::chrono::steady_clock::now();
    TensorList outs = run.run(TensorList{input}, 1000);
    const auto t1 = std::chrono::steady_clock::now();

    require(outs.size() == 1, "run(TensorList) returned wrong tensor count");
    require(elapsed_ms(t0, t1) < 1000, "run(TensorList) exceeded timeout");
  }

  run.stop();

  Graph delayed;
  delayed.add(nodes::Input(input_opt));
  delayed.custom("identity sleep-time=500000");
  delayed.add(nodes::Output(OutputOptions::EveryFrame()));
  RunOptions delayed_opt;
  delayed_opt.preset = RunPreset::Realtime;
  delayed_opt.output_memory = OutputMemory::Owned;
  delayed_opt.startup_preflight = false;
  delayed_opt.input_timeout_ms = 100;
  delayed_opt.advanced.sync_num_buffers_override = 1;

  const auto expect_run_timeout = [&](uint8_t value) {
    try {
      (void)delayed.run(TensorList{make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, value)},
                        delayed_opt);
    } catch (const std::exception& error) {
      require_contains(error.what(), "status=Timeout", "Graph::run failed without a pull timeout");
      return;
    }
    throw std::runtime_error("Graph::run returned late output from the preceding timed-out call");
  };
  expect_run_timeout(0x40);
  // Give an uncancelled first output time to arrive before the next input.
  std::this_thread::sleep_for(std::chrono::milliseconds(650));
  expect_run_timeout(0x80);
});

#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/InputStream.h"
#include "pipeline/NeatError.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/runtime/RunCore.h"
#include "pipeline/runtime/RunInternal.h"
#include "nodes/io/Input.h"
#include "nodes/common/Output.h"
#include "test_main.h"
#include "test_utils.h"
#include <opencv2/core.hpp>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <new>

namespace {
using namespace simaai::neat;
template <class Fn> void rejects(Fn fn, const char* code) {
  try {
    fn();
  } catch (const NeatError& e) {
    require(e.report().error_code == code, "wrong structured rejection code");
    require(!e.report().repro_note.empty(), "rejection must carry diagnostic");
    return;
  }
  throw std::runtime_error("invalid input metadata was accepted");
}
void queue_contract() {
  Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x4A);
  InputOptions options;
  options.payload_type = PayloadType::Image;
  options.format = FormatTag::RGB;
  options.memory_policy = InputMemoryPolicy::SystemMemory;
  options.max_width = 96;
  options.max_height = 96;
  options.max_depth = 3;
  Graph graph;
  graph.add(nodes::Input(options));
  graph.add(nodes::Output(OutputOptions::EveryFrame(8)));
  RunOptions run_options;
  run_options.queue_depth = 1;
  run_options.overflow_policy = OverflowPolicy::KeepLatest;
  run_options.advanced.copy_input = true;
  Run run = graph.build(TensorList{seed}, run_options);
  require(run.try_push(TensorList{seed}), "warm input refused");
  require(!run.pull_tensors(2000).empty(), "warm output missing");
  auto core = std::const_pointer_cast<runtime::RunCore>(run_internal::core(run));
  std::unique_lock gate(core->latency_mu);
  require(run.try_push(TensorList{seed}), "gate input refused");
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  bool parked = false;
  while (std::chrono::steady_clock::now() < deadline) {
    {
      std::lock_guard lock(core->pipeline.in_mu);
      parked = core->pipeline.in_queue.empty();
    }
    if (parked)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  require(parked, "worker did not reach gate");
  require(run.try_push(TensorList{seed}), "pending input refused");
  const auto enqueued = core->inputs_enqueued.load();
  const auto dropped = core->inputs_dropped.load();
  cv::Mat oversized(48, 97, CV_8UC3);
  auto short_tensor = seed;
  short_tensor.storage = make_cpu_owned_storage(1);
  rejects([&] { run.try_push(std::vector<cv::Mat>{oversized}); }, error_codes::kInputCapacity);
  rejects([&] { run.try_push(TensorList{short_tensor}); }, error_codes::kInputShape);
  rejects([&] { run.try_push(Sample{sample_from_tensors(TensorList{short_tensor})}); },
          error_codes::kInputShape);
  require(core->inputs_enqueued.load() == enqueued && core->inputs_dropped.load() == dropped,
          "rejected input consumed queue admission or evicted valid input");
  require(!core->stop_requested.load(), "caller rejection poisoned Run");
  {
    std::lock_guard lock(core->pipeline.in_mu);
    require(core->pipeline.in_queue.size() == 1, "pending work lost");
  }
  gate.unlock();
  require(!run.pull_tensors(2000).empty(), "gate output missing after rejection");
  require(!run.pull_tensors(2000).empty(), "pending output missing after rejection");
  require(!run.run(TensorList{seed}, 2000).empty(), "same Run did not recover after rejection");
  run.stop();
}

void aggregate_queue_contract() {
  Tensor image =
      Tensor::from_vector(std::vector<std::uint8_t>(48 * 64, 7), {48, 64}, TensorMemory::CPU);
  image.layout = TensorLayout::HW;
  Tensor points =
      Tensor::from_vector(std::vector<std::int32_t>(128, 3), {64, 2}, TensorMemory::CPU);
  const TensorList seed{image, image, points};
  InputOptions options;
  options.payload_type = PayloadType::Tensor;
  options.format = FormatTag::UINT8;
  options.memory_policy = InputMemoryPolicy::SystemMemory;
  options.max_width = 64;
  options.max_height = 48;
  options.max_depth = 1;
  Graph graph;
  graph.add(nodes::Input(options));
  graph.add(nodes::Output(OutputOptions::EveryFrame(8)));
  RunOptions run_options;
  run_options.queue_depth = 1;
  run_options.overflow_policy = OverflowPolicy::KeepLatest;
  Run run = graph.build(seed, run_options);
  const auto check_output = [](const TensorList& output) {
    require(output.size() == 3 && output[0].dtype == TensorDType::UInt8 &&
                output[1].dtype == TensorDType::UInt8 && output[2].dtype == TensorDType::Int32,
            "mixed input traversal lost independent member types");
  };
  require(run.try_push(seed), "mixed warm input refused");
  check_output(run.pull_tensors(2000));
  auto core = std::const_pointer_cast<runtime::RunCore>(run_internal::core(run));
  std::unique_lock gate(core->latency_mu);
  require(run.try_push(seed), "mixed gate input refused");
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  bool parked = false;
  while (std::chrono::steady_clock::now() < deadline) {
    {
      std::lock_guard lock(core->pipeline.in_mu);
      parked = core->pipeline.in_queue.empty();
    }
    if (parked)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  require(parked, "mixed worker did not reach gate");
  require(run.try_push(seed), "mixed pending input refused");
  const auto enqueued = core->inputs_enqueued.load();
  const auto dropped = core->inputs_dropped.load();
  auto invalid = seed;
  invalid.back().storage = make_cpu_owned_storage(1);
  rejects([&] { run.try_push(invalid); }, error_codes::kInputShape);
  rejects([&] { run.try_push(Sample{sample_from_tensors(invalid)}); }, error_codes::kInputShape);
  require(core->inputs_enqueued.load() == enqueued && core->inputs_dropped.load() == dropped &&
              !core->stop_requested.load(),
          "invalid later member altered queue or terminal state");
  {
    std::lock_guard lock(core->pipeline.in_mu);
    require(core->pipeline.in_queue.size() == 1, "invalid later member evicted pending input");
  }
  gate.unlock();
  check_output(run.pull_tensors(2000));
  check_output(run.pull_tensors(2000));
  check_output(run.run(seed, 2000));
  run.stop();
}

} // namespace
RUN_TEST("unit_input_metadata_admission_test", ([] {
           queue_contract();
           aggregate_queue_contract();
         }));

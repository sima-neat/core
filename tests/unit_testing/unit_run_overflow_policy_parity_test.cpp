#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "model/Model.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"
#include "pipeline/Run.h"
#include "pipeline/runtime/RunInternal.h"
#include "pipeline/runtime/RunCore.h"
#include "pipeline/Graph.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

namespace {

enum class PushPath {
  Mat = 0,
  Tensor,
  Sample,
  Holder,
};

struct PolicyProbeResult {
  int attempts = 0;
  int accepted = 0;
  bool saw_false = false;
  std::uint64_t inputs_dropped = 0;
};

simaai::neat::Run make_async_rgb_run_with_policy(const simaai::neat::Tensor& seed,
                                                 simaai::neat::OverflowPolicy overflow_policy,
                                                 int queue_depth = 1) {
  using namespace simaai::neat;

  Graph graph;
  InputOptions src_opt;
  src_opt.payload_type = simaai::neat::PayloadType::Image;
  src_opt.format = simaai::neat::FormatTag::RGB;
  src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
  src_opt.max_width = 96;
  src_opt.max_height = 96;
  src_opt.max_depth = 3;
  graph.add(nodes::Input(src_opt));
  graph.add(nodes::Output(OutputOptions::EveryFrame(128)));

  RunOptions run_opt;
  run_opt.queue_depth = queue_depth;
  run_opt.overflow_policy = overflow_policy;
  run_opt.advanced.copy_input = true;

  return graph.build(TensorList{seed}, run_opt);
}

simaai::neat::Sample tensor_to_sample(const simaai::neat::Tensor& tensor) {
  using namespace simaai::neat;
  Sample sample;
  sample.kind = SampleKind::Tensor;
  sample.tensor = tensor;
  sample.payload_tag = "RGB";
  sample.owned = true;
  return sample;
}

PolicyProbeResult probe_policy(simaai::neat::OverflowPolicy policy, PushPath path) {
  using namespace simaai::neat;

  const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x4A);
  Run run = make_async_rgb_run_with_policy(seed, policy, 1);
  cv::Mat mat_seed(48, 64, CV_8UC3, cv::Scalar(40, 100, 180));
  const Sample sample_seed = tensor_to_sample(seed);

  std::shared_ptr<void> holder;
  if (path == PushPath::Holder) {
    const TensorList first = run.run(TensorList{seed}, 1000);
    require(first.size() == 1, "holder parity test: expected one tensor output");
    require(first.front().storage != nullptr, "holder parity test: missing tensor storage");
    require(first.front().storage->holder != nullptr, "holder parity test: missing holder");
    holder = first.front().storage->holder;
  }

  PolicyProbeResult result;
  constexpr int kMaxAttempts = 16384;
  for (int i = 0; i < kMaxAttempts; ++i) {
    ++result.attempts;
    bool ok = false;
    switch (path) {
    case PushPath::Mat:
      ok = run.try_push(std::vector<cv::Mat>{mat_seed});
      break;
    case PushPath::Tensor:
      ok = run.try_push(TensorList{seed});
      break;
    case PushPath::Sample:
      ok = run.try_push(Sample{sample_seed});
      break;
    case PushPath::Holder:
      ok = run.try_push_holder(holder);
      break;
    }

    if (ok) {
      ++result.accepted;
      continue;
    }

    result.saw_false = true;
    if (policy != OverflowPolicy::KeepLatest) {
      break;
    }
  }

  result.inputs_dropped = run_internal::stats(run).inputs_dropped;
  run.stop();
  return result;
}

void require_policy_parity(simaai::neat::OverflowPolicy policy, bool expect_backpressure_false,
                           bool expect_drops) {
  const auto mat = probe_policy(policy, PushPath::Mat);
  const auto tensor = probe_policy(policy, PushPath::Tensor);
  const auto sample = probe_policy(policy, PushPath::Sample);
  const auto holder = probe_policy(policy, PushPath::Holder);

  if (expect_backpressure_false) {
    require(mat.saw_false, "Mat path did not report policy backpressure");
    require(tensor.saw_false, "Tensor path did not report policy backpressure");
    require(sample.saw_false, "Sample path did not report policy backpressure");
    require(holder.saw_false, "Holder path did not report policy backpressure");
  } else {
    require(!mat.saw_false, "Mat path unexpectedly reported false under KeepLatest");
    require(!tensor.saw_false, "Tensor path unexpectedly reported false under KeepLatest");
    require(!sample.saw_false, "Sample path unexpectedly reported false under KeepLatest");
    require(!holder.saw_false, "Holder path unexpectedly reported false under KeepLatest");
  }

  require(mat.accepted > 0, "Mat path accepted no input");
  require(tensor.accepted > 0, "Tensor path accepted no input");
  require(sample.accepted > 0, "Sample path accepted no input");
  require(holder.accepted > 0, "Holder path accepted no input");

  if (expect_drops) {
    require(mat.inputs_dropped > 0, "Mat path expected dropped inputs");
    require(tensor.inputs_dropped > 0, "Tensor path expected dropped inputs");
    require(sample.inputs_dropped > 0, "Sample path expected dropped inputs");
    require(holder.inputs_dropped > 0, "Holder path expected dropped inputs");
  } else {
    require(mat.inputs_dropped == 0, "Mat path unexpectedly dropped inputs");
    require(tensor.inputs_dropped == 0, "Tensor path unexpectedly dropped inputs");
    require(sample.inputs_dropped == 0, "Sample path unexpectedly dropped inputs");
    require(holder.inputs_dropped == 0, "Holder path unexpectedly dropped inputs");
  }
}

simaai::neat::Sample make_tensor_pair(int id) {
  using namespace simaai::neat;
  Sample sample;
  sample.kind = SampleKind::TensorSet;
  sample.payload_type = PayloadType::Tensor;
  sample.media_type = "application/vnd.simaai.tensor";
  sample.frame_id = id;
  sample.stream_id = "stream0";
  sample.pts_ns = id * 1000;
  for (int i = 0; i < 2; ++i) {
    Tensor tensor =
        Tensor::from_vector(std::vector<float>{float(id), float(i)}, {2}, TensorMemory::CPU);
    tensor.route.physical_index = 0;
    tensor.route.memory_index = i;
    sample.tensors.push_back(std::move(tensor));
  }
  return sample;
}

simaai::neat::Run make_tensor_pair_run(const simaai::neat::Sample& seed,
                                       simaai::neat::OverflowPolicy policy) {
  using namespace simaai::neat;
  InputOptions input;
  input.payload_type = PayloadType::Tensor;
  input.format = FormatTag::FP32;
  input.memory_policy = InputMemoryPolicy::SystemMemory;
  Graph graph;
  graph.add(nodes::Input(input));
  graph.add(nodes::Output(OutputOptions::EveryFrame(8)));
  RunOptions options;
  options.queue_depth = 1;
  options.overflow_policy = policy;
  options.advanced.copy_input = true;
  return graph.build(Sample{seed}, options);
}

void require_tensor_pair(const simaai::neat::TensorList& output, int id) {
  using namespace simaai::neat;
  require(output.size() == 2, "tensor-media graph must forward both members");
  for (std::size_t i = 0; i < output.size(); ++i) {
    require(output[i].dtype == TensorDType::Float32 && output[i].shape == std::vector<int64_t>{2},
            "tensor-media graph changed member type or shape");
    auto mapping = output[i].map_read();
    float values[2]{};
    require(mapping.data && mapping.size_bytes >= sizeof(values), "tensor output is truncated");
    std::memcpy(values, mapping.data, sizeof(values));
    require(values[0] == float(id) && values[1] == float(i),
            "tensor-media graph changed member contents or order");
  }
}

void runner_try_push_preserves_policy_and_ownership(simaai::neat::OverflowPolicy policy,
                                                    bool tensor_list) {
  using namespace simaai::neat;
  const Sample seed = make_tensor_pair(7);
  Run run = make_tensor_pair_run(seed, policy);
  auto core = std::const_pointer_cast<runtime::RunCore>(run_internal::core(run));
  Model::Runner runner(std::move(run));
  auto submit = [&](const Sample& sample) {
    return tensor_list ? runner.try_push(sample.tensors) : runner.try_push(sample);
  };
  // Unlike the old RGB fixture, these inputs obey the actual downstream contract.
  require(submit(seed), "Runner warmup must accept the tensor pair");
  require_tensor_pair(runner.pull(2000).tensors, 7);

  Sample incompatible = make_tensor_pair(8);
  incompatible.tensors.front() =
      Tensor::from_vector(std::vector<std::uint8_t>{8, 0}, {2}, TensorMemory::CPU);
  const auto before_invalid = core->inputs_enqueued.load();
  bool rejected = false;
  try {
    (void)submit(incompatible);
  } catch (const NeatError& error) {
    require(error.report().error_code == error_codes::kInputShape,
            "incompatible tensor media must have a structured shape rejection");
    rejected = true;
  }
  require(rejected && core->inputs_enqueued.load() == before_invalid &&
              core->inputs_dropped.load() == 0,
          "incompatible media must not enter or consume the queue");
  require(submit(make_tensor_pair(8)), "Runner must accept valid input after media rejection");
  require_tensor_pair(runner.pull(2000).tensors, 8);

  // Park the worker after dequeue so queue occupancy cannot race the assertions.
  std::unique_lock timing_lock(core->latency_mu);
  require(submit(seed), "Runner gate input must be accepted");
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  bool parked = false;
  while (std::chrono::steady_clock::now() < deadline) {
    {
      std::lock_guard queue_lock(core->pipeline.in_mu);
      parked = core->pipeline.in_queue.empty();
    }
    if (parked)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  require(parked, "Runner input worker must reach the gate");

  Sample old = make_tensor_pair(1);
  std::weak_ptr<TensorBuffer> old_storage = old.tensors[0].storage;
  require(submit(old), "Runner must accept the first pending pair");
  old = {};
  require(!old_storage.expired(), "Queued input must retain its storage");

  Sample latest = make_tensor_pair(2);
  auto pending = std::async(std::launch::async, [&] { return submit(latest); });
  const bool returned = pending.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
  if (!returned)
    runner.close_input();
  const bool accepted = pending.get();
  require(returned, "Runner try_push must not wait for queue space");
  const bool replaces = policy == OverflowPolicy::KeepLatest;
  require(accepted == replaces, "Runner must preserve the configured admission policy");
  require(old_storage.expired() == replaces, "Only replacement may release queued storage");
  require(core->inputs_dropped.load() == (policy == OverflowPolicy::Block ? 0U : 1U),
          "Runner must preserve drop accounting");
  {
    std::lock_guard queue_lock(core->pipeline.in_mu);
    require(core->pipeline.in_queue.size() == 1, "Runner must retain one pending sample");
    const auto& queued = core->pipeline.in_queue.front().msg;
    require(queued.tensors.size() == 2, "Paired inputs must remain together");
    require(queued.tensors[0].route.memory_index == 0 && queued.tensors[1].route.memory_index == 1,
            "Input tensor routes must remain distinct");
    if (replaces) {
      for (int i = 0; i < 2; ++i) {
        require(queued.tensors[i].storage == latest.tensors[i].storage,
                "Runner must not copy tensor payloads");
      }
    }
    if (!tensor_list) {
      require(queued.frame_id == (replaces ? 2 : 1) && queued.stream_id == "stream0" &&
                  queued.pts_ns == (replaces ? 2000 : 1000),
              "Sample identity must stay attached to its pair");
    }
    // Release the inspected pending sample explicitly; actual graph traversal was
    // verified above before parking the worker for deterministic queue assertions.
    core->pipeline.in_queue.clear();
  }
  runner.close_input();
  require(!submit(latest), "Runner must reject input after close_input");
  require(old_storage.expired(), "Removing the queued input must release its storage");
  timing_lock.unlock();
  runner.close();
}

} // namespace

RUN_TEST("unit_run_overflow_policy_parity_test", ([] {
           using namespace simaai::neat;

           require_policy_parity(OverflowPolicy::Block, true, false);
           require_policy_parity(OverflowPolicy::DropIncoming, true, true);
           require_policy_parity(OverflowPolicy::KeepLatest, false, true);
           for (auto policy :
                {OverflowPolicy::Block, OverflowPolicy::DropIncoming, OverflowPolicy::KeepLatest}) {
             for (bool tensor_list : {false, true}) {
               runner_try_push_preserves_policy_and_ownership(policy, tensor_list);
             }
           }
         }));

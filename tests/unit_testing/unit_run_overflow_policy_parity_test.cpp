#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Run.h"
#include "pipeline/runtime/RunInternal.h"
#include "pipeline/Graph.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <cstdint>

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

} // namespace

RUN_TEST("unit_run_overflow_policy_parity_test", ([] {
           using namespace simaai::neat;

           require_policy_parity(OverflowPolicy::Block, true, false);
           require_policy_parity(OverflowPolicy::DropIncoming, true, true);
           require_policy_parity(OverflowPolicy::KeepLatest, false, true);
         }));

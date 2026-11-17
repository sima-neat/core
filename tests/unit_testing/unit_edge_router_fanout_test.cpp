// EdgeRouter fan-out hands every branch the routed Sample. Each branch but the last needs its
// own copy; the last takes the routed Sample itself instead of a further copy, so a two-way split
// (for example a camera's decoder and VideoSender branches) copies one Sample per frame, not two.
#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "pipeline/runtime/EdgeRouter.h"
#include "test_utils.h"

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;

void fan_out_moves_the_routed_sample_into_the_last_branch(std::size_t branches) {
  const std::string label = std::to_string(branches) + "-way fan-out";
  runtime::ExecutionGraphRuntime graph;
  runtime::EdgeRouter router(graph);
  std::vector<Sample> received(branches);
  runtime::EdgeRouterCallbacks callbacks;
  callbacks.stop_requested = [] { return false; };
  callbacks.dispatch_to_stage_group = [&](std::size_t index, graph::PortId, Sample&& sample,
                                          std::size_t) {
    received.at(index) = std::move(sample);
    return true;
  };
  std::vector<runtime::DownstreamTarget> targets;
  for (std::size_t i = 0; i < branches; ++i) {
    targets.push_back({runtime::DownstreamTarget::Kind::StageGroup, i});
  }

  Tensor tensor;
  tensor.storage = make_cpu_owned_storage(4);
  tensor.dtype = TensorDType::UInt8;
  tensor.shape = {4};
  tensor.strides_bytes = {1};
  Sample sample;
  sample.kind = SampleKind::TensorSet;
  sample.frame_id = 7;
  sample.tensors = TensorList{tensor};
  const Tensor* routed_tensors = sample.tensors.data();

  require(router.dispatch_to_targets(targets, std::move(sample), runtime::EdgeRouterOptions{},
                                     callbacks),
          label + ": dispatch failed");
  for (std::size_t i = 0; i < branches; ++i) {
    const Sample& branch = received[i];
    require(branch.frame_id == 7 && branch.tensors.size() == 1 &&
                branch.tensors.front().storage == tensor.storage,
            label + ": branch " + std::to_string(i) + " must receive the routed Sample");
    const bool routed_object = branch.tensors.data() == routed_tensors;
    require(routed_object == (i + 1 == branches),
            label + ": only the last branch may take the routed Sample (branch " +
                std::to_string(i) + ")");
  }
}

} // namespace

int main() {
  try {
    fan_out_moves_the_routed_sample_into_the_last_branch(2);
    fan_out_moves_the_routed_sample_into_the_last_branch(3);
  } catch (const std::exception& e) {
    return fail_test(e.what());
  }
  std::cout << "[OK] unit_edge_router_fanout_test\n";
  return 0;
}

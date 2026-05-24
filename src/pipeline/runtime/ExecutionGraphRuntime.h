#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "graph/GraphTypes.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"
#include "graph/runtime/BlockingQueue.h"
#include "graph/runtime/StageMailbox.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "pipeline/runtime/PipelineSegmentRuntime.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace simaai::neat::runtime {

using BlockingQueueSample = simaai::neat::graph::runtime::BlockingQueue<simaai::neat::Sample>;
using StageKeyBy = simaai::neat::graph::nodes::StageKeyBy;
using StageNodeOptions = simaai::neat::graph::nodes::StageNodeOptions;

struct DownstreamTarget {
  enum class Kind {
    StageGroup,
    PipelineInput,
    GraphSink,
  };

  Kind kind = Kind::StageGroup;
  std::size_t index = 0; // stage group index, pipeline index, or sink node id
  simaai::neat::graph::PortId port = simaai::neat::graph::kInvalidPort;
};

struct RuntimeStageEmitter final : simaai::neat::graph::StageEmitter {
  std::function<bool(simaai::neat::graph::StageOutMsg)> emit_fn;
  std::function<bool()> stop_requested_fn;

  bool emit(simaai::neat::graph::StageOutMsg msg) override {
    return emit_fn ? emit_fn(std::move(msg)) : false;
  }

  bool stop_requested() const override {
    return stop_requested_fn ? stop_requested_fn() : true;
  }
};

struct StageRuntime {
  explicit StageRuntime(std::size_t capacity = 0) : mailbox(capacity) {}

  simaai::neat::graph::NodeId node_id = simaai::neat::graph::kInvalidNode;
  RuntimeStageEmitter emitter;
  std::unique_ptr<simaai::neat::graph::StageExecutor> exec;
  simaai::neat::graph::runtime::StageMailbox mailbox;
  std::thread worker;
  std::atomic<bool> worker_done{false};
  std::vector<simaai::neat::graph::PortId> input_ports;
  std::vector<simaai::neat::graph::PortId> output_ports;
  simaai::neat::graph::StagePorts ports;
};

struct StageGroup {
  simaai::neat::graph::NodeId node_id = simaai::neat::graph::kInvalidNode;
  StageNodeOptions options;
  std::vector<std::size_t> instances;
  std::atomic<std::size_t> rr{0};

  StageGroup() = default;
  StageGroup(const StageGroup&) = delete;
  StageGroup& operator=(const StageGroup&) = delete;
  StageGroup(StageGroup&& other) noexcept
      : node_id(other.node_id), options(other.options), instances(std::move(other.instances)),
        rr(other.rr.load()) {}
  StageGroup& operator=(StageGroup&& other) noexcept {
    if (this != &other) {
      node_id = other.node_id;
      options = other.options;
      instances = std::move(other.instances);
      rr.store(other.rr.load());
    }
    return *this;
  }
};

struct ExecutionGraphRuntime {
  ExecutionGraphPlan plan;
  std::vector<std::string> node_labels;
  std::atomic<std::size_t> output_rr{0};

  std::vector<std::unique_ptr<PipelineSegmentRuntime>> pipelines;
  std::vector<std::unique_ptr<StageRuntime>> stages;
  std::vector<StageGroup> stage_groups;

  std::unordered_map<simaai::neat::graph::NodeId, std::size_t> node_to_pipeline;
  std::unordered_map<simaai::neat::graph::NodeId, std::size_t> node_to_stage_group;
  std::unordered_set<simaai::neat::graph::NodeId> direct_sink_nodes;

  std::unordered_map<std::uint64_t, std::vector<DownstreamTarget>> adjacency;
  std::unordered_map<simaai::neat::graph::NodeId, std::shared_ptr<BlockingQueueSample>> sinks;
};

} // namespace simaai::neat::runtime

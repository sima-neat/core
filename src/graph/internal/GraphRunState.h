#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "graph/GraphRun.h"
#include "graph/nodes/PipelineNode.h"
#include "graph/runtime/BlockingQueue.h"
#include "graph/runtime/StageMailbox.h"
#include "builder/Node.h"
#include "nodes/io/Input.h"
#include "pipeline/PowerTelemetry.h"
#include "pipeline/Run.h"
#include "pipeline/TensorCore.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/PipelineBuild.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "pipeline/runtime/ExecutionGraphRuntime.h"
#include "pipeline/runtime/PipelineSegmentRuntime.h"
#include "pipeline/runtime/RunCore.h"

#include <atomic>
#include <condition_variable>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace simaai::neat::graph {

using simaai::neat::pipeline_internal::env_bool;
using simaai::neat::pipeline_internal::env_int;

using PipelineNode = simaai::neat::graph::nodes::PipelineNode;
using StageNode = simaai::neat::graph::nodes::StageNode;
using StageKeyBy = simaai::neat::runtime::StageKeyBy;
using StageNodeOptions = simaai::neat::runtime::StageNodeOptions;
using BlockingQueueSample = simaai::neat::runtime::BlockingQueueSample;
using DownstreamTarget = simaai::neat::runtime::DownstreamTarget;

bool graph_debug_enabled();
bool graph_push_fail_debug_enabled();
bool stop_trace_enabled();
bool graph_output_rate_enabled();
bool graph_sched_debug_enabled();

void graph_sched_record(NodeId node_id, const std::string& label, const Sample& sample);
void graph_sched_summary(const std::vector<std::string>* labels);
void graph_log_output_rate(NodeId node_id, const Sample& sample, const char* label);
void graph_output_rate_summary(
    const std::vector<std::string>* labels,
    const std::unordered_map<NodeId, std::shared_ptr<BlockingQueueSample>>* sinks);
void graph_debug_sample(const char* tag, const Sample& sample);

inline std::uint64_t port_key(NodeId id, PortId port) {
  return (static_cast<std::uint64_t>(id) << 32) | static_cast<std::uint64_t>(port);
}

bool has_input_appsrc(std::span<const std::shared_ptr<simaai::neat::Node>> nodes);
bool has_output_appsink(std::span<const std::shared_ptr<simaai::neat::Node>> nodes);
bool has_internal_source(std::span<const std::shared_ptr<simaai::neat::Node>> nodes);
std::size_t identity_map_capacity();
void maybe_force_copy_for_backpressure(Sample& sample, std::size_t qsize, const char* where,
                                       std::size_t seg_id);
InputOptions input_opts_from_spec(const OutputSpec& spec, bool complete);
bool is_encoded_sample(const Sample& sample);
std::optional<Sample> sample_from_input_spec(const OutputSpec& spec, std::string* err);
Sample make_bundle_carrier_sample();
void log_first_decoded_once(const Sample& sample, std::size_t segment_id);

struct GraphRun::State {
  std::shared_ptr<simaai::neat::runtime::RunCore> core;

  simaai::neat::runtime::ExecutionGraphRuntime& execution() {
    return core->graph_execution();
  }
  const simaai::neat::runtime::ExecutionGraphRuntime& execution() const {
    return core->graph_execution();
  }
  bool stop_requested() const {
    return !core || core->graph_stop_requested();
  }
};

// =====================================================================================
// Split implementation chunks

} // namespace simaai::neat::graph

#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/runtime/ExecutionGraphRuntime.h"

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::runtime {

struct EdgeRouterOptions {
  std::size_t edge_queue = 0;
  int push_timeout_ms = 0;
};

struct EdgeRouterDispatchOptions {
  bool sanitize_pipeline_input_before_enqueue = false;
  std::optional<std::size_t> sink_backpressure_context;
};

struct EdgeRouterCallbacks {
  std::function<bool(std::size_t, simaai::neat::graph::PortId, Sample&&, std::size_t)>
      dispatch_to_stage_group;
  std::function<bool(std::size_t, const Sample&, std::string*)> ensure_pipeline_built;
  std::function<void(std::size_t, Sample&)> sanitize_pipeline_input;
  std::function<void(simaai::neat::graph::NodeId, Sample&, std::size_t, std::size_t)>
      prepare_sink_sample;
  std::function<void(const std::string&)> request_stop;
  std::function<bool()> stop_requested;
};

class EdgeRouter {
public:
  explicit EdgeRouter(ExecutionGraphRuntime& runtime) : runtime_(&runtime) {}

  const std::vector<DownstreamTarget>* targets(simaai::neat::graph::NodeId node,
                                               simaai::neat::graph::PortId port) const;

  bool push_to_sink(simaai::neat::graph::NodeId sink_node, Sample&& sample, std::size_t edge_index,
                    const EdgeRouterOptions& options, const EdgeRouterCallbacks& callbacks,
                    std::size_t sink_backpressure_context) const;

  bool dispatch_to_target(const DownstreamTarget& target, Sample&& sample,
                          const EdgeRouterOptions& options, const EdgeRouterCallbacks& callbacks,
                          const EdgeRouterDispatchOptions& dispatch_options = {}) const;

  bool dispatch_to_targets(const std::vector<DownstreamTarget>& targets, Sample&& sample,
                           const EdgeRouterOptions& options, const EdgeRouterCallbacks& callbacks,
                           const EdgeRouterDispatchOptions& dispatch_options = {}) const;

private:
  ExecutionGraphRuntime* runtime_ = nullptr;
};

} // namespace simaai::neat::runtime

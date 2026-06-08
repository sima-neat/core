#include "EdgeRouter.h"

#include <cstdint>
#include <atomic>
#include <chrono>
#include <sstream>
#include <utility>

namespace simaai::neat::runtime {
namespace {

std::uint64_t port_key(simaai::neat::graph::NodeId id, simaai::neat::graph::PortId port) {
  return (static_cast<std::uint64_t>(id) << 32) | static_cast<std::uint64_t>(port);
}

bool stop_requested(const EdgeRouterCallbacks& callbacks) {
  return callbacks.stop_requested && callbacks.stop_requested();
}

void request_stop(const EdgeRouterCallbacks& callbacks, const std::string& msg) {
  if (callbacks.request_stop) {
    callbacks.request_stop(msg);
  }
}

const char* graph_backpressure_timeout_explanation() {
  return " This can happen because of graph backpressure: downstream stages, appsinks, or the "
         "application are not draining outputs as fast as inputs are pushed, so an internal "
         "edge/pipeline queue filled before the timeout. Pull outputs concurrently, reduce the "
         "push rate, increase GraphRunOptions.edge_queue/push_timeout_ms, or remove/relax slow "
         "downstream stages.";
}

std::uint64_t elapsed_ns_since(std::chrono::steady_clock::time_point start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start)
          .count());
}

void atomic_add_max(std::atomic<std::uint64_t>& total, std::atomic<std::uint64_t>& max_value,
                    std::uint64_t ns) {
  total.fetch_add(ns, std::memory_order_relaxed);
  std::uint64_t cur = max_value.load(std::memory_order_relaxed);
  while (cur < ns && !max_value.compare_exchange_weak(cur, ns, std::memory_order_relaxed,
                                                      std::memory_order_relaxed)) {
  }
}

} // namespace

const std::vector<DownstreamTarget>* EdgeRouter::targets(simaai::neat::graph::NodeId node,
                                                         simaai::neat::graph::PortId port) const {
  if (!runtime_)
    return nullptr;
  const auto it = runtime_->adjacency.find(port_key(node, port));
  if (it == runtime_->adjacency.end() || it->second.empty())
    return nullptr;
  return &it->second;
}

bool EdgeRouter::push_to_sink(simaai::neat::graph::NodeId sink_node, Sample&& sample,
                              std::size_t edge_index, const EdgeRouterOptions& options,
                              const EdgeRouterCallbacks& callbacks,
                              std::size_t sink_backpressure_context) const {
  if (!runtime_) {
    request_stop(callbacks, "EdgeRouter: missing runtime state");
    return false;
  }

  auto sink_it = runtime_->sinks.find(sink_node);
  if (sink_it == runtime_->sinks.end() || !sink_it->second) {
    return true;
  }

  const std::size_t qsize = sink_it->second->size();
  if (callbacks.prepare_sink_sample) {
    callbacks.prepare_sink_sample(sink_node, sample, qsize, sink_backpressure_context);
  }

  const bool trace = graph_message_trace_enabled(runtime_) && edge_index != invalid_edge_index();
  TraceGraphMessageArgs trace_args;
  if (trace) {
    trace_args = make_trace_graph_message_args(runtime_, edge_index, sample);
    trace_graph_message_event(TraceGraphMessageEventType::EdgeSrcPush, trace_args);
  }

  if (!sink_it->second->push(RuntimeSinkQueueMsg{std::move(sample), edge_index},
                             options.push_timeout_ms)) {
    if (trace) {
      trace_graph_message_event(TraceGraphMessageEventType::Drop, trace_args);
    }
    if (!stop_requested(callbacks)) {
      std::ostringstream msg;
      msg << "GraphRun: sink backpressure timeout (node=" << static_cast<std::size_t>(sink_node)
          << ", edge_queue=" << options.edge_queue
          << ", push_timeout_ms=" << options.push_timeout_ms << ")."
          << graph_backpressure_timeout_explanation();
      request_stop(callbacks, msg.str());
    }
    return false;
  }
  if (trace) {
    trace_graph_message_event(TraceGraphMessageEventType::QueueIn, trace_args);
  }
  return true;
}

bool EdgeRouter::dispatch_to_target(const DownstreamTarget& target, Sample&& sample,
                                    const EdgeRouterOptions& options,
                                    const EdgeRouterCallbacks& callbacks,
                                    const EdgeRouterDispatchOptions& dispatch_options) const {
  if (!runtime_) {
    request_stop(callbacks, "EdgeRouter: missing runtime state");
    return false;
  }

  if (target.kind == DownstreamTarget::Kind::StageGroup) {
    if (!callbacks.dispatch_to_stage_group) {
      request_stop(callbacks, "EdgeRouter: missing stage dispatch callback");
      return false;
    }
    const bool trace =
        graph_message_trace_enabled(runtime_) && target.edge_index != invalid_edge_index();
    TraceGraphMessageArgs trace_args;
    if (trace) {
      trace_args = make_trace_graph_message_args(runtime_, target.edge_index, sample);
      trace_graph_message_event(TraceGraphMessageEventType::EdgeSrcPush, trace_args);
    }
    const bool ok = callbacks.dispatch_to_stage_group(target.index, target.port, std::move(sample),
                                                      target.edge_index);
    if (trace) {
      trace_graph_message_event(
          ok ? TraceGraphMessageEventType::QueueIn : TraceGraphMessageEventType::Drop, trace_args);
    }
    return ok;
  }

  if (target.kind == DownstreamTarget::Kind::PipelineInput) {
    if (target.index >= runtime_->pipelines.size() || !runtime_->pipelines[target.index]) {
      std::ostringstream msg;
      msg << "GraphRun: pipeline input target out of range (index=" << target.index << ")";
      request_stop(callbacks, msg.str());
      return false;
    }

    if (!callbacks.ensure_pipeline_built) {
      request_stop(callbacks, "EdgeRouter: missing pipeline build callback");
      return false;
    }

    auto& pipe = *runtime_->pipelines[target.index];
    auto& telemetry = pipe.transport.telemetry;
    std::string build_err;
    const auto ensure_start = std::chrono::steady_clock::now();
    telemetry.router_ensure_build_calls.fetch_add(1, std::memory_order_relaxed);
    if (!callbacks.ensure_pipeline_built(target.index, sample, &build_err)) {
      atomic_add_max(telemetry.router_ensure_build_ns, telemetry.router_ensure_build_max_ns,
                     elapsed_ns_since(ensure_start));
      request_stop(callbacks, build_err.empty() ? "GraphRun: pipeline build failed" : build_err);
      return false;
    }
    atomic_add_max(telemetry.router_ensure_build_ns, telemetry.router_ensure_build_max_ns,
                   elapsed_ns_since(ensure_start));

    if (dispatch_options.sanitize_pipeline_input_before_enqueue &&
        callbacks.sanitize_pipeline_input) {
      const auto sanitize_start = std::chrono::steady_clock::now();
      callbacks.sanitize_pipeline_input(target.index, sample);
      telemetry.router_sanitize_calls.fetch_add(1, std::memory_order_relaxed);
      atomic_add_max(telemetry.router_sanitize_ns, telemetry.router_sanitize_max_ns,
                     elapsed_ns_since(sanitize_start));
    }

    auto& input_queue = pipe.transport.input_queue;
    if (!input_queue) {
      return true;
    }

    const auto push_start = std::chrono::steady_clock::now();
    telemetry.router_input_push_calls.fetch_add(1, std::memory_order_relaxed);
    const bool trace =
        graph_message_trace_enabled(runtime_) && target.edge_index != invalid_edge_index();
    TraceGraphMessageArgs trace_args;
    if (trace) {
      trace_args = make_trace_graph_message_args(runtime_, target.edge_index, sample);
      trace_graph_message_event(TraceGraphMessageEventType::EdgeSrcPush, trace_args);
    }
    if (!input_queue->push(RuntimePipelineQueueMsg{std::move(sample), target.edge_index},
                           options.push_timeout_ms)) {
      if (trace) {
        trace_graph_message_event(TraceGraphMessageEventType::Drop, trace_args);
      }
      atomic_add_max(telemetry.router_input_push_ns, telemetry.router_input_push_max_ns,
                     elapsed_ns_since(push_start));
      if (!stop_requested(callbacks)) {
        std::ostringstream msg;
        msg << "GraphRun: pipeline input backpressure timeout (seg="
            << static_cast<std::size_t>(runtime_->pipelines[target.index]->seg.id)
            << ", edge_queue=" << options.edge_queue
            << ", push_timeout_ms=" << options.push_timeout_ms << ")."
            << graph_backpressure_timeout_explanation();
        request_stop(callbacks, msg.str());
      }
      return false;
    }
    atomic_add_max(telemetry.router_input_push_ns, telemetry.router_input_push_max_ns,
                   elapsed_ns_since(push_start));
    if (trace) {
      trace_graph_message_event(TraceGraphMessageEventType::QueueIn, trace_args);
    }
    return true;
  }

  const auto sink_node = static_cast<simaai::neat::graph::NodeId>(target.index);
  return push_to_sink(sink_node, std::move(sample), target.edge_index, options, callbacks,
                      dispatch_options.sink_backpressure_context.value_or(target.index));
}

bool EdgeRouter::dispatch_to_targets(const std::vector<DownstreamTarget>& targets, Sample&& sample,
                                     const EdgeRouterOptions& options,
                                     const EdgeRouterCallbacks& callbacks,
                                     const EdgeRouterDispatchOptions& dispatch_options) const {
  if (targets.empty())
    return false;
  if (targets.size() == 1) {
    return dispatch_to_target(targets.front(), std::move(sample), options, callbacks,
                              dispatch_options);
  }

  bool ok = true;
  for (const auto& target : targets) {
    Sample sample_copy = sample;
    ok = dispatch_to_target(target, std::move(sample_copy), options, callbacks, dispatch_options) &&
         ok;
  }
  return ok;
}

} // namespace simaai::neat::runtime

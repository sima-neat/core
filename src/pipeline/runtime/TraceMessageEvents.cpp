#include "TraceMessageEvents.h"

#include "ExecutionGraphRuntime.h"

#include <atomic>
#include <limits>
#include <string>

// Optional platform adapter.
//
// The production provider should live in the SDK trace package / libsimaaitrace.so, not in
// dlopen-able GStreamer plugins.  Keep this TU buildable in source trees where that provider has
// not landed yet.  When the SDK installs sima_neat_metrics_tp.h and defines the expected
// tracepoint helper, this code emits sima_neat_edge:message with the same payload shape that the
// parser consumes below.
#if defined(SIMA_NEAT_ENABLE_EDGE_LTTNG_TRACEPOINTS) &&                                            \
    __has_include(<simaai/trace/sima_neat_metrics_tp.h>)
#include <simaai/trace/sima_neat_metrics_tp.h>
#define SIMA_NEAT_HAS_EDGE_MESSAGE_TRACEPOINT 1
#else
#define SIMA_NEAT_HAS_EDGE_MESSAGE_TRACEPOINT 0
#endif

namespace simaai::neat::runtime {
namespace {

bool edge_event_requires_edge(TraceGraphMessageEventType type) noexcept {
  return type == TraceGraphMessageEventType::EdgeSrcPush ||
         type == TraceGraphMessageEventType::EdgeSinkRecv ||
         type == TraceGraphMessageEventType::QueueIn ||
         type == TraceGraphMessageEventType::QueueOut || type == TraceGraphMessageEventType::Drop;
}

} // namespace

bool graph_message_trace_enabled(const ExecutionGraphRuntime* execution) noexcept {
  return execution && execution->message_trace_enabled.load(std::memory_order_acquire);
}

std::uint64_t make_numeric_message_id(const Sample& sample) noexcept {
  if (sample.orig_input_seq >= 0) {
    return static_cast<std::uint64_t>(sample.orig_input_seq);
  }
  if (sample.input_seq >= 0) {
    return static_cast<std::uint64_t>(sample.input_seq);
  }
  if (sample.frame_id >= 0) {
    return static_cast<std::uint64_t>(sample.frame_id);
  }
  return 0;
}

TraceGraphMessageArgs make_trace_graph_message_args(const ExecutionGraphRuntime* execution,
                                                    std::size_t edge_index,
                                                    const Sample& sample) noexcept {
  TraceGraphMessageArgs args;
  args.edge_index = edge_index;
  args.endpoint = {};
  args.stream_id = sample.stream_id;
  args.frame_id = sample.frame_id;
  args.input_seq = sample.input_seq;
  args.orig_input_seq = sample.orig_input_seq;
  args.pts_ns = sample.pts_ns;
  args.message_id = make_numeric_message_id(sample);
  if (!execution) {
    return args;
  }
  args.run_id_hash = execution->trace_run_id_hash.load(std::memory_order_relaxed);
  args.graph_id_hash = execution->trace_graph_id_hash.load(std::memory_order_relaxed);
  if (edge_index != invalid_edge_index() && edge_index < execution->plan.edges.size()) {
    const auto& edge = execution->plan.edges[edge_index];
    args.src_runtime_node_id =
        edge.from == graph::kInvalidNode ? -1 : static_cast<std::int32_t>(edge.from);
    args.dst_runtime_node_id =
        edge.to == graph::kInvalidNode ? -1 : static_cast<std::int32_t>(edge.to);
  }
  return args;
}

void trace_graph_message_event(TraceGraphMessageEventType type,
                               const TraceGraphMessageArgs& args) noexcept {
#if SIMA_NEAT_HAS_EDGE_MESSAGE_TRACEPOINT
  const std::string endpoint(args.endpoint);
  const std::string stream_id(args.stream_id);
  sima_neat_edge_message_args payload;
  payload.event_type = static_cast<std::uint32_t>(type);
  payload.run_id_hash = args.run_id_hash;
  payload.graph_id_hash = args.graph_id_hash;
  payload.message_id = args.message_id;
  payload.pipeline_segment_id = std::numeric_limits<std::uint32_t>::max();
  payload.edge_id =
      args.edge_index == invalid_edge_index() ? -1 : static_cast<std::int32_t>(args.edge_index);
  payload.src_runtime_node_id = args.src_runtime_node_id;
  payload.dst_runtime_node_id = args.dst_runtime_node_id;
  payload.src_plugin_instance_id = "";
  payload.dst_plugin_instance_id = "";
  payload.src_element = "";
  payload.dst_element = "";
  payload.src_pad =
      type == TraceGraphMessageEventType::GraphEntry && !endpoint.empty() ? endpoint.c_str() : "";
  payload.dst_pad = type == TraceGraphMessageEventType::GraphOutputPull && !endpoint.empty()
                        ? endpoint.c_str()
                        : "";
  payload.stream_id = stream_id.empty() ? "" : stream_id.c_str();
  payload.frame_id = args.frame_id;
  payload.input_seq = args.input_seq;
  payload.orig_input_seq = args.orig_input_seq;
  payload.pts_ns = args.pts_ns >= 0 ? static_cast<std::uint64_t>(args.pts_ns) : 0U;
  payload.bytes = 0U;
  payload.buffer_addr = 0U;
  tracepoint_sima_neat_edge_message(&payload);
#else
  (void)type;
  (void)args;
  // Provider emission intentionally lives in libsimaaitrace.so / platform trace package.  This
  // core TU remains a guarded no-op until that adapter/header is present in the SDK.
#endif
}

void trace_graph_message_event(TraceGraphMessageEventType type,
                               const ExecutionGraphRuntime* execution, std::size_t edge_index,
                               const Sample& sample, std::string_view endpoint) noexcept {
  if (!graph_message_trace_enabled(execution)) {
    return;
  }
  if (edge_event_requires_edge(type) && edge_index == invalid_edge_index()) {
    return;
  }
  TraceGraphMessageArgs args = make_trace_graph_message_args(execution, edge_index, sample);
  args.endpoint = endpoint;
  trace_graph_message_event(type, args);
}

} // namespace simaai::neat::runtime

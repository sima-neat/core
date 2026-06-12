#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "graph/GraphTypes.h"
#include "pipeline/Run.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace simaai::neat::runtime {

struct ExecutionGraphRuntime;

constexpr std::size_t invalid_edge_index() {
  return static_cast<std::size_t>(-1);
}

enum class TraceGraphMessageEventType : std::uint8_t {
  EdgeSrcPush = 0,
  EdgeSinkRecv = 1,
  QueueIn = 2,
  QueueOut = 3,
  Drop = 4,
  GraphEntry = 5,
  GraphOutputPull = 6,
};

struct TraceGraphMessageArgs {
  std::uint64_t run_id_hash = 0;
  std::uint64_t graph_id_hash = 0;
  std::size_t edge_index = invalid_edge_index();
  std::int32_t src_runtime_node_id = -1;
  std::int32_t dst_runtime_node_id = -1;
  std::string_view endpoint;
  std::string_view stream_id;
  std::int64_t frame_id = -1;
  std::int64_t input_seq = -1;
  std::int64_t orig_input_seq = -1;
  std::int64_t pts_ns = -1;
  std::uint64_t message_id = 0;
};

bool graph_message_trace_enabled(const ExecutionGraphRuntime* execution) noexcept;
std::uint64_t make_numeric_message_id(const Sample& sample) noexcept;
TraceGraphMessageArgs make_trace_graph_message_args(const ExecutionGraphRuntime* execution,
                                                    std::size_t edge_index,
                                                    const Sample& sample) noexcept;
void trace_graph_message_event(TraceGraphMessageEventType type,
                               const TraceGraphMessageArgs& args) noexcept;
void trace_graph_message_event(TraceGraphMessageEventType type,
                               const ExecutionGraphRuntime* execution, std::size_t edge_index,
                               const Sample& sample, std::string_view endpoint = {}) noexcept;

} // namespace simaai::neat::runtime

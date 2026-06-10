#pragma once

#include "pipeline/Run.h"
#include "pipeline/internal/RunDiagnostics.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal {

struct GraphQueueLatencySnapshot {
  std::string edge_id;
  std::string name;
  std::int32_t from_runtime_node_id = -1;
  std::int32_t to_runtime_node_id = -1;
  std::size_t pipeline_segment_id = static_cast<std::size_t>(-1);
  std::string queue_kind;
  std::uint64_t residence_count = 0;
  std::uint64_t residence_ns = 0;
  std::uint64_t max_residence_ns = 0;
  bool timing_enabled = false;
};

std::vector<MeasureEdgeLatency> build_edge_latency_from_diag_delta(const RunDiagSnapshot& before,
                                                                   const RunDiagSnapshot& after);

std::vector<GraphQueueLatencySnapshot> snapshot_graph_queue_latencies(const Run& run);
std::vector<MeasureEdgeLatency>
build_graph_queue_latency_delta(const std::vector<GraphQueueLatencySnapshot>& before,
                                const std::vector<GraphQueueLatencySnapshot>& after);
void set_graph_queue_timing_enabled(const Run& run, bool enabled);

void attribute_edge_latency_to_nodes(const std::vector<GraphNodeMetrics>& nodes,
                                     std::vector<MeasureEdgeLatency>* edge_metrics,
                                     std::vector<MeasureEdgeLatency>* unattributed = nullptr);

} // namespace simaai::neat::pipeline_internal

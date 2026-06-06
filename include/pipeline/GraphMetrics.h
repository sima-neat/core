/**
 * @file
 * @ingroup diagnostics
 * @brief Structured graph-level throughput/power and node-level latency metrics.
 */
#pragma once

#include "pipeline/RuntimeMetrics.h"
#include "graph/GraphTypes.h"

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace simaai::neat {

class Run;

struct NodeLatencySummary {
  std::uint64_t samples = 0;
  double total_ms = 0.0;
  double avg_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  /**
   * True when min_ms/max_ms are exact for this summary.
   *
   * Run-lifetime element counters have exact min/max. Measured-window deltas can
   * subtract samples/total exactly, but cumulative min/max cannot be subtracted
   * without window-local counters, so measured-window deltas set this false.
   */
  bool min_max_available = true;
};

struct GraphElementMetrics {
  std::string name;
  NodeLatencySummary latency;
};

struct GraphNodeMetrics {
  std::size_t pipeline_segment_id = static_cast<std::size_t>(-1);
  graph::NodeId runtime_node_id = graph::kInvalidNode;
  std::string node_id;
  std::vector<std::string> public_node_ids;
  std::string kind;
  std::string label;
  std::vector<std::string> element_names;
  std::vector<GraphElementMetrics> elements;
  NodeLatencySummary latency;
};

struct GraphMetricsReport {
  RuntimeMetrics graph_metrics;
  std::string aggregation = "run_lifetime";
  std::string latency_semantics = "sum_element_residency";
  std::string throughput_counting = "all_pulled_outputs";
  std::vector<GraphNodeMetrics> node_metrics;
};

GraphMetricsReport build_graph_metrics_report_run_lifetime(
    const Run& run, const RuntimeMetricsOptions& opt = RuntimeMetricsOptions{});

} // namespace simaai::neat

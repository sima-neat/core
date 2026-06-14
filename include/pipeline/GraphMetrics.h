/**
 * @file
 * @ingroup diagnostics
 * @brief Structured graph-level throughput/power and node-level latency metrics.
 */
#pragma once

#include "pipeline/PowerTelemetry.h"

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

class Run;
namespace runtime {
class RunCore;
} // namespace runtime

using RuntimeNodeId = std::size_t;
static constexpr RuntimeNodeId kInvalidRuntimeNodeId = static_cast<RuntimeNodeId>(-1);

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
  RuntimeNodeId runtime_node_id = kInvalidRuntimeNodeId;
  std::string node_id;
  std::vector<std::string> public_node_ids;
  std::string kind;
  std::string label;
  std::vector<std::string> element_names;
  std::vector<GraphElementMetrics> elements;
  NodeLatencySummary latency;
};

#ifdef SIMA_NEAT_INTERNAL
struct GraphMetricsCounters {
  std::uint64_t inputs_enqueued = 0;
  std::uint64_t inputs_dropped = 0;
  std::uint64_t inputs_pushed = 0;
  std::uint64_t outputs_ready = 0;
  std::uint64_t outputs_pulled = 0;
  std::uint64_t outputs_dropped = 0;
};

struct GraphMetricsSummary {
  double elapsed_seconds = 0.0;
  double throughput_fps = 0.0;
  GraphMetricsCounters counters;
  PowerSummary power;
};

struct GraphMetricsOptions {
  bool include_power = true;
};

struct GraphMetricsReport {
  GraphMetricsSummary graph_metrics;
  std::string aggregation = "run_lifetime";
  std::string latency_semantics = "sum_element_residency";
  std::string throughput_counting = "all_pulled_outputs";
  std::vector<GraphNodeMetrics> node_metrics;
};

GraphMetricsReport build_graph_metrics_report_run_lifetime(const Run& run,
                                                           const GraphMetricsOptions& opt = {});
GraphMetricsReport
build_graph_metrics_report_run_lifetime(const std::shared_ptr<const runtime::RunCore>& core,
                                        const GraphMetricsOptions& opt = {});
#endif

} // namespace simaai::neat

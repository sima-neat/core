/**
 * @file
 * @ingroup diagnostics
 * @brief Unified runtime metrics schema and serializers.
 */
#pragma once

#include "pipeline/PowerTelemetry.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

/**
 * @brief Output format for runtime metrics reports.
 * @ingroup diagnostics
 */
enum class RuntimeMetricsFormat {
  Text,       ///< Human-readable multi-line report.
  Json,       ///< JSON object suitable for tools.
  CompactText ///< One-line headline summary.
};

/**
 * @brief Options controlling collection and rendering of runtime metrics.
 * @ingroup diagnostics
 */
struct RuntimeMetricsOptions {
  bool include_power = true;        ///< Include power telemetry when available.
  bool include_diagnostics = false; ///< Include detailed diagnostic metric groups.
  bool include_pipeline = false;    ///< Include pipeline/graph description metadata.
  bool include_percentiles = false; ///< Include percentile fields when a caller provides them.
};

/**
 * @brief Basic latency summary in milliseconds.
 * @ingroup diagnostics
 */
struct RuntimeLatencyMetrics {
  double avg_ms = 0.0;          ///< Average latency.
  double min_ms = 0.0;          ///< Minimum latency.
  double max_ms = 0.0;          ///< Maximum latency.
  double p50_ms = 0.0;          ///< Optional p50 latency.
  double p95_ms = 0.0;          ///< Optional p95 latency.
  bool has_percentiles = false; ///< True iff percentile fields are populated.
};

/**
 * @brief Common input/output/drop counters.
 * @ingroup diagnostics
 */
struct RuntimeCounters {
  std::uint64_t inputs_enqueued = 0; ///< Inputs accepted into the runtime boundary.
  std::uint64_t inputs_dropped = 0;  ///< Inputs dropped/rejected.
  std::uint64_t inputs_pushed = 0;   ///< Inputs pushed downstream.
  std::uint64_t outputs_ready = 0;   ///< Outputs produced by the runtime.
  std::uint64_t outputs_pulled = 0;  ///< Outputs pulled by the application.
  std::uint64_t outputs_dropped = 0; ///< Outputs dropped before application pull.
};

/**
 * @brief Generic named scalar metric.
 * @ingroup diagnostics
 */
struct RuntimeMetricValue {
  std::string name;   ///< Metric name.
  double value = 0.0; ///< Scalar value.
  std::string unit;   ///< Optional unit, e.g. "ms", "fps", "count".
};

/**
 * @brief Generic named metric group for adapter-specific diagnostics.
 * @ingroup diagnostics
 */
struct RuntimeMetricGroup {
  std::string name;                       ///< Group name.
  std::vector<RuntimeMetricValue> values; ///< Metrics in this group.
};

/**
 * @brief Unified runtime metrics returned by Run, Model::Runner, and tools.
 * @ingroup diagnostics
 */
struct RuntimeMetrics {
  std::string source_kind;       ///< Producer kind: "run", "model", "graph", "perf", ...
  std::string source_name;       ///< Optional producer label.
  double elapsed_seconds = 0.0;  ///< Measurement duration.
  double throughput_fps = 0.0;   ///< Headline throughput.
  RuntimeLatencyMetrics latency; ///< Headline latency.
  RuntimeCounters counters;      ///< Common counters.
  PowerSummary power;            ///< Optional power summary.
  std::vector<std::pair<std::string, std::string>> metadata; ///< String metadata.
  std::vector<RuntimeMetricGroup> groups;                    ///< Adapter-specific details.
};

/**
 * @brief Render runtime metrics in the requested format.
 * @ingroup diagnostics
 */
std::string format_runtime_metrics(const RuntimeMetrics& metrics,
                                   RuntimeMetricsFormat format = RuntimeMetricsFormat::Text);

/**
 * @brief Render runtime metrics as JSON.
 * @ingroup diagnostics
 */
std::string runtime_metrics_to_json(const RuntimeMetrics& metrics, int indent = 0);

/**
 * @brief Render runtime metrics as human-readable text.
 * @ingroup diagnostics
 */
std::string runtime_metrics_to_text(const RuntimeMetrics& metrics);

/**
 * @brief Render runtime metrics as one compact line.
 * @ingroup diagnostics
 */
std::string runtime_metrics_to_compact_text(const RuntimeMetrics& metrics);

} // namespace simaai::neat

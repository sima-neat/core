#include "GraphMetricJson.h"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace simaai::neat::runtime {
namespace {

using json = nlohmann::ordered_json;

json path_stat_json(const MeasurePathStat& stat) {
  return {
      {"samples", stat.samples}, {"avg_ms", stat.avg_ms}, {"p50_ms", stat.p50_ms},
      {"p95_ms", stat.p95_ms},   {"max_ms", stat.max_ms}, {"reliable", stat.reliable},
  };
}

} // namespace

json throughput_json(std::uint64_t output_pulls, double elapsed_s, int logical_batch_size) {
  const bool valid_window = elapsed_s > 0.0 && std::isfinite(elapsed_s);
  const double outputs_per_s = valid_window ? static_cast<double>(output_pulls) / elapsed_s : 0.0;
  const int batch = logical_batch_size <= 0 ? 1 : logical_batch_size;
  return {
      {"unit", "output_pulls_per_second"},
      {"semantics", "public_output_pulls_per_second"},
      {"numerator_counter", "outputs_pulled"},
      {"numerator_value", output_pulls},
      {"denominator", "elapsed_seconds"},
      {"denominator_seconds", elapsed_s},
      {"outputs_pulled", output_pulls},
      {"elapsed_seconds", elapsed_s},
      {"outputs_per_s", outputs_per_s},
      {"logical_batch_size", batch},
      {"logical_inferences_per_s", outputs_per_s * static_cast<double>(batch)},
      {"multi_output_semantics", "each successful public pull increments outputs_pulled once"},
      {"available", valid_window},
      {"status", valid_window ? "collected" : "invalid_window"},
  };
}

json window_json(std::string_view scope, double elapsed_s, int requested_duration_ms,
                 int requested_warmup_ms, std::uint64_t warmup_iterations_excluded) {
  return {
      {"scope", std::string(scope)},
      {"elapsed_seconds", elapsed_s},
      {"requested_duration_ms", requested_duration_ms},
      {"requested_warmup_ms", requested_warmup_ms},
      {"warmup_iterations_excluded", warmup_iterations_excluded},
  };
}

json power_status_json(const PowerSummary& power, bool export_requested, bool monitor_configured,
                       std::string_view scope) {
  std::uint64_t rail_power_errors = 0;
  for (const PowerRailSummary& rail : power.rails) {
    rail_power_errors += rail.power_w.errors;
  }

  std::string status;
  if (!export_requested) {
    status = "disabled_by_options";
  } else if (!monitor_configured) {
    status = "not_configured_on_run";
  } else if (power.samples == 0 && rail_power_errors == 0) {
    status = "enabled_no_samples";
  } else if (power.samples == 0 && rail_power_errors > 0) {
    status = "unavailable_all_rails_failed";
  } else if (rail_power_errors > 0) {
    status = "collected_with_errors";
  } else {
    status = "collected";
  }

  json j = json::parse(power_summary_to_json(power, 0));
  j["status"] = status;
  j["scope"] = std::string(scope);
  j["available"] = status == "collected" || status == "collected_with_errors";
  j["attribution"] = "graph_level_only";
  j["note"] =
      "Power is board/SOM rail telemetry. DVT board readings may be unavailable or unreliable.";
  return j;
}

json graph_e2e_json(const MeasureLatencyStats& stats, bool graph_backed,
                    std::string_view empty_status) {
  return {
      {"available", stats.count > 0},
      {"status", stats.count > 0 ? "collected" : std::string(empty_status)},
      {"semantics", graph_backed ? "queue_inclusive_public_graph_push_to_public_output_pull"
                                 : "queue_inclusive_linear_run_push_to_output_pull"},
      {"interpretation",
       "Single-flight loops approximate per-input latency; async burst/queued windows include "
       "queue wait and should be presented as queue residency, not standalone latency."},
      {"count", stats.count},
      {"avg_ms", stats.avg_ms},
      {"p50_ms", stats.p50_ms},
      {"p90_ms", stats.p90_ms},
      {"p95_ms", stats.p95_ms},
      {"p99_ms", stats.p99_ms},
      {"max_ms", stats.max_ms},
  };
}

json path_timing_to_json(const MeasurePathTiming& timing, bool trace_loss_detected) {
  json out;
  out["available"] = timing.available;
  out["status"] = timing.status.empty() ? (timing.available ? "collected" : "off") : timing.status;
  out["source"] = timing.source.empty() ? "none" : timing.source;
  out["reason"] = timing.reason.empty() ? json(nullptr) : json(timing.reason);
  out["aggregation"] = timing.aggregation;
  out["trace_loss_detected"] = trace_loss_detected;
  out["warnings"] = timing.warnings;
  out["identity"] = {
      {"primary_key", timing.identity.primary_key},
      {"fallback_key", timing.identity.fallback_key},
      {"used_public_fields", timing.identity.used_public_fields},
      {"sample_identity_source", timing.identity.sample_identity_source},
  };

  out["node_arrival"] = json::array();
  for (const auto& row : timing.node_arrival) {
    out["node_arrival"].push_back({
        {"customer_node_id",
         row.customer_node_id.empty() ? json(nullptr) : json(row.customer_node_id)},
        {"lowered_node_id",
         row.lowered_node_id.empty() ? json(nullptr) : json(row.lowered_node_id)},
        {"runtime_node_id", row.runtime_node_id >= 0 ? json(row.runtime_node_id) : json(nullptr)},
        {"plugin_instance_id",
         row.plugin_instance_id.empty() ? json(nullptr) : json(row.plugin_instance_id)},
        {"stream_id", row.stream_id.empty() ? json(nullptr) : json(row.stream_id)},
        {"semantics", row.semantics},
        {"latency_ms", path_stat_json(row.latency)},
    });
  }

  out["inter_plugin_gap"] = json::array();
  for (const auto& row : timing.inter_plugin_gap) {
    json j = {
        {"customer_edge_id",
         row.customer_edge_id.empty() ? json(nullptr) : json(row.customer_edge_id)},
        {"lowered_edge_id",
         row.lowered_edge_id.empty() ? json(nullptr) : json(row.lowered_edge_id)},
        {"from_customer_node_id",
         row.from_customer_node_id.empty() ? json(nullptr) : json(row.from_customer_node_id)},
        {"to_customer_node_id",
         row.to_customer_node_id.empty() ? json(nullptr) : json(row.to_customer_node_id)},
        {"from_runtime_node_id",
         row.from_runtime_node_id >= 0 ? json(row.from_runtime_node_id) : json(nullptr)},
        {"to_runtime_node_id",
         row.to_runtime_node_id >= 0 ? json(row.to_runtime_node_id) : json(nullptr)},
        {"from_plugin_instance_id",
         row.from_plugin_instance_id.empty() ? json(nullptr) : json(row.from_plugin_instance_id)},
        {"to_plugin_instance_id",
         row.to_plugin_instance_id.empty() ? json(nullptr) : json(row.to_plugin_instance_id)},
        {"stream_id", row.stream_id.empty() ? json(nullptr) : json(row.stream_id)},
        {"semantics", row.semantics},
        {"latency_ms", path_stat_json(row.latency)},
    };
    out["inter_plugin_gap"].push_back(j);
  }
  out["inter_plugin_gap_ms"] = out["inter_plugin_gap"];

  out["output_tail"] = json::array();
  for (const auto& row : timing.output_tail) {
    out["output_tail"].push_back({
        {"output_endpoint", row.output_endpoint},
        {"customer_output_node_id",
         row.customer_output_node_id.empty() ? json(nullptr) : json(row.customer_output_node_id)},
        {"lowered_edge_id",
         row.lowered_edge_id.empty() ? json(nullptr) : json(row.lowered_edge_id)},
        {"stream_id", row.stream_id.empty() ? json(nullptr) : json(row.stream_id)},
        {"semantics", row.semantics},
        {"latency_ms", path_stat_json(row.latency)},
    });
  }
  return out;
}

std::string canonical_lowered_edge_id(std::string raw) {
  if (raw.empty()) {
    return {};
  }
  if (raw[0] == 'e' && raw.size() > 1 &&
      std::all_of(raw.begin() + 1, raw.end(),
                  [](unsigned char c) { return std::isdigit(c) != 0; })) {
    return raw;
  }
  if (std::all_of(raw.begin(), raw.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
    return "e" + raw;
  }
  return raw;
}

} // namespace simaai::neat::runtime

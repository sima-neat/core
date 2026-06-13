#include "GraphMetricJson.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

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

MeasureQualityView compute_measure_quality(const MeasureReport& report) {
  MeasureQualityView out;
  out.warnings = report.warnings;

  const std::uint64_t pulled = report.counters.outputs_pulled;
  const std::size_t samples = report.end_to_end.count;

  if (samples == 0U) {
    out.end_to_end_status = pulled == 0U ? "unavailable_no_outputs" : "unavailable_no_samples";
    out.end_to_end_correlation_reliable = false;
    if (pulled > 0U) {
      out.warnings.push_back(
          "End-to-end timing unavailable: outputs were pulled but no latency samples were "
          "correlated");
    }
  } else {
    bool correlation_reliable = true;
    if (samples != static_cast<std::size_t>(pulled)) {
      correlation_reliable = false;
      out.warnings.push_back("End-to-end timing partial: collected " +
                             std::to_string(samples) + " latency samples for " +
                             std::to_string(pulled) + " pulled outputs");
    }
    if (report.graph_sample_timing_unkeyed > 0U) {
      correlation_reliable = false;
      out.warnings.push_back("Graph sample timing had " +
                             std::to_string(report.graph_sample_timing_unkeyed) +
                             " unkeyed graph entries");
    }
    if (report.graph_sample_timing_misses > 0U) {
      correlation_reliable = false;
      out.warnings.push_back("Graph sample timing had " +
                             std::to_string(report.graph_sample_timing_misses) +
                             " output correlation misses");
    }
    out.end_to_end_status = correlation_reliable ? "collected" : "partial";
    out.end_to_end_correlation_reliable = correlation_reliable;
  }

  out.survivor_biased =
      report.counters.inputs_dropped > 0U || report.counters.outputs_dropped > 0U;
  if (out.survivor_biased) {
    out.warnings.push_back(
        "Measured window included dropped inputs/outputs; latency describes surviving "
        "correlated outputs only");
  }
  return out;
}

json throughput_json(std::uint64_t output_pulls, double elapsed_s, int logical_batch_size,
                     std::string_view scope) {
  const bool valid_window = elapsed_s > 0.0 && std::isfinite(elapsed_s);
  const double outputs_per_s = valid_window ? static_cast<double>(output_pulls) / elapsed_s : 0.0;
  const int batch = logical_batch_size <= 0 ? 1 : logical_batch_size;
  return {
      {"unit", "output_pulls_per_second"},
      {"scope", std::string(scope)},
      {"semantics", "public_output_pulls_per_second"},
      {"numerator_counter", "outputs_pulled"},
      {"numerator_value", output_pulls},
      {"denominator", "elapsed_seconds"},
      {"denominator_seconds", elapsed_s},
      {"outputs_pulled", output_pulls},
      {"elapsed_seconds", elapsed_s},
      {"outputs_per_s", outputs_per_s},
      {"batches_per_s", outputs_per_s},
      {"throughput_batches_per_s", outputs_per_s},
      {"logical_batch_size", batch},
      {"logical_inferences_per_s", outputs_per_s * static_cast<double>(batch)},
      {"multi_output_semantics", "each successful public pull increments outputs_pulled once"},
      {"available", valid_window},
      {"status", valid_window ? "collected" : "invalid_window"},
      {"warnings", json::array()},
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

json graph_e2e_json(const MeasureReport& report, bool graph_backed, std::string_view empty_status) {
  const MeasureLatencyStats& stats = report.end_to_end;
  MeasureQualityView quality = compute_measure_quality(report);
  if (stats.count == 0U && quality.end_to_end_status == "unavailable") {
    quality.end_to_end_status = std::string(empty_status);
  }
  return {
      {"available", stats.count > 0},
      {"status", quality.end_to_end_status},
      {"correlation_reliable", quality.end_to_end_correlation_reliable},
      {"survivor_biased", quality.survivor_biased},
      {"unit", "milliseconds"},
      {"scope", graph_backed ? "graph_application" : "linear_run"},
      {"source", "runtime_graph_sample_timing"},
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
      {"warnings", quality.warnings},
  };
}

json output_materialization_json(const RunOptions& opt) {
  std::string mode = "auto";
  std::string semantics = "not_tracked";
  std::string claim_scope = "requested_output_memory_mode_only";
  switch (opt.output_memory) {
  case OutputMemory::Owned:
    mode = "owned";
    semantics = "owned_output_copy";
    break;
  case OutputMemory::ZeroCopy:
    mode = "zero-copy";
    semantics = opt.advanced.prepare_output_cpu_visible ? "lazy_zero_copy_cpu_visible_prepared"
                                                        : "lazy_zero_copy";
    break;
  case OutputMemory::Auto:
    mode = "auto";
    semantics = "auto_runtime_selected_not_tracked";
    claim_scope = "requested_auto_mode_only";
    break;
  }
  json warnings = json::array();
  warnings.push_back("Output materialization reports requested output-memory mode and static "
                     "semantics only; it does not measure lazy map/clone/copy time.");
  if (opt.output_memory == OutputMemory::ZeroCopy) {
    warnings.push_back("Zero-copy benchmark claims must use this as a mode declaration only; "
                       "copy/map materialization timing is unavailable in this report.");
  }
  return {
      {"available", true},
      {"status", "metadata_only"},
      {"claim_status", "requested_mode_only_materialization_timing_unavailable"},
      {"claim_scope", claim_scope},
      {"output_memory_mode", mode},
      {"semantics", semantics},
      {"timing_available", false},
      {"runtime_resolved_mode_available", false},
      {"copy_map_timing_available", false},
      {"prepare_output_cpu_visible", opt.advanced.prepare_output_cpu_visible},
      {"note", "Output memory/materialization mode is reported, but lazy map/clone/copy timing is "
               "not tracked in this report."},
      {"warnings", warnings},
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

#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/GraphMetrics.h"
#include "pipeline/PowerTelemetry.h"
#include "pipeline/Run.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace simaai::neat::runtime {

struct MeasureQualityView {
  std::string end_to_end_status = "unavailable";
  bool end_to_end_correlation_reliable = false;
  bool survivor_biased = false;
  std::vector<std::string> warnings;
};

MeasureQualityView compute_measure_quality(const MeasureReport& report);
nlohmann::ordered_json throughput_json(std::uint64_t output_pulls, double elapsed_s,
                                       int logical_batch_size,
                                       std::string_view scope = "measured_window");
nlohmann::ordered_json window_json(std::string_view scope, double elapsed_s,
                                   int requested_duration_ms, int requested_warmup_ms,
                                   std::uint64_t warmup_iterations_excluded);
nlohmann::ordered_json power_status_json(const PowerSummary& power, bool export_requested,
                                         bool monitor_configured, std::string_view scope);
nlohmann::ordered_json graph_e2e_json(const MeasureReport& report, bool graph_backed,
                                      std::string_view empty_status);
nlohmann::ordered_json output_materialization_json(const RunOptions& opt);
nlohmann::ordered_json path_timing_to_json(const MeasurePathTiming& timing,
                                           bool trace_loss_detected);
std::string canonical_lowered_edge_id(std::string raw);

} // namespace simaai::neat::runtime

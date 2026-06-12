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

namespace simaai::neat::runtime {

nlohmann::ordered_json throughput_json(std::uint64_t output_pulls, double elapsed_s,
                                       int logical_batch_size);
nlohmann::ordered_json window_json(std::string_view scope, double elapsed_s,
                                   int requested_duration_ms, int requested_warmup_ms,
                                   std::uint64_t warmup_iterations_excluded);
nlohmann::ordered_json power_status_json(const PowerSummary& power, bool export_requested,
                                         bool monitor_configured, std::string_view scope);
nlohmann::ordered_json graph_e2e_json(const MeasureLatencyStats& stats, bool graph_backed,
                                      std::string_view empty_status);
nlohmann::ordered_json path_timing_to_json(const MeasurePathTiming& timing,
                                           bool trace_loss_detected);
std::string canonical_lowered_edge_id(std::string raw);

} // namespace simaai::neat::runtime

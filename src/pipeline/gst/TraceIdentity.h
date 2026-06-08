#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/GraphMetrics.h"
#include "pipeline/Run.h"

#include <cstdint>
#include <vector>

namespace simaai::neat::pipeline_internal {

void apply_lttng_trace_identity(const Run& run, const std::vector<GraphNodeMetrics>& nodes,
                                std::uint64_t run_id_hash, std::uint64_t graph_id_hash, bool enable,
                                bool enable_message_events = false);

} // namespace simaai::neat::pipeline_internal

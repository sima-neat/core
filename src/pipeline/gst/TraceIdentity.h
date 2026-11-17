#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/GraphMetrics.h"
#include "pipeline/Run.h"

#include <gst/gst.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace simaai::neat::runtime {
class RunCore;
}

namespace simaai::neat::pipeline_internal {

// Cold property transaction for one pipeline segment. Returned identities are
// the exact parser allowlist, including fallback trace-capable elements.
void configure_lttng_trace_pipeline(GstElement* pipeline,
                                    const std::vector<const GraphNodeMetrics*>& nodes,
                                    std::size_t segment_id, std::uint64_t run_id_hash,
                                    std::uint64_t graph_id_hash, bool enable,
                                    std::uint64_t measurement_epoch,
                                    std::vector<std::string>* configured_instances);

void apply_lttng_trace_identity(const Run& run, const std::vector<GraphNodeMetrics>& nodes,
                                std::uint64_t run_id_hash, std::uint64_t graph_id_hash, bool enable,
                                bool enable_message_events = false,
                                std::uint64_t measurement_epoch = 0,
                                std::vector<std::string>* configured_instances = nullptr);
void apply_lttng_trace_identity(const std::shared_ptr<const runtime::RunCore>& core,
                                const std::vector<GraphNodeMetrics>& nodes,
                                std::uint64_t run_id_hash, std::uint64_t graph_id_hash, bool enable,
                                bool enable_message_events = false,
                                std::uint64_t measurement_epoch = 0,
                                std::vector<std::string>* configured_instances = nullptr);

} // namespace simaai::neat::pipeline_internal

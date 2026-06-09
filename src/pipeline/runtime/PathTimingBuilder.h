#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "LttngMetricsCollector.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "pipeline/runtime/RunCore.h"

namespace simaai::neat::runtime {

MeasurePathTiming build_path_timing(const ExecutionGraphPlan& plan,
                                    const pipeline_internal::LttngParseResult& parsed,
                                    const std::vector<GraphSampleTimingEvent>& graph_entries,
                                    const std::vector<GraphSampleTimingEvent>& graph_pulls);

} // namespace simaai::neat::runtime

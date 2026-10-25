#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "pipeline/internal/contract/ContractFacts.h"

namespace simaai::neat::pipeline_internal::sima {

// Resolve the same execution policy before carrier placement. The command's
// lowering kind stays unchanged; the memory planner consumes its executor.
void resolve_physical_execution_targets(static_contract::PhysicalExecutionPlan& plan,
                                        const ContractCompileInput& input);

// Resolve owned domains after targets and producer edges are known. Imported
// allocations and aliased ownership retain their original authority.
void resolve_stage_memory_placement(SimaPluginStaticManifest& manifest);

} // namespace simaai::neat::pipeline_internal::sima

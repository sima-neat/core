#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "pipeline/internal/sima/static_contract/PhysicalExecutionPlan.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "nodes/common/Output.h"

namespace simaai::neat::pipeline_internal::terminal_output_contract {
struct PublicOutputEndpointSelector;
}

namespace simaai::neat::pipeline_internal::sima {

// Resolve the same execution policy before carrier placement. The command's
// lowering kind stays unchanged; the memory planner consumes its executor.
void resolve_physical_execution_targets(static_contract::PhysicalExecutionPlan& plan,
                                        const ContractCompileInput& input);

// Resolve checked executors before assigning carrier access.
void resolve_compiled_execution_targets(CompiledNodeContract&, const ContractCompileInput&);
std::uint32_t resolved_stage_device_access(const StageStaticSpec&);

// Final owning allocation-local bindings, after checked private requirements exist.
void finalize_pipeline_storage(
    SimaPluginStaticManifest& manifest,
    const terminal_output_contract::PublicOutputEndpointSelector* endpoint = nullptr,
    OutputPlacement placement = OutputPlacement::Auto);

} // namespace simaai::neat::pipeline_internal::sima

#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include <string_view>

namespace simaai::neat::pipeline_internal::sima {

// Resolve the same execution policy before carrier placement. The command's
// lowering kind stays unchanged; the memory planner consumes its executor.
void resolve_physical_execution_targets(static_contract::PhysicalExecutionPlan& plan,
                                        const ContractCompileInput& input);

// Resolve checked executors before assigning carrier access.
void resolve_compiled_execution_targets(CompiledNodeContract&, const ContractCompileInput&);
std::uint32_t resolved_stage_device_access(const StageStaticSpec&);

// Final owning allocation-local bindings, after checked private requirements exist.
void finalize_pipeline_storage(SimaPluginStaticManifest& manifest,
                               std::string_view terminal_host_stage = {});

} // namespace simaai::neat::pipeline_internal::sima

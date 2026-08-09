#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/MlaStaticContractExtractor.h"
#include "pipeline/internal/contract/PluginCompiledContracts.h"
#include "pipeline/internal/sima/static_contract/ModelExecutionPlan.h"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::static_contract {

// One setup-time lowering result. `value_id` is the exact graph value consumed
// by an MLA IFM port and `source_physical_index` selects the canonical
// materialized carrier published by the immediately preceding stage. Names
// and logical tensors are used only to prove this relation at setup.
struct PhysicalPortSource {
  ValueId value_id = 0;
  int source_physical_index = -1;
};

// Resolve MLA input ValueIds to physical roots. A direct value maps to the
// exact upstream physical carrier. A Pack value maps to one parent only when
// its explicit component placement and the producer's logical views prove
// exact ordered coverage. No logical child is used as a runtime address
// anchor.
std::optional<std::vector<PhysicalPortSource>>
resolve_mla_input_physical_sources(
    const ModelExecutionPlan& plan,
    std::span<const LogicalTensorStaticSpec> upstream_outputs,
    std::string* error = nullptr);

// Prove that the existing manifest projection is byte-for-byte the MLA port
// boundary decoded from MPK+ELF. On success, install the exact ELF symbols,
// ordered physical-carrier IFM sources, raw stage-local OFM publication and
// multi-IFM policy; on failure leave the contract unusable by dmabuf-plan.
bool apply_dmabuf_plan_contract_projection(const ModelExecutionPlan& plan,
                                           MlaStaticContract* contract,
                                           std::span<const PhysicalPortSource> input_sources,
                                           std::string* error = nullptr);

enum class ProcessCvuMlaBoundary : std::uint8_t {
  Inputs,
  Outputs,
};

// Select the production CVU kernel ABI and project the exact MLA-boundary
// alignment into every materialized CVU output carrier. Pre-MLA transforms
// inherit the IFM-port contract; post-MLA transforms inherit the OFM-port
// contract. A one-port packed boundary may fan out into multiple CVU output
// regions, in which case the one authoritative alignment applies to each.
bool apply_dmabuf_plan_processcvu_contract_projection(
    const ModelExecutionPlan& plan, ProcessCvuMlaBoundary boundary,
    ProcessCvuStagePayload* payload, ::simaai::neat::CompiledRuntimeContract* runtime,
    ::simaai::neat::CompiledExposedView* exposed_view,
    std::string* error = nullptr);

} // namespace simaai::neat::pipeline_internal::sima::static_contract

#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/MlaStaticContractExtractor.h"
#include "pipeline/internal/contract/PluginCompiledContracts.h"
#include "pipeline/internal/sima/static_contract/ModelExecutionPlan.h"
#include "pipeline/internal/sima/static_contract/PhysicalExecutionPlan.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::static_contract {

struct ModelStorageRoute;

struct PhysicalPortSource {
  ValueId value_id = 0;
  int source_physical_index = -1;
};

// Terminal outputs can own a separate group while upstream storage stays live.
enum class MlaOutputCarrierPolicy : std::uint8_t {
  SharedInputGroup = 0,
  DetachedOutputGroup = 1,
};

[[nodiscard]] MlaOutputCarrierPolicy select_mla_output_carrier_policy(
    const ModelExecutionPlan& plan, const PhysicalExecutionPlan& physical_plan,
    std::size_t mla_stage_index, const ModelStorageRoute* route = nullptr);

[[nodiscard]] std::vector<ValueId>
detached_mla_output_roots(const ModelExecutionPlan& plan,
                          const PhysicalExecutionPlan& physical_plan,
                          const ModelStorageRoute* route = nullptr);

// Resolve semantic IFM roots and public source identities before placement.
std::optional<std::vector<PhysicalPortSource>>
resolve_mla_input_physical_sources(const ModelExecutionPlan& plan, std::size_t mla_stage_index,
                                   std::string* error = nullptr);

// Validate MPK/ELF port semantics and author root-relative bindings. The shared
// planner assigns allocation-local spans after checked recipes are compiled.
bool apply_dmabuf_plan_contract_projection(const ModelExecutionPlan& plan,
                                           const PhysicalExecutionPlan& physical_plan,
                                           std::size_t mla_stage_index, MlaStaticContract* contract,
                                           std::span<const PhysicalPortSource> input_sources,
                                           std::string* error = nullptr);

std::optional<CompiledProcessCvuContract> build_dmabuf_plan_processcvu_command_contract(
    const ModelExecutionPlan& plan, const PhysicalExecutionPlan& physical_plan,
    std::span<const PhysicalCommandId> command_ids, std::string* error = nullptr);

bool apply_dmabuf_plan_processcvu_command_projection(
    const ModelExecutionPlan& plan, const PhysicalExecutionPlan& physical_plan,
    std::span<const PhysicalCommandId> command_ids, ProcessCvuStagePayload* payload,
    ::simaai::neat::CompiledRuntimeContract* runtime,
    ::simaai::neat::CompiledExposedView* exposed_view, std::string* error = nullptr);

// A model-managed preprocessor can replace only the checked ingress prefix.
std::optional<std::vector<PhysicalCommandId>>
resolve_model_managed_preproc_ingress_commands(const ModelExecutionPlan& plan,
                                               const PhysicalExecutionPlan& physical_plan,
                                               std::string* error = nullptr);

bool project_model_managed_preproc_contract(const ModelExecutionPlan& plan,
                                            const PhysicalExecutionPlan& physical_plan,
                                            ::simaai::neat::CompiledProcessCvuContract* contract,
                                            std::vector<PhysicalCommandId>* absorbed_command_ids,
                                            std::string* error = nullptr);

} // namespace simaai::neat::pipeline_internal::sima::static_contract

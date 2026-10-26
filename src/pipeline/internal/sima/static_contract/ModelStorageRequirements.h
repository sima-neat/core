#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/static_contract/CarrierPacking.h"
#include "pipeline/internal/sima/static_contract/PhysicalExecutionPlan.h"

#include <span>
#include <unordered_map>

namespace simaai::neat::pipeline_internal::sima::static_contract {

// Selected public fragment boundaries retain the original physical identities.
struct ModelStorageRoute {
  std::vector<PhysicalCommandId> commands;
  std::vector<ValueId> imported_inputs;
  std::vector<ValueId> published_outputs;
  std::unordered_map<CarrierId, std::uint32_t> external_device_access;
};

// Build-local model requirements and the physical-DAG proof used by the common
// packer. No heap addresses, allocation offsets or cached placement live here.
struct ModelStorageRequirements {
  std::vector<CarrierRequirement> carriers;
  std::unordered_map<std::size_t, std::vector<CarrierRequirement>> detached_outputs;
  std::vector<ValueId> imported_inputs;
  std::vector<ValueId> detached_roots;
  std::vector<std::optional<CarrierId>> value_carriers;
  std::unordered_map<CarrierId, std::vector<PhysicalCommandId>> accesses;
  std::vector<std::vector<bool>> happens_before;

  [[nodiscard]] bool lifetimes_conflict(const CarrierRequirement& lhs,
                                        const CarrierRequirement& rhs) const noexcept;
};

[[nodiscard]] std::optional<ModelStorageRequirements> collect_model_storage_requirements(
    const ModelExecutionPlan& execution_plan, const PhysicalExecutionPlan& physical_plan,
    std::uint64_t default_alignment_bytes, std::string* error = nullptr,
    ::simaai::scheduler::client::AllocationPolicy eligible_dms_policy =
        ::simaai::scheduler::client::AllocationPolicy::RequireDms,
    std::span<const ValueId> detached_roots = {}, const ModelStorageRoute* route = nullptr);

} // namespace simaai::neat::pipeline_internal::sima::static_contract

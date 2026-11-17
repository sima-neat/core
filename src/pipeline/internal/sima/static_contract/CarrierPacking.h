#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <simaai/scheduler/client/frame_layout.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::static_contract {

struct StorageLifetime {
  std::uint64_t first_sequence = 0;
  std::uint64_t last_sequence = 0;
};

enum class StorageReuse { Disabled, DisjointLifetimes };

[[nodiscard]] std::optional<::simaai::scheduler::client::AllocationPolicy>
resolve_allocation_policy(std::uint32_t device_access,
                          ::simaai::scheduler::client::AllocationPolicy eligible_policy =
                              ::simaai::scheduler::client::AllocationPolicy::RequireDms) noexcept;

// Temporary checked requirements for one materialized root carrier. Views do
// not enter the packer: their offsets remain relative to this root.
struct CarrierRequirement {
  std::uint64_t carrier_id = 0;
  std::uint64_t size_bytes = 0;
  std::uint64_t required_alignment_bytes = 0;
  StorageLifetime lifetime;
  ::simaai::scheduler::client::AllocationPolicy policy =
      ::simaai::scheduler::client::AllocationPolicy::RequireDms;
  std::uint32_t required_device_access = 0;
};

struct PackedCarrier {
  std::uint64_t carrier_id = 0;
  ::simaai::scheduler::client::FrameSpan span;
};

// A packing result for one output ownership group. Allocation indices and
// offsets are local to this group, as required by FrameLayout.
struct CarrierPacking {
  std::vector<::simaai::scheduler::client::FrameAllocation> allocations;
  std::vector<PackedCarrier> carriers;

  [[nodiscard]] const ::simaai::scheduler::client::FrameSpan*
  span(std::uint64_t carrier_id) const noexcept;
};

using CarrierLifetimeConflict =
    std::function<bool(const CarrierRequirement&, const CarrierRequirement&)>;

[[nodiscard]] std::optional<CarrierPacking>
pack_carriers(std::vector<CarrierRequirement> carriers, StorageReuse reuse,
              const CarrierLifetimeConflict& lifetimes_conflict, std::string* error = nullptr);

} // namespace simaai::neat::pipeline_internal::sima::static_contract

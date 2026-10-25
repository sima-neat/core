#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/static_contract/FrameSlotArenaPlan.h"
#include <functional>
#include <string>

namespace simaai::neat::pipeline_internal::sima::static_contract {

struct CarrierArenaRegion {
  std::uint64_t carrier_id = 0;
  std::uint64_t byte_offset = 0;
  std::uint64_t size_bytes = 0;
  std::uint64_t required_alignment_bytes = 0;
  ValueLifetime lifetime;
  ArenaStorageDomain storage_domain = ArenaStorageDomain::Unknown;
  std::uint32_t required_device_access = 0;
  bool public_output = false;
};

struct CarrierArenaPacking {
  std::vector<CarrierArenaRegion> regions;
  std::vector<FrameSlotArenaAllocation> allocations;
  std::uint64_t used_bytes = 0;
  std::uint64_t allocation_bytes = 0;
  std::uint64_t allocation_alignment_bytes = 0;
};

using CarrierLifetimeConflict =
    std::function<bool(const CarrierArenaRegion&, const CarrierArenaRegion&)>;

std::optional<CarrierArenaPacking>
pack_carrier_arena(std::vector<CarrierArenaRegion> regions, FrameSlotArenaReuse reuse,
                   const CarrierLifetimeConflict& lifetimes_conflict, ArenaDmsPolicy dms_policy,
                   std::string* error = nullptr);

} // namespace simaai::neat::pipeline_internal::sima::static_contract

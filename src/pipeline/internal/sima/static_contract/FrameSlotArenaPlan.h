#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/static_contract/ModelExecutionPlan.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::static_contract {

struct PhysicalExecutionPlan;

enum class ArenaStorageDomain : std::uint8_t {
  Unknown = 0,
  Cma = 1,
  Dms = 2,
};

enum class ArenaAllocationProvenance : std::uint8_t {
  Unknown = 0,
  CoreAllocated = 1,
  ExternalAdopted = 2,
};

enum class ArenaDeviceAccess : std::uint32_t {
  None = 0U,
  CpuA65 = 1U << 0U,
  Mla = 1U << 1U,
  Ev74 = 1U << 2U,
};

enum class ArenaEscapePolicy : std::uint8_t {
  InternalOnly = 0,
  CpuMappablePublic = 1,
};

enum class ArenaDmsPolicy : std::uint8_t {
  CmaOnly = 0,
  PreferDmsForEligible = 1,
};

// Modalix production placement.  DMS is CPU/A65+MLA visible; any EV74
// requirement still forces CMA in FrameSlotArenaPlan::compile_impl().
// Keeping the rollout choice named and explicit prevents production callers
// from silently inheriting the conservative test/default policy.
inline constexpr ArenaDmsPolicy kModalixProductionArenaDmsPolicy =
    ArenaDmsPolicy::PreferDmsForEligible;

struct ArenaPlacement {
  ArenaStorageDomain domain = ArenaStorageDomain::Unknown;
  ArenaAllocationProvenance provenance = ArenaAllocationProvenance::Unknown;
  std::uint32_t required_device_access = 0U;
  ArenaEscapePolicy escape = ArenaEscapePolicy::InternalOnly;

  [[nodiscard]] bool requires_access(ArenaDeviceAccess access) const noexcept {
    return (required_device_access & static_cast<std::uint32_t>(access)) != 0U;
  }
};

// Operation sequences are closed intervals: an operation's inputs remain live
// while its outputs are produced. Consequently, [a,b] and [b,c] may not share
// storage. This is the conservative rule required by asynchronous devices.
struct ValueLifetime {
  std::uint64_t first_sequence = 0;
  std::uint64_t last_sequence = 0;
};

struct FrameSlotArenaRegion {
  CarrierId carrier_id = 0;
  // First materialized logical value bound to this carrier. Kept for existing
  // diagnostics; region(ValueId) maps every bound value to this one region.
  ValueId value_id = 0;
  std::uint64_t byte_offset = 0;
  std::uint64_t size_bytes = 0;
  std::uint64_t required_alignment_bytes = 0;
  ValueLifetime lifetime;
};

enum class FrameSlotArenaReuse {
  Disabled,
  DisjointLifetimes,
};

// Immutable conservative placement of materialized non-input carriers in one
// graph frame arena. Multiple values/views/direct producer destinations may
// map to one region. A stage projection may detach an exact terminal public
// MLA output into its own typed pool; every remaining region keeps this plan's
// offset and access authority.
class FrameSlotArenaPlan final {
public:
  [[nodiscard]] static std::optional<FrameSlotArenaPlan>
  compile(const ModelExecutionPlan& execution_plan, FrameSlotArenaReuse reuse,
          std::uint64_t default_region_alignment_bytes =
              kLegacyEvoCmaRegionAlignmentBytes,
          std::string* error = nullptr,
          ArenaDmsPolicy dms_policy = ArenaDmsPolicy::CmaOnly);

  // Production placement is compiled against the physical command DAG, not
  // semantic operation order.  Several semantic operations may be one backend
  // submission, while independent commands may execute concurrently.  Storage
  // is reused only when the command DAG proves every access to one carrier
  // happens before every access to the other carrier.
  [[nodiscard]] static std::optional<FrameSlotArenaPlan>
  compile(const ModelExecutionPlan& execution_plan,
          const PhysicalExecutionPlan& physical_plan,
          FrameSlotArenaReuse reuse,
          std::uint64_t default_region_alignment_bytes =
              kLegacyEvoCmaRegionAlignmentBytes,
          std::string* error = nullptr,
          ArenaDmsPolicy dms_policy = ArenaDmsPolicy::CmaOnly,
          std::span<const ValueId> detached_roots = {});

  [[nodiscard]] const std::vector<FrameSlotArenaRegion>& regions() const noexcept;
  [[nodiscard]] std::span<const ValueId> detached_roots() const noexcept;
  [[nodiscard]] bool is_detached_root(ValueId value_id) const noexcept;
  [[nodiscard]] const FrameSlotArenaRegion* region(ValueId value_id) const noexcept;
  [[nodiscard]] std::uint64_t used_bytes() const noexcept;
  [[nodiscard]] std::uint64_t allocation_bytes() const noexcept;
  [[nodiscard]] std::uint64_t allocation_alignment_bytes() const noexcept;
  [[nodiscard]] FrameSlotArenaReuse reuse_policy() const noexcept;
  [[nodiscard]] const ArenaPlacement& placement() const noexcept;

private:
  [[nodiscard]] static std::optional<FrameSlotArenaPlan>
  compile_impl(const ModelExecutionPlan& execution_plan,
               const PhysicalExecutionPlan* physical_plan,
               FrameSlotArenaReuse reuse,
               std::uint64_t default_region_alignment_bytes,
               std::string* error, ArenaDmsPolicy dms_policy,
               std::span<const ValueId> detached_roots);

  std::vector<FrameSlotArenaRegion> regions_;
  std::vector<ValueId> detached_roots_;
  std::vector<std::size_t> value_to_region_;
  std::uint64_t used_bytes_ = 0;
  std::uint64_t allocation_bytes_ = 0;
  std::uint64_t allocation_alignment_bytes_ = 0;
  FrameSlotArenaReuse reuse_policy_ = FrameSlotArenaReuse::Disabled;
  ArenaPlacement placement_;
};

} // namespace simaai::neat::pipeline_internal::sima::static_contract

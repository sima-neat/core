#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/static_contract/ModelExecutionPlan.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::static_contract {

// Operation sequences are closed intervals: an operation's inputs remain live
// while its outputs are produced. Consequently, [a,b] and [b,c] may not share
// storage. This is the conservative rule required by asynchronous devices.
struct ValueLifetime {
  std::uint64_t first_sequence = 0;
  std::uint64_t last_sequence = 0;
};

struct FrameSlotArenaRegion {
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

// Immutable compile-time placement of materialized non-input values in one
// per-frame DMA-BUF. ReadExpression values never receive regions: they extend
// the lifetime of, and remain address expressions over, their root carrier.
class FrameSlotArenaPlan final {
public:
  [[nodiscard]] static std::optional<FrameSlotArenaPlan>
  compile(const ModelExecutionPlan& execution_plan, FrameSlotArenaReuse reuse,
          std::uint64_t default_region_alignment_bytes =
              kLegacyEvoCmaRegionAlignmentBytes,
          std::string* error = nullptr);

  [[nodiscard]] const std::vector<FrameSlotArenaRegion>& regions() const noexcept;
  [[nodiscard]] const FrameSlotArenaRegion* region(ValueId value_id) const noexcept;
  [[nodiscard]] std::uint64_t used_bytes() const noexcept;
  [[nodiscard]] std::uint64_t allocation_bytes() const noexcept;
  [[nodiscard]] std::uint64_t allocation_alignment_bytes() const noexcept;
  [[nodiscard]] FrameSlotArenaReuse reuse_policy() const noexcept;

private:
  std::vector<FrameSlotArenaRegion> regions_;
  std::vector<std::size_t> value_to_region_;
  std::uint64_t used_bytes_ = 0;
  std::uint64_t allocation_bytes_ = 0;
  std::uint64_t allocation_alignment_bytes_ = 0;
  FrameSlotArenaReuse reuse_policy_ = FrameSlotArenaReuse::Disabled;
};

} // namespace simaai::neat::pipeline_internal::sima::static_contract

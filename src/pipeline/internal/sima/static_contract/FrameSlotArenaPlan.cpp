#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/static_contract/FrameSlotArenaPlan.h"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

namespace simaai::neat::pipeline_internal::sima::static_contract {
namespace {

constexpr std::size_t kNoRegion = std::numeric_limits<std::size_t>::max();

bool fail(std::string* error, std::string detail) {
  if (error) {
    *error = std::move(detail);
  }
  return false;
}

bool is_power_of_two(const std::uint64_t value) {
  return value != 0U && (value & (value - 1U)) == 0U;
}

bool checked_add(const std::uint64_t lhs, const std::uint64_t rhs,
                 std::uint64_t* result) {
  if (!result || lhs > std::numeric_limits<std::uint64_t>::max() - rhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

std::optional<std::uint64_t> align_up(const std::uint64_t value,
                                      const std::uint64_t alignment) {
  if (!is_power_of_two(alignment)) {
    return std::nullopt;
  }
  const auto mask = alignment - 1U;
  if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
    return std::nullopt;
  }
  return (value + mask) & ~mask;
}

bool lifetimes_overlap(const ValueLifetime& lhs, const ValueLifetime& rhs) {
  return lhs.first_sequence <= rhs.last_sequence &&
         rhs.first_sequence <= lhs.last_sequence;
}

bool byte_ranges_overlap(const std::uint64_t lhs_offset,
                         const std::uint64_t lhs_size,
                         const std::uint64_t rhs_offset,
                         const std::uint64_t rhs_size) {
  return lhs_offset < rhs_offset + rhs_size && rhs_offset < lhs_offset + lhs_size;
}

ValueId root_value_id(const ModelExecutionPlan& plan, const ValueId value_id) {
  const auto* value = plan.value(value_id);
  return value && value->read_expression.has_value()
             ? value->read_expression->source_value_id
             : value_id;
}

bool place_first_fit(FrameSlotArenaRegion* candidate,
                     const std::vector<FrameSlotArenaRegion>& placed,
                     std::string* error) {
  if (!candidate) {
    return fail(error, "frame-slot arena has no placement candidate");
  }
  std::uint64_t cursor = 0U;
  while (true) {
    const auto aligned = align_up(cursor, candidate->required_alignment_bytes);
    std::uint64_t candidate_end = 0U;
    if (!aligned || !checked_add(*aligned, candidate->size_bytes, &candidate_end)) {
      return fail(error, "frame-slot arena placement overflows");
    }

    bool conflict = false;
    std::uint64_t next_cursor = cursor;
    for (const auto& other : placed) {
      if (!lifetimes_overlap(candidate->lifetime, other.lifetime) ||
          !byte_ranges_overlap(*aligned, candidate->size_bytes,
                               other.byte_offset, other.size_bytes)) {
        continue;
      }
      std::uint64_t other_end = 0U;
      if (!checked_add(other.byte_offset, other.size_bytes, &other_end)) {
        return fail(error, "frame-slot arena contains an overflowing region");
      }
      next_cursor = std::max(next_cursor, other_end);
      conflict = true;
    }
    if (!conflict) {
      candidate->byte_offset = *aligned;
      return true;
    }
    if (next_cursor <= cursor) {
      return fail(error, "frame-slot arena first-fit placement made no progress");
    }
    cursor = next_cursor;
  }
}

} // namespace

std::optional<FrameSlotArenaPlan>
FrameSlotArenaPlan::compile(const ModelExecutionPlan& execution_plan,
                            const FrameSlotArenaReuse reuse,
                            const std::uint64_t default_region_alignment_bytes,
                            std::string* error) {
  if (!is_power_of_two(default_region_alignment_bytes)) {
    fail(error, "frame-slot arena default alignment must be a power of two");
    return std::nullopt;
  }

  const auto& values = execution_plan.values();
  const auto& ops = execution_plan.ops();
  std::vector<bool> is_model_input(values.size(), false);
  std::vector<std::uint64_t> producer_sequence(values.size(), 0U);
  std::vector<std::uint64_t> last_sequence(values.size(), 0U);
  std::vector<std::uint64_t> alignment(values.size(),
                                       default_region_alignment_bytes);

  for (const auto value_id : execution_plan.model_inputs()) {
    is_model_input[value_id] = true;
  }
  for (const auto& op : ops) {
    for (const auto value_id : op.outputs) {
      const auto* value = execution_plan.value(value_id);
      // Unpack/Slice outputs are address expressions over an existing root.
      // They extend that root's lifetime through their consumers, but they do
      // not replace its real producer sequence.
      if (value && !value->read_expression.has_value()) {
        producer_sequence[value_id] = op.sequence;
      }
    }
    for (const auto value_id : op.inputs) {
      const auto root = root_value_id(execution_plan, value_id);
      last_sequence[root] = std::max(last_sequence[root], op.sequence);
    }
  }
  const std::uint64_t public_output_sequence =
      static_cast<std::uint64_t>(ops.size()) + 1U;
  for (const auto& output : execution_plan.model_outputs()) {
    const auto root = root_value_id(execution_plan, output.value_id);
    last_sequence[root] = std::max(last_sequence[root], public_output_sequence);
  }
  for (const auto& port : execution_plan.backend_ports()) {
    const auto root = root_value_id(execution_plan, port.value_id);
    alignment[root] = std::max<std::uint64_t>(
        alignment[root], port.required_alignment_bytes);
  }
  // A materializing transform may conservatively inherit the strongest input
  // offset constraint. This keeps post-MLA CVU outputs legal without inventing
  // a second, stage-local arena policy; the published physical contract still
  // reports the exact device-port requirement rather than this safe placement
  // over-alignment.
  for (const auto& op : ops) {
    std::uint64_t inherited_alignment = default_region_alignment_bytes;
    for (const auto input_id : op.inputs) {
      inherited_alignment =
          std::max(inherited_alignment,
                   alignment[root_value_id(execution_plan, input_id)]);
    }
    for (const auto output_id : op.outputs) {
      const auto root = root_value_id(execution_plan, output_id);
      alignment[root] = std::max(alignment[root], inherited_alignment);
    }
  }

  FrameSlotArenaPlan result;
  result.reuse_policy_ = reuse;
  result.value_to_region_.assign(values.size(), kNoRegion);
  result.regions_.reserve(values.size());
  for (const auto& value : values) {
    if (is_model_input[value.id] || value.read_expression.has_value()) {
      continue;
    }
    if (producer_sequence[value.id] == 0U) {
      fail(error, "frame-slot arena materialized value has no producer");
      return std::nullopt;
    }
    FrameSlotArenaRegion region;
    region.value_id = value.id;
    region.size_bytes = value.required_bytes;
    region.required_alignment_bytes = alignment[value.id];
    region.lifetime.first_sequence = producer_sequence[value.id];
    region.lifetime.last_sequence =
        std::max(producer_sequence[value.id], last_sequence[value.id]);
    result.regions_.push_back(std::move(region));
  }
  if (result.regions_.empty()) {
    fail(error, "frame-slot arena has no materialized internal values");
    return std::nullopt;
  }

  std::stable_sort(result.regions_.begin(), result.regions_.end(),
                   [](const auto& lhs, const auto& rhs) {
                     if (lhs.lifetime.first_sequence != rhs.lifetime.first_sequence) {
                       return lhs.lifetime.first_sequence < rhs.lifetime.first_sequence;
                     }
                     return lhs.value_id < rhs.value_id;
                   });

  std::uint64_t cursor = 0U;
  if (reuse == FrameSlotArenaReuse::Disabled) {
    for (auto& region : result.regions_) {
      const auto offset = align_up(cursor, region.required_alignment_bytes);
      if (!offset || !checked_add(*offset, region.size_bytes, &cursor)) {
        fail(error, "frame-slot no-reuse oracle overflows");
        return std::nullopt;
      }
      region.byte_offset = *offset;
      result.used_bytes_ = cursor;
    }
  } else {
    struct LifetimeGroup {
      ValueLifetime lifetime;
      std::vector<std::size_t> region_indices;
      std::uint64_t used_bytes = 0U;
      std::uint64_t alignment = 0U;
    };
    std::vector<LifetimeGroup> groups;
    for (std::size_t index = 0; index < result.regions_.size(); ++index) {
      const auto lifetime = result.regions_[index].lifetime;
      auto group = std::find_if(groups.begin(), groups.end(), [&](const auto& item) {
        return item.lifetime.first_sequence == lifetime.first_sequence &&
               item.lifetime.last_sequence == lifetime.last_sequence;
      });
      if (group == groups.end()) {
        groups.push_back({lifetime, {}, 0U, 0U});
        group = std::prev(groups.end());
      }
      group->region_indices.push_back(index);
    }
    for (auto& group : groups) {
      std::stable_sort(group.region_indices.begin(), group.region_indices.end(),
                       [&](const auto lhs, const auto rhs) {
                         const auto& left = result.regions_[lhs];
                         const auto& right = result.regions_[rhs];
                         if (left.required_alignment_bytes !=
                             right.required_alignment_bytes) {
                           return left.required_alignment_bytes >
                                  right.required_alignment_bytes;
                         }
                         if (left.size_bytes != right.size_bytes) {
                           return left.size_bytes > right.size_bytes;
                         }
                         return left.value_id < right.value_id;
                       });
      std::uint64_t local_cursor = 0U;
      for (const auto index : group.region_indices) {
        auto& region = result.regions_[index];
        const auto local =
            align_up(local_cursor, region.required_alignment_bytes);
        if (!local || !checked_add(*local, region.size_bytes, &local_cursor)) {
          fail(error, "frame-slot lifetime-group layout overflows");
          return std::nullopt;
        }
        region.byte_offset = *local;
        group.alignment =
            std::max(group.alignment, region.required_alignment_bytes);
      }
      group.used_bytes = local_cursor;
    }
    std::stable_sort(groups.begin(), groups.end(), [](const auto& lhs, const auto& rhs) {
      if (lhs.used_bytes != rhs.used_bytes) {
        return lhs.used_bytes > rhs.used_bytes;
      }
      if (lhs.alignment != rhs.alignment) {
        return lhs.alignment > rhs.alignment;
      }
      if (lhs.lifetime.first_sequence != rhs.lifetime.first_sequence) {
        return lhs.lifetime.first_sequence < rhs.lifetime.first_sequence;
      }
      return lhs.lifetime.last_sequence < rhs.lifetime.last_sequence;
    });

    std::vector<FrameSlotArenaRegion> placed;
    placed.reserve(result.regions_.size());
    for (const auto& group : groups) {
      FrameSlotArenaRegion block;
      block.size_bytes = group.used_bytes;
      block.required_alignment_bytes = group.alignment;
      block.lifetime = group.lifetime;
      if (!place_first_fit(&block, placed, error)) {
        return std::nullopt;
      }
      for (const auto index : group.region_indices) {
        auto& region = result.regions_[index];
        if (!checked_add(block.byte_offset, region.byte_offset,
                         &region.byte_offset)) {
          fail(error, "frame-slot lifetime-group base overflows");
          return std::nullopt;
        }
        std::uint64_t end = 0U;
        if (!checked_add(region.byte_offset, region.size_bytes, &end)) {
          fail(error, "frame-slot arena region end overflows");
          return std::nullopt;
        }
        result.used_bytes_ = std::max(result.used_bytes_, end);
        placed.push_back(region);
      }
    }
  }

  for (std::size_t index = 0; index < result.regions_.size(); ++index) {
    const auto& region = result.regions_[index];
    result.value_to_region_[region.value_id] = index;
    result.allocation_alignment_bytes_ =
        std::max(result.allocation_alignment_bytes_,
                 region.required_alignment_bytes);
    if (region.byte_offset % region.required_alignment_bytes != 0U) {
      fail(error, "frame-slot arena produced an unaligned region");
      return std::nullopt;
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      const auto& other = result.regions_[previous];
      if (lifetimes_overlap(region.lifetime, other.lifetime) &&
          byte_ranges_overlap(region.byte_offset, region.size_bytes,
                              other.byte_offset, other.size_bytes)) {
        fail(error, "frame-slot arena reuses storage across overlapping lifetimes");
        return std::nullopt;
      }
    }
  }
  const auto allocation =
      align_up(result.used_bytes_, result.allocation_alignment_bytes_);
  if (!allocation || *allocation == 0U) {
    fail(error, "frame-slot arena allocation rounding overflows");
    return std::nullopt;
  }
  result.allocation_bytes_ = *allocation;
  if (error) {
    error->clear();
  }
  return result;
}

const std::vector<FrameSlotArenaRegion>&
FrameSlotArenaPlan::regions() const noexcept {
  return regions_;
}

const FrameSlotArenaRegion*
FrameSlotArenaPlan::region(const ValueId value_id) const noexcept {
  if (value_id >= value_to_region_.size()) {
    return nullptr;
  }
  const auto index = value_to_region_[value_id];
  return index == kNoRegion ? nullptr : &regions_[index];
}

std::uint64_t FrameSlotArenaPlan::used_bytes() const noexcept {
  return used_bytes_;
}

std::uint64_t FrameSlotArenaPlan::allocation_bytes() const noexcept {
  return allocation_bytes_;
}

std::uint64_t FrameSlotArenaPlan::allocation_alignment_bytes() const noexcept {
  return allocation_alignment_bytes_;
}

FrameSlotArenaReuse FrameSlotArenaPlan::reuse_policy() const noexcept {
  return reuse_policy_;
}

} // namespace simaai::neat::pipeline_internal::sima::static_contract

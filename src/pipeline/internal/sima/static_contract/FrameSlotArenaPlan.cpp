#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/static_contract/FrameSlotArenaPlan.h"
#include "pipeline/internal/sima/static_contract/PhysicalExecutionPlan.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

struct PhysicalCarrierSchedule {
  std::unordered_map<CarrierId, std::vector<PhysicalCommandId>> accesses;
  // Strict transitive happens-before relation.  The final row/column is a
  // synthetic model-completion command used by public outputs.
  std::vector<std::vector<bool>> happens_before;
  PhysicalCommandId completion = 0;
};

bool all_accesses_happen_before(const std::vector<PhysicalCommandId>& lhs,
                                const std::vector<PhysicalCommandId>& rhs,
                                const PhysicalCarrierSchedule& schedule) {
  if (lhs.empty() || rhs.empty()) {
    return false;
  }
  for (const auto left : lhs) {
    for (const auto right : rhs) {
      if (left >= schedule.happens_before.size() ||
          right >= schedule.happens_before.size() ||
          !schedule.happens_before[left][right]) {
        return false;
      }
    }
  }
  return true;
}

bool physical_lifetimes_overlap(const FrameSlotArenaRegion& lhs,
                                const FrameSlotArenaRegion& rhs,
                                const PhysicalCarrierSchedule& schedule) {
  const auto left = schedule.accesses.find(lhs.carrier_id);
  const auto right = schedule.accesses.find(rhs.carrier_id);
  if (left == schedule.accesses.end() || right == schedule.accesses.end()) {
    // Missing command provenance is invalidated by compile_impl.  Treat it as
    // live here as a final fail-safe rather than permitting unsafe reuse.
    return true;
  }
  return !all_accesses_happen_before(left->second, right->second, schedule) &&
         !all_accesses_happen_before(right->second, left->second, schedule);
}

CarrierId carrier_id(const ModelExecutionPlan& plan, const ValueId value_id) {
  const auto* value = plan.value(value_id);
  return value && value->storage_binding.has_value()
             ? value->storage_binding->carrier_id
             : static_cast<CarrierId>(value_id);
}

using LifetimeConflict =
    std::function<bool(const FrameSlotArenaRegion&, const FrameSlotArenaRegion&)>;

bool place_group_first_fit(
    const std::vector<std::size_t>& group_indices,
    std::vector<FrameSlotArenaRegion>* regions,
    const std::uint64_t group_alignment,
    const std::vector<FrameSlotArenaRegion>& placed,
    const LifetimeConflict& lifetimes_conflict,
    std::uint64_t* group_base, std::string* error) {
  if (!regions || !group_base || group_indices.empty() ||
      !is_power_of_two(group_alignment)) {
    return fail(error, "frame-slot arena has no valid placement group");
  }
  std::uint64_t cursor = 0U;
  while (true) {
    const auto aligned = align_up(cursor, group_alignment);
    if (!aligned) {
      return fail(error, "frame-slot arena placement overflows");
    }

    bool conflict = false;
    std::uint64_t next_cursor = cursor;
    for (const auto index : group_indices) {
      auto& member = (*regions)[index];
      const auto local_offset = member.byte_offset;
      std::uint64_t member_offset = 0U;
      if (!checked_add(*aligned, local_offset, &member_offset)) {
        return fail(error, "frame-slot arena group placement overflows");
      }
      for (const auto& other : placed) {
        if (!lifetimes_conflict(member, other) ||
            !byte_ranges_overlap(member_offset, member.size_bytes,
                                 other.byte_offset, other.size_bytes)) {
          continue;
        }
        std::uint64_t other_end = 0U;
        if (!checked_add(other.byte_offset, other.size_bytes, &other_end)) {
          return fail(error, "frame-slot arena contains an overflowing region");
        }
        // Move this member immediately after the conflicting range.  Because
        // member_offset = group_base + local_offset, subtracting its fixed
        // local offset finds the earliest possible next group base.
        const auto required_base =
            other_end > local_offset ? other_end - local_offset : 0U;
        next_cursor = std::max(next_cursor, required_base);
        conflict = true;
      }
    }
    if (!conflict) {
      *group_base = *aligned;
      return true;
    }
    if (next_cursor <= *aligned) {
      return fail(error, "frame-slot arena first-fit placement made no progress");
    }
    cursor = next_cursor;
  }
}

} // namespace

std::optional<FrameSlotArenaPlan>
FrameSlotArenaPlan::compile_impl(
    const ModelExecutionPlan& execution_plan,
    const PhysicalExecutionPlan* physical_plan,
    const FrameSlotArenaReuse reuse,
    const std::uint64_t default_region_alignment_bytes,
    std::string* error, const ArenaDmsPolicy dms_policy,
    const std::span<const ValueId> detached_roots) {
  if (!is_power_of_two(default_region_alignment_bytes)) {
    fail(error, "frame-slot arena default alignment must be a power of two");
    return std::nullopt;
  }

  const auto& values = execution_plan.values();
  const auto& ops = execution_plan.ops();
  std::unordered_set<CarrierId> external_carriers;
  std::unordered_map<CarrierId, std::uint64_t> producer_sequence;
  std::unordered_map<CarrierId, std::uint64_t> last_sequence;
  std::unordered_map<CarrierId, std::uint64_t> alignment;
  std::unordered_map<ValueId, CarrierId> direct_pack_parent_for_child;
  std::unordered_set<CarrierId> physically_required_carriers;
  std::unordered_set<CarrierId> detached_carriers;
  std::unordered_set<ValueId> detached_values(detached_roots.begin(),
                                              detached_roots.end());
  std::unordered_set<ValueId> elided_values;
  std::optional<PhysicalCarrierSchedule> physical_schedule;

  for (const auto value_id : execution_plan.model_inputs()) {
    external_carriers.emplace(carrier_id(execution_plan, value_id));
  }
  for (const auto value_id : detached_roots) {
    const auto* value = execution_plan.value(value_id);
    if (!value || value->read_expression.has_value() ||
        external_carriers.contains(carrier_id(execution_plan, value_id))) {
      fail(error, "frame-slot detached output root is invalid");
      return std::nullopt;
    }
    detached_carriers.emplace(carrier_id(execution_plan, value_id));
  }
  for (const auto& value : values) {
    if (!value.storage_binding.has_value() || value.read_expression.has_value() ||
        !detached_carriers.contains(value.storage_binding->carrier_id)) {
      continue;
    }
    if (!detached_values.contains(value.id)) {
      fail(error, "frame-slot detached carrier also owns a retained materialized root");
      return std::nullopt;
    }
  }
  for (const auto& carrier : execution_plan.carriers()) {
    alignment[carrier.id] =
        std::max<std::uint64_t>(default_region_alignment_bytes,
                                carrier.required_alignment_bytes);
  }

  // Frozen batch-one contracts describe direct Pack placement with component
  // offsets but predate shared carrier bindings.  The Pack itself is still an
  // address relation: attribute the parent write to the exact commands that
  // produce its children.  Current span contracts already bind children to the
  // parent carrier and therefore need no compatibility projection here.
  for (const auto& op : ops) {
    if (op.kind != OpKind::Pack || op.outputs.size() != 1U) {
      continue;
    }
    const auto* pack = std::get_if<PackOpConfig>(&op.config);
    const auto* parent = execution_plan.value(op.outputs.front());
    if (!pack || pack->materializes || !pack->spans.empty() ||
        pack->components.size() != op.inputs.size() || !parent ||
        !parent->storage_binding) {
      continue;
    }
    for (const auto child : op.inputs) {
      const auto* value = execution_plan.value(child);
      if (value && value->storage_binding &&
          value->storage_binding->carrier_id !=
              parent->storage_binding->carrier_id) {
        direct_pack_parent_for_child.emplace(
            child, parent->storage_binding->carrier_id);
      }
    }
  }

  if (physical_plan) {
    if (physical_plan->commands.size() >=
        static_cast<std::size_t>(std::numeric_limits<PhysicalCommandId>::max())) {
      fail(error, "frame-slot physical command count exceeds ABI capacity");
      return std::nullopt;
    }
    PhysicalCarrierSchedule schedule;
    schedule.completion =
        static_cast<PhysicalCommandId>(physical_plan->commands.size());
    schedule.happens_before.assign(
        physical_plan->commands.size() + 1U,
        std::vector<bool>(physical_plan->commands.size() + 1U, false));

    // A member chain is one physical invocation.  Prove and record every
    // chain-internal edge before considering storage so it can neither acquire
    // an arena region nor accidentally reappear as an outer/backend/public
    // binding.
    for (const auto& command : physical_plan->commands) {
      for (const auto& member : command.members) {
        for (std::size_t chain_index = 1U;
             chain_index < member.semantic_chain.size(); ++chain_index) {
          const auto first_id = member.semantic_chain[chain_index - 1U];
          const auto second_id = member.semantic_chain[chain_index];
          if (first_id >= ops.size() || second_id >= ops.size()) {
            fail(error, "frame-slot fused member has out-of-range semantic provenance");
            return std::nullopt;
          }
          const auto& first = ops[first_id];
          const auto& second = ops[second_id];
          std::vector<ValueId> internal_values;
          if (!resolve_exact_private_ordered_relation_path(
                  execution_plan, first.id, second.id, &internal_values) ||
              internal_values.empty()) {
            fail(error,
                 "frame-slot fused member must have one exact private ordered-view path");
            return std::nullopt;
          }
          elided_values.insert(internal_values.begin(), internal_values.end());
        }
      }
    }

    for (std::size_t index = 0; index < physical_plan->commands.size(); ++index) {
      const auto& command = physical_plan->commands[index];
      if (command.id != index || command.topological_rank != index) {
        fail(error, "frame-slot physical commands must have dense topological ids");
        return std::nullopt;
      }
      for (const auto predecessor : command.predecessors) {
        if (predecessor >= command.id) {
          fail(error, "frame-slot physical command has a non-topological predecessor");
          return std::nullopt;
        }
        schedule.happens_before[predecessor][command.id] = true;
        for (std::size_t ancestor = 0; ancestor < command.id; ++ancestor) {
          if (schedule.happens_before[ancestor][predecessor]) {
            schedule.happens_before[ancestor][command.id] = true;
          }
        }
      }
      schedule.happens_before[command.id][schedule.completion] = true;

      const auto sequence = static_cast<std::uint64_t>(command.id) + 1U;
      for (const auto value_id : command.inputs) {
        if (value_id >= values.size()) {
          fail(error, "frame-slot physical command input is outside the value table");
          return std::nullopt;
        }
        if (elided_values.contains(value_id)) {
          fail(error, "frame-slot fused internal edge appears as a physical input");
          return std::nullopt;
        }
        const auto id = carrier_id(execution_plan, value_id);
        physically_required_carriers.emplace(id);
        schedule.accesses[id].push_back(command.id);
        last_sequence[id] = std::max(last_sequence[id], sequence);
      }
      for (const auto value_id : command.outputs) {
        if (value_id >= values.size()) {
          fail(error, "frame-slot physical command output is outside the value table");
          return std::nullopt;
        }
        if (elided_values.contains(value_id)) {
          fail(error, "frame-slot fused internal edge appears as a physical output");
          return std::nullopt;
        }
        const auto* value = execution_plan.value(value_id);
        const auto id = carrier_id(execution_plan, value_id);
        physically_required_carriers.emplace(id);
        schedule.accesses[id].push_back(command.id);
        last_sequence[id] = std::max(last_sequence[id], sequence);
        // An outer command output is a physical write even when the semantic
        // binding is a proved nonzero-offset view into a producer-direct Pack
        // parent. Attribute production to the root carrier; only relation-only
        // alias operations are absent from PhysicalExecutionPlan::outputs.
        if (value && value->storage_binding.has_value()) {
          auto [iterator, inserted] = producer_sequence.emplace(id, sequence);
          if (!inserted) {
            iterator->second = std::min(iterator->second, sequence);
          }
        }
        const auto direct_parent = direct_pack_parent_for_child.find(value_id);
        if (direct_parent != direct_pack_parent_for_child.end()) {
          physically_required_carriers.emplace(direct_parent->second);
          schedule.accesses[direct_parent->second].push_back(command.id);
          last_sequence[direct_parent->second] =
              std::max(last_sequence[direct_parent->second], sequence);
          auto [iterator, inserted] =
              producer_sequence.emplace(direct_parent->second, sequence);
          if (!inserted) {
            iterator->second = std::min(iterator->second, sequence);
          }
        }
      }
    }
    const auto public_output_sequence =
        static_cast<std::uint64_t>(schedule.completion) + 1U;
    for (const auto& output : execution_plan.model_outputs()) {
      if (elided_values.contains(output.value_id)) {
        fail(error, "frame-slot fused internal edge is a public output");
        return std::nullopt;
      }
      const auto id = carrier_id(execution_plan, output.value_id);
      physically_required_carriers.emplace(id);
      schedule.accesses[id].push_back(schedule.completion);
      last_sequence[id] = std::max(last_sequence[id], public_output_sequence);
    }
    for (auto& [id, accesses] : schedule.accesses) {
      (void)id;
      std::sort(accesses.begin(), accesses.end());
      accesses.erase(std::unique(accesses.begin(), accesses.end()), accesses.end());
    }
    physical_schedule = std::move(schedule);
  } else {
    for (const auto& op : ops) {
      for (const auto value_id : op.outputs) {
        const auto* value = execution_plan.value(value_id);
        if (value && value->storage_binding.has_value() &&
            value->storage_binding->kind != StorageBindingKind::View) {
          const auto id = value->storage_binding->carrier_id;
          auto [iterator, inserted] = producer_sequence.emplace(id, op.sequence);
          if (!inserted) {
            iterator->second = std::min(iterator->second, op.sequence);
          }
        }
      }
      for (const auto value_id : op.inputs) {
        const auto id = carrier_id(execution_plan, value_id);
        last_sequence[id] = std::max(last_sequence[id], op.sequence);
      }
    }
    const std::uint64_t public_output_sequence =
        ops.empty() ? 1U : ops.back().sequence + 1U;
    for (const auto& output : execution_plan.model_outputs()) {
      const auto id = carrier_id(execution_plan, output.value_id);
      last_sequence[id] = std::max(last_sequence[id], public_output_sequence);
    }
  }

  for (const auto& port : execution_plan.backend_ports()) {
    if (physical_plan && elided_values.contains(port.value_id)) {
      fail(error, "frame-slot fused internal edge is bound to a backend port");
      return std::nullopt;
    }
    const auto id = carrier_id(execution_plan, port.value_id);
    if (physical_plan) {
      physically_required_carriers.emplace(id);
    }
    alignment[id] =
        std::max<std::uint64_t>(alignment[id], port.required_alignment_bytes);
  }
  // A materializing transform may conservatively inherit the strongest input
  // offset constraint. This keeps post-MLA CVU outputs legal without inventing
  // a second, stage-local arena policy; the published physical contract still
  // reports the exact device-port requirement rather than this safe placement
  // over-alignment.
  for (const auto& op : ops) {
    std::uint64_t inherited_alignment = default_region_alignment_bytes;
    for (const auto input_id : op.inputs) {
      inherited_alignment = std::max(
          inherited_alignment, alignment[carrier_id(execution_plan, input_id)]);
    }
    for (const auto output_id : op.outputs) {
      const auto id = carrier_id(execution_plan, output_id);
      alignment[id] = std::max(alignment[id], inherited_alignment);
    }
  }

  FrameSlotArenaPlan result;
  result.reuse_policy_ = reuse;
  result.detached_roots_.assign(detached_values.begin(), detached_values.end());
  std::sort(result.detached_roots_.begin(), result.detached_roots_.end());
  result.value_to_region_.assign(values.size(), kNoRegion);
  result.regions_.reserve(execution_plan.carriers().size());
  for (const auto& carrier : execution_plan.carriers()) {
    if (external_carriers.contains(carrier.id)) {
      continue;
    }
    if (detached_carriers.contains(carrier.id)) {
      continue;
    }
    if (physical_plan && !physically_required_carriers.contains(carrier.id)) {
      continue;
    }
    const auto producer = producer_sequence.find(carrier.id);
    if (producer == producer_sequence.end() || producer->second == 0U) {
      fail(error, "frame-slot arena materialized carrier has no producer");
      return std::nullopt;
    }
    if (physical_schedule && !physical_schedule->accesses.contains(carrier.id)) {
      fail(error, "frame-slot arena carrier has no physical command provenance");
      return std::nullopt;
    }
    FrameSlotArenaRegion region;
    region.carrier_id = carrier.id;
    const auto first_value = std::find_if(values.begin(), values.end(), [&](const ValueSpec& value) {
      return value.storage_binding.has_value() &&
             value.storage_binding->carrier_id == carrier.id &&
             value.storage_binding->kind != StorageBindingKind::View;
    });
    if (first_value == values.end()) {
      fail(error, "frame-slot arena carrier has no materialized logical binding");
      return std::nullopt;
    }
    region.value_id = first_value->id;
    region.size_bytes = carrier.required_bytes;
    region.required_alignment_bytes = alignment[carrier.id];
    region.lifetime.first_sequence = producer->second;
    region.lifetime.last_sequence =
        std::max(producer->second, last_sequence[carrier.id]);
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
                     return lhs.carrier_id < rhs.carrier_id;
                   });

  const LifetimeConflict lifetimes_conflict =
      physical_schedule
          ? LifetimeConflict{[&](const FrameSlotArenaRegion& lhs,
                                 const FrameSlotArenaRegion& rhs) {
              return physical_lifetimes_overlap(lhs, rhs, *physical_schedule);
            }}
          : LifetimeConflict{[](const FrameSlotArenaRegion& lhs,
                                const FrameSlotArenaRegion& rhs) {
              return lifetimes_overlap(lhs.lifetime, rhs.lifetime);
            }};

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
                         return left.carrier_id < right.carrier_id;
                       });
      std::uint64_t local_cursor = 0U;
      for (const auto index : group.region_indices) {
        auto& region = result.regions_[index];
        const auto local = align_up(local_cursor, region.required_alignment_bytes);
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
      std::uint64_t block_offset = 0U;
      if (!place_group_first_fit(group.region_indices, &result.regions_,
                                 group.alignment, placed, lifetimes_conflict,
                                 &block_offset, error)) {
        return std::nullopt;
      }
      for (const auto index : group.region_indices) {
        auto& region = result.regions_[index];
        if (!checked_add(block_offset, region.byte_offset,
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

  std::unordered_map<CarrierId, std::size_t> carrier_to_region;
  for (std::size_t index = 0; index < result.regions_.size(); ++index) {
    const auto& region = result.regions_[index];
    carrier_to_region.emplace(region.carrier_id, index);
    result.allocation_alignment_bytes_ =
        std::max(result.allocation_alignment_bytes_,
                 region.required_alignment_bytes);
    if (region.byte_offset % region.required_alignment_bytes != 0U) {
      fail(error, "frame-slot arena produced an unaligned region");
      return std::nullopt;
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      const auto& other = result.regions_[previous];
      if (lifetimes_conflict(region, other) &&
          byte_ranges_overlap(region.byte_offset, region.size_bytes,
                              other.byte_offset, other.size_bytes)) {
        fail(error, "frame-slot arena reuses storage across overlapping lifetimes");
        return std::nullopt;
      }
    }
  }
  for (const auto& value : values) {
    if (!value.storage_binding.has_value()) {
      continue;
    }
    // Elision removes a semantic value's independent allocation, not its
    // address identity.  When an elided producer value and an exposed affine
    // view share a physically required carrier (for example A65 output ->
    // Reshape), both must resolve to that one carrier region.
    const auto found = carrier_to_region.find(value.storage_binding->carrier_id);
    if (found != carrier_to_region.end()) {
      result.value_to_region_[value.id] = found->second;
    }
  }
  if (physical_plan) {
    for (const auto& command : physical_plan->commands) {
      std::vector<CarrierId> referenced;
      referenced.reserve(command.inputs.size() + command.outputs.size());
      for (const auto value_id : command.inputs) {
        referenced.push_back(carrier_id(execution_plan, value_id));
      }
      for (const auto value_id : command.outputs) {
        referenced.push_back(carrier_id(execution_plan, value_id));
      }
      std::sort(referenced.begin(), referenced.end());
      referenced.erase(std::unique(referenced.begin(), referenced.end()), referenced.end());
      for (std::size_t left = 0; left < referenced.size(); ++left) {
        const auto left_region = carrier_to_region.find(referenced[left]);
        if (left_region == carrier_to_region.end()) {
          continue;
        }
        for (std::size_t right = left + 1U; right < referenced.size(); ++right) {
          const auto right_region = carrier_to_region.find(referenced[right]);
          if (right_region == carrier_to_region.end()) {
            continue;
          }
          const auto& lhs = result.regions_[left_region->second];
          const auto& rhs = result.regions_[right_region->second];
          if (byte_ranges_overlap(lhs.byte_offset, lhs.size_bytes,
                                  rhs.byte_offset, rhs.size_bytes)) {
            fail(error, "frame-slot physical command " +
                            std::to_string(command.id) +
                            " references overlapping carrier regions");
            return std::nullopt;
          }
        }
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

  std::uint32_t required_access = 0U;
  const auto add_access = [&](const ArenaDeviceAccess access) {
    required_access |= static_cast<std::uint32_t>(access);
  };
  if (physical_plan) {
    for (const auto& command : physical_plan->commands) {
      const bool touches_arena =
          std::any_of(command.inputs.begin(), command.inputs.end(), [&](const ValueId id) {
            return result.region(id) != nullptr;
          }) ||
          std::any_of(command.outputs.begin(), command.outputs.end(), [&](const ValueId id) {
            return result.region(id) != nullptr;
          });
      if (!touches_arena) {
        continue;
      }
      switch (command.engine) {
      case PhysicalEngine::Mla:
        add_access(ArenaDeviceAccess::Mla);
        break;
      case PhysicalEngine::A65:
        add_access(ArenaDeviceAccess::CpuA65);
        break;
      case PhysicalEngine::Cvu:
        // Physical CVU commands are placement-polymorphic until Core resolves
        // their requested target while rendering.  The arena is compiled
        // earlier, so conservatively require EV74 visibility.  This preserves
        // correctness and CMA placement; an A65-resolved CVU command may only
        // relax this bit in a future single-authority compile pass.
        add_access(ArenaDeviceAccess::Ev74);
        break;
      }
    }
  } else {
    for (const auto& op : execution_plan.ops()) {
      const bool touches_arena =
          std::any_of(op.inputs.begin(), op.inputs.end(), [&](const ValueId id) {
            return result.region(id) != nullptr;
          }) ||
          std::any_of(op.outputs.begin(), op.outputs.end(), [&](const ValueId id) {
            return result.region(id) != nullptr;
          });
      if (!touches_arena) {
        continue;
      }
      if (op.kind == OpKind::Mla) {
        add_access(ArenaDeviceAccess::Mla);
      } else if (op.kind == OpKind::HostTvm) {
        add_access(ArenaDeviceAccess::CpuA65);
      } else if (op.kind != OpKind::PassThrough && op.kind != OpKind::Slice &&
                 op.kind != OpKind::Reshape && op.kind != OpKind::Unpack) {
        add_access(ArenaDeviceAccess::Ev74);
      }
    }
  }

  ArenaEscapePolicy escape = ArenaEscapePolicy::InternalOnly;
  for (const auto& output : execution_plan.model_outputs()) {
    if (result.region(output.value_id) != nullptr) {
      escape = ArenaEscapePolicy::CpuMappablePublic;
      add_access(ArenaDeviceAccess::CpuA65);
    }
  }
  if (required_access == 0U) {
    fail(error, "frame-slot arena has no proved device or CPU consumer");
    return std::nullopt;
  }
  const bool ev74_required =
      (required_access & static_cast<std::uint32_t>(ArenaDeviceAccess::Ev74)) != 0U;
  result.placement_.domain =
      dms_policy == ArenaDmsPolicy::PreferDmsForEligible && !ev74_required
          ? ArenaStorageDomain::Dms
          : ArenaStorageDomain::Cma;
  result.placement_.provenance = ArenaAllocationProvenance::CoreAllocated;
  result.placement_.required_device_access = required_access;
  result.placement_.escape = escape;
  if (error) {
    error->clear();
  }
  return result;
}

std::optional<FrameSlotArenaPlan>
FrameSlotArenaPlan::compile(const ModelExecutionPlan& execution_plan,
                            const FrameSlotArenaReuse reuse,
                            const std::uint64_t default_region_alignment_bytes,
                            std::string* error, const ArenaDmsPolicy dms_policy) {
  return compile_impl(execution_plan, nullptr, reuse,
                      default_region_alignment_bytes, error, dms_policy, {});
}

std::optional<FrameSlotArenaPlan>
FrameSlotArenaPlan::compile(const ModelExecutionPlan& execution_plan,
                            const PhysicalExecutionPlan& physical_plan,
                            const FrameSlotArenaReuse reuse,
                            const std::uint64_t default_region_alignment_bytes,
                            std::string* error, const ArenaDmsPolicy dms_policy,
                            const std::span<const ValueId> detached_roots) {
  return compile_impl(execution_plan, &physical_plan, reuse,
                      default_region_alignment_bytes, error, dms_policy,
                      detached_roots);
}

const std::vector<FrameSlotArenaRegion>&
FrameSlotArenaPlan::regions() const noexcept {
  return regions_;
}

std::span<const ValueId> FrameSlotArenaPlan::detached_roots() const noexcept {
  return detached_roots_;
}

bool FrameSlotArenaPlan::is_detached_root(const ValueId value_id) const noexcept {
  return std::binary_search(detached_roots_.begin(), detached_roots_.end(),
                            value_id);
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

const ArenaPlacement& FrameSlotArenaPlan::placement() const noexcept {
  return placement_;
}

} // namespace simaai::neat::pipeline_internal::sima::static_contract

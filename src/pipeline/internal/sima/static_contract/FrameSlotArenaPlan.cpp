#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/static_contract/FrameSlotArenaPlan.h"
#include "pipeline/internal/sima/static_contract/PhysicalExecutionPlan.h"
#include "pipeline/internal/sima/static_contract/CarrierArenaPacking.h"

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

bool checked_add(const std::uint64_t lhs, const std::uint64_t rhs, std::uint64_t* result) {
  if (!result || lhs > std::numeric_limits<std::uint64_t>::max() - rhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

std::optional<std::uint64_t> align_up(const std::uint64_t value, const std::uint64_t alignment) {
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
  return lhs.first_sequence <= rhs.last_sequence && rhs.first_sequence <= lhs.last_sequence;
}

bool byte_ranges_overlap(const std::uint64_t lhs_offset, const std::uint64_t lhs_size,
                         const std::uint64_t rhs_offset, const std::uint64_t rhs_size) {
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
      if (left >= schedule.happens_before.size() || right >= schedule.happens_before.size() ||
          !schedule.happens_before[left][right]) {
        return false;
      }
    }
  }
  return true;
}

bool physical_lifetimes_overlap(const FrameSlotArenaRegion& lhs, const FrameSlotArenaRegion& rhs,
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
  return value && value->storage_binding.has_value() ? value->storage_binding->carrier_id
                                                     : static_cast<CarrierId>(value_id);
}

using LifetimeConflict = CarrierLifetimeConflict;

bool place_group_first_fit(const std::vector<std::size_t>& group_indices,
                           std::vector<CarrierArenaRegion>* regions,
                           const std::uint64_t group_alignment,
                           const std::vector<CarrierArenaRegion>& placed,
                           const LifetimeConflict& lifetimes_conflict, std::uint64_t* group_base,
                           std::string* error) {
  if (!regions || !group_base || group_indices.empty() || !is_power_of_two(group_alignment)) {
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
        if (member.storage_domain != other.storage_domain || !lifetimes_conflict(member, other) ||
            !byte_ranges_overlap(member_offset, member.size_bytes, other.byte_offset,
                                 other.size_bytes)) {
          continue;
        }
        std::uint64_t other_end = 0U;
        if (!checked_add(other.byte_offset, other.size_bytes, &other_end)) {
          return fail(error, "frame-slot arena contains an overflowing region");
        }
        // Move this member immediately after the conflicting range.  Because
        // member_offset = group_base + local_offset, subtracting its fixed
        // local offset finds the earliest possible next group base.
        const auto required_base = other_end > local_offset ? other_end - local_offset : 0U;
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

std::optional<CarrierArenaPacking>
pack_carrier_arena(std::vector<CarrierArenaRegion> regions, const FrameSlotArenaReuse reuse,
                   const CarrierLifetimeConflict& lifetimes_conflict,
                   const ArenaDmsPolicy dms_policy, std::string* error) {
  CarrierArenaPacking result;
  result.regions = std::move(regions);
  if (result.regions.empty()) {
    fail(error, "carrier arena requires materialized carriers");
    return std::nullopt;
  }
  for (const auto& region : result.regions) {
    if (!region.size_bytes || !is_power_of_two(region.required_alignment_bytes)) {
      fail(error, "carrier arena has an invalid extent or alignment");
      return std::nullopt;
    }
  }
  std::stable_sort(result.regions.begin(), result.regions.end(),
                   [](const auto& lhs, const auto& rhs) {
                     if (lhs.lifetime.first_sequence != rhs.lifetime.first_sequence) {
                       return lhs.lifetime.first_sequence < rhs.lifetime.first_sequence;
                     }
                     return lhs.carrier_id < rhs.carrier_id;
                   });

  if (reuse == FrameSlotArenaReuse::Disabled) {
    std::uint64_t cma_cursor = 0U;
    std::uint64_t dms_cursor = 0U;
    std::uint64_t host_cursor = 0U;
    for (auto& region : result.regions) {
      auto& cursor = region.storage_domain == ArenaStorageDomain::Cma       ? cma_cursor
                     : region.storage_domain == ArenaStorageDomain::HostCpu ? host_cursor
                                                                            : dms_cursor;
      const auto offset = align_up(cursor, region.required_alignment_bytes);
      if (!offset || !checked_add(*offset, region.size_bytes, &cursor)) {
        fail(error, "frame-slot no-reuse oracle overflows");
        return std::nullopt;
      }
      region.byte_offset = *offset;
    }
  } else {
    struct LifetimeGroup {
      ValueLifetime lifetime;
      ArenaStorageDomain domain;
      std::vector<std::size_t> region_indices;
      std::uint64_t used_bytes = 0U;
      std::uint64_t alignment = 0U;
    };
    std::vector<LifetimeGroup> groups;
    for (std::size_t index = 0; index < result.regions.size(); ++index) {
      const auto lifetime = result.regions[index].lifetime;
      auto group = std::find_if(groups.begin(), groups.end(), [&](const auto& item) {
        return item.domain == result.regions[index].storage_domain &&
               item.lifetime.first_sequence == lifetime.first_sequence &&
               item.lifetime.last_sequence == lifetime.last_sequence;
      });
      if (group == groups.end()) {
        groups.push_back({lifetime, result.regions[index].storage_domain, {}, 0U, 0U});
        group = std::prev(groups.end());
      }
      group->region_indices.push_back(index);
    }
    for (auto& group : groups) {
      std::stable_sort(group.region_indices.begin(), group.region_indices.end(),
                       [&](const auto lhs, const auto rhs) {
                         const auto& left = result.regions[lhs];
                         const auto& right = result.regions[rhs];
                         if (left.required_alignment_bytes != right.required_alignment_bytes) {
                           return left.required_alignment_bytes > right.required_alignment_bytes;
                         }
                         if (left.size_bytes != right.size_bytes) {
                           return left.size_bytes > right.size_bytes;
                         }
                         return left.carrier_id < right.carrier_id;
                       });
      std::uint64_t local_cursor = 0U;
      for (const auto index : group.region_indices) {
        auto& region = result.regions[index];
        const auto local = align_up(local_cursor, region.required_alignment_bytes);
        if (!local || !checked_add(*local, region.size_bytes, &local_cursor)) {
          fail(error, "frame-slot lifetime-group layout overflows");
          return std::nullopt;
        }
        region.byte_offset = *local;
        group.alignment = std::max(group.alignment, region.required_alignment_bytes);
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

    std::vector<CarrierArenaRegion> placed;
    placed.reserve(result.regions.size());
    for (const auto& group : groups) {
      std::uint64_t block_offset = 0U;
      if (!place_group_first_fit(group.region_indices, &result.regions, group.alignment, placed,
                                 lifetimes_conflict, &block_offset, error)) {
        return std::nullopt;
      }
      for (const auto index : group.region_indices) {
        auto& region = result.regions[index];
        if (!checked_add(block_offset, region.byte_offset, &region.byte_offset)) {
          fail(error, "frame-slot lifetime-group base overflows");
          return std::nullopt;
        }
        std::uint64_t end = 0U;
        if (!checked_add(region.byte_offset, region.size_bytes, &end)) {
          fail(error, "frame-slot arena region end overflows");
          return std::nullopt;
        }
        placed.push_back(region);
      }
    }
  }

  result.used_bytes = 0U;
  for (const auto& region : result.regions)
    result.allocation_alignment_bytes =
        std::max(result.allocation_alignment_bytes, region.required_alignment_bytes);
  for (const auto domain :
       {ArenaStorageDomain::Cma, ArenaStorageDomain::Dms, ArenaStorageDomain::HostCpu}) {
    std::uint64_t bytes = 0U;
    ArenaPlacement placement;
    placement.domain = domain;
    placement.provenance = ArenaAllocationProvenance::CoreAllocated;
    placement.allocation_policy =
        domain == ArenaStorageDomain::Dms && dms_policy == ArenaDmsPolicy::PreferDmsThenCma
            ? ArenaAllocationPolicy::PreferDmsThenCma
            : ArenaAllocationPolicy::RequiredDomain;
    for (const auto& region : result.regions) {
      if (region.storage_domain != domain)
        continue;
      std::uint64_t end = 0U;
      if (!checked_add(region.byte_offset, region.size_bytes, &end)) {
        fail(error, "frame-slot allocation extent overflows");
        return std::nullopt;
      }
      bytes = std::max(bytes, end);
      placement.required_device_access |= region.required_device_access;

      if (region.public_output)
        placement.escape = domain == ArenaStorageDomain::HostCpu
                               ? ArenaEscapePolicy::CpuOnlyPublic
                               : ArenaEscapePolicy::CpuMappablePublic;
    }
    if (!bytes)
      continue;
    const auto size = align_up(bytes, result.allocation_alignment_bytes);
    const auto base = result.allocation_bytes;
    if (!size || !checked_add(base, *size, &result.allocation_bytes)) {
      fail(error, "frame-slot allocation rounding overflows");
      return std::nullopt;
    }
    result.allocations.push_back({base, *size, placement});
    for (auto& region : result.regions) {
      if (region.storage_domain != domain)
        continue;
      if (!checked_add(base, region.byte_offset, &region.byte_offset)) {
        fail(error, "frame-slot allocation base overflows");
        return std::nullopt;
      }
    }
    result.used_bytes += bytes;
  }
  if (error)
    error->clear();
  return result;
}

std::optional<FrameSlotArenaPlan> FrameSlotArenaPlan::compile_impl(
    const ModelExecutionPlan& execution_plan, const PhysicalExecutionPlan* physical_plan,
    const FrameSlotArenaReuse reuse, const std::uint64_t default_region_alignment_bytes,
    std::string* error, const ArenaDmsPolicy dms_policy,
    const std::span<const ValueId> detached_roots, const FrameSlotArenaRoute* route) {
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
  std::unordered_set<CarrierId> physically_required_carriers;
  std::unordered_set<CarrierId> detached_carriers;
  std::unordered_set<ValueId> detached_values(detached_roots.begin(), detached_roots.end());
  std::unordered_set<ValueId> elided_values;
  std::optional<PhysicalCarrierSchedule> physical_schedule;

  const auto imported_inputs = route ? std::span<const ValueId>(route->imported_inputs)
                                     : std::span<const ValueId>(execution_plan.model_inputs());
  std::vector<ValueId> published_outputs;
  if (route) {
    if (!physical_plan || route->commands.empty()) {
      fail(error, "frame-slot route requires selected physical commands");
      return std::nullopt;
    }
    published_outputs = route->published_outputs;
  } else {
    for (const auto& output : execution_plan.model_outputs()) {
      published_outputs.push_back(output.value_id);
    }
  }
  std::vector<bool> selected(physical_plan ? physical_plan->commands.size() : 0U, !route);
  if (route) {
    for (const auto id : route->commands) {
      if (id >= selected.size() || selected[id]) {
        fail(error, "frame-slot route has invalid or duplicate command identity");
        return std::nullopt;
      }
      selected[id] = true;
    }
  }
  std::unordered_set<ValueId> imported_values;
  for (const auto value_id : imported_inputs) {
    if (!execution_plan.value(value_id) || !imported_values.emplace(value_id).second) {
      fail(error, "frame-slot route has an invalid or duplicate input value");
      return std::nullopt;
    }
    external_carriers.emplace(carrier_id(execution_plan, value_id));
  }
  for (const auto value_id : published_outputs) {
    if (!execution_plan.value(value_id)) {
      fail(error, "frame-slot route has an invalid output value");
      return std::nullopt;
    }
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
        std::max<std::uint64_t>(default_region_alignment_bytes, carrier.required_alignment_bytes);
  }

  if (physical_plan) {
    if (physical_plan->commands.size() >=
        static_cast<std::size_t>(std::numeric_limits<PhysicalCommandId>::max())) {
      fail(error, "frame-slot physical command count exceeds ABI capacity");
      return std::nullopt;
    }
    for (std::size_t index = 0; index < physical_plan->commands.size(); ++index) {
      if (physical_plan->commands[index].id != index ||
          physical_plan->commands[index].topological_rank != index) {
        fail(error, "frame-slot physical commands must have dense topological ids");
        return std::nullopt;
      }
    }
    PhysicalCarrierSchedule schedule;
    schedule.completion = static_cast<PhysicalCommandId>(physical_plan->commands.size());
    schedule.happens_before.assign(physical_plan->commands.size() + 1U,
                                   std::vector<bool>(physical_plan->commands.size() + 1U, false));

    // A member chain is one physical invocation.  Prove and record every
    // chain-internal edge before considering storage so it can neither acquire
    // an arena region nor accidentally reappear as an outer/backend/public
    // binding.
    for (const auto& command : physical_plan->commands) {
      if (!selected[command.id]) {
        continue;
      }
      for (const auto& member : command.members) {
        for (std::size_t chain_index = 1U; chain_index < member.semantic_chain.size();
             ++chain_index) {
          const auto first_id = member.semantic_chain[chain_index - 1U];
          const auto second_id = member.semantic_chain[chain_index];
          if (first_id >= ops.size() || second_id >= ops.size()) {
            fail(error, "frame-slot fused member has out-of-range semantic provenance");
            return std::nullopt;
          }
          const auto& first = ops[first_id];
          const auto& second = ops[second_id];
          std::vector<ValueId> internal_values;
          if (!resolve_exact_private_ordered_relation_path(execution_plan, first.id, second.id,
                                                           &internal_values) ||
              internal_values.empty()) {
            fail(error, "frame-slot fused member must have one exact private ordered-view path");
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
      if (!selected[command.id]) {
        continue;
      }
      for (const auto predecessor : command.predecessors) {
        if (predecessor >= command.id) {
          fail(error, "frame-slot physical command has a non-topological predecessor");
          return std::nullopt;
        }
        if (!selected[predecessor]) {
          continue;
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
        if (external_carriers.contains(id)) {
          fail(error, "frame-slot selected command writes an imported input carrier");
          return std::nullopt;
        }
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
      }
    }
    const auto public_output_sequence = static_cast<std::uint64_t>(schedule.completion) + 1U;
    for (const auto output : published_outputs) {
      if (elided_values.contains(output)) {
        fail(error, "frame-slot fused internal edge is a public output");
        return std::nullopt;
      }
      const auto id = carrier_id(execution_plan, output);
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
    const std::uint64_t public_output_sequence = ops.empty() ? 1U : ops.back().sequence + 1U;
    for (const auto& output : execution_plan.model_outputs()) {
      const auto id = carrier_id(execution_plan, output.value_id);
      last_sequence[id] = std::max(last_sequence[id], public_output_sequence);
    }
  }

  for (const auto& port : execution_plan.backend_ports()) {
    if (route &&
        !physically_required_carriers.contains(carrier_id(execution_plan, port.value_id))) {
      continue;
    }
    if (physical_plan && elided_values.contains(port.value_id)) {
      fail(error, "frame-slot fused internal edge is bound to a backend port");
      return std::nullopt;
    }
    const auto id = carrier_id(execution_plan, port.value_id);
    if (physical_plan) {
      physically_required_carriers.emplace(id);
    }
    alignment[id] = std::max<std::uint64_t>(alignment[id], port.required_alignment_bytes);
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
          std::max(inherited_alignment, alignment[carrier_id(execution_plan, input_id)]);
    }
    for (const auto output_id : op.outputs) {
      const auto id = carrier_id(execution_plan, output_id);
      alignment[id] = std::max(alignment[id], inherited_alignment);
    }
  }

  if (dms_policy != ArenaDmsPolicy::CmaOnly &&
      dms_policy != ArenaDmsPolicy::RequireDmsForEligible &&
      dms_policy != ArenaDmsPolicy::PreferDmsThenCma) {
    fail(error, "frame-slot arena has unknown allocation policy");
    return std::nullopt;
  }
  std::unordered_map<CarrierId, std::uint32_t> carrier_access;
  const auto add_access = [&](ValueId value, ArenaDeviceAccess access) {
    carrier_access[carrier_id(execution_plan, value)] |= static_cast<std::uint32_t>(access);
  };
  if (physical_plan) {
    for (const auto& command : physical_plan->commands) {
      if (!selected[command.id])
        continue;
      const auto engine = command.resolved_executor.value_or(command.engine);
      const auto access = engine == PhysicalEngine::Mla   ? ArenaDeviceAccess::Mla
                          : engine == PhysicalEngine::A65 ? ArenaDeviceAccess::CpuA65
                                                          : ArenaDeviceAccess::Ev74;
      for (const auto value : command.inputs)
        add_access(value, access);
      for (const auto value : command.outputs)
        add_access(value, access);
    }
  } else {
    for (const auto& op : ops) {
      if (is_address_relation_op(op))
        continue;
      const auto access = op.kind == OpKind::Mla       ? ArenaDeviceAccess::Mla
                          : op.kind == OpKind::HostTvm ? ArenaDeviceAccess::CpuA65
                                                       : ArenaDeviceAccess::Ev74;
      for (const auto value : op.inputs)
        add_access(value, access);
      for (const auto value : op.outputs)
        add_access(value, access);
    }
  }
  for (const auto value : published_outputs) {
    add_access(value, ArenaDeviceAccess::CpuA65);
    if (route) {
      const auto id = carrier_id(execution_plan, value);
      if (const auto access = route->external_device_access.find(id);
          access != route->external_device_access.end())
        carrier_access[id] |= access->second;
    }
  }

  FrameSlotArenaPlan result;
  result.reuse_policy_ = reuse;
  result.dms_policy_ = dms_policy;
  result.imported_inputs_.assign(imported_inputs.begin(), imported_inputs.end());
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
    const auto first_value =
        std::find_if(values.begin(), values.end(), [&](const ValueSpec& value) {
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
    region.lifetime.last_sequence = std::max(producer->second, last_sequence[carrier.id]);
    const auto access = carrier_access[carrier.id];
    if (access == 0U) {
      fail(error, "frame-slot carrier has no proved device or CPU consumer");
      return std::nullopt;
    }
    const bool ev74 = (access & static_cast<std::uint32_t>(ArenaDeviceAccess::Ev74)) != 0U;
    region.storage_domain = dms_policy == ArenaDmsPolicy::CmaOnly || ev74 ? ArenaStorageDomain::Cma
                                                                          : ArenaStorageDomain::Dms;
    result.regions_.push_back(std::move(region));
  }
  if (result.regions_.empty()) {
    fail(error, "frame-slot arena has no materialized internal values");
    return std::nullopt;
  }

  const auto lifetimes_conflict = [&](const auto& lhs, const auto& rhs) {
    if (physical_schedule) {
      FrameSlotArenaRegion left, right;
      left.carrier_id = static_cast<CarrierId>(lhs.carrier_id);
      right.carrier_id = static_cast<CarrierId>(rhs.carrier_id);
      return physical_lifetimes_overlap(left, right, *physical_schedule);
    }
    return lifetimes_overlap(lhs.lifetime, rhs.lifetime);
  };
  std::vector<CarrierArenaRegion> carriers;
  carriers.reserve(result.regions_.size());
  for (const auto& region : result.regions_) {
    const bool published =
        std::any_of(published_outputs.begin(), published_outputs.end(), [&](ValueId value) {
          return carrier_id(execution_plan, value) == region.carrier_id;
        });
    carriers.push_back({region.carrier_id, 0, region.size_bytes, region.required_alignment_bytes,
                        region.lifetime, region.storage_domain, carrier_access[region.carrier_id],
                        published});
  }
  auto packed =
      pack_carrier_arena(std::move(carriers), reuse, lifetimes_conflict, dms_policy, error);
  if (!packed)
    return std::nullopt;
  for (const auto& region : packed->regions) {
    const auto target =
        std::find_if(result.regions_.begin(), result.regions_.end(), [&](const auto& candidate) {
          return candidate.carrier_id == region.carrier_id;
        });
    target->byte_offset = region.byte_offset;
  }
  std::stable_sort(result.regions_.begin(), result.regions_.end(),
                   [](const auto& lhs, const auto& rhs) {
                     if (lhs.lifetime.first_sequence != rhs.lifetime.first_sequence)
                       return lhs.lifetime.first_sequence < rhs.lifetime.first_sequence;
                     return lhs.carrier_id < rhs.carrier_id;
                   });
  result.allocations_ = std::move(packed->allocations);
  result.used_bytes_ = packed->used_bytes;
  result.allocation_bytes_ = packed->allocation_bytes;
  result.allocation_alignment_bytes_ = packed->allocation_alignment_bytes;
  result.placement_.domain = result.allocations_.size() == 1
                                 ? result.allocations_.front().placement.domain
                                 : ArenaStorageDomain::Mixed;
  result.placement_.provenance = ArenaAllocationProvenance::CoreAllocated;

  for (const auto& part : result.allocations_) {
    result.placement_.required_device_access |= part.placement.required_device_access;
    if (part.placement.escape == ArenaEscapePolicy::CpuMappablePublic)
      result.placement_.escape = ArenaEscapePolicy::CpuMappablePublic;
  }

  for (std::size_t mla_index = 0; mla_index < execution_plan.mla_stage_count(); ++mla_index) {
    const auto outputs = execution_plan.backend_ports(mla_index, BackendPortDirection::Output);
    if (outputs.empty() || !std::all_of(outputs.begin(), outputs.end(), [&](const auto& port) {
          return detached_carriers.contains(carrier_id(execution_plan, port.value_id));
        }))
      continue;
    std::vector<CarrierArenaRegion> detached_regions;
    for (const auto& port : outputs) {
      const auto id = carrier_id(execution_plan, port.value_id);
      if (std::any_of(detached_regions.begin(), detached_regions.end(),
                      [&](const auto& region) { return region.carrier_id == id; }))
        continue;
      const auto& carrier = *execution_plan.carrier(id);
      const auto access = carrier_access[id];
      const auto domain = (access & static_cast<std::uint32_t>(ArenaDeviceAccess::Ev74))
                              ? ArenaStorageDomain::Cma
                              : ArenaStorageDomain::Dms;
      detached_regions.push_back(
          {id, 0, carrier.required_bytes, alignment[id], {}, domain, access, true});
    }
    auto detached = pack_carrier_arena(
        std::move(detached_regions), FrameSlotArenaReuse::Disabled,
        [](const auto&, const auto&) { return true; }, ArenaDmsPolicy::RequireDmsForEligible,
        error);
    if (!detached)
      return std::nullopt;
    FrameSlotArenaOutputGroup group;
    group.allocation_bytes = detached->allocation_bytes;
    group.allocations = std::move(detached->allocations);
    group.placement.domain = group.allocations.size() == 1
                                 ? group.allocations.front().placement.domain
                                 : ArenaStorageDomain::Mixed;
    group.placement.provenance = ArenaAllocationProvenance::CoreAllocated;
    group.placement.escape = ArenaEscapePolicy::CpuMappablePublic;
    for (const auto& allocation : group.allocations)
      group.placement.required_device_access |= allocation.placement.required_device_access;
    for (const auto& packed_region : detached->regions) {
      FrameSlotArenaRegion region;
      region.carrier_id = static_cast<CarrierId>(packed_region.carrier_id);
      region.byte_offset = packed_region.byte_offset;
      region.size_bytes = packed_region.size_bytes;
      region.required_alignment_bytes = packed_region.required_alignment_bytes;
      region.storage_domain = packed_region.storage_domain;
      group.regions.push_back(region);
    }
    result.detached_output_groups_.emplace(mla_index, std::move(group));
  }

  std::unordered_map<CarrierId, std::size_t> carrier_to_region;
  for (std::size_t index = 0; index < result.regions_.size(); ++index) {
    const auto& region = result.regions_[index];
    carrier_to_region.emplace(region.carrier_id, index);
    if (region.byte_offset % region.required_alignment_bytes != 0U) {
      fail(error, "frame-slot arena produced an unaligned region");
      return std::nullopt;
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      const auto& other = result.regions_[previous];
      if (lifetimes_conflict(region, other) &&
          byte_ranges_overlap(region.byte_offset, region.size_bytes, other.byte_offset,
                              other.size_bytes)) {
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
      if (!selected[command.id]) {
        continue;
      }
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
          if (byte_ranges_overlap(lhs.byte_offset, lhs.size_bytes, rhs.byte_offset,
                                  rhs.size_bytes)) {
            fail(error, "frame-slot physical command " + std::to_string(command.id) +
                            " references overlapping carrier regions");
            return std::nullopt;
          }
        }
      }
    }
  }
  if (error) {
    error->clear();
  }
  return result;
}

std::optional<FrameSlotArenaPlan>
FrameSlotArenaPlan::compile(const ModelExecutionPlan& execution_plan,
                            const FrameSlotArenaReuse reuse,
                            const std::uint64_t default_region_alignment_bytes, std::string* error,
                            const ArenaDmsPolicy dms_policy) {
  return compile_impl(execution_plan, nullptr, reuse, default_region_alignment_bytes, error,
                      dms_policy, {});
}

std::optional<FrameSlotArenaPlan> FrameSlotArenaPlan::compile(
    const ModelExecutionPlan& execution_plan, const PhysicalExecutionPlan& physical_plan,
    const FrameSlotArenaReuse reuse, const std::uint64_t default_region_alignment_bytes,
    std::string* error, const ArenaDmsPolicy dms_policy,
    const std::span<const ValueId> detached_roots, const FrameSlotArenaRoute* route) {
  return compile_impl(execution_plan, &physical_plan, reuse, default_region_alignment_bytes, error,
                      dms_policy, detached_roots, route);
}

const std::vector<FrameSlotArenaRegion>& FrameSlotArenaPlan::regions() const noexcept {
  return regions_;
}

std::span<const ValueId> FrameSlotArenaPlan::imported_inputs() const noexcept {
  return imported_inputs_;
}

std::span<const ValueId> FrameSlotArenaPlan::detached_roots() const noexcept {
  return detached_roots_;
}

const FrameSlotArenaOutputGroup*
FrameSlotArenaPlan::detached_output_group(const std::size_t mla_stage_index) const noexcept {
  const auto group = detached_output_groups_.find(mla_stage_index);
  return group == detached_output_groups_.end() ? nullptr : &group->second;
}

bool FrameSlotArenaPlan::is_detached_root(const ValueId value_id) const noexcept {
  return std::binary_search(detached_roots_.begin(), detached_roots_.end(), value_id);
}

const FrameSlotArenaRegion* FrameSlotArenaPlan::region(const ValueId value_id) const noexcept {
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

ArenaDmsPolicy FrameSlotArenaPlan::dms_policy() const noexcept {
  return dms_policy_;
}

const ArenaLayoutSummary& FrameSlotArenaPlan::placement() const noexcept {
  return placement_;
}

const std::vector<FrameSlotArenaAllocation>& FrameSlotArenaPlan::allocations() const noexcept {
  return allocations_;
}

} // namespace simaai::neat::pipeline_internal::sima::static_contract

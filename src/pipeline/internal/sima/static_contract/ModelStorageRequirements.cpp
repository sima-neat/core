#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/static_contract/ModelStorageRequirements.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace simaai::neat::pipeline_internal::sima::static_contract {
namespace {
namespace client = ::simaai::scheduler::client;

bool fail(std::string* error, std::string message) {
  if (error)
    *error = std::move(message);
  return false;
}

CarrierId carrier_id(const ModelExecutionPlan& plan, ValueId id) {
  const auto* value = plan.value(id);
  return value && value->storage_binding ? value->storage_binding->carrier_id
                                         : static_cast<CarrierId>(id);
}

bool all_before(const std::vector<PhysicalCommandId>& lhs,
                const std::vector<PhysicalCommandId>& rhs,
                const std::vector<std::vector<bool>>& before) noexcept {
  if (lhs.empty() || rhs.empty())
    return false;
  for (const auto left : lhs)
    for (const auto right : rhs)
      if (left >= before.size() || right >= before.size() || !before[left][right])
        return false;
  return true;
}
} // namespace

bool ModelStorageRequirements::lifetimes_conflict(const CarrierRequirement& lhs,
                                                  const CarrierRequirement& rhs) const noexcept {
  if (lhs.carrier_id > std::numeric_limits<CarrierId>::max() ||
      rhs.carrier_id > std::numeric_limits<CarrierId>::max())
    return true;
  const auto left = accesses.find(static_cast<CarrierId>(lhs.carrier_id));
  const auto right = accesses.find(static_cast<CarrierId>(rhs.carrier_id));
  return left == accesses.end() || right == accesses.end() ||
         (!all_before(left->second, right->second, happens_before) &&
          !all_before(right->second, left->second, happens_before));
}

std::optional<ModelStorageRequirements> collect_model_storage_requirements(
    const ModelExecutionPlan& execution, const PhysicalExecutionPlan& physical,
    std::uint64_t default_alignment, std::string* error,
    client::AllocationPolicy eligible_dms_policy, std::span<const ValueId> detached_roots,
    const ModelStorageRoute* route) {
  if (!default_alignment || (default_alignment & (default_alignment - 1U)) ||
      (eligible_dms_policy != client::AllocationPolicy::RequireDms &&
       eligible_dms_policy != client::AllocationPolicy::PreferDmsThenCma)) {
    fail(error, "model storage requires a valid alignment and DMS policy");
    return std::nullopt;
  }
  if (physical.commands.size() >= std::numeric_limits<PhysicalCommandId>::max()) {
    fail(error, "model storage physical command count exceeds identity capacity");
    return std::nullopt;
  }
  const auto& values = execution.values();
  const auto& ops = execution.ops();
  std::vector<bool> selected(physical.commands.size(), !route);
  if (route) {
    if (route->commands.empty()) {
      fail(error, "model storage route requires selected physical commands");
      return std::nullopt;
    }
    for (const auto id : route->commands) {
      if (id >= selected.size() || selected[id]) {
        fail(error, "model storage route has an invalid or duplicate command identity");
        return std::nullopt;
      }
      selected[id] = true;
    }
  }
  ModelStorageRequirements result;
  const auto imported = route ? std::span<const ValueId>(route->imported_inputs)
                              : std::span<const ValueId>(execution.model_inputs());
  result.imported_inputs.assign(imported.begin(), imported.end());
  result.value_carriers.resize(values.size());
  for (const auto& value : values)
    if (value.storage_binding)
      result.value_carriers[value.id] = value.storage_binding->carrier_id;

  std::unordered_set<ValueId> imported_values;
  std::unordered_set<CarrierId> external;
  for (const auto id : imported) {
    if (!execution.value(id) || !imported_values.insert(id).second) {
      fail(error, "model storage route has an invalid or duplicate input value");
      return std::nullopt;
    }
    external.insert(carrier_id(execution, id));
  }
  std::vector<ValueId> published;
  if (route)
    published = route->published_outputs;
  else
    for (const auto& output : execution.model_outputs())
      published.push_back(output.value_id);
  for (const auto id : published)
    if (!execution.value(id)) {
      fail(error, "model storage route has an invalid output value");
      return std::nullopt;
    }

  std::unordered_set<ValueId> detached_values(detached_roots.begin(), detached_roots.end());
  std::unordered_set<CarrierId> detached;
  for (const auto id : detached_roots) {
    const auto* value = execution.value(id);
    if (!value || value->read_expression || external.contains(carrier_id(execution, id))) {
      fail(error, "model storage detached output root is invalid");
      return std::nullopt;
    }
    detached.insert(carrier_id(execution, id));
  }
  for (const auto& value : values)
    if (value.storage_binding && !value.read_expression &&
        detached.contains(value.storage_binding->carrier_id) &&
        !detached_values.contains(value.id)) {
      fail(error, "model storage detached carrier also owns a retained materialized root");
      return std::nullopt;
    }
  result.detached_roots.assign(detached_values.begin(), detached_values.end());
  std::sort(result.detached_roots.begin(), result.detached_roots.end());

  std::unordered_map<CarrierId, std::uint64_t> alignment, first, last;
  for (const auto& carrier : execution.carriers())
    alignment[carrier.id] =
        std::max<std::uint64_t>(default_alignment, carrier.required_alignment_bytes);
  const auto completion = static_cast<PhysicalCommandId>(physical.commands.size());
  result.happens_before.assign(physical.commands.size() + 1U,
                               std::vector<bool>(physical.commands.size() + 1U, false));
  std::unordered_set<ValueId> elided;
  for (std::size_t index = 0; index < physical.commands.size(); ++index) {
    const auto& command = physical.commands[index];
    if (command.id != index || command.topological_rank != index) {
      fail(error, "model storage commands must have dense topological identities");
      return std::nullopt;
    }
    if (!selected[command.id])
      continue;
    for (const auto& member : command.members)
      for (std::size_t chain = 1; chain < member.semantic_chain.size(); ++chain) {
        const auto first_op = member.semantic_chain[chain - 1U];
        const auto second_op = member.semantic_chain[chain];
        std::vector<ValueId> internal;
        if (first_op >= ops.size() || second_op >= ops.size() ||
            !resolve_exact_private_ordered_relation_path(execution, first_op, second_op,
                                                         &internal) ||
            internal.empty()) {
          fail(error, "model storage fused member requires an exact private ordered-view path");
          return std::nullopt;
        }
        elided.insert(internal.begin(), internal.end());
      }
  }

  std::unordered_set<CarrierId> required;
  std::unordered_map<CarrierId, std::uint32_t> access;
  for (const auto& command : physical.commands) {
    if (!selected[command.id])
      continue;
    for (const auto predecessor : command.predecessors) {
      if (predecessor >= command.id) {
        fail(error, "model storage command has a non-topological predecessor");
        return std::nullopt;
      }
      if (!selected[predecessor])
        continue;
      result.happens_before[predecessor][command.id] = true;
      for (std::size_t ancestor = 0; ancestor < command.id; ++ancestor)
        if (result.happens_before[ancestor][predecessor])
          result.happens_before[ancestor][command.id] = true;
    }
    result.happens_before[command.id][completion] = true;
    const auto engine = command.resolved_executor.value_or(command.engine);
    const auto device = engine == PhysicalEngine::Mla   ? client::kAccessMla
                        : engine == PhysicalEngine::A65 ? client::kAccessCpuA65
                                                        : client::kAccessEv74;
    const auto record_access = [&](ValueId value) {
      const auto id = carrier_id(execution, value);
      required.insert(id);
      result.accesses[id].push_back(command.id);
      last[id] = std::max(last[id], static_cast<std::uint64_t>(command.id) + 1U);
      access[id] |= device;
    };
    for (const auto id : command.inputs) {
      if (id >= values.size() || elided.contains(id)) {
        fail(error, "model storage command input is invalid or a fused internal value");
        return std::nullopt;
      }
      record_access(id);
    }
    for (const auto id : command.outputs) {
      if (id >= values.size() || elided.contains(id)) {
        fail(error, "model storage command output is invalid or a fused internal value");
        return std::nullopt;
      }
      const auto root = carrier_id(execution, id);
      if (external.contains(root)) {
        fail(error, "model storage command writes an imported input carrier");
        return std::nullopt;
      }
      record_access(id);
      if (values[id].storage_binding) {
        const auto sequence = static_cast<std::uint64_t>(command.id) + 1U;
        const auto [iterator, inserted] = first.emplace(root, sequence);
        if (!inserted)
          iterator->second = std::min(iterator->second, sequence);
      }
    }
  }
  for (const auto output : published) {
    if (elided.contains(output)) {
      fail(error, "model storage fused internal value is publicly exposed");
      return std::nullopt;
    }
    const auto id = carrier_id(execution, output);
    required.insert(id);
    result.accesses[id].push_back(completion);
    last[id] = std::max(last[id], static_cast<std::uint64_t>(completion) + 1U);
    access[id] |= client::kAccessCpuA65;
    if (route)
      if (const auto consumer = route->external_device_access.find(id);
          consumer != route->external_device_access.end())
        access[id] |= consumer->second;
  }
  for (auto& [id, accesses] : result.accesses) {
    (void)id;
    std::sort(accesses.begin(), accesses.end());
    accesses.erase(std::unique(accesses.begin(), accesses.end()), accesses.end());
  }
  for (const auto& port : execution.backend_ports()) {
    const auto id = carrier_id(execution, port.value_id);
    if (route && !required.contains(id))
      continue;
    if (elided.contains(port.value_id)) {
      fail(error, "model storage fused internal value is bound to a backend port");
      return std::nullopt;
    }
    required.insert(id);
    alignment[id] = std::max<std::uint64_t>(alignment[id], port.required_alignment_bytes);
  }
  // Materializing transforms retain the strongest inherited port constraint.
  for (const auto& op : ops) {
    std::uint64_t inherited = default_alignment;
    for (const auto id : op.inputs)
      inherited = std::max(inherited, alignment[carrier_id(execution, id)]);
    for (const auto id : op.outputs) {
      const auto root = carrier_id(execution, id);
      alignment[root] = std::max(alignment[root], inherited);
    }
  }

  const auto requirement = [&](const CarrierSpec& carrier, client::AllocationPolicy dms_policy) {
    const auto mask = access[carrier.id];
    return CarrierRequirement{carrier.id,
                              carrier.required_bytes,
                              alignment[carrier.id],
                              {first[carrier.id], std::max(first[carrier.id], last[carrier.id])},
                              *resolve_allocation_policy(mask, dms_policy),
                              mask};
  };
  for (const auto& carrier : execution.carriers()) {
    if (external.contains(carrier.id) || detached.contains(carrier.id) ||
        !required.contains(carrier.id))
      continue;
    const auto binding = std::find_if(values.begin(), values.end(), [&](const auto& value) {
      return value.storage_binding && value.storage_binding->carrier_id == carrier.id &&
             value.storage_binding->kind != StorageBindingKind::View;
    });
    if (binding == values.end() || !first.contains(carrier.id) || !first[carrier.id] ||
        !result.accesses.contains(carrier.id) || !access[carrier.id]) {
      fail(error, "model storage carrier has no materialized producer or physical access proof");
      return std::nullopt;
    }
    result.carriers.push_back(requirement(carrier, eligible_dms_policy));
  }
  for (std::size_t mla = 0; mla < execution.mla_stage_count(); ++mla) {
    const auto outputs = execution.backend_ports(mla, BackendPortDirection::Output);
    if (outputs.empty() || !std::all_of(outputs.begin(), outputs.end(), [&](const auto& port) {
          const auto id = carrier_id(execution, port.value_id);
          return detached.contains(id) && required.contains(id) && first.contains(id) && access[id];
        }))
      continue;
    auto& group = result.detached_outputs[mla];
    for (const auto& port : outputs) {
      const auto id = carrier_id(execution, port.value_id);
      if (std::none_of(group.begin(), group.end(),
                       [&](const auto& item) { return item.carrier_id == id; }))
        group.push_back(requirement(*execution.carrier(id), client::AllocationPolicy::RequireDms));
    }
  }
  if (error)
    error->clear();
  return result;
}

} // namespace simaai::neat::pipeline_internal::sima::static_contract

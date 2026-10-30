#include "pipeline/internal/sima/StageMemoryPolicy.h"
#include "model/internal/ModelPack.h"
#include "pipeline/internal/TerminalOutputContractQuery.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/sima/static_contract/CarrierPacking.h"
#include "pipeline/internal/sima/static_contract/ModelStorageRequirements.h"
#include <simaai/scheduler/client/frame_layout.h>
#include "pipeline/internal/sima/InternalEdgeContractResolver.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/ProcessCvuRunTargetPolicy.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemanticsInternal.h"

#include <algorithm>
#include <stdexcept>
#include <limits>
#include <unordered_map>

namespace simaai::neat::pipeline_internal::sima {
namespace {
namespace sc = static_contract;
struct StageConsumers {
  bool shared_allocation = false;
  bool has_consumers = false;
};

} // namespace

std::uint32_t resolved_stage_device_access(const StageStaticSpec& stage) {
  if (stage.native_kernel) {
    switch (stage.native_kernel->target) {
    case ::simaai::scheduler::client::Hardware::EV74:
      return ::simaai::scheduler::client::kAccessEv74;
    case ::simaai::scheduler::client::Hardware::A65:
      return ::simaai::scheduler::client::kAccessCpuA65;
    default:
      throw std::invalid_argument("memory placement requires a resolved kernel executor");
    }
  }
  if (stage.tvm_program)
    return ::simaai::scheduler::client::kAccessCpuA65;
  if (stage.payload_kind == StagePayloadKind::ProcessMla)
    return ::simaai::scheduler::client::kAccessMla;
  if (stage.payload_kind == StagePayloadKind::ProcessCvu) {
    if (stage.processcvu.resolved_exec_backend == "A65")
      return ::simaai::scheduler::client::kAccessCpuA65;
    if (stage.processcvu.resolved_exec_backend == "EV74" ||
        stage.processcvu.resolved_exec_backend == "EVXX")
      return ::simaai::scheduler::client::kAccessEv74;
    throw std::invalid_argument("memory placement requires a resolved CVU executor");
  }
  // A transport does not execute numerics; its consumers contribute access.
  return ::simaai::scheduler::client::kAccessCpuA65;
}

void resolve_physical_execution_targets(static_contract::PhysicalExecutionPlan& plan,
                                        const ContractCompileInput& input) {
  for (auto& command : plan.commands) {
    if (command.engine != static_contract::PhysicalEngine::Cvu) {
      command.resolved_executor = command.engine;
      continue;
    }
    ProcessCvuStagePayload payload;
    payload.graph_name = command.implementation_id;
    payload.graph_family =
        stagesemantics::canonical_family_name_internal(command.implementation_id);
    payload.graph_family_enum =
        stagesemantics::family_enum_from_name_internal(payload.graph_family);
    const auto decision = resolve_processcvu_backend_decision(
        payload, input, command.implementation_id, command.role);
    command.resolved_executor = decision.resolved_exec_backend == ProcessCvuResolvedExecBackend::A65
                                    ? static_contract::PhysicalEngine::A65
                                    : static_contract::PhysicalEngine::Cvu;
  }
}

void resolve_compiled_execution_targets(CompiledNodeContract& stage,
                                        const ContractCompileInput& input) {
  for (auto& child : stage.child_stages)
    resolve_compiled_execution_targets(child, input);
  if (stage.kernel && stage.kernel->recipe) {
    stage.resolved_executor =
        stage.kernel->recipe->target == ::simaai::scheduler::client::Hardware::EV74
            ? sc::PhysicalEngine::Cvu
            : sc::PhysicalEngine::A65;
    return;
  }
  if (stage.processmla) {
    stage.resolved_executor = sc::PhysicalEngine::Mla;
    return;
  }
  if (stage.transport && stage.transport->tvm_program) {
    stage.resolved_executor = sc::PhysicalEngine::A65;
    return;
  }
  auto* payload = stage.processcvu ? &stage.processcvu->payload
                  : stage.transport && stage.transport->processcvu_payload
                      ? &*stage.transport->processcvu_payload
                      : nullptr;
  if (payload) {
    resolve_processcvu_run_target(payload, input, stage.logical_stage_id,
                                  stage.processcvu ? stage.processcvu->physical_command_role
                                                   : std::nullopt);
    stage.resolved_executor =
        payload->resolved_exec_backend == "A65" ? sc::PhysicalEngine::A65 : sc::PhysicalEngine::Cvu;
  }
}

namespace {
namespace client = ::simaai::scheduler::client;

const StageStaticSpec*
public_output_candidate(const SimaPluginStaticManifest& manifest,
                        const terminal_output_contract::PublicOutputEndpointSelector* endpoint,
                        std::span<const StageConsumers> access) {
  namespace query = terminal_output_contract;
  if (!endpoint)
    return nullptr;
  const StageStaticSpec* candidate = nullptr;
  if (endpoint->terminal_stage_key_required && !endpoint->terminal_stage_key.empty()) {
    for (const auto& stage : manifest.stages)
      if (stage.element_name == endpoint->terminal_stage_key ||
          stage.logical_stage_id == endpoint->terminal_stage_key) {
        if (candidate)
          return nullptr;
        candidate = &stage;
      }
  } else if (endpoint->terminal_node_kind == "ModelFragment" &&
             endpoint->allow_unresolved_terminal_stage_fallback) {
    const StageStaticSpec* named_stage = nullptr;
    for (std::size_t i = 0; i < manifest.stages.size(); ++i) {
      const auto& stage = manifest.stages[i];
      if (!endpoint->terminal_stage_key.empty() &&
          (stage.element_name == endpoint->terminal_stage_key ||
           stage.logical_stage_id == endpoint->terminal_stage_key))
        named_stage = &stage;
      const auto role = query::classify_stage_for_publication(stage);
      if (!access[i].has_consumers &&
          (role == query::StagePublicationRole::RealProducer ||
           role == query::StagePublicationRole::MaterializedTransform)) {
        if (candidate)
          return nullptr;
        candidate = &stage;
      }
    }
    if (!candidate || (named_stage && named_stage != candidate) ||
        !query::stage_public_outputs_match_endpoint(*candidate, *endpoint))
      return nullptr;
  }
  return candidate;
}

bool dense_publication(const LogicalTensorStaticSpec& logical) {
  std::uint32_t dtype = 0;
  if (!tensorsemantics::dtype_token_to_ev(logical.dtype, &dtype) || logical.byte_offset < 0 ||
      (!logical.stride_bytes.empty() && logical.stride_bytes.size() != logical.shape.size()))
    return false;
  std::uint64_t bytes = sima_ev_elem_size_bytes(dtype);
  if (!bytes)
    return false;
  for (std::size_t i = logical.shape.size(); i-- > 0;) {
    const auto extent = logical.shape[i];
    if (extent <= 0 ||
        (!logical.stride_bytes.empty() &&
         (logical.stride_bytes[i] < 0 ||
          (extent > 1 && static_cast<std::uint64_t>(logical.stride_bytes[i]) != bytes))) ||
        static_cast<std::uint64_t>(extent) > std::numeric_limits<std::uint64_t>::max() / bytes)
      return false;
    bytes *= static_cast<std::uint64_t>(extent);
  }
  return logical.size_bytes >= bytes;
}

struct KernelStorageAccess {
  std::unordered_map<int, std::vector<std::size_t>> accesses;
  std::unordered_map<int, std::uint32_t> device_access;
  std::vector<std::vector<bool>> before;
  bool all_members_cpu = true;
};

int physical_output_id(const StageStaticSpec& stage, std::size_t tensor) {
  const auto logical = std::find_if(
      stage.logical_outputs.begin(), stage.logical_outputs.end(),
      [&](const auto& value) { return value.backend_output_index == static_cast<int>(tensor); });
  return logical == stage.logical_outputs.end() ? static_cast<int>(tensor)
                                                : logical->physical_index;
}

KernelStorageAccess inspect_stage_storage(StageStaticSpec& stage) {
  KernelStorageAccess result;
  if (!stage.native_kernel)
    return result;
  const auto& recipe = *stage.native_kernel;
  const auto count = recipe.members.size();
  result.before.assign(count + 1, std::vector<bool>(count + 1));
  for (std::size_t index = 0; index < count; ++index) {
    const auto& member = recipe.members[index];
    for (const auto dependency : member.dependencies) {
      if (dependency >= index)
        throw std::invalid_argument("memory placement: non-topological numerical dependency");
      result.before[dependency][index] = true;
      for (std::size_t ancestor = 0; ancestor < dependency; ++ancestor)
        if (result.before[ancestor][dependency])
          result.before[ancestor][index] = true;
    }
    result.before[index][count] = true;
    const auto inspected = client::inspect_kernel(member.executable);
    if (const auto* error = std::get_if<client::Error>(&inspected))
      throw std::invalid_argument("memory placement: " + error->message);
    const auto& checked = std::get<client::KernelInspection>(inspected);
    result.all_members_cpu &= checked.target() == client::Hardware::A65;
    const auto access =
        checked.target() == client::Hardware::EV74 ? client::kAccessEv74 : client::kAccessCpuA65;
    for (std::size_t port = 0; port < member.outputs.size(); ++port) {
      const auto& output = member.outputs[port];
      const auto id = physical_output_id(stage, output.tensor);
      const auto physical =
          std::find_if(stage.physical_outputs.begin(), stage.physical_outputs.end(),
                       [&](const auto& value) { return value.physical_index == id; });
      if (physical == stage.physical_outputs.end() || port >= checked.output_requirements().size())
        throw std::invalid_argument("memory placement: numerical output has no carrier");
      const auto& requirement = checked.output_requirements()[port];
      if (output.byte_offset > physical->size_bytes ||
          requirement.storage_bytes > physical->size_bytes - output.byte_offset)
        throw std::invalid_argument("memory placement: checked output exceeds its carrier");
      physical->required_alignment_bytes =
          std::max<std::uint64_t>(physical->required_alignment_bytes, requirement.alignment_bytes);
      result.accesses[id].push_back(index);
      result.device_access[id] |= access;
    }
    for (std::size_t port = 0; port < member.inputs.size(); ++port) {
      const auto& input = member.inputs[port];
      if (port >= checked.input_requirements().size())
        throw std::invalid_argument("memory placement: numerical input has no checked requirement");
      const auto alignment = checked.input_requirements()[port].alignment_bytes;
      if (input.source == client::TensorStagePortSource::Output) {
        const auto id = physical_output_id(stage, input.tensor);
        const auto physical =
            std::find_if(stage.physical_outputs.begin(), stage.physical_outputs.end(),
                         [&](const auto& value) { return value.physical_index == id; });
        if (physical == stage.physical_outputs.end())
          throw std::invalid_argument("memory placement: dependent input has no carrier");
        physical->required_alignment_bytes =
            std::max<std::uint64_t>(physical->required_alignment_bytes, alignment);
        result.accesses[id].push_back(index);
        result.device_access[id] |= access;
      } else if (input.source == client::TensorStagePortSource::Input) {
        const auto logical = std::find_if(
            stage.logical_inputs.begin(), stage.logical_inputs.end(), [&](const auto& value) {
              return value.backend_input_index == static_cast<int>(input.tensor);
            });
        const auto physical = std::find_if(stage.physical_inputs.begin(),
                                           stage.physical_inputs.end(), [&](const auto& value) {
                                             return logical != stage.logical_inputs.end() &&
                                                    value.physical_index == logical->physical_index;
                                           });
        if (physical == stage.physical_inputs.end())
          throw std::invalid_argument("memory placement: numerical input has no carrier");
        physical->required_alignment_bytes =
            std::max<std::uint64_t>(physical->required_alignment_bytes, alignment);
      }
    }
  }
  for (const auto& logical : stage.logical_outputs) {
    result.accesses[logical.physical_index].push_back(count);
    result.device_access[logical.physical_index] |= client::kAccessCpuA65;
  }
  return result;
}

bool all_before(const std::vector<std::size_t>& first, const std::vector<std::size_t>& second,
                const std::vector<std::vector<bool>>& before) {
  if (first.empty() || second.empty())
    return false;
  for (const auto left : first)
    for (const auto right : second)
      if (left >= before.size() || right >= before[left].size() || !before[left][right])
        return false;
  return true;
}

client::AllocationPolicy owned_policy(std::uint32_t access,
                                      std::optional<client::AllocationPolicy> requested) {
  const auto policy = sc::resolve_allocation_policy(
      access, requested.value_or(client::AllocationPolicy::RequireDms));
  if (!policy)
    throw std::invalid_argument("memory placement: invalid owned allocation policy");
  return *policy;
}

client::FrameSpan subspan(client::FrameSpan root, std::uint64_t offset, std::uint64_t bytes) {
  if (offset > root.size_bytes || bytes > root.size_bytes - offset ||
      offset > std::numeric_limits<std::size_t>::max() - root.byte_offset)
    throw std::invalid_argument("memory placement: view exceeds its root carrier");
  root.byte_offset += static_cast<std::size_t>(offset);
  root.size_bytes = static_cast<std::size_t>(bytes);
  return root;
}

struct Carrier {
  sc::CarrierRequirement requirement;
  std::optional<std::size_t> model_scope;
  std::optional<sc::CarrierId> model_carrier;
  std::optional<std::size_t> kernel_stage;
  int physical_id = -1;
  std::vector<std::size_t> stages;
  std::size_t group = 0;
};
struct Group {
  std::vector<std::size_t> carriers;
  sc::StorageReuse reuse = sc::StorageReuse::DisjointLifetimes;
  sc::CarrierPacking packing;
  bool pinned = false;
};
struct InputRoute {
  std::optional<std::size_t> carrier;
  std::optional<std::size_t> producer;
};
struct StageStorage {
  std::optional<std::size_t> model_scope;
  std::unordered_map<int, std::size_t> outputs;
  std::unordered_map<int, InputRoute> inputs;
  std::optional<std::size_t> output_group;
  KernelStorageAccess kernel;
  std::shared_ptr<const client::FrameLayout> pinned_layout;
};

// All identities and lifetime evidence are build-local. Final stage layouts
// retain only Scheduler's allocations, spans and input ownership routes.
struct StorageCompilation {
  SimaPluginStaticManifest& manifest;
  std::vector<StageStorage> stages;
  std::vector<Carrier> carriers;
  std::vector<Group> groups;
  std::vector<std::size_t> group_roots;
  std::vector<sc::ModelStorageRequirements> models;
  std::vector<std::unordered_map<sc::CarrierId, std::size_t>> model_carriers;
  std::vector<std::vector<bool>> before;
  std::vector<StageConsumers> consumers;

  explicit StorageCompilation(SimaPluginStaticManifest& value)
      : manifest(value), stages(value.stages.size()),
        before(value.stages.size(), std::vector<bool>(value.stages.size())),
        consumers(value.stages.size()) {
    for (std::size_t index = 0; index < value.stages.size(); ++index) {
      const auto& layout = value.stages[index].frame_layout;
      if (layout && (layout->ownership == client::FrameOwnership::ReuseInput ||
                     std::any_of(layout->allocations.begin(), layout->allocations.end(),
                                 [](const auto& allocation) {
                                   return allocation.policy ==
                                          client::AllocationPolicy::PreserveImported;
                                 })))
        stages[index].pinned_layout = layout;
    }
  }

  std::size_t new_group() {
    const auto id = groups.size();
    groups.emplace_back();
    group_roots.push_back(id);
    return id;
  }
  std::size_t group_root(std::size_t id) const {
    while (group_roots[id] != id)
      id = group_roots[id];
    return id;
  }
  void join_group(std::size_t source, std::size_t destination) {
    source = group_root(source);
    destination = group_root(destination);
    if (source != destination)
      group_roots[source] = destination;
  }
  std::size_t add_carrier(sc::CarrierRequirement requirement, std::size_t group,
                          std::optional<std::size_t> scope = std::nullopt,
                          std::optional<sc::CarrierId> model_carrier = std::nullopt) {
    const auto id = carriers.size();
    requirement.carrier_id = id;
    carriers.push_back({std::move(requirement), scope, model_carrier, std::nullopt, -1, {}, group});
    groups[group].carriers.push_back(id);
    return id;
  }
  std::optional<std::size_t> input_producer(std::size_t index,
                                            const InputBindingStaticSpec& binding,
                                            const PhysicalBufferStaticSpec& physical,
                                            std::string* error) const {
    if (physical.model_carrier_id && stages[index].model_scope) {
      for (std::size_t producer = index; producer-- > 0;)
        if (stages[producer].model_scope == stages[index].model_scope &&
            std::any_of(manifest.stages[producer].physical_outputs.begin(),
                        manifest.stages[producer].physical_outputs.end(), [&](const auto& output) {
                          return output.model_carrier_id == physical.model_carrier_id;
                        }))
          return producer;
    }
    return edgecontract::resolve_producer_stage_exact(manifest, index, binding, error);
  }
  const InputRoute& input_route(std::size_t index, const InputBindingStaticSpec& binding,
                                const PhysicalBufferStaticSpec& physical) {
    auto& inputs = stages[index].inputs;
    if (const auto found = inputs.find(binding.local_logical_input_index); found != inputs.end())
      return found->second;
    InputRoute route;
    if (physical.model_carrier_id && stages[index].model_scope) {
      const auto& roots = model_carriers[*stages[index].model_scope];
      if (const auto found = roots.find(*physical.model_carrier_id); found != roots.end()) {
        route.carrier = found->second;
        std::string error;
        route.producer = input_producer(index, binding, physical, &error);
        if (!error.empty())
          throw std::invalid_argument("memory placement: " + error);
        return inputs.emplace(binding.local_logical_input_index, route).first->second;
      }
    }
    if (physical.address_source != PhysicalAddressSource::RuntimePhysicalBinding ||
        !binding.src_stage_id.empty() || binding.src_stage_index >= 0) {
      std::string error;
      route.producer = input_producer(index, binding, physical, &error);
      if (!error.empty())
        throw std::invalid_argument("memory placement: " + error);
      if (route.producer) {
        auto selected = binding;
        selected.src_stage_index = static_cast<int>(*route.producer);
        selected.src_stage_id.clear();
        const auto edge =
            edgecontract::resolve_edge_contract_for_binding(manifest, index, selected, &error);
        if (!edge)
          throw std::invalid_argument("memory placement: " + error);
        const auto found =
            stages[*route.producer].outputs.find(edge->producer_physical_output->physical_index);
        if (found != stages[*route.producer].outputs.end())
          route.carrier = found->second;
      }
    }
    return inputs.emplace(binding.local_logical_input_index, route).first->second;
  }

  void collect_models() {
    if (!manifest.model_storage_scopes)
      return;
    const auto& scopes = manifest.model_storage_scopes->scopes;
    models.reserve(scopes.size());
    model_carriers.resize(scopes.size());
    const auto stage_index = [&](const auto& origin) {
      const auto found =
          std::find_if(manifest.stages.begin(), manifest.stages.end(), [&](const auto& stage) {
            return stage.compile_stage_index == origin.compile_stage_index;
          });
      if (found == manifest.stages.end())
        throw std::invalid_argument("memory placement: model stage identity was lost");
      return static_cast<std::size_t>(found - manifest.stages.begin());
    };
    // Register every invocation before resolving edges: authored value names
    // are reusable between invocations, whereas carrier identities are scoped.
    for (std::size_t scope = 0; scope < scopes.size(); ++scope)
      for (const auto& origin : scopes[scope].stages)
        stages[stage_index(origin)].model_scope = scope;
    for (std::size_t scope_index = 0; scope_index < scopes.size(); ++scope_index) {
      const auto& scope = scopes[scope_index];
      if (!scope.source)
        throw std::invalid_argument("memory placement: model scope has no semantic source");
      auto physical = scope.source->physical;
      sc::ModelStorageRoute route;
      route.imported_inputs = scope.imported_inputs;
      route.published_outputs = scope.published_outputs;
      std::optional<client::AllocationPolicy> policy;
      for (const auto& origin : scope.stages) {
        const auto index = stage_index(origin);
        const auto& stage = manifest.stages[index];
        const auto access = resolved_stage_device_access(stage);
        if (stage.allocation_policy)
          policy = stage.allocation_policy;
        for (const auto command_id : origin.commands) {
          if (command_id >= physical.commands.size())
            throw std::invalid_argument("memory placement: unknown model command");
          physical.commands[command_id].resolved_executor =
              access & client::kAccessEv74  ? sc::PhysicalEngine::Cvu
              : access & client::kAccessMla ? sc::PhysicalEngine::Mla
                                            : sc::PhysicalEngine::A65;
          if (std::find(route.commands.begin(), route.commands.end(), command_id) ==
              route.commands.end())
            route.commands.push_back(command_id);
        }
      }
      // External consumers contribute device reachability before heap choice.
      for (std::size_t consumer = 0; consumer < manifest.stages.size(); ++consumer) {
        if (stages[consumer].model_scope == scope_index)
          continue;
        for (const auto& binding : manifest.stages[consumer].input_bindings) {
          const auto& stage = manifest.stages[consumer];
          const auto logical = std::find_if(
              stage.logical_inputs.begin(), stage.logical_inputs.end(), [&](const auto& input) {
                return input.logical_index == binding.local_logical_input_index;
              });
          const auto physical = std::find_if(
              stage.physical_inputs.begin(), stage.physical_inputs.end(), [&](const auto& input) {
                return logical != stage.logical_inputs.end() &&
                       input.physical_index == logical->physical_index;
              });
          if (physical == stage.physical_inputs.end())
            throw std::invalid_argument("memory placement: input binding has no physical carrier");
          std::string error;
          const auto producer = input_producer(consumer, binding, *physical, &error);
          if (!error.empty())
            throw std::invalid_argument("memory placement: " + error);
          if (!producer || stages[*producer].model_scope != scope_index)
            continue;
          auto selected = binding;
          selected.src_stage_index = static_cast<int>(*producer);
          selected.src_stage_id.clear();
          const auto edge =
              edgecontract::resolve_edge_contract_for_binding(manifest, consumer, selected, &error);
          if (!edge || !edge->producer_physical_output->model_carrier_id)
            throw std::invalid_argument("memory placement: external model edge has no carrier");
          const auto root = *edge->producer_physical_output->model_carrier_id;
          route.external_device_access[root] |=
              resolved_stage_device_access(manifest.stages[consumer]);
          // An external edge retains this carrier through the invocation's
          // completion, even when it exposes an intermediate model value.
          const auto value =
              std::find_if(scope.source->execution.values().begin(),
                           scope.source->execution.values().end(), [&](const auto& candidate) {
                             return candidate.storage_binding &&
                                    candidate.storage_binding->carrier_id == root &&
                                    candidate.storage_binding->kind != sc::StorageBindingKind::View;
                           });
          if (value == scope.source->execution.values().end())
            throw std::invalid_argument(
                "memory placement: external carrier has no materialized value");
          if (std::find(route.published_outputs.begin(), route.published_outputs.end(),
                        value->id) == route.published_outputs.end())
            route.published_outputs.push_back(value->id);
        }
      }
      std::string error;
      auto requirements = sc::collect_model_storage_requirements(
          scope.source->execution, physical, sc::kLegacyEvoCmaRegionAlignmentBytes, &error,
          policy == client::AllocationPolicy::PreferDmsThenCma
              ? client::AllocationPolicy::PreferDmsThenCma
              : client::AllocationPolicy::RequireDms,
          scope.detached_roots, &route);
      if (!requirements)
        throw std::invalid_argument("memory placement: " + error);
      models.push_back(std::move(*requirements));
      const auto add_group = [&](const auto& values) {
        if (values.empty())
          return;
        const auto group = new_group();
        for (const auto& requirement : values) {
          const auto root = static_cast<sc::CarrierId>(requirement.carrier_id);
          model_carriers[scope_index].emplace(root,
                                              add_carrier(requirement, group, scope_index, root));
        }
      };
      add_group(models.back().carriers);
      std::vector<std::size_t> detached_groups;
      for (const auto& [mla, outputs] : models.back().detached_outputs) {
        (void)outputs;
        detached_groups.push_back(mla);
      }
      std::sort(detached_groups.begin(), detached_groups.end());
      for (const auto mla : detached_groups)
        add_group(models.back().detached_outputs.at(mla));
    }
  }

  void collect_outputs() {
    for (std::size_t index = 0; index < manifest.stages.size(); ++index) {
      auto& stage = manifest.stages[index];
      auto& storage = stages[index];
      storage.kernel = inspect_stage_storage(stage);
      const bool independent_outputs =
          stage.native_kernel && !stage.native_kernel->private_buffers.empty();
      if (storage.pinned_layout) {
        storage.output_group = new_group();
        auto& group = groups[*storage.output_group];
        group.pinned = true;
        group.packing.allocations = storage.pinned_layout->allocations;
      }
      for (const auto& output : stage.physical_outputs) {
        if (!storage.pinned_layout && output.model_carrier_id && storage.model_scope) {
          const auto& roots = model_carriers[*storage.model_scope];
          const auto found = roots.find(*output.model_carrier_id);
          if (found != roots.end()) {
            auto& carrier = carriers[found->second];
            if (independent_outputs) {
              if (!storage.output_group)
                storage.output_group = new_group();
              if (carrier.group != *storage.output_group) {
                auto& previous = groups[carrier.group].carriers;
                previous.erase(std::remove(previous.begin(), previous.end(), found->second),
                               previous.end());
                carrier.group = *storage.output_group;
                groups[carrier.group].carriers.push_back(found->second);
              }
              // Checked multi-member recipes own their public and private
              // outputs together, independently of the model's shared group.
              carrier.model_scope.reset();
              carrier.model_carrier.reset();
              carrier.kernel_stage = index;
              carrier.physical_id = output.physical_index;
              carrier.requirement.size_bytes = output.size_bytes;
              carrier.requirement.required_alignment_bytes =
                  std::max<std::uint64_t>(64, output.required_alignment_bytes);
              const auto uses = storage.kernel.accesses.find(output.physical_index);
              const auto access = storage.kernel.device_access.find(output.physical_index);
              if (uses == storage.kernel.accesses.end() || uses->second.empty() ||
                  access == storage.kernel.device_access.end())
                throw std::invalid_argument(
                    "memory placement: independent model output has no checked access");
              carrier.requirement.lifetime = {
                  *std::min_element(uses->second.begin(), uses->second.end()),
                  *std::max_element(uses->second.begin(), uses->second.end())};
              carrier.requirement.required_device_access = access->second;
              carrier.requirement.policy = owned_policy(access->second, stage.allocation_policy);
            }
            storage.outputs.emplace(output.physical_index, found->second);
            storage.output_group = carrier.group;
          }
        }
      }
      for (const auto& output : stage.physical_outputs) {
        auto found = storage.outputs.find(output.physical_index);
        // A byte-preserving transport keeps the producer's carrier identity.
        // Numerical stages have independent writes even when they reuse a buffer group.
        if (found == storage.outputs.end() && stage.reuse_input_pad && !stage.native_kernel &&
            !stage.tvm_program && stage.payload_kind != StagePayloadKind::ProcessMla &&
            output.source_physical_index >= 0) {
          for (const auto& binding : stage.input_bindings) {
            if (binding.sink_pad_index < 0 ||
                *stage.reuse_input_pad != static_cast<std::size_t>(binding.sink_pad_index))
              continue;
            const auto input =
                std::find_if(stage.physical_inputs.begin(), stage.physical_inputs.end(),
                             [&](const auto& physical) {
                               return physical.physical_index == output.source_physical_index;
                             });
            if (input == stage.physical_inputs.end())
              continue;
            if (const auto source = input_route(index, binding, *input).carrier)
              found = storage.outputs.emplace(output.physical_index, *source).first;
          }
        }
        if (found == storage.outputs.end()) {
          if (!storage.output_group)
            storage.output_group = new_group();
          auto access = resolved_stage_device_access(stage);
          if (stage.native_kernel) {
            const auto checked = storage.kernel.device_access.find(output.physical_index);
            if (checked == storage.kernel.device_access.end())
              throw std::invalid_argument("memory placement: output has no checked access");
            access = checked->second;
          }
          const auto policy = storage.pinned_layout ? client::AllocationPolicy::PreserveImported
                                                    : owned_policy(access, stage.allocation_policy);
          if (output.source_byte_offset < 0 ||
              static_cast<std::uint64_t>(output.source_byte_offset) >
                  std::numeric_limits<std::uint64_t>::max() - output.size_bytes)
            throw std::invalid_argument("memory placement: output carrier extent overflows");
          sc::CarrierRequirement requirement{
              0,
              static_cast<std::uint64_t>(output.source_byte_offset) + output.size_bytes,
              std::max<std::uint64_t>(64, output.required_alignment_bytes),
              {},
              policy,
              access};
          const auto uses = storage.kernel.accesses.find(output.physical_index);
          if (uses != storage.kernel.accesses.end() && !uses->second.empty())
            requirement.lifetime = {*std::min_element(uses->second.begin(), uses->second.end()),
                                    *std::max_element(uses->second.begin(), uses->second.end())};
          const auto id = add_carrier(requirement, *storage.output_group);
          carriers[id].kernel_stage = index;
          carriers[id].physical_id = output.physical_index;
          found = storage.outputs.emplace(output.physical_index, id).first;
          if (storage.pinned_layout) {
            const auto logical = std::find_if(
                stage.logical_outputs.begin(), stage.logical_outputs.end(),
                [&](const auto& value) { return value.physical_index == output.physical_index; });
            const auto tensor = logical == stage.logical_outputs.end()
                                    ? static_cast<std::size_t>(output.physical_index)
                                    : static_cast<std::size_t>(logical->backend_output_index);
            const auto fixed = std::find_if(
                storage.pinned_layout->outputs.begin(), storage.pinned_layout->outputs.end(),
                [&](const auto& value) { return value.tensor == tensor; });
            if (fixed == storage.pinned_layout->outputs.end() || output.source_byte_offset < 0 ||
                static_cast<std::uint64_t>(output.source_byte_offset) > fixed->span.byte_offset)
              throw std::invalid_argument("memory placement: imported output has no pinned span");
            auto root = fixed->span;
            root.byte_offset -= static_cast<std::size_t>(output.source_byte_offset);
            root.size_bytes =
                output.size_bytes + static_cast<std::size_t>(output.source_byte_offset);
            groups[*storage.output_group].packing.carriers.push_back({id, root});
          }
        }
        auto& carrier = carriers[found->second];
        if (carrier.stages.empty() && stage.allocation_policy && !storage.pinned_layout)
          carrier.requirement.policy =
              owned_policy(carrier.requirement.required_device_access, stage.allocation_policy);
        carrier.stages.push_back(index);
        carrier.requirement.required_alignment_bytes =
            std::max(carrier.requirement.required_alignment_bytes, output.required_alignment_bytes);
        const auto group = carrier.group;
        if (storage.output_group && group_root(*storage.output_group) != group_root(group))
          throw std::invalid_argument(
              "memory placement: stage outputs have different ownership groups");
        storage.output_group = group;
      }
      if (storage.output_group && stage.native_kernel && stage.native_kernel->image_publication)
        groups[*storage.output_group].reuse = sc::StorageReuse::Disabled;
    }
  }

  void collect_edges() {
    for (std::size_t index = 0; index < manifest.stages.size(); ++index) {
      const auto& stage = manifest.stages[index];
      for (const auto& binding : stage.input_bindings) {
        const auto logical = std::find_if(
            stage.logical_inputs.begin(), stage.logical_inputs.end(), [&](const auto& input) {
              return input.logical_index == binding.local_logical_input_index;
            });
        const auto physical = std::find_if(stage.physical_inputs.begin(),
                                           stage.physical_inputs.end(), [&](const auto& input) {
                                             return logical != stage.logical_inputs.end() &&
                                                    input.physical_index == logical->physical_index;
                                           });
        if (physical == stage.physical_inputs.end())
          throw std::invalid_argument("memory placement: input binding has no physical carrier");
        const auto& route = input_route(index, binding, *physical);
        const auto producer = route.producer;
        if (producer) {
          if (*producer >= index)
            throw std::invalid_argument("memory placement requires topological allocation owners");
          before[*producer][index] = true;
          consumers[*producer].has_consumers = true;
        }
        const auto carrier = route.carrier;
        if (!carrier)
          continue;
        auto& root = carriers[*carrier];
        root.stages.push_back(index);
        root.requirement.required_device_access |= resolved_stage_device_access(stage);
        root.requirement.required_alignment_bytes =
            std::max(root.requirement.required_alignment_bytes, physical->required_alignment_bytes);
        if (!groups[group_root(root.group)].pinned)
          root.requirement.policy =
              owned_policy(root.requirement.required_device_access, root.requirement.policy);
        if (stage.reuse_input_pad == static_cast<std::size_t>(binding.sink_pad_index) &&
            stages[index].output_group) {
          join_group(*stages[index].output_group, root.group);
          consumers[index].shared_allocation = true;
          if (producer)
            consumers[*producer].shared_allocation = true;
        }
      }
      for (std::size_t ancestor = 0; ancestor < index; ++ancestor)
        if (before[ancestor][index])
          for (std::size_t previous = 0; previous < ancestor; ++previous)
            if (before[previous][ancestor])
              before[previous][index] = true;
    }
  }

  bool conflict(const sc::CarrierRequirement& lhs, const sc::CarrierRequirement& rhs) const {
    const auto& left = carriers[lhs.carrier_id];
    const auto& right = carriers[rhs.carrier_id];
    if (left.model_scope && left.model_scope == right.model_scope) {
      auto first = lhs, second = rhs;
      first.carrier_id = *left.model_carrier;
      second.carrier_id = *right.model_carrier;
      return models[*left.model_scope].lifetimes_conflict(first, second);
    }
    if (left.kernel_stage && left.kernel_stage == right.kernel_stage) {
      const auto& access = stages[*left.kernel_stage].kernel;
      const auto first = access.accesses.find(left.physical_id);
      const auto second = access.accesses.find(right.physical_id);
      return first == access.accesses.end() || second == access.accesses.end() ||
             (!all_before(first->second, second->second, access.before) &&
              !all_before(second->second, first->second, access.before));
    }
    return !all_before(left.stages, right.stages, before) &&
           !all_before(right.stages, left.stages, before);
  }

  void select_terminal(const terminal_output_contract::PublicOutputEndpointSelector* endpoint) {
    const auto* terminal = public_output_candidate(manifest, endpoint, consumers);
    if (!terminal)
      return;
    const auto index = static_cast<std::size_t>(terminal - manifest.stages.data());
    const auto& stage = manifest.stages[index];
    const auto& storage = stages[index];
    if (!storage.output_group || stage.allocation_policy || stage.reuse_input_pad ||
        consumers[index].has_consumers || consumers[index].shared_allocation ||
        !stage.native_kernel || stage.native_kernel->target != client::Hardware::A65 ||
        !storage.kernel.all_members_cpu || stage.image_publication ||
        stage.native_kernel->image_publication || stage.native_kernel->frame_adapter_factory ||
        stage.logical_outputs.empty() || stage.output_order.empty() ||
        !std::all_of(groups[group_root(*storage.output_group)].carriers.begin(),
                     groups[group_root(*storage.output_group)].carriers.end(),
                     [&](const auto id) {
                       const auto& carrier = carriers[id];
                       return carrier.requirement.required_alignment_bytes <= 64 &&
                              std::all_of(carrier.stages.begin(), carrier.stages.end(),
                                          [&](const auto use) { return use == index; });
                     }) ||
        !std::all_of(stage.logical_outputs.begin(), stage.logical_outputs.end(),
                     dense_publication) ||
        !std::all_of(stage.physical_outputs.begin(), stage.physical_outputs.end(),
                     [&](const auto& output) {
                       return output.source_physical_index == -1 &&
                              output.device_kind == DeviceKind::Cpu && !output.memory_flags &&
                              output.address_source == PhysicalAddressSource::DirectCarrierSpan &&
                              output.required_alignment_bytes <= 64 &&
                              carriers[storage.outputs.at(output.physical_index)]
                                      .requirement.required_device_access == client::kAccessCpuA65;
                     }))
      return;
    for (const auto id : groups[*storage.output_group].carriers)
      carriers[id].requirement.policy =
          owned_policy(carriers[id].requirement.required_device_access,
                       client::AllocationPolicy::RequireHostCpu);
  }

  void require_public_cma(const terminal_output_contract::PublicOutputEndpointSelector* endpoint) {
    const auto* terminal = public_output_candidate(manifest, endpoint, consumers);
    if (!terminal || terminal->logical_outputs.empty())
      throw std::invalid_argument(
          "OutputPlacement::EV74Compatible requires a resolved public tensor output.");
    const auto index = static_cast<std::size_t>(terminal - manifest.stages.data());
    auto publication = *terminal;
    // An Output name may label the aggregate endpoint. A matching tensor/slot
    // selector narrows placement using the same rules as publication.
    if (terminal_output_contract::stage_public_outputs_match_endpoint(*terminal, *endpoint) &&
        !terminal_output_contract::filter_publication_stage_for_endpoint(&publication, *endpoint))
      throw std::invalid_argument("stage '" + terminal->element_name +
                                  "': EV74Compatible has no selected tensor output.");
    for (const auto& output : publication.logical_outputs) {
      auto& carrier = carriers.at(stages[index].outputs.at(output.physical_index));
      if (groups[group_root(carrier.group)].pinned)
        throw std::invalid_argument("stage '" + terminal->element_name +
                                    "': EV74Compatible cannot relocate imported backing; "
                                    "use tensor.cvu() for an explicit copy.");
      carrier.requirement.policy = client::AllocationPolicy::RequireCma;
    }
  }

  void pack_groups() {
    for (std::size_t id = 0; id < groups.size(); ++id)
      if (group_root(id) != id) {
        auto& destination = groups[group_root(id)];
        if (groups[id].pinned) {
          if (destination.pinned)
            throw std::invalid_argument(
                "memory placement: distinct imported groups cannot be merged");
          destination.pinned = true;
          destination.packing = std::move(groups[id].packing);
        }
        destination.carriers.insert(destination.carriers.end(), groups[id].carriers.begin(),
                                    groups[id].carriers.end());
        if (groups[id].reuse == sc::StorageReuse::Disabled)
          destination.reuse = sc::StorageReuse::Disabled;
      }
    for (std::size_t id = 0; id < groups.size(); ++id) {
      if (group_root(id) != id || groups[id].carriers.empty())
        continue;
      if (groups[id].pinned) {
        for (const auto carrier : groups[id].carriers) {
          const auto* fixed = groups[id].packing.span(carrier);
          if (!fixed || fixed->allocation_index >= groups[id].packing.allocations.size())
            throw std::invalid_argument(
                "memory placement: imported group cannot relocate a carrier");
          auto& allocation = groups[id].packing.allocations[fixed->allocation_index];
          if (fixed->byte_offset > allocation.size_bytes ||
              fixed->size_bytes > allocation.size_bytes - fixed->byte_offset)
            throw std::invalid_argument("memory placement: imported span exceeds its allocation");
          allocation.required_device_access |= carriers[carrier].requirement.required_device_access;
        }
        continue;
      }
      std::vector<sc::CarrierRequirement> requirements;
      for (const auto carrier : groups[id].carriers)
        requirements.push_back(carriers[carrier].requirement);
      std::string error;
      auto packed = sc::pack_carriers(
          std::move(requirements), groups[id].reuse,
          [&](const auto& lhs, const auto& rhs) { return conflict(lhs, rhs); }, &error);
      if (!packed)
        throw std::invalid_argument("memory placement: " + error);
      // Keep the existing page-sized owned pool envelope; spans keep their exact capacities.
      auto& last = packed->allocations.back();
      if (last.size_bytes > std::numeric_limits<std::size_t>::max() - 4095U)
        throw std::invalid_argument("memory placement: allocation extent overflows");
      last.size_bytes = (last.size_bytes + 4095U) & ~std::size_t{4095U};
      groups[id].packing = std::move(*packed);
    }
  }

  client::FrameSpan span(std::size_t id) const {
    const auto* value = groups[group_root(carriers[id].group)].packing.span(id);
    if (!value)
      throw std::invalid_argument("memory placement: carrier was not packed");
    return *value;
  }

  std::optional<std::pair<std::size_t, std::uint64_t>>
  imported_model_origin(std::size_t index, const PhysicalBufferStaticSpec& physical) const {
    const auto scope = stages[index].model_scope;
    if (!scope || !physical.model_carrier_id ||
        model_carriers[*scope].contains(*physical.model_carrier_id))
      return std::nullopt;
    const auto& source = manifest.model_storage_scopes->scopes[*scope].source->execution;
    const auto& imported_inputs = models[*scope].imported_inputs;
    for (std::size_t input = 0; input < imported_inputs.size(); ++input) {
      const auto* value = source.value(imported_inputs[input]);
      if (!value || !value->storage_binding ||
          value->storage_binding->carrier_id != *physical.model_carrier_id)
        continue;
      const auto& imported = *value->storage_binding;
      if (physical.source_byte_offset >= 0 &&
          static_cast<std::uint64_t>(physical.source_byte_offset) >= imported.byte_offset) {
        const auto relative =
            static_cast<std::uint64_t>(physical.source_byte_offset) - imported.byte_offset;
        if (relative <= imported.physical_span &&
            physical.size_bytes <= imported.physical_span - relative)
          return std::pair{input, imported.byte_offset};
      }
    }
    return std::nullopt;
  }

  void publish_layouts() {
    for (std::size_t index = 0; index < manifest.stages.size(); ++index) {
      auto& stage = manifest.stages[index];
      const auto& storage = stages[index];
      if (!storage.output_group)
        continue;
      const auto group = group_root(*storage.output_group);
      auto layout = std::make_shared<client::FrameLayout>();
      layout->allocations = groups[group].packing.allocations;
      if (storage.pinned_layout) {
        layout->ownership = storage.pinned_layout->ownership;
        layout->source_pad = storage.pinned_layout->source_pad;
      }
      const auto add_output = [&](std::size_t tensor, int physical_id) {
        if (std::any_of(layout->outputs.begin(), layout->outputs.end(),
                        [&](const auto& output) { return output.tensor == tensor; }))
          return;
        const auto physical =
            std::find_if(stage.physical_outputs.begin(), stage.physical_outputs.end(),
                         [&](const auto& output) { return output.physical_index == physical_id; });
        if (physical == stage.physical_outputs.end() || physical->source_byte_offset < 0)
          throw std::invalid_argument("memory placement: invalid output identity or offset");
        auto offset = static_cast<std::uint64_t>(physical->source_byte_offset);
        auto bytes = physical->size_bytes;
        if (stage.native_kernel && stage.native_kernel->image_publication) {
          const auto plane = std::find_if(
              stage.logical_outputs.begin(), stage.logical_outputs.end(), [&](const auto& logical) {
                return logical.backend_output_index == static_cast<int>(tensor);
              });
          if (plane == stage.logical_outputs.end() || plane->byte_offset < 0 ||
              std::uint64_t(plane->byte_offset) > bytes ||
              plane->size_bytes > bytes - std::uint64_t(plane->byte_offset))
            throw std::invalid_argument("memory placement: image plane exceeds its carrier");
          offset += static_cast<std::uint64_t>(plane->byte_offset);
          bytes = plane->size_bytes;
        }
        layout->outputs.push_back(
            {tensor, subspan(span(storage.outputs.at(physical_id)), offset, bytes)});
      };
      if (stage.native_kernel) {
        for (const auto& member : stage.native_kernel->members)
          for (const auto& port : member.outputs)
            add_output(port.tensor, physical_output_id(stage, port.tensor));
      } else {
        for (const auto& output : stage.physical_outputs)
          add_output(static_cast<std::size_t>(output.physical_index), output.physical_index);
      }
      if (!stage.native_kernel || !stage.native_kernel->frame_adapter_factory)
        for (const auto& logical : stage.logical_inputs) {
          const auto binding = std::find_if(
              stage.input_bindings.begin(), stage.input_bindings.end(), [&](const auto& value) {
                return value.local_logical_input_index == logical.logical_index;
              });
          const auto physical = std::find_if(
              stage.physical_inputs.begin(), stage.physical_inputs.end(),
              [&](const auto& input) { return input.physical_index == logical.physical_index; });
          if (binding == stage.input_bindings.end() || physical == stage.physical_inputs.end() ||
              binding->sink_pad_index < 0 || physical->source_byte_offset < 0 ||
              logical.backend_input_index < 0 || logical.byte_offset < 0)
            throw std::invalid_argument("memory placement: incomplete physical input route");
          const auto pad = static_cast<std::size_t>(binding->sink_pad_index);
          const auto& route = input_route(index, *binding, *physical);
          // Preserve the resolved edge after build-local model provenance is discarded.
          if (route.producer)
            binding->src_stage_index = static_cast<int>(*route.producer);
          const auto owner = route.carrier;
          client::FrameInputBinding input;
          input.view_offset = static_cast<std::size_t>(logical.byte_offset);
          const auto imported_origin = imported_model_origin(index, *physical);
          const bool direct =
              physical->address_source == PhysicalAddressSource::DirectCarrierSpan ||
              (physical->address_source == PhysicalAddressSource::FrameSpan && imported_origin);
          if (owner && physical->address_source != PhysicalAddressSource::RuntimePhysicalBinding) {
            auto offset = static_cast<std::uint64_t>(physical->source_byte_offset);
            if (!physical->model_carrier_id) {
              if (binding->src_physical_byte_offset < 0)
                throw std::invalid_argument("memory placement: negative input view offset");
              offset += static_cast<std::uint64_t>(binding->src_physical_byte_offset);
            }
            const auto source_span = subspan(span(*owner), offset, physical->size_bytes);
            input.source = client::KnownFrameSpan{pad, source_span};
            if (!storage.pinned_layout && group_root(carriers[*owner].group) == group &&
                (!stage.reuse_input_pad || *stage.reuse_input_pad == pad) &&
                (stage.reuse_input_pad || layout->ownership == client::FrameOwnership::Allocate)) {
              layout->ownership = client::FrameOwnership::ReuseInput;
              layout->source_pad = pad;
            }
          } else if (storage.pinned_layout &&
                     static_cast<std::size_t>(logical.backend_input_index) <
                         storage.pinned_layout->inputs.size()) {
            input.source =
                storage.pinned_layout->inputs[static_cast<std::size_t>(logical.backend_input_index)]
                    .source;
          } else {
            const bool root = binding->src_stage_index < 0 && binding->src_stage_id.empty() &&
                              binding->src_logical_output_index < 0 &&
                              binding->src_output_slot < 0 &&
                              binding->src_physical_output_index < 0;
            const auto source =
                imported_origin && physical->address_source == PhysicalAddressSource::FrameSpan
                    ? static_cast<int>(imported_origin->first)
                : root ? physical->physical_index
                       : binding->src_physical_output_index;
            if (source < 0)
              throw std::invalid_argument("memory placement: input has no physical source");
            const auto offset =
                static_cast<std::size_t>(static_cast<std::uint64_t>(physical->source_byte_offset) -
                                         (imported_origin ? imported_origin->second : 0));
            // Public tensor views select this frame's binding, not its allocation
            // ordinal. Resolve this only after producer discovery: classifying
            // inputs as external earlier would hide genuine internal edges.
            const bool public_tensor =
                physical->address_source == PhysicalAddressSource::DirectCarrierSpan &&
                stage.payload_kind == StagePayloadKind::ProcessCvu && stage.native_kernel &&
                !stage.native_kernel->frame_adapter_factory && !physical->model_carrier_id &&
                !storage.model_scope && !storage.pinned_layout && !route.producer && !owner;
            if (direct && !public_tensor)
              input.source = client::DirectMemory{pad, static_cast<std::size_t>(source), offset,
                                                  static_cast<std::size_t>(physical->size_bytes)};
            else if (public_tensor ||
                     physical->address_source == PhysicalAddressSource::RuntimePhysicalBinding)
              input.source =
                  client::PhysicalBinding{pad, static_cast<std::size_t>(source), offset,
                                          static_cast<std::size_t>(physical->size_bytes)};
            else
              throw std::invalid_argument("memory placement: known span has no allocation owner");
          }
          const auto slot = static_cast<std::size_t>(logical.backend_input_index);
          if (slot >= layout->inputs.size())
            layout->inputs.resize(slot + 1);
          layout->inputs[slot] = std::move(input);
        }
      if (index && storage.model_scope && stages[index - 1].model_scope == storage.model_scope &&
          stages[index - 1].output_group && group_root(*stages[index - 1].output_group) == group &&
          layout->ownership == client::FrameOwnership::Allocate && !stage.input_bindings.empty()) {
        const auto pad = stage.reuse_input_pad.value_or(
            static_cast<std::size_t>(stage.input_bindings.front().sink_pad_index));
        if (stage.reuse_input_pad ||
            std::all_of(stage.input_bindings.begin(), stage.input_bindings.end(),
                        [&](const auto& binding) {
                          return binding.sink_pad_index >= 0 &&
                                 static_cast<std::size_t>(binding.sink_pad_index) == pad;
                        })) {
          layout->ownership = client::FrameOwnership::ReuseInput;
          layout->source_pad = pad;
        }
      }
      stage.frame_layout = std::move(layout);
      stage.allocation_policy.reset();
      stage.reuse_input_pad.reset();
    }
  }
  void finalize_tvm_epochs() {
    for (std::size_t index = 0; index < manifest.stages.size(); ++index) {
      auto& stage = manifest.stages[index];
      if (!stages[index].model_scope || !stage.tvm_program || !stage.frame_layout)
        continue;
      auto program = std::make_shared<client::TvmProgram>(*stage.tvm_program);
      program->cpu_epoch_start = program->cpu_epoch_end = true;
      if (index && manifest.stages[index - 1].tvm_program &&
          stages[index - 1].model_scope == stages[index].model_scope &&
          stages[index - 1].output_group && stages[index].output_group &&
          group_root(*stages[index - 1].output_group) == group_root(*stages[index].output_group) &&
          !groups[group_root(*stages[index].output_group)].pinned &&
          stage.frame_layout->ownership == client::FrameOwnership::ReuseInput &&
          stage.frame_layout->allocations.size() == 1) {
        auto previous =
            std::make_shared<client::TvmProgram>(*manifest.stages[index - 1].tvm_program);
        previous->cpu_epoch_end = false;
        manifest.stages[index - 1].tvm_program = std::move(previous);
        program->cpu_epoch_start = false;
      }
      stage.tvm_program = std::move(program);
    }
  }
};
} // namespace

void finalize_pipeline_storage(
    SimaPluginStaticManifest& manifest,
    const terminal_output_contract::PublicOutputEndpointSelector* endpoint,
    OutputPlacement placement, std::span<const std::string> compatible_output_names) {
  StorageCompilation storage(manifest);
  storage.collect_models();
  storage.collect_outputs();
  storage.collect_edges();
  if (placement == OutputPlacement::EV74Compatible) {
    if (compatible_output_names.empty())
      storage.require_public_cma(endpoint);
    else
      for (const auto& name : compatible_output_names) {
        auto selected =
            endpoint ? *endpoint : terminal_output_contract::PublicOutputEndpointSelector{};
        terminal_output_contract::set_output_selector(selected, name);
        storage.require_public_cma(&selected);
      }
  } else {
    storage.select_terminal(endpoint);
  }
  storage.pack_groups();
  storage.publish_layouts();
  storage.finalize_tvm_epochs();
  manifest.model_storage_scopes.reset();
}

} // namespace simaai::neat::pipeline_internal::sima

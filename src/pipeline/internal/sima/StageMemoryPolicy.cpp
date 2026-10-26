#include "pipeline/internal/sima/StageMemoryPolicy.h"
#include "model/internal/ModelPack.h"
#include "pipeline/internal/TerminalOutputContractQuery.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/sima/static_contract/CarrierArenaPacking.h"
#include <simaai/scheduler/client/frame_layout.h>
#include "pipeline/internal/sima/InternalEdgeContractResolver.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/ProcessCvuRunTargetPolicy.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemanticsInternal.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <limits>
#include <unordered_map>

namespace simaai::neat::pipeline_internal::sima {
namespace {
namespace sc = static_contract;
struct StageCarrierAccess {
  std::unordered_map<int, std::uint32_t> outputs;
  bool shared_allocation = false;
  bool has_consumers = false;
};

} // namespace

std::uint32_t resolved_stage_device_access(const StageStaticSpec& stage) {
  if (stage.native_kernel) {
    switch (stage.native_kernel->target) {
    case ::simaai::scheduler::client::Hardware::EV74:
      return static_cast<std::uint32_t>(sc::ArenaDeviceAccess::Ev74);
    case ::simaai::scheduler::client::Hardware::A65:
      return static_cast<std::uint32_t>(sc::ArenaDeviceAccess::CpuA65);
    default:
      throw std::invalid_argument("memory placement requires a resolved kernel executor");
    }
  }
  if (stage.tvm_program)
    return static_cast<std::uint32_t>(sc::ArenaDeviceAccess::CpuA65);
  if (stage.payload_kind == StagePayloadKind::ProcessMla)
    return static_cast<std::uint32_t>(sc::ArenaDeviceAccess::Mla);
  if (stage.payload_kind == StagePayloadKind::ProcessCvu) {
    if (stage.processcvu.resolved_exec_backend == "A65")
      return static_cast<std::uint32_t>(sc::ArenaDeviceAccess::CpuA65);
    if (stage.processcvu.resolved_exec_backend == "EV74" ||
        stage.processcvu.resolved_exec_backend == "EVXX")
      return static_cast<std::uint32_t>(sc::ArenaDeviceAccess::Ev74);
    throw std::invalid_argument("memory placement requires a resolved CVU executor");
  }
  // Transport-only stages already carry their compiled access contract.
  return stage.frame_arena_required_device_access;
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

static std::vector<StageCarrierAccess>
resolve_memory_placement(SimaPluginStaticManifest& manifest) {
  const auto count = manifest.stages.size();
  std::vector<StageCarrierAccess> carrier_access(count);
  std::vector<std::uint32_t> stage_access;
  stage_access.reserve(count);
  for (const auto& stage : manifest.stages)
    stage_access.push_back(resolved_stage_device_access(stage));
  std::vector<std::size_t> allocation_roots(count);
  std::iota(allocation_roots.begin(), allocation_roots.end(), 0U);
  const auto find_allocation_root = [&](std::size_t index) {
    while (allocation_roots[index] != index)
      index = allocation_roots[index];
    return index;
  };
  std::vector<std::vector<std::size_t>> producers(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto& stage = manifest.stages[index];
    for (const auto& binding : stage.input_bindings) {
      const auto logical = std::find_if(
          stage.logical_inputs.begin(), stage.logical_inputs.end(), [&](const auto& input) {
            return input.logical_index == binding.local_logical_input_index;
          });
      const auto physical = std::find_if(stage.physical_inputs.begin(), stage.physical_inputs.end(),
                                         [&](const auto& input) {
                                           return logical != stage.logical_inputs.end() &&
                                                  input.physical_index == logical->physical_index;
                                         });
      if (physical != stage.physical_inputs.end() &&
          physical->address_source == PhysicalAddressSource::RuntimePhysicalBinding &&
          binding.src_stage_id.empty() && binding.src_stage_index < 0)
        continue;
      std::string error;
      const auto producer =
          edgecontract::resolve_producer_stage_exact(manifest, index, binding, &error);
      if (!error.empty())
        throw std::invalid_argument("memory placement: " + error);
      if (!producer)
        continue; // Public input: retain the producer's allocation and admission.
      if (*producer >= index)
        throw std::invalid_argument("memory placement requires topological allocation owners");
      const bool frame_input =
          physical != stage.physical_inputs.end()
              ? physical->address_source == PhysicalAddressSource::FrameArenaSpan
              : !stage.physical_inputs.empty() &&
                    std::all_of(stage.physical_inputs.begin(), stage.physical_inputs.end(),
                                [](const auto& input) {
                                  return input.address_source ==
                                         PhysicalAddressSource::FrameArenaSpan;
                                });
      const bool shared = stage.frame_arena_role == FrameArenaRole::ReuseInput && frame_input;
      producers[index].push_back(*producer);
      if (shared)
        allocation_roots[find_allocation_root(index)] = find_allocation_root(*producer);
      carrier_access[*producer].has_consumers = true;
      const auto& source = manifest.stages[*producer];
      if (source.native_kernel && !source.model_storage_planned &&
          source.frame_arena_role == FrameArenaRole::Allocate) {
        auto selected = binding;
        selected.src_stage_index = static_cast<int>(*producer);
        selected.src_stage_id.clear();
        const auto edge =
            edgecontract::resolve_edge_contract_for_binding(manifest, index, selected, &error);
        if (!edge)
          throw std::invalid_argument("memory placement: " + error);
        carrier_access[*producer].outputs[edge->producer_physical_output->physical_index] |=
            stage_access[index];
      }
    }
  }
  std::vector<bool> shared_allocations(count, false);
  for (std::size_t index = 0; index < count; ++index)
    if (manifest.stages[index].frame_arena_role == FrameArenaRole::ReuseInput)
      shared_allocations[find_allocation_root(index)] = true;
  std::vector<std::uint32_t> access(count, 0U);
  for (std::size_t index = 0; index < count; ++index) {
    carrier_access[index].shared_allocation = shared_allocations[find_allocation_root(index)];
    const auto required = stage_access[index];
    access[find_allocation_root(index)] |= required;
    for (const auto producer : producers[index])
      access[find_allocation_root(producer)] |= required;
  }
  for (std::size_t index = 0; index < count; ++index) {
    auto& stage = manifest.stages[index];
    if (stage.model_storage_planned || (!stage.frame_arena_size_bytes && !stage.native_kernel) ||
        stage.frame_arena_provenance != sc::ArenaAllocationProvenance::CoreAllocated)
      continue;
    if (!stage.frame_arena_allocations.empty()) {
      // Model carrier placement was resolved before packing. Validate each
      // actual numerical access without collapsing the separate heaps.
      if (stage.native_kernel || stage.payload_kind == StagePayloadKind::ProcessCvu ||
          stage.payload_kind == StagePayloadKind::ProcessMla) {
        const auto required = resolved_stage_device_access(stage);
        const auto validate_span = [&](const PhysicalBufferStaticSpec& physical) {
          if (physical.source_byte_offset < 0)
            throw std::invalid_argument("memory placement: negative arena offset");
          const auto offset = static_cast<std::uint64_t>(physical.source_byte_offset);
          const auto part =
              std::find_if(stage.frame_arena_allocations.begin(),
                           stage.frame_arena_allocations.end(), [&](const auto& allocation) {
                             return offset >= allocation.byte_offset &&
                                    offset - allocation.byte_offset <= allocation.size_bytes &&
                                    physical.size_bytes <=
                                        allocation.size_bytes - (offset - allocation.byte_offset);
                           });
          if (part == stage.frame_arena_allocations.end() ||
              (part->placement.required_device_access & required) != required ||
              (part->placement.domain == sc::ArenaStorageDomain::Dms &&
               (required & static_cast<std::uint32_t>(sc::ArenaDeviceAccess::Ev74))))
            throw std::invalid_argument(
                "memory placement: executor cannot access the authored carrier");
        };
        for (const auto& output : stage.physical_outputs)
          validate_span(output);
        for (const auto& input : stage.physical_inputs)
          if (input.address_source == PhysicalAddressSource::FrameArenaSpan)
            validate_span(input);
      }
      continue;
    }
    if (stage.native_kernel && !carrier_access[index].shared_allocation)
      continue; // Independent outputs are placed once, per checked carrier below.
    const auto required = access[find_allocation_root(index)];
    if (!required)
      throw std::invalid_argument("memory placement requires an allocation access contract");
    const bool ev74 = required & static_cast<std::uint32_t>(sc::ArenaDeviceAccess::Ev74);
    stage.frame_arena_storage_domain =
        ev74 ? sc::ArenaStorageDomain::Cma : sc::ArenaStorageDomain::Dms;
    stage.frame_arena_required_device_access = required;
  }
  return carrier_access;
}

namespace {
namespace client = ::simaai::scheduler::client;

const StageStaticSpec*
terminal_host_candidate(const SimaPluginStaticManifest& manifest,
                        const terminal_output_contract::PublicOutputEndpointSelector* endpoint,
                        std::span<const StageCarrierAccess> access) {
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

client::AllocationPolicy allocation_policy(const sc::ArenaPlacement& placement) {
  if (placement.provenance == sc::ArenaAllocationProvenance::ExternalAdopted)
    return client::AllocationPolicy::PreserveImported;
  if (placement.domain == sc::ArenaStorageDomain::HostCpu)
    return client::AllocationPolicy::RequireHostCpu;
  if (placement.domain == sc::ArenaStorageDomain::Dms &&
      placement.allocation_policy == sc::ArenaAllocationPolicy::PreferDmsThenCma)
    return client::AllocationPolicy::PreferDmsThenCma;
  return placement.domain == sc::ArenaStorageDomain::Dms ? client::AllocationPolicy::RequireDms
                                                         : client::AllocationPolicy::RequireCma;
}

client::FrameSpan allocation_span(const StageStaticSpec& stage, std::uint64_t offset,
                                  std::uint64_t bytes) {
  if (stage.frame_arena_allocations.empty()) {
    if (offset > stage.frame_arena_size_bytes || bytes > stage.frame_arena_size_bytes - offset)
      throw std::invalid_argument("memory placement: carrier exceeds its allocation");
    return {0, static_cast<std::size_t>(offset), static_cast<std::size_t>(bytes)};
  }
  for (std::size_t index = 0; index < stage.frame_arena_allocations.size(); ++index) {
    const auto& part = stage.frame_arena_allocations[index];
    if (offset >= part.byte_offset && offset - part.byte_offset <= part.size_bytes &&
        bytes <= part.size_bytes - (offset - part.byte_offset))
      return {index, static_cast<std::size_t>(offset - part.byte_offset),
              static_cast<std::size_t>(bytes)};
  }
  throw std::invalid_argument("memory placement: carrier straddles allocations");
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

void pack_stage_outputs(StageStaticSpec& stage, const StageCarrierAccess& consumers,
                        bool terminal_host_candidate) {
  const auto& recipe = *stage.native_kernel;
  auto required_access = consumers.outputs;
  const auto physical_id = [&](std::size_t tensor) {
    const auto logical = std::find_if(
        stage.logical_outputs.begin(), stage.logical_outputs.end(),
        [&](const auto& value) { return value.backend_output_index == static_cast<int>(tensor); });
    return logical == stage.logical_outputs.end() ? static_cast<int>(tensor)
                                                  : logical->physical_index;
  };
  const auto count = recipe.members.size();
  std::vector<std::vector<bool>> before(count + 1, std::vector<bool>(count + 1));
  std::unordered_map<std::uint64_t, std::vector<std::size_t>> accesses;
  bool all_members_cpu = true;
  for (std::size_t index = 0; index < count; ++index) {
    const auto& member = recipe.members[index];
    for (const auto dependency : member.dependencies) {
      if (dependency >= index)
        throw std::invalid_argument("memory placement: non-topological numerical dependency");
      before[dependency][index] = true;
      for (std::size_t ancestor = 0; ancestor < dependency; ++ancestor)
        if (before[ancestor][dependency])
          before[ancestor][index] = true;
    }
    before[index][count] = true;
    const auto inspected = client::inspect_kernel(member.executable);
    if (const auto* error = std::get_if<client::Error>(&inspected))
      throw std::invalid_argument("memory placement: " + error->message);
    const auto& checked = std::get<client::KernelInspection>(inspected);
    all_members_cpu = all_members_cpu && checked.target() == client::Hardware::A65;
    const auto member_access = static_cast<std::uint32_t>(checked.target() == client::Hardware::EV74
                                                              ? sc::ArenaDeviceAccess::Ev74
                                                              : sc::ArenaDeviceAccess::CpuA65);
    for (std::size_t port = 0; port < member.outputs.size(); ++port) {
      const auto& output = member.outputs[port];
      const auto output_id = physical_id(output.tensor);
      const auto physical =
          std::find_if(stage.physical_outputs.begin(), stage.physical_outputs.end(),
                       [&](const auto& value) { return value.physical_index == output_id; });
      if (physical == stage.physical_outputs.end() || port >= checked.output_requirements().size())
        throw std::invalid_argument("memory placement: numerical output has no carrier");
      const auto& requirement = checked.output_requirements()[port];
      if (output.byte_offset > physical->size_bytes ||
          requirement.storage_bytes > physical->size_bytes - output.byte_offset)
        throw std::invalid_argument("memory placement: checked output exceeds its carrier");
      physical->required_alignment_bytes =
          std::max<std::uint64_t>(physical->required_alignment_bytes, requirement.alignment_bytes);
      accesses[static_cast<std::uint64_t>(output_id)].push_back(index);
      required_access[output_id] |= member_access;
    }
    for (const auto& input : member.inputs)
      if (input.source == client::TensorStagePortSource::Output) {
        const auto input_id = physical_id(input.tensor);
        accesses[static_cast<std::uint64_t>(input_id)].push_back(index);
        required_access[input_id] |= member_access;
      }
  }
  for (const auto& logical : stage.logical_outputs) {
    accesses[static_cast<std::uint64_t>(logical.physical_index)].push_back(count);
    required_access[logical.physical_index] |=
        static_cast<std::uint32_t>(sc::ArenaDeviceAccess::CpuA65);
  }
  const auto cpu_access = static_cast<std::uint32_t>(sc::ArenaDeviceAccess::CpuA65);
  // The checked member/carrier pass above already supplies access and lifetime
  // facts. Only a fresh, unconsumed public CPU endpoint may choose host storage;
  // Scheduler independently proves the actual owned stage-to-appsink edge.
  const bool terminal_host =
      terminal_host_candidate && stage.core_fresh_default_arena && !consumers.shared_allocation &&
      !consumers.has_consumers &&
      stage.frame_arena_allocation_policy == sc::ArenaAllocationPolicy::RequiredDomain &&
      stage.frame_arena_escape_policy == sc::ArenaEscapePolicy::CpuMappablePublic &&
      recipe.target == client::Hardware::A65 && all_members_cpu && !stage.image_publication &&
      !recipe.image_publication && !recipe.frame_adapter_factory &&
      !stage.logical_outputs.empty() && !stage.output_order.empty() &&
      std::all_of(stage.logical_outputs.begin(), stage.logical_outputs.end(), dense_publication) &&
      std::all_of(stage.physical_outputs.begin(), stage.physical_outputs.end(),
                  [&](const auto& output) {
                    return output.source_physical_index == -1 &&
                           output.device_kind == DeviceKind::Cpu && !output.memory_flags &&
                           output.address_source == PhysicalAddressSource::DirectCarrierSpan &&
                           output.required_alignment_bytes <= 64 &&
                           required_access[output.physical_index] == cpu_access;
                  });
  std::vector<sc::CarrierArenaRegion> regions;
  for (const auto& physical : stage.physical_outputs) {
    const auto& used = accesses[static_cast<std::uint64_t>(physical.physical_index)];
    if (used.empty())
      throw std::invalid_argument("memory placement: output carrier has no checked access");
    const bool published = std::any_of(
        stage.logical_outputs.begin(), stage.logical_outputs.end(),
        [&](const auto& logical) { return logical.physical_index == physical.physical_index; });
    auto access = required_access[physical.physical_index];
    // Reused envelopes keep the already shared allocation's access contract.
    if (consumers.shared_allocation)
      access |= stage.frame_arena_required_device_access;
    const auto domain = terminal_host ? sc::ArenaStorageDomain::HostCpu
                        : (access & static_cast<std::uint32_t>(sc::ArenaDeviceAccess::Ev74))
                            ? sc::ArenaStorageDomain::Cma
                            : sc::ArenaStorageDomain::Dms;
    regions.push_back(
        {static_cast<std::uint64_t>(physical.physical_index),
         0,
         physical.size_bytes,
         std::max<std::uint64_t>(64, physical.required_alignment_bytes),
         {*std::min_element(used.begin(), used.end()), *std::max_element(used.begin(), used.end())},
         domain,
         access,
         published});
  }
  const auto conflict = [&](const auto& lhs, const auto& rhs) {
    const auto all_before = [&](const auto& first, const auto& second) {
      for (const auto left : first)
        for (const auto right : second)
          if (!before[left][right])
            return false;
      return true;
    };
    return !all_before(accesses[lhs.carrier_id], accesses[rhs.carrier_id]) &&
           !all_before(accesses[rhs.carrier_id], accesses[lhs.carrier_id]);
  };
  std::string error;
  auto packed = sc::pack_carrier_arena(
      std::move(regions),
      recipe.image_publication ? sc::FrameSlotArenaReuse::Disabled
                               : sc::FrameSlotArenaReuse::DisjointLifetimes,
      conflict,
      stage.frame_arena_allocation_policy == sc::ArenaAllocationPolicy::PreferDmsThenCma
          ? sc::ArenaDmsPolicy::PreferDmsThenCma
          : sc::ArenaDmsPolicy::RequireDmsForEligible,
      &error);
  if (!packed)
    throw std::invalid_argument("memory placement: " + error);
  // Preserve the pool's page-sized allocation envelope; carriers keep exact extents.
  auto& tail = packed->allocations.back();
  const auto rounded = (tail.size_bytes + 4095U) & ~std::uint64_t{4095U};
  packed->allocation_bytes += rounded - tail.size_bytes;
  tail.size_bytes = rounded;
  for (auto& physical : stage.physical_outputs) {
    const auto region =
        std::find_if(packed->regions.begin(), packed->regions.end(), [&](const auto& value) {
          return value.carrier_id == static_cast<std::uint64_t>(physical.physical_index);
        });
    physical.source_byte_offset = static_cast<std::int64_t>(region->byte_offset);
  }
  stage.frame_arena_size_bytes = packed->allocation_bytes;
  stage.frame_arena_required_device_access = 0;
  stage.frame_arena_storage_domain = packed->allocations.front().placement.domain;
  for (const auto& allocation : packed->allocations) {
    stage.frame_arena_required_device_access |= allocation.placement.required_device_access;
    if (stage.frame_arena_storage_domain != allocation.placement.domain)
      stage.frame_arena_storage_domain = sc::ArenaStorageDomain::Mixed;
  }
  stage.frame_arena_allocations = std::move(packed->allocations);
  if (terminal_host)
    stage.frame_arena_escape_policy = sc::ArenaEscapePolicy::CpuOnlyPublic;
}
} // namespace

void finalize_pipeline_storage(
    SimaPluginStaticManifest& manifest,
    const terminal_output_contract::PublicOutputEndpointSelector* terminal_host) {
  internal::project_model_fragment_storage(manifest);
  std::vector<bool> unplaced(manifest.stages.size());
  for (std::size_t index = 0; index < manifest.stages.size(); ++index) {
    auto& stage = manifest.stages[index];
    unplaced[index] = stage.native_kernel && !stage.model_storage_planned &&
                      stage.frame_arena_role != FrameArenaRole::ReuseInput;
    if (unplaced[index]) {
      stage.frame_arena_size_bytes = 0;
      stage.frame_arena_allocations.clear();
      stage.frame_arena_role = FrameArenaRole::Allocate;
      stage.frame_arena_provenance = sc::ArenaAllocationProvenance::CoreAllocated;
    }
  }
  const auto carrier_access = resolve_memory_placement(manifest);
  const auto* terminal = terminal_host_candidate(manifest, terminal_host, carrier_access);
  for (std::size_t index = 0; index < manifest.stages.size(); ++index) {
    auto& stage = manifest.stages[index];
    if (unplaced[index])
      pack_stage_outputs(stage, carrier_access[index], &stage == terminal);
    if (!stage.frame_arena_size_bytes)
      continue;
    auto layout = std::make_shared<client::FrameLayout>();
    layout->ownership = stage.frame_arena_role == FrameArenaRole::ReuseInput
                            ? client::FrameOwnership::ReuseInput
                            : client::FrameOwnership::Allocate;
    if (stage.frame_arena_allocations.empty()) {
      sc::ArenaPlacement placement;
      placement.domain = stage.frame_arena_storage_domain;
      placement.provenance = stage.frame_arena_provenance;
      placement.allocation_policy = stage.frame_arena_allocation_policy;
      layout->allocations.push_back({static_cast<std::size_t>(stage.frame_arena_size_bytes),
                                     stage.frame_arena_required_device_access,
                                     allocation_policy(placement)});
    } else {
      for (const auto& part : stage.frame_arena_allocations)
        layout->allocations.push_back({static_cast<std::size_t>(part.size_bytes),
                                       part.placement.required_device_access,
                                       allocation_policy(part.placement)});
    }
    const auto add_output = [&](std::size_t tensor, int physical_id) {
      if (std::any_of(layout->outputs.begin(), layout->outputs.end(),
                      [&](const auto& value) { return value.tensor == tensor; }))
        return;
      const auto output =
          std::find_if(stage.physical_outputs.begin(), stage.physical_outputs.end(),
                       [&](const auto& value) { return value.physical_index == physical_id; });
      if (output == stage.physical_outputs.end() || output->source_byte_offset < 0)
        throw std::invalid_argument("memory placement: invalid output identity or offset");
      auto offset = static_cast<std::uint64_t>(output->source_byte_offset);
      auto bytes = output->size_bytes;
      if (stage.native_kernel && stage.native_kernel->image_publication) {
        const auto plane = std::find_if(
            stage.logical_outputs.begin(), stage.logical_outputs.end(), [&](const auto& value) {
              return value.backend_output_index == static_cast<int>(tensor);
            });
        if (plane == stage.logical_outputs.end() || plane->byte_offset < 0 ||
            std::uint64_t(plane->byte_offset) > bytes ||
            plane->size_bytes > bytes - std::uint64_t(plane->byte_offset))
          throw std::invalid_argument("memory placement: image plane exceeds its carrier");
        offset += std::uint64_t(plane->byte_offset);
        bytes = plane->size_bytes;
      }
      layout->outputs.push_back({tensor, allocation_span(stage, offset, bytes)});
    };
    if (stage.native_kernel) {
      for (const auto& member : stage.native_kernel->members)
        for (const auto& port : member.outputs) {
          const auto logical = std::find_if(
              stage.logical_outputs.begin(), stage.logical_outputs.end(), [&](const auto& value) {
                return value.backend_output_index == static_cast<int>(port.tensor);
              });
          add_output(port.tensor, logical == stage.logical_outputs.end()
                                      ? static_cast<int>(port.tensor)
                                      : logical->physical_index);
        }
    } else {
      for (const auto& output : stage.physical_outputs)
        add_output(static_cast<std::size_t>(output.physical_index), output.physical_index);
    }
    if (!stage.native_kernel || !stage.native_kernel->frame_adapter_factory)
      for (const auto& logical : stage.logical_inputs) {
        if (logical.backend_input_index < 0 || logical.byte_offset < 0)
          throw std::invalid_argument("memory placement: invalid input identity or view offset");
        const auto binding = std::find_if(
            stage.input_bindings.begin(), stage.input_bindings.end(), [&](const auto& value) {
              return value.local_logical_input_index == logical.logical_index;
            });
        const auto physical = std::find_if(
            stage.physical_inputs.begin(), stage.physical_inputs.end(),
            [&](const auto& value) { return value.physical_index == logical.physical_index; });
        if (binding == stage.input_bindings.end() || physical == stage.physical_inputs.end() ||
            binding->sink_pad_index < 0 || physical->source_byte_offset < 0 ||
            binding->src_physical_byte_offset < 0)
          throw std::invalid_argument("memory placement: incomplete physical input route");
        const bool root = binding->src_stage_index < 0 && binding->src_stage_id.empty() &&
                          binding->src_logical_output_index < 0 && binding->src_output_slot < 0 &&
                          binding->src_physical_output_index < 0 &&
                          !binding->src_physical_size_bytes && !binding->src_physical_byte_offset;
        const auto carrier = root ? physical->physical_index : binding->src_physical_output_index;
        // Model inputs already carry the final physical-DAG span. Edge
        // selectors identify its owner; they do not supply a second offset.
        const auto capacity = stage.model_storage_planned || root
                                  ? physical->size_bytes
                                  : binding->src_physical_size_bytes;
        const auto relative = stage.model_storage_planned
                                  ? std::uint64_t{0}
                                  : static_cast<std::uint64_t>(binding->src_physical_byte_offset);
        if (carrier < 0 || relative > physical->size_bytes || !capacity ||
            capacity > physical->size_bytes - relative)
          throw std::invalid_argument("memory placement: contradictory input carrier extent");
        const auto pad = static_cast<std::size_t>(binding->sink_pad_index);
        const auto offset = static_cast<std::uint64_t>(physical->source_byte_offset) + relative;
        client::FrameInputBinding input;
        input.view_offset = static_cast<std::size_t>(logical.byte_offset);
        switch (physical->address_source) {
        case PhysicalAddressSource::RuntimePhysicalBinding:
          input.source = client::PhysicalBinding{pad, static_cast<std::size_t>(carrier),
                                                 static_cast<std::size_t>(offset),
                                                 static_cast<std::size_t>(capacity)};
          break;
        case PhysicalAddressSource::DirectCarrierSpan:
          input.source = client::DirectMemory{pad, static_cast<std::size_t>(carrier),
                                              static_cast<std::size_t>(offset),
                                              static_cast<std::size_t>(capacity)};
          break;
        case PhysicalAddressSource::FrameArenaSpan: {
          std::string error;
          const auto producer =
              edgecontract::resolve_producer_stage_exact(manifest, index, *binding, &error);
          if (!error.empty())
            throw std::invalid_argument("memory placement: " + error);
          const auto* owner = producer ? &manifest.stages[*producer]
                              : layout->ownership == client::FrameOwnership::ReuseInput ? &stage
                                                                                        : nullptr;
          if (!owner)
            throw std::invalid_argument("memory placement: unresolved input arena owner");
          auto producer_offset = offset;
          if (producer && !stage.model_storage_planned) {
            auto selected_binding = *binding;
            selected_binding.src_stage_index = static_cast<int>(*producer);
            selected_binding.src_stage_id.clear();
            const auto edge = edgecontract::resolve_edge_contract_for_binding(
                manifest, index, selected_binding, &error);
            if (!edge || edge->producer_physical_output->source_byte_offset < 0)
              throw std::invalid_argument("memory placement: " + error);
            const auto& source = *edge->producer_physical_output;
            if (relative > source.size_bytes || capacity > source.size_bytes - relative)
              throw std::invalid_argument("memory placement: input exceeds its producer carrier");
            producer_offset = static_cast<std::uint64_t>(source.source_byte_offset) + relative;
          }
          input.source =
              client::KnownArena{pad, allocation_span(*owner, producer_offset, capacity)};
          if (layout->ownership == client::FrameOwnership::ReuseInput)
            layout->source_pad = pad;
          break;
        }
        }
        const auto slot = static_cast<std::size_t>(logical.backend_input_index);
        if (slot >= layout->inputs.size())
          layout->inputs.resize(slot + 1);
        layout->inputs[slot] = std::move(input);
      }
    stage.frame_layout = std::move(layout);
  }
}

} // namespace simaai::neat::pipeline_internal::sima

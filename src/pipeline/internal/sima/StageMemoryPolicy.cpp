#include "pipeline/internal/sima/StageMemoryPolicy.h"
#include "pipeline/internal/sima/InternalEdgeContractResolver.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/ProcessCvuRunTargetPolicy.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemanticsInternal.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace simaai::neat::pipeline_internal::sima {
namespace {
namespace sc = static_contract;

std::uint32_t execution_access(const StageStaticSpec& stage) {
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

} // namespace

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

void resolve_stage_memory_placement(SimaPluginStaticManifest& manifest) {
  const auto count = manifest.stages.size();
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
    }
  }
  std::vector<std::uint32_t> access(count, 0U);
  for (std::size_t index = 0; index < count; ++index) {
    const auto required = execution_access(manifest.stages[index]);
    access[find_allocation_root(index)] |= required;
    for (const auto producer : producers[index])
      access[find_allocation_root(producer)] |= required;
  }
  for (std::size_t index = 0; index < count; ++index) {
    auto& stage = manifest.stages[index];
    if (!stage.frame_arena_size_bytes ||
        stage.frame_arena_provenance != sc::ArenaAllocationProvenance::CoreAllocated)
      continue;
    if (!stage.frame_arena_allocations.empty()) {
      // Model carrier placement was resolved before packing. Validate each
      // actual numerical access without collapsing the separate heaps.
      if (stage.native_kernel || stage.payload_kind == StagePayloadKind::ProcessCvu ||
          stage.payload_kind == StagePayloadKind::ProcessMla) {
        const auto required = execution_access(stage);
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
    const auto required = access[find_allocation_root(index)];
    if (!required)
      throw std::invalid_argument("memory placement requires an allocation access contract");
    const bool ev74 = required & static_cast<std::uint32_t>(sc::ArenaDeviceAccess::Ev74);
    if (ev74 && stage.frame_arena_allocation_policy == sc::ArenaAllocationPolicy::PreferDmsThenCma)
      throw std::invalid_argument("memory placement: DMS fallback cannot authorize EV74 access");
    stage.frame_arena_storage_domain =
        ev74 ? sc::ArenaStorageDomain::Cma : sc::ArenaStorageDomain::Dms;
    stage.frame_arena_required_device_access = required;
  }
}

} // namespace simaai::neat::pipeline_internal::sima

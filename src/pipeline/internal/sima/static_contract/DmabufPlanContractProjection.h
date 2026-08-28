#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/MlaStaticContractExtractor.h"
#include "pipeline/internal/contract/PluginCompiledContracts.h"
#include "pipeline/internal/sima/static_contract/ModelExecutionPlan.h"
#include "pipeline/internal/sima/static_contract/FrameSlotArenaPlan.h"
#include "pipeline/internal/sima/static_contract/PhysicalExecutionPlan.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::static_contract {

// One setup-time lowering result. `value_id` is the exact graph value consumed
// by an MLA IFM port and `source_physical_index` selects the canonical
// materialized carrier published by the immediately preceding stage. Names
// and logical tensors are used only to prove this relation at setup.
struct PhysicalPortSource {
  ValueId value_id = 0;
  int source_physical_index = -1;
};

// A stage normally addresses its IFMs and OFMs in the graph-wide frame arena.
// When a terminal MLA's complete physical output is a public CPU boundary and
// an upstream internal carrier must remain live, the MLA owns a separate,
// right-sized output carrier instead.  A public-input-only terminal MLA keeps
// its sole already-compact DMS arena; there is nothing useful to detach from.
// The policy is selected from the physical command DAG and model-output
// ValueIds; runtime element names and model families are deliberately not
// inputs to the decision.
enum class MlaOutputCarrierPolicy : std::uint8_t {
  SharedFrameArena = 0,
  SeparateCpuVisible = 1,
};

[[nodiscard]] MlaOutputCarrierPolicy select_mla_output_carrier_policy(
    const ModelExecutionPlan& plan, const PhysicalExecutionPlan& physical_plan,
    std::size_t mla_stage_index);

// Complete materialized roots removed from the shared FrameSlotArena because
// the exact terminal policy above assigns them to a stage-owned output pool.
// This is the sole bridge between topology selection and arena filtering.
[[nodiscard]] std::vector<ValueId> detached_mla_output_roots(
    const ModelExecutionPlan& plan, const PhysicalExecutionPlan& physical_plan);

// Read the already-compiled placement decision; no topology is re-inferred
// while rendering stage contracts.
[[nodiscard]] MlaOutputCarrierPolicy mla_output_carrier_policy_from_arena(
    const ModelExecutionPlan& plan, const FrameSlotArenaPlan& arena,
    std::size_t mla_stage_index);

// Resolve MLA input ValueIds to physical roots. A direct value maps to the
// exact upstream physical carrier. A Pack value maps to one parent only when
// its explicit component placement and the producer's logical views prove
// exact ordered coverage. No logical child is used as a runtime address
// anchor.
std::optional<std::vector<PhysicalPortSource>>
resolve_mla_input_physical_sources(const ModelExecutionPlan& plan, std::size_t mla_stage_index,
                                   std::span<const LogicalTensorStaticSpec> upstream_outputs,
                                   std::string* error = nullptr);

// Arena-aware multi-stage entry point. Any materialized internal IFM is read
// as an offset view of the one retained frame-arena DMA-BUF; several MLA IFM
// ports may therefore bind the same source memory at different byte offsets.
std::optional<std::vector<PhysicalPortSource>>
resolve_mla_input_physical_sources(const ModelExecutionPlan& plan, std::size_t mla_stage_index,
                                   const FrameSlotArenaPlan& arena,
                                   std::span<const LogicalTensorStaticSpec> upstream_outputs,
                                   std::string* error = nullptr);

// Source-compatible single-stage entry point. It rejects an ambiguous plan.
std::optional<std::vector<PhysicalPortSource>>
resolve_mla_input_physical_sources(const ModelExecutionPlan& plan,
                                   std::span<const LogicalTensorStaticSpec> upstream_outputs,
                                   std::string* error = nullptr);

// Prove that the existing manifest projection is byte-for-byte the MLA port
// boundary decoded from MPK+ELF. On success, install the exact ELF symbols,
// ordered physical-carrier IFM sources, raw stage-local OFM publication and
// multi-IFM policy; on failure leave the contract unusable by dmabuf-plan.
// `output_carrier_policy` must equal the policy already authored by `arena`;
// the explicit form remains available for low-level fail-closed tests only.
bool apply_dmabuf_plan_contract_projection(const ModelExecutionPlan& plan,
                                           std::size_t mla_stage_index,
                                           const FrameSlotArenaPlan& arena,
                                           MlaOutputCarrierPolicy output_carrier_policy,
                                           MlaStaticContract* contract,
                                           std::span<const PhysicalPortSource> input_sources,
                                           std::string* error = nullptr);

bool apply_dmabuf_plan_contract_projection(const ModelExecutionPlan& plan,
                                           std::size_t mla_stage_index,
                                           const FrameSlotArenaPlan& arena,
                                           MlaStaticContract* contract,
                                           std::span<const PhysicalPortSource> input_sources,
                                           std::string* error = nullptr);

bool apply_dmabuf_plan_contract_projection(const ModelExecutionPlan& plan,
                                           std::size_t mla_stage_index, MlaStaticContract* contract,
                                           std::span<const PhysicalPortSource> input_sources,
                                           std::string* error = nullptr);

bool apply_dmabuf_plan_contract_projection(const ModelExecutionPlan& plan,
                                           MlaStaticContract* contract,
                                           std::span<const PhysicalPortSource> input_sources,
                                           std::string* error = nullptr);

enum class ProcessCvuMlaBoundary : std::uint8_t {
  Inputs,
  Outputs,
};

// Select the production CVU kernel ABI and project the exact MLA-boundary
// alignment into every materialized CVU output carrier. Pre-MLA transforms
// inherit the IFM-port contract; post-MLA transforms inherit the OFM-port
// contract. A one-port packed boundary may fan out into multiple CVU output
// regions, in which case the one authoritative alignment applies to each.
bool apply_dmabuf_plan_processcvu_contract_projection(
    const ModelExecutionPlan& plan, std::size_t adjacent_mla_stage_index,
    const FrameSlotArenaPlan& arena, ProcessCvuMlaBoundary boundary,
    ProcessCvuStagePayload* payload, ::simaai::neat::CompiledRuntimeContract* runtime,
    ::simaai::neat::CompiledExposedView* exposed_view, std::string* error = nullptr);

// Project one physical CVU cohort without reasoning from an adjacent MLA
// position. `command_ids` are the bounded capacity chunks of one cohort. Each
// member contributes its complete typed semantic chain but only its outer
// input/output values become runtime bindings; fused intermediate values are
// intentionally absent. This is the command-authoritative path for first,
// middle, last, CVU-only, branched, and grouped schedules.
std::optional<CompiledProcessCvuContract> build_dmabuf_plan_processcvu_command_contract(
    const ModelExecutionPlan& plan, const PhysicalExecutionPlan& physical_plan,
    std::span<const PhysicalCommandId> command_ids, const FrameSlotArenaPlan& arena,
    std::string* error = nullptr);

bool apply_dmabuf_plan_processcvu_command_projection(
    const ModelExecutionPlan& plan, const PhysicalExecutionPlan& physical_plan,
    std::span<const PhysicalCommandId> command_ids,
    const FrameSlotArenaPlan& arena, ProcessCvuStagePayload* payload,
    ::simaai::neat::CompiledRuntimeContract* runtime,
    ::simaai::neat::CompiledExposedView* exposed_view, std::string* error = nullptr);

// Resolve the exact linear compiler-authored ingress transform prefix that a
// model-managed graph-200 image preprocessor may replace.  Every semantic op
// must belong to one strict CVU Ingress command, intermediate values must not
// escape, and the prefix must terminate at the first MLA's sole IFM.  The
// returned command ids are the only physical stages that the compatibility
// renderer may omit; an empty/ambiguous/multi-IFM route fails closed.
std::optional<std::vector<PhysicalCommandId>>
resolve_model_managed_preproc_ingress_commands(
    const ModelExecutionPlan& plan, const PhysicalExecutionPlan& physical_plan,
    std::string* error = nullptr);

// Project an already-compiled graph-200 contract onto the exact output carrier
// of the absorbed compiler ingress prefix.  This reuses the one FrameSlotArena
// and the existing MLA-boundary projection; it neither invents a second arena
// contract nor relaxes graph-200/MLA tensor semantics.
bool project_model_managed_preproc_contract(
    const ModelExecutionPlan& plan, const PhysicalExecutionPlan& physical_plan,
    const FrameSlotArenaPlan& arena, ::simaai::neat::CompiledProcessCvuContract* contract,
    std::vector<PhysicalCommandId>* absorbed_command_ids,
    std::string* error = nullptr);

bool apply_dmabuf_plan_processcvu_contract_projection(
    const ModelExecutionPlan& plan, std::size_t adjacent_mla_stage_index,
    ProcessCvuMlaBoundary boundary, ProcessCvuStagePayload* payload,
    ::simaai::neat::CompiledRuntimeContract* runtime,
    ::simaai::neat::CompiledExposedView* exposed_view, std::string* error = nullptr);

bool apply_dmabuf_plan_processcvu_contract_projection(
    const ModelExecutionPlan& plan, ProcessCvuMlaBoundary boundary, ProcessCvuStagePayload* payload,
    ::simaai::neat::CompiledRuntimeContract* runtime,
    ::simaai::neat::CompiledExposedView* exposed_view, std::string* error = nullptr);

} // namespace simaai::neat::pipeline_internal::sima::static_contract

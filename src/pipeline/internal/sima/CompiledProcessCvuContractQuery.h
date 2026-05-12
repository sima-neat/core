/**
 * @file
 * @ingroup internal_sima
 * @brief **Framework internal — reach-through tier.** Query helpers over compiled process-CVU
 *        contracts.
 *
 * The route planner uses these helpers to reason about how a compiled process-CVU stage hands
 * off to a downstream MLA stage: whether the upstream output is packed or dense, how to rewrite
 * the MLA's logical input from the upstream view, and how to apply a strict single-output
 * handoff. These are header-only inline helpers; the planner pulls them in directly.
 *
 * @see CompiledProcessCvuContract
 * @see MlaStaticContract
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/contract/PluginCompiledContracts.h"
#include "pipeline/internal/sima/MlaStaticContractExtractor.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace simaai::neat::pipeline_internal::sima {

/// Returns the `cm_output_name` if non-empty, else the `segment_name`.
inline std::string output_name_from_route(const StageOutputRoute& route) {
  return !route.cm_output_name.empty() ? route.cm_output_name : route.segment_name;
}

/**
 * @brief True when the upstream process-CVU contract's primary output uses packed transport.
 *
 * Tessellated and quant-tess outputs are intrinsically packed; an explicit `Packed` transport
 * flag also forces this path. Used by the MLA handoff rewriter to decide whether to preserve
 * the IFM envelope (packed) or to rewrite from the dense exposed view.
 */
inline bool processcvu_contract_primary_output_uses_packed_transport(
    const CompiledProcessCvuContract& contract) {
  return contract.payload.primary_output_transport_kind == ProcessCvuOutputTransportKind::Packed ||
         contract.payload.primary_output_semantic_kind ==
             ProcessCvuOutputSemanticKind::TessellatedImage ||
         contract.payload.primary_output_semantic_kind ==
             ProcessCvuOutputSemanticKind::QuantTessTensor;
}

/**
 * @brief Adjust an MLA contract's logical input to match a packed upstream handoff.
 *
 * When the upstream process-CVU emits packed transport, the MLA must read the entire physical
 * envelope as-is. This helper widens the MLA's `max_stride` to the physical size while leaving
 * shape/dtype unchanged. Returns false if the contract cannot be normalized (e.g., empty
 * inputs, mismatched sizes).
 */
inline bool preserve_mla_input_for_packed_handoff(MlaStaticContract* mla_contract) {
  if (!mla_contract || mla_contract->logical_inputs.empty() ||
      mla_contract->physical_inputs.empty()) {
    return false;
  }

  auto& logical_input = mla_contract->logical_inputs.front();
  const auto& physical_input = mla_contract->physical_inputs.front();
  if (logical_input.shape.empty() || logical_input.dtype.empty()) {
    return false;
  }

  const std::uint64_t logical_size_bytes =
      specbuilders::tensor_size_bytes_from_shape_dtype(logical_input.shape, logical_input.dtype);
  if (logical_size_bytes == 0U || physical_input.size_bytes < logical_size_bytes) {
    return false;
  }

  logical_input.max_stride = static_cast<int>(std::min<std::uint64_t>(
      physical_input.size_bytes, static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
  return true;
}

/**
 * @brief Rewrite the MLA's logical input from an upstream dense process-CVU exposed view.
 *
 * Copies shape / dtype / layout (normalized) and computes max_h / max_w / max_stride from the
 * exposed logical output. Used when the upstream contract's primary output is dense (not
 * packed). Returns false when the exposed view lacks the required fields.
 */
inline bool rewrite_mla_input_from_processcvu_dense_handoff(
    const CompiledProcessCvuContract& upstream_handoff_contract, MlaStaticContract* mla_contract) {
  if (!mla_contract || mla_contract->physical_inputs.empty() ||
      upstream_handoff_contract.exposed_view.exposed_logical_outputs.empty()) {
    return false;
  }

  const auto& exposed = upstream_handoff_contract.exposed_view.exposed_logical_outputs.front();
  if (exposed.shape.empty() || exposed.dtype.empty()) {
    return false;
  }

  if (mla_contract->logical_inputs.empty()) {
    mla_contract->logical_inputs.resize(1U);
  }

  auto& logical = mla_contract->logical_inputs.front();
  const auto& physical = mla_contract->physical_inputs.front();
  logical.tensor_index = 0;
  logical.shape = exposed.shape;
  logical.dtype = exposed.dtype;
  logical.layout = !exposed.layout.empty()
                       ? tensorsemantics::normalize_layout_token(exposed.layout)
                       : upstream_handoff_contract.payload.logical_output_layout_token();
  logical.max_stride = static_cast<int>(
      std::min<std::uint64_t>(exposed.size_bytes > 0U ? exposed.size_bytes : physical.size_bytes,
                              static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
  if (exposed.shape.size() >= 3U) {
    logical.max_h =
        static_cast<int>(std::max<std::int64_t>(0, exposed.shape[exposed.shape.size() - 3U]));
    logical.max_w =
        static_cast<int>(std::max<std::int64_t>(0, exposed.shape[exposed.shape.size() - 2U]));
  } else if (exposed.shape.size() == 2U) {
    logical.max_h = static_cast<int>(std::max<std::int64_t>(0, exposed.shape[0]));
    logical.max_w = static_cast<int>(std::max<std::int64_t>(0, exposed.shape[1]));
  } else if (exposed.shape.size() == 1U) {
    logical.max_w = static_cast<int>(std::max<std::int64_t>(0, exposed.shape.front()));
    logical.max_h = 1;
  }
  logical.semantic_tag =
      !exposed.logical_name.empty()
          ? exposed.logical_name
          : (!physical.segment_name.empty() ? physical.segment_name : std::string("mla_input_0"));
  return true;
}

/**
 * @brief Resolved single-output handoff facts for a process-CVU stage.
 *
 * Populated from a strict-handoff contract; carries the segment / slot / physical-index /
 * offset / size needed to bind a downstream MLA's first input.
 */
struct ProcessCvuSingleHandoffOutput {
  std::string segment_name;      ///< Source segment name on the upstream stage.
  int logical_output_index = -1; ///< Upstream logical-output index.
  int output_slot = -1;          ///< Upstream output slot.
  int physical_index = -1;       ///< Upstream physical-output index.
  std::int64_t byte_offset = 0;  ///< Byte offset within the upstream physical buffer.
  std::uint64_t size_bytes = 0U; ///< Size in bytes of the handoff slice.
};

/**
 * @brief Resolve the strict single-output handoff for a process-CVU contract.
 *
 * Returns the handoff facts when the contract advertises an explicit single-output handoff or
 * has exactly one exposed output. Returns nullopt when neither condition holds. Throws
 * `std::runtime_error` when the contract is malformed (e.g., missing primary output name in
 * a strict-handoff configuration).
 */
inline std::optional<ProcessCvuSingleHandoffOutput>
resolve_processcvu_single_handoff_output(const CompiledProcessCvuContract& contract) {
  const bool explicit_single_handoff = contract.preproc_single_output_handoff;
  const bool single_exposed_output = contract.exposed_view.exposed_output_order.size() == 1U &&
                                     contract.exposed_view.exposed_logical_outputs.size() == 1U;
  if (!explicit_single_handoff && !single_exposed_output) {
    return std::nullopt;
  }
  if (contract.exposed_view.primary_output_name.empty() && !single_exposed_output) {
    throw std::runtime_error(
        "ModelFragment: strict processcvu handoff missing exposed primary output name");
  }
  if (!single_exposed_output) {
    throw std::runtime_error(
        "ModelFragment: strict processcvu handoff requires exactly one exposed output");
  }

  const auto& route = contract.exposed_view.exposed_output_order.front();
  const auto& logical = contract.exposed_view.exposed_logical_outputs.front();
  const std::string route_name = output_name_from_route(route);
  const std::string primary_output_name =
      !contract.exposed_view.primary_output_name.empty()
          ? contract.exposed_view.primary_output_name
          : (!route_name.empty() ? route_name : logical.segment_name);
  if (!route_name.empty() && !primary_output_name.empty() && route_name != primary_output_name) {
    throw std::runtime_error(
        "ModelFragment: strict processcvu handoff primary output does not match exposed route");
  }

  const std::string segment_name =
      !route.segment_name.empty()
          ? route.segment_name
          : (!logical.segment_name.empty() ? logical.segment_name : primary_output_name);
  if (segment_name.empty()) {
    throw std::runtime_error(
        "ModelFragment: strict processcvu handoff missing upstream segment name");
  }

  ProcessCvuSingleHandoffOutput handoff;
  handoff.segment_name = segment_name;
  handoff.logical_output_index =
      route.logical_output_index >= 0 ? route.logical_output_index : logical.logical_index;
  handoff.output_slot = route.output_slot >= 0 ? route.output_slot : logical.output_slot;
  handoff.physical_index = logical.physical_index;
  handoff.byte_offset = logical.byte_offset;
  if (logical.physical_index >= 0 && static_cast<std::size_t>(logical.physical_index) <
                                         contract.runtime_contract.physical_outputs.size()) {
    handoff.size_bytes =
        contract.runtime_contract.physical_outputs[static_cast<std::size_t>(logical.physical_index)]
            .size_bytes;
  }
  if (handoff.size_bytes == 0U) {
    handoff.size_bytes = logical.size_bytes;
  }
  return handoff;
}

/**
 * @brief Apply a strict process-CVU single-output handoff to an MLA contract.
 *
 * Rewrites the MLA's first physical input (segment, source physical index, size, offset) and
 * its first input binding to point at the upstream handoff target. For packed-transport
 * upstreams it preserves the IFM envelope; for dense upstreams it rewrites the logical view.
 * Throws `std::runtime_error` on contract violations (missing handoff facts, no physical
 * inputs, etc.).
 *
 * @param upstream_handoff_contract  Optional upstream contract; no-op when `std::nullopt`.
 * @param mla_contract               In/out MLA contract to rewrite.
 * @param stage_name                 Stage name used in error diagnostics.
 * @param upstream_stage_id          Source stage id stamped onto the rewritten input binding.
 */
inline void apply_processcvu_single_handoff_to_mla_contract(
    const std::optional<CompiledProcessCvuContract>& upstream_handoff_contract,
    MlaStaticContract* mla_contract, const std::string& stage_name,
    const std::string& upstream_stage_id = {}) {
  if (!mla_contract) {
    throw std::runtime_error(
        "ModelFragment: strict MLA handoff rewrite requires a valid MLA contract");
  }
  if (mla_contract->physical_inputs.empty()) {
    throw std::runtime_error(
        "ModelFragment: strict MLA contract missing physical inputs for stage '" + stage_name +
        "'");
  }
  if (!upstream_handoff_contract.has_value()) {
    return;
  }
  if (mla_contract->logical_inputs.size() > 1U) {
    return;
  }

  const auto handoff = resolve_processcvu_single_handoff_output(*upstream_handoff_contract);
  if (!handoff.has_value()) {
    throw std::runtime_error(
        "ModelFragment: strict processcvu handoff facts missing for MLA stage '" + stage_name +
        "'");
  }

  auto& mla_input = mla_contract->physical_inputs.front();
  mla_input.segment_name = handoff->segment_name;
  if (handoff->physical_index >= 0) {
    mla_input.source_physical_index = handoff->physical_index;
  }
  if (handoff->size_bytes > 0U) {
    mla_input.size_bytes = handoff->size_bytes;
  }
  mla_input.source_byte_offset = std::max<std::int64_t>(0, handoff->byte_offset);

  if (mla_contract->input_bindings.empty()) {
    mla_contract->input_bindings.resize(
        std::max(mla_contract->logical_inputs.size(), std::size_t{1}));
  }
  auto& binding = mla_contract->input_bindings.front();
  binding.local_logical_input_index = 0;
  binding.src_stage_id = upstream_stage_id;
  binding.src_logical_output_index = handoff->logical_output_index;
  binding.src_output_slot = handoff->output_slot;
  binding.src_physical_output_index =
      handoff->physical_index >= 0 ? handoff->physical_index : mla_input.source_physical_index;
  binding.src_physical_size_bytes =
      handoff->size_bytes > 0U ? handoff->size_bytes : mla_input.size_bytes;
  binding.src_physical_byte_offset = mla_input.source_byte_offset;
  binding.source_segment_name = handoff->segment_name;

  if (processcvu_contract_primary_output_uses_packed_transport(*upstream_handoff_contract)) {
    if (!preserve_mla_input_for_packed_handoff(mla_contract)) {
      throw std::runtime_error("ModelFragment: strict MLA handoff rewrite could not normalize the "
                               "MLA logical input contract for stage '" +
                               stage_name + "'");
    }
    return;
  }

  if (!rewrite_mla_input_from_processcvu_dense_handoff(*upstream_handoff_contract, mla_contract)) {
    throw std::runtime_error("ModelFragment: strict MLA handoff rewrite could not map the "
                             "dense upstream processcvu contract onto MLA stage '" +
                             stage_name + "'");
  }
}

} // namespace simaai::neat::pipeline_internal::sima

/**
 * @file
 * @ingroup internal_sima
 * @brief **Framework internal — reach-through tier.** Process-CVU stage family resolver.
 *
 * The process-CVU stages come in a small family of variants — Preproc, Quant, Tess, QuantTess,
 * CastTess, Detess, DetessCast, DetessDequant, Dequant — each with its own compiled graph.
 * The route planner uses these helpers to map a raw kernel name or an `ExecutionStageKind` to
 * the canonical family token used downstream.
 *
 * @see ProcessCvuGraphFamily (in SimaPluginStaticManifest.h)
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <cstdint>
#include <string>

namespace simaai::neat::internal {
enum class ExecutionStageKind : std::uint8_t;
}

namespace simaai::neat::pipeline_internal::sima {

/// Returns the canonical process-CVU family token (e.g., `"quanttess"`) for a raw kernel name.
std::string canonical_processcvu_family_from_kernel(std::string kernel);

/// Returns the canonical process-CVU family token for an internal stage-kind enum.
std::string processcvu_graph_family_for_stage_kind(
    ::simaai::neat::internal::ExecutionStageKind kind);

} // namespace simaai::neat::pipeline_internal::sima

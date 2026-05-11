/**
 * @file
 * @ingroup internal_sima
 * @brief **Framework internal — reach-through tier.** Renders compiled pipeline contracts into
 *        a `SimaPluginStaticManifest`.
 *
 * The contract render is the bridge between the planner's compiled contract set
 * (`CompiledPipelineContracts`) and the static manifest the SiMa GStreamer plugins consume.
 * Diagnostics produced during rendering are collected into the optional
 * `ManifestBuildDiagnostics` out-parameter so the caller can surface warnings/errors.
 *
 * @see SimaPluginStaticManifest
 * @see CompiledPipelineContracts
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include <optional>

namespace simaai::neat {

/**
 * @brief Render a `SimaPluginStaticManifest` from a fully compiled pipeline contract set.
 *
 * @param compiled       The set of compiled per-node contracts produced by the planner.
 * @param compile_input  Original compile-time input (session options, target board, etc.).
 * @param diagnostics    Optional out-parameter that collects warnings and errors during render.
 * @return The rendered manifest, or `std::nullopt` if a fatal error occurred (see `diagnostics`).
 */
std::optional<pipeline_internal::sima::SimaPluginStaticManifest>
render_manifest_from_compiled_contracts(const CompiledPipelineContracts& compiled,
                                        const ContractCompileInput& compile_input,
                                        pipeline_internal::sima::ManifestBuildDiagnostics* diagnostics);

} // namespace simaai::neat

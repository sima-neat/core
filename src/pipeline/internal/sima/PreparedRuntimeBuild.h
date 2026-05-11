/**
 * @file
 * @ingroup internal_sima
 * @brief **Framework internal — reach-through tier.** Builder for the "prepared runtime" handed
 *        off to the SiMa GStreamer plugins.
 *
 * Composes the transformed static manifest, original (pre-transform) manifest, parsed pipeline
 * element specs, and model source paths into a `PreparedRuntimeDescriptor` — the data structure
 * the runtime bridge consumes to wire compiled contracts and buffer-segment policies into the
 * live pipeline.
 *
 * @see SimaPluginStaticManifest
 * @see PreparedRuntimeDescriptor (in neat/PreparedRuntimeBridge.h)
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/PipelineBuild.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include <neat/PreparedRuntimeBridge.h>

#include <optional>
#include <string>

namespace simaai::neat::pipeline_internal::sima {

/**
 * @brief Build the prepared-runtime descriptor for a SiMa pipeline.
 *
 * @param static_manifest_context  GStreamer context carrying the static manifest, or null.
 * @param transformed_manifest     Manifest after stage-transform rules have been applied.
 * @param original_manifest        Pre-transform manifest (kept for diagnostics / overlay).
 * @param pipeline_elements        Parsed elements from the GStreamer launch string.
 * @param model_source_paths       Model package paths backing the manifest (for resolving assets).
 * @param name_transform           Element-name transform used during planning.
 * @param error_message            Optional out-parameter populated on failure.
 * @return Prepared-runtime descriptor on success; `std::nullopt` otherwise.
 */
std::optional<simaai::neat::PreparedRuntimeDescriptor> build_prepared_runtime_context(
    const GstContext* static_manifest_context,
    const SimaPluginStaticManifest& transformed_manifest,
    const std::optional<SimaPluginStaticManifest>& original_manifest,
    const std::vector<PipelineElementSpec>& pipeline_elements,
    const std::vector<std::string>& model_source_paths,
    const NameTransform& name_transform,
    std::string* error_message = nullptr);

} // namespace simaai::neat::pipeline_internal::sima

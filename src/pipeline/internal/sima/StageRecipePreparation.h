/** Cold checked numerical recipes and final tensor publication. */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/PipelineBuild.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include <simaai/scheduler/client/kernel.h>

namespace simaai::scheduler::client {
struct TensorStagePublication;
}

#include <optional>
#include <string>

namespace simaai::neat::pipeline_internal::sima {

// Reuse the canonical manifest-to-TensorBuffer projection for named kernels.
// This prepares only caps/metadata, never device resources or tensor storage.
bool build_kernel_stage_publication(const StageStaticSpec&,
                                    ::simaai::scheduler::client::TensorStagePublication*,
                                    std::string* error_message);

enum class GraphTiledChannelEncoding { CompactPixelMajor, PaddedHwcC16, CBlock16 };
bool resolve_graph_processcvu_tiled_channel_encoding(const std::string&, bool c16_packed,
                                                     bool cblock, GraphTiledChannelEncoding*,
                                                     std::string* error_message = nullptr);

// Complete only missing upstream route fields using the existing Core edge
// resolver. Explicit coordinates, including a complete zero offset, remain
// authoritative. This is metadata normalization, not numerical admission.
::simaai::scheduler::client::Status complete_kernel_input_routes(SimaPluginStaticManifest*,
                                                                 std::size_t stage_index);

// Cold metadata-only lowering. Completes MPK-owned contracts, checks canonical
// requests once, and retains their owned executables for graph-wide admission.
::simaai::scheduler::client::Status prepare_kernel_stage_recipes(
    SimaPluginStaticManifest*, const std::optional<SimaPluginStaticManifest>& original_manifest,
    const std::vector<PipelineElementSpec>&, const std::vector<std::string>& model_source_paths,
    const NameTransform&);

// Freeze public metadata only after the final allocation bindings exist.
::simaai::scheduler::client::Status finalize_stage_publications(SimaPluginStaticManifest*);

} // namespace simaai::neat::pipeline_internal::sima

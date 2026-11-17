#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/GraphReport.h"
#include "pipeline/internal/PipelineBuild.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include <optional>
#include <vector>

namespace simaai::neat::pipeline_internal {

/**
 * What a seeded pipeline-segment build resolved after storage and publication planning.
 *
 * `Graph::build()` records it on the segment's DiagCtx for `Model::execution_plan()`.
 */
struct SegmentBuildPlan {
  /// Maps manifest stage names to the instance element names in `node_reports`.
  NameTransform name_transform;
  std::vector<NodeReport> node_reports;
  std::optional<sima::SimaPluginStaticManifest> rendered_manifest;
};

} // namespace simaai::neat::pipeline_internal

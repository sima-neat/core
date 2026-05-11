#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include <vector>

namespace simaai::neat::session_test {

void reset_rendered_manifests();
std::vector<pipeline_internal::sima::SimaPluginStaticManifest> get_rendered_manifests();
void record_rendered_manifest(
    const pipeline_internal::sima::SimaPluginStaticManifest& manifest);

} // namespace simaai::neat::session_test

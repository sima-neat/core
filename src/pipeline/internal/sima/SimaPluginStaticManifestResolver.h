#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include <string>

namespace simaai::neat::pipeline_internal::sima {

SimaPluginStaticManifest
resolve_manifest_from_pipeline(const std::string& pipeline_string, const std::string& session_id,
                               ManifestBuildDiagnostics* diagnostics = nullptr);

} // namespace simaai::neat::pipeline_internal::sima

#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include <memory>
#include <vector>

namespace simaai::neat {
class Node;
struct InputOptions;
} // namespace simaai::neat

namespace simaai::neat::session_test {

void reset_rendered_manifests();
std::vector<pipeline_internal::sima::SimaPluginStaticManifest> get_rendered_manifests();
void record_rendered_manifest(const pipeline_internal::sima::SimaPluginStaticManifest& manifest);
bool apply_auto_memory_policy_from_downstream_for_test(
    InputOptions& src_opt, const std::vector<std::shared_ptr<Node>>& nodes);
int parse_sdp_fps_for_rtp_payload_for_test(const char* sdp_text, int payload_type,
                                           const char* encoding_name);

} // namespace simaai::neat::session_test

#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/OutputTensorOverride.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include <optional>
#include <string>

namespace simaai::neat::pipeline_internal::terminal_output_contract {

enum class StagePublicationRole {
  RealProducer,
  MaterializedTransform,
  BoundaryView,
  TransportOnly,
};

struct PublicOutputEndpointSelector {
  std::string terminal_stage_key;
  std::string output_segment_name;
  int output_slot = -1;
  int route_slot = -1;
};

StagePublicationRole classify_stage_for_publication(const sima::StageStaticSpec& stage);

const sima::StageStaticSpec* find_terminal_real_producer_for_endpoint(
    const sima::SimaPluginStaticManifest& manifest,
    const PublicOutputEndpointSelector& endpoint = {});

sima::StageStaticSpec make_publication_stage_for_terminal(
    const sima::StageStaticSpec& terminal_stage);

bool validate_publication_stage_strict(const sima::StageStaticSpec& publication_stage,
                                       std::string* error_message = nullptr);

std::optional<OutputTensorOverride> build_output_override_from_manifest(
    const sima::SimaPluginStaticManifest& manifest,
    const PublicOutputEndpointSelector& endpoint = {}, std::string* error_message = nullptr);

} // namespace simaai::neat::pipeline_internal::terminal_output_contract

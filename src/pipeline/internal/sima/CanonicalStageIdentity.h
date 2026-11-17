#pragma once

#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "pipeline/internal/sima/static_contract/TransformOperation.h"

#include <string>
#include <string_view>

namespace simaai::neat::pipeline_internal::sima {

// Model spelling normalization belongs to Core; no firmware identities are used.
inline std::string_view canonical_transform_operation(std::string_view name) noexcept {
  if (name == "quantize")
    return "quant";
  if (name == "dequantize")
    return "dequant";
  if (name == "tessellate")
    return "tess";
  if (name == "detessellate")
    return "detess";
  if (name == "quantizetessellate" || name == "quantize_tessellate")
    return "quanttess";
  return name;
}

inline bool initialize_canonical_stage_contract(ProcessCvuStagePayload& payload,
                                                std::string* error = nullptr) {
  const auto name = canonical_transform_operation(payload.graph_name);
  const auto* rule = static_contract::transform_operation(name);
  if (!rule && name != "preproc" && name != "feature_histogram" && name != "grider_fast" &&
      name != "track_descriptor" && name != "track_klt" && name != "simor_depth_map") {
    if (error)
      *error = "unknown canonical numerical stage '" + std::string(name) + "'";
    return false;
  }
  payload.graph_name = name;
  // Legacy static-manifest fields remain serializable for ABI compatibility but
  // are not numerical preparation authority in newly authored canonical stages.
  payload.graph_id = 0;
  payload.descriptor_abi_id = 0U;
  payload.descriptor_contract_version = 0U;
  payload.binding_schema_version = 0U;
  payload.supported_placement_mask = 0U;
  payload.allowed_frame_patch_mask = 0U;
  payload.maximum_members = rule ? rule->maximum_members : 1U;
  if (error)
    error->clear();
  return true;
}

} // namespace simaai::neat::pipeline_internal::sima

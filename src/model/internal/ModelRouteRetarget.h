/**
 * @file
 * @brief Internal helpers for model-bound route retargeting.
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/Model.h"

#include <memory>
#include <optional>
#include <string>

namespace simaai::neat::internal {

enum class PostRouteStageKind;

enum class RequestedPostRouteKind {
  Auto = 0,
  BoxDecode = 1,
  Detess = 2,
  DetessDequant = 3,
  Dequant = 4,
};

enum class ModelLineageStageRole {
  Preprocess = 0,
  Infer = 1,
  ManualPost = 2,
};

struct ModelLineageBinding {
  std::string lineage_key;
  std::string source_path;
  Model::Options base_options;
  ModelLineageStageRole stage_role = ModelLineageStageRole::Infer;
  RequestedPostRouteKind requested_post = RequestedPostRouteKind::Auto;
  std::string requester_kind;
};

class ModelLineageProvider {
public:
  virtual ~ModelLineageProvider() = default;
  virtual const ModelLineageBinding* model_lineage_binding() const = 0;
};

RequestedPostRouteKind requested_post_route_from_stage_kind(PostRouteStageKind kind);
PostRouteStageKind requested_post_route_to_stage_kind(RequestedPostRouteKind kind);
std::string requested_post_route_name(RequestedPostRouteKind kind);
std::shared_ptr<const ModelLineageBinding>
make_model_lineage_binding(const Model& model, ModelLineageStageRole stage_role,
                           RequestedPostRouteKind requested_post,
                           const std::string& requester_kind = {});
bool requested_post_route_supported(RequestedPostRouteKind kind);
std::shared_ptr<Model> build_effective_model_for_requested_post(const ModelLineageBinding& binding,
                                                                BoxDecodeType requested_decode_type,
                                                                bool* changed = nullptr,
                                                                std::string* err = nullptr);

} // namespace simaai::neat::internal

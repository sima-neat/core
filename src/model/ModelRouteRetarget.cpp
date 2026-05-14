#include "model/internal/ModelRouteRetarget.h"

#include "model/internal/ModelInternal.h"

namespace simaai::neat::internal {
namespace {

Model::Options internal_retarget_model_options(Model::Options opt) {
  opt.verbose.progress = false;
  opt.verbose.progress_force = false;
  if (opt.verbose.level != VerbosityLevel::Verbose && !opt.verbose.planner) {
    opt.verbose.level = VerbosityLevel::Quiet;
  }
  return opt;
}

} // namespace

RequestedPostRouteKind requested_post_route_from_stage_kind(PostRouteStageKind kind) {
  switch (kind) {
  case PostRouteStageKind::BoxDecode:
    return RequestedPostRouteKind::BoxDecode;
  case PostRouteStageKind::Detess:
    return RequestedPostRouteKind::Detess;
  case PostRouteStageKind::DetessDequant:
    return RequestedPostRouteKind::DetessDequant;
  case PostRouteStageKind::Dequantize:
    return RequestedPostRouteKind::Dequant;
  case PostRouteStageKind::None:
  case PostRouteStageKind::Cast:
  case PostRouteStageKind::Unknown:
    return RequestedPostRouteKind::Auto;
  }
  return RequestedPostRouteKind::Auto;
}

PostRouteStageKind requested_post_route_to_stage_kind(RequestedPostRouteKind kind) {
  switch (kind) {
  case RequestedPostRouteKind::BoxDecode:
    return PostRouteStageKind::BoxDecode;
  case RequestedPostRouteKind::Detess:
    return PostRouteStageKind::Detess;
  case RequestedPostRouteKind::DetessDequant:
    return PostRouteStageKind::DetessDequant;
  case RequestedPostRouteKind::Dequant:
    return PostRouteStageKind::Dequantize;
  case RequestedPostRouteKind::Auto:
    return PostRouteStageKind::None;
  }
  return PostRouteStageKind::Unknown;
}

std::string requested_post_route_name(RequestedPostRouteKind kind) {
  switch (kind) {
  case RequestedPostRouteKind::Auto:
    return "auto";
  case RequestedPostRouteKind::BoxDecode:
    return "boxdecode";
  case RequestedPostRouteKind::Detess:
    return "detess";
  case RequestedPostRouteKind::DetessDequant:
    return "detessdequant";
  case RequestedPostRouteKind::Dequant:
    return "dequant";
  }
  return "auto";
}

std::shared_ptr<const ModelLineageBinding>
make_model_lineage_binding(const Model& model, ModelLineageStageRole stage_role,
                           RequestedPostRouteKind requested_post,
                           const std::string& requester_kind) {
  auto binding = std::make_shared<ModelLineageBinding>();
  binding->lineage_key = ModelAccess::model_id(model);
  binding->source_path = ModelAccess::source_path(model);
  binding->base_options = ModelAccess::options(model);
  binding->stage_role = stage_role;
  binding->requested_post = requested_post;
  binding->requester_kind = requester_kind;
  return binding;
}

bool requested_post_route_supported(RequestedPostRouteKind kind) {
  return kind == RequestedPostRouteKind::Auto || kind == RequestedPostRouteKind::BoxDecode;
}

std::shared_ptr<Model> build_effective_model_for_requested_post(const ModelLineageBinding& binding,
                                                                BoxDecodeType requested_decode_type,
                                                                bool* changed, std::string* err) {
  if (changed) {
    *changed = false;
  }
  if (err) {
    err->clear();
  }

  const Model::Options base_opt = internal_retarget_model_options(binding.base_options);
  auto model = std::make_shared<Model>(binding.source_path, base_opt);
  const RequestedPostRouteKind current =
      requested_post_route_from_stage_kind(ModelAccess::resolved_post_kind(*model));
  const RequestedPostRouteKind requested = binding.requested_post;
  if (requested == RequestedPostRouteKind::Auto || requested == current) {
    if (requested == RequestedPostRouteKind::BoxDecode &&
        requested_decode_type != BoxDecodeType::Unspecified &&
        base_opt.decode_type != requested_decode_type) {
      Model::Options opt = base_opt;
      opt.decode_type = requested_decode_type;
      auto rebuilt = std::make_shared<Model>(ModelAccess::clone_with_options(*model, opt));
      if (changed) {
        *changed = true;
      }
      return rebuilt;
    }
    return model;
  }

  if (!requested_post_route_supported(requested)) {
    if (err) {
      *err = "requested post route '" + requested_post_route_name(requested) +
             "' is not supported for automatic retargeting";
    }
    return nullptr;
  }

  Model::Options opt = base_opt;
  switch (requested) {
  case RequestedPostRouteKind::BoxDecode:
    opt.decode_type = requested_decode_type != BoxDecodeType::Unspecified ? requested_decode_type
                                                                          : opt.decode_type;
    if (opt.decode_type == BoxDecodeType::Unspecified) {
      if (err) {
        *err = "boxdecode retarget requires a concrete decode_type";
      }
      return nullptr;
    }
    break;
  case RequestedPostRouteKind::Auto:
  case RequestedPostRouteKind::Detess:
  case RequestedPostRouteKind::DetessDequant:
  case RequestedPostRouteKind::Dequant:
    if (err) {
      *err = "requested post route '" + requested_post_route_name(requested) +
             "' is not implemented for automatic retargeting";
    }
    return nullptr;
  }

  auto rebuilt = std::make_shared<Model>(ModelAccess::clone_with_options(*model, opt));
  if (requested_post_route_from_stage_kind(ModelAccess::resolved_post_kind(*rebuilt)) !=
      requested) {
    if (err) {
      *err = "retargeted model route remained '" +
             requested_post_route_name(
                 requested_post_route_from_stage_kind(ModelAccess::resolved_post_kind(*rebuilt))) +
             "' instead of requested '" + requested_post_route_name(requested) + "'";
    }
    return nullptr;
  }
  if (changed) {
    *changed = true;
  }
  return rebuilt;
}

} // namespace simaai::neat::internal

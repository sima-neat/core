#include "pipeline/internal/InputPolicy.h"

#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/InputStreamUtil.h"

#include <algorithm>

namespace simaai::neat::pipeline_internal {

InputOptions normalize_shape_bounds(const InputOptions& in) {
  InputOptions out = in;
  if (out.max_width <= 0 && out.width > 0)
    out.max_width = out.width;
  if (out.max_height <= 0 && out.height > 0)
    out.max_height = out.height;
  if (out.max_depth <= 0 && out.depth > 0)
    out.max_depth = out.depth;
  return out;
}

bool has_explicit_shape_bounds(const InputOptions& opt) {
  return opt.width > 0 || opt.height > 0 || opt.depth > 0 || opt.max_width > 0 ||
         opt.max_height > 0 || opt.max_depth > 0;
}

InputStreamOptions::ShapePolicy resolve_shape_policy(const InputOptions& opt) {
  if (!opt.caps_override.empty())
    return InputStreamOptions::ShapePolicy::LockedByCapsOverride;
  if (has_explicit_shape_bounds(opt))
    return InputStreamOptions::ShapePolicy::BoundedDynamic;
  return InputStreamOptions::ShapePolicy::ElasticDynamic;
}

InputStreamOptions::ResolvedShapeLimits resolve_shape_limits(const InputOptions& opt,
                                                             const SampleSpec& seed) {
  InputStreamOptions::ResolvedShapeLimits limits{};

  if (opt.width > 0) {
    limits.seed_width = opt.width;
    limits.seed_width_origin = InputStreamOptions::LimitOrigin::UserSeed;
  } else if (seed.width > 0) {
    limits.seed_width = seed.width;
    limits.seed_width_origin = InputStreamOptions::LimitOrigin::SeedInput;
  }

  if (opt.height > 0) {
    limits.seed_height = opt.height;
    limits.seed_height_origin = InputStreamOptions::LimitOrigin::UserSeed;
  } else if (seed.height > 0) {
    limits.seed_height = seed.height;
    limits.seed_height_origin = InputStreamOptions::LimitOrigin::SeedInput;
  }

  if (opt.depth > 0) {
    limits.seed_depth = opt.depth;
    limits.seed_depth_origin = InputStreamOptions::LimitOrigin::UserSeed;
  } else if (seed.depth > 0) {
    limits.seed_depth = seed.depth;
    limits.seed_depth_origin = InputStreamOptions::LimitOrigin::SeedInput;
  }

  if (opt.max_width > 0) {
    limits.max_width = opt.max_width;
    limits.max_width_origin = InputStreamOptions::LimitOrigin::UserMax;
  } else if (opt.width > 0) {
    limits.max_width = opt.width;
    limits.max_width_origin = InputStreamOptions::LimitOrigin::UserSeed;
  }

  if (opt.max_height > 0) {
    limits.max_height = opt.max_height;
    limits.max_height_origin = InputStreamOptions::LimitOrigin::UserMax;
  } else if (opt.height > 0) {
    limits.max_height = opt.height;
    limits.max_height_origin = InputStreamOptions::LimitOrigin::UserSeed;
  }

  if (opt.max_depth > 0) {
    limits.max_depth = opt.max_depth;
    limits.max_depth_origin = InputStreamOptions::LimitOrigin::UserMax;
  } else if (opt.depth > 0) {
    limits.max_depth = opt.depth;
    limits.max_depth_origin = InputStreamOptions::LimitOrigin::UserSeed;
  }

  limits.max_width_explicit = opt.max_width > 0;
  limits.max_height_explicit = opt.max_height > 0;
  limits.max_depth_explicit = opt.max_depth > 0;
  return limits;
}

std::optional<std::string>
validate_shape_limits(const InputStreamOptions::ResolvedShapeLimits& limits) {
  if (limits.max_width > 0 && limits.seed_width > 0 && limits.seed_width > limits.max_width) {
    return std::string("input width seed exceeds max_width (") +
           std::to_string(limits.seed_width) + " > " + std::to_string(limits.max_width) + ")";
  }
  if (limits.max_height > 0 && limits.seed_height > 0 && limits.seed_height > limits.max_height) {
    return std::string("input height seed exceeds max_height (") +
           std::to_string(limits.seed_height) + " > " + std::to_string(limits.max_height) +
           ")";
  }
  if (limits.max_depth > 0 && limits.seed_depth > 0 && limits.seed_depth > limits.max_depth) {
    return std::string("input depth seed exceeds max_depth (") +
           std::to_string(limits.seed_depth) + " > " + std::to_string(limits.max_depth) + ")";
  }
  return std::nullopt;
}

int default_depth_for_image_format(const std::string& fmt, int fallback) {
  const std::string up = upper_copy(fmt);
  if (up == "GRAY" || up == "GRAY8")
    return 1;
  if (up == "RGB" || up == "BGR")
    return 3;
  return fallback;
}

std::size_t default_elastic_input_bytes_guard() {
  const int mb = std::max(1, env_int("SIMA_INPUTSTREAM_ELASTIC_MAX_MB", 256));
  return static_cast<std::size_t>(mb) * 1024ULL * 1024ULL;
}

std::size_t resolve_input_bytes_guard(std::size_t requested_max_input_bytes,
                                      InputStreamOptions::ShapePolicy shape_policy,
                                      std::size_t bounded_estimate_bytes,
                                      InputStreamOptions::ByteGuardOrigin* out_origin) {
  if (requested_max_input_bytes > 0) {
    if (out_origin)
      *out_origin = InputStreamOptions::ByteGuardOrigin::User;
    return requested_max_input_bytes;
  }

  if (shape_policy == InputStreamOptions::ShapePolicy::ElasticDynamic) {
    if (out_origin)
      *out_origin = InputStreamOptions::ByteGuardOrigin::DerivedElasticDefault;
    return default_elastic_input_bytes_guard();
  }

  if (out_origin)
    *out_origin = InputStreamOptions::ByteGuardOrigin::DerivedBoundedEstimate;
  return bounded_estimate_bytes;
}

SessionInputPolicyResult resolve_session_input_policy(const InputOptions& opt,
                                                      const SampleSpec& seed,
                                                      std::size_t requested_max_input_bytes,
                                                      std::size_t bounded_estimate_bytes) {
  SessionInputPolicyResult out;
  out.shape_policy = resolve_shape_policy(opt);
  out.shape_limits = resolve_shape_limits(opt, seed);
  out.max_input_bytes_guard = resolve_input_bytes_guard(requested_max_input_bytes, out.shape_policy,
                                                        bounded_estimate_bytes,
                                                        &out.byte_guard_origin);
  return out;
}

ModelInputPolicyResult resolve_model_input_policy(const ModelInputPolicyRequest& req) {
  ModelInputPolicyResult out;

  out.resolved_input_format =
      req.preproc_input_img_type.has_value() ? *req.preproc_input_img_type : req.format;
  out.resolved_input_width = req.preproc_input_width.value_or(0);
  out.resolved_input_height = req.preproc_input_height.value_or(0);

  if (req.input_max_depth > 0) {
    out.resolved_input_depth = req.input_max_depth;
  } else {
    out.resolved_input_depth = default_depth_for_image_format(out.resolved_input_format, 0);
  }

  if (req.input_max_width > 0) {
    out.resolved_max_input_width = req.input_max_width;
  } else {
    out.resolved_max_input_width = (out.resolved_input_width > 0) ? out.resolved_input_width : 1920;
  }

  if (req.input_max_height > 0) {
    out.resolved_max_input_height = req.input_max_height;
  } else {
    out.resolved_max_input_height =
        (out.resolved_input_height > 0) ? out.resolved_input_height : 1080;
  }

  if (req.input_max_depth > 0) {
    out.resolved_max_input_depth = req.input_max_depth;
  } else {
    const int by_format = default_depth_for_image_format(out.resolved_input_format, 3);
    out.resolved_max_input_depth = (out.resolved_input_depth > 0) ? out.resolved_input_depth : by_format;
  }

  out.resolved_normalize = req.preproc_normalize.value_or(false);
  return out;
}

SessionInputPolicyResult resolve_for_session(const InputOptions& opt,
                                             const SampleSpec& seed,
                                             std::size_t requested_max_input_bytes,
                                             std::size_t bounded_estimate_bytes) {
  return resolve_session_input_policy(opt, seed, requested_max_input_bytes, bounded_estimate_bytes);
}

ModelInputPolicyResult resolve_for_model(const ModelInputPolicyRequest& req) {
  return resolve_model_input_policy(req);
}
} // namespace simaai::neat::pipeline_internal

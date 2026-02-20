#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "internal/InputStream.h"
#include "nodes/io/Input.h"

#include <cstddef>
#include <optional>
#include <string>

namespace simaai::neat {

class SampleSpec;

namespace pipeline_internal {

struct ModelInputPolicyRequest {
  std::string format;
  std::optional<int> preproc_input_width;
  std::optional<int> preproc_input_height;
  std::optional<std::string> preproc_input_img_type;
  std::optional<bool> preproc_normalize;
  int input_max_width = 0;
  int input_max_height = 0;
  int input_max_depth = 0;
};

struct ModelInputPolicyResult {
  std::string resolved_input_format;
  int resolved_input_width = 0;
  int resolved_input_height = 0;
  int resolved_input_depth = 0;
  int resolved_max_input_width = 0;
  int resolved_max_input_height = 0;
  int resolved_max_input_depth = 0;
  bool resolved_normalize = false;
};

struct SessionInputPolicyResult {
  InputStreamOptions::ShapePolicy shape_policy =
      InputStreamOptions::ShapePolicy::BoundedDynamic;
  InputStreamOptions::ResolvedShapeLimits shape_limits{};
  std::size_t max_input_bytes_guard = 0;
  InputStreamOptions::ByteGuardOrigin byte_guard_origin =
      InputStreamOptions::ByteGuardOrigin::Unset;
};

using ResolvedLimits = InputStreamOptions::ResolvedShapeLimits;

InputOptions normalize_shape_bounds(const InputOptions& opt);

bool has_explicit_shape_bounds(const InputOptions& opt);

InputStreamOptions::ShapePolicy resolve_shape_policy(const InputOptions& opt);

InputStreamOptions::ResolvedShapeLimits resolve_shape_limits(const InputOptions& opt,
                                                             const SampleSpec& seed);

std::optional<std::string>
validate_shape_limits(const InputStreamOptions::ResolvedShapeLimits& limits);

int default_depth_for_image_format(const std::string& fmt, int fallback = -1);

std::size_t default_elastic_input_bytes_guard();

std::size_t resolve_input_bytes_guard(std::size_t requested_max_input_bytes,
                                      InputStreamOptions::ShapePolicy shape_policy,
                                      std::size_t bounded_estimate_bytes,
                                      InputStreamOptions::ByteGuardOrigin* out_origin);

SessionInputPolicyResult resolve_session_input_policy(const InputOptions& opt,
                                                      const SampleSpec& seed,
                                                      std::size_t requested_max_input_bytes,
                                                      std::size_t bounded_estimate_bytes);

ModelInputPolicyResult resolve_model_input_policy(const ModelInputPolicyRequest& req);

SessionInputPolicyResult resolve_for_session(const InputOptions& opt,
                                             const SampleSpec& seed,
                                             std::size_t requested_max_input_bytes,
                                             std::size_t bounded_estimate_bytes);

ModelInputPolicyResult resolve_for_model(const ModelInputPolicyRequest& req);

} // namespace pipeline_internal
} // namespace simaai::neat

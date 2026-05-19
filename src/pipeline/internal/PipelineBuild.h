#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/SessionOptions.h"
#include "pipeline/internal/TensorMath.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

class Node;

struct NameTransform {
  std::string prefix;
  std::string suffix;
};

NameTransform make_name_transform(const SessionOptions& opt);

bool name_transform_enabled(const NameTransform& t);

std::string apply_name_transform(const NameTransform& t, const std::string& name);
std::vector<std::string> apply_name_transform(const NameTransform& t,
                                              const std::vector<std::string>& names);

// Buffer-name transform (no RTSP payloader special-casing).
std::string apply_name_transform_once(const std::string& value, const NameTransform& transform);

} // namespace simaai::neat

namespace simaai::neat::pipeline_internal {

class PipelineBuildContext {
public:
  explicit PipelineBuildContext(const SessionOptions& opt);
  explicit PipelineBuildContext(const NameTransform& transform);

  const NameTransform& name_transform() const {
    return name_transform_;
  }

  std::string resolve_buffer_name(const std::string& raw) const;
  std::vector<std::string> resolve_expected_buffer_names(const std::string& raw) const;

private:
  NameTransform name_transform_;
};

std::string resolve_buffer_name(const std::string& raw, const NameTransform& transform);
std::vector<std::string> resolve_expected_buffer_names(const std::string& raw,
                                                       const NameTransform& transform);

} // namespace simaai::neat::pipeline_internal

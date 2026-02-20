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

struct ConfigWiringIssue {
  std::string message;
};

struct BuildWiringReport {
  std::vector<ConfigWiringIssue> issues;
};

class PipelineBuildContext {
public:
  explicit PipelineBuildContext(const SessionOptions& opt);
  explicit PipelineBuildContext(const NameTransform& transform);

  const NameTransform& name_transform() const {
    return name_transform_;
  }

  std::string resolve_buffer_name(const std::string& raw) const;
  std::vector<std::string> resolve_expected_buffer_names(const std::string& raw) const;
  // Legacy JSON-wiring hooks retained for ABI/source compatibility; currently no-op.
  void apply_name_transform_to_configs(const std::vector<std::shared_ptr<Node>>& nodes) const;
  void wire_configs_by_order(const std::vector<std::shared_ptr<Node>>& nodes) const;
  void dump_mla_config_wiring(const std::vector<std::shared_ptr<Node>>& nodes) const;
  BuildWiringReport check_config_wiring(const std::vector<std::shared_ptr<Node>>& nodes) const;

private:
  NameTransform name_transform_;
};

std::string resolve_buffer_name(const std::string& raw, const NameTransform& transform);
std::vector<std::string> resolve_expected_buffer_names(const std::string& raw,
                                                       const NameTransform& transform);

} // namespace simaai::neat::pipeline_internal

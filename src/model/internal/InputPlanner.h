#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/Model.h"
#include "model/internal/ModelPack.h"
#include "model/internal/RoutePlanner.h"

#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace simaai::neat::internal {

enum class RequiredPreprocessOp {
  Resize = 0,
  ColorConvert = 1,
  LayoutConvert = 2,
  Normalize = 3,
  Quantize = 4,
  Tessellate = 5,
};

enum class RequirementSeverity {
  HardError = 0,
  Warning = 1,
};

enum class RequirementSource {
  MlaContract = 0,
  GraphFamilyCapability = 1,
  GraphDefault = 2,
  Preset = 3,
  UserOverride = 4,
  RuntimeObserved = 5,
};

struct RequirementIssue {
  RequiredPreprocessOp op = RequiredPreprocessOp::Resize;
  RequirementSeverity severity = RequirementSeverity::Warning;
  RequirementSource source = RequirementSource::GraphDefault;
  std::string code;
  std::string reason;
  std::string fix_hint;
};

std::string requirement_issue_message(const RequirementIssue& issue);
std::string required_preprocess_op_name(RequiredPreprocessOp op);
std::string requirement_severity_name(RequirementSeverity severity);
std::string requirement_source_name(RequirementSource source);

struct GraphFamilyCapabilities {
  bool available = false;
  bool supports_resize = false;
  bool supports_color_convert = false;
  bool supports_layout_convert = false;
  bool supports_normalize = false;
  bool supports_quantize = false;
  bool supports_tessellate = false;
};

struct PreprocessCapabilities {
  GraphFamilyCapabilities preproc;
  GraphFamilyCapabilities quant;
  GraphFamilyCapabilities tess;
  GraphFamilyCapabilities quanttess;
  PreprocessGraphFamily tensor_auto_family = PreprocessGraphFamily::Disabled;
  bool has_model_input_normalization = false;
  std::array<float, 3> model_input_mean = {0.0f, 0.0f, 0.0f};
  std::array<float, 3> model_input_stddev = {1.0f, 1.0f, 1.0f};
};

struct PreprocessPlannerResult {
  ResolvedPreprocessPlan resolved_plan;
  std::vector<RequirementIssue> requirement_issues;
  PipelineType pipeline_type = PipelineType::Preproc;
  bool include_preprocess_stage = true;
  bool include_postprocess_stage = true;
  bool infer_only_route = false;
  bool mla_tessellation = false;
  std::string route_selected_post_kind = "none";
  bool route_cast_symmetry_ok = true;
  std::vector<std::string> route_diagnostics;
  SessionRoutePlan session_route_plan;

  std::string modelpack_media_type;
  std::string modelpack_format;
  int modelpack_input_depth = 0;
  int modelpack_max_width = 0;
  int modelpack_max_height = 0;
  int modelpack_max_depth = 0;
  bool normalize = false;
  std::vector<float> mean;
  std::vector<float> stddev;
};

PreprocessCapabilities inspect_preprocess_capabilities(const ModelPack& pack);
PreprocessPlannerResult plan_preprocess(const Model::Options& options,
                                        const PreprocessCapabilities& capabilities);

} // namespace simaai::neat::internal

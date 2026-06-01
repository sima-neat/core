#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "builder/InputContractConfigurable.h"
#include "builder/OutputSpec.h"
#include "pipeline/GraphOptions.h"

#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {

struct CompiledNodeContract;

struct InputContractFacts {
  std::optional<InputContract> ingress_contract;
  std::optional<OutputSpec> ingress_spec;
  std::optional<Sample> ingress_sample;
  std::vector<std::string> preprocess_meta_fields;
};

struct PreprocessMetaFacts {
  std::vector<std::string> required_fields;
};

inline const std::vector<std::string>& default_preprocess_meta_required_fields() {
  static const std::vector<std::string> kFields = {
      "preproc_original_width", "preproc_original_height", "preproc_resized_width",
      "preproc_resized_height", "preproc_scaled_width",    "preproc_scaled_height",
      "preproc_pad_left",       "preproc_pad_right",       "preproc_pad_top",
      "preproc_pad_bottom",     "preproc_resize_mode",     "preproc_color_in",
      "preproc_color_out",      "preproc_axis_perm",       "preproc_normalize",
      "preproc_quantize",       "preproc_tessellate",      "preproc_affine_m00",
      "preproc_affine_m01",     "preproc_affine_m02",      "preproc_affine_m10",
      "preproc_affine_m11",     "preproc_affine_m12",      "preproc_affine_scale_x",
      "preproc_affine_scale_y", "preproc_affine_offset_x", "preproc_affine_offset_y",
  };
  return kFields;
}

struct ContractCompileInput {
  std::string pipeline_label;
  int node_index = 0;
  InputContractFacts ingress;
  const CompiledNodeContract* immediate_upstream = nullptr;
  bool strict = true;
  std::string processcvu_requested_run_target = "AUTO";
  ProcessCvuOptions processcvu;
};

} // namespace simaai::neat

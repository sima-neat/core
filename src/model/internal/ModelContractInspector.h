#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/Model.h"
#include "model/internal/ModelPack.h"
#include "mpk/MpKLoader.h"

#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat::internal {

enum class ModelContractStageFilter : std::uint8_t {
  All = 0,
  Pre = 1,
  Infer = 2,
  Post = 3,
};

struct ModelContractReportOptions {
  ModelContractStageFilter stage_filter = ModelContractStageFilter::All;
  std::vector<std::string> plugin_filters;
  bool show_gst = false;
};

struct ModelContractReportContext {
  const simaai::neat::mpk::MpKManifest* manifest = nullptr;
  const Model::ModelInfo* model_info = nullptr;
  const ResolvedPreprocessPlan* preprocess_plan = nullptr;
  std::string selected_post_kind;
};

struct ModelContractInspectionResult {
  bool archive_ok = false;
  bool raw_contract_ok = false;
  bool model_ok = false;
  std::string report;
};

std::string build_model_contract_report(const ModelPack& pack,
                                        const ModelContractReportOptions& options,
                                        const ModelContractReportContext& context = {});

ModelContractInspectionResult
inspect_model_contract_archive(const std::string& tar_gz,
                               const ModelContractReportOptions& options);

} // namespace simaai::neat::internal

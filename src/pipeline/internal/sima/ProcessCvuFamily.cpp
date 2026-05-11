#include "pipeline/internal/sima/ProcessCvuFamily.h"

#include "model/internal/ModelPack.h"

#include <algorithm>
#include <cctype>
#include <string>

namespace simaai::neat::pipeline_internal::sima {

std::string canonical_processcvu_family_from_kernel(std::string kernel) {
  std::transform(kernel.begin(), kernel.end(), kernel.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (kernel.find("detesscast") != std::string::npos || kernel.find("casttess") != std::string::npos) {
    return kernel.find("detesscast") != std::string::npos ? "detesscast" : "casttess";
  }
  if (kernel.find("detess") != std::string::npos || kernel.find("dequant") != std::string::npos ||
      kernel.find("boxdecode") != std::string::npos) {
    return {};
  }
  if (kernel.find("quanttess") != std::string::npos ||
      kernel.find("quantizetessellate") != std::string::npos ||
      kernel.find("quantize_tessellate") != std::string::npos) {
    return "quanttess";
  }
  if (kernel.find("preprocess") != std::string::npos || kernel.find("preproc") != std::string::npos) {
    return "preproc";
  }
  if (kernel.find("tess") != std::string::npos) {
    return "tess";
  }
  if (kernel.find("quant") != std::string::npos) {
    return "quant";
  }
  if (kernel.find("cast") != std::string::npos) {
    return "cast";
  }
  return {};
}

std::string processcvu_graph_family_for_stage_kind(
    ::simaai::neat::internal::ExecutionStageKind kind) {
  using ::simaai::neat::internal::ExecutionStageKind;
  switch (kind) {
  case ExecutionStageKind::Preproc:
    return "preproc";
  case ExecutionStageKind::Quant:
    return "quantize";
  case ExecutionStageKind::Tess:
    return "tessellate";
  case ExecutionStageKind::QuantTess:
    return "quanttess";
  case ExecutionStageKind::CastTess:
    return "casttess";
  case ExecutionStageKind::Detess:
    return "detessellate";
  case ExecutionStageKind::DetessCast:
    return "detesscast";
  case ExecutionStageKind::DetessDequant:
    return "detessdequant";
  case ExecutionStageKind::Dequant:
    return "dequantize";
  case ExecutionStageKind::Cast:
    return "cast";
  case ExecutionStageKind::Mla:
  case ExecutionStageKind::BoxDecode:
  case ExecutionStageKind::Unknown:
    break;
  }
  return {};
}

} // namespace simaai::neat::pipeline_internal::sima

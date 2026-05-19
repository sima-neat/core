#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/contract/CompiledNodeContract.h"

namespace simaai::neat {

inline const CompiledRuntimeContract*
compiled_runtime_contract_from_stage(const CompiledNodeContract* stage) {
  if (!stage) {
    return nullptr;
  }
  if (stage->processcvu.has_value()) {
    return &stage->processcvu->runtime_contract;
  }
  if (stage->processmla.has_value()) {
    return &stage->processmla->runtime_contract;
  }
  if (stage->boxdecode.has_value()) {
    return &stage->boxdecode->runtime_contract;
  }
  if (stage->dequant.has_value()) {
    return &stage->dequant->runtime_contract;
  }
  if (stage->transport.has_value()) {
    return &stage->transport->runtime_contract;
  }
  return nullptr;
}

inline const CompiledNodeContract* last_effective_child_stage(const CompiledNodeContract* stage) {
  if (!stage) {
    return nullptr;
  }
  if (stage->child_stages.empty()) {
    return stage;
  }
  return last_effective_child_stage(&stage->child_stages.back());
}

} // namespace simaai::neat

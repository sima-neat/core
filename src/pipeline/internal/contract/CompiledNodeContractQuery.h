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
  if (stage->kernel.has_value()) {
    return &stage->kernel->runtime_contract;
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

inline const CompiledNodeContract* first_effective_child_stage(const CompiledNodeContract* stage) {
  while (stage && !stage->child_stages.empty())
    stage = &stage->child_stages.front();
  return stage;
}

inline const CompiledNodeContract* last_effective_child_stage(const CompiledNodeContract* stage) {
  while (stage && !stage->child_stages.empty())
    stage = &stage->child_stages.back();
  return stage;
}

inline const CompiledProcessCvuContract*
last_preprocess_contract(const CompiledNodeContract& stage) {
  for (auto child = stage.child_stages.rbegin(); child != stage.child_stages.rend(); ++child)
    if (const auto* found = last_preprocess_contract(*child))
      return found;
  return stage.processcvu && stage.processcvu->payload.graph_name == "preproc" ? &*stage.processcvu
                                                                               : nullptr;
}

inline const CompiledProcessCvuContract*
last_preprocess_contract(const std::vector<CompiledNodeContract>& stages,
                         const CompiledProcessCvuContract* fallback = nullptr) {
  for (auto stage = stages.rbegin(); stage != stages.rend(); ++stage)
    if (const auto* found = last_preprocess_contract(*stage))
      return found;
  return fallback;
}

} // namespace simaai::neat

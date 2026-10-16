#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/PreprocessPlan.h"
#include "pipeline/internal/contract/PluginCompiledContracts.h"

namespace simaai::neat::pipeline_internal::sima {
struct BoxDecodeGeometryOptions {
  int model_width = 0, model_height = 0;
  int original_width = 0, original_height = 0;
  ResizeMode resize = ResizeMode::Letterbox;
};
struct BoxDecodeKernelRecipeResult {
  CompiledKernelContract compiled;
  std::size_t decode_member = 0;
};
// Cold lowering only: immutable numerical admission, explicit format transforms
// and Core wire publication. No input bytes, device connection or per-frame math.
// Geometry must come from node/preprocess contracts, never guessed from heads.
BoxDecodeKernelRecipeResult prepare_boxdecode_kernel(const CompiledBoxDecodeContract&,
                                                     const BoxDecodeGeometryOptions&);
} // namespace simaai::neat::pipeline_internal::sima

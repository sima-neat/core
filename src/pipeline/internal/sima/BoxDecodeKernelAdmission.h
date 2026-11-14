#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/BoxDecodeKernelRecipe.h"

namespace simaai::neat {
struct ContractCompileInput;
}
namespace simaai::neat::pipeline_internal::sima {
// Shared graph-build admission for public nodes and authored MPK terminals.
// Preserve authoritative carriers and install dynamic geometry resolution once.
CompiledKernelContract
compile_boxdecode_kernel(const CompiledBoxDecodeContract&, const ContractCompileInput&,
                         BoxDecodeGeometryOptions,
                         std::optional<ResizeMode> resize_mode = std::nullopt);

// True when a letterbox is not centered: published pads differ by more than the
// one-pixel centering remainder, or, without published facts, the nearest
// Preproc policy names a corner placement. Other resize modes are never off-center.
bool boxdecode_letterbox_is_off_center(const PreprocessRuntimeMeta* observed,
                                       const ProcessCvuStagePayload* upstream_preprocess);
} // namespace simaai::neat::pipeline_internal::sima

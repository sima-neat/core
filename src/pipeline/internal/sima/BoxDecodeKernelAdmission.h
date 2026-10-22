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
} // namespace simaai::neat::pipeline_internal::sima

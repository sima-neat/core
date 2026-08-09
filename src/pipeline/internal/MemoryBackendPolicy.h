#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

namespace simaai::neat::pipeline_internal {

// One Core-owned parser for the memory architecture selected by the process.
// ModelPack uses it to prove static stage contracts, while public Tensor
// placement uses it to choose the matching transport allocation.
enum class MemoryBackendPolicy {
  Legacy,
  DmaBufPlan,
};

// Reads SIMA_NEAT_MEMORY_BACKEND. An unset value means Legacy. Unknown values
// fail closed rather than selecting either memory implementation implicitly.
MemoryBackendPolicy selected_memory_backend_policy();

} // namespace simaai::neat::pipeline_internal

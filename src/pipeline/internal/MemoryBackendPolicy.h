#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

namespace simaai::neat::pipeline_internal {

// Temporary Phase-1 migration selector. Phase 7B deletes this enum and parser
// after strict execution becomes the only implementation.
enum class MemoryBackendPolicy {
  Legacy,
  DmaBufPlan,
};

const char* memory_backend_policy_name(MemoryBackendPolicy policy) noexcept;

// Pure exact parser used by unit tests and the one process-level loader. Null
// means the migration default (Legacy). Empty, whitespace-altered, case-altered,
// and unknown values fail closed.
MemoryBackendPolicy parse_memory_backend_policy(const char* raw);

struct ProcessMemoryBackendSelection {
  MemoryBackendPolicy policy = MemoryBackendPolicy::Legacy;
  bool explicitly_configured = false;
};

// Reads SIMA_NEAT_MEMORY_BACKEND exactly once and returns an immutable process
// selection. Lower layers consume this object or an explicitly propagated
// policy; they never reread mutable environment state.
const ProcessMemoryBackendSelection& process_memory_backend_selection();

} // namespace simaai::neat::pipeline_internal

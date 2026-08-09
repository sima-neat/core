#include "pipeline/internal/MemoryBackendPolicy.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace simaai::neat::pipeline_internal {

const char* memory_backend_policy_name(const MemoryBackendPolicy policy) noexcept {
  switch (policy) {
  case MemoryBackendPolicy::Legacy:
    return "legacy";
  case MemoryBackendPolicy::DmaBufPlan:
    return "dmabuf-plan";
  }
  return "unknown";
}

MemoryBackendPolicy parse_memory_backend_policy(const char* raw) {
  if (!raw) {
    return MemoryBackendPolicy::Legacy;
  }
  if (std::strcmp(raw, "legacy") == 0) {
    return MemoryBackendPolicy::Legacy;
  }
  if (std::strcmp(raw, "dmabuf-plan") == 0) {
    return MemoryBackendPolicy::DmaBufPlan;
  }
  throw std::runtime_error(
      std::string("invalid SIMA_NEAT_MEMORY_BACKEND='") + raw +
      "' (expected legacy or dmabuf-plan)");
}

const ProcessMemoryBackendSelection& process_memory_backend_selection() {
  static const ProcessMemoryBackendSelection selection = [] {
    const char* raw = std::getenv("SIMA_NEAT_MEMORY_BACKEND");
    return ProcessMemoryBackendSelection{parse_memory_backend_policy(raw), raw != nullptr};
  }();
  return selection;
}

} // namespace simaai::neat::pipeline_internal

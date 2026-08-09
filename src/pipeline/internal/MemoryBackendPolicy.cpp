#include "pipeline/internal/MemoryBackendPolicy.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace simaai::neat::pipeline_internal {

MemoryBackendPolicy selected_memory_backend_policy() {
  const char* raw = std::getenv("SIMA_NEAT_MEMORY_BACKEND");
  if (!raw || !*raw || std::strcmp(raw, "legacy") == 0) {
    return MemoryBackendPolicy::Legacy;
  }
  if (std::strcmp(raw, "dmabuf-plan") == 0) {
    return MemoryBackendPolicy::DmaBufPlan;
  }
  throw std::runtime_error(
      std::string("invalid SIMA_NEAT_MEMORY_BACKEND='") + raw +
      "' (expected legacy or dmabuf-plan)");
}

} // namespace simaai::neat::pipeline_internal

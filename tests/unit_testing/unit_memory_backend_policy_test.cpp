#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/MemoryBackendPolicy.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

#define CHECK(expr)                                                                                \
  do {                                                                                             \
    if (!(expr)) {                                                                                 \
      std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK failed: " #expr << '\n';                \
      ++failures;                                                                                  \
    }                                                                                              \
  } while (false)

using simaai::neat::pipeline_internal::MemoryBackendPolicy;

void expect_rejected(const char* value) {
  bool rejected = false;
  try {
    (void)simaai::neat::pipeline_internal::parse_memory_backend_policy(value);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  CHECK(rejected);
}

} // namespace

int main() {
  using namespace simaai::neat::pipeline_internal;

  CHECK(parse_memory_backend_policy(nullptr) == MemoryBackendPolicy::Legacy);
  CHECK(parse_memory_backend_policy("legacy") == MemoryBackendPolicy::Legacy);
  CHECK(parse_memory_backend_policy("dmabuf-plan") == MemoryBackendPolicy::DmaBufPlan);
  CHECK(std::string(memory_backend_policy_name(MemoryBackendPolicy::Legacy)) == "legacy");
  CHECK(std::string(memory_backend_policy_name(MemoryBackendPolicy::DmaBufPlan)) == "dmabuf-plan");

  expect_rejected("");
  expect_rejected("Legacy");
  expect_rejected("DMABUF-PLAN");
  expect_rejected(" dmabuf-plan");
  expect_rejected("dmabuf-plan ");
  expect_rejected("auto");
  expect_rejected("probe");

  CHECK(::setenv("SIMA_NEAT_MEMORY_BACKEND", "dmabuf-plan", 1) == 0);
  const auto& first = process_memory_backend_selection();
  CHECK(first.policy == MemoryBackendPolicy::DmaBufPlan);
  CHECK(first.explicitly_configured);
  CHECK(::setenv("SIMA_NEAT_MEMORY_BACKEND", "legacy", 1) == 0);
  const auto& second = process_memory_backend_selection();
  CHECK(&first == &second);
  CHECK(second.policy == MemoryBackendPolicy::DmaBufPlan);

  if (failures != 0) {
    std::cerr << failures << " memory-backend policy checks failed\n";
    return 1;
  }
  std::cout << "memory-backend policy checks passed\n";
  return 0;
}

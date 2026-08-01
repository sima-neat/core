#include "pipeline/graph/internal/GraphTestHooks.h"

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace simaai::neat::session_test {
namespace {

std::mutex& manifest_mutex() {
  static std::mutex m;
  return m;
}

std::vector<pipeline_internal::sima::SimaPluginStaticManifest>& manifest_store() {
  static std::vector<pipeline_internal::sima::SimaPluginStaticManifest> manifests;
  return manifests;
}

std::atomic<CompositionFailurePoint> composition_failure_point{CompositionFailurePoint::None};
std::atomic<std::size_t> composition_failure_hits{0U};

} // namespace

void arm_composition_failure_for_test(CompositionFailurePoint point,
                                      std::size_t successful_hits_before_failure) {
  composition_failure_hits.store(successful_hits_before_failure, std::memory_order_relaxed);
  composition_failure_point.store(point, std::memory_order_release);
}

void clear_composition_failure_for_test() noexcept {
  composition_failure_point.store(CompositionFailurePoint::None, std::memory_order_release);
  composition_failure_hits.store(0U, std::memory_order_relaxed);
}

void maybe_throw_composition_failure_for_test(CompositionFailurePoint point) {
  if (composition_failure_point.load(std::memory_order_acquire) != point) {
    return;
  }
  std::size_t remaining = composition_failure_hits.load(std::memory_order_relaxed);
  while (remaining > 0U) {
    if (composition_failure_hits.compare_exchange_weak(
            remaining, remaining - 1U, std::memory_order_acq_rel, std::memory_order_relaxed)) {
      return;
    }
  }
  clear_composition_failure_for_test();
  throw std::runtime_error("injected Graph composition failure");
}

void reset_rendered_manifests() {
  std::lock_guard<std::mutex> lock(manifest_mutex());
  manifest_store().clear();
}

std::vector<pipeline_internal::sima::SimaPluginStaticManifest> get_rendered_manifests() {
  std::lock_guard<std::mutex> lock(manifest_mutex());
  return manifest_store();
}

void record_rendered_manifest(const pipeline_internal::sima::SimaPluginStaticManifest& manifest) {
  std::lock_guard<std::mutex> lock(manifest_mutex());
  manifest_store().push_back(manifest);
}

} // namespace simaai::neat::session_test

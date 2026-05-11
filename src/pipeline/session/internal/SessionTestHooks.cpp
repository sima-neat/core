#include "pipeline/session/internal/SessionTestHooks.h"

#include <mutex>
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

} // namespace

void reset_rendered_manifests() {
  std::lock_guard<std::mutex> lock(manifest_mutex());
  manifest_store().clear();
}

std::vector<pipeline_internal::sima::SimaPluginStaticManifest> get_rendered_manifests() {
  std::lock_guard<std::mutex> lock(manifest_mutex());
  return manifest_store();
}

void record_rendered_manifest(
    const pipeline_internal::sima::SimaPluginStaticManifest& manifest) {
  std::lock_guard<std::mutex> lock(manifest_mutex());
  manifest_store().push_back(manifest);
}

} // namespace simaai::neat::session_test

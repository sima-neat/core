// Executable-local allocation fault against the real context/manifest provider.
// No production fault hook, copied implementation, numerical admission or device access.
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include <gst/SimaPluginStaticManifestAbi.h>
#include <gst/gst.h>
#include <dlfcn.h>
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>

namespace {
thread_local bool arm_copy_failure = false;
thread_local bool fail_selected_allocation = false;
thread_local std::size_t contexts_created = 0, contexts_destroyed = 0, failures = 0;
constexpr std::size_t copied_string_bytes = 8192;
void need(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
} // namespace
void* operator new(std::size_t bytes) {
  if (fail_selected_allocation && bytes == copied_string_bytes) {
    fail_selected_allocation = false;
    ++failures;
    throw std::bad_alloc();
  }
  if (auto* result = std::malloc(bytes ? bytes : 1))
    return result;
  throw std::bad_alloc();
}
// All replaced unaligned forms use malloc/free. Aligned forms are untouched and
// remain paired in the standard library; std::string uses ordinary allocation.
void* operator new[](std::size_t bytes) {
  return ::operator new(bytes);
}
void* operator new(std::size_t bytes, const std::nothrow_t&) noexcept {
  try {
    return ::operator new(bytes);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t bytes, const std::nothrow_t&) noexcept {
  try {
    return ::operator new[](bytes);
  } catch (...) {
    return nullptr;
  }
}
void operator delete(void* memory) noexcept {
  std::free(memory);
}
void operator delete[](void* memory) noexcept {
  std::free(memory);
}
void operator delete(void* memory, std::size_t) noexcept {
  std::free(memory);
}
void operator delete[](void* memory, std::size_t) noexcept {
  std::free(memory);
}
void operator delete(void* memory, const std::nothrow_t&) noexcept {
  std::free(memory);
}
void operator delete[](void* memory, const std::nothrow_t&) noexcept {
  std::free(memory);
}
extern "C" GstContext* gst_context_new(const gchar* type, gboolean persistent) {
  using Create = GstContext* (*)(const gchar*, gboolean);
  static auto create = reinterpret_cast<Create>(dlsym(RTLD_NEXT, "gst_context_new"));
  if (!create)
    std::abort();
  auto* context = create(type, persistent);
  if (context && arm_copy_failure && g_str_equal(type, SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE)) {
    arm_copy_failure = false;
    ++contexts_created;
    gst_mini_object_weak_ref(
        GST_MINI_OBJECT(context), +[](gpointer, GstMiniObject*) { ++contexts_destroyed; }, nullptr);
    // Arm only after the genuine context exists. The source manifest's first
    // string copy in ManifestContextOwner then throws, not its allocation.
    fail_selected_allocation = true;
  }
  return context;
}
int exercise() {
  gst_init(nullptr, nullptr);
  auto* pipeline = gst_pipeline_new("throwing-copy");
  simaai::neat::pipeline_internal::sima::SimaPluginStaticManifest manifest;
  manifest.session_id = std::string(copied_string_bytes - 1, 'x');
  bool threw = false;
  arm_copy_failure = true;
  try {
    std::string error;
    simaai::neat::pipeline_internal::sima::attach_manifest_context(pipeline, manifest, &error);
  } catch (const std::bad_alloc&) {
    threw = true;
  }
  gst_object_unref(pipeline);
  need(threw && failures == 1 && contexts_created == 1,
       "real metadata owner string-copy exception was not observed");
  need(contexts_destroyed == 1, "GstContext leaked while metadata owner copy threw");
  std::cout << "[OK] real GstContext created, actual ManifestContextOwner string copy threw "
               "bad_alloc, original exception preserved, context weak-finalized once\n";
  return 0;
}
int main() {
  try {
    return exercise();
  } catch (const std::exception& failure) {
    std::cerr << "[FAIL] " << failure.what() << "\n";
    return 1;
  }
}

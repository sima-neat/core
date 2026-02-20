#include "pipeline/internal/SyncBuild.h"

#include <atomic>
#include <cstdio>

namespace simaai::neat::pipeline_internal {

thread_local bool g_sync_build_mode = false;
static std::atomic<bool> g_sync_warned{false};

void set_sync_build_mode(bool enabled) {
  g_sync_build_mode = enabled;
}

bool sync_build_mode() {
  return g_sync_build_mode;
}

void warn_sync_override(const char* group_name) {
  if (!g_sync_warned.exchange(true)) {
    std::fprintf(stderr,
                 "[sync] %s: run() enforces sync_mode=true; "
                 "async settings ignored. Use build()/push()/pull() for async.\n",
                 group_name ? group_name : "NodeGroup");
  }
}

ScopedSyncBuild::ScopedSyncBuild(bool enabled) {
  prev_ = g_sync_build_mode;
  g_sync_build_mode = enabled;
}

ScopedSyncBuild::~ScopedSyncBuild() {
  g_sync_build_mode = prev_;
}

} // namespace simaai::neat::pipeline_internal

#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

namespace simaai::neat::pipeline_internal {

// Thread-local flag to indicate sync build mode for backend_fragment generation.
void set_sync_build_mode(bool enabled);
bool sync_build_mode();

void warn_sync_override(const char* group_name);

class ScopedSyncBuild {
public:
  explicit ScopedSyncBuild(bool enabled);
  ~ScopedSyncBuild();

private:
  bool prev_ = false;
};

} // namespace simaai::neat::pipeline_internal

#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

namespace simaai::neat::pipeline_internal {

// InputStream teardown has two different synchronous requirements. Live
// sources prefer a bounded NULL transition, but may still be handed to the
// reaper if it cannot finish. Driver-backed async stages must retain their
// owning runtime until their stop callback has reaped every submitted job.
enum class InputStreamTeardownPolicy {
  Deferred,
  BoundedPreferred,
  MustReachNull,
};

} // namespace simaai::neat::pipeline_internal

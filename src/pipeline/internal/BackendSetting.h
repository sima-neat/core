#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// The process's backend (neat/runtime.h: set_backend, backend): one value for the whole process,
// fixed by the first Model constructed or Graph built. A Model's constructor and every call that
// builds a Run take it from use_backend(); a call that exists on the GStreamer backend only asks
// require_gstreamer_backend(). Compiled in every build of the library; one that has no Dataflow
// runtime starts on GStreamer and stays there.

#include "pipeline/Run.h"
#include "pipeline/internal/DataflowNotCarried.h"

namespace simaai::neat::pipeline_internal {

// The backend of the Model being constructed or the Graph being built. The first call fixes the
// process's backend: from then on set_backend() with the other value throws. Any thread.
RunBackend use_backend() noexcept;

// For a call that only the GStreamer backend carries (`where`: the customer's call): takes the
// process's backend as use_backend() does and, when it is Dataflow, throws the not-carried error
// of `what`.
void require_gstreamer_backend(const char* where, const DataflowNotCarried& what);

// Whether this library was built with the Dataflow runtime.
bool has_dataflow_backend() noexcept;

} // namespace simaai::neat::pipeline_internal

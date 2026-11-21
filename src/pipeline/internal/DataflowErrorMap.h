#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/GraphOptions.h"
#include "pipeline/NeatError.h"
#include "pipeline/internal/DataflowRuntime.h"

#include <string>
#include <string_view>

#pragma GCC visibility push(hidden)

namespace simaai::neat::pipeline_internal {

// One row of the Dataflow error table: what a runtime Error (code, origin, phase, operator) becomes
// on Core's surface. `status` is what a pull reports; `code` is the error_codes:: family a build,
// push or pull error carries.
struct DataflowErrorMapping {
  PullStatus status = PullStatus::Error;
  const char* code = "";
  bool terminal = true; ///< latched as the Run's last_error() when seen by a pull
  /// Raised while a Run is built on Dataflow, the error is Dataflow's own: RunBackend::Auto builds
  /// the Run on GStreamer instead (DataflowBackendSelect.cpp). False where GStreamer would meet
  /// the same error: the process lost a device, or the MLA does not load the model's file.
  bool retry_on_gstreamer = true;
  const char* hint = ""; ///< what to do, where the runtime's text does not say it
};

// The table: one lookup for every runtime Error Core can meet.
DataflowErrorMapping map_dataflow_error(const df::Error& error) noexcept;

// "<where>: <runtime message> (Dataflow backend)", the Core text of a runtime Error.
std::string dataflow_error_text(const df::Error& error, std::string_view where);

// The NeatError of a runtime Error whose row says retry_on_gstreamer false. RunBackend::Auto lets
// it out of Graph::build as it is; an application catches it as any NeatError.
class DataflowErrorOnEveryBackend final : public NeatError {
public:
  using NeatError::NeatError;
};

// Throws the NeatError the table gives for `error` (build, push and other non-pull paths), with
// `detail` as a second line and `hint` after the row's own.
[[noreturn]] void throw_dataflow_error(const df::Error& error, std::string_view where,
                                       std::string_view detail = {}, std::string_view hint = {});

} // namespace simaai::neat::pipeline_internal

#pragma GCC visibility pop

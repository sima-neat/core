#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/GraphOptions.h"
#include "pipeline/NeatError.h"
#include "pipeline/internal/DataflowNotCarried.h"
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
  /// The error is the Dataflow backend's own: what to do about it is to run the program on the
  /// GStreamer backend (dataflow_backend_action). False where GStreamer would meet the same
  /// error: the process lost a device, or the MLA does not load the model's file.
  bool backend_action = true;
  const char* hint = ""; ///< what else to do, where the runtime's text does not say it
};

// Where a runtime Error reaches the application, which decides the action its text carries.
enum class DataflowErrorSite : unsigned char {
  /// While a Run is built (Graph::build): a row's backend action is DataflowRemedy::Backend.
  Build,
  /// While an input that binds at its first push is built there: the same action, and how to
  /// find the error at the build.
  FirstPush,
  /// On a running Session (a push, an alloc): only what the backend does not implement
  /// (ErrorCode::Unsupported) names the backend action; a failure of a device or of the
  /// application's call names none.
  Running,
};

// The table: one lookup for every runtime Error Core can meet.
DataflowErrorMapping map_dataflow_error(const df::Error& error) noexcept;

// "<where>: <runtime message> (Dataflow backend)", the Core text of a runtime Error.
std::string dataflow_error_text(const df::Error& error, std::string_view where);

// Throws the NeatError the table gives for `error` (build, push and other non-pull paths), with
// `detail` as a second line and, as its hint, the row's own and the action `site` gives the row.
[[noreturn]] void throw_dataflow_error(const df::Error& error, std::string_view where,
                                       std::string_view detail = {},
                                       DataflowErrorSite site = DataflowErrorSite::Running);

} // namespace simaai::neat::pipeline_internal

#pragma GCC visibility pop

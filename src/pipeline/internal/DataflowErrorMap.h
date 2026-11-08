#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/GraphOptions.h"
#include "pipeline/internal/DataflowRuntime.h"

#include <string>
#include <string_view>

#pragma GCC visibility push(hidden)

namespace simaai::neat::pipeline_internal {

// One row of the Dataflow error table: what a runtime Error (code, phase, origin) becomes on
// Core's surface. `status` is what a pull reports; `code` is the error_codes:: family a build,
// push or pull error carries.
struct DataflowErrorMapping {
  PullStatus status = PullStatus::Error;
  const char* code = "";
  bool terminal = true; // latched as the Run's last_error() when seen by a pull
};

// The table (H0 plan 3.8): one lookup for every runtime Error Core can meet.
DataflowErrorMapping map_dataflow_error(const df::Error& error) noexcept;

// "<where>: <runtime message> (Dataflow backend)", the Core text of a runtime Error.
std::string dataflow_error_text(const df::Error& error, std::string_view where);

// Throws the NeatError the table gives for `error` (build, push and other non-pull paths).
[[noreturn]] void throw_dataflow_error(const df::Error& error, std::string_view where,
                                       std::string_view detail = {});

} // namespace simaai::neat::pipeline_internal

#pragma GCC visibility pop

#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/SessionReport.h"
#include "pipeline/ErrorCodes.h"

#include <string>

namespace simaai::neat::pipeline_internal {

constexpr const char* kDispatcherUnavailableError =
    simaai::neat::error_codes::kDispatcherUnavailable;
constexpr const char* kDispatcherUnavailableErrorLegacy =
    simaai::neat::error_codes::kDispatcherUnavailableLegacy;

bool match_dispatcher_unavailable(const std::string& message);
bool is_dispatcher_unavailable(const SessionReport& report);
bool attempt_dispatcher_recovery(SessionReport* report, bool auto_recover);

} // namespace simaai::neat::pipeline_internal

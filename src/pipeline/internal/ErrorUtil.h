#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/SessionError.h"
#include "pipeline/SessionOptions.h"
#include "pipeline/SessionReport.h"

#include <optional>
#include <string>
#include <string_view>

namespace simaai::neat::pipeline_internal::error_util {

std::string decorate_error(std::string_view code, std::string_view message);
std::string append_hint(std::string_view message, std::string_view hint);

SessionReport make_report(std::string_view code, std::string_view summary,
                          std::string_view pipeline_string = {}, std::string_view hint = {});

[[noreturn]] void throw_session_error(std::string_view code, std::string_view summary,
                                      std::string_view pipeline_string = {},
                                      std::string_view hint = {});

void set_pull_error(PullError* err, std::string code, std::string message,
                    std::optional<SessionReport> report = std::nullopt);

} // namespace simaai::neat::pipeline_internal::error_util

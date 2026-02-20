/**
 * @file
 * @ingroup diagnostics
 * @brief SessionError exception carrying a SessionReport.
 */
#pragma once

#include "pipeline/SessionReport.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace simaai::neat {

// Exception that carries a structured report.
class SessionError : public std::runtime_error {
public:
  explicit SessionError(std::string msg) : std::runtime_error(std::move(msg)), report_{} {}
  SessionError(std::string msg, SessionReport report)
      : std::runtime_error(std::move(msg)), report_(std::move(report)) {}
  const SessionReport& report() const {
    return report_;
  }

private:
  SessionReport report_;
};

} // namespace simaai::neat

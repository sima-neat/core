/**
 * @file
 * @ingroup diagnostics
 * @brief SessionError — the framework's only exception type, carrying a structured SessionReport.
 *
 * Every `Session::build()`, `Model::run()`, `Run::push()`, etc. that fails throws `SessionError`.
 * The exception's `what()` string is human-readable; `.report()` returns the structured
 * `SessionReport` with `error_code`, `repro_note`, GStreamer bus messages, a standalone
 * `gst-launch-1.0` reproducer, and DOT graph paths. Catch this exception type to handle all
 * framework failures uniformly.
 */
#pragma once

#include "pipeline/SessionReport.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace simaai::neat {

/**
 * @brief Framework exception type carrying a structured `SessionReport`.
 *
 * Catch `SessionError` (or `std::runtime_error`) from any framework call that can fail. Use
 * the `report()` accessor to get machine-readable error details for triage.
 *
 * @code
 *   try {
 *     auto run = sess.build(input);
 *   } catch (const sima::SessionError& e) {
 *     std::cerr << e.what() << "\n";
 *     auto& r = e.report();
 *     std::cerr << "error_code: " << r.error_code << "\n"
 *               << "repro: " << r.repro_gst_launch << "\n";
 *   }
 * @endcode
 *
 * @see SessionReport for the structured payload
 * @see "Validation API" (§29 of the design deep dive)
 * @ingroup diagnostics
 */
class SessionError : public std::runtime_error {
public:
  /// Construct from a message only (no structured report — used for early/internal failures).
  explicit SessionError(std::string msg) : std::runtime_error(std::move(msg)), report_{} {}
  /// Construct from a message and an attached structured report.
  SessionError(std::string msg, SessionReport report)
      : std::runtime_error(std::move(msg)), report_(std::move(report)) {}
  /// Returns the structured `SessionReport` (machine-readable error details for triage).
  const SessionReport& report() const {
    return report_;
  }

private:
  SessionReport report_;
};

} // namespace simaai::neat

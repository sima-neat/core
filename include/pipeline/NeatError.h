/**
 * @file
 * @ingroup diagnostics
 * @brief NeatError — the framework's only exception type, carrying a structured GraphReport.
 *
 * Every `Graph::build()`, `Model::run()`, `Run::push()`, etc. that fails throws `NeatError`.
 * The exception's `what()` string is human-readable; `.report()` returns the structured
 * `GraphReport` with `error_code`, `repro_note`, GStreamer bus messages, a standalone
 * `gst-launch-1.0` reproducer, and DOT graph paths. Catch this exception type to handle all
 * framework failures uniformly.
 */
#pragma once

#include "pipeline/GraphReport.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace simaai::neat {

/**
 * @brief Framework exception type carrying a structured `GraphReport`.
 *
 * Catch `NeatError` (or `std::runtime_error`) from any framework call that can fail. Use
 * the `report()` accessor to get machine-readable error details for triage.
 *
 * @code
 *   try {
 *     auto run = graph.build(input);
 *   } catch (const sima::NeatError& e) {
 *     std::cerr << e.what() << "\n";
 *     auto& r = e.report();
 *     std::cerr << "error_code: " << r.error_code << "\n"
 *               << "repro: " << r.repro_gst_launch << "\n";
 *   }
 * @endcode
 *
 * @see GraphReport for the structured payload
 * @see "Validation API" (§29 of the design deep dive)
 * @ingroup diagnostics
 */
class NeatError : public std::runtime_error {
public:
  /// Construct from a message only (no structured report — used for early/internal failures).
  explicit NeatError(std::string msg) : std::runtime_error(std::move(msg)), report_{} {}
  /// Construct from a message and an attached structured report.
  NeatError(std::string msg, GraphReport report)
      : std::runtime_error(std::move(msg)), report_(std::move(report)) {}
  /// Returns the structured `GraphReport` (machine-readable error details for triage).
  const GraphReport& report() const {
    return report_;
  }

private:
  GraphReport report_;
};

} // namespace simaai::neat

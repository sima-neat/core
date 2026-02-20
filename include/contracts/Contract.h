/**
 * @file
 * @ingroup contracts
 * @brief Builder-level validation contracts.
 */
// include/contracts/Contract.h
#pragma once

#include <memory>
#include <string>

#include "builder/NodeGroup.h"
#include "contracts/ContractTypes.h"
#include "contracts/ValidationReport.h"

namespace simaai::neat {

// Forward-declare policy surface (contracts may consult it, but must remain STL-only).
class Policy;

/**
 * @brief Context passed to contracts during validation.
 *
 * Contracts are *builder-level* checks (no GStreamer). They validate structural
 * expectations before Session builds a gst-launch string.
 */
struct ValidationContext {
  enum class Mode {
    Validate = 0, // "validate()" path (structural checks only, no PLAYING implied here)
    Run,          // "run()" path (expects Output terminal)
    Rtsp,         // "run_rtsp()" path (expects StillImageInput, encoder/pay if you model them)
  };

  Mode mode = Mode::Validate;

  // Desired runner memory posture (used only for reporting / soft checks at builder-level).
  MemoryContract runner_memory_contract = MemoryContract::AllowEitherButReport;

  // Optional: a policy bundle. Contracts must not require it.
  const Policy* policy = nullptr;

  // If false, contracts should prefer WARN over ERROR when reasonable.
  bool strict = true;
};

/**
 * @brief A single validation rule.
 *
 * Contracts must:
 * - never include or call GStreamer APIs
 * - only use Node/NodeGroup/Graph/policy and STL
 * - report issues via ValidationReport (do not throw for normal violations)
 */
class Contract {
public:
  virtual ~Contract() = default;

  /// Stable identifier (used for filtering, report grouping).
  virtual std::string id() const = 0;

  /// Human-readable description (optional).
  virtual std::string description() const {
    return "";
  }

  /// Validate the node group and append issues to report.
  virtual void validate(const NodeGroup& nodes, const ValidationContext& ctx,
                        ValidationReport& report) const = 0;
};

} // namespace simaai::neat

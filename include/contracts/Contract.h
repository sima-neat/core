/**
 * @file
 * @ingroup contracts
 * @brief Builder-level validation contracts.
 *
 * Defines the `Contract` base class — a typed predicate over a `NodeGroup` —
 * plus the `ValidationContext` passed to each contract during validation.
 * Contracts are pure builder-level checks (STL only, no GStreamer); they
 * report findings through `ValidationReport` rather than throwing.
 *
 * @see ContractRegistry
 * @see Validators
 * @see ValidationReport
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
 *
 * @ingroup contracts
 * @see Contract
 */
struct ValidationContext {
  /**
   * @brief Which Session entry point is being validated.
   *
   * Some contracts only fire in specific modes (e.g., `SinkLastForRun` only
   * checks for the `Output` terminal in `Run` mode).
   */
  enum class Mode {
    Validate = 0, ///< `validate()` path (structural checks only, no PLAYING implied).
    Run,          ///< `run()` path (expects an Output terminal).
    Rtsp,         ///< `run_rtsp()` path (expects StillImageInput, encoder/pay if modeled).
  };

  Mode mode = Mode::Validate;  ///< Active Session mode.

  /// @brief Desired runner memory posture (used only for reporting / soft checks).
  MemoryContract runner_memory_contract = MemoryContract::AllowEitherButReport;

  /// @brief Optional policy bundle. Contracts must not require it (may be null).
  const Policy* policy = nullptr;

  /// @brief If false, contracts should prefer WARN over ERROR when reasonable.
  bool strict = true;
};

/**
 * @brief A single validation rule applied to a NodeGroup.
 *
 * Contracts must:
 * - never include or call GStreamer APIs
 * - only use Node/NodeGroup/Graph/policy and STL
 * - report issues via ValidationReport (do not throw for normal violations)
 *
 * Concrete contracts implement `id()`, optionally `description()`, and the
 * core `validate()` method that inspects the NodeGroup and appends issues to
 * the report.
 *
 * @ingroup contracts
 * @see ContractRegistry
 * @see ValidationReport
 * @see Validators (built-in implementations)
 */
class Contract {
public:
  /// Virtual destructor — required for the abstract base.
  virtual ~Contract() = default;

  /// @brief Stable identifier (used for filtering, report grouping).
  virtual std::string id() const = 0;

  /// @brief Human-readable description (optional).
  virtual std::string description() const {
    return "";
  }

  /**
   * @brief Validate the node group and append issues to `report`.
   *
   * @param nodes  The NodeGroup being validated.
   * @param ctx    Validation context (mode, policy, strictness).
   * @param report Output report; the contract appends issues to it.
   */
  virtual void validate(const NodeGroup& nodes, const ValidationContext& ctx,
                        ValidationReport& report) const = 0;
};

} // namespace simaai::neat

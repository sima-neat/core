/**
 * @file
 * @ingroup internal_sima
 * @brief **Framework internal — reach-through tier.** Run-target policy for process-CVU stages.
 *
 * Each process-CVU stage can run on the EVxx CVU or on the A65 host fallback. This header
 * exposes the policy surface the planner uses to: (a) read the backend capabilities advertised
 * by a stage payload, (b) resolve a final run-target/exec-backend decision given the user's
 * compile input, and (c) stamp the resolved values back onto the stage payload.
 *
 * @see ProcessCvuStagePayload (in SimaPluginStaticManifest.h)
 */
#pragma once

#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include <string>
#include <string_view>

namespace simaai::neat::pipeline_internal::sima {

/// Which executor a process-CVU stage actually runs on after policy resolution.
enum class ProcessCvuResolvedExecBackend : std::uint8_t {
  Evxx = 0, ///< Run on the on-die EVxx CVU.
  A65 = 1,  ///< Run on the A65 host fallback.
};

/**
 * @brief Backend capabilities advertised by a process-CVU stage payload.
 *
 * Reports whether the stage's compiled artifacts support EV74 and/or A65, and what the
 * `AUTO` policy resolves to for this stage.
 */
struct ProcessCvuBackendCapabilities {
  bool supports_ev74 = true;            ///< Stage has an EV74 path.
  bool supports_a65 = true;             ///< Stage has an A65 path.
  std::string auto_run_target = "AUTO"; ///< Run-target token chosen by `AUTO`.
  ProcessCvuResolvedExecBackend auto_exec_backend =
      ProcessCvuResolvedExecBackend::Evxx; ///< Backend chosen by `AUTO`.
  std::string reason;                      ///< Why `AUTO` resolved this way (diagnostics).
};

/**
 * @brief Final run-target decision for a process-CVU stage.
 *
 * Captures the user's request, the effective token after normalization, the resolved backend,
 * and whether a fallback path was taken.
 */
struct ProcessCvuBackendDecision {
  std::string requested_run_target = "AUTO"; ///< What the user asked for.
  std::string effective_run_target = "AUTO"; ///< The token actually applied.
  ProcessCvuResolvedExecBackend resolved_exec_backend =
      ProcessCvuResolvedExecBackend::Evxx; ///< Resolved backend.
  std::string reason; ///< Human-readable reason for the decision (for diagnostics).
};

/// Normalize a run-target token (case-fold, alias collapse) — e.g., `"a65"` -> `"A65"`.
std::string normalize_processcvu_run_target_token(std::string value);

/// Stable string form of a `ProcessCvuResolvedExecBackend` value.
const char* processcvu_resolved_exec_backend_token(ProcessCvuResolvedExecBackend backend);

/// Inspect a stage payload and report what backends it can run on.
ProcessCvuBackendCapabilities
processcvu_backend_capabilities(const ProcessCvuStagePayload& payload);

/// Compute the final run-target decision for a stage given the compile input.
/// @throws std::invalid_argument when an explicit target is unsupported by the stage.
ProcessCvuBackendDecision
resolve_processcvu_backend_decision(const ProcessCvuStagePayload& payload,
                                    const ContractCompileInput& compile_input,
                                    std::string_view stage_identity = {});

/// Resolve the run-target and stamp the result back onto the stage payload in place.
void resolve_processcvu_run_target(ProcessCvuStagePayload* payload,
                                   const ContractCompileInput& compile_input,
                                   std::string_view stage_identity = {});

} // namespace simaai::neat::pipeline_internal::sima

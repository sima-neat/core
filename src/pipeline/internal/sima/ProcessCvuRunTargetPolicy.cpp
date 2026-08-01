#include "pipeline/internal/sima/ProcessCvuRunTargetPolicy.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace simaai::neat::pipeline_internal::sima {
namespace {

std::string upper_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

// SIMA_PROCESSCVU_RUN_TARGET is retained as a legacy/debug override for
// callers that do not use explicit Model/Graph processcvu placement.
std::string env_run_target_override() {
  const char* raw = std::getenv("SIMA_PROCESSCVU_RUN_TARGET");
  if (!raw || !*raw) {
    return {};
  }
  std::string v = upper_copy(std::string(raw));
  if (v == "A65" || v == "APU" || v == "CPU" || v == "SIMA_EV_RUN_A65") {
    return "A65";
  }
  if (v == "EV" || v == "EVXX" || v == "CVU" || v == "EV74" || v == "SIMA_EV_RUN_EV74") {
    return "EV74";
  }
  if (v == "AUTO" || v == "SIMA_EV_RUN_AUTO") {
    return "AUTO";
  }
  return {};
}

std::string normalize_processcvu_run_target_token_no_env(std::string value) {
  value = upper_copy(std::move(value));
  if (value.empty()) {
    return "AUTO";
  }
  if (value == "APU" || value == "CPU" || value == "SIMA_EV_RUN_A65") {
    return "A65";
  }
  if (value == "EV" || value == "EVXX" || value == "CVU" || value == "SIMA_EV_RUN_EV74") {
    return "EV74";
  }
  if (value == "SIMA_EV_RUN_AUTO") {
    return "AUTO";
  }
  if (value == "A65" || value == "EV74" || value == "AUTO") {
    return value;
  }
  return "AUTO";
}

bool explicit_run_target_token(const std::string& value) {
  return normalize_processcvu_run_target_token_no_env(value) != "AUTO";
}

enum class ProcessCvuStageRole {
  Unknown,
  Pre,
  Post,
};

ProcessCvuStageRole processcvu_stage_role(const ProcessCvuStagePayload& payload,
                                          std::string_view stage_identity = {}) {
  switch (payload.graph_family_enum) {
  case ProcessCvuGraphFamily::Preproc:
  case ProcessCvuGraphFamily::Quant:
  case ProcessCvuGraphFamily::Tess:
  case ProcessCvuGraphFamily::QuantTess:
  case ProcessCvuGraphFamily::CastTess:
  case ProcessCvuGraphFamily::VisualFrontend:
    return ProcessCvuStageRole::Pre;
  case ProcessCvuGraphFamily::Detess:
  case ProcessCvuGraphFamily::Dequant:
  case ProcessCvuGraphFamily::DetessCast:
  case ProcessCvuGraphFamily::DetessDequant:
    return ProcessCvuStageRole::Post;
  case ProcessCvuGraphFamily::Cast: {
    // Cast can appear on either side. Prefer the rendered model-stage identity
    // ("pre_*"/"post_*") over the exact MPK plugin name: post-side casts often
    // preserve an MPK producer name like "cast_2", which is not enough to infer
    // role. Bare casts with no post/output identity are pre-adapters.
    const std::string identity = lower_copy(std::string(stage_identity));
    if (identity.find("post") != std::string::npos ||
        identity.find("output") != std::string::npos) {
      return ProcessCvuStageRole::Post;
    }
    if (identity.find("pre") != std::string::npos || identity.find("input") != std::string::npos) {
      return ProcessCvuStageRole::Pre;
    }
    const std::string stage = lower_copy(payload.exact_stage_name_or_id);
    if (stage.find("post") != std::string::npos || stage.find("output") != std::string::npos) {
      return ProcessCvuStageRole::Post;
    }
    return ProcessCvuStageRole::Pre;
  }
  case ProcessCvuGraphFamily::Unknown:
  default:
    break;
  }
  return ProcessCvuStageRole::Unknown;
}

struct ExplicitProcessCvuTarget {
  std::string run_target;
  std::string source;
};

[[noreturn]] void throw_unsupported_explicit_target(const std::string& requested_target,
                                                    const ProcessCvuBackendCapabilities& caps,
                                                    const std::string& requested_source,
                                                    std::string_view stage_identity) {
  std::string supported;
  if (caps.supports_ev74) {
    supported = "EV74";
  }
  if (caps.supports_a65) {
    if (!supported.empty()) {
      supported += ", ";
    }
    supported += "A65";
  }
  if (supported.empty()) {
    supported = "none";
  }

  std::string message = "processcvu stage";
  if (!stage_identity.empty()) {
    message += " '" + std::string(stage_identity) + "'";
  }
  message += " cannot run on explicitly requested target " + requested_target +
             " (source=" + requested_source + "). Supported target(s): " + supported +
             ". Use AUTO or a supported target.";
  throw std::invalid_argument(message);
}

std::optional<ExplicitProcessCvuTarget>
explicit_prepost_target(const ProcessCvuStagePayload& payload, const ProcessCvuOptions& options,
                        std::string_view stage_identity = {}) {
  switch (processcvu_stage_role(payload, stage_identity)) {
  case ProcessCvuStageRole::Pre:
    if (explicit_run_target_token(options.pre_run_target)) {
      return ExplicitProcessCvuTarget{
          normalize_processcvu_run_target_token_no_env(options.pre_run_target), "processcvu_pre"};
    }
    break;
  case ProcessCvuStageRole::Post:
    if (explicit_run_target_token(options.post_run_target)) {
      return ExplicitProcessCvuTarget{
          normalize_processcvu_run_target_token_no_env(options.post_run_target), "processcvu_post"};
    }
    break;
  case ProcessCvuStageRole::Unknown:
  default:
    break;
  }
  return std::nullopt;
}

} // namespace

std::string normalize_processcvu_run_target_token(std::string value) {
  if (auto override_v = env_run_target_override(); !override_v.empty()) {
    return override_v;
  }
  return normalize_processcvu_run_target_token_no_env(std::move(value));
}

const char* processcvu_resolved_exec_backend_token(ProcessCvuResolvedExecBackend backend) {
  switch (backend) {
  case ProcessCvuResolvedExecBackend::A65:
    return "A65";
  case ProcessCvuResolvedExecBackend::Evxx:
  default:
    return "EVXX";
  }
}

ProcessCvuBackendCapabilities
processcvu_backend_capabilities(const ProcessCvuStagePayload& payload) {
  ProcessCvuBackendCapabilities caps;
  caps.supports_ev74 = true;
  caps.supports_a65 = false;
  caps.auto_run_target = "AUTO";
  caps.auto_exec_backend = ProcessCvuResolvedExecBackend::Evxx;
  caps.reason = "generic_ev_default_policy";

  switch (payload.graph_family_enum) {
  case ProcessCvuGraphFamily::VisualFrontend:
    caps.supports_ev74 = true;
    caps.supports_a65 = false;
    caps.auto_run_target = "EV74";
    caps.auto_exec_backend = ProcessCvuResolvedExecBackend::Evxx;
    caps.reason = "native_visual_ev74_only";
    break;
  case ProcessCvuGraphFamily::Cast:
    caps.supports_ev74 = true;
    caps.supports_a65 = true;
    caps.auto_run_target = "EV74";
    caps.auto_exec_backend = ProcessCvuResolvedExecBackend::Evxx;
    caps.reason = "ev74_preferred_dual_backend";
    break;
  case ProcessCvuGraphFamily::Quant:
    caps.supports_ev74 = true;
    caps.supports_a65 = true;
    caps.auto_run_target = "EV74";
    caps.auto_exec_backend = ProcessCvuResolvedExecBackend::Evxx;
    caps.reason = "ev74_preferred_dual_backend";
    break;
  case ProcessCvuGraphFamily::QuantTess:
    caps.supports_ev74 = true;
    caps.supports_a65 = true;
    caps.auto_run_target = "EV74";
    caps.auto_exec_backend = ProcessCvuResolvedExecBackend::Evxx;
    caps.reason = "ev74_preferred_dual_backend";
    break;
  case ProcessCvuGraphFamily::CastTess:
    caps.supports_ev74 = true;
    caps.supports_a65 = true;
    caps.auto_run_target = "EV74";
    caps.auto_exec_backend = ProcessCvuResolvedExecBackend::Evxx;
    caps.reason = "ev74_preferred_dual_backend";
    break;
  case ProcessCvuGraphFamily::Dequant:
    caps.supports_ev74 = true;
    caps.supports_a65 = true;
    caps.auto_run_target = "EV74";
    caps.auto_exec_backend = ProcessCvuResolvedExecBackend::Evxx;
    caps.reason = "ev74_preferred_dual_backend";
    break;
  case ProcessCvuGraphFamily::DetessCast:
    // Graph 225 (detesscast) was previously A65-only because the legacy EV74
    // detesscast kernel produced accuracy-corrupt boxes on YOLOv8 BF16 mpk.
    // The Phase 1 port replaced that body with the FLAT tile-walker kernel
    // (ported from graph 227's d227 hot path, BF16->FP32 specialised), so
    // detesscast now ships dual-backend with EV74 as the preferred target.
    // NOTE: this flip is gated on the YOLOv8 BF16 inference smoke test from
    // porting_kernels.md Phase 5.3 -- the code below reflects the post-port
    // intent; if the matrix test regresses boxdecode accuracy the policy
    // must revert to A65-only here.
    caps.supports_ev74 = true;
    caps.supports_a65 = true;
    caps.auto_run_target = "EV74";
    caps.auto_exec_backend = ProcessCvuResolvedExecBackend::Evxx;
    caps.reason = "ev74_preferred_dual_backend";
    break;
  case ProcessCvuGraphFamily::DetessDequant:
    // Canonical pattern (matches cast/casttess/quantize/quanttess):
    //  - EV74: detessdequant.cc::detessdequant_ev74() dispatches the OCL hot
    //          path against libsima_detessdequant.a (7 specialized kernels:
    //          int8/16/32 input × fp16/fp32 output × NHWC/NCHW formats, plus
    //          large-tile variants).
    //  - A65 : detessdequant.cc::detessdequant_a65() runs the scalar
    //          reference using sima_ev_a65:: helpers.
    // Top-level entry detessdequant() switches on cfg->hdr.run_target.
    // Graph id 227 is the canonical detessdequant slot; the old
    // standalone detessdequant_opt anti-pattern was retired because
    // libsima_detessdequant already owns the OCL hot path.
    caps.supports_ev74 = true;
    caps.supports_a65 = true;
    caps.auto_run_target = "EV74";
    caps.auto_exec_backend = ProcessCvuResolvedExecBackend::Evxx;
    caps.reason = "ev74_preferred_dual_backend";
    break;
  default:
    break;
  }
  return caps;
}

ProcessCvuBackendDecision
resolve_processcvu_backend_decision(const ProcessCvuStagePayload& payload,
                                    const ContractCompileInput& compile_input,
                                    std::string_view stage_identity) {
  ProcessCvuBackendDecision decision;
  std::string requested_source = "legacy_or_env";
  if (explicit_run_target_token(payload.requested_run_target)) {
    decision.requested_run_target =
        normalize_processcvu_run_target_token_no_env(payload.requested_run_target);
    requested_source = "payload_stage";
  } else if (auto match =
                 explicit_prepost_target(payload, compile_input.processcvu, stage_identity)) {
    decision.requested_run_target = match->run_target;
    requested_source = match->source;
  } else {
    decision.requested_run_target =
        normalize_processcvu_run_target_token(compile_input.processcvu_requested_run_target);
  }
  const ProcessCvuBackendCapabilities caps = processcvu_backend_capabilities(payload);

  if (decision.requested_run_target == "A65") {
    if (caps.supports_a65) {
      decision.effective_run_target = "A65";
      decision.resolved_exec_backend = ProcessCvuResolvedExecBackend::A65;
      decision.reason = "requested_a65_supported:" + requested_source;
    } else {
      throw_unsupported_explicit_target(decision.requested_run_target, caps, requested_source,
                                        stage_identity);
    }
    return decision;
  }

  if (decision.requested_run_target == "EV74") {
    if (caps.supports_ev74) {
      decision.effective_run_target = "EV74";
      decision.resolved_exec_backend = ProcessCvuResolvedExecBackend::Evxx;
      decision.reason = "requested_ev74_supported:" + requested_source;
    } else {
      throw_unsupported_explicit_target(decision.requested_run_target, caps, requested_source,
                                        stage_identity);
    }
    return decision;
  }

  if (processcvu_stage_role(payload, stage_identity) == ProcessCvuStageRole::Post &&
      caps.supports_a65) {
    // AUTO should keep pre/adaptor stages on EV74, but post stages are CPU-facing
    // in the common terminal route and the A65 reference path is measurably
    // faster for YOLO INT8 post/dequant. Explicit per-stage/session/env targets
    // above still win; this only changes the unresolved AUTO policy.
    decision.effective_run_target = "A65";
    decision.resolved_exec_backend = ProcessCvuResolvedExecBackend::A65;
    decision.reason = "a65_preferred_auto_post:" + requested_source;
  } else {
    decision.effective_run_target =
        normalize_processcvu_run_target_token_no_env(caps.auto_run_target);
    decision.resolved_exec_backend = caps.auto_exec_backend;
    decision.reason = (caps.reason.empty() ? "auto_policy" : caps.reason) + ":" + requested_source;
  }
  return decision;
}

void resolve_processcvu_run_target(ProcessCvuStagePayload* payload,
                                   const ContractCompileInput& compile_input,
                                   std::string_view stage_identity) {
  if (!payload) {
    return;
  }
  const ProcessCvuBackendDecision decision =
      resolve_processcvu_backend_decision(*payload, compile_input, stage_identity);
  payload->requested_run_target = decision.requested_run_target;
  payload->run_target = decision.effective_run_target;
  payload->resolved_exec_backend =
      processcvu_resolved_exec_backend_token(decision.resolved_exec_backend);
  payload->run_target_resolution_reason = decision.reason;
}

} // namespace simaai::neat::pipeline_internal::sima

#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/MemoryBackendPolicy.h"
#include "pipeline/internal/sima/static_contract/ModelExecutionPlan.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal {

enum class DmabufEligibilityCode {
  NotEvaluated,
  Eligible,
  MissingMpkManifest,
  MissingMlaExecutable,
  ElfTopologyUnreadable,
  InvalidJson,
  MissingRequiredField,
  InvalidField,
  UnsupportedContractVersion,
  UnsupportedKernel,
  InvalidKernelArity,
  DuplicateSequence,
  DuplicateProducer,
  MissingProducer,
  ValueSizeMismatch,
  ConfigurationMismatch,
  MissingMlaStage,
  MultipleMlaStages,
  MissingPublicationStage,
  InvalidPublicationStage,
  ElfTopologyInvalid,
  ElfTopologyMismatch,
  PlanValidationFailed,
  ArenaPlanInvalid,
  IoError,
  InternalError,
};

const char* dmabuf_eligibility_code_name(DmabufEligibilityCode code) noexcept;

struct DmabufProofFact {
  std::string subject;
  std::string evidence;
};

struct DmabufEligibilityReport {
  DmabufEligibilityCode code = DmabufEligibilityCode::NotEvaluated;
  std::string source;
  std::string location;
  std::string detail;
  std::string contract_version;
  std::string artifact_digest;
  std::vector<DmabufProofFact> proof;

  [[nodiscard]] bool eligible() const noexcept {
    return code == DmabufEligibilityCode::Eligible;
  }
};

struct DmabufPlanCompileResult {
  std::optional<sima::static_contract::ModelExecutionPlan> plan;
  DmabufEligibilityReport report;
  std::string plan_digest;

  [[nodiscard]] bool eligible() const noexcept {
    return plan.has_value() && report.eligible() && !plan_digest.empty();
  }
};

// Pure structural admission. It reads only the explicitly supplied MPK manifest
// and MLA ELF. It allocates no device memory, opens no accelerator device, and
// publishes no plugin manifest.
DmabufPlanCompileResult
try_compile_dmabuf_plan(const std::filesystem::path& mpk_manifest,
                        const std::filesystem::path& mla_executable) noexcept;

// Stable canonical rendering and SHA-256 digest of the accepted immutable plan.
std::string canonical_dmabuf_plan_json(const sima::static_contract::ModelExecutionPlan& plan);
std::string dmabuf_plan_digest(const sima::static_contract::ModelExecutionPlan& plan);

// Versioned, machine-readable audit record. Paths are represented by basenames
// and content digests; customer filesystem paths are not emitted.
std::string dmabuf_plan_audit_json(const DmabufPlanCompileResult& result,
                                   const std::filesystem::path& mpk_manifest,
                                   const std::filesystem::path& mla_executable,
                                   bool pretty = false);

struct MemoryBackendDecision {
  MemoryBackendPolicy backend = MemoryBackendPolicy::Legacy;
  DmabufEligibilityReport admission;
  std::string plan_digest;

  [[nodiscard]] bool uses_dmabuf_plan() const noexcept {
    return backend == MemoryBackendPolicy::DmaBufPlan;
  }
};

} // namespace simaai::neat::pipeline_internal

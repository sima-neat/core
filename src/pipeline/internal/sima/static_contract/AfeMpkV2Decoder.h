#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/MlaElfIoTopology.h"
#include "pipeline/internal/sima/static_contract/ModelExecutionPlan.h"

#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::static_contract {

enum class AfeMpkV2DecodeErrorCode {
  InvalidJson,
  MissingRequiredField,
  InvalidField,
  UnsupportedContractVersion,
  UnsupportedKernel,
  UnsupportedHostModule,
  InvalidKernelArity,
  DuplicateSequence,
  DuplicateProducer,
  MissingProducer,
  ValueSizeMismatch,
  ConfigurationMismatch,
  MissingMlaStage,
  MultipleMlaStages,
  MissingMlaExecutableEvidence,
  AmbiguousMlaExecutableEvidence,
  UnexpectedMlaExecutableEvidence,
  MissingPublicationStage,
  InvalidPublicationStage,
  ElfTopologyInvalid,
  ElfTopologyMismatch,
  PlanValidationFailed,
  IoError,
};

// Setup-time evidence for one exact MPK MLA operation.  The decoder joins by
// both compiler-authored logical identity and manifest executable token; file
// enumeration order and filename heuristics are deliberately irrelevant.
struct MlaStageExecutableEvidence {
  std::string logical_stage_id;
  std::string executable;
  MlaElfIoTopology topology;
};

struct AfeMpkV2DecodeError {
  AfeMpkV2DecodeErrorCode code = AfeMpkV2DecodeErrorCode::InvalidJson;
  std::string source;
  std::string json_path;
  std::string detail;
};

struct AfeMpkV2ProofFact {
  std::string subject;
  std::string evidence;
};

struct AfeMpkV2DecodeResult {
  std::optional<ModelExecutionPlan> plan;
  std::vector<AfeMpkV2ProofFact> proof;
  std::optional<AfeMpkV2DecodeError> error;

  explicit operator bool() const noexcept {
    return plan.has_value() && !error.has_value();
  }
};

// Quarantined inverse for the frozen, untyped AFE MPK contract. It accepts an
// explicitly supplied manifest and explicitly supplied ELF topology; archive
// names, model names, sidecar JSON, environment, and runtime metadata are not
// consulted. New typed MPK versions must use a direct parser instead.
class AfeMpkV2Decoder final {
public:
  AfeMpkV2DecodeResult decode_json(std::string_view mpk_json,
                                   std::span<const MlaStageExecutableEvidence> executable_evidence,
                                   std::string source_label = "<memory>") const noexcept;

  AfeMpkV2DecodeResult decode_json(std::string_view mpk_json, const MlaElfIoTopology& elf_topology,
                                   std::string source_label = "<memory>") const noexcept;

  AfeMpkV2DecodeResult
  decode_file(const std::filesystem::path& mpk_manifest,
              std::span<const MlaStageExecutableEvidence> executable_evidence) const noexcept;

  AfeMpkV2DecodeResult decode_file(const std::filesystem::path& mpk_manifest,
                                   const MlaElfIoTopology& elf_topology) const noexcept;
};

} // namespace simaai::neat::pipeline_internal::sima::static_contract

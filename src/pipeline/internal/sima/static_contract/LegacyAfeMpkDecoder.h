#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/MlaElfIoTopology.h"
#include "pipeline/internal/sima/static_contract/ModelExecutionPlan.h"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::static_contract {

enum class LegacyAfeDecodeErrorCode {
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
  IoError,
};

struct LegacyAfeDecodeError {
  LegacyAfeDecodeErrorCode code = LegacyAfeDecodeErrorCode::InvalidJson;
  std::string source;
  std::string json_path;
  std::string detail;
};

struct LegacyAfeProofFact {
  std::string subject;
  std::string evidence;
};

struct LegacyAfeDecodeResult {
  std::optional<ModelExecutionPlan> plan;
  std::vector<LegacyAfeProofFact> proof;
  std::optional<LegacyAfeDecodeError> error;

  explicit operator bool() const noexcept {
    return plan.has_value() && !error.has_value();
  }
};

// Quarantined inverse for the frozen, untyped AFE MPK contract. It accepts an
// explicitly supplied manifest and explicitly supplied ELF topology; archive
// names, model names, sidecar JSON, environment, and runtime metadata are not
// consulted. New typed MPK versions must use a direct parser instead.
class LegacyAfeMpkDecoder final {
public:
  LegacyAfeDecodeResult decode_json(std::string_view mpk_json, const MlaElfIoTopology& elf_topology,
                                    std::string source_label = "<memory>") const noexcept;

  LegacyAfeDecodeResult decode_file(const std::filesystem::path& mpk_manifest,
                                    const MlaElfIoTopology& elf_topology) const noexcept;
};

} // namespace simaai::neat::pipeline_internal::sima::static_contract

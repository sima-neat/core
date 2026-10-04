#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/MlaElfIoTopology.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/static_contract/ModelExecutionPlan.h"

#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::static_contract {

enum class MpkDecodeErrorCode {
  InvalidJson,
  MissingRequiredField,
  InvalidField,
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
  std::uint64_t byte_length = 0;
  std::string sha256;
};

struct HostTvmExecutableEvidence {
  std::string logical_stage_id;
  std::string executable;
  std::vector<std::string> input_names;
  std::vector<HostTensorTypeSpec> input_types;
  std::vector<HostTensorTypeSpec> output_types;
  std::vector<std::int32_t> output_alias_input;
  std::uint64_t byte_length = 0;
  std::string sha256;
  // Complete GraphExecutor arg-node table. For historical evidence producers
  // these vectors may be empty and input_names/input_types remain the complete
  // table. The target-ready decoder uses the complete table to prove the
  // external-versus-linked classification.
  std::vector<std::string> argument_names;
  std::vector<HostTensorTypeSpec> argument_types;
};

struct MpkDecodeError {
  MpkDecodeErrorCode code = MpkDecodeErrorCode::InvalidJson;
  std::string source;
  std::string json_path;
  std::string detail;
};

struct MpkProofFact {
  std::string subject;
  std::string evidence;
};

struct MpkDecodeResult {
  std::optional<ModelExecutionPlan> plan;
  std::vector<MpkProofFact> proof;
  std::optional<MpkDecodeError> error;

  explicit operator bool() const noexcept {
    return plan.has_value() && !error.has_value();
  }
};

// An executable named by one MPK stage, resolved inside its model package.
struct MpkExecutableArtifact {
  std::string logical_stage_id;
  std::string manifest_executable;
  std::filesystem::path resolved_path;
};

// The exact inputs of one model package: the manifest bytes, read once, and the
// executables its MLA and A65 stages name.
struct MpkPackage {
  std::filesystem::path manifest_path;
  std::string manifest_bytes;
  std::vector<MpkExecutableArtifact> mla_executables;
  std::vector<MpkExecutableArtifact> host_executables;
};

// A loaded model package and the semantic contract built from its manifest bytes.
struct LoadedMpk {
  MpkPackage package;
  MpkContract contract;
};

// SHA-256 of a file's bytes, or nullopt when it cannot be read.
std::optional<std::string> sha256_file(const std::filesystem::path& path);

// Resolves an MLA executable named by the manifest. Candidates are
// <package>/share/<name>, <package>/<name>, then <name>; the first existing file
// wins, otherwise the first candidate. An empty name resolves to an empty path.
std::filesystem::path resolve_mla_executable(const std::filesystem::path& mpk_manifest,
                                             const std::string& executable);

// Resolves an A65 host module named by the manifest. Candidates are
// <package>/lib/<name>, <package>/<name>, <package>/share/<name>, then <name>,
// with the same selection rule as resolve_mla_executable.
std::filesystem::path resolve_host_executable(const std::filesystem::path& mpk_manifest,
                                              const std::string& executable);

struct MpkEvidenceError {
  enum class Kind { MissingIdentity, MissingFile, DigestFailed, UnreadableStructure };
  Kind kind = Kind::MissingIdentity;
  bool host_module = false;
  std::filesystem::path resolved_path;
  std::string detail;
  // Artifact identity through the failing executable's digest, when it was hashed.
  std::string artifact_identity;
};

struct MpkExecutableEvidence {
  std::vector<MlaStageExecutableEvidence> mla;
  std::vector<HostTvmExecutableEvidence> host;
  // "mpk=<sha256>" followed by each executable's stage, token, and digest.
  std::string artifact_identity;
  std::optional<MpkEvidenceError> error;
};

// Exact decoder for supported AFE MPK processor/kernel capabilities.
// It accepts an explicitly supplied manifest plus exact setup-time MLA ELF and
// A65 GraphExecutor evidence; archive names, filename suffixes, sidecar JSON,
// environment, and runtime metadata are never semantic authority.
class MpkDecoder final {
public:
  // Single entry point for a model package. Reads the manifest once, builds the
  // semantic MpkContract from those bytes, and resolves the executables its
  // stages name without reading them.
  static std::optional<LoadedMpk> load(const std::filesystem::path& mpk_manifest,
                                       std::string* error = nullptr);
  static std::optional<LoadedMpk> load_package_root(const std::filesystem::path& package_root,
                                                    std::string* error = nullptr);

  // Reads each executable's digest and structure: MLA ELF topology and A65
  // GraphExecutor contract.
  static MpkExecutableEvidence read_executable_evidence(const MpkPackage& package);

  MpkDecodeResult decode(const MpkPackage& package) const noexcept;

  MpkDecodeResult decode_json(std::string_view mpk_json,
                              std::span<const MlaStageExecutableEvidence> executable_evidence,
                              std::span<const HostTvmExecutableEvidence> host_evidence,
                              std::string source_label = "<memory>") const noexcept;

  MpkDecodeResult decode_json(std::string_view mpk_json,
                              std::span<const MlaStageExecutableEvidence> executable_evidence,
                              std::string source_label = "<memory>") const noexcept;

  MpkDecodeResult decode_json(std::string_view mpk_json, const MlaElfIoTopology& elf_topology,
                              std::string source_label = "<memory>") const noexcept;

  MpkDecodeResult
  decode_file(const std::filesystem::path& mpk_manifest,
              std::span<const MlaStageExecutableEvidence> executable_evidence,
              std::span<const HostTvmExecutableEvidence> host_evidence) const noexcept;

  MpkDecodeResult
  decode_file(const std::filesystem::path& mpk_manifest,
              std::span<const MlaStageExecutableEvidence> executable_evidence) const noexcept;

  MpkDecodeResult decode_file(const std::filesystem::path& mpk_manifest,
                              const MlaElfIoTopology& elf_topology) const noexcept;
};

} // namespace simaai::neat::pipeline_internal::sima::static_contract

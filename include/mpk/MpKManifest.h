/**
 * @file
 * @ingroup mpk
 * @brief MPK manifest, archive-entry metadata, and the loader's error taxonomy.
 *
 * The `MpKManifest` is the single authoritative description of a model pack — produced by
 * `MpKLoader::inspect()` / `MpKLoader::extract()`, and the only JSON the framework reads
 * inside an MPK archive. This header also defines the loader's structured error type
 * (`MpKError`) and the `ErrorClass` taxonomy used in support tickets and triage tools.
 *
 * @see MpKLoader for how to read these
 * @see "MPK contract" (§0.16 of the design deep dive)
 */
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace simaai::neat::mpk {

/**
 * @brief Coarse error categories raised by the MPK loader.
 *
 * Each MPK validation failure maps to exactly one of these classes. The class is the
 * triage hook — support tickets and CI tools bucket failures by `ErrorClass`.
 * @ingroup mpk
 */
enum class ErrorClass {
  InvalidArchive,     ///< Archive is corrupt, truncated, or in an unrecognized format.
  PathTraversal,      ///< Archive contains a path that would escape the extraction root.
  SchemaError,        ///< Manifest is structurally valid JSON but violates the schema.
  UnsupportedVersion, ///< Manifest declares a version this loader doesn't understand.
  SizeLimitExceeded,  ///< Archive or one of its entries exceeds a configured size cap.
};

/// Returns a stable string identifier for an `ErrorClass` (used in error messages and JSON serialization).
const char* error_class_name(ErrorClass code);

/**
 * @brief Exception thrown by the MPK loader when validation fails.
 *
 * Carries the `ErrorClass` for triage plus a human-readable message. Often re-wrapped into a
 * `SessionError` (with a `SessionReport`) at higher layers.
 * @see ErrorClass
 * @ingroup mpk
 */
class MpKError : public std::runtime_error {
public:
  /// Construct from an `ErrorClass` and a descriptive message.
  MpKError(ErrorClass code, const std::string& message);

  /// Returns the error category (for switch-on-class triage).
  ErrorClass code() const noexcept {
    return code_;
  }

private:
  ErrorClass code_;
};

/**
 * @brief Metadata for a single entry inside the MPK archive.
 *
 * Captured during inspection/extraction. `path` is the raw archive entry path; `normalized_path`
 * is the post-validation safe path (used for filesystem operations). `type` is the tar entry
 * type code (`'0'` = regular file, `'5'` = directory, etc.).
 * @ingroup mpk
 */
struct ArchiveEntry {
  std::string path;            ///< Raw entry path as recorded in the archive.
  std::string normalized_path; ///< Validated, normalized path (relative, no traversal, valid UTF-8).
  char type = '?';             ///< Tar entry type code.
  std::uint64_t size_bytes = 0;///< Uncompressed entry size in bytes.
};

/**
 * @brief Validated manifest of an MPK archive — the framework's view of a model pack.
 *
 * Returned from `MpKLoader::inspect()` and `MpKLoader::extract()`. The `entries` vector is
 * populated only after a full inspection/extraction; the boolean flags signal whether the
 * archive contains the required sections (pipeline sequence, model binary). When loaded via
 * `Model`, this is the authoritative source for everything the route planner needs.
 * @ingroup mpk
 */
struct MpKManifest {
  std::string archive_path;     ///< Filesystem path the manifest was loaded from.
  std::string package_name;     ///< Package identifier (typically the archive's top-level directory name).
  std::string version = "1";    ///< Manifest schema version this archive declares.
  std::uint64_t archive_size_bytes = 0; ///< Compressed archive size on disk.

  bool has_pipeline_sequence = false;   ///< True if the archive contains a `pipeline_sequence.json`.
  bool has_model_binary = false;        ///< True if the archive contains at least one model binary (`.lm`/`.so`/etc.).

  std::vector<ArchiveEntry> entries;    ///< All archive entries, in the order encountered.
};

} // namespace simaai::neat::mpk

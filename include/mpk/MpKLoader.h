/**
 * @file
 * @ingroup mpk
 * @brief MPK loader with deterministic validation and secure extraction.
 *
 * The `MpKLoader` is the framework's gateway for accepting model packs (`.tar.gz` /
 * `.tgz` / `.mpk` / `.tar`) from outside the system. Because model packs are user-
 * uploaded artifacts, the loader is **fail-closed**: every accepted pack passes a strict
 * validation matrix (size limits, path traversal protection, UTF-8 sanity, manifest
 * schema, kernel allowlist). The complete defense matrix is described in §91 of the
 * design deep dive.
 *
 * Use `inspect()` for cheap manifest-only reads (no extraction), `extract()` to fully
 * unpack the archive into a controlled directory.
 *
 * @see MpKManifest for the parsed manifest structure
 * @see "MPK contract" (§0.16) and "MPK security matrix" (§91) of the design deep dive
 */
#pragma once

#include "mpk/MpKManifest.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace simaai::neat::mpk {

/**
 * @brief Tunable safety/size limits applied to every MPK load.
 *
 * Defaults are conservative ceilings appropriate for production model packs. Tighten
 * these for sandboxed/multi-tenant deployments where untrusted packs flow through the
 * loader. All limits are enforced before any extraction; an over-limit pack fails with
 * `error_code = "io.parse"` and an actionable repro note.
 * @ingroup mpk
 */
struct MpKLoaderOptions {
  std::size_t max_archive_bytes =
      512ULL * 1024ULL * 1024ULL; ///< Hard cap on the on-disk archive size (default 512 MiB).
  std::size_t max_entry_bytes =
      256ULL * 1024ULL * 1024ULL; ///< Hard cap on any single extracted entry (default 256 MiB).
  std::size_t max_total_json_bytes =
      32ULL * 1024ULL *
      1024ULL; ///< Hard cap on aggregate JSON size across all manifest/config files.
  std::size_t max_entries = 2048; ///< Hard cap on the number of entries the archive may contain.
  std::size_t max_json_depth =
      64; ///< Hard cap on JSON nesting depth (defends against parser-stack exhaustion).

  bool require_pipeline_sequence =
      true; ///< If true, reject packs that lack a `pipeline_sequence` manifest section.
  bool require_model_binary = true; ///< If true, reject packs that don't ship a model binary.
  bool reject_unsupported_file_types =
      true; ///< If true, reject packs containing file types outside the allowlist.
  bool reject_duplicate_json_keys =
      true; ///< If true, reject manifests with duplicate keys at any nesting level.
  bool reject_invalid_utf8_paths =
      true; ///< If true, reject archive entries whose paths aren't valid UTF-8.
  bool reject_unicode_path_confusables =
      true; ///< If true, reject paths containing visually-confusable Unicode (defends against
            ///< typosquatting).
};

/**
 * @brief Result of `MpKLoader::extract()` — the unpacked MPK on disk plus the parsed manifest.
 *
 * Directory layout after extraction:
 *   - `package_root` — root of the extracted package
 *   - `etc_dir` — per-stage configs and runtime metadata
 *   - `lib_dir` — kernel binaries and dispatcher payloads
 *   - `share_dir` — model weights and shared resources
 *
 * The `manifest` field contains the parsed and validated `mpk.json`.
 * @ingroup mpk
 */
struct MpKExtractResult {
  std::string package_root; ///< Absolute path to the extracted package's root.
  std::string etc_dir;      ///< `<package_root>/etc/` — per-stage config files.
  std::string lib_dir;      ///< `<package_root>/lib/` — kernel binaries and runtime libraries.
  std::string share_dir;    ///< `<package_root>/share/` — model weights and shared assets.

  MpKManifest manifest; ///< Parsed manifest (the only authoritative JSON in the pack).
};

/**
 * @brief Static factory for opening an MPK — inspect (manifest only) or extract (full unpack).
 *
 * Stateless: every call validates from scratch. Use `inspect()` when you only need the
 * manifest (fast, no disk writes); use `extract()` when you need the binaries and configs
 * on disk for a `Model` to consume.
 * @ingroup mpk
 */
class MpKLoader {
public:
  /**
   * @brief Read and validate the manifest of an MPK archive without extracting binaries.
   *
   * Cheaper than `extract()` because it skips on-disk extraction of `.so`/`.lm`/etc.
   * Useful for tooling that just wants to inspect a model pack's metadata.
   *
   * @param archive_path Path to the `.tar.gz`/`.mpk`/etc.
   * @param opt          Safety/size limits for validation.
   * @return The parsed `MpKManifest`.
   * @throws SessionError on any validation failure.
   */
  static MpKManifest inspect(const std::string& archive_path, const MpKLoaderOptions& opt = {});

  /**
   * @brief Validate, extract, and stage an MPK on disk for runtime consumption.
   *
   * Extracts into `output_root/<package-name>/`. Runs the full validation matrix; rejects
   * any pack that fails. On success, returns paths to the standard sub-directories plus the
   * parsed manifest.
   *
   * @param archive_path Path to the `.tar.gz`/`.mpk`/etc.
   * @param output_root  Directory under which the package will be unpacked.
   * @param opt          Safety/size limits for validation.
   * @return Populated `MpKExtractResult` on success.
   * @throws SessionError on any validation or I/O failure.
   */
  static MpKExtractResult extract(const std::string& archive_path, const std::string& output_root,
                                  const MpKLoaderOptions& opt = {});
};

} // namespace simaai::neat::mpk

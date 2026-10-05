/**
 * @file
 * @brief Internal secure loader for .tar.gz model archives.
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace simaai::neat::internal {

/// Coarse error categories raised by the internal model archive loader.
enum class ModelArchiveErrorClass {
  InvalidArchive,
  PathTraversal,
  SchemaError,
  UnsupportedVersion,
  SizeLimitExceeded,
  UnsupportedExtension,
  OutputStorageUnavailable,
};

const char* model_archive_error_class_name(ModelArchiveErrorClass code);

class ModelArchiveError : public std::runtime_error {
public:
  ModelArchiveError(ModelArchiveErrorClass code, const std::string& message);

  ModelArchiveErrorClass code() const noexcept {
    return code_;
  }

private:
  ModelArchiveErrorClass code_;
};

struct ModelArchiveEntry {
  std::string path;
  std::string normalized_path;
  char type = '?';
  std::uint64_t size_bytes = 0;
};

struct ModelArchiveManifest {
  std::string archive_path;
  std::string package_name;
  std::uint64_t archive_size_bytes = 0;

  bool has_model_binary = false;

  std::vector<ModelArchiveEntry> entries;
};

struct ModelArchiveLoaderOptions {
  std::size_t max_archive_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  // Gzip-bomb guard for the staging copy: the compressed size says nothing about how far a
  // hostile archive expands.
  std::size_t max_inflated_archive_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  std::size_t max_entry_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  std::size_t max_total_json_bytes = 32ULL * 1024ULL * 1024ULL;
  std::size_t max_entries = 2048;
  std::size_t max_json_depth = 64;
  std::uint64_t min_output_free_bytes = 16ULL * 1024ULL * 1024ULL;
  // Filesystem for the inflated snapshot. Empty stages under TMPDIR.
  std::string staging_base;
  // Optional physical package-directory leaf. Empty preserves the archive-derived package name.
  // Runtime ModelPack uses a compact leaf inside its already unique archive-identity directory so
  // downstream fixed-width transport paths do not redundantly include the logical archive name.
  std::string physical_package_leaf;

  bool require_model_binary = true;
  bool reject_unsupported_file_types = true;
  bool reject_duplicate_json_keys = true;
  bool reject_invalid_utf8_paths = true;
  bool reject_unicode_path_confusables = true;
  bool check_output_free_space = true;
  // Two distinct archive entries can flatten to the same extraction destination
  // (e.g. a/config.json and b/config.json both -> etc/config.json); the later one
  // silently overwrites the earlier. Default = warn (loud diagnostic). Set true to
  // hard-reject — gated on an archive-in-the-wild audit before flipping the default.
  bool reject_destination_collisions = false;
};

struct ModelArchiveExtractResult {
  std::string package_root;
  std::string etc_dir;
  std::string lib_dir;
  std::string share_dir;

  ModelArchiveManifest manifest;
};

/// Selects the extraction root from the validated manifest, before any output file is created.
using ChooseModelArchiveOutputRoot = std::function<std::string(const ModelArchiveManifest&)>;

/// Reads .tar.gz model archives. Every entry point, inspect() included, inflates the archive
/// once into a staging copy and so needs temp space for the inflated size.
class ModelArchiveLoader {
public:
  static ModelArchiveManifest inspect(const std::string& archive_path,
                                      const ModelArchiveLoaderOptions& opt = {});

  static ModelArchiveExtractResult extract(const std::string& archive_path,
                                           const std::string& output_root,
                                           const ModelArchiveLoaderOptions& opt = {});

  /// extract() for callers whose root choice depends on the extracted size, without costing
  /// them a second validation pass.
  static ModelArchiveExtractResult extract(const std::string& archive_path,
                                           const ChooseModelArchiveOutputRoot& choose_output_root,
                                           const ModelArchiveLoaderOptions& opt = {});

  /// Total archive inflations performed by this process. #653 made one load cost exactly one
  /// inflation; a regression to repeated passes is otherwise only visible as elapsed time.
  static std::uint64_t inflation_count();
};

} // namespace simaai::neat::internal

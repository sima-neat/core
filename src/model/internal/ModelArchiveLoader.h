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
  std::string version = "1";
  std::uint64_t archive_size_bytes = 0;

  bool has_pipeline_sequence = false;
  bool has_model_binary = false;

  std::vector<ModelArchiveEntry> entries;
};

struct ModelArchiveLoaderOptions {
  std::size_t max_archive_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  std::size_t max_entry_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  std::size_t max_total_json_bytes = 32ULL * 1024ULL * 1024ULL;
  std::size_t max_entries = 2048;
  std::size_t max_json_depth = 64;

  bool require_pipeline_sequence = true;
  bool require_model_binary = true;
  bool reject_unsupported_file_types = true;
  bool reject_duplicate_json_keys = true;
  bool reject_invalid_utf8_paths = true;
  bool reject_unicode_path_confusables = true;
};

struct ModelArchiveExtractResult {
  std::string package_root;
  std::string etc_dir;
  std::string lib_dir;
  std::string share_dir;

  ModelArchiveManifest manifest;
};

class ModelArchiveLoader {
public:
  static ModelArchiveManifest inspect(const std::string& archive_path,
                                      const ModelArchiveLoaderOptions& opt = {});

  static ModelArchiveExtractResult extract(const std::string& archive_path,
                                           const std::string& output_root,
                                           const ModelArchiveLoaderOptions& opt = {});
};

} // namespace simaai::neat::internal

/**
 * @file
 * @ingroup mpk
 * @brief MPK manifest and error taxonomy types.
 */
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace simaai::neat::mpk {

enum class ErrorClass {
  InvalidArchive,
  PathTraversal,
  SchemaError,
  UnsupportedVersion,
  SizeLimitExceeded,
};

const char* error_class_name(ErrorClass code);

class MpKError : public std::runtime_error {
public:
  MpKError(ErrorClass code, const std::string& message);

  ErrorClass code() const noexcept {
    return code_;
  }

private:
  ErrorClass code_;
};

struct ArchiveEntry {
  std::string path;
  std::string normalized_path;
  char type = '?';
  std::uint64_t size_bytes = 0;
};

struct MpKManifest {
  std::string archive_path;
  std::string package_name;
  std::string version = "1";
  std::uint64_t archive_size_bytes = 0;

  bool has_pipeline_sequence = false;
  bool has_model_binary = false;

  std::vector<ArchiveEntry> entries;
};

} // namespace simaai::neat::mpk

/**
 * @file
 * @ingroup mpk
 * @brief MPK loader with deterministic validation and secure extraction.
 */
#pragma once

#include "mpk/MpKManifest.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace simaai::neat::mpk {

struct MpKLoaderOptions {
  std::size_t max_archive_bytes = 512ULL * 1024ULL * 1024ULL;
  std::size_t max_entry_bytes = 256ULL * 1024ULL * 1024ULL;
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

struct MpKExtractResult {
  std::string package_root;
  std::string etc_dir;
  std::string lib_dir;
  std::string share_dir;

  MpKManifest manifest;
};

class MpKLoader {
public:
  static MpKManifest inspect(const std::string& archive_path, const MpKLoaderOptions& opt = {});

  static MpKExtractResult extract(const std::string& archive_path, const std::string& output_root,
                                  const MpKLoaderOptions& opt = {});
};

} // namespace simaai::neat::mpk

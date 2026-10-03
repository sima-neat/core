/**
 * @file
 * @brief Copy a VLM image into the daemon's serve root so the card can pull it,
 *        and delete the copy when the request is done (RAII).
 */
#pragma once

#include <filesystem>
#include <string>

namespace simaai::neat::pcie::genai::internal {

// Copies one image into <stage_dir> under a per-run name and removes it on
// destruction. Construct one for the lifetime of a single generate() call.
class StagedImage {
public:
  // Sweeps stage_dir of files older than one hour whose owner process is gone
  // (leftovers from a killed process), then copies `source` to <stage_dir>/<run_id><source
  // extension>. Throws std::runtime_error if the source is unreadable or the copy fails.
  StagedImage(const std::filesystem::path& stage_dir, const std::string& run_id,
              const std::filesystem::path& source);
  ~StagedImage();

  StagedImage(const StagedImage&) = delete;
  StagedImage& operator=(const StagedImage&) = delete;

  // Name for the genai.prompt "image" field, relative to the serve root:
  // "<stage_dir last component>/<run_id><ext>" (e.g. "pcie-genai/h1-1.jpg").
  const std::string& relative_name() const {
    return relative_name_;
  }
  const std::filesystem::path& staged_path() const {
    return staged_path_;
  }

private:
  std::filesystem::path staged_path_;
  std::string relative_name_;
};

} // namespace simaai::neat::pcie::genai::internal

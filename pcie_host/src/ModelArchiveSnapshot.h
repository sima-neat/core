#pragma once

#include <filesystem>
#include <string>

namespace simaai::neat::pcie::internal {

class ModelArchiveSnapshot {
public:
  explicit ModelArchiveSnapshot(const std::string& source_path);
  ~ModelArchiveSnapshot() noexcept;

  ModelArchiveSnapshot(const ModelArchiveSnapshot&) = delete;
  ModelArchiveSnapshot& operator=(const ModelArchiveSnapshot&) = delete;
  ModelArchiveSnapshot(ModelArchiveSnapshot&&) = delete;
  ModelArchiveSnapshot& operator=(ModelArchiveSnapshot&&) = delete;

  const std::string& path() const noexcept {
    return archive_path_;
  }

private:
  std::filesystem::path directory_;
  std::string archive_path_;
};

} // namespace simaai::neat::pcie::internal

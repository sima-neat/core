#include "ModelArchiveSnapshot.h"

#include <stdexcept>
#include <system_error>

namespace fs = std::filesystem;

namespace simaai::neat::pcie::internal {

ModelArchiveSnapshot::ModelArchiveSnapshot(const std::string& source_path)
    : directory_("sima-neat-pcie-model") {
  const fs::path source(source_path);
  std::error_code ec;
  if (!fs::is_regular_file(source, ec) || ec) {
    throw std::runtime_error("PCIe model archive is not a readable regular file: " + source_path);
  }

  const fs::path snapshot = directory_.path() / source.filename();
  fs::copy_file(source, snapshot, fs::copy_options::none);
#ifndef _WIN32
  fs::permissions(snapshot, fs::perms::owner_read, fs::perm_options::replace);
#endif
  archive_path_ = host_path_string(snapshot);
}

ModelArchiveSnapshot::~ModelArchiveSnapshot() noexcept = default;

} // namespace simaai::neat::pcie::internal

#include "ModelArchiveSnapshot.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <system_error>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;

namespace simaai::neat::pcie::internal {

ModelArchiveSnapshot::ModelArchiveSnapshot(const std::string& source_path) {
  const fs::path source(source_path);
  std::error_code ec;
  if (!fs::is_regular_file(source, ec) || ec) {
    throw std::runtime_error("PCIe model archive is not a readable regular file: " + source_path);
  }

  std::string directory_template =
      (fs::temp_directory_path() / "sima-neat-pcie-model-XXXXXX").string();
  std::vector<char> writable_template(directory_template.begin(), directory_template.end());
  writable_template.push_back('\0');
  const char* created = ::mkdtemp(writable_template.data());
  if (!created) {
    throw std::runtime_error(std::string("failed to create PCIe model snapshot directory: ") +
                             std::strerror(errno));
  }
  directory_ = created;

  try {
    const fs::path snapshot = directory_ / source.filename();
    fs::copy_file(source, snapshot, fs::copy_options::none);
    fs::permissions(snapshot, fs::perms::owner_read, fs::perm_options::replace);
    archive_path_ = snapshot.string();
  } catch (...) {
    fs::remove_all(directory_, ec);
    throw;
  }
}

ModelArchiveSnapshot::~ModelArchiveSnapshot() noexcept {
  std::error_code ec;
  fs::remove_all(directory_, ec);
}

} // namespace simaai::neat::pcie::internal

#include "ModelArchiveSnapshot.h"

#include <stdexcept>
#include <string_view>
#include <system_error>

namespace fs = std::filesystem;

namespace simaai::neat::pcie::internal {

ModelArchiveSnapshot::ModelArchiveSnapshot(const std::string& source_path)
    : directory_("sima-neat-pcie-model") {
  const fs::path source = host_path_from_string(source_path);
  std::error_code ec;
  if (!fs::is_regular_file(source, ec) || ec) {
    throw std::runtime_error("PCIe model archive is not a readable regular file: " + source_path);
  }
  fs::path snapshot = directory_.path() / source.filename();
#ifdef _WIN32
  const std::string source_name = host_path_string(source.filename());
  constexpr std::string_view suffix = ".tar.gz";
  if (source_name.size() < suffix.size() ||
      source_name.compare(source_name.size() - suffix.size(), suffix.size(), suffix) != 0) {
    throw std::runtime_error("model archive must use .tar.gz: " + source_path);
  }
  // Keep the Windows private snapshot path ASCII-only. The archive loader
  // invokes the operating system's tar program, while the customer path may
  // contain any valid Unicode characters. Inspection and upload still use
  // this exact same immutable snapshot.
  snapshot = directory_.path() / "model.tar.gz";
#endif
  fs::copy_file(source, snapshot, fs::copy_options::none);
#ifndef _WIN32
  fs::permissions(snapshot, fs::perms::owner_read, fs::perm_options::replace);
#endif
  archive_path_ = host_path_string(snapshot);
}

ModelArchiveSnapshot::~ModelArchiveSnapshot() noexcept = default;

} // namespace simaai::neat::pcie::internal

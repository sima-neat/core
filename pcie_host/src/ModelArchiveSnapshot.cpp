#include "ModelArchiveSnapshot.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace fs = std::filesystem;

namespace simaai::neat::pcie::internal {
namespace {

constexpr const char* kSnapshotPrefix = "sima-neat-pcie-model";
constexpr const char* kOwnerFile = "owner";

// Removes snapshots left by ended processes of this host, for example after a crash or kill.
// Snapshots without an owner record, from other hosts or of running processes are kept.
void remove_ended_snapshots(const fs::path& own) noexcept {
  const std::string prefix = std::string(kSnapshotPrefix) + "-";
  std::error_code ec;
  for (fs::directory_iterator it(own.parent_path(), ec), end; !ec && it != end; it.increment(ec)) {
    try {
      if (it->path() == own || host_path_string(it->path().filename()).rfind(prefix, 0) != 0) {
        continue;
      }
      HostProcessOwner owner;
      std::ifstream record(it->path() / kOwnerFile);
      for (std::string line; std::getline(record, line);) {
        const auto equals = line.find('=');
        const std::string key = line.substr(0, equals);
        const std::string value = equals == std::string::npos ? "" : line.substr(equals + 1);
        if (key == "host_id") {
          owner.host_id = value;
        } else if (key == "host_pid") {
          owner.pid = std::stoull(value);
        } else if (key == "host_start") {
          owner.start = value;
        }
      }
      if (host_process_ended(owner)) {
        std::error_code remove_error;
        fs::remove_all(it->path(), remove_error);
      }
    } catch (...) {
    }
  }
}

} // namespace

ModelArchiveSnapshot::ModelArchiveSnapshot(const std::string& source_path)
    : directory_(kSnapshotPrefix) {
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
  // Give the Windows snapshot a fixed ASCII name: the card-side upload name
  // embeds it, while the customer's file name may hold any Unicode characters.
  // Inspection and upload still use this one immutable snapshot.
  snapshot = directory_.path() / "model.tar.gz";
#endif
  const HostProcessOwner owner = current_host_process_owner();
  std::ofstream(directory_.path() / kOwnerFile)
      << "host_id=" << owner.host_id << "\nhost_pid=" << owner.pid << "\nhost_start=" << owner.start
      << "\n";
  remove_ended_snapshots(directory_.path());
  fs::copy_file(source, snapshot, fs::copy_options::none);
#ifndef _WIN32
  fs::permissions(snapshot, fs::perms::owner_read, fs::perm_options::replace);
#endif
  archive_path_ = host_path_string(snapshot);
}

ModelArchiveSnapshot::~ModelArchiveSnapshot() noexcept = default;

} // namespace simaai::neat::pcie::internal

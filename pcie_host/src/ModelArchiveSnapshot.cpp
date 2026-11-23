#include "ModelArchiveSnapshot.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace fs = std::filesystem;

namespace simaai::neat::pcie::internal {
namespace {

constexpr const char* kSnapshotPrefix = "sima-neat-pcie-model";
constexpr const char* kOwnerFile = "owner";
constexpr std::size_t kMaxOwnerRecordBytes = 4096;

// The owner record of a snapshot directory, or "" when the entry is not a directory holding a
// regular owner file. Other local users can create entries in a shared temporary directory, so
// neither is followed through a symbolic link, and a FIFO or device is never opened.
std::string read_owner_record(const fs::directory_entry& entry) {
  std::error_code ec;
  const fs::path path = entry.path() / kOwnerFile;
  if (entry.symlink_status(ec).type() != fs::file_type::directory ||
      fs::symlink_status(path, ec).type() != fs::file_type::regular) {
    return "";
  }
  std::ifstream file(path, std::ios::binary);
  std::string record(kMaxOwnerRecordBytes, '\0');
  file.read(record.data(), static_cast<std::streamsize>(record.size()));
  record.resize(static_cast<std::size_t>(file.gcount()));
  return record;
}

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
      std::istringstream record(read_owner_record(*it));
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
  // Binary, so that the record holds the same LF lines on every host.
  std::ofstream(directory_.path() / kOwnerFile, std::ios::binary)
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

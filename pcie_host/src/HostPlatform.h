#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace simaai::neat::pcie::internal {

class ScopedTempDirectory {
public:
  explicit ScopedTempDirectory(const std::string& prefix);
  ~ScopedTempDirectory() noexcept;

  ScopedTempDirectory(const ScopedTempDirectory&) = delete;
  ScopedTempDirectory& operator=(const ScopedTempDirectory&) = delete;
  ScopedTempDirectory(ScopedTempDirectory&&) = delete;
  ScopedTempDirectory& operator=(ScopedTempDirectory&&) = delete;

  const std::filesystem::path& path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

std::optional<std::filesystem::path> host_home_directory();
std::filesystem::path host_path_from_string(const std::string& path);
std::string host_path_string(const std::filesystem::path& path);
bool same_host_path(const std::filesystem::path& left, const std::filesystem::path& right);
std::string open_ssh_executable(const char* command);
std::uint64_t host_process_id();
void host_secure_random(unsigned char* buffer, std::size_t size);

// A host process, recorded so that a later process can tell whether it has ended.
struct HostProcessOwner {
  std::string host_id; // this OS instance (on Linux, also the PID namespace); empty if unknown
  std::string host;    // host name, for messages
  std::uint64_t pid = 0;
  std::string start; // process start time, so that a reused PID is not taken for this process
};

HostProcessOwner current_host_process_owner();
// True only when the owner was recorded on this host and that process is known to have ended.
bool host_process_ended(const HostProcessOwner& owner);

} // namespace simaai::neat::pcie::internal

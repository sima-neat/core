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
std::string open_ssh_executable(const char* command);
std::uint64_t host_process_id();
void host_secure_random(unsigned char* buffer, std::size_t size);

} // namespace simaai::neat::pcie::internal

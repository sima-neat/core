#include "HostPlatform.h"

#include <cstdlib>
#include <stdexcept>

#ifdef _WIN32
#include <process.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace simaai::neat::pcie::internal {

std::optional<std::filesystem::path> host_home_directory() {
#ifdef _WIN32
  const wchar_t* home = _wgetenv(L"USERPROFILE");
#else
  const char* home = std::getenv("HOME");
#endif
  if (!home || !*home) {
    return std::nullopt;
  }
  return std::filesystem::path(home);
}

std::string host_path_string(const std::filesystem::path& path) {
#ifdef _WIN32
  const auto utf8 = path.u8string();
  return {utf8.begin(), utf8.end()};
#else
  return path.string();
#endif
}

std::string open_ssh_executable(const char* command) {
  if (!command || !*command) {
    throw std::invalid_argument("OpenSSH command must not be empty");
  }
#ifdef _WIN32
  wchar_t system_directory[MAX_PATH];
  const UINT length = GetSystemDirectoryW(system_directory, MAX_PATH);
  if (length > 0 && length < MAX_PATH) {
    const auto candidate = std::filesystem::path(system_directory) / L"OpenSSH" /
                           (std::filesystem::path(command).wstring() + L".exe");
    std::error_code ec;
    if (std::filesystem::is_regular_file(candidate, ec)) {
      return host_path_string(candidate);
    }
  }
  return std::string(command) + ".exe";
#else
  return command;
#endif
}

std::uint64_t host_process_id() {
#ifdef _WIN32
  return static_cast<std::uint64_t>(_getpid());
#else
  return static_cast<std::uint64_t>(getpid());
#endif
}

} // namespace simaai::neat::pcie::internal

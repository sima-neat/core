#include "HostPlatform.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
#include <process.h>
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace simaai::neat::pcie::internal {
namespace {

bool valid_temp_prefix(const std::string& prefix) {
  if (prefix.empty()) {
    return false;
  }
  for (const unsigned char ch : prefix) {
    if (!(std::isalnum(ch) || ch == '-' || ch == '_')) {
      return false;
    }
  }
  return true;
}

std::string temp_token() {
  static std::atomic<std::uint64_t> sequence{0};
  std::random_device random;
  std::ostringstream token;
  token << std::hex << std::setfill('0') << std::setw(8) << random() << std::setw(8) << random()
        << std::setw(16) << host_process_id() << std::setw(16)
        << sequence.fetch_add(1, std::memory_order_relaxed) << std::setw(16)
        << std::chrono::steady_clock::now().time_since_epoch().count();
  return token.str();
}

} // namespace

ScopedTempDirectory::ScopedTempDirectory(const std::string& prefix) {
  if (!valid_temp_prefix(prefix)) {
    throw std::invalid_argument("temporary directory prefix must use letters, digits, '-' or '_'");
  }
  const auto parent = std::filesystem::temp_directory_path();
  for (int attempt = 0; attempt < 128; ++attempt) {
    const auto candidate = parent / (prefix + "-" + temp_token());
#ifdef _WIN32
    if (CreateDirectoryW(candidate.c_str(), nullptr)) {
      path_ = candidate;
      return;
    }
    const auto error = GetLastError();
    if (error != ERROR_ALREADY_EXISTS) {
      throw std::system_error(static_cast<int>(error), std::system_category(),
                              "failed to create temporary directory");
    }
#else
    if (::mkdir(candidate.c_str(), S_IRWXU) == 0) {
      path_ = candidate;
      return;
    }
    if (errno != EEXIST) {
      throw std::system_error(errno, std::generic_category(),
                              "failed to create temporary directory");
    }
#endif
  }
  throw std::runtime_error("failed to allocate a unique temporary directory");
}

ScopedTempDirectory::~ScopedTempDirectory() noexcept {
  std::error_code ec;
  std::filesystem::remove_all(path_, ec);
}

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

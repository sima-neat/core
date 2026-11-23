#include "HostPlatform.h"
#include "pipeline/internal/Utf8Path.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <process.h>
#include <windows.h>
#include <bcrypt.h>
#include <shlobj.h>
#else
#include <signal.h>
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

#ifdef _WIN32
std::filesystem::path host_common_application_data_directory() {
  PWSTR path = nullptr;
  const HRESULT status =
      SHGetKnownFolderPath(FOLDERID_ProgramData, KF_FLAG_DEFAULT, nullptr, &path);
  std::unique_ptr<wchar_t, decltype(&CoTaskMemFree)> allocated(path, &CoTaskMemFree);
  if (FAILED(status) || !allocated) {
    throw std::runtime_error("Windows CommonApplicationData (FOLDERID_ProgramData) is unavailable");
  }
  return std::filesystem::path(allocated.get());
}
#endif

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

std::filesystem::path host_path_from_string(const std::string& path) {
  return pipeline_internal::utf8_path(path);
}

std::string host_path_string(const std::filesystem::path& path) {
  return pipeline_internal::utf8_string(path);
}

bool same_host_path(const std::filesystem::path& left, const std::filesystem::path& right) {
#ifdef _WIN32
  // Windows paths compare case-insensitively; canonical() returns the on-disk case.
  return CompareStringOrdinal(left.c_str(), -1, right.c_str(), -1, TRUE) == CSTR_EQUAL;
#else
  return left == right;
#endif
}

std::string open_ssh_executable(const char* command) {
  if (!command || !*command) {
    throw std::invalid_argument("OpenSSH command must not be empty");
  }
#ifdef _WIN32
  const auto executable = std::filesystem::path(command).wstring() + L".exe";
  wchar_t system_directory[MAX_PATH];
  const UINT length = GetSystemDirectoryW(system_directory, MAX_PATH);
  if (length > 0 && length < MAX_PATH) {
    const auto candidate = std::filesystem::path(system_directory) / L"OpenSSH" / executable;
    std::error_code ec;
    if (std::filesystem::is_regular_file(candidate, ec)) {
      return host_path_string(candidate);
    }
  }
  // CreateProcessW resolves a bare application name against the current directory, never
  // PATH, so search PATH's absolute entries here (e.g. C:\Program Files\OpenSSH).
  const DWORD size = GetEnvironmentVariableW(L"PATH", nullptr, 0);
  std::wstring path_list(size, L'\0');
  path_list.resize(size ? GetEnvironmentVariableW(L"PATH", path_list.data(), size) : 0);
  for (std::size_t begin = 0; begin <= path_list.size();) {
    const auto end = std::min(path_list.find(L';', begin), path_list.size());
    const auto candidate = std::filesystem::path(path_list.substr(begin, end - begin)) / executable;
    std::error_code ec;
    if (candidate.is_absolute() && std::filesystem::is_regular_file(candidate, ec)) {
      return host_path_string(candidate);
    }
    begin = end + 1;
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

void host_secure_random(unsigned char* buffer, std::size_t size) {
  if (!buffer && size != 0) {
    throw std::invalid_argument("random output buffer must not be null");
  }
  if (size == 0) {
    return;
  }
#ifdef _WIN32
  while (size > 0) {
    const auto chunk = static_cast<ULONG>(
        std::min(size, static_cast<std::size_t>(std::numeric_limits<ULONG>::max())));
    const NTSTATUS status =
        BCryptGenRandom(nullptr, buffer, chunk, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (status != 0) {
      throw std::runtime_error("BCryptGenRandom failed with NTSTATUS " +
                               std::to_string(static_cast<unsigned long>(status)));
    }
    buffer += chunk;
    size -= chunk;
  }
#else
  std::ifstream random("/dev/urandom", std::ios::binary);
  if (size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) ||
      !random.read(reinterpret_cast<char*>(buffer), static_cast<std::streamsize>(size))) {
    throw std::runtime_error("failed to read secure host randomness");
  }
#endif
}

namespace {

#ifdef _WIN32
// Process creation time, or "" when the process cannot be queried.
std::string process_start(const HANDLE process) {
  FILETIME created{}, exited{}, kernel{}, user{};
  if (!GetProcessTimes(process, &created, &exited, &kernel, &user)) {
    return "";
  }
  return std::to_string((static_cast<std::uint64_t>(created.dwHighDateTime) << 32) |
                        created.dwLowDateTime);
}

std::string host_computer_name() {
  wchar_t name[MAX_COMPUTERNAME_LENGTH + 1]{};
  DWORD size = MAX_COMPUTERNAME_LENGTH + 1;
  return GetComputerNameW(name, &size) ? host_path_string(std::filesystem::path(name)) : "";
}
#else
std::string first_line(const std::filesystem::path& path) {
  std::ifstream in(path);
  std::string line;
  std::getline(in, line);
  return line;
}

// The process state and start time (clock ticks after boot) from /proc/<pid>/stat.
std::optional<std::pair<char, std::string>> process_state_and_start(const std::uint64_t pid) {
  std::ifstream in("/proc/" + std::to_string(pid) + "/stat");
  const std::string stat((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const auto name_end = stat.rfind(')');
  if (name_end == std::string::npos) {
    return std::nullopt;
  }
  std::istringstream fields(stat.substr(name_end + 1));
  std::string state, field;
  fields >> state;
  // The state is field 3 and the start time field 22.
  for (int index = 4; index <= 22; ++index) {
    fields >> field;
  }
  if (!fields || state.size() != 1) {
    return std::nullopt;
  }
  return std::pair{state.front(), field};
}
#endif

} // namespace

HostProcessOwner current_host_process_owner() {
  HostProcessOwner owner;
  owner.pid = host_process_id();
#ifdef _WIN32
  owner.host = host_computer_name();
  if (!owner.host.empty()) {
    owner.host_id = "windows:" + owner.host;
  }
  owner.start = process_start(GetCurrentProcess());
#else
  char name[256]{};
  if (gethostname(name, sizeof(name) - 1) == 0) {
    owner.host = name;
  }
  // PIDs are only meaningful within one boot and one PID namespace (a container has its own).
  const std::string boot = first_line("/proc/sys/kernel/random/boot_id");
  std::error_code ec;
  const auto pid_namespace = std::filesystem::read_symlink("/proc/self/ns/pid", ec);
  if (!boot.empty() && !ec) {
    owner.host_id = "linux:" + boot + ":" + pid_namespace.string();
  }
  if (const auto stat = process_state_and_start(owner.pid)) {
    owner.start = stat->second;
  }
#endif
  return owner;
}

bool host_process_ended(const HostProcessOwner& owner) {
  if (owner.host_id.empty() || owner.pid == 0 ||
      owner.host_id != current_host_process_owner().host_id) {
    return false;
  }
#ifdef _WIN32
  if (owner.pid > std::numeric_limits<DWORD>::max()) {
    return false;
  }
  const HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE,
                                     static_cast<DWORD>(owner.pid));
  if (!process) {
    return GetLastError() == ERROR_INVALID_PARAMETER; // no process has this PID
  }
  const bool exited = WaitForSingleObject(process, 0) == WAIT_OBJECT_0;
  const std::string start = process_start(process);
  CloseHandle(process);
  return exited || (!owner.start.empty() && !start.empty() && start != owner.start);
#else
  if (owner.pid > static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
    return false;
  }
  if (::kill(static_cast<pid_t>(owner.pid), 0) != 0 && errno == ESRCH) {
    return true;
  }
  const auto stat = process_state_and_start(owner.pid);
  if (!stat) {
    return false; // hidden by /proc mount options, or already gone: kill() decides above
  }
  return stat->first == 'Z' || (!owner.start.empty() && stat->second != owner.start);
#endif
}

} // namespace simaai::neat::pcie::internal

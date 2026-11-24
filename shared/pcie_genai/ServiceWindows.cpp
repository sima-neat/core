#include "Service.h"

#include <simaai_svc.h>

#include <cerrno>
#include <climits>
#include <filesystem>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <windows.h>

namespace simaai::neat::pcie::genai::wire {
namespace {
void check(int rc, const char* operation) {
  if (rc)
    throw std::runtime_error(std::string("PCIe service ") + operation +
                             " failed: " + std::to_string(rc));
}

// The Platform runtime is the directory of the SiMaMlaDaemon service executable, as for
// sima-pcie-host-env.cmd and PyNeatPCIe. A bare DLL name would search the current directory.
std::filesystem::path platform_service_directory() {
  constexpr auto key = L"SYSTEM\\CurrentControlSet\\Services\\SiMaMlaDaemon";
  DWORD bytes = 0;
  if (RegGetValueW(HKEY_LOCAL_MACHINE, key, L"ImagePath", RRF_RT_REG_SZ, nullptr, nullptr,
                   &bytes) != ERROR_SUCCESS)
    return {};
  std::wstring command(bytes / sizeof(wchar_t), L'\0');
  if (RegGetValueW(HKEY_LOCAL_MACHINE, key, L"ImagePath", RRF_RT_REG_SZ, nullptr, command.data(),
                   &bytes) != ERROR_SUCCESS)
    return {};
  command.resize(command.find(L'\0'));
  command.erase(0, command.find_first_not_of(L" \t"));
  const bool quoted = !command.empty() && command.front() == L'"';
  const auto executable = quoted ? command.substr(1, command.find(L'"', 1) - 1)
                                 : command.substr(0, command.find_first_of(L" \t"));
  return std::filesystem::path(executable).parent_path();
}

std::wstring platform_library() {
  wchar_t directory[32768]{};
  const DWORD size = GetEnvironmentVariableW(L"SIMA_WINDOWS_PLATFORM_RUNTIME_DIR", directory,
                                             static_cast<DWORD>(std::size(directory)));
  if (size >= std::size(directory))
    throw std::runtime_error("SIMA_WINDOWS_PLATFORM_RUNTIME_DIR is too long");
  const auto runtime = size ? std::filesystem::path(directory) : platform_service_directory();
  if (!size && !runtime.is_absolute())
    throw std::runtime_error("The SiMaMlaDaemon service is not installed; install matching "
                             "Windows PCIe userspace or set SIMA_WINDOWS_PLATFORM_RUNTIME_DIR");
  return (runtime / L"SiMaPCIe.dll").wstring();
}

// SiMaPCIe.dll opens host files with ANSI APIs, so pass UTF-8 host paths in the active code page.
std::string code_page_path(const std::string& path) {
  const UINT code_page = GetACP();
  if (path.empty() || code_page == CP_UTF8)
    return path;
  const auto unrepresentable = [&] {
    return std::invalid_argument("Host path cannot be represented in the system code page " +
                                 std::to_string(code_page) +
                                 "; keep GenAI model folders on paths it can represent: " + path);
  };
  if (path.size() > INT_MAX)
    throw unrepresentable();
  const int length = static_cast<int>(path.size());
  const int wide_length =
      MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.data(), length, nullptr, 0);
  if (wide_length <= 0)
    throw unrepresentable();
  std::wstring wide(static_cast<size_t>(wide_length), L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.data(), length, wide.data(), wide_length);
  BOOL defaulted = FALSE;
  const int narrow_length = WideCharToMultiByte(code_page, WC_NO_BEST_FIT_CHARS, wide.data(),
                                                wide_length, nullptr, 0, nullptr, &defaulted);
  if (narrow_length <= 0 || defaulted)
    throw unrepresentable();
  std::string narrow(static_cast<size_t>(narrow_length), '\0');
  WideCharToMultiByte(code_page, WC_NO_BEST_FIT_CHARS, wide.data(), wide_length, narrow.data(),
                      narrow_length, nullptr, nullptr);
  return narrow;
}
} // namespace

struct Service::Impl {
  HMODULE library = nullptr;
  simaai_svc* handle = nullptr;
  decltype(&simaai_svc_close) close = nullptr;
  decltype(&simaai_svc_subscribe) subscribe = nullptr;
  decltype(&simaai_svc_notify) notify = nullptr;
  decltype(&simaai_svc_recv) recv = nullptr;
  decltype(&simaai_svc_get_file) fetch = nullptr;
  decltype(&simaai_svc_put_file) put = nullptr;
  decltype(&simaai_svc_handle) native_handle = nullptr;
  std::vector<char> buffer = std::vector<char>(SIMAAI_SVC_PAYLOAD_MAX);

  template <class T> T symbol(const char* name) {
    const auto address = GetProcAddress(library, name);
    if (!address)
      throw std::runtime_error(std::string("Installed PCIe userspace lacks ") + name);
    return reinterpret_cast<T>(address);
  }

  ~Impl() {
    if (handle && close)
      close(handle);
    if (library)
      FreeLibrary(library);
  }
};

Service::Service(int card, bool endpoint) : impl_(std::make_unique<Impl>()) {
  const auto library_path = platform_library();
  impl_->library = LoadLibraryW(library_path.c_str());
  if (!impl_->library)
    throw std::runtime_error("Install matching Windows PCIe userspace or set "
                             "SIMA_WINDOWS_PLATFORM_RUNTIME_DIR");
  impl_->close = impl_->symbol<decltype(impl_->close)>("simaai_svc_close");
  impl_->subscribe = impl_->symbol<decltype(impl_->subscribe)>("simaai_svc_subscribe");
  impl_->notify = impl_->symbol<decltype(impl_->notify)>("simaai_svc_notify");
  impl_->recv = impl_->symbol<decltype(impl_->recv)>("simaai_svc_recv");
  impl_->fetch = impl_->symbol<decltype(impl_->fetch)>("simaai_svc_get_file");
  impl_->put = impl_->symbol<decltype(impl_->put)>("simaai_svc_put_file");
  impl_->native_handle = impl_->symbol<decltype(impl_->native_handle)>("simaai_svc_handle");
  if (endpoint) {
    const auto open = impl_->symbol<int (*)(const char*, simaai_svc**)>("simaai_svc_open");
    check(open(nullptr, &impl_->handle), "open endpoint");
  } else {
    if (card < 0)
      throw std::invalid_argument("card_id must be nonnegative");
    const auto open = impl_->symbol<int (*)(uint32_t, simaai_svc**)>("simaai_svc_open_card");
    // Platform reports a stopped SiMaMlaDaemon service only as an error code, -2 among them.
    if (const int rc = open(static_cast<uint32_t>(card), &impl_->handle))
      throw std::runtime_error(
          "PCIe service open card failed: " + std::to_string(rc) +
          ". The SiMa Platform service may not be running: check it with Get-Service "
          "SiMaMlaDaemon and start it with Start-Service SiMaMlaDaemon from an elevated "
          "PowerShell. To check the whole host, run "
          R"(& "$env:ProgramFiles\SimaPCIeHost\bin\sima-pcie-host-doctor.cmd")");
  }
}

Service::~Service() = default;

void Service::interrupt() noexcept {
  if (!impl_ || !impl_->handle || !impl_->native_handle)
    return;
  const HANDLE handle = impl_->native_handle(impl_->handle);
  if (handle != INVALID_HANDLE_VALUE)
    CancelIoEx(handle, nullptr);
}

void Service::subscribe(const std::string& tag) {
  check(impl_->subscribe(impl_->handle, tag.c_str()), "subscribe");
}

void Service::send(const std::string& tag, const std::string& payload) {
  if (payload.size() > SIMAAI_SVC_PAYLOAD_MAX)
    throw std::length_error("PCIe message too large");
  simaai_svc_note note{};
  note.type = SIMAAI_NOTE_EVENT;
  note.severity = SIMAAI_NOTE_SEV_INFO;
  note.tag = tag.c_str();
  note.payload = payload.data();
  note.payload_len = payload.size();
  check(impl_->notify(impl_->handle, &note, nullptr), "notify");
}

std::optional<std::string> Service::receive(int timeout_ms) {
  simaai_svc_note note{};
  const int rc =
      impl_->recv(impl_->handle, &note, impl_->buffer.data(), impl_->buffer.size(), timeout_ms);
  if (rc == -EAGAIN || rc == -ETIMEDOUT)
    return std::nullopt;
  check(rc, "receive");
  return std::string(static_cast<const char*>(note.payload), note.payload_len);
}

void Service::fetch(const std::string& root, const std::string& source,
                    const std::string& destination) {
  simaai_svc_xfer_opts options{};
  options.flags = SIMAAI_SVC_XF_OVERWRITE;
  simaai_svc_xfer statistics{};
  const auto local = code_page_path(destination);
  check(impl_->fetch(impl_->handle, root.c_str(), source.c_str(), local.c_str(), &options,
                     &statistics),
        "fetch");
}

void Service::put(const std::string& source, const std::string& destination) {
  simaai_svc_xfer_opts options{};
  options.flags = SIMAAI_SVC_XF_OVERWRITE;
  simaai_svc_xfer statistics{};
  const auto local = code_page_path(source);
  check(
      impl_->put(impl_->handle, local.c_str(), nullptr, destination.c_str(), &options, &statistics),
      "put model asset");
}
} // namespace simaai::neat::pcie::genai::wire

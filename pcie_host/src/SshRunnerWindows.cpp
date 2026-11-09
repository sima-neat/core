#include "SshRunner.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

#define NOMINMAX
#include <windows.h>

namespace simaai::neat::pcie::internal {
namespace {

std::runtime_error windows_error(const char* operation, DWORD error = GetLastError());

class WinHandle {
public:
  WinHandle() = default;
  explicit WinHandle(HANDLE handle) : handle_(handle) {}
  ~WinHandle() {
    reset();
  }
  WinHandle(const WinHandle&) = delete;
  WinHandle& operator=(const WinHandle&) = delete;
  WinHandle(WinHandle&& other) noexcept : handle_(other.release()) {}
  WinHandle& operator=(WinHandle&& other) noexcept {
    if (this != &other) {
      reset(other.release());
    }
    return *this;
  }
  HANDLE get() const {
    return handle_;
  }
  HANDLE release() {
    const HANDLE handle = handle_;
    handle_ = nullptr;
    return handle;
  }
  void reset(HANDLE handle = nullptr) {
    if (handle_ && handle_ != INVALID_HANDLE_VALUE) {
      CloseHandle(handle_);
    }
    handle_ = handle;
  }
  explicit operator bool() const {
    return handle_ && handle_ != INVALID_HANDLE_VALUE;
  }

private:
  HANDLE handle_ = nullptr;
};

class ProcessAttributeList {
public:
  ProcessAttributeList() {
    SIZE_T bytes = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
    if (bytes == 0 || GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
      throw windows_error("InitializeProcThreadAttributeList(size)");
    }
    storage_.resize(bytes);
    list_ = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage_.data());
    if (!InitializeProcThreadAttributeList(list_, 1, 0, &bytes)) {
      throw windows_error("InitializeProcThreadAttributeList");
    }
  }
  ~ProcessAttributeList() {
    if (list_) {
      DeleteProcThreadAttributeList(list_);
    }
  }
  ProcessAttributeList(const ProcessAttributeList&) = delete;
  ProcessAttributeList& operator=(const ProcessAttributeList&) = delete;

  void inherit_handles(HANDLE* handles, const std::size_t count) {
    if (!UpdateProcThreadAttribute(list_, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, handles,
                                   count * sizeof(HANDLE), nullptr, nullptr)) {
      throw windows_error("UpdateProcThreadAttribute(handle list)");
    }
  }

  LPPROC_THREAD_ATTRIBUTE_LIST get() const {
    return list_;
  }

private:
  std::vector<std::byte> storage_;
  LPPROC_THREAD_ATTRIBUTE_LIST list_ = nullptr;
};

std::runtime_error windows_error(const char* operation, const DWORD error) {
  return std::runtime_error(std::string(operation) + " failed with Windows error " +
                            std::to_string(error));
}

std::wstring utf8_to_wide(const std::string& value) {
  if (value.empty()) {
    return {};
  }
  if (value.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error("command argument is too large");
  }
  const int input_size = static_cast<int>(value.size());
  const int size =
      MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), input_size, nullptr, 0);
  if (size <= 0) {
    throw windows_error("UTF-8 command argument conversion");
  }
  std::wstring out(static_cast<std::size_t>(size), L'\0');
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), input_size, out.data(),
                          size) != size) {
    throw windows_error("UTF-8 command argument conversion");
  }
  return out;
}

std::wstring windows_command_line(const std::vector<std::string>& args) {
  std::wstring command_line;
  for (const auto& arg : args) {
    if (!command_line.empty()) {
      command_line.push_back(L' ');
    }
    command_line += utf8_to_wide(SshRunner::windows_quote_argument(arg));
  }
  return command_line;
}

DWORD windows_timeout(const std::chrono::milliseconds timeout) {
  const auto count = std::max<std::chrono::milliseconds::rep>(timeout.count(), 1);
  return static_cast<DWORD>(std::min<std::chrono::milliseconds::rep>(
      count, static_cast<std::chrono::milliseconds::rep>(INFINITE - 1)));
}

} // namespace

CommandResult SshRunner::run_for(const std::vector<std::string>& args,
                                 const std::chrono::milliseconds timeout) {
  if (args.empty()) {
    throw std::invalid_argument("SshRunner::run requires a command");
  }

  SECURITY_ATTRIBUTES security{};
  security.nLength = sizeof(security);
  security.bInheritHandle = TRUE;

  HANDLE read_raw = nullptr;
  HANDLE write_raw = nullptr;
  if (!CreatePipe(&read_raw, &write_raw, &security, 0)) {
    throw windows_error("CreatePipe");
  }
  WinHandle read_pipe(read_raw);
  WinHandle write_pipe(write_raw);
  if (!SetHandleInformation(read_pipe.get(), HANDLE_FLAG_INHERIT, 0)) {
    throw windows_error("SetHandleInformation");
  }

  WinHandle null_input(CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                   &security, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
  if (!null_input) {
    throw windows_error("CreateFileW(NUL)");
  }

  WinHandle job(CreateJobObjectW(nullptr, nullptr));
  if (!job) {
    throw windows_error("CreateJobObjectW");
  }
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION job_limits{};
  job_limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if (!SetInformationJobObject(job.get(), JobObjectExtendedLimitInformation, &job_limits,
                               sizeof(job_limits))) {
    throw windows_error("SetInformationJobObject");
  }

  HANDLE inherited_handles[] = {null_input.get(), write_pipe.get()};
  ProcessAttributeList attributes;
  attributes.inherit_handles(inherited_handles, std::size(inherited_handles));

  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = null_input.get();
  startup.StartupInfo.hStdOutput = write_pipe.get();
  startup.StartupInfo.hStdError = write_pipe.get();
  startup.lpAttributeList = attributes.get();
  PROCESS_INFORMATION process_info{};
  std::wstring executable = utf8_to_wide(args.front());
  std::wstring command_line = windows_command_line(args);
  if (!CreateProcessW(executable.c_str(), command_line.data(), nullptr, nullptr, TRUE,
                      CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT, nullptr,
                      nullptr, &startup.StartupInfo, &process_info)) {
    throw windows_error("CreateProcessW");
  }
  WinHandle process(process_info.hProcess);
  WinHandle process_thread(process_info.hThread);
  write_pipe.reset();
  null_input.reset();

  if (!AssignProcessToJobObject(job.get(), process.get())) {
    const DWORD error = GetLastError();
    TerminateProcess(process.get(), 1);
    WaitForSingleObject(process.get(), 5000);
    throw windows_error("AssignProcessToJobObject", error);
  }
  if (ResumeThread(process_thread.get()) == static_cast<DWORD>(-1)) {
    const DWORD error = GetLastError();
    TerminateJobObject(job.get(), 1);
    WaitForSingleObject(process.get(), 5000);
    throw windows_error("ResumeThread", error);
  }
  process_thread.reset();

  CommandResult result;
  std::thread reader([&result, handle = read_pipe.get()] {
    std::array<char, 4096> buffer{};
    DWORD count = 0;
    while (ReadFile(handle, buffer.data(), static_cast<DWORD>(buffer.size()), &count, nullptr)) {
      if (count > 0) {
        result.output.append(buffer.data(), static_cast<std::size_t>(count));
      }
    }
  });

  const DWORD wait_result = WaitForSingleObject(
      process.get(), windows_timeout(std::max(timeout, std::chrono::milliseconds(1))));
  DWORD wait_error = ERROR_SUCCESS;
  if (wait_result == WAIT_TIMEOUT) {
    result.timed_out = true;
    if (!TerminateJobObject(job.get(), 124)) {
      wait_error = GetLastError();
    }
    WaitForSingleObject(process.get(), 5000);
  } else if (wait_result == WAIT_FAILED) {
    wait_error = GetLastError();
    TerminateJobObject(job.get(), 1);
    WaitForSingleObject(process.get(), 5000);
  }

  DWORD exit_code = 1;
  if (!result.timed_out && wait_result == WAIT_OBJECT_0 &&
      GetExitCodeProcess(process.get(), &exit_code) &&
      exit_code <= static_cast<DWORD>(std::numeric_limits<int>::max())) {
    result.exit_code = static_cast<int>(exit_code);
  } else if (result.timed_out) {
    result.exit_code = 124;
  }

  // Closing the kill-on-close job retires descendants that inherited the
  // redirected handles. This guarantees EOF before joining the reader.
  job.reset();
  reader.join();
  read_pipe.reset();

  if (wait_error != ERROR_SUCCESS) {
    throw windows_error(wait_result == WAIT_FAILED ? "WaitForSingleObject" : "TerminateJobObject",
                        wait_error);
  }
  return result;
}

} // namespace simaai::neat::pcie::internal

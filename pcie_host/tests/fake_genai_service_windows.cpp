// Local-only daemon substitute. Never opens a PCIe device or connects to SSH.
#include <simaai_svc.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#define NOMINMAX
#include <windows.h>

struct simaai_svc {
  HANDLE reader = INVALID_HANDLE_VALUE;
  HANDLE writer = INVALID_HANDLE_VALUE;
  std::string tag;
  std::deque<std::string> messages;
};

namespace {
std::mutex mutex;
std::vector<simaai_svc*> clients;
std::filesystem::path receive_root;
std::atomic<int> copies{0}, put_error{0};
std::atomic<bool> block_put{false}, pause_put{false}, put_paused{false}, put_waiting{false};
std::atomic<unsigned long> pipe_sequence{0};

int windows_error() {
  return -EIO;
}

bool make_pipe(simaai_svc& client) {
  const auto name = L"\\\\.\\pipe\\neat-genai-fake-" + std::to_wstring(GetCurrentProcessId()) +
                    L"-" + std::to_wstring(++pipe_sequence);
  client.reader =
      CreateNamedPipeW(name.c_str(), PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED,
                       PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 1, 4096, 4096, 0, nullptr);
  if (client.reader == INVALID_HANDLE_VALUE)
    return false;

  OVERLAPPED connected{};
  connected.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
  if (!connected.hEvent)
    return false;
  const BOOL pending = ConnectNamedPipe(client.reader, &connected);
  const DWORD connect_error = pending ? ERROR_SUCCESS : GetLastError();
  if (!pending && connect_error != ERROR_IO_PENDING && connect_error != ERROR_PIPE_CONNECTED) {
    CloseHandle(connected.hEvent);
    return false;
  }
  client.writer = CreateFileW(name.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
  if (client.writer == INVALID_HANDLE_VALUE) {
    CancelIoEx(client.reader, &connected);
    CloseHandle(connected.hEvent);
    return false;
  }
  if (connect_error == ERROR_IO_PENDING &&
      WaitForSingleObject(connected.hEvent, 5000) != WAIT_OBJECT_0) {
    CancelIoEx(client.reader, &connected);
    CloseHandle(connected.hEvent);
    return false;
  }
  CloseHandle(connected.hEvent);
  return true;
}

int wait_for_signal(simaai_svc* client, int timeout_ms, std::atomic<bool>* pending = nullptr) {
  OVERLAPPED operation{};
  operation.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
  if (!operation.hEvent)
    return windows_error();
  char byte = 0;
  DWORD count = 0;
  const BOOL complete = ReadFile(client->reader, &byte, 1, &count, &operation);
  if (complete) {
    CloseHandle(operation.hEvent);
    return count == 1 ? 0 : windows_error();
  }
  if (GetLastError() != ERROR_IO_PENDING) {
    CloseHandle(operation.hEvent);
    return windows_error();
  }
  if (pending)
    *pending = true;
  const DWORD timeout = timeout_ms < 0 ? INFINITE : static_cast<DWORD>(timeout_ms);
  const DWORD waited = WaitForSingleObject(operation.hEvent, timeout);
  if (waited == WAIT_TIMEOUT) {
    CancelIoEx(client->reader, &operation);
    WaitForSingleObject(operation.hEvent, INFINITE);
    const BOOL completed = GetOverlappedResult(client->reader, &operation, &count, FALSE);
    CloseHandle(operation.hEvent);
    return completed && count == 1 ? 0 : -EAGAIN;
  }
  if (waited != WAIT_OBJECT_0 || !GetOverlappedResult(client->reader, &operation, &count, FALSE)) {
    CloseHandle(operation.hEvent);
    return windows_error();
  }
  CloseHandle(operation.hEvent);
  return count == 1 ? 0 : windows_error();
}

int deliver(simaai_svc* client, std::string message) {
  client->messages.push_back(std::move(message));
  const char byte = 'x';
  DWORD count = 0;
  return WriteFile(client->writer, &byte, 1, &count, nullptr) && count == 1 ? 0 : windows_error();
}
} // namespace

extern "C" {
void test_genai_receive_root(const char* root) {
  receive_root = root;
}
int test_genai_copies() {
  return copies;
}
void test_genai_put_error(int error) {
  put_error = error;
}
void test_genai_block_put(bool block) {
  block_put = block;
}
// Holds put() before its read is pending, the window a one-shot CancelIoEx misses.
void test_genai_pause_put(bool pause) {
  pause_put = pause;
}
bool test_genai_put_paused() {
  return put_paused;
}
// True once a blocked put() has a read pending that CancelIoEx can cancel.
bool test_genai_put_waiting() {
  return put_waiting;
}
int test_genai_open_clients() {
  std::lock_guard lock(mutex);
  return static_cast<int>(clients.size());
}
int simaai_svc_open_card(uint32_t, simaai_svc** out) {
  auto* client = new simaai_svc;
  if (!make_pipe(*client)) {
    if (client->reader != INVALID_HANDLE_VALUE)
      CloseHandle(client->reader);
    if (client->writer != INVALID_HANDLE_VALUE)
      CloseHandle(client->writer);
    delete client;
    return windows_error();
  }
  std::lock_guard lock(mutex);
  clients.push_back(client);
  *out = client;
  return 0;
}
int simaai_svc_open(const char*, simaai_svc** out) {
  return simaai_svc_open_card(0, out);
}
void simaai_svc_close(simaai_svc* client) {
  if (!client)
    return;
  std::lock_guard lock(mutex);
  clients.erase(std::find(clients.begin(), clients.end(), client));
  CloseHandle(client->reader);
  CloseHandle(client->writer);
  delete client;
}
HANDLE simaai_svc_handle(const simaai_svc* client) {
  return client ? client->reader : INVALID_HANDLE_VALUE;
}
int simaai_svc_subscribe(simaai_svc* client, const char* tag) {
  std::lock_guard lock(mutex);
  client->tag = tag;
  return 0;
}
int simaai_svc_notify(simaai_svc*, const simaai_svc_note* note, unsigned int*) {
  std::lock_guard lock(mutex);
  for (auto* client : clients) {
    if (client->tag == note->tag) {
      const int result =
          deliver(client, std::string(static_cast<const char*>(note->payload), note->payload_len));
      if (result)
        return result;
    }
  }
  return 0;
}
int simaai_svc_recv(simaai_svc* client, simaai_svc_note* note, void* buffer, size_t capacity,
                    int timeout_ms) {
  const int waited = wait_for_signal(client, timeout_ms);
  if (waited)
    return waited;
  std::lock_guard lock(mutex);
  if (client->messages.empty())
    return -EAGAIN;
  auto message = std::move(client->messages.front());
  client->messages.pop_front();
  if (message.size() > capacity)
    return -EMSGSIZE;
  std::memcpy(buffer, message.data(), message.size());
  note->payload = buffer;
  note->payload_len = message.size();
  return 0;
}
int simaai_svc_get_file(simaai_svc*, const char*, const char*, const char*,
                        const simaai_svc_xfer_opts*, simaai_svc_xfer*) {
  return -ENOSYS;
}
int simaai_svc_put_file(simaai_svc* client, const char* source, const char* root,
                        const char* destination, const simaai_svc_xfer_opts*, simaai_svc_xfer*) {
  while (pause_put) {
    put_paused = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  put_paused = false;
  put_waiting = false;
  ++copies;
  if (block_put) {
    const int result = wait_for_signal(client, -1, &put_waiting);
    if (result)
      return result;
  }
  if (put_error)
    return put_error;
  // Decodes the host path in the active code page, as the released SiMaPCIe.dll does.
  const std::filesystem::path local(source);
  if (root != nullptr || !local.is_absolute())
    return -EINVAL;
  const auto target = receive_root / destination;
  std::filesystem::create_directories(target.parent_path());
  std::error_code error;
  std::filesystem::copy_file(local, target, std::filesystem::copy_options::overwrite_existing,
                             error);
  return error ? -ENOENT : 0;
}
}

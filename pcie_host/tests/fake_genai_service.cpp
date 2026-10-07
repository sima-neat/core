// Local-only daemon substitute. Never opens a PCIe device or connects to SSH.
#include <simaai_svc.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <poll.h>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

struct simaai_svc {
  int sockets[2];
  std::string tag;
  std::deque<std::string> messages;
};
namespace {
std::mutex mutex;
std::vector<simaai_svc*> clients;
std::filesystem::path receive_root;
std::atomic<int> copies{0}, put_error{0};
std::atomic<bool> block_put{false};
std::atomic<bool> auto_reply{false};
int deliver(simaai_svc* client, std::string message) {
  client->messages.push_back(std::move(message));
  return send(client->sockets[1], "x", 1, MSG_NOSIGNAL) == 1 ? 0 : -ECONNRESET;
}
bool disconnected(simaai_svc* client, int timeout) {
  pollfd fd{client->sockets[0], POLLIN, 0};
  return poll(&fd, 1, timeout) > 0 && (fd.revents & (POLLHUP | POLLERR));
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
void test_genai_auto_reply(bool enabled) {
  auto_reply = enabled;
}
int simaai_svc_open_card(uint32_t, simaai_svc** out) {
  auto* client = new simaai_svc;
  if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, client->sockets) != 0) {
    delete client;
    return -errno;
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
  std::lock_guard lock(mutex);
  clients.erase(std::find(clients.begin(), clients.end(), client));
  close(client->sockets[0]);
  close(client->sockets[1]);
  delete client;
}
int simaai_svc_fd(const simaai_svc* client) {
  return client->sockets[0];
}
int simaai_svc_subscribe(simaai_svc* client, const char* tag) {
  std::lock_guard lock(mutex);
  client->tag = tag;
  return 0;
}
int simaai_svc_notify(simaai_svc* sender, const simaai_svc_note* note, unsigned int*) {
  if (disconnected(sender, 0))
    return -ECONNRESET;
  std::lock_guard lock(mutex);
  for (auto* client : clients) {
    if (client->tag == note->tag) {
      if (deliver(client, std::string(static_cast<const char*>(note->payload), note->payload_len)))
        return -ECONNRESET;
    }
  }
  if (auto_reply && std::string_view(note->tag).ends_with(".c")) {
    auto reply = nlohmann::json::parse(
        std::string_view(static_cast<const char*>(note->payload), note->payload_len));
    if (reply.at("kind") == "hello") {
      reply["capabilities"] = {{"text", true}, {"image", true}, {"audio", false}};
    } else if (reply.at("kind") == "generate" && reply.at("body").at("prompt") != "blocked") {
      reply["kind"] = "sample";
      reply["sequence"] = uint64_t{1};
      reply["body"] = {{"text", "answer"}, {"reasoning", ""},
                       {"final", true},    {"finish_reason", "stop"},
                       {"language", ""},   {"tool_calls", nlohmann::json::array()},
                       {"tokens", 1},      {"ttft", 0.0},
                       {"tps", 0.0}};
    } else if (reply.at("kind") == "cancel") {
      reply["kind"] = "sample";
      reply["sequence"] = uint64_t{1};
      reply["body"] = {{"text", ""},     {"reasoning", ""},
                       {"final", true},  {"finish_reason", "interrupted"},
                       {"language", ""}, {"tool_calls", nlohmann::json::array()},
                       {"tokens", 0},    {"ttft", 0.0},
                       {"tps", 0.0}};
    } else {
      return 0;
    }
    return deliver(sender, reply.dump());
  }
  return 0;
}
int simaai_svc_recv(simaai_svc* client, simaai_svc_note* note, void* buffer, size_t capacity,
                    int timeout) {
  pollfd fd{client->sockets[0], POLLIN, 0};
  if (poll(&fd, 1, timeout) <= 0)
    return -EAGAIN;
  char byte;
  if (read(fd.fd, &byte, 1) != 1)
    return -ECONNRESET;
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
  ++copies;
  while (block_put)
    if (disconnected(client, 10))
      return -ECONNRESET;
  if (put_error)
    return put_error;
  if (root != nullptr || !std::filesystem::path(source).is_absolute())
    return -EINVAL;
  const auto target = receive_root / destination;
  std::filesystem::create_directories(target.parent_path());
  std::filesystem::copy_file(source, target, std::filesystem::copy_options::overwrite_existing);
  return 0;
}
}

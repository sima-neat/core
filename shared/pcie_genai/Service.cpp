#include "Service.h"
#include <simaai_svc.h>
#include <cerrno>
#include <dlfcn.h>
#include <stdexcept>
#include <sys/socket.h>
#include <vector>

namespace simaai::neat::pcie::genai::wire {
namespace {
void check(int rc, const char* operation) {
  if (rc)
    throw std::runtime_error(std::string("PCIe service ") + operation +
                             " failed: " + std::to_string(rc));
}
} // namespace
struct Service::Impl {
  void* library = nullptr;
  simaai_svc* handle = nullptr;
  decltype(&simaai_svc_close) close = nullptr;
  decltype(&simaai_svc_subscribe) subscribe = nullptr;
  decltype(&simaai_svc_notify) notify = nullptr;
  decltype(&simaai_svc_recv) recv = nullptr;
  decltype(&simaai_svc_get_file) fetch = nullptr;
  decltype(&simaai_svc_put_file) put = nullptr;
  decltype(&simaai_svc_fd) fd = nullptr;
  std::vector<char> buffer = std::vector<char>(SIMAAI_SVC_PAYLOAD_MAX);
  template <class T> T symbol(const char* name) {
    auto fn = reinterpret_cast<T>(dlsym(library, name));
    if (!fn)
      throw std::runtime_error(std::string("Installed PCIe userspace lacks ") + name);
    return fn;
  }
  ~Impl() {
    if (handle && close)
      close(handle);
    if (library)
      dlclose(library);
  }
};
Service::Service(int card, bool endpoint) : impl_(std::make_unique<Impl>()) {
  const char* name = endpoint ? "libsimaaipep.so" : "libsimaaipcie.so";
  impl_->library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  if (!impl_->library)
    throw std::runtime_error(std::string("Install matching PCIe userspace: ") + dlerror());
  impl_->close = impl_->symbol<decltype(impl_->close)>("simaai_svc_close");
  impl_->subscribe = impl_->symbol<decltype(impl_->subscribe)>("simaai_svc_subscribe");
  impl_->notify = impl_->symbol<decltype(impl_->notify)>("simaai_svc_notify");
  impl_->recv = impl_->symbol<decltype(impl_->recv)>("simaai_svc_recv");
  impl_->fetch = impl_->symbol<decltype(impl_->fetch)>("simaai_svc_get_file");
  impl_->put = impl_->symbol<decltype(impl_->put)>("simaai_svc_put_file");
  impl_->fd = impl_->symbol<decltype(impl_->fd)>("simaai_svc_fd");
  if (endpoint) {
    auto open = impl_->symbol<int (*)(const char*, simaai_svc**)>("simaai_svc_open");
    check(open(nullptr, &impl_->handle), "open endpoint");
  } else {
    if (card < 0)
      throw std::invalid_argument("card_id must be nonnegative");
    auto open = impl_->symbol<int (*)(uint32_t, simaai_svc**)>("simaai_svc_open_card");
    check(open(static_cast<uint32_t>(card), &impl_->handle), "open card");
  }
}
Service::~Service() = default;
void Service::interrupt() noexcept {
  shutdown(impl_->fd(impl_->handle), SHUT_RDWR);
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
bool Service::fetch(const std::string& root, const std::string& source,
                    const std::string& destination, bool optional) {
  simaai_svc_xfer_opts opts{};
  opts.flags = SIMAAI_SVC_XF_OVERWRITE;
  simaai_svc_xfer stats{};
  const int rc =
      impl_->fetch(impl_->handle, root.c_str(), source.c_str(), destination.c_str(), &opts, &stats);
  if (optional && rc == -ENOENT)
    return false;
  check(rc, "fetch");
  return true;
}
void Service::put(const std::string& source, const std::string& destination) {
  simaai_svc_xfer_opts opts{};
  opts.flags = SIMAAI_SVC_XF_OVERWRITE;
  simaai_svc_xfer stats{};
  // A null destination selects the card daemon's default receive root.
  check(impl_->put(impl_->handle, source.c_str(), nullptr, destination.c_str(), &opts, &stats),
        "put model asset");
}
} // namespace simaai::neat::pcie::genai::wire

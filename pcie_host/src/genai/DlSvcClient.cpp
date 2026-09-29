// DlSvcClient: loads libsimaaipcie.so at run time and wraps the few
// simaai_svc calls SvcTransport needs (open_card, subscribe, notify, recv).
// It turns the C error codes into RecvStatus or exceptions.

#include "genai/DlSvcClient.h"

#include "simaai_svc.h" // header only: struct simaai_svc_note, SIMAAI_* constants

#include <cerrno>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <dlfcn.h>

namespace simaai::neat::pcie::genai::internal {

namespace {

struct SvcApi {
  using open_card_fn = int (*)(uint32_t, struct simaai_svc**);
  using close_fn = void (*)(struct simaai_svc*);
  using subscribe_fn = int (*)(struct simaai_svc*, const char*);
  using notify_fn = int (*)(struct simaai_svc*, const struct simaai_svc_note*, unsigned int*);
  using recv_fn = int (*)(struct simaai_svc*, struct simaai_svc_note*, void*, size_t, int);

  open_card_fn open_card = nullptr;
  close_fn close = nullptr;
  subscribe_fn subscribe = nullptr;
  notify_fn notify = nullptr;
  recv_fn recv = nullptr;
};

// dlopen, not a link: the host package does not depend on libsimaaipcie.so.
// A host without the PCIe host package still runs everything else, and
// pcie-genai fails with a clear message instead of a loader error.
// RTLD_LOCAL, not RTLD_GLOBAL: on the card, RTLD_GLOBAL once let the
// svc library bind a clashing global symbol and a 670 MB pull failed with
// -EPROTO. We keep the same safe rule here; dlsym on the handle still works.
SvcApi load_api(const char* library) {
  void* handle = ::dlopen(library, RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char* why = ::dlerror();
    throw std::runtime_error(std::string("PCIe support unavailable: cannot load ") + library +
                             " (" + (why ? why : "unknown error") +
                             "). Install the sima PCIe host package.");
  }
  SvcApi api;
  api.open_card = reinterpret_cast<SvcApi::open_card_fn>(::dlsym(handle, "simaai_svc_open_card"));
  api.close = reinterpret_cast<SvcApi::close_fn>(::dlsym(handle, "simaai_svc_close"));
  api.subscribe = reinterpret_cast<SvcApi::subscribe_fn>(::dlsym(handle, "simaai_svc_subscribe"));
  api.notify = reinterpret_cast<SvcApi::notify_fn>(::dlsym(handle, "simaai_svc_notify"));
  api.recv = reinterpret_cast<SvcApi::recv_fn>(::dlsym(handle, "simaai_svc_recv"));
  if (!api.open_card || !api.close || !api.subscribe || !api.notify || !api.recv) {
    throw std::runtime_error(std::string("PCIe support unavailable: ") + library +
                             " is missing simaai_svc symbols");
  }
  return api; // the handle stays open for the process lifetime, like on the card
}

} // namespace

struct DlSvcClient::Impl {
  SvcApi api;
  struct simaai_svc* handle = nullptr;
  std::vector<char> buffer = std::vector<char>(SIMAAI_SVC_PAYLOAD_MAX);
};

DlSvcClient::DlSvcClient(const std::uint32_t card_id, const char* library)
    : impl_(std::make_unique<Impl>()) {
  impl_->api = load_api(library);
  const int rc = impl_->api.open_card(card_id, &impl_->handle);
  if (rc != 0) {
    throw std::runtime_error("simaai_svc_open_card(" + std::to_string(card_id) +
                             ") failed: rc=" + std::to_string(rc) + " (is simaai-mla-daemon@" +
                             std::to_string(card_id) + " running?)");
  }
}

DlSvcClient::~DlSvcClient() {
  if (impl_ && impl_->handle != nullptr) {
    impl_->api.close(impl_->handle);
  }
}

void DlSvcClient::subscribe(const std::string& tag) {
  const int rc = impl_->api.subscribe(impl_->handle, tag.c_str());
  if (rc != 0) {
    throw std::runtime_error("simaai_svc_subscribe(" + tag + ") failed: rc=" + std::to_string(rc));
  }
}

unsigned DlSvcClient::notify(const std::string& tag, const std::string& payload) {
  struct simaai_svc_note note {};
  note.type = SIMAAI_NOTE_EVENT;
  note.severity = SIMAAI_NOTE_SEV_INFO;
  note.tag = tag.c_str();
  note.payload = payload.empty() ? nullptr : payload.data();
  note.payload_len = payload.size();
  unsigned int subscribers = 0;
  const int rc = impl_->api.notify(impl_->handle, &note, &subscribers);
  if (rc != 0) {
    throw std::runtime_error("simaai_svc_notify(" + tag + ") failed: rc=" + std::to_string(rc));
  }
  return subscribers;
}

RecvStatus DlSvcClient::recv(SvcNote& out, const int timeout_ms) {
  struct simaai_svc_note note {};
  const int rc =
      impl_->api.recv(impl_->handle, &note, impl_->buffer.data(), impl_->buffer.size(), timeout_ms);
  // -EAGAIN: nothing arrived in timeout_ms (normal; the caller loops).
  // -ECONNRESET: the local daemon went away; the caller stops the run.
  // -ENOSPC: the note was bigger than our buffer (see below).
  if (rc == -EAGAIN) {
    return RecvStatus::Timeout;
  }
  if (rc == -ECONNRESET) {
    return RecvStatus::Disconnected;
  }
  if (rc == -ENOSPC) {
    // The note was consumed but did not fit the 1 MiB buffer; skip it.
    // It is already gone from the queue, so report a timeout and go on.
    std::cerr << "pcie-genai: dropped an oversize notification (" << note.payload_len
              << " bytes)\n";
    return RecvStatus::Timeout;
  }
  if (rc != 0) {
    throw std::runtime_error("simaai_svc_recv failed: rc=" + std::to_string(rc));
  }
  out.tag = note.tag != nullptr ? note.tag : note.tag_buf;
  out.payload.assign(static_cast<const char*>(note.payload), note.payload_len);
  // Some tools send C strings together with their trailing NUL. Our
  // payloads never contain a NUL, so strip it; otherwise the JSON parse
  // or the printed text would get a stray '\0'.
  if (!out.payload.empty() && out.payload.back() == '\0') {
    out.payload.pop_back();
  }
  return RecvStatus::Ok;
}

} // namespace simaai::neat::pcie::genai::internal

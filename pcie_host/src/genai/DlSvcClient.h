/**
 * @file
 * @brief The real host SvcClient, backed by libsimaaipcie.so.
 *
 * SvcTransport -> DlSvcClient -> simaai_mla_daemon@<card> -> PCIe -> card.
 * The library is loaded at run time (dlopen), never linked, so the host
 * package builds and starts without the PCIe host library installed.
 */
#pragma once

#include "genai/SvcClient.h"

#include <cstdint>
#include <memory>

namespace simaai::neat::pcie::genai::internal {

/**
 * @brief The real SvcClient: one handle to the host simaai_mla_daemon for a card.
 *
 * libsimaaipcie.so is loaded with dlopen(RTLD_NOW | RTLD_LOCAL), never linked,
 * so hosts without the PCIe host package still start and get a clear error.
 * RTLD_LOCAL keeps its symbols private (the fix for a large-transfer
 * symbol clash on the card).
 */
class DlSvcClient final : public SvcClient {
public:
  explicit DlSvcClient(std::uint32_t card_id, const char* library = "libsimaaipcie.so");
  ~DlSvcClient() override;

  DlSvcClient(const DlSvcClient&) = delete;
  DlSvcClient& operator=(const DlSvcClient&) = delete;

  void subscribe(const std::string& tag) override;
  unsigned notify(const std::string& tag, const std::string& payload) override;
  RecvStatus recv(SvcNote& out, int timeout_ms) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat::pcie::genai::internal

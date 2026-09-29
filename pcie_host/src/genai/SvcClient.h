/**
 * @file
 * @brief Small interface for one simaai_svc client handle on the host.
 *
 * Chain: pcie-genai CLI -> GenAIModel -> SvcTransport -> SvcClient ->
 * simaai_mla_daemon -> PCIe -> pcie-genai-backend on the card.
 * SvcTransport uses only this interface, so its tests use a fake and need no
 * daemon and no card. The card has the same interface (svc_client.hpp).
 */
#pragma once

#include <string>

namespace simaai::neat::pcie::genai::internal {

/// One received notification: the tag it arrived on and its payload bytes.
struct SvcNote {
  std::string tag;
  std::string payload;
};

enum class RecvStatus {
  Ok,           ///< a note was received into the output
  Timeout,      ///< nothing arrived within the wait
  Disconnected, ///< the local simaai_svc daemon went away
};

/**
 * @brief Minimal view of one simaai_svc client handle.
 *
 * The real implementation (DlSvcClient) talks to the local daemon; tests use a
 * scripted fake. SvcTransport depends only on this interface, which is what
 * lets its logic be tested with no daemon and no card.
 */
class SvcClient {
public:
  virtual ~SvcClient() = default;

  /// Receive every notification sent on @p tag from now on. Throws on failure.
  virtual void subscribe(const std::string& tag) = 0;

  /// Send one notification. Returns how many far-side apps subscribe to the
  /// tag (meaningful on the host; the card side always reports 0). Throws on failure.
  virtual unsigned notify(const std::string& tag, const std::string& payload) = 0;

  /// Wait up to @p timeout_ms for the next notification on any subscribed tag.
  virtual RecvStatus recv(SvcNote& out, int timeout_ms) = 0;
};

} // namespace simaai::neat::pcie::genai::internal

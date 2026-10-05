#pragma once
#include "Protocol.h"
#include <chrono>

namespace simaai::neat::pcie::genai::wire {
// One outstanding asset per provider. Callbacks allow the exchange to be tested
// without opening a daemon or a PCIe queue.
template <class Send, class Receive, class Cancelled>
std::optional<uint64_t> request_asset(const std::string& session, uint64_t id,
                                      const std::string& name, int timeout_ms, Send send,
                                      Receive receive, Cancelled cancelled) {
  using Clock = std::chrono::steady_clock;
  auto request = envelope(session, id, "asset");
  request["name"] = relative_name(name);
  const auto text = request.dump();
  const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  auto sent = Clock::time_point{};
  while (Clock::now() < deadline) {
    if (cancelled())
      throw std::runtime_error("Model asset transfer cancelled");
    if (Clock::now() - sent >= std::chrono::seconds(1)) {
      send(text);
      sent = Clock::now();
    }
    const auto payload = receive(100);
    if (!payload)
      continue;
    const Json reply = parse(*payload, session);
    const uint64_t received = reply.at("request");
    if (received < id)
      continue;
    if (received != id || reply.at("kind") != "asset")
      throw std::runtime_error("Unexpected model asset reply");
    if (reply.contains("error"))
      throw std::runtime_error(reply.at("error").get<std::string>());
    if (!reply.at("found").get<bool>())
      return std::nullopt;
    return reply.at("bytes").get<uint64_t>();
  }
  throw std::runtime_error("Model asset transfer timed out: " + name);
}
} // namespace simaai::neat::pcie::genai::wire

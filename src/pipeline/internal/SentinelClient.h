/**
 * @file
 * @brief Client for the Sentinel board telemetry daemon (not part of the public API).
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <nlohmann/json.hpp>

#include <chrono>
#include <string>

namespace simaai::neat::pipeline_internal {

inline constexpr const char* kSentinelSocketPath = "/run/simaai-sentinel/api.sock";

/**
 * @brief Calls the Sentinel HTTP API (`/v1`) on its Unix socket.
 *
 * Sentinel samples board power, temperature, CPU, and memory every 2 s and keeps the latest
 * samples in a cache. Every method throws `std::runtime_error` naming the failed action and
 * Sentinel's reason; when the socket cannot be reached, the message names it and says how to
 * install and start Sentinel.
 */
class SentinelClient {
public:
  explicit SentinelClient(std::string socket_path = kSentinelSocketPath,
                          std::chrono::seconds timeout = std::chrono::seconds(5));

  /// Read the latest sample: its `timestamp` and metric `values`.
  nlohmann::json latest_sample() const;
  /// Read the cache: Sentinel's `version`, the metric definitions in `metrics`, and the latest
  /// `samples`, oldest first.
  nlohmann::json cache() const;

private:
  std::string socket_path_;
  std::chrono::seconds timeout_;
};

} // namespace simaai::neat::pipeline_internal

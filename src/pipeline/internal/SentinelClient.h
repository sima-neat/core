/**
 * @file
 * @brief Client for the Sentinel board telemetry daemon (not part of the public API).
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal {

inline constexpr const char* kSentinelSocketPath = "/run/simaai-sentinel/api.sock";

/**
 * @brief Calls the Sentinel HTTP API (`/v1`) on its Unix socket.
 *
 * Sentinel samples board power, temperature, CPU, and memory. A trace records those samples
 * between its start and stop and is saved as a run. Every method throws `std::runtime_error`
 * naming the failed action and Sentinel's reason.
 */
class SentinelClient {
public:
  explicit SentinelClient(std::string socket_path = kSentinelSocketPath);

  /// Start recording a trace; returns its ID. Fails while another trace is recording.
  std::string start_trace(const std::string& name, const std::string& note,
                          const std::vector<std::string>& tags) const;
  /// Stop the trace with this ID.
  void stop_trace(const std::string& trace_id) const;
  /// Read a saved run: its metadata, metric definitions, and samples.
  nlohmann::json read_run(const std::string& trace_id) const;

private:
  std::string socket_path_;
};

} // namespace simaai::neat::pipeline_internal

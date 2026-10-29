/**
 * @file
 * @ingroup diagnostics
 * @brief Board power telemetry read from Sentinel.
 *
 * @deprecated Use `TelemetryMonitor` (`pipeline/TelemetryMonitor.h`), which records power,
 * temperatures, CPU usage, and memory. This API keeps working unchanged.
 *
 * Power readings come from Sentinel, the board telemetry daemon, which samples the
 * PMBus rails every 2 s. Core does not access I2C. The rail tables, profiles, and
 * I2C fields below are deprecated and ignored.
 */
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

namespace internal {
struct PowerMonitorAccess;
} // namespace internal

/**
 * @brief Named board-level power monitor profiles.
 * @deprecated Ignored: Sentinel reports the rails of the running board.
 * @ingroup diagnostics
 */
enum class PowerMonitorProfile {
  Auto,       ///< Detect from the running board model when possible.
  ModalixSom, ///< Modalix SOM PMIC table from the board scripts.
  ModalixDvt, ///< Modalix DVT stable single-PMIC power path.
  Custom,     ///< Caller-provided explicit rail table.
};

/**
 * @brief One board power rail.
 *
 * Only `name` is set: Sentinel's rail name, such as `mla` for the
 * `power_rail_mla_watts` metric. The I2C fields are deprecated and ignored.
 * @ingroup diagnostics
 */
struct PowerRailConfig {
  std::string name;          ///< Rail name.
  int i2c_bus = 3;           ///< Deprecated and ignored.
  std::uint8_t i2c_addr = 0; ///< Deprecated and ignored.
  std::uint8_t page = 0;     ///< Deprecated and ignored.
  int vout_exponent = -8;    ///< Deprecated and ignored.
  int iout_exponent = -6;    ///< Deprecated and ignored.
  int pout_exponent = -5;    ///< Deprecated and ignored.
};

/**
 * @brief One scalar rail reading.
 * @ingroup diagnostics
 */
struct PowerFieldReading {
  bool available = false; ///< True when `value` was read successfully.
  std::uint8_t raw = 0;   ///< Deprecated: always 0.
  double value = 0.0;     ///< Scaled value in V, A, or W depending on field.
  std::string error;      ///< Non-empty on read failure.
};

/**
 * @brief Snapshot for one rail.
 *
 * Sentinel reports rail power only; voltage and current stay unavailable.
 * @ingroup diagnostics
 */
struct PowerRailReading {
  PowerRailConfig config;      ///< Rail this reading belongs to.
  PowerFieldReading voltage_v; ///< Deprecated: always unavailable.
  PowerFieldReading current_a; ///< Deprecated: always unavailable.
  PowerFieldReading power_w;   ///< Rail power in watts.
};

/**
 * @brief Point-in-time board power snapshot.
 * @ingroup diagnostics
 */
struct PowerSnapshot {
  std::chrono::steady_clock::time_point timestamp{}; ///< Capture time.
  std::vector<PowerRailReading> rails;               ///< Per-rail readings.
  double total_watts = 0.0;                          ///< Sum of available rail POUT values.
  std::uint64_t rails_with_power = 0;                ///< Rails contributing to `total_watts`.
};

/**
 * @brief Options for a `PowerMonitor`.
 * @deprecated Use `TelemetryMonitor`.
 * @ingroup diagnostics
 */
struct PowerMonitorOptions {
  bool enabled = false; ///< Read power from Sentinel while running.
  int sample_interval_ms =
      100; ///< Deprecated and ignored: Sentinel samples every 2 s and is polled every 1 s.
  PowerMonitorProfile profile = PowerMonitorProfile::Auto; ///< Deprecated and ignored.
  std::vector<PowerRailConfig> rails;                      ///< Deprecated and ignored.
};

/**
 * @brief Summary statistics for one scalar rail field.
 * @ingroup diagnostics
 */
struct PowerFieldSummary {
  std::uint64_t samples = 0; ///< Successful samples.
  std::uint64_t errors = 0;  ///< Samples without a reading.
  double avg = 0.0;          ///< Average successful value.
  double min = 0.0;          ///< Minimum successful value.
  double max = 0.0;          ///< Maximum successful value.
};

/**
 * @brief Summary statistics for one rail.
 * @ingroup diagnostics
 */
struct PowerRailSummary {
  PowerRailConfig config;      ///< Rail configuration.
  PowerFieldSummary voltage_v; ///< VOUT summary.
  PowerFieldSummary current_a; ///< IOUT summary.
  PowerFieldSummary power_w;   ///< POUT summary.
};

/**
 * @brief Aggregated power monitor result.
 * @deprecated Use `TelemetryMonitor`, which returns `Telemetry`.
 * @ingroup diagnostics
 */
struct PowerSummary {
  bool enabled = false;                ///< True when the monitor was configured to sample.
  std::uint64_t samples = 0;           ///< Sentinel samples taken while running.
  double duration_seconds = 0.0;       ///< Wall-clock monitor duration.
  double total_avg_watts = 0.0;        ///< Estimated average total SOM rail power.
  double total_min_watts = 0.0;        ///< Estimated minimum total SOM rail power.
  double total_max_watts = 0.0;        ///< Estimated maximum total SOM rail power.
  double energy_joules = 0.0;          ///< Estimated energy: avg watts * duration seconds.
  std::vector<PowerRailSummary> rails; ///< Per-rail summaries.
};

/**
 * @brief Default Modalix SOM PMIC rail table from the board measurement scripts.
 * @ingroup diagnostics
 */
[[deprecated("Ignored: Sentinel reports the rails of the running board")]]
std::vector<PowerRailConfig> default_modalix_som_power_rails();

/**
 * @brief Default Modalix DVT PMIC rail table.
 * @ingroup diagnostics
 */
[[deprecated("Ignored: Sentinel reports the rails of the running board")]]
std::vector<PowerRailConfig> default_modalix_dvt_power_rails();

/**
 * @brief Detect the most likely built-in board power profile for the current host.
 * @ingroup diagnostics
 */
[[deprecated("Ignored: Sentinel reports the rails of the running board")]]
PowerMonitorProfile detect_default_power_monitor_profile();

/**
 * @brief Human-readable name for a power monitor profile.
 * @ingroup diagnostics
 */
[[deprecated("Ignored: Sentinel reports the rails of the running board")]]
std::string power_monitor_profile_name(PowerMonitorProfile profile);

/**
 * @brief Resolve a built-in board profile to its rail table.
 * @ingroup diagnostics
 */
[[deprecated("Ignored: Sentinel reports the rails of the running board")]]
std::vector<PowerRailConfig> power_rails_for_profile(PowerMonitorProfile profile);

/**
 * @brief Options that enable board power monitoring.
 *
 * Both arguments are deprecated and ignored.
 * @deprecated Use `TelemetryMonitor`.
 * @ingroup diagnostics
 */
PowerMonitorOptions
board_power_monitor_options(int sample_interval_ms = 100,
                            PowerMonitorProfile profile = PowerMonitorProfile::Auto);

/**
 * @brief Convenience options enabling Modalix SOM power monitoring.
 * @ingroup diagnostics
 */
[[deprecated("Use board_power_monitor_options()")]]
PowerMonitorOptions modalix_som_power_monitor_options(int sample_interval_ms = 100);

/**
 * @brief Convenience options enabling Modalix DVT power monitoring.
 * @ingroup diagnostics
 */
[[deprecated("Use board_power_monitor_options()")]]
PowerMonitorOptions modalix_dvt_power_monitor_options(int sample_interval_ms = 100);

/**
 * @brief Read Sentinel's latest power sample.
 *
 * `options` is ignored. Throws `std::runtime_error` when Sentinel cannot be read.
 * @deprecated Use `TelemetryMonitor`.
 * @ingroup diagnostics
 */
[[deprecated("Use TelemetryMonitor")]]
PowerSnapshot read_power_snapshot(const PowerMonitorOptions& options);

/**
 * @brief Format a compact human-readable power summary.
 * @ingroup diagnostics
 */
std::string format_power_summary(const PowerSummary& summary);

/**
 * @brief Format power summary as a JSON object.
 * @ingroup diagnostics
 */
std::string power_summary_to_json(const PowerSummary& summary, int indent = 2);

/**
 * @brief Collects Sentinel power samples over a measured interval.
 *
 * Start immediately before a measured interval and stop immediately after it. Only
 * samples Sentinel takes in between are kept, so an interval shorter than about 2 s
 * may have none. Never throws: when Sentinel cannot be read, the monitor warns once
 * on stderr and collects no samples.
 * @deprecated Use `TelemetryMonitor`, which also records temperatures, CPU usage, and memory.
 * @ingroup diagnostics
 */
class [[deprecated("Use TelemetryMonitor")]] PowerMonitor {
public:
  explicit PowerMonitor(PowerMonitorOptions options = {});
  ~PowerMonitor();

  PowerMonitor(const PowerMonitor&) = delete;
  PowerMonitor& operator=(const PowerMonitor&) = delete;

  PowerMonitor(PowerMonitor&&) noexcept;
  PowerMonitor& operator=(PowerMonitor&&) noexcept;

  /** @brief Start background sampling if enabled. */
  void start();
  /** @brief Stop background sampling and keep the final summary available. */
  void stop();
  /** @brief Add Sentinel's latest sample now when it was not already added. */
  void sample_once();
  /** @brief Return the current aggregate summary. */
  PowerSummary summary() const;
  /** @brief True while the sampling thread is active. */
  bool running() const;

private:
  friend struct internal::PowerMonitorAccess;
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat

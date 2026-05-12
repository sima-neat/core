/**
 * @file
 * @ingroup diagnostics
 * @brief PMBus rail power telemetry.
 */
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Named board-level power monitor profiles.
 * @ingroup diagnostics
 */
enum class PowerMonitorProfile {
  Auto,       ///< Detect from the running board model when possible.
  ModalixSom, ///< Modalix SOM PMIC table from the board scripts.
  ModalixDvt, ///< Modalix DVT stable single-PMIC power path.
  Custom,     ///< Caller-provided explicit rail table.
};

/**
 * @brief Configuration for one PMBus rail exposed by a paged PMIC.
 *
 * The default Modalix SOM table is available via `default_modalix_som_power_rails()`.
 * Exponents intentionally mirror the board-team scripts: scaled value = raw byte *
 * 2^exponent.
 * @ingroup diagnostics
 */
struct PowerRailConfig {
  std::string name;          ///< Human-readable rail name.
  int i2c_bus = 3;           ///< Linux I2C bus number (`/dev/i2c-<bus>`).
  std::uint8_t i2c_addr = 0; ///< 7-bit PMIC I2C address.
  std::uint8_t page = 0;     ///< PMBus page selected through register 0x00.
  int vout_exponent = -8;    ///< Scaling exponent for VOUT (register 0x8b).
  int iout_exponent = -6;    ///< Scaling exponent for IOUT (register 0x8c).
  int pout_exponent = -5;    ///< Scaling exponent for POUT (register 0x96).
};

/**
 * @brief One scalar rail field read from a PMIC.
 * @ingroup diagnostics
 */
struct PowerFieldReading {
  bool available = false; ///< True when `value` was read successfully.
  std::uint8_t raw = 0;   ///< Raw byte returned by the PMIC register.
  double value = 0.0;     ///< Scaled value in V, A, or W depending on field.
  std::string error;      ///< Non-empty on read failure.
};

/**
 * @brief Snapshot for one rail.
 *
 * Current native PMBus sampling intentionally reads POUT only. Voltage/current
 * fields are retained for API compatibility and remain unavailable unless a
 * backend supplies them.
 * @ingroup diagnostics
 */
struct PowerRailReading {
  PowerRailConfig config;      ///< Rail configuration used for this read.
  PowerFieldReading voltage_v; ///< VOUT register value scaled to volts.
  PowerFieldReading current_a; ///< IOUT register value scaled to amps.
  PowerFieldReading power_w;   ///< POUT register value scaled to watts.
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
 * @ingroup diagnostics
 */
struct PowerMonitorOptions {
  bool enabled = false;         ///< Disabled by default to avoid I2C overhead.
  int sample_interval_ms = 100; ///< Periodic sampling interval while running.
  PowerMonitorProfile profile =
      PowerMonitorProfile::Auto; ///< Built-in board profile when `rails` is empty.
  std::vector<PowerRailConfig>
      rails; ///< Empty means use the selected built-in profile when enabled.
};

/**
 * @brief Summary statistics for one scalar rail field.
 * @ingroup diagnostics
 */
struct PowerFieldSummary {
  std::uint64_t samples = 0; ///< Successful samples.
  std::uint64_t errors = 0;  ///< Failed read attempts.
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
 * @ingroup diagnostics
 */
struct PowerSummary {
  bool enabled = false;                ///< True when the monitor was configured to sample.
  std::uint64_t samples = 0;           ///< Snapshot samples collected.
  double duration_seconds = 0.0;       ///< Wall-clock monitor duration.
  double total_avg_watts = 0.0;        ///< Average total SOM rail power.
  double total_min_watts = 0.0;        ///< Minimum total SOM rail power.
  double total_max_watts = 0.0;        ///< Maximum total SOM rail power.
  double energy_joules = 0.0;          ///< Estimated energy: avg watts * duration seconds.
  std::vector<PowerRailSummary> rails; ///< Per-rail summaries.
};

/**
 * @brief Default Modalix SOM PMIC rail table from the board measurement scripts.
 * @ingroup diagnostics
 */
std::vector<PowerRailConfig> default_modalix_som_power_rails();

/**
 * @brief Default Modalix DVT PMIC rail table.
 *
 * DVT Linux PMBus access is intentionally limited to the verified stable
 * reading: bus 4, PMIC 0x4d, page 0, POUT register 0x96, exponent -2.
 * @ingroup diagnostics
 */
std::vector<PowerRailConfig> default_modalix_dvt_power_rails();

/**
 * @brief Detect the most likely built-in board power profile for the current host.
 * @ingroup diagnostics
 */
PowerMonitorProfile detect_default_power_monitor_profile();

/**
 * @brief Human-readable name for a power monitor profile.
 * @ingroup diagnostics
 */
std::string power_monitor_profile_name(PowerMonitorProfile profile);

/**
 * @brief Resolve a built-in board profile to its rail table.
 * @ingroup diagnostics
 */
std::vector<PowerRailConfig> power_rails_for_profile(PowerMonitorProfile profile);

/**
 * @brief Convenience options enabling board power monitoring with optional auto-detect.
 * @ingroup diagnostics
 */
PowerMonitorOptions
board_power_monitor_options(int sample_interval_ms = 100,
                            PowerMonitorProfile profile = PowerMonitorProfile::Auto);

/**
 * @brief Convenience options enabling Modalix SOM power monitoring.
 * @ingroup diagnostics
 */
PowerMonitorOptions modalix_som_power_monitor_options(int sample_interval_ms = 100);

/**
 * @brief Convenience options enabling Modalix DVT power monitoring.
 * @ingroup diagnostics
 */
PowerMonitorOptions modalix_dvt_power_monitor_options(int sample_interval_ms = 100);

/**
 * @brief Read one immediate power snapshot using native Linux I2C access.
 * @ingroup diagnostics
 */
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
 * @brief Background SOM power sampler.
 *
 * Start immediately before a measured interval and stop immediately after it.
 * Read failures are counted per field and never throw; unavailable fields are
 * omitted from averages.
 * @ingroup diagnostics
 */
class PowerMonitor {
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
  /** @brief Collect one sample synchronously. Useful for tests and one-shot diagnostics. */
  void sample_once();
  /** @brief Return the current aggregate summary. */
  PowerSummary summary() const;
  /** @brief True while the sampling thread is active. */
  bool running() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat

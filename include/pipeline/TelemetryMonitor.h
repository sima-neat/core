/**
 * @file
 * @ingroup diagnostics
 * @brief Board telemetry from Sentinel over a window the caller chooses.
 *
 * Sentinel, the board telemetry daemon, samples power, temperatures, CPU usage, and memory
 * every 2 s. `TelemetryMonitor` reads those samples between `start()` and `stop()` and
 * summarizes each measurement as it arrives. `Model::benchmark()` uses the same monitor for its
 * throughput window.
 */
#pragma once

#include <filesystem>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

namespace internal {
struct TelemetryMonitorAccess;
} // namespace internal

/**
 * @brief One Sentinel measurement summarized over a telemetry window.
 *
 * `samples == 0` means the board did not report the measurement during the window, and the
 * statistics are then zero.
 * @ingroup diagnostics
 */
struct TelemetryStats {
  std::string key;   ///< Sentinel metric key, for example `power_current_watts`.
  std::string label; ///< Sentinel display label.
  std::string unit;  ///< Unit of the values.
  int samples = 0;   ///< Sentinel samples that carried a value.
  double mean = 0.0; ///< Mean over those samples.
  double minimum = 0.0;
  double maximum = 0.0;
  double last = 0.0; ///< Value in the newest of those samples.
};

/**
 * @brief Board telemetry Sentinel recorded during a window.
 *
 * Every value is board-wide and includes whatever else is running, including Sentinel and the
 * monitor themselves. Each power value is one 250 ms PMBus reading per Sentinel sample, so the
 * mean and maximum come from one reading in eight and can miss short spikes.
 * @ingroup diagnostics
 */
struct Telemetry {
  /// `"ok"` once a sample was counted; otherwise why there is no telemetry, or
  /// `"not started"` before the monitor first started.
  std::string status;
  std::string sentinel_version;   ///< Version of the Sentinel daemon that took the samples.
  int sample_interval_ms = 0;     ///< Time between Sentinel samples.
  int samples = 0;                ///< Sentinel samples counted in the window.
  double duration_ms = 0.0;       ///< Window length so far, or up to `stop()`.
  TelemetryStats power_som_watts; ///< Total SoM input power.
  TelemetryStats power_mla_watts; ///< MLA rail power; not reported on boards without that rail.
  std::vector<TelemetryStats> power_rails_watts;  ///< Every power rail the board reports.
  std::vector<TelemetryStats> die_temperatures_c; ///< Die sensors at the MLA, APU, CVU, and TOP.
  TelemetryStats die_temperature_max_c; ///< Hottest die sensor per sample, then summarized.
  std::vector<TelemetryStats> board_temperatures_c; ///< SoM top-side and bottom-side sensors.
  /// CPU utilization. Each sample averages the preceding interval, so `maximum` is the busiest
  /// interval rather than an instantaneous peak.
  TelemetryStats cpu_usage_pct;
  /// Utilization of each CPU core, in core order, summarized like `cpu_usage_pct`. A single busy
  /// core shows here even when the all-core average stays low.
  std::vector<TelemetryStats> cpu_cores_usage_pct;
  /// DMA buffers the MLA holds, in MiB: buffers from the MLA memory pool, plus shared buffers
  /// only the MLA uses. A shared buffer EV74 also uses counts in `ev74_memory_mib`, so the two
  /// add up. With nothing else running this is the model's MLA memory.
  TelemetryStats mla_memory_mib;
  /// DMA buffers EV74 holds, in MiB, excluding buffers from the MLA memory pool and frames a
  /// video decoder or camera ISP produced. With nothing else running this is the model's EV74
  /// memory, such as the input frames EV74 preprocesses and hands to the MLA.
  TelemetryStats ev74_memory_mib;
  /// Memory the whole board uses (Linux `MemTotal - MemAvailable`), in MiB. It covers every
  /// process and the buffers allocated from Linux contiguous memory (CMA), such as EV74's, so do
  /// not add it to the memory above.
  TelemetryStats linux_memory_used_mib;

  /**
   * @brief Write the telemetry as YAML or JSON to the exact destination filename.
   *
   * A `.json` extension selects JSON; `.yaml` and all other endings select YAML. Creates
   * missing parent directories and atomically replaces an existing file. Returns `file_path`
   * unchanged. Throws on invalid values or filesystem/write failures.
   */
  std::filesystem::path save(const std::filesystem::path& file_path) const;
};

/// Write `telemetry` as the YAML text `Telemetry::save()` writes to a `.yaml` file.
std::ostream& operator<<(std::ostream& out, const Telemetry& telemetry);

/**
 * @brief Records board telemetry from Sentinel between `start()` and `stop()`.
 *
 * The monitor counts each Sentinel sample once into a running summary, so memory stays the
 * same however long the window. It holds nothing in Sentinel: several monitors, in one process
 * or many, and `Model::benchmark()` can run at the same time, and a killed process leaves
 * nothing behind.
 *
 * Never throws for Sentinel problems. When Sentinel cannot be read, the monitor warns once on
 * stderr, `status` names the reason, and the monitor keeps retrying and counts samples once
 * Sentinel answers. All methods are thread-safe. The destructor stops a running monitor.
 *
 * @code
 * simaai::neat::TelemetryMonitor monitor;
 * monitor.start();
 * run_workload();
 * const simaai::neat::Telemetry telemetry = monitor.stop();
 * std::cout << telemetry;
 * telemetry.save("telemetry.yaml");
 * @endcode
 * @ingroup diagnostics
 */
class TelemetryMonitor {
public:
  /// Create a stopped monitor; contacts nothing until `start()`.
  TelemetryMonitor();
  /// Use an explicit Sentinel Unix socket instead of `/run/simaai-sentinel/api.sock`.
  /// Creates a stopped monitor; contacts nothing until `start()`.
  explicit TelemetryMonitor(std::string socket_path);
  ~TelemetryMonitor();

  TelemetryMonitor(const TelemetryMonitor&) = delete;
  TelemetryMonitor& operator=(const TelemetryMonitor&) = delete;

  TelemetryMonitor(TelemetryMonitor&&) noexcept;
  TelemetryMonitor& operator=(TelemetryMonitor&&) noexcept;

  /// Start a new window from zero. Samples Sentinel took before this call are not counted.
  /// Does nothing while running.
  void start();
  /// The summary so far, without stopping or contacting Sentinel. After `stop()`, its result.
  Telemetry read() const;
  /// Stop the window and return the final summary, including Sentinel's newest sample taken
  /// before this call. When already stopped, returns the previous result.
  Telemetry stop();
  /// True between `start()` and `stop()`.
  bool running() const;

private:
  friend struct internal::TelemetryMonitorAccess;
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat

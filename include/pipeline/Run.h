/**
 * @file
 * @ingroup pipeline
 * @brief Run — the live pipeline returned by Session::build, plus runtime options and diagnostics.
 *
 * `Run` is what a `Session` becomes when built. It owns the running GStreamer pipeline,
 * its internal threads (typically 5–15: one per element thread boundary, plus dispatcher
 * workers, plus a bus watcher), and its bounded queues. Application code interacts with a
 * Run by `push()`-ing inputs and `pull()`-ing outputs — or `run()` for a synchronous
 * shortcut. A Run can operate in Async mode (continuous pipeline, push/pull at user pace)
 * or Sync mode (one frame in, one result out, repeat).
 *
 * This header also defines:
 *   - `OverflowPolicy` — how `push()` behaves when the input queue is full.
 *   - `RunPreset` — preset bundles for common workloads (realtime, balanced, reliable).
 *   - `RunOptions` — runtime knobs (queue depth, overflow policy, output memory, metrics).
 *   - `RunStats` / `InputStreamStats` / `RunDiagSnapshot` — telemetry surfaces.
 *
 * @see Session::build for how a Run is constructed
 * @see SessionOptions for build-time options (Run takes runtime options here)
 * @see "Runs: the live pipeline (and the timing decision)" (§0.13 of the design deep dive)
 */
#pragma once

#include "nodes/io/Input.h"
#include "pipeline/PowerTelemetry.h"
#include "pipeline/RuntimeMetrics.h"
#include "pipeline/SessionOptions.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cv {
class Mat;
} // namespace cv

namespace simaai::neat {

class InputStream;
class LatencyProfiler;
struct InputStreamOptions;
namespace pipeline_internal {
struct InputRouteProcessor;
/// Shared pointer to a const internal route-processor (framework-internal use).
using InputRouteProcessorPtr = std::shared_ptr<const InputRouteProcessor>;
}

/**
 * @brief What `push()` does when the input queue is full.
 *
 * The right choice depends on the input source — file batches want lossless,
 * live cameras want freshness, network feeds with chokepoint pipelines want bounded memory.
 * @ingroup pipeline
 */
enum class OverflowPolicy {
  Block = 0,    ///< `push()` blocks until queue space frees up. Lossless. Use for batch processing.
  KeepLatest,   ///< Drop the oldest queued frame to make room. Use for live cameras (freshness > completeness).
  DropIncoming, ///< Drop the new frame; keep what's queued. Use when the pipeline is the chokepoint.
};

/**
 * @brief Convenience preset bundles for `RunOptions`.
 *
 * Each preset adjusts queue depth, overflow policy, and metrics flags to a profile that's
 * known to work well for one workload class.
 * @ingroup pipeline
 */
enum class RunPreset {
  Realtime, ///< Low-latency; small queues; KeepLatest overflow; metrics off.
  Balanced, ///< Default; moderate queues; Block overflow; metrics off.
  Reliable, ///< Lossless; deeper queues; Block overflow; metrics on.
};

/**
 * @brief How output `Tensor`s relate to the underlying GStreamer buffers.
 *
 * `Auto` lets the framework pick based on platform and pipeline shape. `ZeroCopy` shares
 * storage with GStreamer (faster but lifetime-coupled to the Run). `Owned` copies into a
 * framework-owned buffer (safer, slightly slower).
 * @ingroup pipeline
 */
enum class OutputMemory {
  Auto = 0,
  ZeroCopy,
  Owned,
};

/// Advanced runtime tuning knobs (most users never set these).
struct RunAdvancedOptions {
  bool copy_input = false;          ///< Force a copy of every pushed input (useful when the source buffer is short-lived).
  std::size_t max_input_bytes = 0;  ///< Reject pushes larger than this many bytes (0 = no cap).
  int sync_num_buffers_override = -1; ///< Override the appsrc `num-buffers` for sync-mode runs (-1 = auto).
  /**
   * @brief Prepare zero-copy Gst/SiMa outputs for CPU reads on the InputStream worker.
   *
   * When enabled for a zero-copy output Run, cached SiMa-backed tensors are made
   * CPU-visible before they are placed in the Run output queue. This does not
   * clone or copy tensor payloads; it only performs cache visibility maintenance
   * when producer metadata says a CPU reader could observe stale data.
   *
   * Keep this disabled for Graph-internal forwarding paths where an appsink is
   * used only as transport to another device-facing Session. Tensor::map(Read)
   * remains the correctness fallback for all outputs.
   */
  bool prepare_output_cpu_visible = false;
};

/**
 * @brief Per-Run runtime options. Passed to `Session::build()` and `Model::run()`.
 *
 * Most fields default to sensible values. The most commonly customized are `queue_depth`
 * (deeper for jittery sources, shallower for low latency) and `overflow_policy` (driven by
 * the input source's behavior — see the OverflowPolicy enum docs).
 * @ingroup pipeline
 */
struct RunOptions {
  RunPreset preset = RunPreset::Balanced;          ///< Convenience preset that tunes the other fields.
  int queue_depth = 4;                             ///< Capacity of the input/output buffer queues.
  OverflowPolicy overflow_policy = OverflowPolicy::Block; ///< What to do when the input queue is full.
  OutputMemory output_memory = OutputMemory::Auto; ///< Whether output tensors are zero-copy or owned.
  bool enable_metrics = false;                     ///< Collect detailed per-stage metrics in the Run's stats vectors.
  /**
   * @brief Default pull timeout for `build()`/`run()` input-mode paths, in milliseconds.
   *
   * `-1` keeps framework defaults (and legacy env-var fallback). Override per-call by
   * passing an explicit `timeout_ms` to the relevant `pull()` or `run()` overload.
   */
  int input_timeout_ms = -1;
  RunAdvancedOptions advanced{};                   ///< Advanced tuning (rarely needed).
  PowerMonitorOptions power_monitor{};             ///< Optional board rail power telemetry.

  /**
   * @brief Enable board power monitoring using built-in auto-detect.
   *
   * This is the preferred no-env path for capturing latency, throughput, and power
   * together from any Run.
   */
  RunOptions& enable_board_power(int sample_interval_ms = 100) {
    power_monitor = board_power_monitor_options(sample_interval_ms);
    return *this;
  }

  /**
   * @brief Enable default Modalix SOM PMIC rail power monitoring for this Run.
   *
   * Prefer `enable_board_power()` unless you need to force the SOM profile:
   *
   * @code
   * RunOptions opt;
   * opt.enable_board_power();
   * auto run = session.build(inputs, RunMode::Async, opt);
   * auto metrics = run.metrics();
   * @endcode
   */
  RunOptions& enable_modalix_som_power(int sample_interval_ms = 100) {
    power_monitor = modalix_som_power_monitor_options(sample_interval_ms);
    return *this;
  }

  /**
   * @brief Enable the built-in Modalix DVT PMIC table for this Run.
   */
  RunOptions& enable_modalix_dvt_power(int sample_interval_ms = 100) {
    power_monitor = modalix_dvt_power_monitor_options(sample_interval_ms);
    return *this;
  }

  /**
   * @brief Disable power monitoring after it was enabled/configured.
   */
  RunOptions& disable_power_monitor() {
    power_monitor = PowerMonitorOptions{};
    return *this;
  }

  /**
   * @brief Optional callback invoked when an input frame is dropped by the overflow policy.
   *
   * Useful for telemetry — log the drop reason, update a counter, etc. Callback runs on the
   * pushing thread; keep it short.
   */
  std::function<void(const struct InputDropInfo&)> on_input_drop;
};

/**
 * @brief Diagnostic record for a dropped input frame.
 *
 * Passed to the `on_input_drop` callback in `RunOptions`. Contains enough info to identify
 * which frame was dropped and why, for telemetry and post-mortem analysis.
 * @ingroup diagnostics
 */
struct InputDropInfo {
  SampleKind kind = SampleKind::Unknown; ///< What kind of sample was dropped (image, tensor, etc.).
  std::string media_type;                ///< MIME-style media type of the dropped sample.
  std::string format;                    ///< Format token (e.g., `"NV12"`, `"FP32"`).
  int width = -1;                        ///< Frame width in pixels (-1 if not applicable).
  int height = -1;                       ///< Frame height in pixels (-1 if not applicable).
  int depth = -1;                        ///< Channel depth (-1 if not applicable).
  int64_t frame_id = -1;                 ///< Source-assigned frame ID, if any.
  std::string stream_id;                 ///< Stream identifier (multi-stream pipelines).
  std::string port_name;                 ///< Ingress port name (multi-input models).
  std::string reason;                    ///< Human-readable reason (e.g., `"queue_full"`, `"size_limit_exceeded"`).
};

/**
 * @brief Per-Run input-side telemetry: counts, drops, and timing averages.
 *
 * Returned by `Run::input_stats()`. Useful for monitoring whether the application is
 * keeping up with the input source and where time is being spent on the push side.
 * @ingroup diagnostics
 */
struct InputStreamStats {
  std::uint64_t push_count = 0;          ///< Total successful `push()` calls.
  std::uint64_t push_failures = 0;       ///< Pushes that failed (queue full, EOS, error).
  std::uint64_t pull_count = 0;          ///< Total successful `pull()` calls (output side).
  std::uint64_t poll_count = 0;          ///< Total `pull()` calls (including timeouts).
  std::uint64_t dropped_frames = 0;      ///< Frames dropped due to OverflowPolicy.
  std::uint64_t renegotiations = 0;      ///< Times caps were re-negotiated mid-stream.
  std::uint64_t alloc_grows = 0;         ///< Times the buffer pool grew due to demand.
  std::uint64_t growth_blocked = 0;      ///< Pool-grow attempts that hit a configured cap.
  std::uint64_t renegotiation_blocked = 0; ///< Renegotiation attempts that were rejected by downstream.
  double avg_alloc_us = 0.0;             ///< Average buffer allocation time (microseconds).
  double avg_map_us = 0.0;               ///< Average buffer map (lock for write) time.
  double avg_copy_us = 0.0;              ///< Average input copy time (when `copy_input=true`).
  double avg_push_us = 0.0;              ///< Average end-to-end `push()` call time.
  double avg_pull_wait_us = 0.0;         ///< Average time `pull()` waited for output.
  double avg_decode_us = 0.0;            ///< Average decode time (input-side decoders, e.g., H.264).
};

/**
 * @brief Per-Run end-to-end statistics: counts and latency.
 *
 * Returned by `Run::stats()`. The headline metric is `avg_latency_ms` — wall-clock time
 * from `push()` to `pull()` for a frame.
 * @ingroup diagnostics
 */
struct RunStats {
  std::uint64_t inputs_enqueued = 0;     ///< Inputs that were accepted into the input queue.
  std::uint64_t inputs_dropped = 0;      ///< Inputs rejected by OverflowPolicy.
  std::uint64_t inputs_pushed = 0;       ///< Inputs successfully pushed into the pipeline.
  std::uint64_t outputs_ready = 0;       ///< Outputs the pipeline produced (may exceed pulls if backlogged).
  std::uint64_t outputs_pulled = 0;      ///< Outputs the application pulled.
  std::uint64_t outputs_dropped = 0;     ///< Outputs dropped before the application could pull (output-side overflow).
  double avg_latency_ms = 0.0;           ///< Average push-to-pull latency.
  double min_latency_ms = 0.0;           ///< Minimum observed latency.
  double max_latency_ms = 0.0;           ///< Maximum observed latency.
};

/**
 * @brief One-call runtime measurement summary.
 *
 * Combines the existing latency counters, derived throughput, and optional SOM
 * PMIC power telemetry. `throughput_fps` is computed from pulled outputs over
 * the Run lifetime so applications can get latency, throughput, and power
 * without env-var driven perf harnesses.
 * @ingroup diagnostics
 */
struct RunMeasurementSummary {
  RunStats stats;                  ///< End-to-end Run counters and latency.
  InputStreamStats input_stats;    ///< Input-side counters.
  double elapsed_seconds = 0.0;    ///< Measured wall-clock duration.
  double throughput_fps = 0.0;     ///< Pulled outputs per elapsed second.
  PowerSummary power;              ///< Optional power telemetry summary.
};

/**
 * @brief Per-stage timing telemetry — how long each stage takes per sample.
 * @ingroup diagnostics
 */
struct RunStageStats {
  std::string stage_name;       ///< Stage label (typically the NodeGroup or major Node name).
  std::uint64_t samples = 0;    ///< Number of samples processed by this stage.
  std::uint64_t total_us = 0;   ///< Cumulative time spent in this stage (microseconds).
  std::uint64_t max_us = 0;     ///< Maximum per-sample time observed.
};

/**
 * @brief Per-element timing — finer-grained than per-stage; one row per GStreamer element.
 * @ingroup diagnostics
 */
struct RunElementTimingStats {
  std::string element_name;     ///< Deterministic element name (e.g., `"n3_videoconvert"`).
  std::uint64_t samples = 0;    ///< Buffers processed.
  std::uint64_t total_us = 0;   ///< Cumulative residency (sink-arrival → src-emit; INCLUDES backpressure wait).
  std::uint64_t max_us = 0;     ///< Maximum per-buffer residency.
  std::uint64_t min_us = 0;     ///< Minimum per-buffer residency.
  std::uint64_t missed_in = 0;  ///< Buffers expected on the input pad but never arrived.
  std::uint64_t missed_out = 0; ///< Buffers expected on the output pad but never produced.
};

/**
 * @brief Per-element data-flow telemetry — buffer and byte counts, plus caps changes.
 * @ingroup diagnostics
 */
struct RunElementFlowStats {
  std::string element_name;       ///< Deterministic element name.
  std::uint64_t in_buffers = 0;   ///< Buffers received on input pads.
  std::uint64_t out_buffers = 0;  ///< Buffers produced on output pads.
  std::uint64_t in_bytes = 0;     ///< Bytes received.
  std::uint64_t out_bytes = 0;    ///< Bytes produced.
  std::uint64_t caps_changes = 0; ///< Mid-stream caps renegotiations on this element.
};

/**
 * @brief Per-pad timing — finest-grained telemetry, one row per (element, pad).
 *
 * Tracks inter-arrival jitter and queue-wait time per pad. Most useful for diagnosing
 * specific bottlenecks (e.g., which pad is slow to receive, which is slow to drain).
 * @ingroup diagnostics
 */
struct RunElementPadTimingStats {
  std::string element_name;               ///< Deterministic element name owning this pad.
  std::string pad_name;                   ///< Pad name within the element.
  bool        is_sink = false;            ///< True for input (sink) pads; false for output (src) pads.
  std::uint64_t samples = 0;              ///< Number of buffers seen on this pad.
  std::uint64_t inter_arrival_total_us = 0; ///< Cumulative time between consecutive buffer arrivals.
  std::uint64_t inter_arrival_max_us = 0; ///< Maximum observed inter-arrival gap, in microseconds.
  std::uint64_t queue_wait_samples = 0;     ///< Samples that had to wait in a queue before being processed.
  std::uint64_t queue_wait_total_us = 0;  ///< Cumulative queue-wait time across `queue_wait_samples`.
  std::uint64_t queue_wait_max_us = 0;    ///< Maximum observed per-sample queue-wait time.
  std::uint64_t bytes = 0;                ///< Cumulative byte count seen on this pad.
};

/**
 * @brief Aggregate diagnostic snapshot: stages, boundaries, per-element, per-pad.
 *
 * Returned by `Run::diag_snapshot()`. Holds vectors of the per-X telemetry structs so the
 * caller can iterate and render reports.
 * @ingroup diagnostics
 */
struct RunDiagSnapshot {
  std::vector<RunStageStats> stages;                       ///< Per-stage timing.
  std::vector<BoundaryFlowStats> boundaries;               ///< Per-boundary (between Nodes) flow stats.
  std::vector<RunElementTimingStats> element_timings;      ///< Per-element timing.
  std::vector<RunElementFlowStats> element_flows;          ///< Per-element flow.
  /**
   * @brief Per-pad timing rows (Phase-A diagnostics).
   *
   * Always appended after `element_flows` so older consumers reading the first four vectors
   * keep working unchanged.
   */
  std::vector<RunElementPadTimingStats> element_pad_timings;
};


/**
 * @brief Options for framework-owned runtime measurement.
 *
 * The defaults collect the measurements most applications need without asking examples to
 * reimplement timers, percentile math, profiler aggregation, or terminal formatting.
 */
struct MeasureOptions {
  int duration_ms = 10000;        ///< Timed measurement window.
  int warmup_ms = 1000;           ///< Warmup window excluded from latency/profiler results.
  int timeout_ms = 5000;          ///< Per-output pull timeout.
  /// Capture per-plugin/kernel latency through the NEAT profiler.  This is precise, but can
  /// be disabled for absolute maximum throughput sweeps.
  bool include_plugin_latency = true;
  bool include_power = true;      ///< Include power telemetry when enabled on the Run.

  /// Optional report metadata.  Model-owned wrappers/examples can fill these in so the
  /// standardized report is informative without custom formatting code.
  std::string title = "NEAT measurement";
  std::string model;
  std::string input;
  std::string placement;
  int logical_batch_size = 1;
};

/**
 * @brief Percentile summary for a measured latency series.
 */
struct MeasureLatencyStats {
  std::size_t count = 0;
  double avg_ms = 0.0;
  double p50_ms = 0.0;
  double p90_ms = 0.0;
  double p95_ms = 0.0;
  double p99_ms = 0.0;
  double max_ms = 0.0;
};

/**
 * @brief Aggregated per-plugin/kernel timing captured during a measurement window.
 */
struct MeasurePluginLatency {
  std::string name;
  std::uint64_t calls = 0;
  double avg_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
};

/**
 * @brief Framework-owned report returned by `MeasureScope::stop()`.
 */
struct MeasureReport {
  MeasureOptions options;
  std::size_t warmup_iterations = 0;
  std::size_t outputs = 0;
  double elapsed_s = 0.0;
  double throughput_batches_per_s = 0.0;
  double throughput_inferences_per_s = 0.0;

  MeasureLatencyStats end_to_end;
  MeasureLatencyStats frame_gap;
  bool latency_samples_collected = false;
  std::vector<MeasurePluginLatency> plugin_latency;

  std::uint64_t inputs_pushed = 0;
  std::uint64_t outputs_pulled = 0;
  std::uint64_t inputs_dropped = 0;
  std::uint64_t outputs_dropped = 0;
  RunStats final_run_stats{};
  PowerSummary power{};

  /// Render a compact customer-facing terminal report.
  std::string to_text() const;
};

/**
 * @brief Observation scope for measuring an application-owned push/pull interval.
 *
 * This does not own the inference loop and does not consume outputs. Start it before
 * normal application push/pull code, then call `stop()` to get throughput, app-visible
 * latency samples, counter deltas, profiler aggregation, and optional power telemetry.
 */
class MeasureScope {
 public:
  MeasureScope(MeasureScope&&) noexcept;
  MeasureScope& operator=(MeasureScope&&) noexcept;
  ~MeasureScope();

  MeasureScope(const MeasureScope&) = delete;
  MeasureScope& operator=(const MeasureScope&) = delete;

  MeasureReport stop();
  bool stopped() const;

 private:
  friend class Run;
  struct Impl;
  explicit MeasureScope(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Toggles for what `Run::report()` includes in its formatted text output.
 *
 * Each field corresponds to one section of the human-readable report. Defaults produce a
 * useful general-purpose report; toggle off to focus on a specific subsystem.
 * @ingroup diagnostics
 */
struct RunReportOptions {
  bool include_pipeline = true;       ///< Include the rendered pipeline launch string.
  bool include_stage_timings = true;  ///< Include the per-stage timing table.
  bool include_element_timings = true;///< Include the per-element timing table.
  bool include_boundaries = true;     ///< Include per-boundary flow stats between Nodes.
  bool include_flow_stats = true;     ///< Include per-element buffer/byte flow stats.
  bool include_node_reports = false;  ///< Off by default — verbose.
  bool include_next_cpu = false;      ///< Off by default — internal routing detail.
  bool include_queue_depth = true;    ///< Include configured queue depth in the report header.
  bool include_num_buffers = true;    ///< Include configured num-buffers in the report header.
  bool include_run_stats = true;      ///< Include end-to-end RunStats.
  bool include_input_stats = true;    ///< Include input-side stream stats.
  bool include_power = true;          ///< Include power telemetry when the Run enabled it.
  bool include_system_info = false;   ///< Off by default — system-wide info, not Run-specific.
};

/**
 * @brief Live pipeline handle: push inputs in, pull outputs out.
 *
 * A `Run` is what `Session::build()` returns. It owns the live GStreamer pipeline plus its
 * internal worker threads (typically 5–15 per Run, including streaming threads, dispatcher
 * pool threads, and a bus watcher). Application code drives the Run by `push()`-ing inputs
 * and `pull()`-ing outputs (or `run()` for one-shot synchronous use).
 *
 * @code
 *   auto run = sess.build(input);
 *   run.push(another_input);
 *   auto sample = run.pull(/ * timeout_ms = * / 100);
 * @endcode
 *
 * **Thread safety**: `push()` from one thread and `pull()` from another is safe. **Multiple
 * threads `push`-ing the same Run concurrently is NOT safe** — serialize push from one thread
 * or use external synchronization.
 *
 * Runs are **non-copyable** but **movable**. Destroying a Run shuts down its pipeline cleanly
 * (sends EOS, drains, transitions to NULL).
 *
 * @see Session::build for how a Run is constructed
 * @see RunOptions for runtime configuration
 * @see "Runs and the parallelism story" (§0.13 of the design deep dive)
 * @ingroup pipeline
 */
class Run {
public:
  /// Construct an empty Run; assign from `Session::build()` before use.
  Run() = default;
  Run(const Run&) = delete;            ///< Non-copyable.
  Run& operator=(const Run&) = delete; ///< Non-copyable.

  Run(Run&&) noexcept;                  ///< Move-constructible.
  Run& operator=(Run&&) noexcept;       ///< Move-assignable.
  ~Run();                               ///< Cleanly tears down the pipeline.

  /// Returns `true` if the Run is alive (constructed by Session::build, not yet stopped).
  explicit operator bool() const noexcept;
  /// Returns `true` if the input side accepts pushes (not closed).
  bool can_push() const;
  /// Returns `true` if the output side may produce more samples (pipeline not at EOS).
  bool can_pull() const;
  /// Returns `true` if the pipeline is in PLAYING state.
  bool running() const;

  /**
   * @brief Push `cv::Mat` inputs into the pipeline. Multi-input models accept one Mat per ingress.
   * @return `true` on success; behavior on full queue is governed by `RunOptions::overflow_policy`.
   */
  bool push(const std::vector<cv::Mat>& inputs);
  /// Non-blocking variant of `push`; returns `false` immediately if the queue is full.
  bool try_push(const std::vector<cv::Mat>& inputs);
  /// Push `Tensor` inputs (one per ingress port).
  bool push(const TensorList& inputs);
  /// Non-blocking variant.
  bool try_push(const TensorList& inputs);
  /// Push full `Sample` inputs (carrying per-buffer metadata).
  bool push(const SampleList& msgs);
  /// Non-blocking variant.
  bool try_push(const SampleList& msgs);
  /// **Internal**: pushes a `GstBuffer` held by a tensor ref to preserve plugin metadata.
  bool push_holder(const std::shared_ptr<void>& holder);
  /// Non-blocking variant of `push_holder`.
  bool try_push_holder(const std::shared_ptr<void>& holder);
  /// Send EOS into the pipeline. Drain remaining outputs by continuing to pull until `PullStatus::Closed`.
  void close_input();
  /**
   * @brief Pull the next output sample with a structured status return.
   * @param timeout_ms Wait up to this many ms; `-1` waits forever; `0` is non-blocking.
   * @param out  Filled with the next sample on `Ok`.
   * @param err  Optional out-parameter populated on `Error` with a structured `PullError`.
   * @return `Ok`, `Timeout`, `Closed`, or `Error`.
   */
  PullStatus pull(int timeout_ms, Sample& out, PullError* err = nullptr);
  /// Convenience pull returning an optional `Sample` (empty on timeout/closed/error).
  std::optional<Sample> pull(int timeout_ms = -1);
  /// Pull and unpack the next sample as a `TensorList`.
  TensorList pull_tensors(int timeout_ms = -1);
  /// Pull the next sample as a `SampleList` (preserves per-sample metadata).
  SampleList pull_samples(int timeout_ms = -1);
  /// One-shot synchronous push+pull from `cv::Mat` inputs.
  TensorList run(const std::vector<cv::Mat>& inputs, int timeout_ms = -1);
  /// One-shot synchronous push+pull from `Tensor` inputs.
  TensorList run(const TensorList& inputs, int timeout_ms = -1);
  /// One-shot synchronous push+pull from `Sample` inputs.
  SampleList run(const SampleList& inputs, int timeout_ms = -1);

  /// Start observing a caller-owned push/pull interval without consuming outputs.
  MeasureScope start_measurement(const MeasureOptions& opt = {});
  /**
   * @brief Run `warm` warm-up inferences before measurement begins.
   *
   * Useful for stable performance numbers — first inferences pay one-time setup costs
   * (kernel load, JIT, cache fill). Pass `warm = -1` for a sensible default.
   */
  int warmup(const std::vector<cv::Mat>& inputs, int warm = -1, int timeout_ms = -1);

  /// Returns end-to-end push/pull/latency stats.
  RunStats stats() const;
  /// Returns input-side telemetry (push counts, drops, timing).
  InputStreamStats input_stats() const;
  /// Returns the full diagnostic snapshot (per-stage, per-element, per-pad).
  RunDiagSnapshot diag_snapshot() const;
  /// Returns the optional SOM power telemetry summary for this Run.
  PowerSummary power_summary() const;
  /// Returns latency, throughput, input stats, and optional power telemetry in one call.
  RunMeasurementSummary measurement_summary() const;
  /// Returns the preferred unified runtime metrics surface.
  RuntimeMetrics metrics(const RuntimeMetricsOptions& opt = {}) const;
  /// Render unified runtime metrics in the requested format.
  std::string metrics_report(const RuntimeMetricsOptions& opt = {},
                             RuntimeMetricsFormat format = RuntimeMetricsFormat::Text) const;
  /// Convenience overload for selecting the output format with default options.
  std::string metrics_report(RuntimeMetricsFormat format) const;
  /// Returns a human-readable formatted report combining stats and diagnostics.
  std::string report(const RunReportOptions& opt = {}) const;
  /// Returns the most recent runtime error string (empty if no error occurred).
  std::string last_error() const;
  /// Returns a short diagnostics summary suitable for logging or error reports.
  std::string diagnostics_summary() const;

  /// Stop the pipeline immediately (transitions to NULL). After stop, the Run is no longer running.
  void stop();
  /// Alias for `stop()`. Releases resources.
  void close();

private:
  friend class MeasureScope;
  struct State;
  std::shared_ptr<State> state_;
  bool owns_ref_ = false;

  explicit Run(std::shared_ptr<State> state);
  void require_async_mode(const char* where) const;
  void require_async_pull_mode(const char* where) const;
  void enqueue_run_images(const std::vector<cv::Mat>& inputs);
  void enqueue_run_tensors(const TensorList& inputs);
  void enqueue_run_samples(const SampleList& inputs);
  TensorList pull_tensors_strict(int timeout_ms);
  SampleList pull_samples_strict(int timeout_ms);
  bool push_impl(const cv::Mat& input, bool block);
  bool push_impl(const simaai::neat::Tensor& input, bool block);
  bool push_holder_impl(const std::shared_ptr<void>& holder, bool block);
  bool push_message_impl(const Sample& msg, bool block);
  bool push_sample_impl(const Sample& msg, bool block);
  static Run create(InputStream stream, const RunOptions& opt,
                    const struct InputStreamOptions& stream_opt,
                    RunMode mode = RunMode::Async,
                    const std::optional<InputOptions>& tensor_input_opt_for_cv = std::nullopt,
                    pipeline_internal::InputRouteProcessorPtr input_route_processor = nullptr);
  friend class Session;
};

} // namespace simaai::neat

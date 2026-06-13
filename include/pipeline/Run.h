/**
 * @file
 * @ingroup pipeline
 * @brief Run — the live pipeline returned by Graph::build, plus runtime options and diagnostics.
 *
 * `Run` is what a `Graph` becomes when built. It owns the running GStreamer pipeline,
 * its internal threads (typically 5–15: one per element thread boundary, plus dispatcher
 * workers, plus a bus watcher), and its bounded queues. Application code interacts with a
 * Run by `push()`-ing inputs and `pull()`-ing outputs — or `run()` for a push+pull
 * convenience call on an existing runner. Use `Graph::run(...)` when you want one-shot
 * graph execution without owning a long-lived runner.
 *
 * This header also defines:
 *   - `OverflowPolicy` — how `push()` behaves when the input queue is full.
 *   - `RunPreset` — preset bundles for common workloads (realtime, balanced, reliable).
 *   - `RunOptions` — runtime knobs (queue depth, overflow policy, output memory).
 *   - `MeasureScope` / `MeasureReport` — the canonical measurement surface.
 *
 * @see Graph::build for how a Run is constructed
 * @see GraphOptions for build-time options (Run takes runtime options here)
 * @see "Runs: the live pipeline (and the timing decision)" (§0.13 of the design deep dive)
 */
#pragma once

#include "nodes/io/Input.h"
#include "pipeline/GraphMetrics.h"
#include "pipeline/PowerTelemetry.h"
#include "pipeline/GraphOptions.h"

#include <cstdint>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cv {
class Mat;
} // namespace cv

namespace simaai::neat {

class InputStream;
class MeasureScope;
struct MeasureOptions;
class Run;
struct InputStreamOptions;
namespace pipeline_internal {
struct InputRouteProcessor;
/// Shared pointer to a const internal route-processor (framework-internal use).
using InputRouteProcessorPtr = std::shared_ptr<const InputRouteProcessor>;
} // namespace pipeline_internal
namespace runtime {
class RunCore;
} // namespace runtime
#ifdef SIMA_NEAT_INTERNAL
namespace run_internal {
std::shared_ptr<runtime::RunCore> release_core(Run& run);
std::shared_ptr<const runtime::RunCore> core(const Run& run);
MeasureScope start_measurement_on_core(std::shared_ptr<runtime::RunCore> core,
                                       const MeasureOptions& opt);
} // namespace run_internal
#endif

/**
 * @brief What `push()` does when the input queue is full.
 *
 * The right choice depends on the input source — file batches want lossless,
 * live cameras want freshness, network feeds with chokepoint pipelines want bounded memory.
 * @ingroup pipeline
 */
enum class OverflowPolicy {
  Block = 0,    ///< `push()` blocks until queue space frees up. Lossless. Use for batch processing.
  KeepLatest,   ///< Drop the oldest queued frame to make room. Use for live cameras (freshness >
                ///< completeness).
  DropIncoming, ///< Drop the new frame; keep what's queued. Use when the pipeline is the
                ///< chokepoint.
};

/**
 * @brief Convenience preset bundles for `RunOptions`.
 *
 * Each preset adjusts queue depth, overflow policy, and output ownership to a profile that's
 * known to work well for one workload class. Use `start_measurement()` when you want
 * performance data for any preset.
 * @ingroup pipeline
 */
enum class RunPreset {
  Realtime, ///< Low-latency; small queues; KeepLatest overflow.
  Balanced, ///< Default; moderate queues; Block overflow.
  Reliable, ///< Lossless; deeper queues; Block overflow.
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
  bool copy_input =
      false; ///< Force a copy of every pushed input (useful when the source buffer is short-lived).
  std::size_t max_input_bytes = 0; ///< Reject pushes larger than this many bytes (0 = no cap).
  int sync_num_buffers_override =
      -1; ///< Override the appsrc `num-buffers` for sync-mode runs (-1 = auto).
  /**
   * @brief Prepare zero-copy Gst/SiMa outputs for CPU reads on the InputStream worker.
   *
   * When enabled for a zero-copy output Run, cached SiMa-backed tensors are made
   * CPU-visible before they are placed in the Run output queue. This does not
   * clone or copy tensor payloads; it only performs cache visibility maintenance
   * when producer metadata says a CPU reader could observe stale data.
   *
   * Keep this disabled for Graph-internal forwarding paths where an appsink is
   * used only as transport to another device-facing Graph. Tensor::map(Read)
   * remains the correctness fallback for all outputs.
   */
  bool prepare_output_cpu_visible = false;
};

/**
 * @brief Optional build-time Run JSON export.
 *
 * Set `path` before calling `Graph::build(...)` to write a topology snapshot as
 * soon as the Run is built. This is useful for CI artifacts and graph
 * visualization even if the application has not pushed samples yet.
 *
 * For final throughput/latency/power numbers, call `save_run_json(run, ...)`
 * after the run has executed/drained. Build-time export is an initial snapshot.
 */
struct RunAutoExportOptions {
  std::string path;  ///< Empty disables build-time export.
  std::string label; ///< Optional label; empty uses the exporter default.
  bool include_metrics = true;
  bool include_power = true;
  bool include_node_metrics = true;       ///< Include node-level latency rows.
  bool include_plugin_metrics = true;     ///< Include plugin/kernel latency rows when available.
  bool include_empty_node_metrics = true; ///< Keep attributed node rows even before samples arrive.
  int indent = 2;
};

/// Backward-compatible name for the previous graph-run export option spelling.
using GraphRunAutoExportOptions = RunAutoExportOptions;

/**
 * @brief Per-Run runtime options. Passed to `Graph::build()` and `Model::run()`.
 *
 * Most fields default to sensible values. The most commonly customized are `queue_depth`
 * (deeper for jittery sources, shallower for low latency) and `overflow_policy` (driven by
 * the input source's behavior — see the OverflowPolicy enum docs).
 * @ingroup pipeline
 */
struct RunOptions {
  RunPreset preset = RunPreset::Balanced; ///< Convenience preset that tunes the other fields.
  int queue_depth = 4;                    ///< Capacity of the input/output buffer queues.
  OverflowPolicy overflow_policy =
      OverflowPolicy::Block; ///< What to do when the input queue is full.
  OutputMemory output_memory =
      OutputMemory::Auto; ///< Whether output tensors are zero-copy or owned.
  /**
   * @brief Default pull timeout for `build()`/`run()` input-mode paths, in milliseconds.
   *
   * `-1` keeps framework defaults (and legacy env-var fallback). Override per-call by
   * passing an explicit `timeout_ms` to the relevant `pull()` or `run()` overload.
   */
  int input_timeout_ms = -1;
  /**
   * @brief Validate seeded `build(input, ...)` calls by pushing/pulling the seed once.
   *
   * This applies only to build overloads that receive an initial `cv::Mat`, `TensorList`, or
   * `Sample`. It catches payload-level failures (for example invalid encoded bitstreams or
   * caps that only the GStreamer element can reject) at build time instead of letting the first
   * failure surface asynchronously after a `Run` has already been returned. Seedless/source builds
   * ignore this option.
   *
   * Set to `false` for latency-sensitive callers that only want the seed for caps/shape
   * adaptation and are prepared to observe first-sample errors through `Run::last_error()` or
   * `pull()`.
   */
  bool startup_preflight = true;
  RunAdvancedOptions advanced{};       ///< Advanced tuning (rarely needed).
  PowerMonitorOptions power_monitor{}; ///< Optional board rail power telemetry.
  RunAutoExportOptions run_export{};   ///< Optional build-time Run/graph JSON export.
  /// Backward-compatible spelling for the previous graph-run export field.
  RunAutoExportOptions graph_run_export{};

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
   * auto run = graph.build(inputs, opt);
   * auto scope = run.start_measurement();
   * // push/pull workload
   * auto report = scope.stop();
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
  std::string reason; ///< Human-readable reason (e.g., `"queue_full"`, `"size_limit_exceeded"`).
};

/**
 * @brief Metrics trace backend selection for framework-owned measurements.
 *
 * `Auto` prefers the low-overhead LTTng collector.  If LTTng is unavailable the report marks
 * the requested metric class as unavailable; it does not silently fall back to a second metrics
 * backend.
 */
enum class MetricsTraceSource {
  Auto,
  Off,
  Lttng,
};

/**
 * @brief Options for framework-owned runtime measurement.
 *
 * Default construction is the production-safe/customer path: public E2E latency and throughput
 * only.  It does not start plugin tracing, message tracing, graph queue probes, or measurement-
 * local power monitoring.  Set `include_plugin_latency=true` only when you intentionally want
 * deeper per-plugin/kernel profiling.
 */
struct MeasureOptions {
  int duration_ms = 10000; ///< Timed measurement window.
  int warmup_ms = 1000;    ///< Warmup window excluded from latency/metrics results.
  int timeout_ms = 5000;   ///< Per-output pull timeout.
  /// Capture per-plugin/kernel execution latency through the LTTng metrics collector.
  bool include_plugin_latency = false;
  /// Backend used for plugin execution latency.
  MetricsTraceSource plugin_latency_source = MetricsTraceSource::Lttng;
  /// Include low-overhead inter-plugin/edge/queue diagnostics in the report.
  bool include_edge_latency = false;
  /// Enable exact per-message LTTng edge tracing.  Higher volume; off by default.
  bool include_message_latency = false;
  /// Backend used for exact per-message edge tracing.
  MetricsTraceSource message_latency_source = MetricsTraceSource::Lttng;
  /// Keep the private LTTng CTF trace directory after parsing for support/debug.
  bool retain_metrics_trace = false;
  /// Optional parent/output directory for retained or temporary metrics traces.
  std::string metrics_trace_dir;
  bool include_power = false; ///< Include power telemetry when enabled on the Run.

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
  std::string backend;
  std::string phase;
  std::string kernel_name;
  std::string stage_name;
  std::int32_t physical_input_index = -1;
  std::int32_t output_slot = -1;
  std::uint64_t run_id_hash = 0;            ///< Stable run-id hash, 0 if trace payload omitted it.
  std::int32_t pipeline_segment_id = -1;    ///< Graph pipeline segment, -1 if unavailable.
  std::int32_t runtime_node_id = -1;        ///< Lowered runtime node id, -1 if unavailable.
  std::int32_t public_node_id = -1;         ///< Optional public graph node id, -1 if unavailable.
  std::vector<std::string> public_node_ids; ///< Public graph node ids, e.g. "p2".
  std::string gst_element_name;             ///< Owning GStreamer element name, if available.
  std::string stream_id;                    ///< Stream identity when provided by LTTng.
  std::string plugin_instance_id;           ///< Stable plugin instance id when provided by LTTng.
  std::string source;                       ///< lttng or diagnostics.
  std::string attribution_source;           ///< lttng_v2_identity, lttng_element_name, etc.
  std::string mapping_error;                ///< Non-empty when the row could not be attributed.
  bool reliable = true;                     ///< False when trace loss/unmatched pairs affect row.
  std::uint64_t calls = 0;
  double total_ms = 0.0;
  double avg_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
};

/**
 * @brief Aggregated handoff/queue/transport timing between plugins or runtime nodes.
 *
 * Edge/message latency is diagnostic and non-additive.  It must not be summed into plugin
 * execution latency or graph throughput/power.
 */
struct MeasureEdgeLatency {
  std::string edge_id;
  std::string name;
  std::string from_node_id;
  std::string to_node_id;
  std::int32_t from_runtime_node_id = -1;
  std::int32_t to_runtime_node_id = -1;
  std::string from_element_name;
  std::string to_element_name;
  std::string from_plugin_instance_id;
  std::string to_plugin_instance_id;
  std::string stream_id;

  std::uint64_t samples = 0;
  double total_ms = 0.0;
  double avg_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;

  std::string source;             ///< diagnostics or lttng.
  std::string timing_semantics;   ///< queue_residence, edge_transport, pad_wait, etc.
  std::string attribution_source; ///< graph_edge_identity, element_link, unattributed.
  std::string mapping_error;
  bool non_additive = true;
  bool reliable = true;
};

struct MeasurePathStat {
  std::uint64_t samples = 0;
  double avg_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  double max_ms = 0.0;
  bool reliable = true;
};

struct MeasurePathIdentity {
  std::string primary_key;
  std::string fallback_key;
  std::vector<std::string> used_public_fields;
  std::string sample_identity_source;
};

struct MeasurePathNodeArrival {
  std::string customer_node_id;
  std::string lowered_node_id;
  std::int32_t runtime_node_id = -1;
  std::string plugin_instance_id;
  std::string stream_id;
  std::string semantics = "graph_entry_to_first_node_observation";
  MeasurePathStat latency;
};

struct MeasurePathInterPluginGap {
  std::string customer_edge_id;
  std::string lowered_edge_id;
  std::string from_customer_node_id;
  std::string to_customer_node_id;
  std::int32_t from_runtime_node_id = -1;
  std::int32_t to_runtime_node_id = -1;
  std::string from_plugin_instance_id;
  std::string to_plugin_instance_id;
  std::string stream_id;
  std::string semantics = "upstream_plugin_end_to_downstream_plugin_start";
  MeasurePathStat latency;
};

struct MeasurePathOutputTail {
  std::string output_endpoint;
  std::string customer_output_node_id;
  std::string lowered_edge_id;
  std::string stream_id;
  std::string semantics = "last_observed_work_to_public_pull";
  MeasurePathStat latency;
};

struct MeasurePathTiming {
  bool available = false;
  std::string status;
  std::string source;
  std::string reason;
  std::string aggregation = "measured_window";
  std::vector<std::string> warnings;
  MeasurePathIdentity identity;
  std::vector<MeasurePathNodeArrival> node_arrival;
  std::vector<MeasurePathInterPluginGap> inter_plugin_gap;
  std::vector<MeasurePathOutputTail> output_tail;
};

/**
 * @brief Common input/output/drop counters captured during a measurement window.
 *
 * This is the public counter shape. Raw runtime counters remain implementation details.
 */
struct MeasureCounters {
  std::uint64_t inputs_enqueued = 0;
  std::uint64_t inputs_pushed = 0;
  std::uint64_t outputs_ready = 0;
  std::uint64_t outputs_pulled = 0;
  std::uint64_t inputs_dropped = 0;
  std::uint64_t outputs_dropped = 0;
};

/**
 * @brief Input-boundary telemetry captured during a measurement window.
 */
struct MeasureInputStats {
  std::uint64_t push_count = 0;
  std::uint64_t push_failures = 0;
  std::uint64_t pull_count = 0;
  std::uint64_t poll_count = 0;
  std::uint64_t dropped_frames = 0;
  std::uint64_t renegotiations = 0;
  std::uint64_t alloc_grows = 0;
  std::uint64_t growth_blocked = 0;
  std::uint64_t renegotiation_blocked = 0;

  double avg_alloc_us = 0.0;
  double avg_map_us = 0.0;
  double avg_copy_us = 0.0;
  double avg_push_us = 0.0;
  double avg_pull_wait_us = 0.0;
  double avg_decode_us = 0.0;
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

  /// Queue-inclusive graph-entry to public-pull timing. In a single-flight loop this approximates
  /// user-visible latency; in async burst / queued workloads it includes queue wait and should be
  /// presented as queue residency, not standalone model latency.
  MeasureLatencyStats end_to_end;
  MeasureLatencyStats frame_gap;
  bool latency_samples_collected = false;
  std::string end_to_end_semantics = "queue_inclusive_graph_entry_to_public_pull";
  std::string end_to_end_interpretation =
      "Single-flight: approximates per-input latency. Async burst/queued: includes queue wait and "
      "should be presented as queue residency, not standalone latency.";
  MeasureCounters counters;
  MeasureInputStats input;
  std::vector<MeasurePluginLatency> plugin_latency;
  std::vector<MeasurePluginLatency> plugin_latency_unattributed;
  std::vector<MeasureEdgeLatency> edge_latency;
  std::vector<MeasureEdgeLatency> edge_latency_unattributed;
  std::vector<GraphNodeMetrics> node_metrics;
  MeasurePathTiming path_timing;

  std::string plugin_latency_status;  ///< off, collected, unavailable, or failed.
  std::string plugin_latency_source;  ///< lttng or none.
  std::string message_latency_status; ///< off, collected, unavailable, or failed.
  std::string message_latency_source; ///< diagnostics, lttng, or none.
  std::string metrics_trace_dir;      ///< Retained CTF trace dir, otherwise empty.
  std::vector<std::string> warnings;
  bool trace_loss_detected = false;

  std::uint64_t graph_sample_timing_unkeyed = 0;
  std::uint64_t graph_sample_timing_misses = 0;
  PowerSummary power{};

  /// Render a compact customer-facing terminal report.
  std::string to_text() const;
  /// Render the same measured-window report as JSON.
  std::string to_json(int indent = 2) const;
};

/**
 * @brief Observation scope for measuring an application-owned push/pull interval.
 *
 * This does not own the inference loop and does not consume outputs. Start it before
 * normal application push/pull code, then call `stop()` to get throughput, app-visible
 * latency samples, counter deltas, plugin/edge aggregation, and optional power telemetry.
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
#ifdef SIMA_NEAT_INTERNAL
  friend MeasureScope run_internal::start_measurement_on_core(std::shared_ptr<runtime::RunCore>,
                                                             const MeasureOptions&);
#endif
  struct Impl;
  explicit MeasureScope(std::unique_ptr<Impl> impl);
  static void disable_lttng_trace_identity_noexcept(Impl* impl);
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Live pipeline handle: push inputs in, pull outputs out.
 *
 * A `Run` is what `Graph::build()` returns. It owns the live GStreamer pipeline plus its
 * internal worker threads (typically 5–15 per Run, including streaming threads, dispatcher
 * pool threads, and a bus watcher). Application code drives the Run by `push()`-ing inputs
 * and `pull()`-ing outputs (or `run()` for one-shot synchronous use).
 *
 * @code
 *   auto run = graph.build(TensorList{input});
 *   run.push(TensorList{another_input});
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
 * @see Graph::build for how a Run is constructed
 * @see RunOptions for runtime configuration
 * @see "Runs and the parallelism story" (§0.13 of the design deep dive)
 * @ingroup pipeline
 */
class Run {
public:
  /// Construct an empty Run; assign from `Graph::build()` before use.
  Run() = default;
  Run(const Run&) = delete;            ///< Non-copyable.
  Run& operator=(const Run&) = delete; ///< Non-copyable.

  Run(Run&&) noexcept;            ///< Move-constructible.
  Run& operator=(Run&&) noexcept; ///< Move-assignable.
  ~Run();                         ///< Cleanly tears down the pipeline.

  /// Returns `true` if the Run is alive (constructed by Graph::build, not yet stopped).
  explicit operator bool() const noexcept;
  /// Returns `true` if the input side accepts pushes (not closed).
  bool can_push() const;
  /// Returns `true` if the output side may produce more samples (pipeline not at EOS).
  bool can_pull() const;
  /// Returns `true` if the pipeline is in PLAYING state.
  bool running() const;
  /// Names accepted by `push(name, ...)` for graph-backed Runs. Empty for unnamed/linear Runs.
  std::vector<std::string> input_names() const;
  /// Names accepted by `pull(name, ...)` for graph-backed Runs. Empty for unnamed/linear Runs.
  std::vector<std::string> output_names() const;

  /**
   * @brief Push `cv::Mat` inputs into the pipeline. Multi-input models accept one Mat per ingress.
   * @return `true` on success; behavior on full queue is governed by `RunOptions::overflow_policy`.
   */
  bool push(const std::vector<cv::Mat>& inputs);
  /// Push `cv::Mat` inputs into a named graph ingress. Use for multi-input graphs.
  bool push(std::string_view input_name, const std::vector<cv::Mat>& inputs);
  /// Non-blocking variant of `push`; returns `false` immediately if the queue is full.
  bool try_push(const std::vector<cv::Mat>& inputs);
  /// Non-blocking named-ingress variant.
  bool try_push(std::string_view input_name, const std::vector<cv::Mat>& inputs);
  /// Push `Tensor` inputs (one per ingress port).
  bool push(const TensorList& inputs);
  /// Push `Tensor` inputs into a named graph ingress. Use for multi-input graphs.
  bool push(std::string_view input_name, const TensorList& inputs);
  /// Non-blocking variant.
  bool try_push(const TensorList& inputs);
  /// Non-blocking named-ingress variant.
  bool try_push(std::string_view input_name, const TensorList& inputs);
  /// Push full `Sample` inputs (carrying per-buffer metadata).
  bool push(const Sample& msgs);
  /// Push full `Sample` inputs into a named graph ingress. Use for multi-input graphs.
  bool push(std::string_view input_name, const Sample& msgs);
  /// Non-blocking variant.
  bool try_push(const Sample& msgs);
  /// Non-blocking named-ingress variant.
  bool try_push(std::string_view input_name, const Sample& msgs);
  /// **Internal**: pushes a `GstBuffer` held by a tensor ref to preserve plugin metadata.
  bool push_holder(const std::shared_ptr<void>& holder);
  /// Non-blocking variant of `push_holder`.
  bool try_push_holder(const std::shared_ptr<void>& holder);
  /// Send EOS into the pipeline. Drain remaining outputs by continuing to pull until
  /// `PullStatus::Closed`.
  void close_input();
  /**
   * @brief Pull the next output sample with a structured status return.
   * @param timeout_ms Wait up to this many ms; `-1` waits forever; `0` is non-blocking.
   * @param out  Filled with the next sample on `Ok`.
   * @param err  Optional out-parameter populated on `Error` with a structured `PullError`.
   * @return `Ok`, `Timeout`, `Closed`, or `Error`.
   */
  PullStatus pull(int timeout_ms, Sample& out, PullError* err = nullptr);
  /**
   * @brief Pull from a named graph output with structured status.
   *
   * Use this when a graph has multiple terminal outputs. `pull()` without a name remains the
   * default-output convenience API for the common single-output case.
   */
  PullStatus pull(std::string_view output_name, int timeout_ms, Sample& out,
                  PullError* err = nullptr);
  /// Convenience pull returning an optional `Sample` (empty on timeout/closed; throws on error).
  std::optional<Sample> pull(int timeout_ms = -1);
  /// Convenience named-output pull (empty on timeout/closed; throws on error).
  std::optional<Sample> pull(std::string_view output_name, int timeout_ms = -1);
  /// Pull and unpack the next sample as a `TensorList`.
  TensorList pull_tensors(int timeout_ms = -1);
  /// Pull and unpack from a named graph output as a `TensorList`.
  TensorList pull_tensors(std::string_view output_name, int timeout_ms = -1);
  /// Pull the next sample as a `Sample` (preserves per-sample metadata).
  Sample pull_samples(int timeout_ms = -1);
  /// Pull samples from a named graph output (preserves per-sample metadata).
  Sample pull_samples(std::string_view output_name, int timeout_ms = -1);
  /// One-shot synchronous push+pull from `cv::Mat` inputs.
  TensorList run(const std::vector<cv::Mat>& inputs, int timeout_ms = -1);
  /// One-shot synchronous push+pull from `Tensor` inputs.
  TensorList run(const TensorList& inputs, int timeout_ms = -1);
  /// One-shot synchronous push+pull from `Sample` inputs.
  Sample run(const Sample& inputs, int timeout_ms = -1);

  /// Start observing a caller-owned push/pull interval without consuming outputs.
  MeasureScope start_measurement(const MeasureOptions& opt = {});
  /// Convenience overload: false = E2E latency/throughput only, true = include plugin latency.
  MeasureScope start_measurement(bool include_plugin_latency);
  /// Returns the most recent runtime error string (empty if no error occurred).
  std::string last_error() const;

  /// Stop the pipeline immediately (transitions to NULL). After stop, the Run is no longer running.
  void stop();
  /// Alias for `stop()`. Releases resources.
  void close();

private:
  friend class MeasureScope;
#ifdef SIMA_NEAT_INTERNAL
  friend std::shared_ptr<runtime::RunCore> run_internal::release_core(Run& run);
  friend std::shared_ptr<const runtime::RunCore> run_internal::core(const Run& run);
#endif
  std::shared_ptr<runtime::RunCore> core_;

  explicit Run(std::shared_ptr<runtime::RunCore> core);
  void require_async_mode(const char* where) const;
  void require_async_pull_mode(const char* where) const;
  void enqueue_run_images(const std::vector<cv::Mat>& inputs);
  void enqueue_run_tensors(const TensorList& inputs);
  void enqueue_run_samples(const Sample& inputs);
  TensorList pull_tensors_strict(int timeout_ms);
  Sample pull_samples_strict(int timeout_ms);
  bool push_impl(const cv::Mat& input, bool block);
  bool push_impl(const simaai::neat::Tensor& input, bool block);
  bool push_holder_impl(const std::shared_ptr<void>& holder, bool block);
  bool push_message_impl(const Sample& msg, bool block);
  bool push_sample_impl(const Sample& msg, bool block);
  static Run create(InputStream stream, const RunOptions& opt,
                    const struct InputStreamOptions& stream_opt, RunMode mode = RunMode::Async,
                    const std::optional<InputOptions>& tensor_input_opt_for_cv = std::nullopt,
                    pipeline_internal::InputRouteProcessorPtr input_route_processor = nullptr);
  friend class Graph;
};

} // namespace simaai::neat

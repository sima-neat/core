/**
 * @file
 * @ingroup pipeline
 * @brief Per-kernel latency profiler — V1 surface.
 *
 * Attach a `LatencyProfiler` to a `simaai::neat::Run` (or to a Graph that has
 * produced one) BEFORE you start pushing frames. After your run loop, call
 * `finalize()` to get a `ProfilerReport` and pass it to `to_text()` /
 * `to_chrome_trace()` to dump a human-readable summary or a JSON trace file
 * loadable in chrome://tracing or Perfetto.
 *
 * The profiler aggregates four classes of telemetry:
 *   1. Per-kernel-invocation events (MLA, A65, EV74, BoxDecode, Memcpy) drained
 *      from libsimaaineatprofiler.so's cross-shared-library ring. Each event
 *      carries (start_ns, end_ns, backend, phase, physical_input_index,
 *      output_slot, frame_id, request_id, kernel_name, stage_name, in/out
 *      segment names, bytes).
 *   2. Per-element aggregate timings (existing `Run::diag_snapshot()`).
 *   3. End-to-end per-frame stats (existing `Run::stats()`).
 *   4. Per-site memcpy totals (calls / total_ns / total_bytes) for the five hot
 *      copy sites the runtime instruments.
 *
 * Off-path overhead is gated by `sima_neat_profiler_enabled()` — when no
 * profiler is attached, every emit site is one atomic-load + branch.
 *
 * @see Run.h for `RunStats`, `InputStreamStats`, `RunDiagSnapshot`.
 */

// Per-kernel latency profiler library — V1 surface.
//
// Attach a `LatencyProfiler` to a `simaai::neat::Run` (or to a Graph that
// has produced one) BEFORE you start pushing frames.  After your run loop,
// call `finalize()` to get a `Report` and pass it to `to_text()` /
// `to_chrome_trace()` to dump a human-readable summary or a JSON trace file
// loadable in chrome://tracing or Perfetto.
//
// The profiler aggregates four classes of telemetry:
//   1. Per-kernel-invocation events (MLA, A65, EV74, BoxDecode, Memcpy) —
//      drained from libsimaaineatprofiler.so's cross-shared-library ring.
//      Each event carries (start_ns, end_ns, backend, phase,
//      physical_input_index, output_slot, frame_id, request_id, kernel_name,
//      stage_name, in/out segment names, bytes).
//   2. Per-element aggregate timings (existing `Run::diag_snapshot()`).
//   3. End-to-end per-frame stats (existing `Run::stats()`).
//   4. Per-site memcpy totals (calls / total_ns / total_bytes) for the five
//      hot copy sites the runtime instruments.
//
// Off-path overhead is gated by `sima_neat_profiler_enabled()` — when no
// profiler is attached, every emit site is one atomic-load + branch.

#ifndef SIMAAI_NEAT_PIPELINE_LATENCY_PROFILER_H_
#define SIMAAI_NEAT_PIPELINE_LATENCY_PROFILER_H_

#include "pipeline/Run.h"

#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat {

class Graph; // forward

/**
 * @brief One kernel-invocation telemetry event.
 *
 * Captures the wall-clock start/end timestamps of a single kernel run plus the
 * routing context (which backend, which physical input, which output slot,
 * which logical frame) so events from different backends can be cross-correlated
 * into a unified timeline.
 *
 * @ingroup pipeline
 */
struct ProfilerKernelInvocation {
  std::uint64_t start_ns = 0;             ///< Event start timestamp (CLOCK_MONOTONIC, ns).
  std::uint64_t end_ns = 0;               ///< Event end timestamp (CLOCK_MONOTONIC, ns).
  std::string backend;                    ///< "MLA", "A65", "EV74", "BoxDecode", "Memcpy".
  std::string phase;                      ///< "Run", "Load", "GetHandle", ...
  std::int32_t physical_input_index = -1; ///< Physical input index, -1 if N/A.
  std::int32_t output_slot = -1;          ///< Output slot, -1 if N/A.
  std::int64_t frame_id = -1;             ///< Logical frame id, -1 if not tagged.
  std::uint32_t request_id = 0;           ///< Backend request id (0 if unset).
  std::uint32_t bytes = 0;                ///< Bytes moved/processed (0 if N/A).
  std::string kernel_name;                ///< Kernel name (backend-specific).
  std::string stage_name;                 ///< Pipeline stage name.
  std::string in_segment;                 ///< Input segment label.
  std::string out_segment;                ///< Output segment label.

  /// @brief Convenience: event duration in milliseconds.
  double duration_ms() const {
    return static_cast<double>(end_ns - start_ns) / 1.0e6;
  }
};

/**
 * @brief Aggregate counters for one instrumented memcpy site.
 *
 * The runtime instruments a small fixed set of hot copy sites (e.g.,
 * `MEMCPY_NV12_Y`); each `ProfilerMemcpySite` rolls up call count, total
 * nanoseconds spent, total bytes moved, and the worst-case single-call latency
 * for one of those sites.
 *
 * @ingroup pipeline
 */
struct ProfilerMemcpySite {
  std::string site_name;         ///< Site label, e.g. "MEMCPY_NV12_Y".
  std::uint64_t calls = 0;       ///< Number of times this site fired.
  std::uint64_t total_ns = 0;    ///< Total wall-clock time across all calls (ns).
  std::uint64_t total_bytes = 0; ///< Total bytes copied across all calls.
  std::uint64_t max_ns = 0;      ///< Worst-case single-call latency (ns).

  /// @brief Total time spent at this site, in milliseconds.
  double total_ms() const {
    return static_cast<double>(total_ns) / 1.0e6;
  }
  /// @brief Average time per call at this site, in milliseconds.
  double avg_ms() const {
    return calls > 0 ? (static_cast<double>(total_ns) / 1.0e6) / static_cast<double>(calls) : 0.0;
  }
};

/**
 * @brief Aggregated timings for one (backend, kernel, stage, slot) tuple.
 *
 * Bucketed view over `ProfilerKernelInvocation` records: call count plus
 * total/min/max latency in milliseconds. Use `avg_ms()` for the mean.
 *
 * @ingroup pipeline
 */
struct ProfilerKernelAggregate {
  std::string backend;                    ///< Backend label ("MLA", "A65", ...).
  std::string kernel_name;                ///< Kernel name within the backend.
  std::string stage_name;                 ///< Pipeline stage name.
  std::int32_t physical_input_index = -1; ///< Physical input index, -1 if N/A.
  std::int32_t output_slot = -1;          ///< Output slot, -1 if N/A.
  std::uint64_t count = 0;                ///< Number of invocations in the bucket.
  double total_ms = 0.0;                  ///< Total time across invocations (ms).
  double min_ms = 0.0;                    ///< Minimum single-invocation time (ms).
  double max_ms = 0.0;                    ///< Maximum single-invocation time (ms).
  /// @brief Mean latency per invocation, in milliseconds.
  double avg_ms() const {
    return count > 0 ? (total_ms / static_cast<double>(count)) : 0.0;
  }
};

/**
 * @brief Snapshot bundle returned by `LatencyProfiler::finalize()`.
 *
 * Combines the existing per-frame and per-element telemetry surfaces with the
 * new per-invocation kernel events and memcpy-site totals. Optional fields
 * (`model_path`, `description`, `frames_total`, `warmup_frames`) let the caller
 * stamp the report with run-identifying metadata before serialising.
 *
 * @ingroup pipeline
 * @see LatencyProfiler::to_text
 * @see LatencyProfiler::to_chrome_trace
 */
struct ProfilerReport {
  // Reused snapshots
  RunStats end_to_end{};           ///< End-to-end per-frame stats (Run::stats()).
  InputStreamStats input_stream{}; ///< Input stream backpressure stats.
  RunDiagSnapshot diag{};          ///< Per-element aggregate timings.

  // New
  std::vector<ProfilerKernelInvocation> kernel_invocations; ///< Per-event timeline.
  std::vector<ProfilerKernelAggregate> kernel_aggregates;   ///< Bucketed event totals.
  std::vector<ProfilerMemcpySite> memcpy_sites;             ///< Per-site memcpy counters.

  std::uint64_t profiler_emits = 0;   ///< Number of events the runtime emitted.
  std::uint64_t profiler_dropped = 0; ///< Number of events dropped (ring full).

  std::string model_path;        ///< Optional: caller may set to identify the loaded model archive.
  std::string description;       ///< Optional: caller-supplied label for the run.
  std::int64_t frames_total = 0; ///< Optional: total frames pushed during the run.
  std::int64_t warmup_frames = 0; ///< Optional: warmup frames excluded from measurements.
};

/**
 * @brief Construction options for `LatencyProfiler`.
 *
 * @ingroup pipeline
 */
struct LatencyProfilerOptions {
  bool capture_kernels = true;      ///< Capture per-kernel-invocation events.
  bool capture_memcpy = true;       ///< Capture per-site memcpy totals.
  std::size_t ring_capacity = 8192; ///< Event ring capacity (oldest dropped on overflow).
  std::int64_t warmup_frames = 0;   ///< Frames to skip before starting measurement.
};

/**
 * @brief Per-sample latency tracker; attach to a `Run` to capture timing telemetry.
 *
 * Lifetime: construct, `attach()` to a `Run` (or `Graph`) before pushing
 * frames, push frames, optionally call `mark_warmup_done()` after the warmup
 * window, then call `finalize()` to obtain a `ProfilerReport`. The profiler
 * is single-owner (non-copyable) and detaches automatically on destruction.
 *
 * @ingroup pipeline
 * @see ProfilerReport
 * @see Run
 */
class LatencyProfiler {
public:
  /// Tunable knobs for the profiler.
  using Options = LatencyProfilerOptions;

  /// @brief Construct a profiler with the given options.
  explicit LatencyProfiler(Options o = Options());
  /// @brief Destructor; detaches from any attached Run/Graph.
  ~LatencyProfiler();

  /// Deleted copy constructor; the profiler owns thread-bound state.
  LatencyProfiler(const LatencyProfiler&) = delete;
  /// Deleted copy assignment; the profiler owns thread-bound state.
  LatencyProfiler& operator=(const LatencyProfiler&) = delete;

  /**
   * @brief Attach to a `Run` (Graph-level path).
   *
   * After this call, every kernel event emitted by the runtime is captured in
   * the profiler's ring. Must be called before frames are pushed for events
   * to be observed.
   */
  // Attach to a Run (Graph-level path).  After this call, every kernel
  // event emitted by the runtime is captured in the profiler's ring.
  void attach(Run& run);

  /**
   * @brief Optional: attach a Graph directly.
   *
   * Reserved for V2: would let per-output `frame_id` stamping hook the existing
   * tensor_callback. No-op for V1.
   */
  // Optional: attach a Graph directly so per-output frame_id stamping can
  // hook the existing tensor_callback.  No-op for V1 (placeholder for V2).
  void attach(Graph& graph);

  /**
   * @brief Reset all counters/event ring to mark the warmup→measured boundary.
   *
   * Call after pushing `Options::warmup_frames` inputs. Subsequent telemetry
   * starts fresh from this point.
   */
  // Reset all counters/event ring to mark the boundary between warmup and
  // measured frames.  Call after pushing `Options::warmup_frames` inputs.
  void mark_warmup_done();

  /**
   * @brief Drain the event ring and snapshot all telemetry sources.
   *
   * Safe to call multiple times; each call drains incremental events since the
   * previous drain (use `mark_warmup_done()` to discard rather than emit).
   *
   * @return Snapshot bundle ready for serialisation via `to_text()` or
   *         `to_chrome_trace()`.
   */
  // Drain the event ring and snapshot every reused telemetry source into a
  // Report.  Safe to call multiple times; each call drains incremental
  // events since the previous drain (use mark_warmup_done() to discard).
  ProfilerReport finalize();

  // Convenience helpers for serialization.
  /// @brief Render @p report as a human-readable text summary.
  static std::string to_text(const ProfilerReport& report);
  /// @brief Render @p report as a Chrome/Perfetto trace JSON document.
  static std::string to_chrome_trace(const ProfilerReport& report);

private:
  Options options_;
  Run* attached_run_ = nullptr;
  Graph* attached_graph_ = nullptr;
  bool enabled_at_attach_ = false;
};

} // namespace simaai::neat

#endif // SIMAAI_NEAT_PIPELINE_LATENCY_PROFILER_H_

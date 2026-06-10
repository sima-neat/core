#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/GraphReport.h"
#include "pipeline/internal/InputStreamStats.h"

#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat {

struct RunStats {
  std::uint64_t inputs_enqueued = 0; ///< Inputs that were accepted into the input queue.
  std::uint64_t inputs_dropped = 0;  ///< Inputs rejected by OverflowPolicy.
  std::uint64_t inputs_pushed = 0;   ///< Inputs successfully pushed into the pipeline.
  std::uint64_t outputs_ready =
      0; ///< Outputs the pipeline produced (may exceed pulls if backlogged).
  std::uint64_t outputs_pulled = 0; ///< Outputs the application pulled.
  std::uint64_t outputs_dropped =
      0; ///< Outputs dropped before the application could pull (output-side overflow).
  double avg_latency_ms = 0.0; ///< Average push-to-pull latency.
  double min_latency_ms = 0.0; ///< Minimum observed latency.
  double max_latency_ms = 0.0; ///< Maximum observed latency.
};

/**
 * @brief Per-stage timing telemetry — how long each stage takes per sample.
 * @ingroup diagnostics
 */
struct RunStageStats {
  std::string stage_name;     ///< Stage label (typically the Graph fragment or major Node name).
  std::uint64_t samples = 0;  ///< Number of samples processed by this stage.
  std::uint64_t total_us = 0; ///< Cumulative time spent in this stage (microseconds).
  std::uint64_t max_us = 0;   ///< Maximum per-sample time observed.
};

/**
 * @brief Per-element timing — finer-grained than per-stage; one row per GStreamer element.
 * @ingroup diagnostics
 */
struct RunElementTimingStats {
  std::string element_name;  ///< Deterministic element name (e.g., `"n3_videoconvert"`).
  std::uint64_t samples = 0; ///< Buffers processed.
  std::uint64_t total_us =
      0; ///< Cumulative residency (sink-arrival → src-emit; INCLUDES backpressure wait).
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
  std::string element_name; ///< Deterministic element name owning this pad.
  std::string pad_name;     ///< Pad name within the element.
  bool is_sink = false;     ///< True for input (sink) pads; false for output (src) pads.
  std::string transport_from_element_name; ///< Upstream element that stamped src departure.
  std::string transport_to_element_name;   ///< Sink element that observed transport arrival.
  std::uint64_t samples = 0;               ///< Number of buffers seen on this pad.
  std::uint64_t inter_arrival_total_us =
      0;                                  ///< Cumulative time between consecutive buffer arrivals.
  std::uint64_t inter_arrival_max_us = 0; ///< Maximum observed inter-arrival gap, in microseconds.
  std::uint64_t queue_wait_samples =
      0; ///< Samples that had to wait in a queue before being processed.
  std::uint64_t queue_wait_total_us =
      0;                               ///< Cumulative queue-wait time across `queue_wait_samples`.
  std::uint64_t queue_wait_max_us = 0; ///< Maximum observed per-sample queue-wait time.
  std::uint64_t bytes = 0;             ///< Cumulative byte count seen on this pad.
};

/**
 * @brief Aggregate diagnostic snapshot: stages, boundaries, per-element, per-pad.
 *
 * Internal diagnostic snapshot. Measurement reports expose the customer-facing diagnostics.
 * @ingroup diagnostics
 */
struct RunDiagSnapshot {
  std::vector<RunStageStats> stages;                  ///< Per-stage timing.
  std::vector<BoundaryFlowStats> boundaries;          ///< Per-boundary (between Nodes) flow stats.
  std::vector<RunElementTimingStats> element_timings; ///< Per-element timing.
  std::vector<RunElementFlowStats> element_flows;     ///< Per-element flow.
  /**
   * @brief Per-pad timing rows (Phase-A diagnostics).
   *
   * Always appended after `element_flows` so older consumers reading the first four vectors
   * keep working unchanged.
   */
  std::vector<RunElementPadTimingStats> element_pad_timings;
};

} // namespace simaai::neat

#include "pipeline/Run.h"
#include "RunInternal.h"

#include "pipeline/internal/Diagnostics.h"

#include <mutex>
#include <utility>

namespace simaai::neat {

RunStats runtime::RunCore::stats() const {
  RunStats out;
  out.inputs_enqueued = inputs_enqueued.load();
  out.inputs_dropped = inputs_dropped.load();
  out.inputs_pushed = inputs_pushed.load();
  out.outputs_ready = outputs_ready.load();
  out.outputs_pulled = outputs_pulled.load();
  out.outputs_dropped = outputs_dropped.load();
  {
    std::lock_guard<std::mutex> lock(latency_mu);
    if (latency_count > 0) {
      out.avg_latency_ms = latency_mean_ms;
      out.min_latency_ms = latency_min_ms;
      out.max_latency_ms = latency_max_ms;
    }
  }
  return out;
}

InputStreamStats runtime::RunCore::input_stats() const {
  return pipeline.stream.stats();
}

RunStats run_internal::stats(const Run& run) {
  const auto st = run_internal::core(run);
  return st ? st->stats() : RunStats{};
}

InputStreamStats run_internal::input_stats(const Run& run) {
  const auto st = run_internal::core(run);
  return st ? st->input_stats() : InputStreamStats{};
}

PowerSummary run_internal::power_summary(const Run& run) {
  const auto st = run_internal::core(run);
  if (!st || !st->power_monitor)
    return {};
  return st->power_monitor->summary();
}

RunDiagSnapshot runtime::RunCore::diag_snapshot() const {
  RunDiagSnapshot out;
  const auto diag = pipeline.stream.diag_ctx();
  if (!diag)
    return out;

  out.stages.reserve(diag->stage_timings.size());
  for (const auto& stage : diag->stage_timings) {
    if (!stage)
      continue;
    const auto snap = stage->snapshot();
    RunStageStats s;
    s.stage_name = snap.stage_name;
    s.samples = snap.samples;
    s.total_us = snap.total_us;
    s.max_us = snap.max_us;
    out.stages.push_back(std::move(s));
  }

  out.boundaries.reserve(diag->boundaries.size());
  for (const auto& boundary : diag->boundaries) {
    if (!boundary)
      continue;
    out.boundaries.push_back(boundary->snapshot());
  }

  out.element_timings.reserve(diag->element_timings.size());
  for (const auto& timing : diag->element_timings) {
    if (!timing)
      continue;
    const auto snap = timing->snapshot();
    RunElementTimingStats s;
    s.element_name = snap.element_name;
    s.samples = snap.samples;
    s.total_us = snap.total_us;
    s.max_us = snap.max_us;
    s.min_us = snap.min_us;
    s.missed_in = snap.missed_in;
    s.missed_out = snap.missed_out;
    out.element_timings.push_back(std::move(s));
  }

  out.element_flows.reserve(diag->element_flows.size());
  for (const auto& flow : diag->element_flows) {
    if (!flow)
      continue;
    const auto snap = flow->snapshot();
    RunElementFlowStats s;
    s.element_name = snap.element_name;
    s.in_buffers = snap.in_buffers;
    s.out_buffers = snap.out_buffers;
    s.in_bytes = snap.in_bytes;
    s.out_bytes = snap.out_bytes;
    s.caps_changes = snap.caps_changes;
    out.element_flows.push_back(std::move(s));
  }

  // Phase A: per-pad timings.  Hold the diag mutex briefly because the pad
  // timings vector is appended to lazily on first probe fire (so a buffer
  // landing on a never-before-seen pad mid-stream must not race with us).
  {
    std::lock_guard<std::mutex> lk(diag->element_pad_timings_mu);
    out.element_pad_timings.reserve(diag->element_pad_timings.size());
    for (const auto& pad_t : diag->element_pad_timings) {
      if (!pad_t)
        continue;
      const auto snap = pad_t->snapshot();
      RunElementPadTimingStats s;
      s.element_name = snap.element_name;
      s.pad_name = snap.pad_name;
      s.is_sink = snap.is_sink;
      s.transport_from_element_name = snap.transport_from_element_name;
      s.transport_to_element_name = snap.transport_to_element_name;
      s.samples = snap.samples;
      s.inter_arrival_total_us = snap.inter_arrival_total_us;
      s.inter_arrival_max_us = snap.inter_arrival_max_us;
      s.queue_wait_samples = snap.queue_wait_samples;
      s.queue_wait_total_us = snap.queue_wait_total_us;
      s.queue_wait_max_us = snap.queue_wait_max_us;
      s.bytes = snap.bytes;
      out.element_pad_timings.push_back(std::move(s));
    }
  }

  return out;
}

RunDiagSnapshot run_internal::diag_snapshot(const Run& run) {
  const auto st = run_internal::core(run);
  return st ? st->diag_snapshot() : RunDiagSnapshot{};
}

} // namespace simaai::neat

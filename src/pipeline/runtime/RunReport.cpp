#include "pipeline/Run.h"
#include "RunInternal.h"

#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"

#include <chrono>
#include <mutex>
#include <sstream>
#include <string>
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

RunStats Run::stats() const {
  return core_ ? core_->stats() : RunStats{};
}

InputStreamStats Run::input_stats() const {
  return core_ ? core_->input_stats() : InputStreamStats{};
}

PowerSummary Run::power_summary() const {
  if (!core_ || !core_->power_monitor)
    return {};
  return core_->power_monitor->summary();
}

RuntimeMetrics Run::metrics(const RuntimeMetricsOptions& opt) const {
  RuntimeMetrics out;
  out.source_kind = "run";
  if (!core_)
    return out;
  auto st = core_;

  const RunStats run_stats = stats();
  out.counters.inputs_enqueued = run_stats.inputs_enqueued;
  out.counters.inputs_dropped = run_stats.inputs_dropped;
  out.counters.inputs_pushed = run_stats.inputs_pushed;
  out.counters.outputs_ready = run_stats.outputs_ready;
  out.counters.outputs_pulled = run_stats.outputs_pulled;
  out.counters.outputs_dropped = run_stats.outputs_dropped;
  out.latency.avg_ms = run_stats.avg_latency_ms;
  out.latency.min_ms = run_stats.min_latency_ms;
  out.latency.max_ms = run_stats.max_latency_ms;
  if (opt.include_power) {
    out.power = power_summary();
  }

  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  {
    std::lock_guard<std::mutex> lock(st->latency_mu);
    start = st->created_at;
    if (st->pull_timing_init) {
      end = st->last_pull_at;
    } else if (st->output_timing_init) {
      end = st->last_output_at;
    } else {
      end = std::chrono::steady_clock::now();
    }
  }
  if (start != std::chrono::steady_clock::time_point{} && end > start) {
    out.elapsed_seconds = std::chrono::duration<double>(end - start).count();
    if (out.elapsed_seconds > 0.0) {
      out.throughput_fps = static_cast<double>(out.counters.outputs_pulled) / out.elapsed_seconds;
    }
  }

  const InputStreamStats is = input_stats();
  RuntimeMetricGroup input_group;
  input_group.name = "input_stream";
  input_group.values = {
      {"push_count", static_cast<double>(is.push_count), "count"},
      {"push_failures", static_cast<double>(is.push_failures), "count"},
      {"pull_count", static_cast<double>(is.pull_count), "count"},
      {"poll_count", static_cast<double>(is.poll_count), "count"},
      {"dropped_frames", static_cast<double>(is.dropped_frames), "count"},
      {"avg_push_us", is.avg_push_us, "us"},
      {"avg_pull_wait_us", is.avg_pull_wait_us, "us"},
  };
  out.groups.push_back(std::move(input_group));

  const auto diag = st->pipeline.stream.diag_ctx();
  if (opt.include_pipeline && diag && !diag->pipeline_string.empty()) {
    out.metadata.emplace_back("pipeline", diag->pipeline_string);
  }
  if (opt.include_diagnostics) {
    const RunDiagSnapshot snap = diag_snapshot();
    for (const auto& stage : snap.stages) {
      RuntimeMetricGroup group;
      group.name = "stage:" + stage.stage_name;
      group.values = {
          {"samples", static_cast<double>(stage.samples), "count"},
          {"total_us", static_cast<double>(stage.total_us), "us"},
          {"max_us", static_cast<double>(stage.max_us), "us"},
      };
      if (stage.samples > 0) {
        group.values.push_back(
            {"avg_us", static_cast<double>(stage.total_us) / static_cast<double>(stage.samples),
             "us"});
      }
      out.groups.push_back(std::move(group));
    }
    for (const auto& elem : snap.element_timings) {
      RuntimeMetricGroup group;
      group.name = "element:" + elem.element_name;
      group.values = {
          {"samples", static_cast<double>(elem.samples), "count"},
          {"total_us", static_cast<double>(elem.total_us), "us"},
          {"min_us", static_cast<double>(elem.min_us), "us"},
          {"max_us", static_cast<double>(elem.max_us), "us"},
      };
      if (elem.samples > 0) {
        group.values.push_back(
            {"avg_us", static_cast<double>(elem.total_us) / static_cast<double>(elem.samples),
             "us"});
      }
      out.groups.push_back(std::move(group));
    }
  }
  return out;
}

std::string Run::metrics_report(const RuntimeMetricsOptions& opt,
                                RuntimeMetricsFormat format) const {
  return format_runtime_metrics(metrics(opt), format);
}

std::string Run::metrics_report(RuntimeMetricsFormat format) const {
  return metrics_report(RuntimeMetricsOptions{}, format);
}

RunMeasurementSummary Run::measurement_summary() const {
  RunMeasurementSummary out;
  if (!core_)
    return out;
  out.stats = stats();
  out.input_stats = input_stats();
  const RuntimeMetrics unified = metrics();
  out.power = unified.power;
  out.elapsed_seconds = unified.elapsed_seconds;
  out.throughput_fps = unified.throughput_fps;
  return out;
}

RunDiagSnapshot Run::diag_snapshot() const {
  RunDiagSnapshot out;
  if (!core_)
    return out;
  const auto diag = core_->pipeline.stream.diag_ctx();
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

std::string Run::report(const RunReportOptions& opt) const {
  if (!core_)
    return {};
  auto st = core_;
  const auto diag = st->pipeline.stream.diag_ctx();
  std::ostringstream oss;
  oss << "[REPORT] Run\n";

  if (opt.include_pipeline && diag && !diag->pipeline_string.empty()) {
    oss << "Pipeline:\n" << diag->pipeline_string << "\n";
  }

  if (diag) {
    if (opt.include_boundaries) {
      const std::string boundary = pipeline_internal::boundary_summary(diag);
      if (!boundary.empty())
        oss << boundary;
    }
    if (opt.include_stage_timings) {
      const std::string stages = pipeline_internal::stage_timing_summary(diag);
      if (!stages.empty())
        oss << stages;
    }
    if (opt.include_element_timings) {
      const std::string elements = pipeline_internal::element_timing_summary(diag);
      if (!elements.empty())
        oss << elements;
    }
    if (opt.include_flow_stats) {
      const std::string flow = pipeline_internal::element_flow_summary(diag);
      if (!flow.empty())
        oss << flow;
    }
    if (opt.include_node_reports) {
      std::ostringstream mf;
      bool first = true;
      for (const auto& nr : diag->node_reports) {
        if (nr.kind != "ModelFragment" && nr.kind != "Preproc")
          continue;
        if (!first)
          mf << "; ";
        first = false;
        mf << nr.user_label;
        if (!nr.elements.empty()) {
          mf << " [";
          for (size_t i = 0; i < nr.elements.size(); ++i) {
            if (i)
              mf << ",";
            mf << nr.elements[i];
          }
          mf << "]";
        }
      }
      const std::string mf_str = mf.str();
      if (!mf_str.empty()) {
        oss << "Model fragments: " << mf_str << "\n";
      }
    }
    if (opt.include_next_cpu && !diag->next_cpu_decisions.empty()) {
      oss << "Next CPU decisions:\n";
      for (const auto& d : diag->next_cpu_decisions) {
        oss << "  - node=" << d.node_index << " kind=" << d.node_kind;
        if (!d.node_label.empty())
          oss << " label=" << d.node_label;
        oss << " next_cpu=" << d.next_cpu << " applied=" << (d.applied ? "1" : "0") << "\n";
      }
    }
    if (opt.include_queue_depth || opt.include_num_buffers) {
      const std::string pipeline = diag->pipeline_string;
      if (opt.include_queue_depth) {
        const int queue2_depth =
            diag->queue2_enabled ? diag->queue2_depth : run_internal::parse_queue2_depth(pipeline);
        if (!diag->queue2_enabled) {
          if (queue2_depth > 0) {
            oss << "queue2 depth=" << queue2_depth << " (manual)\n";
          } else {
            oss << "queue2 disabled\n";
          }
        } else if (queue2_depth > 0) {
          oss << "queue2 depth=" << queue2_depth << "\n";
        }
      }
      if (opt.include_num_buffers) {
        int num_cvu = run_internal::parse_num_buffers_for(pipeline, "neatprocesscvu");
        int num_mla = run_internal::parse_num_buffers_for(pipeline, "neatprocessmla");
        if (num_cvu > 0 || num_mla > 0) {
          oss << "num_buffers_cvu=" << num_cvu << " num_buffers_mla=" << num_mla << "\n";
        }
      }
    }
  }

  if (opt.include_run_stats) {
    const RunMeasurementSummary measurement = measurement_summary();
    const RunStats run_stats = measurement.stats;
    oss << "RunStats: inputs_enqueued=" << run_stats.inputs_enqueued
        << " inputs_dropped=" << run_stats.inputs_dropped
        << " inputs_pushed=" << run_stats.inputs_pushed
        << " outputs_ready=" << run_stats.outputs_ready
        << " outputs_pulled=" << run_stats.outputs_pulled
        << " outputs_dropped=" << run_stats.outputs_dropped
        << " avg_latency_ms=" << run_stats.avg_latency_ms
        << " min_latency_ms=" << run_stats.min_latency_ms
        << " max_latency_ms=" << run_stats.max_latency_ms
        << " elapsed_s=" << measurement.elapsed_seconds
        << " throughput_fps=" << measurement.throughput_fps << "\n";
  }
  if (opt.include_input_stats) {
    const InputStreamStats is = input_stats();
    oss << "InputStreamStats: push_count=" << is.push_count << " push_failures=" << is.push_failures
        << " pull_count=" << is.pull_count << " poll_count=" << is.poll_count
        << " dropped_frames=" << is.dropped_frames << " renegotiations=" << is.renegotiations
        << " alloc_grows=" << is.alloc_grows << " growth_blocked=" << is.growth_blocked
        << " renegotiation_blocked=" << is.renegotiation_blocked
        << " avg_alloc_us=" << is.avg_alloc_us << " avg_map_us=" << is.avg_map_us
        << " avg_copy_us=" << is.avg_copy_us << " avg_push_us=" << is.avg_push_us
        << " avg_pull_wait_us=" << is.avg_pull_wait_us << " avg_decode_us=" << is.avg_decode_us
        << "\n";
  }
  if (opt.include_power && st->power_monitor) {
    const std::string power = format_power_summary(st->power_monitor->summary());
    if (!power.empty()) {
      oss << power;
    }
  }
  if (opt.include_system_info && !st->diag_sysinfo.empty()) {
    oss << "System: " << st->diag_sysinfo << "\n";
  }

  return oss.str();
}

} // namespace simaai::neat

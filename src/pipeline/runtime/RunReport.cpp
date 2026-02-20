#include "pipeline/Run.h"
#include "RunInternal.h"

#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"

#include <mutex>
#include <sstream>
#include <string>
#include <utility>

namespace simaai::neat {

RunStats Run::stats() const {
  RunStats out;
  if (!state_)
    return out;
  auto st = state_;
  out.inputs_enqueued = st->inputs_enqueued.load();
  out.inputs_dropped = st->inputs_dropped.load();
  out.inputs_pushed = st->inputs_pushed.load();
  out.outputs_ready = st->outputs_ready.load();
  out.outputs_pulled = st->outputs_pulled.load();
  out.outputs_dropped = st->outputs_dropped.load();
  {
    std::lock_guard<std::mutex> lock(st->latency_mu);
    if (st->latency_count > 0) {
      out.avg_latency_ms = st->latency_mean_ms;
      out.min_latency_ms = st->latency_min_ms;
      out.max_latency_ms = st->latency_max_ms;
    }
  }
  return out;
}

InputStreamStats Run::input_stats() const {
  if (!state_)
    return {};
  return state_->stream.stats();
}

RunDiagSnapshot Run::diag_snapshot() const {
  RunDiagSnapshot out;
  if (!state_)
    return out;
  const auto diag = state_->stream.diag_ctx();
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

  return out;
}

std::string Run::report(const RunReportOptions& opt) const {
  if (!state_)
    return {};
  auto st = state_;
  const auto diag = st->stream.diag_ctx();
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
    const RunStats run_stats = stats();
    oss << "RunStats: inputs_enqueued=" << run_stats.inputs_enqueued
        << " inputs_dropped=" << run_stats.inputs_dropped
        << " inputs_pushed=" << run_stats.inputs_pushed
        << " outputs_ready=" << run_stats.outputs_ready
        << " outputs_pulled=" << run_stats.outputs_pulled
        << " outputs_dropped=" << run_stats.outputs_dropped
        << " avg_latency_ms=" << run_stats.avg_latency_ms
        << " min_latency_ms=" << run_stats.min_latency_ms
        << " max_latency_ms=" << run_stats.max_latency_ms << "\n";
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
  if (opt.include_system_info && !st->diag_sysinfo.empty()) {
    oss << "System: " << st->diag_sysinfo << "\n";
  }

  return oss.str();
}

} // namespace simaai::neat

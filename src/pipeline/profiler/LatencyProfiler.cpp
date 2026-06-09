#include "pipeline/LatencyProfiler.h"

#include "pipeline/Run.h"
#include "pipeline/Graph.h"

#include <algorithm>
#include <atomic>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

#if defined(SIMA_HAS_NEAT_PROFILER)
#include "profiler_events.h"
#include "profiler_scoped_timer.h"
#endif

namespace simaai::neat {

namespace {

#if defined(SIMA_HAS_NEAT_PROFILER)
std::atomic_bool& active_latency_profiler() {
  static std::atomic_bool active{false};
  return active;
}

const char* backend_name(uint8_t backend) {
  switch (backend) {
  case SIMA_NEAT_PROFILER_BACKEND_MLA:
    return "MLA";
  case SIMA_NEAT_PROFILER_BACKEND_A65:
    return "A65";
  case SIMA_NEAT_PROFILER_BACKEND_EV74:
    return "EV74";
  case SIMA_NEAT_PROFILER_BACKEND_BOXDECODE:
    return "BoxDecode";
  case SIMA_NEAT_PROFILER_BACKEND_MEMCPY:
    return "Memcpy";
  case SIMA_NEAT_PROFILER_BACKEND_DISPATCHER:
    return "Dispatcher";
  default:
    return "Unknown";
  }
}

const char* phase_name(uint8_t phase) {
  switch (phase) {
  case SIMA_NEAT_PROFILER_PHASE_RUN:
    return "Run";
  case SIMA_NEAT_PROFILER_PHASE_LOAD:
    return "Load";
  case SIMA_NEAT_PROFILER_PHASE_GET_HANDLE:
    return "GetHandle";
  case SIMA_NEAT_PROFILER_PHASE_BUILD_IO:
    return "BuildIO";
  case SIMA_NEAT_PROFILER_PHASE_COLLECT:
    return "Collect";
  case SIMA_NEAT_PROFILER_PHASE_EXEC:
    return "Exec";
  case SIMA_NEAT_PROFILER_PHASE_POST_FIXUP:
    return "PostFixup";
  default:
    return "Unknown";
  }
}
#endif

void compute_aggregates(std::vector<ProfilerKernelInvocation>& invocations,
                        std::vector<ProfilerKernelAggregate>* out) {
  std::map<std::tuple<std::string, std::string, std::string, std::string, int32_t, int32_t,
                      std::uint64_t, std::int32_t, std::int32_t, std::int32_t, std::string>,
           ProfilerKernelAggregate>
      groups;
  for (const auto& inv : invocations) {
    auto key = std::make_tuple(inv.backend, inv.phase, inv.kernel_name, inv.stage_name,
                               inv.physical_input_index, inv.output_slot, inv.run_id_hash,
                               inv.pipeline_segment_id, inv.runtime_node_id, inv.public_node_id,
                               inv.gst_element_name);
    auto& agg = groups[key];
    if (agg.count == 0) {
      agg.backend = inv.backend;
      agg.phase = inv.phase;
      agg.kernel_name = inv.kernel_name;
      agg.stage_name = inv.stage_name;
      agg.physical_input_index = inv.physical_input_index;
      agg.output_slot = inv.output_slot;
      agg.run_id_hash = inv.run_id_hash;
      agg.pipeline_segment_id = inv.pipeline_segment_id;
      agg.runtime_node_id = inv.runtime_node_id;
      agg.public_node_id = inv.public_node_id;
      agg.gst_element_name = inv.gst_element_name;
      agg.min_ms = inv.duration_ms();
      agg.max_ms = inv.duration_ms();
    } else {
      agg.min_ms = std::min(agg.min_ms, inv.duration_ms());
      agg.max_ms = std::max(agg.max_ms, inv.duration_ms());
    }
    agg.count += 1;
    agg.total_ms += inv.duration_ms();
  }
  out->clear();
  out->reserve(groups.size());
  for (auto& kv : groups) {
    out->push_back(std::move(kv.second));
  }
  std::sort(out->begin(), out->end(),
            [](const ProfilerKernelAggregate& a, const ProfilerKernelAggregate& b) {
              if (a.backend != b.backend)
                return a.backend < b.backend;
              if (a.phase != b.phase)
                return a.phase < b.phase;
              if (a.stage_name != b.stage_name)
                return a.stage_name < b.stage_name;
              if (a.kernel_name != b.kernel_name)
                return a.kernel_name < b.kernel_name;
              if (a.physical_input_index != b.physical_input_index)
                return a.physical_input_index < b.physical_input_index;
              if (a.output_slot != b.output_slot)
                return a.output_slot < b.output_slot;
              if (a.pipeline_segment_id != b.pipeline_segment_id)
                return a.pipeline_segment_id < b.pipeline_segment_id;
              if (a.runtime_node_id != b.runtime_node_id)
                return a.runtime_node_id < b.runtime_node_id;
              if (a.public_node_id != b.public_node_id)
                return a.public_node_id < b.public_node_id;
              return a.gst_element_name < b.gst_element_name;
            });
}

} // namespace

LatencyProfiler::LatencyProfiler(Options o) : options_(o) {
#if defined(SIMA_HAS_NEAT_PROFILER)
  bool expected = false;
  if (!active_latency_profiler().compare_exchange_strong(expected, true,
                                                         std::memory_order_acq_rel,
                                                         std::memory_order_acquire)) {
    throw std::runtime_error(
        "LatencyProfiler: another profiler is already active; profiler events are process-global");
  }
  owns_global_profiler_ = true;
  enabled_at_attach_ = sima_neat_profiler_enabled() != 0;
  if (options_.ring_capacity > 0U) {
    sima_neat_profiler_set_capacity(options_.ring_capacity);
  }
  sima_neat_profiler_reset();
  sima_neat_profiler_memcpy_reset();
  sima_neat_profiler_set_enabled(1);
#endif
}

LatencyProfiler::~LatencyProfiler() {
#if defined(SIMA_HAS_NEAT_PROFILER)
  // Restore prior global gate.  This is best-effort: if the user keeps the
  // profiler attached for the lifetime of the process, leave it enabled.
  if (!enabled_at_attach_) {
    sima_neat_profiler_set_enabled(0);
  }
  if (owns_global_profiler_) {
    active_latency_profiler().store(false, std::memory_order_release);
  }
#endif
}

void LatencyProfiler::attach(Run& run) {
  attached_run_ = &run;
}

void LatencyProfiler::attach(Graph& graph) {
  attached_graph_ = &graph;
}

void LatencyProfiler::mark_warmup_done() {
#if defined(SIMA_HAS_NEAT_PROFILER)
  sima_neat_profiler_reset();
  sima_neat_profiler_memcpy_reset();
#endif
}

ProfilerReport LatencyProfiler::finalize() {
  ProfilerReport report;
  report.warmup_frames = options_.warmup_frames;
  if (attached_run_) {
    report.end_to_end = attached_run_->stats();
    report.input_stream = attached_run_->input_stats();
    report.diag = attached_run_->diag_snapshot();
    report.frames_total = static_cast<std::int64_t>(report.end_to_end.outputs_pulled);
  }

#if defined(SIMA_HAS_NEAT_PROFILER)
  // Drain kernel events.
  if (options_.capture_kernels) {
    const std::size_t available = sima_neat_profiler_drain(nullptr, 0U);
    std::vector<SimaNeatProfilerEvent> raw(available);
    if (!raw.empty()) {
      const std::size_t got = sima_neat_profiler_drain(raw.data(), raw.size());
      raw.resize(got);
    }
    report.kernel_invocations.reserve(raw.size());
    for (const auto& e : raw) {
      ProfilerKernelInvocation inv;
      inv.start_ns = e.start_ns;
      inv.end_ns = e.end_ns;
      inv.backend = backend_name(e.backend);
      inv.phase = phase_name(e.phase);
      inv.physical_input_index = e.physical_input_index;
      inv.output_slot = e.output_slot;
      inv.frame_id = e.frame_id;
      inv.request_id = e.request_id;
      inv.bytes = e.bytes;
      inv.kernel_name = e.kernel_name;
      inv.stage_name = e.stage_name;
      inv.in_segment = e.in_segment;
      inv.out_segment = e.out_segment;
      report.kernel_invocations.push_back(std::move(inv));
    }
    compute_aggregates(report.kernel_invocations, &report.kernel_aggregates);
  }

  // Memcpy site snapshots.
  if (options_.capture_memcpy) {
    for (int i = 0; i < SIMA_NEAT_PROFILER_MEMCPY_SITE_COUNT; ++i) {
      const auto site = static_cast<SimaNeatProfilerMemcpySite>(i);
      const auto stats = sima_neat_profiler_memcpy_stats(site);
      ProfilerMemcpySite m;
      m.site_name = sima_neat_profiler_memcpy_site_name(site);
      m.calls = stats.calls;
      m.total_ns = stats.total_ns;
      m.total_bytes = stats.total_bytes;
      m.max_ns = stats.max_ns;
      report.memcpy_sites.push_back(std::move(m));
    }
  }

  report.profiler_emits = sima_neat_profiler_emit_count();
  report.profiler_dropped = sima_neat_profiler_dropped_count();
#endif

  return report;
}

} // namespace simaai::neat

#include "pipeline/LatencyProfiler.h"

#include "pipeline/Run.h"
#include "pipeline/Session.h"

#include <algorithm>
#include <map>
#include <sstream>
#include <utility>

#if defined(SIMA_HAS_NEAT_PROFILER)
#include "profiler_events.h"
#include "profiler_scoped_timer.h"
#endif

namespace simaai::neat {

namespace {

#if defined(SIMA_HAS_NEAT_PROFILER)
const char* backend_name(uint8_t backend) {
  switch (backend) {
    case SIMA_NEAT_PROFILER_BACKEND_MLA:       return "MLA";
    case SIMA_NEAT_PROFILER_BACKEND_A65:       return "A65";
    case SIMA_NEAT_PROFILER_BACKEND_EV74:      return "EV74";
    case SIMA_NEAT_PROFILER_BACKEND_BOXDECODE: return "BoxDecode";
    case SIMA_NEAT_PROFILER_BACKEND_MEMCPY:    return "Memcpy";
    default:                                   return "Unknown";
  }
}

const char* phase_name(uint8_t phase) {
  switch (phase) {
    case SIMA_NEAT_PROFILER_PHASE_RUN:        return "Run";
    case SIMA_NEAT_PROFILER_PHASE_LOAD:       return "Load";
    case SIMA_NEAT_PROFILER_PHASE_GET_HANDLE: return "GetHandle";
    case SIMA_NEAT_PROFILER_PHASE_BUILD_IO:   return "BuildIO";
    case SIMA_NEAT_PROFILER_PHASE_COLLECT:    return "Collect";
    case SIMA_NEAT_PROFILER_PHASE_EXEC:       return "Exec";
    case SIMA_NEAT_PROFILER_PHASE_POST_FIXUP: return "PostFixup";
    default:                                  return "Unknown";
  }
}
#endif

void compute_aggregates(std::vector<ProfilerKernelInvocation>& invocations,
                        std::vector<ProfilerKernelAggregate>* out) {
  std::map<std::tuple<std::string, std::string, std::string, int32_t, int32_t>,
           ProfilerKernelAggregate>
      groups;
  for (const auto& inv : invocations) {
    auto key = std::make_tuple(inv.backend, inv.kernel_name, inv.stage_name,
                               inv.physical_input_index, inv.output_slot);
    auto& agg = groups[key];
    if (agg.count == 0) {
      agg.backend              = inv.backend;
      agg.kernel_name          = inv.kernel_name;
      agg.stage_name           = inv.stage_name;
      agg.physical_input_index = inv.physical_input_index;
      agg.output_slot          = inv.output_slot;
      agg.min_ms               = inv.duration_ms();
      agg.max_ms               = inv.duration_ms();
    } else {
      agg.min_ms = std::min(agg.min_ms, inv.duration_ms());
      agg.max_ms = std::max(agg.max_ms, inv.duration_ms());
    }
    agg.count    += 1;
    agg.total_ms += inv.duration_ms();
  }
  out->clear();
  out->reserve(groups.size());
  for (auto& kv : groups) {
    out->push_back(std::move(kv.second));
  }
  std::sort(out->begin(), out->end(),
            [](const ProfilerKernelAggregate& a,
               const ProfilerKernelAggregate& b) {
              if (a.backend != b.backend) return a.backend < b.backend;
              if (a.stage_name != b.stage_name) return a.stage_name < b.stage_name;
              if (a.physical_input_index != b.physical_input_index)
                return a.physical_input_index < b.physical_input_index;
              return a.output_slot < b.output_slot;
            });
}

}  // namespace

LatencyProfiler::LatencyProfiler(Options o) : options_(o) {
#if defined(SIMA_HAS_NEAT_PROFILER)
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
#endif
}

void LatencyProfiler::attach(Run& run) { attached_run_ = &run; }

void LatencyProfiler::attach(Session& session) {
  attached_session_ = &session;
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
    report.end_to_end   = attached_run_->stats();
    report.input_stream = attached_run_->input_stats();
    report.diag         = attached_run_->diag_snapshot();
    report.frames_total = static_cast<std::int64_t>(report.end_to_end.outputs_pulled);
  }

#if defined(SIMA_HAS_NEAT_PROFILER)
  // Drain kernel events.
  if (options_.capture_kernels) {
    const std::size_t available =
        sima_neat_profiler_drain(nullptr, 0U);
    std::vector<SimaNeatProfilerEvent> raw(available);
    if (!raw.empty()) {
      const std::size_t got =
          sima_neat_profiler_drain(raw.data(), raw.size());
      raw.resize(got);
    }
    report.kernel_invocations.reserve(raw.size());
    for (const auto& e : raw) {
      ProfilerKernelInvocation inv;
      inv.start_ns             = e.start_ns;
      inv.end_ns               = e.end_ns;
      inv.backend              = backend_name(e.backend);
      inv.phase                = phase_name(e.phase);
      inv.physical_input_index = e.physical_input_index;
      inv.output_slot          = e.output_slot;
      inv.frame_id             = e.frame_id;
      inv.request_id           = e.request_id;
      inv.bytes                = e.bytes;
      inv.kernel_name          = e.kernel_name;
      inv.stage_name           = e.stage_name;
      inv.in_segment           = e.in_segment;
      inv.out_segment          = e.out_segment;
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
      m.site_name   = sima_neat_profiler_memcpy_site_name(site);
      m.calls       = stats.calls;
      m.total_ns    = stats.total_ns;
      m.total_bytes = stats.total_bytes;
      m.max_ns      = stats.max_ns;
      report.memcpy_sites.push_back(std::move(m));
    }
  }

  report.profiler_emits   = sima_neat_profiler_emit_count();
  report.profiler_dropped = sima_neat_profiler_dropped_count();
#endif

  return report;
}

}  // namespace simaai::neat

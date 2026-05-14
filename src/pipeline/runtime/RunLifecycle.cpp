#include "pipeline/Run.h"
#include "RunInternal.h"

#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <cstdio>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace simaai::neat {

namespace {

bool stop_trace_enabled() {
  return pipeline_internal::env_bool("SIMA_STOP_TRACE", false);
}

bool abort_on_hung_stop_threads() {
  return pipeline_internal::env_bool("SIMA_PIPELINE_ABORT_ON_HUNG_STOP_THREADS", false);
}

} // namespace

void Run::close_input() {
  if (!state_)
    return;
  auto st = state_;
  {
    std::lock_guard<std::mutex> lock(st->in_mu);
    st->input_closed = true;
  }
  st->in_cv.notify_all();
}

void Run::stop() {
  if (!state_)
    return;
  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] Run::stop begin\n");
  }
  auto st = state_;
  if (pipeline_internal::env_bool("SIMA_PIPELINE_DEBUG", false) ||
      pipeline_internal::env_bool("SIMA_GRAPH_DEBUG", false)) {
    std::fprintf(stderr, "[PIPELINE] stop called\n");
  }
  // Ensure the input thread can exit cleanly.
  close_input();
  {
    std::lock_guard<std::mutex> lock(st->in_mu);
    st->in_queue.clear();
  }
  st->stop_requested.store(true);
  if (st->power_monitor) {
    st->power_monitor->stop();
  }
  st->in_cv.notify_all();
  st->out_cv.notify_all();
  st->stream.stop_async();
  // Stop the underlying stream first to unblock any appsrc push waiting on downstream.
  const int stream_stop_timeout_ms =
      std::max(0, pipeline_internal::env_int("SIMA_PIPELINE_STREAM_STOP_TIMEOUT_MS", 2000));
  const int stream_stop_timeout_ms_2 = stream_stop_timeout_ms;
  if (stream_stop_timeout_ms <= 0) {
    st->stream.stop();
  } else {
    struct StopCtx {
      std::atomic<bool> done{false};
    };
    auto ctx = std::make_shared<StopCtx>();
    std::thread stop_thread([st, ctx]() {
      try {
        st->stream.stop();
      } catch (const std::exception& e) {
        if (pipeline_internal::env_bool("SIMA_PIPELINE_DEBUG", false) ||
            pipeline_internal::env_bool("SIMA_GRAPH_DEBUG", false)) {
          std::fprintf(stderr, "[PIPELINE] stop stream error: %s\n", e.what());
        }
      }
      ctx->done.store(true, std::memory_order_relaxed);
    });
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(stream_stop_timeout_ms);
    while (!ctx->done.load(std::memory_order_relaxed)) {
      if (std::chrono::steady_clock::now() >= deadline)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (ctx->done.load(std::memory_order_relaxed)) {
      stop_thread.join();
    } else {
      if (pipeline_internal::env_bool("SIMA_PIPELINE_DEBUG", false) ||
          pipeline_internal::env_bool("SIMA_GRAPH_DEBUG", false)) {
        std::fprintf(stderr,
                     "[PIPELINE] stop: stream.stop did not exit within %dms; forcing stop\n",
                     stream_stop_timeout_ms);
      }
      st->stream.stop_async();
      if (stream_stop_timeout_ms_2 > 0) {
        const auto extra_deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(stream_stop_timeout_ms_2);
        while (!ctx->done.load(std::memory_order_relaxed)) {
          if (std::chrono::steady_clock::now() >= extra_deadline)
            break;
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }
      if (ctx->done.load(std::memory_order_relaxed)) {
        stop_thread.join();
      } else {
        const int waited_ms = stream_stop_timeout_ms + stream_stop_timeout_ms_2;
        if (abort_on_hung_stop_threads()) {
          std::fprintf(stderr,
                       "[PIPELINE] stop: stream.stop did not exit after %dms; aborting "
                       "(SIMA_PIPELINE_ABORT_ON_HUNG_STOP_THREADS=1)\n",
                       waited_ms);
          std::terminate();
        }
        std::fprintf(stderr,
                     "[PIPELINE] stop: stream.stop did not exit after %dms; detaching "
                     "stop thread and continuing\n",
                     waited_ms);
        stop_thread.detach();
      }
    }
  }
  if (st->input_thread.joinable()) {
    const int timeout_ms =
        std::max(0, pipeline_internal::env_int("SIMA_PIPELINE_INPUT_THREAD_STOP_TIMEOUT_MS", 2000));
    const int timeout_ms_2 = timeout_ms;
    if (timeout_ms > 0) {
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
      while (!st->input_thread_done.load(std::memory_order_relaxed)) {
        if (std::chrono::steady_clock::now() >= deadline)
          break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    if (!st->input_thread_done.load(std::memory_order_relaxed)) {
      if (pipeline_internal::env_bool("SIMA_PIPELINE_DEBUG", false) ||
          pipeline_internal::env_bool("SIMA_GRAPH_DEBUG", false)) {
        std::fprintf(stderr,
                     "[PIPELINE] stop: input_thread did not exit within %dms; forcing stop\n",
                     timeout_ms);
      }
      st->stream.stop_async();
      if (timeout_ms_2 > 0) {
        const auto extra_deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms_2);
        while (!st->input_thread_done.load(std::memory_order_relaxed)) {
          if (std::chrono::steady_clock::now() >= extra_deadline)
            break;
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }
    }
    if (!st->input_thread_done.load(std::memory_order_relaxed)) {
      const int waited_ms = timeout_ms + timeout_ms_2;
      if (abort_on_hung_stop_threads()) {
        std::fprintf(stderr,
                     "[PIPELINE] stop: input_thread did not exit after %dms; aborting "
                     "(SIMA_PIPELINE_ABORT_ON_HUNG_STOP_THREADS=1)\n",
                     waited_ms);
        std::terminate();
      }
      std::fprintf(stderr,
                   "[PIPELINE] stop: input_thread did not exit after %dms; detaching "
                   "input thread and continuing\n",
                   waited_ms);
      st->input_thread.detach();
      return;
    }
    st->input_thread.join();
  }
  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] Run::stop end\n");
  }
}

void Run::close() {
  if (!state_)
    return;
  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] Run::close begin\n");
  }
  auto st = state_;
  if (!owns_ref_) {
    if (pipeline_internal::env_bool("SIMA_PIPELINE_TEARDOWN_DEBUG", false)) {
      const int refs = st->handle_refs.load(std::memory_order_relaxed);
      std::printf("[DBG] Run::close: no-handle-ref handle_refs=%d\n", refs);
    }
    state_.reset();
    return;
  }
  owns_ref_ = false;
  const int remaining = st->handle_refs.fetch_sub(1, std::memory_order_acq_rel) - 1;
  if (pipeline_internal::env_bool("SIMA_PIPELINE_TEARDOWN_DEBUG", false)) {
    std::printf("[DBG] Run::close: handle_refs=%d action=%s\n", remaining,
                (remaining > 0) ? "defer" : "teardown");
  }
  if (remaining > 0) {
    state_.reset();
    return;
  }
  if (remaining < 0) {
    st->handle_refs.store(0, std::memory_order_relaxed);
  }
  stop();
  const int drain_ms = pipeline_internal::env_int("SIMA_PIPELINE_DRAIN_BEFORE_TEARDOWN_MS", 1500);
  const int drain_min_outputs = pipeline_internal::env_int("SIMA_PIPELINE_DRAIN_MIN_OUTPUTS", 1);
  if (drain_ms > 0 && st->supports_pull &&
      st->outputs_pulled.load(std::memory_order_relaxed) <= drain_min_outputs) {
    st->stream.drain_before_teardown(drain_ms);
  }
  if (st->diag_enabled && !st->diag_logged.exchange(true)) {
    auto log_diag = [&](auto& st_ref) {
      const auto diag = st_ref.stream.diag_ctx();
      std::ostringstream oss;
      oss << "[DIAG] async_tput\n";
      if (diag && !diag->pipeline_string.empty()) {
        oss << "Pipeline:\n" << diag->pipeline_string << "\n";
      }
      if (diag) {
        const std::string boundary = pipeline_internal::boundary_summary(diag);
        if (!boundary.empty())
          oss << boundary;
        const std::string stages = pipeline_internal::stage_timing_summary(diag);
        if (!stages.empty())
          oss << stages;
        const std::string elements = pipeline_internal::element_timing_summary(diag);
        if (!elements.empty())
          oss << elements;
        const std::string flow = pipeline_internal::element_flow_summary(diag);
        if (!flow.empty())
          oss << flow;

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
        if (!diag->next_cpu_decisions.empty()) {
          oss << "Next CPU decisions:\n";
          for (const auto& d : diag->next_cpu_decisions) {
            oss << "  - node=" << d.node_index << " kind=" << d.node_kind;
            if (!d.node_label.empty())
              oss << " label=" << d.node_label;
            oss << " next_cpu=" << d.next_cpu << " applied=" << (d.applied ? "1" : "0") << "\n";
          }
        }
      }

      const std::string pipeline = diag ? diag->pipeline_string : std::string();
      const int queue2_depth = (diag && diag->queue2_enabled)
                                   ? diag->queue2_depth
                                   : run_internal::parse_queue2_depth(pipeline);
      if (diag && !diag->queue2_enabled) {
        if (queue2_depth > 0) {
          oss << "queue2 depth=" << queue2_depth << " (manual)\n";
        } else {
          oss << "queue2 disabled\n";
        }
      } else if (queue2_depth > 0) {
        oss << "queue2 depth=" << queue2_depth << "\n";
      }

      int num_cvu = run_internal::parse_num_buffers_for(pipeline, "neatprocesscvu");
      int num_mla = run_internal::parse_num_buffers_for(pipeline, "neatprocessmla");
      if (num_cvu > 0 || num_mla > 0) {
        oss << "num_buffers_cvu=" << num_cvu << " num_buffers_mla=" << num_mla << "\n";
      }

      RunStats run_stats;
      run_stats.inputs_enqueued = st_ref.inputs_enqueued.load();
      run_stats.inputs_dropped = st_ref.inputs_dropped.load();
      run_stats.inputs_pushed = st_ref.inputs_pushed.load();
      run_stats.outputs_ready = st_ref.outputs_ready.load();
      run_stats.outputs_pulled = st_ref.outputs_pulled.load();
      run_stats.outputs_dropped = st_ref.outputs_dropped.load();
      {
        std::lock_guard<std::mutex> lock(st_ref.latency_mu);
        if (st_ref.latency_count > 0) {
          run_stats.avg_latency_ms = st_ref.latency_mean_ms;
          run_stats.min_latency_ms = st_ref.latency_min_ms;
          run_stats.max_latency_ms = st_ref.latency_max_ms;
        }
      }

      oss << "RunStats: inputs_enqueued=" << run_stats.inputs_enqueued
          << " inputs_dropped=" << run_stats.inputs_dropped
          << " inputs_pushed=" << run_stats.inputs_pushed
          << " outputs_ready=" << run_stats.outputs_ready
          << " outputs_pulled=" << run_stats.outputs_pulled
          << " outputs_dropped=" << run_stats.outputs_dropped
          << " avg_latency_ms=" << run_stats.avg_latency_ms
          << " min_latency_ms=" << run_stats.min_latency_ms
          << " max_latency_ms=" << run_stats.max_latency_ms << "\n";

      const InputStreamStats is = st_ref.stream.stats();
      oss << "InputStreamStats: push_count=" << is.push_count
          << " push_failures=" << is.push_failures << " pull_count=" << is.pull_count
          << " poll_count=" << is.poll_count << " dropped_frames=" << is.dropped_frames
          << " renegotiations=" << is.renegotiations << " alloc_grows=" << is.alloc_grows
          << " growth_blocked=" << is.growth_blocked
          << " renegotiation_blocked=" << is.renegotiation_blocked
          << " avg_alloc_us=" << is.avg_alloc_us << " avg_map_us=" << is.avg_map_us
          << " avg_copy_us=" << is.avg_copy_us << " avg_push_us=" << is.avg_push_us
          << " avg_pull_wait_us=" << is.avg_pull_wait_us << " avg_decode_us=" << is.avg_decode_us
          << "\n";

      if (!st_ref.diag_sysinfo.empty()) {
        oss << "System: " << st_ref.diag_sysinfo << "\n";
      }

      std::printf("%s", oss.str().c_str());
    };
    log_diag(*st);
  }
  st->stream.close();
  state_.reset();
  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] Run::close end\n");
  }
}

} // namespace simaai::neat

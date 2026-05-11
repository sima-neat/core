#include "internal/GraphRunState.h"

namespace simaai::neat::graph {
void GraphRun::stop() {
  if (!state_)
    return;
  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] GraphRun::stop begin\n");
  }
  if (graph_debug_enabled()) {
    std::fprintf(stderr, "[GRAPH] GraphRun::stop called\n");
  }
  if (state_->power_monitor) {
    state_->power_monitor->stop();
  }
  for (auto& pipe : state_->pipelines) {
    if (!pipe)
      continue;
    if (!pipe->built.load(std::memory_order_acquire))
      continue;
    try {
      pipe->run.close_input();
      pipe->run.stop();
    } catch (const std::exception& e) {
      if (graph_debug_enabled()) {
        std::fprintf(stderr, "[GRAPH] stop_error seg=%zu where=run.stop err=%s\n",
                     static_cast<std::size_t>(pipe->seg.id), e.what());
      }
    }
    if (graph_debug_enabled()) {
      std::fprintf(
          stderr,
          "[GRAPH] identity_diag seg=%zu identity_rewrite_count=%lld "
          "identity_map_miss_count=%lld\n",
          static_cast<std::size_t>(pipe->seg.id),
          static_cast<long long>(pipe->identity_rewrite_count.load(std::memory_order_relaxed)),
          static_cast<long long>(pipe->identity_map_miss_count.load(std::memory_order_relaxed)));
    }
  }
  state_->signal_stop();
  if (graph_output_rate_enabled() && !state_->output_rate_reported.exchange(true)) {
    graph_output_rate_summary(&state_->node_labels, &state_->sinks);
  }
  if (graph_sched_debug_enabled() && !state_->sched_reported.exchange(true)) {
    graph_sched_summary(&state_->node_labels);
  }
  if (pipeline_internal::env_bool("SIMA_GRAPH_TEARDOWN_DEBUG", false) || graph_debug_enabled()) {
    std::size_t stage_workers = 0;
    std::size_t pull_threads = 0;
    std::size_t push_threads = 0;
    for (const auto& st : state_->stages) {
      if (st && st->worker.joinable())
        stage_workers++;
    }
    for (const auto& pipe : state_->pipelines) {
      if (!pipe)
        continue;
      if (pipe->pull_thread.joinable())
        pull_threads++;
      if (pipe->push_thread.joinable())
        push_threads++;
    }
    std::fprintf(stderr, "[GRAPH] GraphRun::stop threads stage=%zu pull=%zu push=%zu\n",
                 stage_workers, pull_threads, push_threads);
  }

  const int stop_timeout_ms = std::max(0, env_int("SIMA_GRAPH_STOP_TIMEOUT_MS", 2000));
  auto wait_done = [&](std::atomic<bool>& done_flag) -> bool {
    if (done_flag.load(std::memory_order_relaxed))
      return true;
    if (stop_timeout_ms <= 0)
      return done_flag.load(std::memory_order_relaxed);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(stop_timeout_ms);
    while (!done_flag.load(std::memory_order_relaxed)) {
      if (std::chrono::steady_clock::now() >= deadline)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return done_flag.load(std::memory_order_relaxed);
  };

  for (auto& pipe : state_->pipelines) {
    if (!pipe)
      continue;
    if (pipe->built.load(std::memory_order_acquire)) {
      if (pipeline_internal::env_bool("SIMA_GRAPH_DIAG_ON_STOP", false)) {
        const std::string diag = pipe->run.diagnostics_summary();
        if (!diag.empty()) {
          std::fprintf(stderr, "[GRAPH] diag seg=%zu\n%s\n", static_cast<std::size_t>(pipe->seg.id),
                       diag.c_str());
        }
      }
    }
  }

  for (std::size_t i = 0; i < state_->stages.size(); ++i) {
    auto& st = state_->stages[i];
    if (st && st->worker.joinable()) {
      if (wait_done(st->worker_done)) {
        try {
          st->worker.join();
        } catch (const std::exception& e) {
          if (graph_debug_enabled()) {
            std::fprintf(stderr,
                         "[GRAPH] stop_error stage_index=%zu node=%zu where=stage.join err=%s\n", i,
                         static_cast<std::size_t>(st->node_id), e.what());
          }
          throw;
        }
      } else {
        if (graph_debug_enabled()) {
          std::fprintf(stderr,
                       "[GRAPH] stop_timeout stage_index=%zu node=%zu timeout_ms=%d; detaching\n",
                       i, static_cast<std::size_t>(st->node_id), stop_timeout_ms);
        }
        st->worker.detach();
      }
    }
  }
  for (std::size_t i = 0; i < state_->pipelines.size(); ++i) {
    auto& pipe = state_->pipelines[i];
    if (!pipe)
      continue;
    if (pipe->pull_thread.joinable()) {
      if (wait_done(pipe->pull_done)) {
        try {
          pipe->pull_thread.join();
        } catch (const std::exception& e) {
          if (graph_debug_enabled()) {
            std::fprintf(stderr, "[GRAPH] stop_error seg=%zu where=pull.join err=%s\n",
                         static_cast<std::size_t>(pipe->seg.id), e.what());
          }
          throw;
        }
      } else {
        if (graph_debug_enabled()) {
          std::fprintf(stderr,
                       "[GRAPH] stop_timeout seg=%zu where=pull.join timeout_ms=%d; detaching\n",
                       static_cast<std::size_t>(pipe->seg.id), stop_timeout_ms);
        }
        pipe->pull_thread.detach();
      }
    }
    if (pipe->push_thread.joinable()) {
      if (wait_done(pipe->push_done)) {
        try {
          pipe->push_thread.join();
        } catch (const std::exception& e) {
          if (graph_debug_enabled()) {
            std::fprintf(stderr, "[GRAPH] stop_error seg=%zu where=push.join err=%s\n",
                         static_cast<std::size_t>(pipe->seg.id), e.what());
          }
          throw;
        }
      } else {
        if (graph_debug_enabled()) {
          std::fprintf(stderr,
                       "[GRAPH] stop_timeout seg=%zu where=push.join timeout_ms=%d; detaching\n",
                       static_cast<std::size_t>(pipe->seg.id), stop_timeout_ms);
        }
        pipe->push_thread.detach();
      }
    }
  }
  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] GraphRun::stop end\n");
  }
}
} // namespace simaai::neat::graph

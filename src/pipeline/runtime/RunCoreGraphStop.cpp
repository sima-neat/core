#include "RunCore.h"

#include "pipeline/internal/DecoderAdmissionClient.h"
#include "pipeline/internal/EnvUtil.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace simaai::neat::graph {

bool graph_debug_enabled();
bool graph_output_rate_enabled();
bool graph_sched_debug_enabled();
bool stop_trace_enabled();
void graph_output_rate_summary(
    const std::vector<std::string>* labels,
    const std::unordered_map<NodeId, std::shared_ptr<simaai::neat::runtime::BlockingQueueSample>>*
        sinks);
void graph_sched_summary(const std::vector<std::string>* labels);

} // namespace simaai::neat::graph

namespace simaai::neat::runtime {

void RunCore::stop_graph() {
  if (!graph_execution_) {
    return;
  }
  if (simaai::neat::graph::stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] GraphRun::stop begin\n");
  }
  if (simaai::neat::graph::graph_debug_enabled()) {
    std::fprintf(stderr, "[GRAPH] GraphRun::stop called\n");
  }
  if (power_monitor) {
    power_monitor->stop();
  }

  ExecutionGraphRuntime& execution = graph_execution();
  for (auto& pipe : execution.pipelines) {
    if (!pipe) {
      continue;
    }
    if (!pipe->transport.built.load(std::memory_order_acquire)) {
      continue;
    }
    try {
      if (pipe->run_core) {
        pipe->run_core->close_input();
        pipe->run_core->stop();
      }
    } catch (const std::exception& e) {
      if (simaai::neat::graph::graph_debug_enabled()) {
        std::fprintf(stderr, "[GRAPH] stop_error seg=%zu where=run.stop err=%s\n",
                     static_cast<std::size_t>(pipe->seg.id), e.what());
      }
    }
    if (simaai::neat::graph::graph_debug_enabled()) {
      std::fprintf(stderr,
                   "[GRAPH] identity_diag seg=%zu identity_rewrite_count=%lld "
                   "identity_map_miss_count=%lld\n",
                   static_cast<std::size_t>(pipe->seg.id),
                   static_cast<long long>(
                       pipe->transport.identity_rewrite_count.load(std::memory_order_relaxed)),
                   static_cast<long long>(
                       pipe->transport.identity_map_miss_count.load(std::memory_order_relaxed)));
    }
  }

  if (execution.decoder_admission_active) {
    std::string release_error;
    const bool released = pipeline_internal::release_decoder_graph(
        execution.decoder_admission_group_uuid, &release_error);
    if (!released && (pipeline_internal::env_bool("SIMA_DECODER_ADMISSION_DEBUG", false) ||
                      simaai::neat::graph::graph_debug_enabled())) {
      std::fprintf(stderr, "[GRAPH] decoder_admission_release_failed group=%s err=%s\n",
                   pipeline_internal::decoder_admission_uuid_to_string(
                       execution.decoder_admission_group_uuid)
                       .c_str(),
                   release_error.empty() ? "<unknown>" : release_error.c_str());
    } else if (released && (pipeline_internal::env_bool("SIMA_DECODER_ADMISSION_DEBUG", false) ||
                            simaai::neat::graph::graph_debug_enabled())) {
      std::fprintf(stderr, "[GRAPH] decoder_admission_released group=%s\n",
                   pipeline_internal::decoder_admission_uuid_to_string(
                       execution.decoder_admission_group_uuid)
                       .c_str());
    }
    execution.decoder_admission_active = false;
  }

  graph_signal_stop();

  if (simaai::neat::graph::graph_output_rate_enabled() &&
      !graph_output_rate_reported.exchange(true)) {
    simaai::neat::graph::graph_output_rate_summary(&execution.node_labels, nullptr);
  }
  if (simaai::neat::graph::graph_sched_debug_enabled() && !graph_sched_reported.exchange(true)) {
    simaai::neat::graph::graph_sched_summary(&execution.node_labels);
  }
  if (pipeline_internal::env_bool("SIMA_GRAPH_TEARDOWN_DEBUG", false) ||
      simaai::neat::graph::graph_debug_enabled()) {
    std::size_t stage_workers = 0;
    std::size_t realtime_links = 0;
    std::size_t pull_threads = 0;
    std::size_t push_threads = 0;
    for (const auto& link : execution.realtime_links) {
      if (link) {
        realtime_links++;
      }
    }
    for (const auto& st : execution.stages) {
      if (st && st->worker.joinable()) {
        stage_workers++;
      }
    }
    for (const auto& pipe : execution.pipelines) {
      if (!pipe) {
        continue;
      }
      if (pipe->transport.pull_thread.joinable()) {
        pull_threads++;
      }
      if (pipe->transport.push_thread.joinable()) {
        push_threads++;
      }
    }
    std::fprintf(stderr,
                 "[GRAPH] GraphRun::stop threads stage=%zu realtime_links=%zu pull=%zu "
                 "push=%zu\n",
                 stage_workers, realtime_links, pull_threads, push_threads);
  }

  const int stop_timeout_ms =
      std::max(0, pipeline_internal::env_int("SIMA_GRAPH_STOP_TIMEOUT_MS", 2000));
  auto wait_done = [&](std::atomic<bool>& done_flag) -> bool {
    if (done_flag.load(std::memory_order_relaxed)) {
      return true;
    }
    if (stop_timeout_ms <= 0) {
      return done_flag.load(std::memory_order_relaxed);
    }
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(stop_timeout_ms);
    while (!done_flag.load(std::memory_order_relaxed)) {
      if (std::chrono::steady_clock::now() >= deadline) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return done_flag.load(std::memory_order_relaxed);
  };

  for (auto& pipe : execution.pipelines) {
    if (!pipe) {
      continue;
    }
    if (pipe->transport.built.load(std::memory_order_acquire) &&
        pipeline_internal::env_bool("SIMA_GRAPH_DIAG_ON_STOP", false)) {
      const std::string diag =
          pipe->run_core ? pipe->run_core->diagnostics_summary() : std::string{};
      if (!diag.empty()) {
        std::fprintf(stderr, "[GRAPH] diag seg=%zu\n%s\n", static_cast<std::size_t>(pipe->seg.id),
                     diag.c_str());
      }
    }
  }

  for (std::size_t i = 0; i < execution.stages.size(); ++i) {
    auto& st = execution.stages[i];
    if (st && st->worker.joinable()) {
      if (wait_done(st->worker_done)) {
        try {
          st->worker.join();
        } catch (const std::exception& e) {
          if (simaai::neat::graph::graph_debug_enabled()) {
            std::fprintf(stderr,
                         "[GRAPH] stop_error stage_index=%zu node=%zu where=stage.join err=%s\n", i,
                         static_cast<std::size_t>(st->node_id), e.what());
          }
          throw;
        }
      } else {
        if (simaai::neat::graph::graph_debug_enabled()) {
          std::fprintf(stderr,
                       "[GRAPH] stop_timeout stage_index=%zu node=%zu timeout_ms=%d; detaching\n",
                       i, static_cast<std::size_t>(st->node_id), stop_timeout_ms);
        }
        st->worker.detach();
      }
    }
  }

  for (auto& link : execution.realtime_links) {
    if (link) {
      link->join();
    }
  }

  for (std::size_t i = 0; i < execution.pipelines.size(); ++i) {
    auto& pipe = execution.pipelines[i];
    if (!pipe) {
      continue;
    }
    if (pipe->transport.pull_thread.joinable()) {
      if (wait_done(pipe->transport.pull_done)) {
        try {
          pipe->transport.pull_thread.join();
        } catch (const std::exception& e) {
          if (simaai::neat::graph::graph_debug_enabled()) {
            std::fprintf(stderr, "[GRAPH] stop_error seg=%zu where=pull.join err=%s\n",
                         static_cast<std::size_t>(pipe->seg.id), e.what());
          }
          throw;
        }
      } else {
        if (simaai::neat::graph::graph_debug_enabled()) {
          std::fprintf(stderr,
                       "[GRAPH] stop_timeout seg=%zu where=pull.join timeout_ms=%d; detaching\n",
                       static_cast<std::size_t>(pipe->seg.id), stop_timeout_ms);
        }
        pipe->transport.pull_thread.detach();
      }
    }

    if (pipe->transport.push_thread.joinable()) {
      if (wait_done(pipe->transport.push_done)) {
        try {
          pipe->transport.push_thread.join();
        } catch (const std::exception& e) {
          if (simaai::neat::graph::graph_debug_enabled()) {
            std::fprintf(stderr, "[GRAPH] stop_error seg=%zu where=push.join err=%s\n",
                         static_cast<std::size_t>(pipe->seg.id), e.what());
          }
          throw;
        }
      } else {
        if (simaai::neat::graph::graph_debug_enabled()) {
          std::fprintf(stderr,
                       "[GRAPH] stop_timeout seg=%zu where=push.join timeout_ms=%d; detaching\n",
                       static_cast<std::size_t>(pipe->seg.id), stop_timeout_ms);
        }
        pipe->transport.push_thread.detach();
      }
    }
  }

  if (simaai::neat::graph::stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] GraphRun::stop end\n");
  }
}

} // namespace simaai::neat::runtime

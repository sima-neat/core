#include "RunCore.h"

#include "pipeline/internal/DecoderAdmissionClient.h"
#include "pipeline/internal/EnvUtil.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
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
namespace {

double avg_ms(std::uint64_t total_ns, std::uint64_t count) {
  if (count == 0) {
    return 0.0;
  }
  return static_cast<double>(total_ns) / static_cast<double>(count) / 1000000.0;
}

double ns_to_ms(std::uint64_t ns) {
  return static_cast<double>(ns) / 1000000.0;
}

unsigned long long ull(std::uint64_t value) {
  return static_cast<unsigned long long>(value);
}

bool decoder_cma_debug_enabled() {
  return pipeline_internal::env_bool("SIMA_DECODER_CMA_DEBUG", false) ||
         pipeline_internal::env_bool("SIMA_DECODER_ADMISSION_DEBUG", false);
}

void log_decoder_cma_snapshot(const char* event) {
  if (!decoder_cma_debug_enabled()) {
    return;
  }
  std::ifstream in("/proc/meminfo");
  if (!in.is_open()) {
    std::fprintf(stderr, "[DECCMA] event=%s read_failed=1\n", event ? event : "snapshot");
    return;
  }
  long mem_free_kb = -1;
  long mem_available_kb = -1;
  long cma_total_kb = -1;
  long cma_free_kb = -1;
  std::string key;
  long value = 0;
  std::string unit;
  while (in >> key >> value >> unit) {
    if (key == "MemFree:") {
      mem_free_kb = value;
    } else if (key == "MemAvailable:") {
      mem_available_kb = value;
    } else if (key == "CmaTotal:") {
      cma_total_kb = value;
    } else if (key == "CmaFree:") {
      cma_free_kb = value;
    }
  }
  std::fprintf(stderr,
               "[DECCMA] event=%s mem_free_kb=%ld mem_available_kb=%ld cma_total_kb=%ld "
               "cma_free_kb=%ld cma_used_kb=%ld\n",
               event ? event : "snapshot", mem_free_kb, mem_available_kb, cma_total_kb, cma_free_kb,
               (cma_total_kb >= 0 && cma_free_kb >= 0) ? (cma_total_kb - cma_free_kb) : -1);
}

void dump_realtime_link_diag(const ExecutionGraphRuntime& execution) {
  for (std::size_t i = 0; i < execution.realtime_links.size(); ++i) {
    const auto& link = execution.realtime_links[i];
    if (!link) {
      continue;
    }
    const RealtimeLatestLink::Stats stats = link->stats();
    const DownstreamTarget& downstream = link->downstream();
    const std::string stream_ids = link->debug_stream_ids();
    const std::uint64_t dispatched = stats.scheduled + stats.dispatch_failed;
    std::fprintf(stderr,
                 "[GRAPH] realtime_link index=%zu downstream_kind=%d downstream_index=%zu "
                 "edge=%zu streams=%s offered=%llu scheduled=%llu overwritten=%llu "
                 "dispatch_failed=%llu "
                 "ready=%zu avg_ready_wait_ms=%.3f max_ready_wait_ms=%.3f "
                 "avg_dispatch_ms=%.3f max_dispatch_ms=%.3f no_credit_skips=%llu "
                 "credit_inflight=%zu credit_limit=%zu credit_registered=%llu "
                 "credit_released=%llu/%llu credit_missing_key=%llu\n",
                 i, static_cast<int>(downstream.kind), downstream.index, downstream.edge_index,
                 stream_ids.empty() ? "<none>" : stream_ids.c_str(), ull(stats.offered),
                 ull(stats.scheduled), ull(stats.overwritten), ull(stats.dispatch_failed),
                 stats.ready, avg_ms(stats.ready_wait_ns, dispatched),
                 ns_to_ms(stats.ready_wait_max_ns), avg_ms(stats.dispatch_ns, dispatched),
                 ns_to_ms(stats.dispatch_max_ns), ull(stats.no_credit_skips), stats.credit_inflight,
                 stats.credit_limit, ull(stats.credit_registered),
                 ull(stats.credit_released_by_output), ull(stats.credit_released_without_output),
                 ull(stats.credit_missing_key));
  }
}

void dump_transport_diag(const ExecutionGraphRuntime& execution) {
  for (const auto& pipe : execution.pipelines) {
    if (!pipe) {
      continue;
    }
    const auto& telemetry = pipe->transport.telemetry;
    if (pipe->transport.input_queue) {
      const auto q = pipe->transport.input_queue->stats();
      std::fprintf(stderr,
                   "[GRAPH] transport_queue seg=%zu push=%llu pop=%llu push_timeout=%llu "
                   "pop_timeout=%llu high_water=%zu size=%zu capacity=%zu timing=%d "
                   "avg_queue_residence_ms=%.3f max_queue_residence_ms=%.3f "
                   "avg_push_wait_ms=%.3f max_push_wait_ms=%.3f\n",
                   static_cast<std::size_t>(pipe->seg.id), ull(q.push_count), ull(q.pop_count),
                   ull(q.push_timeout_count), ull(q.pop_timeout_count), q.high_watermark,
                   q.current_size, q.capacity, static_cast<int>(q.timing_enabled),
                   avg_ms(q.residence_ns, q.residence_count), ns_to_ms(q.max_residence_ns),
                   avg_ms(q.push_wait_ns, q.push_count + q.push_timeout_count),
                   ns_to_ms(q.max_push_wait_ns));
    }
    const auto print_counter = [&](const char* name, std::uint64_t calls, std::uint64_t miss,
                                   std::uint64_t total_ns, std::uint64_t max_ns) {
      if (calls == 0 && miss == 0 && total_ns == 0 && max_ns == 0) {
        return;
      }
      std::fprintf(stderr,
                   "[GRAPH] transport_timing seg=%zu name=%s calls=%llu miss=%llu avg_ms=%.3f "
                   "max_ms=%.3f\n",
                   static_cast<std::size_t>(pipe->seg.id), name, ull(calls), ull(miss),
                   avg_ms(total_ns, calls + miss), ns_to_ms(max_ns));
    };
    print_counter("router_input_push",
                  telemetry.router_input_push_calls.load(std::memory_order_relaxed), 0,
                  telemetry.router_input_push_ns.load(std::memory_order_relaxed),
                  telemetry.router_input_push_max_ns.load(std::memory_order_relaxed));
    print_counter("push_thread_pop",
                  telemetry.push_thread_pop_calls.load(std::memory_order_relaxed),
                  telemetry.push_thread_pop_miss.load(std::memory_order_relaxed),
                  telemetry.push_thread_pop_wait_ns.load(std::memory_order_relaxed),
                  telemetry.push_thread_pop_wait_max_ns.load(std::memory_order_relaxed));
    print_counter("push_thread_push_samples",
                  telemetry.push_thread_push_samples_calls.load(std::memory_order_relaxed), 0,
                  telemetry.push_thread_push_samples_ns.load(std::memory_order_relaxed),
                  telemetry.push_thread_push_samples_max_ns.load(std::memory_order_relaxed));
    print_counter("pull_thread_pull",
                  telemetry.pull_thread_pull_calls.load(std::memory_order_relaxed),
                  telemetry.pull_thread_pull_miss.load(std::memory_order_relaxed),
                  telemetry.pull_thread_pull_ns.load(std::memory_order_relaxed),
                  telemetry.pull_thread_pull_max_ns.load(std::memory_order_relaxed));
    print_counter("pull_thread_route",
                  telemetry.pull_thread_route_calls.load(std::memory_order_relaxed), 0,
                  telemetry.pull_thread_route_ns.load(std::memory_order_relaxed),
                  telemetry.pull_thread_route_max_ns.load(std::memory_order_relaxed));
  }
}

void close_fused_encoded_output_sinks(ExecutionGraphRuntime& execution) {
  std::unordered_set<simaai::neat::graph::NodeId> closed;
  for (const auto& segment : execution.plan.pipeline_segments) {
    if (!segment.fused_realtime_ingress.has_value()) {
      continue;
    }
    for (const auto& branch : segment.fused_realtime_ingress->branches) {
      if (!branch.encoded_output.has_value() ||
          branch.encoded_output->sink_node == simaai::neat::graph::kInvalidNode ||
          !closed.insert(branch.encoded_output->sink_node).second) {
        continue;
      }
      const auto sink = execution.sinks.find(branch.encoded_output->sink_node);
      if (sink != execution.sinks.end() && sink->second) {
        sink->second->close();
      }
    }
  }
}

} // namespace

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
  // An EveryFrame encoded tap may be blocked in its source streaming thread
  // waiting for Run::pull() to make room. Close only these auxiliary sink
  // queues before stopping source pipelines so the producer wakes promptly;
  // ordinary graph sinks remain open for the normal close_input() drain.
  close_fused_encoded_output_sinks(execution);
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
    log_decoder_cma_snapshot("before_admission_release");
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
    log_decoder_cma_snapshot(released ? "after_admission_release"
                                      : "after_admission_release_failed");
  }

  graph_signal_stop();

  if (simaai::neat::graph::graph_output_rate_enabled() &&
      !graph_output_rate_reported.exchange(true)) {
    simaai::neat::graph::graph_output_rate_summary(&execution.node_labels, nullptr);
  }
  if (simaai::neat::graph::graph_sched_debug_enabled() && !graph_sched_reported.exchange(true)) {
    simaai::neat::graph::graph_sched_summary(&execution.node_labels);
  }
  if (pipeline_internal::env_bool("SIMA_GRAPH_DIAG_ON_STOP", false)) {
    dump_realtime_link_diag(execution);
    dump_transport_diag(execution);
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

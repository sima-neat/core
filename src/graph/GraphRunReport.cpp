#include "internal/GraphRunState.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace simaai::neat::graph {

namespace {

RuntimeMetricGroup graph_snapshot_group(const GraphRunStats::Snapshot& snap,
                                        const std::string& label) {
  RuntimeMetricGroup group;
  group.name =
      label.empty() ? ("graph_node:" + std::to_string(snap.node_id)) : ("graph_node:" + label);
  const double secs =
      std::chrono::duration_cast<std::chrono::duration<double>>(snap.last - snap.first).count();
  const double fps = (secs > 0.0) ? (static_cast<double>(snap.total) / secs) : 0.0;
  group.values = {
      {"node_id", static_cast<double>(snap.node_id), "id"},
      {"samples", static_cast<double>(snap.total), "count"},
      {"elapsed_s", secs, "s"},
      {"fps", fps, "fps"},
  };
  for (const auto& kv : snap.counts) {
    group.values.push_back({"stream:" + kv.first, static_cast<double>(kv.second), "count"});
  }
  return group;
}

double ns_to_us(std::uint64_t ns) {
  return static_cast<double>(ns) / 1000.0;
}

double avg_ns_to_us(std::uint64_t total_ns, std::uint64_t count) {
  return count > 0 ? ns_to_us(total_ns) / static_cast<double>(count) : 0.0;
}

std::uint64_t load_u64(const std::atomic<std::uint64_t>& v) {
  return v.load(std::memory_order_relaxed);
}

template <typename QueueStats>
void append_queue_stats(RuntimeMetricGroup& group, const std::string& prefix,
                        const QueueStats& q) {
  const std::uint64_t push_attempts = q.push_count + q.push_timeout_count + q.push_closed_count;
  const std::uint64_t pop_attempts = q.pop_count + q.pop_timeout_count + q.pop_closed_empty_count;
  group.values.push_back({prefix + "push_count", static_cast<double>(q.push_count), "count"});
  group.values.push_back({prefix + "pop_count", static_cast<double>(q.pop_count), "count"});
  group.values.push_back(
      {prefix + "push_timeout_count", static_cast<double>(q.push_timeout_count), "count"});
  group.values.push_back(
      {prefix + "pop_timeout_count", static_cast<double>(q.pop_timeout_count), "count"});
  group.values.push_back(
      {prefix + "push_closed_count", static_cast<double>(q.push_closed_count), "count"});
  group.values.push_back(
      {prefix + "pop_closed_empty_count", static_cast<double>(q.pop_closed_empty_count), "count"});
  group.values.push_back(
      {prefix + "avg_push_wait_us", avg_ns_to_us(q.push_wait_ns, push_attempts), "us"});
  group.values.push_back(
      {prefix + "max_push_wait_us", ns_to_us(q.max_push_wait_ns), "us"});
  group.values.push_back(
      {prefix + "avg_pop_wait_us", avg_ns_to_us(q.pop_wait_ns, pop_attempts), "us"});
  group.values.push_back({prefix + "max_pop_wait_us", ns_to_us(q.max_pop_wait_ns), "us"});
  group.values.push_back({prefix + "high_watermark", static_cast<double>(q.high_watermark), "count"});
  group.values.push_back({prefix + "current_size", static_cast<double>(q.current_size), "count"});
  group.values.push_back({prefix + "capacity", static_cast<double>(q.capacity), "count"});
}

void append_graph_runtime_groups(RuntimeMetrics& out,
                                 const simaai::neat::runtime::ExecutionGraphRuntime& execution,
                                 const RuntimeMetricsOptions& opt) {
  for (std::size_t i = 0; i < execution.pipelines.size(); ++i) {
    const auto& pipe_ptr = execution.pipelines[i];
    if (!pipe_ptr)
      continue;
    const auto& pipe = *pipe_ptr;
    const auto& t = pipe.transport.telemetry;

    RuntimeMetricGroup group;
    group.name = "graph_pipeline:" + std::to_string(pipe.seg.id);
    group.values = {
        {"index", static_cast<double>(i), "index"},
        {"segment_id", static_cast<double>(pipe.seg.id), "id"},
        {"built", pipe.transport.built.load(std::memory_order_acquire) ? 1.0 : 0.0, "bool"},
        {"has_input", pipe.transport.has_input ? 1.0 : 0.0, "bool"},
        {"has_output", pipe.transport.has_output ? 1.0 : 0.0, "bool"},
        {"identity_rewrite_count",
         static_cast<double>(pipe.transport.identity_rewrite_count.load(std::memory_order_relaxed)),
         "count"},
        {"identity_map_miss_count",
         static_cast<double>(pipe.transport.identity_map_miss_count.load(std::memory_order_relaxed)),
         "count"},
    };
    if (pipe.transport.input_queue) {
      append_queue_stats(group, "input_queue_", pipe.transport.input_queue->stats());
    }

    const std::uint64_t push_pop_calls = load_u64(t.push_thread_pop_calls);
    const std::uint64_t push_pop_attempts = push_pop_calls + load_u64(t.push_thread_pop_miss);
    const std::uint64_t sanitize_calls = load_u64(t.push_thread_sanitize_calls);
    const std::uint64_t ensure_calls = load_u64(t.push_thread_ensure_build_calls);
    const std::uint64_t push_samples_calls = load_u64(t.push_thread_push_samples_calls);
    const std::uint64_t pull_calls = load_u64(t.pull_thread_pull_calls);
    const std::uint64_t route_calls = load_u64(t.pull_thread_route_calls);
    const std::uint64_t router_ensure_calls = load_u64(t.router_ensure_build_calls);
    const std::uint64_t router_sanitize_calls = load_u64(t.router_sanitize_calls);
    const std::uint64_t router_input_push_calls = load_u64(t.router_input_push_calls);
    const std::uint64_t ensure_build_calls = load_u64(t.ensure_build_calls);

    group.values.push_back(
        {"push_thread_pop_calls", static_cast<double>(push_pop_calls), "count"});
    group.values.push_back(
        {"push_thread_pop_miss", static_cast<double>(load_u64(t.push_thread_pop_miss)), "count"});
    group.values.push_back({"push_thread_avg_pop_wait_us",
                            avg_ns_to_us(load_u64(t.push_thread_pop_wait_ns), push_pop_attempts),
                            "us"});
    group.values.push_back({"push_thread_max_pop_wait_us",
                            ns_to_us(load_u64(t.push_thread_pop_wait_max_ns)), "us"});
    group.values.push_back({"push_thread_avg_sanitize_us",
                            avg_ns_to_us(load_u64(t.push_thread_sanitize_ns), sanitize_calls),
                            "us"});
    group.values.push_back({"push_thread_max_sanitize_us",
                            ns_to_us(load_u64(t.push_thread_sanitize_max_ns)), "us"});
    group.values.push_back({"push_thread_avg_ensure_build_us",
                            avg_ns_to_us(load_u64(t.push_thread_ensure_build_ns), ensure_calls),
                            "us"});
    group.values.push_back({"push_thread_max_ensure_build_us",
                            ns_to_us(load_u64(t.push_thread_ensure_build_max_ns)), "us"});
    group.values.push_back({"push_thread_avg_push_samples_us",
                            avg_ns_to_us(load_u64(t.push_thread_push_samples_ns),
                                         push_samples_calls),
                            "us"});
    group.values.push_back({"push_thread_max_push_samples_us",
                            ns_to_us(load_u64(t.push_thread_push_samples_max_ns)), "us"});
    group.values.push_back({"pull_thread_pull_calls", static_cast<double>(pull_calls), "count"});
    group.values.push_back(
        {"pull_thread_pull_miss", static_cast<double>(load_u64(t.pull_thread_pull_miss)), "count"});
    group.values.push_back(
        {"pull_thread_avg_pull_us", avg_ns_to_us(load_u64(t.pull_thread_pull_ns), pull_calls), "us"});
    group.values.push_back(
        {"pull_thread_max_pull_us", ns_to_us(load_u64(t.pull_thread_pull_max_ns)), "us"});
    group.values.push_back({"pull_thread_avg_route_us",
                            avg_ns_to_us(load_u64(t.pull_thread_route_ns), route_calls), "us"});
    group.values.push_back(
        {"pull_thread_max_route_us", ns_to_us(load_u64(t.pull_thread_route_max_ns)), "us"});
    group.values.push_back({"router_avg_ensure_build_us",
                            avg_ns_to_us(load_u64(t.router_ensure_build_ns),
                                         router_ensure_calls),
                            "us"});
    group.values.push_back({"router_max_ensure_build_us",
                            ns_to_us(load_u64(t.router_ensure_build_max_ns)), "us"});
    group.values.push_back({"router_avg_sanitize_us",
                            avg_ns_to_us(load_u64(t.router_sanitize_ns), router_sanitize_calls),
                            "us"});
    group.values.push_back({"router_max_sanitize_us",
                            ns_to_us(load_u64(t.router_sanitize_max_ns)), "us"});
    group.values.push_back({"router_avg_input_push_us",
                            avg_ns_to_us(load_u64(t.router_input_push_ns),
                                         router_input_push_calls),
                            "us"});
    group.values.push_back({"router_max_input_push_us",
                            ns_to_us(load_u64(t.router_input_push_max_ns)), "us"});
    group.values.push_back({"ensure_build_calls", static_cast<double>(ensure_build_calls), "count"});
    group.values.push_back(
        {"ensure_build_failures", static_cast<double>(load_u64(t.ensure_build_failures)), "count"});
    group.values.push_back({"ensure_build_avg_wait_us",
                            avg_ns_to_us(load_u64(t.ensure_build_wait_ns), ensure_build_calls),
                            "us"});
    group.values.push_back({"ensure_build_max_wait_us",
                            ns_to_us(load_u64(t.ensure_build_wait_max_ns)), "us"});
    group.values.push_back({"ensure_build_avg_canonicalize_us",
                            avg_ns_to_us(load_u64(t.ensure_build_canonicalize_ns),
                                         ensure_build_calls),
                            "us"});
    group.values.push_back({"ensure_build_avg_segment_us",
                            avg_ns_to_us(load_u64(t.ensure_build_segment_ns), ensure_build_calls),
                            "us"});
    group.values.push_back({"ensure_build_avg_total_us",
                            avg_ns_to_us(load_u64(t.ensure_build_total_ns), ensure_build_calls),
                            "us"});
    group.values.push_back(
        {"ensure_build_max_total_us", ns_to_us(load_u64(t.ensure_build_total_max_ns)), "us"});

    if (pipe.run_core) {
      const RunStats stats = pipe.run_core->stats();
      const InputStreamStats input = pipe.run_core->input_stats();
      group.values.push_back(
          {"run_inputs_enqueued", static_cast<double>(stats.inputs_enqueued), "count"});
      group.values.push_back({"run_inputs_pushed", static_cast<double>(stats.inputs_pushed), "count"});
      group.values.push_back(
          {"run_outputs_ready", static_cast<double>(stats.outputs_ready), "count"});
      group.values.push_back(
          {"run_outputs_pulled", static_cast<double>(stats.outputs_pulled), "count"});
      group.values.push_back(
          {"input_stream_push_count", static_cast<double>(input.push_count), "count"});
      group.values.push_back(
          {"input_stream_push_failures", static_cast<double>(input.push_failures), "count"});
      group.values.push_back(
          {"input_stream_pull_count", static_cast<double>(input.pull_count), "count"});
      group.values.push_back({"input_stream_avg_push_us", input.avg_push_us, "us"});
      group.values.push_back({"input_stream_avg_pull_wait_us", input.avg_pull_wait_us, "us"});
    }
    out.groups.push_back(std::move(group));

    if (opt.include_diagnostics && pipe.run_core) {
      const RunDiagSnapshot diag = pipe.run_core->diag_snapshot();
      for (const auto& elem : diag.element_timings) {
        RuntimeMetricGroup elem_group;
        elem_group.name =
            "graph_pipeline:" + std::to_string(pipe.seg.id) + ":element:" + elem.element_name;
        elem_group.values = {
            {"samples", static_cast<double>(elem.samples), "count"},
            {"total_us", static_cast<double>(elem.total_us), "us"},
            {"min_us", static_cast<double>(elem.min_us), "us"},
            {"max_us", static_cast<double>(elem.max_us), "us"},
            {"missed_in", static_cast<double>(elem.missed_in), "count"},
            {"missed_out", static_cast<double>(elem.missed_out), "count"},
        };
        if (elem.samples > 0) {
          elem_group.values.push_back(
              {"avg_us", static_cast<double>(elem.total_us) / static_cast<double>(elem.samples),
               "us"});
        }
        out.groups.push_back(std::move(elem_group));
      }
      for (const auto& pad : diag.element_pad_timings) {
        RuntimeMetricGroup pad_group;
        pad_group.name = "graph_pipeline:" + std::to_string(pipe.seg.id) + ":pad:" +
                         pad.element_name + ":" + pad.pad_name;
        pad_group.values = {
            {"is_sink", pad.is_sink ? 1.0 : 0.0, "bool"},
            {"samples", static_cast<double>(pad.samples), "count"},
            {"inter_arrival_total_us", static_cast<double>(pad.inter_arrival_total_us), "us"},
            {"inter_arrival_max_us", static_cast<double>(pad.inter_arrival_max_us), "us"},
            {"queue_wait_samples", static_cast<double>(pad.queue_wait_samples), "count"},
            {"queue_wait_total_us", static_cast<double>(pad.queue_wait_total_us), "us"},
            {"queue_wait_max_us", static_cast<double>(pad.queue_wait_max_us), "us"},
            {"bytes", static_cast<double>(pad.bytes), "bytes"},
        };
        out.groups.push_back(std::move(pad_group));
      }
    }
  }

  for (std::size_t i = 0; i < execution.stages.size(); ++i) {
    const auto& stage_ptr = execution.stages[i];
    if (!stage_ptr)
      continue;
    const auto& st = *stage_ptr;
    const auto& t = st.telemetry;
    const std::uint64_t pop_calls = load_u64(t.mailbox_pop_calls);
    const std::uint64_t pop_attempts = pop_calls + load_u64(t.mailbox_pop_miss);
    const std::uint64_t exec_calls = load_u64(t.on_input_calls);
    const std::uint64_t route_calls = load_u64(t.route_output_calls);
    std::string label;
    if (st.node_id < execution.node_labels.size()) {
      label = execution.node_labels[st.node_id];
    }
    RuntimeMetricGroup group;
    group.name = label.empty() ? "graph_stage:" + std::to_string(st.node_id)
                               : "graph_stage:" + label;
    group.values = {
        {"index", static_cast<double>(i), "index"},
        {"node_id", static_cast<double>(st.node_id), "id"},
        {"mailbox_pop_calls", static_cast<double>(pop_calls), "count"},
        {"mailbox_pop_miss", static_cast<double>(load_u64(t.mailbox_pop_miss)), "count"},
        {"mailbox_avg_pop_wait_us",
         avg_ns_to_us(load_u64(t.mailbox_pop_wait_ns), pop_attempts), "us"},
        {"mailbox_max_pop_wait_us", ns_to_us(load_u64(t.mailbox_pop_wait_max_ns)), "us"},
        {"avg_on_input_us", avg_ns_to_us(load_u64(t.on_input_ns), exec_calls), "us"},
        {"max_on_input_us", ns_to_us(load_u64(t.on_input_max_ns)), "us"},
        {"avg_route_output_us", avg_ns_to_us(load_u64(t.route_output_ns), route_calls), "us"},
        {"max_route_output_us", ns_to_us(load_u64(t.route_output_max_ns)), "us"},
    };
    append_queue_stats(group, "mailbox_", st.mailbox.inbox.stats());
    out.groups.push_back(std::move(group));
  }

  for (const auto& sink : execution.sinks) {
    if (!sink.second)
      continue;
    RuntimeMetricGroup group;
    group.name = "graph_sink:" + std::to_string(sink.first);
    if (sink.first < execution.node_labels.size()) {
      group.name += ":" + execution.node_labels[sink.first];
    }
    append_queue_stats(group, "", sink.second->stats());
    out.groups.push_back(std::move(group));
  }
}

} // namespace

void GraphRun::emit_rate_summary(const GraphRunStats& stats) const {
  const auto snaps = stats.snapshot();
  if (snaps.empty())
    return;
  const auto now = std::chrono::steady_clock::now();
  for (const auto& snap : snaps) {
    std::string label;
    if (state_ && snap.node_id < state_->execution().node_labels.size()) {
      label = state_->execution().node_labels[snap.node_id];
    }
    const double secs =
        std::chrono::duration_cast<std::chrono::duration<double>>(snap.last - snap.first).count();
    const double fps = (secs > 0.0) ? (static_cast<double>(snap.total) / secs) : 0.0;
    std::cout << "[graph_rate] node=" << snap.node_id;
    if (!label.empty()) {
      std::cout << " label=" << label;
    }
    std::cout << " total=" << snap.total << " secs=" << secs << " fps=" << fps << "\n";
    for (const auto& it : snap.counts) {
      const auto last_it = snap.last_seen.find(it.first);
      const int age_ms =
          (last_it == snap.last_seen.end())
              ? -1
              : static_cast<int>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - last_it->second)
                        .count());
      std::cout << "[graph_rate] node=" << snap.node_id << " stream=" << it.first
                << " count=" << it.second << " age_ms=" << age_ms << "\n";
    }
  }
}

void GraphRun::emit_rate_summary() const {
  const GraphRunStats* graph_stats = stats();
  if (!graph_stats)
    return;
  emit_rate_summary(*graph_stats);
}

void GraphRun::emit_stream_summary(const GraphRunStats& stats) const {
  const auto snaps = stats.snapshot();
  if (snaps.empty())
    return;
  for (const auto& snap : snaps) {
    std::string label;
    if (state_ && snap.node_id < state_->execution().node_labels.size()) {
      label = state_->execution().node_labels[snap.node_id];
    }
    if (snap.counts.empty())
      continue;
    int64_t min_count = std::numeric_limits<int64_t>::max();
    int64_t max_count = 0;
    double sum = 0.0;
    for (const auto& kv : snap.counts) {
      min_count = std::min(min_count, kv.second);
      max_count = std::max(max_count, kv.second);
      sum += static_cast<double>(kv.second);
    }
    const double avg = sum / static_cast<double>(snap.counts.size());
    std::cout << "[graph_stream_summary] node=" << snap.node_id;
    if (!label.empty()) {
      std::cout << " label=" << label;
    }
    std::cout << " streams=" << snap.counts.size() << " min=" << min_count << " max=" << max_count
              << " avg=" << avg << "\n";
  }
}

void GraphRun::emit_stream_summary() const {
  const GraphRunStats* graph_stats = stats();
  if (!graph_stats)
    return;
  emit_stream_summary(*graph_stats);
}

void GraphRun::emit_summary(const GraphRunStats& stats) const {
  emit_rate_summary(stats);
  emit_stream_summary(stats);
}

void GraphRun::emit_summary() const {
  const GraphRunStats* graph_stats = stats();
  if (!graph_stats)
    return;
  emit_summary(*graph_stats);
}

RuntimeMetrics GraphRun::metrics(const RuntimeMetricsOptions& opt) const {
  RuntimeMetrics out;
  out.source_kind = "graph";
  if (!state_)
    return out;
  if (opt.include_pipeline) {
    out.metadata.emplace_back("graph", describe());
  }
  if (opt.include_power && state_->core && state_->core->power_monitor) {
    out.power = state_->core->power_monitor->summary();
  }

  const GraphRunStats* graph_stats = stats();
  if (!graph_stats)
    return out;

  const auto snaps = graph_stats->snapshot();
  std::chrono::steady_clock::time_point first{};
  std::chrono::steady_clock::time_point last{};
  std::int64_t total_events = 0;
  for (const auto& snap : snaps) {
    if (snap.total <= 0)
      continue;
    if (first == std::chrono::steady_clock::time_point{} || snap.first < first)
      first = snap.first;
    if (last == std::chrono::steady_clock::time_point{} || snap.last > last)
      last = snap.last;
    total_events += snap.total;

    std::string label;
    if (snap.node_id < state_->execution().node_labels.size()) {
      label = state_->execution().node_labels[snap.node_id];
    }
    out.groups.push_back(graph_snapshot_group(snap, label));
  }
  append_graph_runtime_groups(out, state_->execution(), opt);
  out.counters.outputs_ready = static_cast<std::uint64_t>(std::max<std::int64_t>(0, total_events));
  out.counters.outputs_pulled = out.counters.outputs_ready;
  if (first != std::chrono::steady_clock::time_point{} && last > first) {
    out.elapsed_seconds = std::chrono::duration<double>(last - first).count();
    if (out.elapsed_seconds > 0.0) {
      out.throughput_fps = static_cast<double>(total_events) / out.elapsed_seconds;
    }
  }
  return out;
}

std::string GraphRun::metrics_report(const RuntimeMetricsOptions& opt,
                                     RuntimeMetricsFormat format) const {
  return format_runtime_metrics(metrics(opt), format);
}

std::string GraphRun::metrics_report(RuntimeMetricsFormat format) const {
  return metrics_report(RuntimeMetricsOptions{}, format);
}

std::string GraphRun::describe() const {
  if (!state_)
    return {};
  std::ostringstream oss;

  auto port_name = [&](PortId id) -> std::string {
    if (id == kInvalidPort)
      return "auto";
    if (id < state_->execution().plan.port_names.size())
      return state_->execution().plan.port_names[id];
    return "port" + std::to_string(id);
  };

  oss << "pipelines=" << state_->execution().pipelines.size()
      << " stages=" << state_->execution().stages.size()
      << " stage_groups=" << state_->execution().stage_groups.size() << "\n";

  for (const auto& pipe : state_->execution().pipelines) {
    if (!pipe)
      continue;
    oss << "pipeline[" << pipe->seg.id << "] nodes=" << pipe->seg.node_ids.size()
        << " source=" << (pipe->seg.boundary.source_like ? "true" : "false")
        << " in_edges=" << pipe->seg.input_edges.size()
        << " out_edges=" << pipe->seg.output_edges.size() << "\n";
  }

  for (const auto& st : state_->execution().stages) {
    if (!st)
      continue;
    oss << "stage node=" << st->node_id << " in_ports=" << st->input_ports.size()
        << " out_ports=" << st->output_ports.size() << "\n";
  }

  for (std::size_t i = 0; i < state_->execution().plan.edges.size(); ++i) {
    const auto& e = state_->execution().plan.edges[i];
    oss << "edge[" << i << "] " << e.from << ":" << port_name(e.from_port) << " -> " << e.to << ":"
        << port_name(e.to_port);
    oss << " spec=" << (e.spec_complete ? "complete" : "partial");
    oss << "\n";
  }

  return oss.str();
}

} // namespace simaai::neat::graph

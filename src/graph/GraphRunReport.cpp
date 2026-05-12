#include "internal/GraphRunState.h"

#include <algorithm>
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

} // namespace

void GraphRun::emit_rate_summary(const GraphRunStats& stats) const {
  const auto snaps = stats.snapshot();
  if (snaps.empty())
    return;
  const auto now = std::chrono::steady_clock::now();
  for (const auto& snap : snaps) {
    std::string label;
    if (state_ && snap.node_id < state_->node_labels.size()) {
      label = state_->node_labels[snap.node_id];
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
  if (!state_ || !state_->stats)
    return;
  emit_rate_summary(*state_->stats);
}

void GraphRun::emit_stream_summary(const GraphRunStats& stats) const {
  const auto snaps = stats.snapshot();
  if (snaps.empty())
    return;
  for (const auto& snap : snaps) {
    std::string label;
    if (state_ && snap.node_id < state_->node_labels.size()) {
      label = state_->node_labels[snap.node_id];
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
  if (!state_ || !state_->stats)
    return;
  emit_stream_summary(*state_->stats);
}

void GraphRun::emit_summary(const GraphRunStats& stats) const {
  emit_rate_summary(stats);
  emit_stream_summary(stats);
}

void GraphRun::emit_summary() const {
  if (!state_ || !state_->stats)
    return;
  emit_summary(*state_->stats);
}

RuntimeMetrics GraphRun::metrics(const RuntimeMetricsOptions& opt) const {
  RuntimeMetrics out;
  out.source_kind = "graph";
  if (!state_)
    return out;
  if (opt.include_pipeline) {
    out.metadata.emplace_back("graph", describe());
  }
  if (opt.include_power && state_->power_monitor) {
    out.power = state_->power_monitor->summary();
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
    if (snap.node_id < state_->node_labels.size()) {
      label = state_->node_labels[snap.node_id];
    }
    out.groups.push_back(graph_snapshot_group(snap, label));
  }
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
    if (id < state_->compiled.port_names.size())
      return state_->compiled.port_names[id];
    return "port" + std::to_string(id);
  };

  oss << "pipelines=" << state_->pipelines.size() << " stages=" << state_->stages.size()
      << " stage_groups=" << state_->stage_groups.size() << "\n";

  for (const auto& pipe : state_->pipelines) {
    if (!pipe)
      continue;
    oss << "pipeline[" << pipe->seg.id << "] nodes=" << pipe->seg.node_ids.size()
        << " source=" << (pipe->seg.source_like ? "true" : "false")
        << " in_edges=" << pipe->seg.input_edges.size()
        << " out_edges=" << pipe->seg.output_edges.size() << "\n";
  }

  for (const auto& st : state_->stages) {
    if (!st)
      continue;
    oss << "stage node=" << st->node_id << " in_ports=" << st->input_ports.size()
        << " out_ports=" << st->output_ports.size() << "\n";
  }

  for (std::size_t i = 0; i < state_->compiled.edges.size(); ++i) {
    const Edge& e = state_->compiled.edges[i];
    oss << "edge[" << i << "] " << e.from << ":" << port_name(e.from_port) << " -> " << e.to << ":"
        << port_name(e.to_port);
    if (i < state_->compiled.edge_specs.size()) {
      const auto& es = state_->compiled.edge_specs[i];
      oss << " spec=" << (es.complete ? "complete" : "partial");
    }
    oss << "\n";
  }

  return oss.str();
}

} // namespace simaai::neat::graph

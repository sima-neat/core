#include "internal/GraphRunState.h"

namespace simaai::neat::graph {

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

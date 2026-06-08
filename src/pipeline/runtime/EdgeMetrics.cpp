#include "EdgeMetrics.h"

#include "pipeline/runtime/ExecutionGraphRuntime.h"
#include "pipeline/runtime/RunCore.h"
#include "pipeline/runtime/RunInternal.h"

#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>

namespace simaai::neat::pipeline_internal {
namespace {

template <typename T> T delta_floor(T before, T after) {
  return after >= before ? after - before : after;
}

double us_to_ms(std::uint64_t us) {
  return static_cast<double>(us) / 1000.0;
}
double ns_to_ms(std::uint64_t ns) {
  return static_cast<double>(ns) / 1000000.0;
}

using PadKey = std::tuple<std::string, std::string, bool>;

PadKey key_for(const RunElementPadTimingStats& s) {
  return {s.element_name, s.pad_name, s.is_sink};
}

std::string pad_edge_id(const RunElementPadTimingStats& s) {
  return s.element_name + ":" + s.pad_name + (s.is_sink ? ":sink" : ":src");
}

void append_pad_metric(const RunElementPadTimingStats& before,
                       const RunElementPadTimingStats& after,
                       std::vector<MeasureEdgeLatency>* out) {
  const std::uint64_t samples = delta_floor(before.samples, after.samples);
  const std::uint64_t queue_samples =
      delta_floor(before.queue_wait_samples, after.queue_wait_samples);
  const std::uint64_t queue_total_us =
      delta_floor(before.queue_wait_total_us, after.queue_wait_total_us);
  if (samples == 0 && queue_samples == 0 && queue_total_us == 0) {
    return;
  }
  // Inter-arrival is cadence, not transport latency.  The edge table should show
  // upstream src departure -> downstream sink arrival.  Src pad probes stamp that departure on the
  // buffer; sink pad probes aggregate it as queue_wait_*.
  if (!after.is_sink || queue_samples == 0) {
    return;
  }

  MeasureEdgeLatency metric;
  metric.from_element_name = after.transport_from_element_name;
  metric.to_element_name = !after.transport_to_element_name.empty()
                               ? after.transport_to_element_name
                               : after.element_name;
  metric.edge_id =
      (!metric.from_element_name.empty() && !metric.to_element_name.empty())
          ? metric.from_element_name + "->" + metric.to_element_name + ":" + after.pad_name
          : pad_edge_id(after);
  metric.name = (!metric.from_element_name.empty() && !metric.to_element_name.empty())
                    ? metric.from_element_name + " -> " + metric.to_element_name
                    : after.element_name + "/" + after.pad_name;
  metric.source = "diagnostics";
  metric.timing_semantics = "edge_transport";
  metric.attribution_source = "src_depart_to_sink_arrival";
  metric.samples = queue_samples;
  metric.total_ms = us_to_ms(queue_total_us);
  metric.avg_ms = metric.samples > 0 ? metric.total_ms / static_cast<double>(metric.samples) : 0.0;
  metric.min_ms = 0.0;
  // Max is a cumulative high-water mark; it cannot be window-delta-subtracted like samples/total.
  metric.max_ms = us_to_ms(after.queue_wait_max_us);
  metric.p50_ms = 0.0;
  metric.p95_ms = 0.0;
  metric.non_additive = true;
  metric.reliable = true;
  out->push_back(std::move(metric));
}

void append_boundary_metric(const BoundaryFlowStats& before, const BoundaryFlowStats& after,
                            std::vector<MeasureEdgeLatency>* out) {
  const std::uint64_t in_delta = delta_floor(before.in_buffers, after.in_buffers);
  const std::uint64_t out_delta = delta_floor(before.out_buffers, after.out_buffers);
  if (in_delta == 0 && out_delta == 0) {
    return;
  }

  MeasureEdgeLatency metric;
  metric.edge_id = after.boundary_name;
  metric.name = after.boundary_name;
  metric.from_runtime_node_id = after.after_node_index;
  metric.to_runtime_node_id = after.before_node_index;
  metric.from_node_id =
      after.after_node_index >= 0 ? "n" + std::to_string(after.after_node_index) : "";
  metric.to_node_id =
      after.before_node_index >= 0 ? "n" + std::to_string(after.before_node_index) : "";
  metric.source = "diagnostics";
  metric.timing_semantics = "boundary_flow";
  metric.attribution_source = "graph_boundary";
  metric.samples = std::max(in_delta, out_delta);
  if (after.last_in_wall_us > 0 && after.last_out_wall_us >= after.last_in_wall_us) {
    metric.total_ms =
        us_to_ms(static_cast<std::uint64_t>(after.last_out_wall_us - after.last_in_wall_us));
    metric.avg_ms = metric.total_ms;
    metric.max_ms = metric.total_ms;
  }
  metric.non_additive = true;
  metric.reliable = true;
  out->push_back(std::move(metric));
}

std::string runtime_node_name(graph::NodeId node) {
  return node == graph::kInvalidNode ? std::string() : "n" + std::to_string(node);
}

std::string port_suffix(const runtime::ExecutionGraphPlan& plan, graph::PortId from_port,
                        graph::PortId to_port) {
  auto port_name = [&](graph::PortId port) -> std::string {
    if (port == graph::kInvalidPort || port >= plan.port_names.size()) {
      return {};
    }
    return plan.port_names[port];
  };
  const std::string from = port_name(from_port);
  const std::string to = port_name(to_port);
  if (from.empty() && to.empty()) {
    return {};
  }
  return "[" + (from.empty() ? "-" : from) + "->" + (to.empty() ? "-" : to) + "]";
}

GraphQueueLatencySnapshot
queue_snapshot_from_stats(std::string edge_id, std::string name, std::string queue_kind,
                          std::size_t pipeline_segment_id, graph::NodeId from, graph::NodeId to,
                          std::uint64_t residence_count, std::uint64_t residence_ns,
                          std::uint64_t max_residence_ns, bool timing_enabled) {
  GraphQueueLatencySnapshot snapshot;
  snapshot.edge_id = std::move(edge_id);
  snapshot.name = std::move(name);
  snapshot.queue_kind = std::move(queue_kind);
  snapshot.pipeline_segment_id = pipeline_segment_id;
  snapshot.from_runtime_node_id =
      from == graph::kInvalidNode ? -1 : static_cast<std::int32_t>(from);
  snapshot.to_runtime_node_id = to == graph::kInvalidNode ? -1 : static_cast<std::int32_t>(to);
  snapshot.residence_count = residence_count;
  snapshot.residence_ns = residence_ns;
  snapshot.max_residence_ns = max_residence_ns;
  snapshot.timing_enabled = timing_enabled;
  return snapshot;
}

std::vector<std::size_t> incoming_edges_for_node(const runtime::ExecutionGraphPlan& plan,
                                                 graph::NodeId node) {
  std::vector<std::size_t> out;
  for (std::size_t i = 0; i < plan.edges.size(); ++i) {
    if (plan.edges[i].to == node) {
      out.push_back(i);
    }
  }
  return out;
}

} // namespace

std::vector<MeasureEdgeLatency> build_edge_latency_from_diag_delta(const RunDiagSnapshot& before,
                                                                   const RunDiagSnapshot& after) {
  std::vector<MeasureEdgeLatency> out;

  std::map<PadKey, RunElementPadTimingStats> before_pad;
  for (const RunElementPadTimingStats& s : before.element_pad_timings) {
    before_pad[key_for(s)] = s;
  }
  for (const RunElementPadTimingStats& s : after.element_pad_timings) {
    const auto it = before_pad.find(key_for(s));
    append_pad_metric(it == before_pad.end() ? RunElementPadTimingStats{} : it->second, s, &out);
  }

  std::map<std::string, BoundaryFlowStats> before_boundaries;
  for (const BoundaryFlowStats& s : before.boundaries) {
    before_boundaries[s.boundary_name] = s;
  }
  for (const BoundaryFlowStats& s : after.boundaries) {
    const auto it = before_boundaries.find(s.boundary_name);
    append_boundary_metric(it == before_boundaries.end() ? BoundaryFlowStats{} : it->second, s,
                           &out);
  }
  return out;
}

std::vector<GraphQueueLatencySnapshot> snapshot_graph_queue_latencies(const Run& run) {
  std::vector<GraphQueueLatencySnapshot> out;
  const std::shared_ptr<const runtime::RunCore> core = run_internal::core(run);
  if (!core || !core->graph_execution_) {
    return out;
  }
  const runtime::ExecutionGraphRuntime& execution = core->graph_execution();
  const auto& plan = execution.plan;

  for (const auto& pipe : execution.pipelines) {
    if (!pipe || !pipe->transport.input_queue) {
      continue;
    }
    const auto stats = pipe->transport.input_queue->stats();
    if (pipe->seg.input_edges.size() == 1U && pipe->seg.input_edges.front() < plan.edges.size()) {
      const std::size_t edge_index = pipe->seg.input_edges.front();
      const auto& edge = plan.edges[edge_index];
      const std::string edge_id = "e" + std::to_string(edge_index);
      const std::string name = runtime_node_name(edge.from) + "->" + runtime_node_name(edge.to) +
                               port_suffix(plan, edge.from_port, edge.to_port);
      out.push_back(queue_snapshot_from_stats(
          edge_id, name, "pipeline_input_queue", pipe->seg.id, edge.from, edge.to,
          stats.residence_count, stats.residence_ns, stats.max_residence_ns, stats.timing_enabled));
    } else {
      const graph::NodeId to =
          pipe->seg.node_ids.empty() ? graph::kInvalidNode : pipe->seg.node_ids.front();
      std::ostringstream edge_id;
      edge_id << "seg" << pipe->seg.id << ":input_queue";
      std::ostringstream name;
      name << "segment " << pipe->seg.id << " input queue";
      out.push_back(queue_snapshot_from_stats(
          edge_id.str(), name.str(), "pipeline_input_queue", pipe->seg.id, graph::kInvalidNode, to,
          stats.residence_count, stats.residence_ns, stats.max_residence_ns, stats.timing_enabled));
    }
  }

  for (std::size_t i = 0; i < execution.stages.size(); ++i) {
    const auto& stage = execution.stages[i];
    if (!stage) {
      continue;
    }
    const auto stats = stage->inbox.stats();
    const auto incoming = incoming_edges_for_node(plan, stage->node_id);
    if (incoming.size() == 1U && incoming.front() < plan.edges.size()) {
      const std::size_t edge_index = incoming.front();
      const auto& edge = plan.edges[edge_index];
      out.push_back(queue_snapshot_from_stats(
          "e" + std::to_string(edge_index),
          runtime_node_name(edge.from) + "->" + runtime_node_name(edge.to) +
              port_suffix(plan, edge.from_port, edge.to_port),
          "stage_mailbox", static_cast<std::size_t>(-1), edge.from, edge.to, stats.residence_count,
          stats.residence_ns, stats.max_residence_ns, stats.timing_enabled));
    } else {
      out.push_back(queue_snapshot_from_stats(
          "stage" + std::to_string(stage->node_id) + ":mailbox:" + std::to_string(i),
          "stage " + runtime_node_name(stage->node_id) + " mailbox", "stage_mailbox",
          static_cast<std::size_t>(-1), graph::kInvalidNode, stage->node_id, stats.residence_count,
          stats.residence_ns, stats.max_residence_ns, stats.timing_enabled));
    }
  }

  for (const auto& [sink_node, queue] : execution.sinks) {
    if (!queue) {
      continue;
    }
    const auto stats = queue->stats();
    const auto incoming = incoming_edges_for_node(plan, sink_node);
    if (incoming.size() == 1U && incoming.front() < plan.edges.size()) {
      const std::size_t edge_index = incoming.front();
      const auto& edge = plan.edges[edge_index];
      out.push_back(queue_snapshot_from_stats(
          "e" + std::to_string(edge_index),
          runtime_node_name(edge.from) + "->" + runtime_node_name(edge.to) +
              port_suffix(plan, edge.from_port, edge.to_port),
          "graph_sink_queue", static_cast<std::size_t>(-1), edge.from, edge.to,
          stats.residence_count, stats.residence_ns, stats.max_residence_ns, stats.timing_enabled));
    } else {
      out.push_back(queue_snapshot_from_stats(
          "sink" + std::to_string(sink_node) + ":queue",
          "sink " + runtime_node_name(sink_node) + " queue", "graph_sink_queue",
          static_cast<std::size_t>(-1), graph::kInvalidNode, sink_node, stats.residence_count,
          stats.residence_ns, stats.max_residence_ns, stats.timing_enabled));
    }
  }
  return out;
}

std::vector<MeasureEdgeLatency>
build_graph_queue_latency_delta(const std::vector<GraphQueueLatencySnapshot>& before,
                                const std::vector<GraphQueueLatencySnapshot>& after) {
  std::map<std::string, GraphQueueLatencySnapshot> before_by_edge;
  for (const auto& snapshot : before) {
    before_by_edge[snapshot.edge_id + "\x1f" + snapshot.queue_kind] = snapshot;
  }

  std::vector<MeasureEdgeLatency> out;
  for (const auto& snapshot : after) {
    const std::string key = snapshot.edge_id + "\x1f" + snapshot.queue_kind;
    const auto it = before_by_edge.find(key);
    const GraphQueueLatencySnapshot empty;
    const GraphQueueLatencySnapshot& base = it == before_by_edge.end() ? empty : it->second;
    const std::uint64_t samples = delta_floor(base.residence_count, snapshot.residence_count);
    const std::uint64_t total_ns = delta_floor(base.residence_ns, snapshot.residence_ns);
    if (samples == 0 && total_ns == 0) {
      continue;
    }

    MeasureEdgeLatency metric;
    metric.edge_id = snapshot.edge_id;
    metric.name = snapshot.name.empty() ? snapshot.edge_id : snapshot.name;
    metric.from_runtime_node_id = snapshot.from_runtime_node_id;
    metric.to_runtime_node_id = snapshot.to_runtime_node_id;
    metric.from_node_id = snapshot.from_runtime_node_id >= 0
                              ? "n" + std::to_string(snapshot.from_runtime_node_id)
                              : "";
    metric.to_node_id =
        snapshot.to_runtime_node_id >= 0 ? "n" + std::to_string(snapshot.to_runtime_node_id) : "";
    metric.source = "diagnostics";
    metric.timing_semantics = "queue_residence";
    metric.attribution_source = snapshot.queue_kind;
    metric.samples = samples;
    metric.total_ms = ns_to_ms(total_ns);
    metric.avg_ms = samples > 0 ? metric.total_ms / static_cast<double>(samples) : 0.0;
    // Max is a queue lifetime high-water mark, because BlockingQueue currently stores cumulative
    // max only. Keep it visible but leave percentile fields unset.
    metric.max_ms = ns_to_ms(snapshot.max_residence_ns);
    metric.non_additive = true;
    metric.reliable = true;
    out.push_back(std::move(metric));
  }
  return out;
}

void set_graph_queue_timing_enabled(const Run& run, bool enabled) {
  const std::shared_ptr<const runtime::RunCore> core = run_internal::core(run);
  if (!core || !core->graph_execution_) {
    return;
  }
  const auto level = enabled ? graph::runtime::QueueTelemetryLevel::Timing
                             : graph::runtime::QueueTelemetryLevel::CountersOnly;
  const runtime::ExecutionGraphRuntime& execution = core->graph_execution();
  for (const auto& pipe : execution.pipelines) {
    if (pipe && pipe->transport.input_queue) {
      pipe->transport.input_queue->set_telemetry_level(level);
    }
  }
  for (const auto& stage : execution.stages) {
    if (stage) {
      stage->inbox.set_telemetry_level(level);
    }
  }
  for (const auto& [node, queue] : execution.sinks) {
    (void)node;
    if (queue) {
      queue->set_telemetry_level(level);
    }
  }
}

void attribute_edge_latency_to_nodes(const std::vector<GraphNodeMetrics>& nodes,
                                     std::vector<MeasureEdgeLatency>* edge_metrics,
                                     std::vector<MeasureEdgeLatency>* unattributed) {
  auto map_one = [&](std::vector<MeasureEdgeLatency>* metrics) {
    if (!metrics) {
      return;
    }

    std::unordered_map<std::string, const GraphNodeMetrics*> by_element;
    for (const GraphNodeMetrics& node : nodes) {
      for (const std::string& element : node.element_names) {
        if (!element.empty()) {
          by_element[element] = &node;
        }
      }
      for (const GraphElementMetrics& element : node.elements) {
        if (!element.name.empty()) {
          by_element[element.name] = &node;
        }
      }
    }

    for (MeasureEdgeLatency& edge : *metrics) {
      bool mapped_from = edge.from_runtime_node_id >= 0 || edge.from_element_name.empty();
      bool mapped_to = edge.to_runtime_node_id >= 0 || edge.to_element_name.empty();

      if (edge.from_runtime_node_id < 0 && !edge.from_element_name.empty()) {
        const auto it = by_element.find(edge.from_element_name);
        if (it != by_element.end() && it->second) {
          edge.from_runtime_node_id = static_cast<std::int32_t>(it->second->runtime_node_id);
          edge.from_node_id = it->second->node_id;
          mapped_from = true;
        }
      }
      if (edge.to_runtime_node_id < 0 && !edge.to_element_name.empty()) {
        const auto it = by_element.find(edge.to_element_name);
        if (it != by_element.end() && it->second) {
          edge.to_runtime_node_id = static_cast<std::int32_t>(it->second->runtime_node_id);
          edge.to_node_id = it->second->node_id;
          mapped_to = true;
        }
      }
      if ((!mapped_from || !mapped_to) && edge.mapping_error.empty()) {
        edge.mapping_error = "edge endpoint element did not match exported node metrics";
      }
      if (!edge.from_node_id.empty() && !edge.to_node_id.empty() &&
          edge.timing_semantics == "edge_transport") {
        edge.name = edge.from_node_id + " -> " + edge.to_node_id;
      }
    }
  };

  map_one(edge_metrics);
  map_one(unattributed);
}

} // namespace simaai::neat::pipeline_internal

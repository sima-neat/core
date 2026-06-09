#include "pipeline/GraphMetrics.h"

#include "pipeline/Run.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/runtime/RunCore.h"
#include "RunInternal.h"

#include <algorithm>
#include <map>
#include <memory>
#include <chrono>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

namespace simaai::neat {
namespace {

using runtime::MaterializedNodeAttribution;

std::string lowered_node_id(graph::NodeId id) {
  if (id == graph::kInvalidNode) {
    return {};
  }
  return "n" + std::to_string(static_cast<std::size_t>(id));
}

struct ElementTimingSnapshot {
  std::uint64_t samples = 0;
  std::uint64_t total_us = 0;
  std::uint64_t min_us = 0;
  std::uint64_t max_us = 0;
};

std::unordered_map<std::string, ElementTimingSnapshot>
snapshot_element_timings(const std::shared_ptr<pipeline_internal::DiagCtx>& diag) {
  std::unordered_map<std::string, ElementTimingSnapshot> out;
  if (!diag) {
    return out;
  }
  for (const auto& counter : diag->element_timings) {
    if (!counter) {
      continue;
    }
    const auto snap = counter->snapshot();
    ElementTimingSnapshot& dst = out[snap.element_name];
    dst.samples += snap.samples;
    dst.total_us += snap.total_us;
    if (snap.samples > 0) {
      if (dst.min_us == 0 || snap.min_us < dst.min_us) {
        dst.min_us = snap.min_us;
      }
      dst.max_us = std::max(dst.max_us, snap.max_us);
    }
  }
  return out;
}

struct NodeAccumulator {
  GraphNodeMetrics metrics;
  std::set<std::string> elements_seen;
  std::uint64_t total_us = 0;
  std::uint64_t max_samples = 0;
};

void finalize_node_accumulator(NodeAccumulator& acc) {
  acc.metrics.latency.samples = acc.max_samples;
  acc.metrics.latency.total_ms = static_cast<double>(acc.total_us) / 1000.0;
  if (acc.max_samples > 0) {
    acc.metrics.latency.avg_ms =
        static_cast<double>(acc.total_us) / static_cast<double>(acc.max_samples) / 1000.0;
    if (acc.metrics.elements.size() == 1U) {
      acc.metrics.latency.min_ms = acc.metrics.elements.front().latency.min_ms;
      acc.metrics.latency.max_ms = acc.metrics.elements.front().latency.max_ms;
      acc.metrics.latency.min_max_available =
          acc.metrics.elements.front().latency.min_max_available;
    } else {
      // Summing per-element min/max is not a valid node min/max when a node expands to
      // multiple GStreamer elements. Keep total/avg exact and mark min/max unavailable.
      acc.metrics.latency.min_ms = 0.0;
      acc.metrics.latency.max_ms = 0.0;
      acc.metrics.latency.min_max_available = false;
    }
  } else {
    acc.metrics.latency.min_max_available = false;
  }
}

void add_node_report_to_accumulator(
    std::map<std::pair<std::size_t, graph::NodeId>, NodeAccumulator>& nodes, std::size_t segment_id,
    graph::NodeId runtime_node, const std::string& label, const NodeReport& node_report,
    const std::unordered_map<std::string, ElementTimingSnapshot>& element_timings) {
  if (runtime_node == graph::kInvalidNode) {
    return;
  }
  auto key = std::make_pair(segment_id, runtime_node);
  NodeAccumulator& acc = nodes[key];
  acc.metrics.pipeline_segment_id = segment_id;
  acc.metrics.runtime_node_id = runtime_node;
  acc.metrics.node_id = lowered_node_id(runtime_node);
  if (acc.metrics.label.empty()) {
    acc.metrics.label = !label.empty() ? label : node_report.user_label;
  }
  if (acc.metrics.kind.empty()) {
    acc.metrics.kind = node_report.kind;
  }

  for (const std::string& element : node_report.elements) {
    if (element.empty() || !acc.elements_seen.insert(element).second) {
      continue;
    }
    acc.metrics.element_names.push_back(element);
    auto it = element_timings.find(element);
    if (it == element_timings.end()) {
      continue;
    }
    const ElementTimingSnapshot& timing = it->second;
    GraphElementMetrics element_metric;
    element_metric.name = element;
    element_metric.latency.samples = timing.samples;
    element_metric.latency.total_ms = static_cast<double>(timing.total_us) / 1000.0;
    if (timing.samples > 0) {
      element_metric.latency.avg_ms =
          static_cast<double>(timing.total_us) / static_cast<double>(timing.samples) / 1000.0;
      element_metric.latency.min_ms = static_cast<double>(timing.min_us) / 1000.0;
      element_metric.latency.max_ms = static_cast<double>(timing.max_us) / 1000.0;
    } else {
      element_metric.latency.min_max_available = false;
    }
    acc.metrics.elements.push_back(std::move(element_metric));
    acc.total_us += timing.total_us;
    if (timing.samples > 0) {
      acc.max_samples = std::max(acc.max_samples, timing.samples);
    }
  }
}

void collect_linear_node_metrics(
    const std::shared_ptr<const runtime::RunCore>& core,
    std::map<std::pair<std::size_t, graph::NodeId>, NodeAccumulator>& nodes) {
  if (!core) {
    return;
  }
  const auto diag = core->pipeline.stream.diag_ctx();
  if (!diag) {
    return;
  }
  const auto element_timings = snapshot_element_timings(diag);
  constexpr std::size_t kLinearSegment = 0;
  for (const NodeReport& node_report : diag->node_reports) {
    const graph::NodeId runtime_node = node_report.index >= 0
                                           ? static_cast<graph::NodeId>(node_report.index)
                                           : graph::kInvalidNode;
    add_node_report_to_accumulator(nodes, kLinearSegment, runtime_node, node_report.user_label,
                                   node_report, element_timings);
  }
}

void collect_graph_node_metrics(
    const std::shared_ptr<const runtime::RunCore>& core,
    std::map<std::pair<std::size_t, graph::NodeId>, NodeAccumulator>& nodes) {
  if (!core || !core->graph_execution_) {
    return;
  }
  const runtime::ExecutionGraphRuntime& execution = core->graph_execution();
  for (const auto& pipe : execution.pipelines) {
    if (!pipe || !pipe->run_core) {
      continue;
    }
    const auto diag = pipe->run_core->pipeline.stream.diag_ctx();
    if (!diag) {
      continue;
    }
    const auto element_timings = snapshot_element_timings(diag);
    const std::size_t segment_id = pipe->seg.id;
    const auto& mapping = pipe->seg.materialized_node_attribution;
    for (const NodeReport& node_report : diag->node_reports) {
      if (node_report.index < 0 || static_cast<std::size_t>(node_report.index) >= mapping.size()) {
        continue;
      }
      const MaterializedNodeAttribution& attr =
          mapping[static_cast<std::size_t>(node_report.index)];
      if (attr.role != MaterializedNodeAttribution::Role::SegmentNode) {
        continue;
      }
      std::string label;
      if (attr.runtime_node < execution.node_labels.size()) {
        label = execution.node_labels[attr.runtime_node];
      }
      add_node_report_to_accumulator(nodes, segment_id, attr.runtime_node, label, node_report,
                                     element_timings);
    }
  }
}

void attach_public_node_ids(
    const runtime::ExecutionGraphPlan& plan,
    std::map<std::pair<std::size_t, graph::NodeId>, NodeAccumulator>& accumulators) {
  std::unordered_map<graph::NodeId, std::vector<std::string>> public_ids_by_runtime;
  for (const auto& node : plan.public_nodes) {
    if (node.runtime_node == graph::kInvalidNode) {
      continue;
    }
    public_ids_by_runtime[node.runtime_node].push_back("p" + std::to_string(node.id));
  }

  for (auto& kv : accumulators) {
    const graph::NodeId runtime_node = kv.first.second;
    auto it = public_ids_by_runtime.find(runtime_node);
    if (it == public_ids_by_runtime.end()) {
      continue;
    }
    kv.second.metrics.public_node_ids = it->second;
  }
}

} // namespace

GraphMetricsReport build_graph_metrics_report_run_lifetime(const Run& run,
                                                           const GraphMetricsOptions& opt) {
  GraphMetricsReport out;
  const std::shared_ptr<const runtime::RunCore> core = run_internal::core(run);
  const RunStats stats = run_internal::stats(run);
  out.graph_metrics.counters.inputs_enqueued = stats.inputs_enqueued;
  out.graph_metrics.counters.inputs_dropped = stats.inputs_dropped;
  out.graph_metrics.counters.inputs_pushed = stats.inputs_pushed;
  out.graph_metrics.counters.outputs_ready = stats.outputs_ready;
  out.graph_metrics.counters.outputs_pulled = stats.outputs_pulled;
  out.graph_metrics.counters.outputs_dropped = stats.outputs_dropped;
  if (opt.include_power) {
    out.graph_metrics.power = run_internal::power_summary(run);
  }
  if (core) {
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    {
      std::lock_guard<std::mutex> lock(core->latency_mu);
      start = core->created_at;
      if (core->pull_timing_init) {
        end = core->last_pull_at;
      } else if (core->output_timing_init) {
        end = core->last_output_at;
      } else {
        end = std::chrono::steady_clock::now();
      }
    }
    if (start != std::chrono::steady_clock::time_point{} && end > start) {
      out.graph_metrics.elapsed_seconds = std::chrono::duration<double>(end - start).count();
      if (out.graph_metrics.elapsed_seconds > 0.0) {
        out.graph_metrics.throughput_fps =
            static_cast<double>(out.graph_metrics.counters.outputs_pulled) /
            out.graph_metrics.elapsed_seconds;
      }
    }
  }

  std::map<std::pair<std::size_t, graph::NodeId>, NodeAccumulator> accumulators;
  if (core && core->graph_execution_) {
    collect_graph_node_metrics(core, accumulators);
    attach_public_node_ids(core->graph_execution().plan, accumulators);
  } else {
    collect_linear_node_metrics(core, accumulators);
  }

  out.node_metrics.reserve(accumulators.size());
  for (auto& kv : accumulators) {
    finalize_node_accumulator(kv.second);
    out.node_metrics.push_back(std::move(kv.second.metrics));
  }
  return out;
}

} // namespace simaai::neat

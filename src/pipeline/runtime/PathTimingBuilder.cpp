#include "PathTimingBuilder.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>

namespace simaai::neat::runtime {
namespace {

std::string runtime_node_id(graph::NodeId id) {
  return id == graph::kInvalidNode ? std::string() : "n" + std::to_string(id);
}

std::string public_node_for_runtime(const ExecutionGraphPlan& plan, graph::NodeId id) {
  for (const auto& node : plan.public_nodes) {
    if (node.runtime_node == id) {
      return "p" + std::to_string(node.id);
    }
  }
  return {};
}

std::string public_edge_for_lowered(const ExecutionGraphPlan& plan, std::size_t edge_index) {
  for (const auto& edge : plan.public_edges) {
    if (std::find(edge.runtime_edge_indices.begin(), edge.runtime_edge_indices.end(), edge_index) !=
        edge.runtime_edge_indices.end()) {
      return "pe" + std::to_string(edge.id);
    }
  }
  return {};
}

std::string sample_key(std::uint64_t run_hash, const std::string& stream,
                       const std::string& message_id, std::int64_t orig_input_seq,
                       std::int64_t input_seq, std::int64_t frame_id) {
  std::string id;
  if (!message_id.empty() && message_id != "0") {
    id = "m:" + message_id;
  } else if (orig_input_seq >= 0) {
    id = "orig:" + std::to_string(orig_input_seq);
  } else if (input_seq >= 0) {
    id = "input:" + std::to_string(input_seq);
  } else if (frame_id >= 0) {
    id = "frame:" + std::to_string(frame_id);
  } else {
    id = "missing";
  }
  return std::to_string(run_hash) + "\x1f" + (stream.empty() ? "default" : stream) + "\x1f" + id;
}

std::string sample_key(const pipeline_internal::ParsedPluginSpan& span) {
  return sample_key(span.run_id_hash, span.stream_id, span.message_id, span.orig_input_seq,
                    span.input_seq, span.frame_id);
}

std::string sample_key(const pipeline_internal::ParsedEdgeSpan& span) {
  return sample_key(span.run_id_hash, span.stream_id, span.message_id, span.orig_input_seq,
                    span.input_seq, span.frame_id);
}

std::string sample_key(const pipeline_internal::ParsedGraphMessageEvent& ev) {
  return sample_key(ev.run_id_hash, ev.stream_id, ev.message_id, ev.orig_input_seq, ev.input_seq,
                    ev.frame_id);
}

MeasurePathStat summarize(std::vector<double> values, bool reliable) {
  MeasurePathStat out;
  out.samples = values.size();
  out.reliable = reliable;
  if (values.empty()) {
    return out;
  }
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  out.avg_ms = sum / static_cast<double>(values.size());
  std::sort(values.begin(), values.end());
  auto percentile = [&](double p) {
    const double index = p * static_cast<double>(values.size() - 1U);
    const auto lo = static_cast<std::size_t>(index);
    const auto hi = std::min(lo + 1U, values.size() - 1U);
    const double frac = index - static_cast<double>(lo);
    return values[lo] + ((values[hi] - values[lo]) * frac);
  };
  out.p50_ms = percentile(0.50);
  out.p95_ms = percentile(0.95);
  out.max_ms = values.back();
  return out;
}

using GapAggKey = std::tuple<std::string, std::string, std::string, std::string, std::string>;

using NodeArrivalAggKey =
    std::tuple<std::string, std::string, std::int32_t, std::string, std::string>;
using EdgeAggKey = std::tuple<std::string, std::string, std::string, std::string, std::int32_t,
                              std::int32_t, std::string, std::string, std::string, std::string>;
using OutputTailAggKey = std::tuple<std::string, std::string, std::string, std::string>;

constexpr std::uint32_t kGraphEntryEventType = 5;
constexpr std::uint32_t kGraphOutputPullEventType = 6;

std::optional<std::size_t> lowered_edge_index_from_id(const std::string& id) {
  if (id.size() < 2U || id[0] != 'e') {
    return std::nullopt;
  }
  try {
    std::size_t pos = 0;
    const std::size_t value = std::stoull(id.substr(1), &pos, 10);
    if (pos != id.size() - 1U) {
      return std::nullopt;
    }
    return value;
  } catch (...) {
    return std::nullopt;
  }
}

std::string lowered_edge_id_from_metric(const MeasureEdgeLatency& metric) {
  const std::string& id = metric.edge_id;
  if (id.size() > 1U && id[0] == 'e' &&
      std::all_of(id.begin() + 1, id.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
    return id;
  }
  if (!id.empty() &&
      std::all_of(id.begin(), id.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
    return "e" + id;
  }
  return {};
}

graph::NodeId output_node_for_endpoint(const ExecutionGraphPlan& plan,
                                       const std::string& endpoint) {
  if (!endpoint.empty()) {
    const auto it = plan.named_outputs.find(endpoint);
    if (it != plan.named_outputs.end()) {
      return it->second.node;
    }
  }
  if (plan.default_output.has_value()) {
    return plan.default_output->node;
  }
  return graph::kInvalidNode;
}

std::string single_incoming_lowered_edge(const ExecutionGraphPlan& plan, graph::NodeId node) {
  std::string out;
  for (std::size_t i = 0; i < plan.edges.size(); ++i) {
    if (plan.edges[i].to != node) {
      continue;
    }
    if (!out.empty()) {
      return {};
    }
    out = "e" + std::to_string(i);
  }
  return out;
}

} // namespace

MeasurePathTiming build_path_timing(const ExecutionGraphPlan& plan,
                                    const pipeline_internal::LttngParseResult& parsed,
                                    const std::vector<GraphSampleTimingEvent>& graph_entries,
                                    const std::vector<GraphSampleTimingEvent>& graph_pulls) {
  (void)graph_entries;
  (void)graph_pulls;
  MeasurePathTiming timing;
  timing.source = "lttng_message_events";
  timing.identity.primary_key = "run_id_hash + stream_id + message_id";
  timing.identity.fallback_key = "run_id_hash + stream_id + orig_input_seq/input_seq/frame_id";
  timing.identity.used_public_fields = {"stream_id", "orig_input_seq", "input_seq", "frame_id"};
  timing.identity.sample_identity_source = "Sample identity fields propagated into LTTng payloads";

  if (parsed.raw_edge_spans.empty() && parsed.raw_plugin_spans.empty()) {
    timing.available = false;
    timing.status = "unavailable_message_trace_disabled";
    timing.source = "none";
    timing.reason = "No per-message LTTng edge/plugin spans were parsed.";
    return timing;
  }
  if (parsed.raw_edge_spans.empty() && parsed.raw_graph_events.empty() &&
      !parsed.raw_plugin_spans.empty()) {
    timing.source = "lttng_plugin_spans";
    timing.warnings.push_back(
        "Exact core message events were not parsed; inter-plugin path rows use plugin-span "
        "start/end fallback and do not include graph entry/output-tail transport.");
  }

  std::map<std::string, std::vector<const pipeline_internal::ParsedPluginSpan*>> plugins_by_sample;
  for (const auto& span : parsed.raw_plugin_spans) {
    plugins_by_sample[sample_key(span)].push_back(&span);
  }

  struct GraphEventChoice {
    double timestamp_s = 0.0;
    std::string endpoint;
    bool valid = false;
    bool ambiguous = false;
  };

  std::map<std::string, GraphEventChoice> entry_by_sample;
  std::map<std::string, GraphEventChoice> pull_by_sample;
  for (const auto& ev : parsed.raw_graph_events) {
    const std::string key = sample_key(ev);
    if (key.find("missing") != std::string::npos) {
      continue;
    }
    auto& target =
        ev.event_type == kGraphEntryEventType ? entry_by_sample[key] : pull_by_sample[key];
    if (target.valid) {
      target.ambiguous = true;
      if (ev.event_type == kGraphEntryEventType) {
        target.timestamp_s = std::min(target.timestamp_s, ev.timestamp_s);
      } else {
        target.timestamp_s = std::max(target.timestamp_s, ev.timestamp_s);
      }
      if (target.endpoint != ev.endpoint) {
        target.endpoint.clear();
      }
    } else {
      target.valid = true;
      target.timestamp_s = ev.timestamp_s;
      target.endpoint = ev.endpoint;
    }
  }

  bool all_reliable = !parsed.trace_loss_detected;
  std::map<NodeArrivalAggKey, std::vector<double>> node_arrival_samples;
  for (const auto& [key, spans] : plugins_by_sample) {
    const auto entry_it = entry_by_sample.find(key);
    if (entry_it == entry_by_sample.end() || !entry_it->second.valid ||
        entry_it->second.ambiguous) {
      if (entry_it != entry_by_sample.end() && entry_it->second.ambiguous) {
        all_reliable = false;
      }
      continue;
    }

    std::map<std::int32_t, const pipeline_internal::ParsedPluginSpan*> earliest_by_node;
    for (const auto* span : spans) {
      if (!span || span->metric_identity.runtime_node_id < 0 ||
          span->start_s < entry_it->second.timestamp_s) {
        continue;
      }
      auto& slot = earliest_by_node[span->metric_identity.runtime_node_id];
      if (!slot || span->start_s < slot->start_s) {
        slot = span;
      }
    }
    for (const auto& [runtime_node, span] : earliest_by_node) {
      if (!span) {
        continue;
      }
      const double arrival_ms = (span->start_s - entry_it->second.timestamp_s) * 1000.0;
      if (arrival_ms < 0.0) {
        continue;
      }
      const std::string lowered_node = "n" + std::to_string(runtime_node);
      const std::string customer_node =
          public_node_for_runtime(plan, static_cast<graph::NodeId>(runtime_node));
      const NodeArrivalAggKey agg_key{customer_node, lowered_node, runtime_node,
                                      span->metric_identity.plugin_instance_id,
                                      span->metric_identity.stream_id};
      node_arrival_samples[agg_key].push_back(arrival_ms);
    }
  }

  for (auto& [key, values] : node_arrival_samples) {
    MeasurePathNodeArrival row;
    row.customer_node_id = std::get<0>(key);
    row.lowered_node_id = std::get<1>(key);
    row.runtime_node_id = std::get<2>(key);
    row.plugin_instance_id = std::get<3>(key);
    row.stream_id = std::get<4>(key);
    row.latency = summarize(std::move(values), all_reliable);
    timing.node_arrival.push_back(std::move(row));
  }

  std::map<EdgeAggKey, std::vector<double>> edge_transport_samples;
  for (const auto& span : parsed.raw_edge_spans) {
    const std::string lowered_edge = lowered_edge_id_from_metric(span.metric_identity);
    if (lowered_edge.empty()) {
      continue;
    }
    const auto edge_index = lowered_edge_index_from_id(lowered_edge);
    if (!edge_index.has_value() || *edge_index >= plan.edges.size()) {
      continue;
    }
    const auto& edge = plan.edges[*edge_index];
    const std::string public_edge = public_edge_for_lowered(plan, *edge_index);
    const std::string from_customer = public_node_for_runtime(plan, edge.from);
    const std::string to_customer = public_node_for_runtime(plan, edge.to);
    const double latency_ms = (span.end_s - span.start_s) * 1000.0;
    if (latency_ms < 0.0) {
      all_reliable = false;
      continue;
    }
    const std::string semantics = span.metric_identity.timing_semantics.empty()
                                      ? "edge_transport"
                                      : span.metric_identity.timing_semantics;
    const EdgeAggKey agg_key{public_edge,
                             lowered_edge,
                             from_customer,
                             to_customer,
                             span.metric_identity.from_runtime_node_id,
                             span.metric_identity.to_runtime_node_id,
                             span.metric_identity.from_plugin_instance_id,
                             span.metric_identity.to_plugin_instance_id,
                             span.metric_identity.stream_id,
                             semantics};
    edge_transport_samples[agg_key].push_back(latency_ms);
  }

  for (auto& [key, values] : edge_transport_samples) {
    MeasurePathInterPluginGap row;
    row.customer_edge_id = std::get<0>(key);
    row.lowered_edge_id = std::get<1>(key);
    row.from_customer_node_id = std::get<2>(key);
    row.to_customer_node_id = std::get<3>(key);
    row.from_runtime_node_id = std::get<4>(key);
    row.to_runtime_node_id = std::get<5>(key);
    row.from_plugin_instance_id = std::get<6>(key);
    row.to_plugin_instance_id = std::get<7>(key);
    row.stream_id = std::get<8>(key);
    row.semantics = std::get<9>(key);
    row.latency = summarize(std::move(values), all_reliable);
    timing.inter_plugin_gap.push_back(std::move(row));
  }

  std::map<GapAggKey, std::vector<double>> gap_samples;
  for (std::size_t edge_index = 0; edge_index < plan.edges.size(); ++edge_index) {
    const auto& edge = plan.edges[edge_index];
    if (edge.from == graph::kInvalidNode || edge.to == graph::kInvalidNode) {
      continue;
    }
    const std::string lowered_edge = "e" + std::to_string(edge_index);
    const std::string public_edge = public_edge_for_lowered(plan, edge_index);
    const std::string from_customer = public_node_for_runtime(plan, edge.from);
    const std::string to_customer = public_node_for_runtime(plan, edge.to);
    for (const auto& [key, spans] : plugins_by_sample) {
      (void)key;
      const pipeline_internal::ParsedPluginSpan* upstream = nullptr;
      const pipeline_internal::ParsedPluginSpan* downstream = nullptr;
      for (const auto* span : spans) {
        if (!span) {
          continue;
        }
        if (span->metric_identity.runtime_node_id == static_cast<std::int32_t>(edge.from)) {
          if (!upstream || span->end_s > upstream->end_s) {
            upstream = span;
          }
        }
        if (span->metric_identity.runtime_node_id == static_cast<std::int32_t>(edge.to)) {
          if (!downstream || span->start_s < downstream->start_s) {
            downstream = span;
          }
        }
      }
      if (!upstream || !downstream || downstream->start_s < upstream->end_s) {
        continue;
      }
      const double gap_ms = (downstream->start_s - upstream->end_s) * 1000.0;
      const GapAggKey agg_key{public_edge, lowered_edge, from_customer, to_customer,
                              downstream->metric_identity.stream_id};
      gap_samples[agg_key].push_back(gap_ms);
    }
  }

  for (auto& [key, values] : gap_samples) {
    MeasurePathInterPluginGap row;
    row.customer_edge_id = std::get<0>(key);
    row.lowered_edge_id = std::get<1>(key);
    row.from_customer_node_id = std::get<2>(key);
    row.to_customer_node_id = std::get<3>(key);
    row.stream_id = std::get<4>(key);
    row.semantics = "upstream_plugin_end_to_downstream_plugin_start";
    row.latency = summarize(std::move(values), all_reliable);
    if (!row.lowered_edge_id.empty()) {
      const std::size_t edge_index =
          static_cast<std::size_t>(std::stoull(row.lowered_edge_id.substr(1)));
      if (edge_index < plan.edges.size()) {
        row.from_runtime_node_id = static_cast<std::int32_t>(plan.edges[edge_index].from);
        row.to_runtime_node_id = static_cast<std::int32_t>(plan.edges[edge_index].to);
      }
    }
    timing.inter_plugin_gap.push_back(std::move(row));
  }

  std::map<OutputTailAggKey, std::vector<double>> output_tail_samples;
  for (const auto& [key, spans] : plugins_by_sample) {
    const auto pull_it = pull_by_sample.find(key);
    if (pull_it == pull_by_sample.end() || !pull_it->second.valid || pull_it->second.ambiguous) {
      if (pull_it != pull_by_sample.end() && pull_it->second.ambiguous) {
        all_reliable = false;
      }
      continue;
    }
    const pipeline_internal::ParsedPluginSpan* latest = nullptr;
    for (const auto* span : spans) {
      if (!span || span->end_s > pull_it->second.timestamp_s) {
        continue;
      }
      if (!latest || span->end_s > latest->end_s) {
        latest = span;
      }
    }
    if (!latest) {
      continue;
    }
    const double tail_ms = (pull_it->second.timestamp_s - latest->end_s) * 1000.0;
    if (tail_ms < 0.0) {
      all_reliable = false;
      continue;
    }
    const std::string endpoint =
        pull_it->second.endpoint.empty() ? "default" : pull_it->second.endpoint;
    const graph::NodeId output_node = output_node_for_endpoint(plan, endpoint);
    const std::string customer_node = public_node_for_runtime(plan, output_node);
    const std::string lowered_edge = single_incoming_lowered_edge(plan, output_node);
    const OutputTailAggKey agg_key{endpoint, customer_node, lowered_edge, latest->stream_id};
    output_tail_samples[agg_key].push_back(tail_ms);
  }

  for (auto& [key, values] : output_tail_samples) {
    MeasurePathOutputTail row;
    row.output_endpoint = std::get<0>(key);
    row.customer_output_node_id = std::get<1>(key);
    row.lowered_edge_id = std::get<2>(key);
    row.stream_id = std::get<3>(key);
    row.latency = summarize(std::move(values), all_reliable);
    timing.output_tail.push_back(std::move(row));
  }

  timing.available = !timing.node_arrival.empty() || !timing.inter_plugin_gap.empty() ||
                     !timing.output_tail.empty();
  timing.status =
      timing.available ? (all_reliable ? "collected" : "partial") : "unavailable_no_path_rows";
  if (!timing.available) {
    timing.reason = "Message/plugin spans were parsed, but no unambiguous per-sample path rows "
                    "matched topology.";
  }
  if (parsed.trace_loss_detected) {
    timing.warnings.push_back("LTTng trace loss detected; path timing rows may be incomplete.");
  }
  for (const auto& [key, choice] : entry_by_sample) {
    (void)key;
    if (choice.ambiguous) {
      timing.warnings.push_back("Ambiguous graph-entry event identity; node-arrival rows skipped "
                                "for at least one sample.");
      break;
    }
  }
  for (const auto& [key, choice] : pull_by_sample) {
    (void)key;
    if (choice.ambiguous) {
      timing.warnings.push_back("Ambiguous graph-output event identity; output-tail rows skipped "
                                "for at least one sample.");
      break;
    }
  }
  return timing;
}

} // namespace simaai::neat::runtime

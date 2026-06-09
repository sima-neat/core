#include "LttngMetricsCollector.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <deque>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <tuple>

namespace simaai::neat::pipeline_internal {
namespace {

struct RawEvent {
  double timestamp_s = 0.0;
  std::string provider;
  std::string event_name;
  std::map<std::string, std::string> fields;
};

std::string trim(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

std::string strip_quotes(std::string s) {
  s = trim(std::move(s));
  if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
    return s.substr(1, s.size() - 2);
  }
  return s;
}

std::uint64_t parse_u64(const std::map<std::string, std::string>& fields, const std::string& key,
                        std::uint64_t fallback = 0) {
  const auto it = fields.find(key);
  if (it == fields.end() || it->second.empty()) {
    return fallback;
  }
  try {
    return static_cast<std::uint64_t>(std::stoull(it->second, nullptr, 0));
  } catch (...) {
    return fallback;
  }
}

std::int32_t parse_i32(const std::map<std::string, std::string>& fields, const std::string& key,
                       std::int32_t fallback = -1) {
  const auto it = fields.find(key);
  if (it == fields.end() || it->second.empty()) {
    return fallback;
  }
  try {
    return static_cast<std::int32_t>(std::stoll(it->second, nullptr, 0));
  } catch (...) {
    return fallback;
  }
}

std::int64_t parse_i64(const std::map<std::string, std::string>& fields, const std::string& key,
                       std::int64_t fallback = -1) {
  const auto it = fields.find(key);
  if (it == fields.end() || it->second.empty()) {
    return fallback;
  }
  try {
    return static_cast<std::int64_t>(std::stoll(it->second, nullptr, 0));
  } catch (...) {
    return fallback;
  }
}

std::string field_string(const std::map<std::string, std::string>& fields, const std::string& key) {
  const auto it = fields.find(key);
  return it == fields.end() ? std::string() : strip_quotes(it->second);
}

std::optional<RawEvent> parse_line(const std::string& line) {
  const std::size_t lb = line.find('[');
  const std::size_t rb = line.find(']', lb == std::string::npos ? 0 : lb + 1);
  if (lb == std::string::npos || rb == std::string::npos || rb <= lb + 1) {
    return std::nullopt;
  }

  RawEvent ev;
  try {
    ev.timestamp_s = std::stod(line.substr(lb + 1, rb - lb - 1));
  } catch (...) {
    return std::nullopt;
  }

  static const std::regex provider_event_re(
      R"((\b[A-Za-z_][A-Za-z0-9_]*):([A-Za-z_][A-Za-z0-9_]*):)");
  std::smatch match;
  if (!std::regex_search(line, match, provider_event_re) || match.size() < 3) {
    return std::nullopt;
  }
  ev.provider = match[1].str();
  ev.event_name = match[2].str();

  static const std::regex field_re(
      R"(([A-Za-z_][A-Za-z0-9_]*)\s*=\s*("(?:[^"\\]|\\.)*"|0x[0-9A-Fa-f]+|-?[0-9]+|[^,}\s]+))");
  for (auto it = std::sregex_iterator(line.begin(), line.end(), field_re);
       it != std::sregex_iterator(); ++it) {
    ev.fields[(*it)[1].str()] = (*it)[2].str();
  }
  return ev;
}

bool is_start(const RawEvent& ev) {
  return parse_i32(ev.fields, "event_type", -1) == 0;
}
bool is_end(const RawEvent& ev) {
  return parse_i32(ev.fields, "event_type", -1) == 1;
}

std::string backend_from_event(const RawEvent& ev) {
  if (ev.provider == "sima_neat_plugin") {
    return field_string(ev.fields, "backend");
  }
  if (ev.provider == "pipeline") {
    return ev.event_name;
  }
  return ev.provider + ":" + ev.event_name;
}

struct SpanStart {
  RawEvent ev;
};

struct SpanAgg {
  MeasurePluginLatency metric;
  std::vector<double> durations_ms;
};

std::string join_key(std::initializer_list<std::string> parts) {
  std::ostringstream os;
  bool first = true;
  for (const auto& part : parts) {
    if (!first) {
      os << '\x1f';
    }
    first = false;
    os << part;
  }
  return os.str();
}

bool expected_hash_mismatch(const RawEvent& ev, std::uint64_t expected_run_id_hash,
                            std::uint64_t expected_graph_id_hash) {
  const std::uint64_t run_id = parse_u64(ev.fields, "run_id_hash", 0);
  if (expected_run_id_hash != 0 && run_id != 0 && run_id != expected_run_id_hash) {
    return true;
  }
  const std::uint64_t graph_id = parse_u64(ev.fields, "graph_id_hash", 0);
  if (expected_graph_id_hash != 0 && graph_id != 0 && graph_id != expected_graph_id_hash) {
    return true;
  }
  return false;
}

std::string message_identity_key(const RawEvent& ev) {
  const std::string message_id = field_string(ev.fields, "message_id");
  if (!message_id.empty() && message_id != "0") {
    return "m:" + message_id;
  }
  const std::string orig = field_string(ev.fields, "orig_input_seq");
  if (!orig.empty() && orig != "-1") {
    return "orig:" + orig;
  }
  const std::string input = field_string(ev.fields, "input_seq");
  if (!input.empty() && input != "-1") {
    return "input:" + input;
  }
  const std::string frame = field_string(ev.fields, "frame_id");
  if (!frame.empty() && frame != "-1") {
    return "frame:" + frame;
  }
  return {};
}

std::string plugin_pair_key(const RawEvent& ev) {
  if (ev.provider == "sima_neat_plugin") {
    return join_key(
        {field_string(ev.fields, "run_id_hash"), field_string(ev.fields, "graph_id_hash"),
         field_string(ev.fields, "pipeline_segment_id"), field_string(ev.fields, "runtime_node_id"),
         field_string(ev.fields, "plugin_instance_id"), field_string(ev.fields, "element_name"),
         field_string(ev.fields, "backend"), field_string(ev.fields, "phase"),
         field_string(ev.fields, "kernel_name"), field_string(ev.fields, "stream_id"),
         field_string(ev.fields, "frame_id"), field_string(ev.fields, "request_id"),
         field_string(ev.fields, "message_id")});
  }
  return join_key({field_string(ev.fields, "vpid"), ev.provider, ev.event_name,
                   field_string(ev.fields, "plugin_id"), field_string(ev.fields, "stream_id"),
                   field_string(ev.fields, "frame_id")});
}

MeasurePluginLatency metric_from_span(const RawEvent& start, const RawEvent& end,
                                      double duration_ms, std::uint64_t expected_run_id_hash,
                                      bool* belongs_to_other_run) {
  MeasurePluginLatency p;
  p.source = "lttng";
  p.reliable = true;
  p.backend = backend_from_event(start);
  p.phase = field_string(start.fields, "phase");
  if (p.phase.empty()) {
    p.phase = "Run";
  }
  p.kernel_name = field_string(start.fields, "kernel_name");
  p.stream_id = field_string(start.fields, "stream_id");
  p.run_id_hash = parse_u64(start.fields, "run_id_hash", 0);
  if (p.run_id_hash == 0) {
    p.run_id_hash = parse_u64(end.fields, "run_id_hash", 0);
  }
  if (belongs_to_other_run) {
    *belongs_to_other_run =
        expected_run_id_hash != 0 && p.run_id_hash != 0 && p.run_id_hash != expected_run_id_hash;
  }

  if (start.provider == "sima_neat_plugin") {
    p.pipeline_segment_id = parse_i32(start.fields, "pipeline_segment_id", -1);
    p.runtime_node_id = parse_i32(start.fields, "runtime_node_id", -1);
    p.public_node_id = parse_i32(start.fields, "public_node_id", -1);
    p.plugin_instance_id = field_string(start.fields, "plugin_instance_id");
    p.gst_element_name = field_string(start.fields, "element_name");
    p.stage_name = p.gst_element_name;
    p.attribution_source = (p.runtime_node_id >= 0 && p.pipeline_segment_id >= 0)
                               ? "lttng_v2_identity"
                               : "unattributed";
  } else {
    p.gst_element_name = field_string(start.fields, "plugin_id");
    p.stage_name = p.gst_element_name;
    p.attribution_source = "lttng_element_name";
  }
  p.name = p.backend + ":" +
           (!p.plugin_instance_id.empty()
                ? p.plugin_instance_id
                : (!p.gst_element_name.empty() ? p.gst_element_name : p.kernel_name));
  p.calls = 1;
  p.total_ms = duration_ms;
  p.avg_ms = duration_ms;
  p.min_ms = duration_ms;
  p.max_ms = duration_ms;
  return p;
}

std::string plugin_agg_key(const MeasurePluginLatency& p) {
  return join_key({p.backend, p.phase, p.kernel_name, p.plugin_instance_id, p.gst_element_name,
                   p.stream_id, std::to_string(p.run_id_hash),
                   std::to_string(p.pipeline_segment_id), std::to_string(p.runtime_node_id),
                   std::to_string(p.public_node_id)});
}

void add_plugin_metric(std::map<std::string, SpanAgg>* aggs, MeasurePluginLatency metric) {
  const std::string key = plugin_agg_key(metric);
  SpanAgg& agg = (*aggs)[key];
  if (agg.metric.calls == 0) {
    agg.metric = metric;
    agg.durations_ms.push_back(metric.total_ms);
    return;
  }
  agg.metric.calls += metric.calls;
  agg.metric.total_ms += metric.total_ms;
  agg.metric.avg_ms = agg.metric.total_ms / static_cast<double>(agg.metric.calls);
  agg.metric.min_ms = std::min(agg.metric.min_ms, metric.min_ms);
  agg.metric.max_ms = std::max(agg.metric.max_ms, metric.max_ms);
  agg.metric.reliable = agg.metric.reliable && metric.reliable;
  agg.durations_ms.push_back(metric.total_ms);
}

struct EdgeStart {
  RawEvent ev;
};

std::string edge_pair_key(const RawEvent& ev) {
  const int type = parse_i32(ev.fields, "event_type", -1);
  const std::string span_kind = (type == 0 || type == 1)   ? "edge_transport"
                                : (type == 2 || type == 3) ? "queue_residence"
                                                           : "unknown";
  return join_key({field_string(ev.fields, "run_id_hash"), field_string(ev.fields, "graph_id_hash"),
                   field_string(ev.fields, "edge_id"), field_string(ev.fields, "stream_id"),
                   message_identity_key(ev), span_kind});
}

bool is_edge_first(const RawEvent& ev) {
  const int type = parse_i32(ev.fields, "event_type", -1);
  return type == 0 || type == 2;
}

bool is_edge_second(const RawEvent& ev) {
  const int type = parse_i32(ev.fields, "event_type", -1);
  return type == 1 || type == 3;
}

bool is_graph_lifecycle_message(const RawEvent& ev) {
  const int type = parse_i32(ev.fields, "event_type", -1);
  return type == 5 || type == 6;
}

ParsedGraphMessageEvent graph_event_from_raw(const RawEvent& ev) {
  ParsedGraphMessageEvent out;
  out.timestamp_s = ev.timestamp_s;
  out.event_type = static_cast<std::uint32_t>(parse_i32(ev.fields, "event_type", 0));
  out.run_id_hash = parse_u64(ev.fields, "run_id_hash", 0);
  out.graph_id_hash = parse_u64(ev.fields, "graph_id_hash", 0);
  out.endpoint = field_string(ev.fields, "endpoint");
  if (out.endpoint.empty()) {
    out.endpoint = field_string(ev.fields, "src_pad");
  }
  if (out.endpoint.empty()) {
    out.endpoint = field_string(ev.fields, "dst_pad");
  }
  out.stream_id = field_string(ev.fields, "stream_id");
  out.message_id = field_string(ev.fields, "message_id");
  out.frame_id = parse_i64(ev.fields, "frame_id", -1);
  out.input_seq = parse_i64(ev.fields, "input_seq", -1);
  out.orig_input_seq = parse_i64(ev.fields, "orig_input_seq", -1);
  return out;
}

MeasureEdgeLatency edge_metric_from_span(const RawEvent& start, const RawEvent& end,
                                         double duration_ms) {
  MeasureEdgeLatency m;
  m.source = "lttng";
  const int type = parse_i32(start.fields, "event_type", -1);
  m.timing_semantics = type == 2 ? "queue_residence" : "edge_transport";
  m.edge_id = field_string(start.fields, "edge_id");
  if (m.edge_id.empty()) {
    m.edge_id = field_string(end.fields, "edge_id");
  }
  if (!m.edge_id.empty() && std::all_of(m.edge_id.begin(), m.edge_id.end(),
                                        [](unsigned char c) { return std::isdigit(c) != 0; })) {
    m.edge_id = "e" + m.edge_id;
  }
  m.name = m.edge_id.empty() ? "edge" : m.edge_id;
  m.from_runtime_node_id = parse_i32(start.fields, "src_runtime_node_id", -1);
  m.to_runtime_node_id = parse_i32(start.fields, "dst_runtime_node_id", -1);
  m.from_node_id = m.from_runtime_node_id >= 0 ? "n" + std::to_string(m.from_runtime_node_id) : "";
  m.to_node_id = m.to_runtime_node_id >= 0 ? "n" + std::to_string(m.to_runtime_node_id) : "";
  m.from_plugin_instance_id = field_string(start.fields, "src_plugin_instance_id");
  m.to_plugin_instance_id = field_string(start.fields, "dst_plugin_instance_id");
  m.from_element_name = field_string(start.fields, "src_element");
  m.to_element_name = field_string(start.fields, "dst_element");
  m.stream_id = field_string(start.fields, "stream_id");
  m.samples = 1;
  m.total_ms = duration_ms;
  m.avg_ms = duration_ms;
  m.min_ms = duration_ms;
  m.max_ms = duration_ms;
  m.p50_ms = duration_ms;
  m.p95_ms = duration_ms;
  m.attribution_source = m.edge_id.empty() ? "unattributed" : "graph_edge_identity";
  if (m.edge_id.empty()) {
    m.mapping_error = "MISSING_MESSAGE_ID";
  }
  return m;
}

struct EdgeAgg {
  MeasureEdgeLatency metric;
  std::vector<double> durations_ms;
};

std::string edge_agg_key(const MeasureEdgeLatency& m) {
  return join_key({m.edge_id, m.from_node_id, m.to_node_id, m.from_plugin_instance_id,
                   m.to_plugin_instance_id, m.stream_id, m.timing_semantics});
}

void finalize_edge_agg(EdgeAgg& agg) {
  if (agg.durations_ms.empty()) {
    return;
  }
  std::sort(agg.durations_ms.begin(), agg.durations_ms.end());
  auto percentile = [&](double p) {
    const double index = p * static_cast<double>(agg.durations_ms.size() - 1U);
    const auto lo = static_cast<std::size_t>(index);
    const auto hi = std::min(lo + 1U, agg.durations_ms.size() - 1U);
    const double frac = index - static_cast<double>(lo);
    return agg.durations_ms[lo] + ((agg.durations_ms[hi] - agg.durations_ms[lo]) * frac);
  };
  agg.metric.p50_ms = percentile(0.50);
  agg.metric.p95_ms = percentile(0.95);
}

void add_edge_metric(std::map<std::string, EdgeAgg>* aggs, MeasureEdgeLatency metric) {
  const std::string key = edge_agg_key(metric);
  EdgeAgg& agg = (*aggs)[key];
  if (agg.metric.samples == 0) {
    agg.metric = metric;
    agg.durations_ms.push_back(metric.total_ms);
    return;
  }
  agg.metric.samples += metric.samples;
  agg.metric.total_ms += metric.total_ms;
  agg.metric.avg_ms = agg.metric.total_ms / static_cast<double>(agg.metric.samples);
  agg.metric.min_ms = std::min(agg.metric.min_ms, metric.min_ms);
  agg.metric.max_ms = std::max(agg.metric.max_ms, metric.max_ms);
  agg.metric.reliable = agg.metric.reliable && metric.reliable;
  agg.durations_ms.push_back(metric.total_ms);
}

} // namespace

LttngParseResult parse_lttng_trace_text(const std::string& text, std::uint64_t expected_run_id_hash,
                                        std::uint64_t expected_graph_id_hash,
                                        bool allow_pipeline_fallback) {
  LttngParseResult result;
  result.parsed = true;

  std::map<std::string, std::deque<SpanStart>> plugin_starts;
  std::map<std::string, SpanAgg> plugin_aggs;
  std::map<std::string, std::deque<EdgeStart>> edge_starts;
  std::map<std::string, EdgeAgg> edge_aggs;

  std::istringstream input(text);
  std::string line;
  while (std::getline(input, line)) {
    if (line.find("discarded") != std::string::npos || line.find("lost") != std::string::npos) {
      result.trace_loss_detected = true;
      result.warnings.push_back("TRACE_LOSS_DETECTED: babeltrace reported discarded/lost events");
    }
    auto ev_opt = parse_line(line);
    if (!ev_opt) {
      continue;
    }
    RawEvent ev = std::move(*ev_opt);

    const bool plugin_event = ev.provider == "sima_neat_plugin" && ev.event_name == "plugin_span";
    const bool pipeline_event = allow_pipeline_fallback && ev.provider == "pipeline";
    if ((plugin_event || pipeline_event) && (is_start(ev) || is_end(ev))) {
      if (expected_hash_mismatch(ev, expected_run_id_hash, expected_graph_id_hash)) {
        continue;
      }
      const std::string key = plugin_pair_key(ev);
      if (is_start(ev)) {
        plugin_starts[key].push_back({std::move(ev)});
      } else {
        auto it = plugin_starts.find(key);
        if (it == plugin_starts.end() || it->second.empty()) {
          result.warnings.push_back("UNMATCHED_PLUGIN_END: " + key);
          continue;
        }
        RawEvent start = std::move(it->second.front().ev);
        it->second.pop_front();
        const double duration_ms = (ev.timestamp_s - start.timestamp_s) * 1000.0;
        if (duration_ms < 0.0 || !std::isfinite(duration_ms)) {
          result.warnings.push_back("NEGATIVE_PLUGIN_LATENCY: " + key);
          continue;
        }
        bool other_run = false;
        MeasurePluginLatency metric =
            metric_from_span(start, ev, duration_ms, expected_run_id_hash, &other_run);
        if (other_run) {
          result.warnings.push_back("PLUGIN_SPAN_OTHER_RUN: dropped span with run_id_hash=" +
                                    std::to_string(metric.run_id_hash));
          continue;
        }
        ParsedPluginSpan raw;
        raw.start_s = start.timestamp_s;
        raw.end_s = ev.timestamp_s;
        raw.metric_identity = metric;
        raw.run_id_hash = parse_u64(start.fields, "run_id_hash", metric.run_id_hash);
        raw.graph_id_hash = parse_u64(start.fields, "graph_id_hash", 0);
        raw.message_id = field_string(start.fields, "message_id");
        raw.stream_id = metric.stream_id;
        raw.frame_id = parse_i64(start.fields, "frame_id", -1);
        raw.input_seq = parse_i64(start.fields, "input_seq", -1);
        raw.orig_input_seq = parse_i64(start.fields, "orig_input_seq", -1);
        result.raw_plugin_spans.push_back(raw);
        add_plugin_metric(&plugin_aggs, std::move(metric));
      }
      continue;
    }

    if (ev.provider == "sima_neat_edge" && ev.event_name == "message") {
      if (expected_hash_mismatch(ev, expected_run_id_hash, expected_graph_id_hash)) {
        continue;
      }
      if (is_graph_lifecycle_message(ev)) {
        if (message_identity_key(ev).empty()) {
          result.warnings.push_back("UNKEYED_GRAPH_MESSAGE_EVENT: event_type=" +
                                    field_string(ev.fields, "event_type"));
          continue;
        }
        result.raw_graph_events.push_back(graph_event_from_raw(ev));
        continue;
      }
      if (!is_edge_first(ev) && !is_edge_second(ev)) {
        continue;
      }
      const std::string key = edge_pair_key(ev);
      if (message_identity_key(ev).empty() || field_string(ev.fields, "edge_id").empty()) {
        MeasureEdgeLatency bad;
        bad.source = "lttng";
        bad.attribution_source = "unattributed";
        bad.mapping_error = "MISSING_MESSAGE_ID";
        bad.reliable = false;
        result.edge_metrics_unattributed.push_back(std::move(bad));
        continue;
      }
      if (is_edge_first(ev)) {
        edge_starts[key].push_back({std::move(ev)});
      } else {
        auto it = edge_starts.find(key);
        if (it == edge_starts.end() || it->second.empty()) {
          result.warnings.push_back("UNMATCHED_EDGE_RECEIVE: " + key);
          continue;
        }
        RawEvent start = std::move(it->second.front().ev);
        it->second.pop_front();
        const double duration_ms = (ev.timestamp_s - start.timestamp_s) * 1000.0;
        if (duration_ms < 0.0 || !std::isfinite(duration_ms)) {
          result.warnings.push_back("NEGATIVE_EDGE_LATENCY: " + key);
          continue;
        }
        MeasureEdgeLatency metric = edge_metric_from_span(start, ev, duration_ms);
        ParsedEdgeSpan raw;
        raw.start_s = start.timestamp_s;
        raw.end_s = ev.timestamp_s;
        raw.metric_identity = metric;
        raw.start_type = static_cast<std::uint32_t>(parse_i32(start.fields, "event_type", 0));
        raw.end_type = static_cast<std::uint32_t>(parse_i32(ev.fields, "event_type", 1));
        raw.run_id_hash = parse_u64(start.fields, "run_id_hash", 0);
        raw.graph_id_hash = parse_u64(start.fields, "graph_id_hash", 0);
        raw.message_id = field_string(start.fields, "message_id");
        raw.stream_id = metric.stream_id;
        raw.frame_id = parse_i64(start.fields, "frame_id", -1);
        raw.input_seq = parse_i64(start.fields, "input_seq", -1);
        raw.orig_input_seq = parse_i64(start.fields, "orig_input_seq", -1);
        result.raw_edge_spans.push_back(raw);
        add_edge_metric(&edge_aggs, std::move(metric));
      }
    }
  }

  for (const auto& [key, starts] : plugin_starts) {
    if (!starts.empty()) {
      result.warnings.push_back("UNMATCHED_PLUGIN_START: " + key);
    }
  }
  for (const auto& [key, starts] : edge_starts) {
    if (!starts.empty()) {
      result.warnings.push_back("UNMATCHED_EDGE_EMIT: " + key);
    }
  }

  auto duplicated_by_v2 = [&](const MeasurePluginLatency& fallback) {
    if (fallback.attribution_source != "lttng_element_name") {
      return false;
    }
    for (const auto& [other_key, other_agg] : plugin_aggs) {
      (void)other_key;
      const MeasurePluginLatency& v2 = other_agg.metric;
      if (v2.attribution_source != "lttng_v2_identity") {
        continue;
      }
      if (v2.backend == fallback.backend && v2.gst_element_name == fallback.gst_element_name &&
          v2.stream_id == fallback.stream_id) {
        return true;
      }
    }
    return false;
  };
  for (auto& [key, agg] : plugin_aggs) {
    (void)key;
    if (duplicated_by_v2(agg.metric)) {
      continue;
    }
    if (result.trace_loss_detected) {
      agg.metric.reliable = false;
    }
    result.plugin_metrics.push_back(std::move(agg.metric));
  }
  for (auto& [key, agg] : edge_aggs) {
    (void)key;
    finalize_edge_agg(agg);
    if (result.trace_loss_detected) {
      agg.metric.reliable = false;
    }
    result.edge_metrics.push_back(std::move(agg.metric));
  }
  return result;
}

} // namespace simaai::neat::pipeline_internal

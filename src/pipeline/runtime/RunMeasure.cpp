#include "pipeline/Run.h"
#include "EdgeMetrics.h"
#include "LttngMetricsCollector.h"
#include "PathTimingBuilder.h"
#include "RunInternal.h"
#include "TraceAttribution.h"
#include "gst/TraceIdentity.h"
#include "pipeline/GraphMetrics.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
namespace {

using Clock = std::chrono::steady_clock;

RunStats delta_counters(const RunStats& before, const RunStats& after) {
  RunStats out;
  out.inputs_enqueued = after.inputs_enqueued >= before.inputs_enqueued
                            ? after.inputs_enqueued - before.inputs_enqueued
                            : 0;
  out.inputs_dropped = after.inputs_dropped >= before.inputs_dropped
                           ? after.inputs_dropped - before.inputs_dropped
                           : 0;
  out.inputs_pushed =
      after.inputs_pushed >= before.inputs_pushed ? after.inputs_pushed - before.inputs_pushed : 0;
  out.outputs_ready =
      after.outputs_ready >= before.outputs_ready ? after.outputs_ready - before.outputs_ready : 0;
  out.outputs_pulled = after.outputs_pulled >= before.outputs_pulled
                           ? after.outputs_pulled - before.outputs_pulled
                           : 0;
  out.outputs_dropped = after.outputs_dropped >= before.outputs_dropped
                            ? after.outputs_dropped - before.outputs_dropped
                            : 0;
  return out;
}

MeasureInputStats delta_input_stats(const InputStreamStats& before, const InputStreamStats& after) {
  const auto delta = [](std::uint64_t b, std::uint64_t a) -> std::uint64_t {
    return a >= b ? a - b : 0;
  };
  const auto delta_avg = [](double before_avg, std::uint64_t before_count, double after_avg,
                            std::uint64_t after_count, std::uint64_t delta_count) -> double {
    if (delta_count == 0) {
      return 0.0;
    }
    const double before_total = before_avg * static_cast<double>(before_count);
    const double after_total = after_avg * static_cast<double>(after_count);
    if (after_total < before_total) {
      return after_avg;
    }
    return (after_total - before_total) / static_cast<double>(delta_count);
  };

  MeasureInputStats out;
  out.push_count = delta(before.push_count, after.push_count);
  out.push_failures = delta(before.push_failures, after.push_failures);
  out.pull_count = delta(before.pull_count, after.pull_count);
  out.poll_count = delta(before.poll_count, after.poll_count);
  out.dropped_frames = delta(before.dropped_frames, after.dropped_frames);
  out.renegotiations = delta(before.renegotiations, after.renegotiations);
  out.alloc_grows = delta(before.alloc_grows, after.alloc_grows);
  out.growth_blocked = delta(before.growth_blocked, after.growth_blocked);
  out.renegotiation_blocked = delta(before.renegotiation_blocked, after.renegotiation_blocked);
  out.avg_alloc_us = delta_avg(before.avg_alloc_us, before.push_count, after.avg_alloc_us,
                               after.push_count, out.push_count);
  out.avg_map_us = delta_avg(before.avg_map_us, before.push_count, after.avg_map_us,
                             after.push_count, out.push_count);
  out.avg_copy_us = delta_avg(before.avg_copy_us, before.push_count, after.avg_copy_us,
                              after.push_count, out.push_count);
  out.avg_push_us = delta_avg(before.avg_push_us, before.push_count, after.avg_push_us,
                              after.push_count, out.push_count);
  out.avg_pull_wait_us = delta_avg(before.avg_pull_wait_us, before.pull_count,
                                   after.avg_pull_wait_us, after.pull_count, out.pull_count);
  out.avg_decode_us = delta_avg(before.avg_decode_us, before.pull_count, after.avg_decode_us,
                                after.pull_count, out.pull_count);
  return out;
}

MeasureLatencyStats summarize_samples(std::vector<double> samples) {
  MeasureLatencyStats out;
  out.count = samples.size();
  if (samples.empty())
    return out;

  const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  out.avg_ms = sum / static_cast<double>(samples.size());
  std::sort(samples.begin(), samples.end());

  auto percentile = [&](double p) {
    if (samples.empty())
      return 0.0;
    const double index = p * static_cast<double>(samples.size() - 1U);
    const auto lo = static_cast<std::size_t>(index);
    const auto hi = std::min(lo + 1U, samples.size() - 1U);
    const double frac = index - static_cast<double>(lo);
    return samples[lo] + ((samples[hi] - samples[lo]) * frac);
  };

  out.p50_ms = percentile(0.50);
  out.p90_ms = percentile(0.90);
  out.p95_ms = percentile(0.95);
  out.p99_ms = percentile(0.99);
  out.max_ms = samples.back();
  return out;
}

void validate_options(const MeasureOptions& opt) {
  if (opt.duration_ms <= 0)
    throw std::runtime_error("MeasureOptions::duration_ms must be > 0");
  if (opt.warmup_ms < 0)
    throw std::runtime_error("MeasureOptions::warmup_ms must be >= 0");
  if (opt.timeout_ms <= 0)
    throw std::runtime_error("MeasureOptions::timeout_ms must be > 0");
}

MetricsTraceSource apply_metrics_trace_env(MetricsTraceSource source) {
  const char* env = std::getenv("SIMA_NEAT_METRICS_TRACE_SOURCE");
  if (!env || !*env) {
    return source;
  }
  std::string value(env);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (value == "off") {
    return MetricsTraceSource::Off;
  }
  if (value == "lttng") {
    return MetricsTraceSource::Lttng;
  }
  return MetricsTraceSource::Auto;
}

bool trace_source_requests_lttng(MetricsTraceSource source) {
  return source == MetricsTraceSource::Auto || source == MetricsTraceSource::Lttng;
}

pipeline_internal::InternalMetricsTraceOptions make_lttng_options(const MeasureOptions& opt,
                                                                  bool enable_message_events) {
  pipeline_internal::InternalMetricsTraceOptions out;
  out.trace_dir = opt.metrics_trace_dir;
  out.retain_trace = opt.retain_metrics_trace;
  out.enable_message_events = enable_message_events;
  out.enable_pipeline_fallback = true;
  if (const char* retain = std::getenv("SIMA_NEAT_METRICS_RETAIN_TRACE")) {
    out.retain_trace = std::string(retain) == "1" || std::string(retain) == "true";
  }
  if (const char* dir = std::getenv("SIMA_NEAT_METRICS_TRACE_DIR")) {
    if (*dir) {
      out.trace_dir = dir;
    }
  }
  if (const char* remote = std::getenv("SIMA_NEAT_LTTNG_ENABLE_REMOTE_CORE")) {
    out.enable_remote_core_debug = std::string(remote) == "1" || std::string(remote) == "true";
  }
  return out;
}

pipeline_internal::TraceIdentityContext trace_identity_for_run(const Run& run) {
  pipeline_internal::TraceIdentityContext ctx;
  const std::shared_ptr<const runtime::RunCore> core = run_internal::core(run);
  if (core) {
    ctx.run_id_hash = pipeline_internal::stable_trace_hash(core->run_id);
    if (core->graph_execution_) {
      const auto& plan = core->graph_execution().plan;
      ctx.graph_id_hash = pipeline_internal::stable_trace_hash(
          std::to_string(plan.public_graph_id) + ":" + std::to_string(plan.public_graph_version));
    } else {
      ctx.graph_id_hash = ctx.run_id_hash;
    }
  }
  return ctx;
}

void print_latency_row(std::ostream& os, const char* name, const MeasureLatencyStats& s) {
  os << std::left << std::setw(28) << name << std::right << std::setw(9) << s.count << std::setw(11)
     << s.avg_ms << std::setw(11) << s.p50_ms << std::setw(11) << s.p90_ms << std::setw(11)
     << s.p95_ms << std::setw(11) << s.p99_ms << std::setw(11) << s.max_ms << "\n";
}

void print_path_row(std::ostream& os, const std::string& item, const std::string& stream,
                    const std::string& semantics, const MeasurePathStat& stat) {
  os << std::left << std::setw(26) << item.substr(0, 25) << std::setw(14)
     << (stream.empty() ? "-" : stream).substr(0, 13) << std::setw(28)
     << (semantics.empty() ? "-" : semantics).substr(0, 27) << std::right << std::setw(9)
     << stat.samples << std::setw(11) << stat.avg_ms << std::setw(11) << stat.p50_ms
     << std::setw(11) << stat.p95_ms << std::setw(11) << stat.max_ms << std::setw(10)
     << (stat.reliable ? "yes" : "no") << "\n";
}

std::string join_strings(const std::vector<std::string>& values, const char* sep = ",") {
  std::ostringstream os;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      os << sep;
    }
    os << values[i];
  }
  return os.str();
}

std::string node_display_name(const GraphNodeMetrics& node) {
  if (!node.public_node_ids.empty()) {
    return join_strings(node.public_node_ids);
  }
  if (!node.node_id.empty()) {
    return node.node_id;
  }
  if (node.runtime_node_id != kInvalidRuntimeNodeId) {
    return "n" + std::to_string(node.runtime_node_id);
  }
  return "-";
}

std::string node_label_or_kind(const GraphNodeMetrics& node) {
  if (!node.label.empty()) {
    return node.label;
  }
  if (!node.kind.empty()) {
    return node.kind;
  }
  return "-";
}

std::string plugin_node_display_name(const MeasurePluginLatency& plugin) {
  if (!plugin.public_node_ids.empty()) {
    return join_strings(plugin.public_node_ids);
  }
  if (plugin.public_node_id >= 0) {
    return "p" + std::to_string(plugin.public_node_id);
  }
  if (plugin.runtime_node_id >= 0) {
    return "n" + std::to_string(plugin.runtime_node_id);
  }
  return "-";
}

PowerMonitorOptions measurement_power_options_for_run(const Run& run) {
  const std::shared_ptr<const runtime::RunCore> core = run_internal::core(run);
  if (!core) {
    return {};
  }
  if (core->graph_execution_) {
    return core->graph_options.power_monitor;
  }
  return core->opt.power_monitor;
}

NodeLatencySummary delta_latency_summary(const NodeLatencySummary& before,
                                         const NodeLatencySummary& after) {
  NodeLatencySummary out;
  out.samples = after.samples >= before.samples ? after.samples - before.samples : after.samples;
  out.total_ms =
      after.total_ms >= before.total_ms ? after.total_ms - before.total_ms : after.total_ms;
  out.avg_ms = out.samples > 0 ? out.total_ms / static_cast<double>(out.samples) : 0.0;
  // Cumulative min/max counters cannot be subtracted exactly. Leave these unset for
  // measured-window deltas until window-local min/max counters exist.
  out.min_ms = 0.0;
  out.max_ms = 0.0;
  out.min_max_available = false;
  return out;
}

std::vector<GraphNodeMetrics> delta_node_metrics(const GraphMetricsReport& before,
                                                 const GraphMetricsReport& after) {
  using NodeKey = std::pair<std::size_t, RuntimeNodeId>;
  std::map<NodeKey, GraphNodeMetrics> before_by_node;
  for (const GraphNodeMetrics& node : before.node_metrics) {
    before_by_node[{node.pipeline_segment_id, node.runtime_node_id}] = node;
  }

  std::vector<GraphNodeMetrics> out;
  out.reserve(after.node_metrics.size());
  for (const GraphNodeMetrics& after_node : after.node_metrics) {
    GraphNodeMetrics delta = after_node;
    const auto before_it =
        before_by_node.find({after_node.pipeline_segment_id, after_node.runtime_node_id});
    const GraphNodeMetrics* before_node =
        before_it == before_by_node.end() ? nullptr : &before_it->second;

    const NodeLatencySummary empty_latency;
    delta.latency = delta_latency_summary(before_node ? before_node->latency : empty_latency,
                                          after_node.latency);

    std::map<std::string, GraphElementMetrics> before_elements;
    if (before_node) {
      for (const GraphElementMetrics& element : before_node->elements) {
        before_elements[element.name] = element;
      }
    }
    delta.elements.clear();
    for (const GraphElementMetrics& after_element : after_node.elements) {
      GraphElementMetrics element_delta = after_element;
      const auto before_element = before_elements.find(after_element.name);
      if (before_element != before_elements.end()) {
        element_delta.latency =
            delta_latency_summary(before_element->second.latency, after_element.latency);
      } else {
        element_delta.latency = delta_latency_summary(empty_latency, after_element.latency);
      }
      delta.elements.push_back(std::move(element_delta));
    }

    if (delta.latency.samples > 0 || delta.latency.total_ms > 0.0 || !delta.element_names.empty() ||
        !delta.elements.empty()) {
      out.push_back(std::move(delta));
    }
  }
  return out;
}

} // namespace

struct MeasureScope::Impl {
  Run* run = nullptr;
  MeasureOptions options;
  RunStats before{};
  InputStreamStats before_input{};
  RunDiagSnapshot before_diag{};
  GraphMetricsReport before_graph_metrics{};
  std::vector<pipeline_internal::GraphQueueLatencySnapshot> before_graph_queue_metrics;
  std::uint64_t before_graph_sample_timing_unkeyed = 0;
  std::uint64_t before_graph_sample_timing_misses = 0;
  Clock::time_point start{};
  std::unique_ptr<pipeline_internal::LttngMetricsCollector> lttng_metrics;
  pipeline_internal::TraceIdentityContext trace_context{};
  bool lttng_trace_identity_applied = false;
  std::unique_ptr<PowerMonitor> power_monitor;
  std::vector<std::string> warnings;
  std::string plugin_latency_status = "off";
  std::string plugin_latency_source = "none";
  std::string message_latency_status = "off";
  std::string message_latency_source = "none";
  bool stopped = false;
  MeasureReport cached{};
};

void MeasureScope::disable_lttng_trace_identity_noexcept(Impl* impl) {
  if (!impl || !impl->lttng_trace_identity_applied || !impl->run) {
    return;
  }
  try {
    const GraphMetricsReport graph_metrics = build_graph_metrics_report_run_lifetime(
        *impl->run, GraphMetricsOptions{.include_power = false});
    pipeline_internal::apply_lttng_trace_identity(*impl->run, graph_metrics.node_metrics,
                                                  impl->trace_context.run_id_hash,
                                                  impl->trace_context.graph_id_hash, false);
  } catch (...) {
  }
  impl->lttng_trace_identity_applied = false;
}

MeasureScope::MeasureScope(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MeasureScope::MeasureScope(MeasureScope&&) noexcept = default;
MeasureScope& MeasureScope::operator=(MeasureScope&& other) noexcept {
  if (this != &other) {
    if (impl_ && !impl_->stopped) {
      MeasureScope::disable_lttng_trace_identity_noexcept(impl_.get());
    }
    if (impl_ && !impl_->stopped && impl_->run && impl_->run->core_) {
      auto st = impl_->run->core_;
      std::lock_guard<std::mutex> lock(st->latency_mu);
      st->measurement_active = false;
      st->measurement_output_timing_init = false;
      st->measurement_latencies_ms.clear();
      st->measurement_frame_gaps_ms.clear();
      st->measurement_graph_entries.clear();
      st->measurement_graph_pulls.clear();
    }
    if (impl_ && impl_->run) {
      pipeline_internal::set_graph_queue_timing_enabled(*impl_->run, false);
    }
    if (impl_ && impl_->power_monitor) {
      impl_->power_monitor->stop();
    }
    impl_ = std::move(other.impl_);
  }
  return *this;
}
MeasureScope::~MeasureScope() {
  if (impl_ && !impl_->stopped) {
    MeasureScope::disable_lttng_trace_identity_noexcept(impl_.get());
  }
  if (impl_ && !impl_->stopped && impl_->run && impl_->run->core_) {
    auto st = impl_->run->core_;
    std::lock_guard<std::mutex> lock(st->latency_mu);
    st->measurement_active = false;
    st->measurement_output_timing_init = false;
    st->measurement_latencies_ms.clear();
    st->measurement_frame_gaps_ms.clear();
    st->measurement_graph_entries.clear();
    st->measurement_graph_pulls.clear();
  }
  if (impl_ && !impl_->stopped && impl_->run) {
    pipeline_internal::set_graph_queue_timing_enabled(*impl_->run, false);
  }
  if (impl_ && impl_->power_monitor) {
    impl_->power_monitor->stop();
  }
}

bool MeasureScope::stopped() const {
  return !impl_ || impl_->stopped;
}

MeasureReport MeasureScope::stop() {
  if (!impl_)
    throw std::runtime_error("MeasureScope::stop: invalid moved-from scope");
  if (impl_->stopped)
    return impl_->cached;

  const auto end = Clock::now();
  const RunStats after = run_internal::stats(*impl_->run);
  const RunStats measured = delta_counters(impl_->before, after);
  const InputStreamStats after_input = run_internal::input_stats(*impl_->run);

  MeasureReport report;
  report.options = impl_->options;
  if (report.options.logical_batch_size <= 0)
    report.options.logical_batch_size = 1;
  report.counters.inputs_enqueued = measured.inputs_enqueued;
  report.counters.inputs_pushed = measured.inputs_pushed;
  report.counters.outputs_ready = measured.outputs_ready;
  report.counters.outputs_pulled = measured.outputs_pulled;
  report.counters.inputs_dropped = measured.inputs_dropped;
  report.counters.outputs_dropped = measured.outputs_dropped;
  report.input = delta_input_stats(impl_->before_input, after_input);
  report.outputs = static_cast<std::size_t>(measured.outputs_pulled);
  report.elapsed_s = std::chrono::duration<double>(end - impl_->start).count();
  report.throughput_batches_per_s =
      report.elapsed_s > 0.0 ? static_cast<double>(report.outputs) / report.elapsed_s : 0.0;
  report.throughput_inferences_per_s =
      report.throughput_batches_per_s * static_cast<double>(report.options.logical_batch_size);
  report.plugin_latency_status = impl_->plugin_latency_status;
  report.plugin_latency_source = impl_->plugin_latency_source;
  report.message_latency_status = impl_->message_latency_status;
  report.message_latency_source = impl_->message_latency_source;
  report.warnings = impl_->warnings;
  std::vector<double> latency_samples;
  std::vector<double> frame_gap_samples;
  std::vector<runtime::GraphSampleTimingEvent> graph_entry_events;
  std::vector<runtime::GraphSampleTimingEvent> graph_pull_events;
  if (impl_->run->core_) {
    auto st = impl_->run->core_;
    std::lock_guard<std::mutex> lock(st->latency_mu);
    st->measurement_active = false;
    latency_samples = st->measurement_latencies_ms;
    frame_gap_samples = st->measurement_frame_gaps_ms;
    graph_entry_events = st->measurement_graph_entries;
    graph_pull_events = st->measurement_graph_pulls;
    st->measurement_latencies_ms.clear();
    st->measurement_frame_gaps_ms.clear();
    st->measurement_graph_entries.clear();
    st->measurement_graph_pulls.clear();
    st->measurement_output_timing_init = false;
    const std::uint64_t unkeyed = st->graph_sample_timing_unkeyed.load(std::memory_order_relaxed);
    const std::uint64_t misses = st->graph_sample_timing_misses.load(std::memory_order_relaxed);
    report.graph_sample_timing_unkeyed = unkeyed >= impl_->before_graph_sample_timing_unkeyed
                                             ? unkeyed - impl_->before_graph_sample_timing_unkeyed
                                             : unkeyed;
    report.graph_sample_timing_misses = misses >= impl_->before_graph_sample_timing_misses
                                            ? misses - impl_->before_graph_sample_timing_misses
                                            : misses;
  }
  report.end_to_end = summarize_samples(std::move(latency_samples));
  report.frame_gap = summarize_samples(std::move(frame_gap_samples));
  report.latency_samples_collected = report.end_to_end.count > 0 || report.frame_gap.count > 0;
  const GraphMetricsReport after_graph_metrics = build_graph_metrics_report_run_lifetime(
      *impl_->run, GraphMetricsOptions{.include_power = false});
  report.node_metrics = delta_node_metrics(impl_->before_graph_metrics, after_graph_metrics);
  const RunDiagSnapshot after_diag = run_internal::diag_snapshot(*impl_->run);
  if (impl_->options.include_power) {
    if (impl_->power_monitor) {
      impl_->power_monitor->stop();
      report.power = impl_->power_monitor->summary();
      impl_->power_monitor.reset();
    } else {
      report.power = run_internal::power_summary(*impl_->run);
    }
  }
  if (impl_->lttng_metrics) {
    std::string err;
    if (!impl_->lttng_metrics->stop_and_destroy(&err)) {
      report.warnings.push_back("LTTng metrics stop/destroy failed: " + err);
      if (impl_->options.include_plugin_latency) {
        report.plugin_latency_status = "failed";
      }
      if (impl_->options.include_message_latency) {
        report.message_latency_status = "failed";
      }
    } else {
      pipeline_internal::LttngParseResult parsed = impl_->lttng_metrics->parse(&err);
      if (!parsed.parsed) {
        report.warnings.push_back("LTTng metrics parse failed: " + err);
        impl_->lttng_metrics->retain_trace_for_debug();
        report.metrics_trace_dir = impl_->lttng_metrics->trace_dir();
        if (impl_->options.include_plugin_latency) {
          report.plugin_latency_status = "failed";
        }
        if (impl_->options.include_message_latency) {
          report.message_latency_status = "failed";
        }
      } else {
        report.plugin_latency = std::move(parsed.plugin_metrics);
        report.plugin_latency_unattributed = std::move(parsed.plugin_metrics_unattributed);
        report.edge_latency.insert(report.edge_latency.end(),
                                   std::make_move_iterator(parsed.edge_metrics.begin()),
                                   std::make_move_iterator(parsed.edge_metrics.end()));
        report.edge_latency_unattributed.insert(
            report.edge_latency_unattributed.end(),
            std::make_move_iterator(parsed.edge_metrics_unattributed.begin()),
            std::make_move_iterator(parsed.edge_metrics_unattributed.end()));
        for (auto& warning : parsed.warnings) {
          report.warnings.push_back(std::move(warning));
        }
        if (impl_->options.include_message_latency && impl_->run->core_ &&
            impl_->run->core_->graph_execution_) {
          report.path_timing =
              runtime::build_path_timing(impl_->run->core_->graph_execution_->plan, parsed,
                                         graph_entry_events, graph_pull_events);
        }
        if (parsed.trace_loss_detected) {
          report.trace_loss_detected = true;
          report.warnings.push_back(
              "LTTng trace loss detected; plugin/edge latencies may be incomplete");
        }
        if (impl_->lttng_metrics->retained()) {
          report.metrics_trace_dir = parsed.trace_dir;
        }
        if (impl_->options.include_plugin_latency && report.plugin_latency.empty() &&
            report.plugin_latency_unattributed.empty()) {
          report.plugin_latency_status = "unavailable";
          report.warnings.push_back(
              "LTTng plugin latency collected no plugin spans; no trace-capable plugin emitted "
              "during the measured window or the active plugin build lacks metrics tracepoints");
        }
        if (impl_->options.include_message_latency && parsed.raw_edge_spans.empty() &&
            parsed.raw_graph_events.empty()) {
          report.message_latency_status = "unavailable";
          report.warnings.push_back(
              "LTTng message latency requested but no sima_neat_edge:message spans were collected");
        } else if (impl_->options.include_message_latency && parsed.raw_edge_spans.empty()) {
          report.warnings.push_back(
              "LTTng message latency collected graph lifecycle events but no edge transport spans");
        }
      }
    }
    impl_->lttng_metrics->cleanup_noexcept();
    impl_->lttng_metrics.reset();
  }
  if (impl_->lttng_trace_identity_applied) {
    pipeline_internal::apply_lttng_trace_identity(*impl_->run, after_graph_metrics.node_metrics,
                                                  impl_->trace_context.run_id_hash,
                                                  impl_->trace_context.graph_id_hash, false);
    impl_->lttng_trace_identity_applied = false;
  }

  if (impl_->options.include_edge_latency) {
    auto diag_edges =
        pipeline_internal::build_edge_latency_from_diag_delta(impl_->before_diag, after_diag);
    report.edge_latency.insert(report.edge_latency.end(),
                               std::make_move_iterator(diag_edges.begin()),
                               std::make_move_iterator(diag_edges.end()));
    auto after_queue = pipeline_internal::snapshot_graph_queue_latencies(*impl_->run);
    pipeline_internal::set_graph_queue_timing_enabled(*impl_->run, false);
    auto queue_edges = pipeline_internal::build_graph_queue_latency_delta(
        impl_->before_graph_queue_metrics, after_queue);
    report.edge_latency.insert(report.edge_latency.end(),
                               std::make_move_iterator(queue_edges.begin()),
                               std::make_move_iterator(queue_edges.end()));
    if (!report.edge_latency.empty() && report.message_latency_status == "off") {
      report.message_latency_status = "collected";
      report.message_latency_source = "diagnostics";
    }
  } else {
    pipeline_internal::set_graph_queue_timing_enabled(*impl_->run, false);
  }
  if (!impl_->options.include_message_latency) {
    report.path_timing.available = false;
    report.path_timing.status = report.edge_latency.empty() ? "unavailable_message_trace_disabled"
                                                            : "diagnostic_aggregate_only";
    report.path_timing.source = report.edge_latency.empty() ? "none" : "diagnostics";
    report.path_timing.warnings.push_back(
        "Exact path timing requires MeasureOptions.include_message_latency=true.");
  } else if (report.path_timing.status.empty()) {
    report.path_timing.available = false;
    report.path_timing.status = "unavailable_message_trace_disabled";
    report.path_timing.source = "none";
    report.path_timing.reason = "No exact message trace data was available for path timing.";
  }
  pipeline_internal::attribute_edge_latency_to_nodes(report.node_metrics, &report.edge_latency,
                                                     &report.edge_latency_unattributed);
  pipeline_internal::attribute_plugin_latency_to_nodes(report.node_metrics, &report.plugin_latency,
                                                       &report.plugin_latency_unattributed);

  impl_->cached = report;
  impl_->stopped = true;
  return impl_->cached;
}

MeasureScope Run::start_measurement(const MeasureOptions& opt) {
  validate_options(opt);
  auto impl = std::unique_ptr<MeasureScope::Impl>(new MeasureScope::Impl());
  impl->run = this;
  impl->options = opt;
  impl->before = run_internal::stats(*this);
  impl->before_input = run_internal::input_stats(*this);
  impl->before_diag = run_internal::diag_snapshot(*this);
  impl->before_graph_metrics =
      build_graph_metrics_report_run_lifetime(*this, GraphMetricsOptions{.include_power = false});
  if (core_) {
    impl->before_graph_sample_timing_unkeyed =
        core_->graph_sample_timing_unkeyed.load(std::memory_order_relaxed);
    impl->before_graph_sample_timing_misses =
        core_->graph_sample_timing_misses.load(std::memory_order_relaxed);
  }
  pipeline_internal::set_graph_queue_timing_enabled(*this, opt.include_edge_latency ||
                                                               opt.include_message_latency);
  impl->before_graph_queue_metrics = pipeline_internal::snapshot_graph_queue_latencies(*this);

  MetricsTraceSource plugin_source = apply_metrics_trace_env(opt.plugin_latency_source);
  MetricsTraceSource message_source = apply_metrics_trace_env(opt.message_latency_source);
  if (!opt.include_plugin_latency || plugin_source == MetricsTraceSource::Off) {
    impl->plugin_latency_status = "off";
    impl->plugin_latency_source = "none";
  }
  if (!opt.include_message_latency || message_source == MetricsTraceSource::Off) {
    impl->message_latency_status = "off";
    impl->message_latency_source = "none";
  }

  const bool effective_plugin_lttng = opt.include_plugin_latency &&
                                      plugin_source != MetricsTraceSource::Off &&
                                      trace_source_requests_lttng(plugin_source);
  const bool effective_message_lttng = opt.include_message_latency &&
                                       message_source != MetricsTraceSource::Off &&
                                       trace_source_requests_lttng(message_source);
  const bool need_lttng = effective_plugin_lttng || effective_message_lttng;
  if (need_lttng) {
    auto lttng_opt = make_lttng_options(opt, effective_message_lttng);
    impl->trace_context = trace_identity_for_run(*this);
    pipeline_internal::apply_lttng_trace_identity(
        *this, impl->before_graph_metrics.node_metrics, impl->trace_context.run_id_hash,
        impl->trace_context.graph_id_hash, true, effective_message_lttng);
    impl->lttng_trace_identity_applied = true;
    impl->lttng_metrics = std::make_unique<pipeline_internal::LttngMetricsCollector>(
        std::move(lttng_opt), impl->trace_context);
    std::string err;
    if (!impl->lttng_metrics->start(&err)) {
      pipeline_internal::apply_lttng_trace_identity(*this, impl->before_graph_metrics.node_metrics,
                                                    impl->trace_context.run_id_hash,
                                                    impl->trace_context.graph_id_hash, false);
      impl->lttng_trace_identity_applied = false;
      const bool required =
          plugin_source == MetricsTraceSource::Lttng || message_source == MetricsTraceSource::Lttng;
      impl->lttng_metrics.reset();
      if (required) {
        pipeline_internal::set_graph_queue_timing_enabled(*this, false);
        throw std::runtime_error("LTTng metrics start failed: " + err);
      }
      impl->warnings.push_back("LTTng metrics unavailable: " + err);
      if (opt.include_plugin_latency && plugin_source != MetricsTraceSource::Off) {
        impl->plugin_latency_status = "unavailable";
        impl->plugin_latency_source = "none";
      }
      if (opt.include_message_latency && message_source != MetricsTraceSource::Off) {
        impl->message_latency_status = "unavailable";
        impl->message_latency_source = "none";
      }
    } else {
      if (opt.include_plugin_latency && plugin_source != MetricsTraceSource::Off) {
        impl->plugin_latency_status = "collected";
        impl->plugin_latency_source = "lttng";
      }
      if (opt.include_message_latency && message_source != MetricsTraceSource::Off) {
        impl->message_latency_status = "collected";
        impl->message_latency_source = "lttng";
      }
    }
  }
  if (opt.include_power) {
    const PowerMonitorOptions power_opt = measurement_power_options_for_run(*this);
    if (power_opt.enabled) {
      impl->power_monitor = std::make_unique<PowerMonitor>(power_opt);
      impl->power_monitor->start();
    }
  }
  impl->start = Clock::now();
  if (core_) {
    std::lock_guard<std::mutex> lock(core_->latency_mu);
    core_->measurement_latencies_ms.clear();
    core_->measurement_frame_gaps_ms.clear();
    core_->measurement_graph_entries.clear();
    core_->measurement_graph_pulls.clear();
    core_->measurement_started_at = impl->start;
    core_->measurement_output_timing_init = false;
    core_->measurement_active = true;
  }
  return MeasureScope(std::move(impl));
}

std::string MeasureReport::to_text() const {
  std::ostringstream os;
  os << std::fixed << std::setprecision(3);
  os << "\n============================================================\n"
     << " " << (options.title.empty() ? "NEAT measurement" : options.title) << "\n"
     << "============================================================\n";
  if (!options.model.empty())
    os << "Model                 : " << options.model << "\n";
  if (options.logical_batch_size > 0) {
    os << "Compiled batch size   : " << options.logical_batch_size << "\n";
  }
  if (!options.input.empty())
    os << "Input                 : " << options.input << "\n";
  if (!options.placement.empty())
    os << "Placement             : " << options.placement << "\n";
  os << "Measure mode          : graph-level measured window\n";
  if (warmup_iterations > 0) {
    os << "Warmup iterations     : " << warmup_iterations << " (excluded from all numbers below)\n";
  }
  if (options.warmup_ms > 0) {
    os << "Warmup request        : " << options.warmup_ms
       << " ms (caller-owned observer metadata)\n";
  }
  os << "Measured outputs      : " << outputs << "\n"
     << "Elapsed               : " << elapsed_s << " s\n"
     << "Throughput            : " << throughput_batches_per_s << " batches/s\n"
     << "Logical throughput    : " << throughput_inferences_per_s << " inferences/s\n";

  os << "\nGraph entry->public pull residency; excludes app decode/draw/write (ms):\n"
     << "  semantics: " << end_to_end_semantics << "\n"
     << "  note: single-flight ~= latency; async burst/queued windows include queue wait and must "
        "not be presented as standalone model latency.\n";
  if (latency_samples_collected) {
    os << std::left << std::setw(28) << "metric" << std::right << std::setw(9) << "count"
       << std::setw(11) << "avg" << std::setw(11) << "p50" << std::setw(11) << "p90"
       << std::setw(11) << "p95" << std::setw(11) << "p99" << std::setw(11) << "max" << "\n";
    print_latency_row(os, "queue-inclusive e2e", end_to_end);
    print_latency_row(os, "between output frames", frame_gap);
  } else {
    os << "  no graph-entry/public-pull timing samples were produced during this observer window\n";
  }

  os << "\nPer-node diagnostic latency during measured window (ms):\n";
  os << std::left << std::setw(18) << "node" << std::setw(30) << "label/kind" << std::right
     << std::setw(9) << "samples" << std::setw(11) << "avg" << std::setw(11) << "min"
     << std::setw(11) << "max" << std::setw(12) << "total" << "\n";
  if (node_metrics.empty()) {
    os << "  no node latency samples were reported\n";
  } else {
    for (const auto& node : node_metrics) {
      os << std::left << std::setw(18) << node_display_name(node).substr(0, 17) << std::setw(30)
         << node_label_or_kind(node).substr(0, 29) << std::right << std::setw(9)
         << node.latency.samples;
      if (node.latency.samples == 0 && node.latency.total_ms == 0.0) {
        os << std::setw(11) << "N/A" << std::setw(11) << "N/A" << std::setw(11) << "N/A"
           << std::setw(12) << "N/A";
      } else {
        os << std::setw(11) << node.latency.avg_ms;
        if (node.latency.min_max_available) {
          os << std::setw(11) << node.latency.min_ms << std::setw(11) << node.latency.max_ms;
        } else {
          os << std::setw(11) << "-" << std::setw(11) << "-";
        }
        os << std::setw(12) << node.latency.total_ms;
      }
      os << "\n";
    }
  }

  os << "\nPer-plugin / kernel latency during measured window (ms):\n"
     << "  note: diagnostic rows may be nested/overlapped; do not sum plugin totals.\n"
     << "  note: processcvu Exec/processcvu_dispatcher is a dispatcher/device window, not pure "
        "kernel math.\n";
  if (!plugin_latency_status.empty()) {
    os << "  source/status: " << (plugin_latency_source.empty() ? "none" : plugin_latency_source)
       << "/" << plugin_latency_status << "\n";
  }
  os << std::left << std::setw(28) << "plugin" << std::setw(10) << "phase" << std::setw(24)
     << "kernel" << std::setw(10) << "node" << std::setw(14) << "stream" << std::right
     << std::setw(9) << "calls" << std::setw(11) << "avg" << std::setw(11) << "min" << std::setw(11)
     << "max" << std::setw(12) << "total" << "\n";
  if (plugin_latency.empty()) {
    os << "  no plugin/kernel timing samples were reported\n";
  } else {
    for (const auto& p : plugin_latency) {
      os << std::left << std::setw(28) << p.name << std::setw(10)
         << (p.phase.empty() ? "-" : p.phase) << std::setw(24)
         << (p.kernel_name.empty() ? "-" : p.kernel_name).substr(0, 23) << std::setw(10)
         << plugin_node_display_name(p) << std::setw(14)
         << (p.stream_id.empty() ? "-" : p.stream_id).substr(0, 13) << std::right << std::setw(9)
         << p.calls << std::setw(11) << p.avg_ms << std::setw(11) << p.min_ms << std::setw(11)
         << p.max_ms << std::setw(12) << p.total_ms << "\n";
    }
  }
  if (!plugin_latency_unattributed.empty()) {
    os << "  unattributed plugin rows: " << plugin_latency_unattributed.size()
       << " (see JSON/HTML for mapping errors)\n";
  }

  os << "\nInter-plugin message / edge latency during measured window (ms):\n"
     << "  note: handoff/queue/transport diagnostics; do not add to plugin or graph latency.\n";
  os << std::left << std::setw(28) << "edge" << std::setw(18) << "semantics" << std::setw(14)
     << "source" << std::right << std::setw(9) << "samples" << std::setw(11) << "avg"
     << std::setw(11) << "p50" << std::setw(11) << "p95" << std::setw(11) << "max" << "\n";
  if (edge_latency.empty()) {
    os << "  no edge/message timing samples were reported\n";
  } else {
    for (const auto& e : edge_latency) {
      os << std::left << std::setw(28) << (e.name.empty() ? e.edge_id : e.name).substr(0, 27)
         << std::setw(18) << (e.timing_semantics.empty() ? "-" : e.timing_semantics).substr(0, 17)
         << std::setw(14) << (e.source.empty() ? "-" : e.source).substr(0, 13) << std::right
         << std::setw(9) << e.samples << std::setw(11) << e.avg_ms << std::setw(11) << e.p50_ms
         << std::setw(11) << e.p95_ms << std::setw(11) << e.max_ms << "\n";
    }
  }
  if (!edge_latency_unattributed.empty()) {
    os << "  unattributed edge/message rows: " << edge_latency_unattributed.size()
       << " (see JSON/HTML for reason codes)\n";
  }

  os << "\nPer-sample path timing during measured window (ms):\n"
     << "  source/status: " << (path_timing.source.empty() ? "none" : path_timing.source) << "/"
     << (path_timing.status.empty() ? "off" : path_timing.status) << "\n";
  if (!path_timing.reason.empty()) {
    os << "  reason: " << path_timing.reason << "\n";
  }
  os << std::left << std::setw(26) << "item" << std::setw(14) << "stream" << std::setw(28)
     << "semantics" << std::right << std::setw(9) << "samples" << std::setw(11) << "avg"
     << std::setw(11) << "p50" << std::setw(11) << "p95" << std::setw(11) << "max" << std::setw(10)
     << "reliable" << "\n";
  if (!path_timing.available) {
    os << "  no path timing rows were reported\n";
  } else {
    for (const auto& row : path_timing.node_arrival) {
      const std::string item =
          !row.customer_node_id.empty()
              ? row.customer_node_id
              : (!row.lowered_node_id.empty() ? row.lowered_node_id
                                              : std::to_string(row.runtime_node_id));
      print_path_row(os, item, row.stream_id, row.semantics, row.latency);
    }
    for (const auto& row : path_timing.inter_plugin_gap) {
      const std::string item =
          !row.customer_edge_id.empty() ? row.customer_edge_id : row.lowered_edge_id;
      print_path_row(os, item.empty() ? "-" : item, row.stream_id, row.semantics, row.latency);
    }
    for (const auto& row : path_timing.output_tail) {
      const std::string item =
          !row.output_endpoint.empty() ? row.output_endpoint : row.customer_output_node_id;
      print_path_row(os, item.empty() ? "-" : item, row.stream_id, row.semantics, row.latency);
    }
  }
  for (const std::string& warning : path_timing.warnings) {
    os << "  path warning: " << warning << "\n";
  }

  os << "\nNEAT counters during measured window:\n"
     << "  inputs_enqueued      : " << counters.inputs_enqueued << "\n"
     << "  inputs_pushed        : " << counters.inputs_pushed << "\n"
     << "  outputs_ready        : " << counters.outputs_ready << "\n"
     << "  outputs_pulled       : " << counters.outputs_pulled << "\n"
     << "  inputs_dropped       : " << counters.inputs_dropped << "\n"
     << "  outputs_dropped      : " << counters.outputs_dropped << "\n"
     << "  graph_timing_unkeyed : " << graph_sample_timing_unkeyed << "\n"
     << "  graph_timing_misses  : " << graph_sample_timing_misses << "\n";

  os << "\nInput boundary during measured window:\n"
     << "  push_count           : " << input.push_count << "\n"
     << "  push_failures        : " << input.push_failures << "\n"
     << "  pull_count           : " << input.pull_count << "\n"
     << "  poll_count           : " << input.poll_count << "\n"
     << "  dropped_frames       : " << input.dropped_frames << "\n"
     << "  renegotiations       : " << input.renegotiations << "\n"
     << "  avg_alloc_us         : " << input.avg_alloc_us << "\n"
     << "  avg_map_us           : " << input.avg_map_us << "\n"
     << "  avg_copy_us          : " << input.avg_copy_us << "\n"
     << "  avg_push_us          : " << input.avg_push_us << "\n"
     << "  avg_pull_wait_us     : " << input.avg_pull_wait_us << "\n"
     << "  avg_decode_us        : " << input.avg_decode_us << "\n";

  if (power.enabled && !power.rails.empty()) {
    os << "\nPower telemetry:\n"
       << "  note: graph-level board/SOM rail telemetry; DVT readings may be unavailable or "
          "unreliable.\n"
       << format_power_summary(power);
  } else if (power.enabled) {
    os << "\nPower telemetry:\n"
       << "  requested, but no rail samples were collected during the measured window\n"
       << "  note: DVT board power can be unavailable/unreliable; keep enabled for SOM "
          "validation\n";
  } else {
    os << "\nPower telemetry:\n"
       << "  disabled for this run; enable on Modalix SOMs for graph-level board rail average "
          "power\n";
  }
  if (!warnings.empty()) {
    os << "\nMetrics warnings:\n";
    for (const std::string& warning : warnings) {
      os << "  - " << warning << "\n";
    }
  }
  return os.str();
}

} // namespace simaai::neat

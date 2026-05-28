#include "RunCore.h"

#include "ExecutionGraphPlan.h"
#include "gst/GstInit.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/internal/BuildTiming.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <mutex>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace simaai::neat::graph {
bool graph_debug_enabled();
bool graph_sched_debug_enabled();
void graph_debug_sample(const char* tag, const Sample& sample);
void graph_sched_record(NodeId node_id, const std::string& label, const Sample& sample);
std::size_t identity_map_capacity();
bool has_input_appsrc(std::span<const std::shared_ptr<simaai::neat::Node>> nodes);
bool has_output_appsink(std::span<const std::shared_ptr<simaai::neat::Node>> nodes);
bool has_internal_source(std::span<const std::shared_ptr<simaai::neat::Node>> nodes);
InputOptions input_opts_from_spec(const OutputSpec& spec, bool complete);
} // namespace simaai::neat::graph

namespace simaai::neat::runtime {
namespace {

std::string make_run_uuid() {
  std::array<unsigned char, 16> bytes{};
  std::random_device rd;
  for (unsigned char& b : bytes) {
    b = static_cast<unsigned char>(rd());
  }
  bytes[6] = static_cast<unsigned char>((bytes[6] & 0x0FU) | 0x40U);
  bytes[8] = static_cast<unsigned char>((bytes[8] & 0x3FU) | 0x80U);

  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    if (i == 4U || i == 6U || i == 8U || i == 10U) {
      oss << '-';
    }
    oss << std::setw(2) << static_cast<unsigned int>(bytes[i]);
  }
  return oss.str();
}

bool graph_debug_enabled_for_core() {
  return pipeline_internal::env_bool("SIMA_GRAPH_DEBUG", false);
}

bool graph_serial_pipeline_build_enabled_for_core() {
  return pipeline_internal::env_bool("SIMA_GRAPH_SERIAL_PIPELINE_BUILD", false);
}

std::mutex& graph_pipeline_build_mu_for_core() {
  static std::mutex mu;
  return mu;
}

const char* graph_backpressure_timeout_explanation() {
  return " This can happen because of graph backpressure: downstream stages, appsinks, or the "
         "application are not draining outputs as fast as inputs are pushed, so an internal "
         "edge/pipeline queue filled before the timeout. Pull outputs concurrently, reduce the "
         "push rate, increase GraphRunOptions.edge_queue/push_timeout_ms, or remove/relax slow "
         "downstream stages.";
}

using SampleIdentity = PipelineSegmentRuntime::GraphTransport::SampleIdentity;

bool is_internal_stream_id(const std::string& id) {
  return !id.empty() && id.find("neatprocess") != std::string::npos;
}

SampleIdentity capture_sample_identity(const Sample& sample) {
  SampleIdentity id;
  id.frame_id = sample.frame_id;
  id.pts_ns = sample.pts_ns;
  id.dts_ns = sample.dts_ns;
  id.duration_ns = sample.duration_ns;
  id.input_seq = sample.input_seq;
  id.orig_input_seq = sample.orig_input_seq;
  id.stream_id = sample.stream_id;
  id.stream_label = sample.stream_label;
  id.port_name = sample.port_name;
  return id;
}

bool has_sample_identity(const SampleIdentity& id) {
  return id.frame_id >= 0 || id.pts_ns >= 0 || id.dts_ns >= 0 || id.duration_ns >= 0 ||
         id.input_seq >= 0 || id.orig_input_seq >= 0 || !id.stream_id.empty() ||
         !id.stream_label.empty() || !id.port_name.empty();
}

void restore_sample_identity_if_needed(Sample& sample, const SampleIdentity& id,
                                       bool prefer_mapped) {
  if (id.frame_id >= 0) {
    sample.frame_id = id.frame_id;
  }
  if (id.pts_ns >= 0) {
    sample.pts_ns = id.pts_ns;
  }
  if (id.dts_ns >= 0) {
    sample.dts_ns = id.dts_ns;
  }
  if (id.duration_ns >= 0) {
    sample.duration_ns = id.duration_ns;
  }
  if (sample.orig_input_seq < 0 && id.orig_input_seq >= 0) {
    sample.orig_input_seq = id.orig_input_seq;
  }
  if (!id.stream_id.empty() &&
      (prefer_mapped || sample.stream_id.empty() || is_internal_stream_id(sample.stream_id))) {
    sample.stream_id = id.stream_id;
  }
  if (sample.stream_label.empty() && !id.stream_label.empty()) {
    sample.stream_label = id.stream_label;
  }
  if (sample.port_name.empty() && !id.port_name.empty()) {
    sample.port_name = id.port_name;
  }
}

std::vector<std::shared_ptr<simaai::neat::Node>>
materialize_segment_nodes(const PipelineSegmentPlan& segment) {
  std::vector<std::shared_ptr<simaai::neat::Node>> nodes = segment.nodes;

  if (!segment.boundary.source_like && segment.input_edges.empty() &&
      simaai::neat::graph::has_internal_source(segment.nodes) &&
      !simaai::neat::graph::has_input_appsrc(segment.nodes)) {
    throw std::runtime_error(
        "RunCore::start_pipeline_segment: pipeline segment has internal source but is not "
        "source_like. Refusing to inject Input into a source pipeline.");
  }

  if (segment.boundary.needs_input && !segment.boundary.source_like &&
      !simaai::neat::graph::has_input_appsrc(segment.nodes)) {
    InputOptions opt_src;
    if (segment.boundary_hints.has_value() && !segment.boundary_hints->ingress_inputs.empty()) {
      opt_src = segment.boundary_hints->ingress_inputs.front();
    } else {
      opt_src =
          simaai::neat::graph::input_opts_from_spec(segment.input_spec, segment.input_complete);
    }
    nodes.insert(nodes.begin(), simaai::neat::nodes::Input(opt_src));
  }

  if (segment.boundary.needs_output && !simaai::neat::graph::has_output_appsink(segment.nodes)) {
    nodes.push_back(simaai::neat::nodes::Output());
  }

  return nodes;
}

} // namespace

void initialize_run_identity(RunCore& core) {
  core.created_at = std::chrono::steady_clock::now();
  core.created_wall_at = std::chrono::system_clock::now();
  core.run_id = make_run_uuid();
}

std::shared_ptr<RunCore> RunCore::create_graph_compat() {
  auto core = std::make_shared<RunCore>();
  initialize_run_identity(*core);
  core->graph_execution_ = std::make_unique<ExecutionGraphRuntime>();
  return core;
}

ExecutionGraphRuntime& RunCore::graph_execution() {
  if (!graph_execution_) {
    throw std::runtime_error("RunCore::graph_execution: graph runtime is not initialized");
  }
  return *graph_execution_;
}

const ExecutionGraphRuntime& RunCore::graph_execution() const {
  if (!graph_execution_) {
    throw std::runtime_error("RunCore::graph_execution: graph runtime is not initialized");
  }
  return *graph_execution_;
}

bool RunCore::graph_stop_requested() const {
  return stop_requested.load(std::memory_order_relaxed);
}

void RunCore::graph_signal_stop() {
  const bool already_stopping = stop_requested.exchange(true, std::memory_order_relaxed);
  if (!already_stopping && graph_debug_enabled_for_core()) {
    std::fprintf(stderr, "[GRAPH] signal_stop\n");
  }

  if (!graph_execution_)
    return;

  for (auto& stage : graph_execution_->stages) {
    if (stage) {
      if (!already_stopping && stage->exec) {
        try {
          stage->exec->request_stop();
        } catch (const std::exception& e) {
          if (graph_debug_enabled_for_core()) {
            std::fprintf(stderr, "[GRAPH] stage_request_stop_error node=%zu err=%s\n",
                         static_cast<std::size_t>(stage->node_id), e.what());
          }
        }
      }
      stage->mailbox.inbox.close();
    }
  }
  for (auto& sink : graph_execution_->sinks) {
    if (sink.second)
      sink.second->close();
  }
  for (auto& pipe : graph_execution_->pipelines) {
    if (!pipe)
      continue;
    if (pipe->transport.input_queue)
      pipe->transport.input_queue->close();
    pipe->transport.cv.notify_all();
  }
}

void RunCore::graph_request_stop(const std::string& err) {
  if (graph_debug_enabled_for_core()) {
    std::fprintf(stderr, "[GRAPH] request_stop err=%s\n", err.empty() ? "<empty>" : err.c_str());
  }
  {
    std::lock_guard<std::mutex> lock(error_mu);
    if (error.empty())
      error = err;
  }
  graph_signal_stop();
}

bool RunCore::ensure_graph_pipeline_built(std::size_t index, const Sample& sample, std::string* err,
                                          bool allow_startup_preflight) {
  const auto total_start = pipeline_internal::build_timing_now();
  ExecutionGraphRuntime& execution = graph_execution();
  if (index >= execution.pipelines.size()) {
    if (err)
      *err = "GraphRun: pipeline index out of range";
    return false;
  }

  auto& pipe = *execution.pipelines[index];
  if (pipe.transport.built.load(std::memory_order_acquire))
    return true;
  if (graph_stop_requested()) {
    if (err)
      *err = "GraphRun: stopped";
    return false;
  }

  {
    std::unique_lock<std::mutex> lock(pipe.transport.mu);
    if (pipe.transport.built.load(std::memory_order_acquire))
      return true;
    if (pipe.transport.building) {
      pipe.transport.cv.wait(lock, [&] {
        return pipe.transport.built.load(std::memory_order_acquire) || graph_stop_requested();
      });
      if (pipe.transport.built.load(std::memory_order_acquire))
        return true;
      if (err)
        *err = "GraphRun: stopped";
      return false;
    }
    pipe.transport.building = true;
  }

  const auto canonicalize_start = pipeline_internal::build_timing_now();
  Sample build_sample =
      simaai::neat::pipeline_internal::canonicalize_tensor_transport_sample(sample);
  const auto canonicalize_us = pipeline_internal::build_timing_us(canonicalize_start);

  try {
    if (graph_debug_enabled_for_core()) {
      std::fprintf(stderr, "[GRAPH] pipeline_build seg=%zu\n",
                   static_cast<std::size_t>(pipe.seg.id));
    }
    std::unique_lock<std::mutex> serial_lock;
    if (graph_serial_pipeline_build_enabled_for_core()) {
      if (graph_debug_enabled_for_core()) {
        std::fprintf(stderr, "[GRAPH] pipeline_build_lock_wait seg=%zu\n",
                     static_cast<std::size_t>(pipe.seg.id));
      }
      serial_lock = std::unique_lock<std::mutex>(graph_pipeline_build_mu_for_core());
      if (graph_debug_enabled_for_core()) {
        std::fprintf(stderr, "[GRAPH] pipeline_build_lock_acquired seg=%zu\n",
                     static_cast<std::size_t>(pipe.seg.id));
      }
    }

    RunCoreStartOptions start_opt;
    start_opt.run_options = pipe.run_options;
    start_opt.mode = RunMode::Async;
    start_opt.seed = std::move(build_sample);
    start_opt.last_pipeline = &pipe.last_pipeline;
    start_opt.owner = &pipe;
    start_opt.allow_startup_preflight = allow_startup_preflight;
    start_opt.push_sample_policy = PushSamplePolicy::PreserveSample;
    const auto segment_start = pipeline_internal::build_timing_now();
    auto run_core = RunCore::start_pipeline_segment(pipe.seg, std::move(start_opt));
    const auto segment_us = pipeline_internal::build_timing_us(segment_start);
    {
      std::lock_guard<std::mutex> lock(pipe.transport.mu);
      pipe.run_core = std::move(run_core);
      pipe.transport.built.store(true, std::memory_order_release);
      pipe.transport.building = false;
    }
    if (graph_debug_enabled_for_core()) {
      std::fprintf(stderr, "[GRAPH] pipeline_built seg=%zu\n",
                   static_cast<std::size_t>(pipe.seg.id));
    }
    pipeline_internal::emit_build_timing(
        "ensure_graph_pipeline_built",
        {{"canonicalize_sample", canonicalize_us},
         {"start_pipeline_segment", segment_us},
         {"total", pipeline_internal::build_timing_us(total_start)}},
        "seg=" + std::to_string(pipe.seg.id));
    pipe.transport.cv.notify_all();
    return true;
  } catch (const std::exception& e) {
    {
      std::lock_guard<std::mutex> lock(pipe.transport.mu);
      pipe.transport.building = false;
    }
    pipe.transport.cv.notify_all();
    if (err)
      *err = e.what();
    return false;
  }
}

bool RunCore::graph_dispatch_to_stage_group(std::size_t group_index,
                                            simaai::neat::graph::PortId port, Sample&& sample,
                                            const EdgeRouterOptions& options) {
  ExecutionGraphRuntime& execution = graph_execution();
  if (group_index >= execution.stage_groups.size())
    return false;
  auto& group = execution.stage_groups[group_index];
  if (group.instances.empty())
    return false;

  std::size_t pick = 0;
  const std::size_t count = group.instances.size();
  if (count == 1) {
    pick = group.instances[0];
  } else if (group.options.key_by == StageKeyBy::StreamId && !sample.stream_id.empty()) {
    const std::size_t h = std::hash<std::string>{}(sample.stream_id);
    pick = group.instances[h % count];
  } else {
    const std::size_t rr_idx = group.rr.fetch_add(1);
    pick = group.instances[rr_idx % count];
  }

  if (simaai::neat::graph::graph_debug_enabled()) {
    simaai::neat::graph::graph_debug_sample("dispatch_to_stage", sample);
  }
  if (simaai::neat::graph::graph_sched_debug_enabled()) {
    const std::string& label = (group.node_id < execution.node_labels.size())
                                   ? execution.node_labels[group.node_id]
                                   : std::string();
    simaai::neat::graph::graph_sched_record(group.node_id, label, sample);
  }

  auto& stage = *execution.stages[pick];
  simaai::neat::graph::StageMsg next{.in_port = port, .sample = std::move(sample)};
  const bool ok = stage.mailbox.inbox.push(std::move(next), options.push_timeout_ms);
  if (!ok && !graph_stop_requested()) {
    std::ostringstream msg;
    msg << "GraphRun: stage inbox backpressure timeout (node="
        << static_cast<std::size_t>(group.node_id) << ", edge_queue=" << options.edge_queue
        << ", push_timeout_ms=" << options.push_timeout_ms << ")."
        << graph_backpressure_timeout_explanation();
    graph_request_stop(msg.str());
  }
  return ok;
}

bool RunCore::graph_push(simaai::neat::graph::NodeId node_id, simaai::neat::graph::PortId port,
                         bool has_port, const Sample& sample, const EdgeRouterOptions& options) {
  ExecutionGraphRuntime& execution = graph_execution();
  auto it_pipe = execution.node_to_pipeline.find(node_id);
  if (it_pipe != execution.node_to_pipeline.end()) {
    auto& pipe = *execution.pipelines[it_pipe->second];
    if (pipe.seg.boundary.direct_graph_source) {
      EdgeRouter router(execution);
      auto* targets = router.targets(node_id, has_port ? port : simaai::neat::graph::kInvalidPort);
      if (!targets && !has_port) {
        for (const std::size_t edge_index : pipe.seg.output_edges) {
          if (edge_index < execution.plan.edges.size()) {
            targets = router.targets(node_id, execution.plan.edges[edge_index].from_port);
            if (targets) {
              break;
            }
          }
        }
      }
      if (!targets) {
        graph_request_stop("GraphRun::push failed: direct graph source has no downstream target");
        return false;
      }

      EdgeRouterCallbacks callbacks;
      callbacks.dispatch_to_stage_group = [this, &options](std::size_t group,
                                                           simaai::neat::graph::PortId target_port,
                                                           Sample&& next) {
        return graph_dispatch_to_stage_group(group, target_port, std::move(next), options);
      };
      callbacks.ensure_pipeline_built = [this](std::size_t index, const Sample& seed,
                                               std::string* err) {
        return ensure_graph_pipeline_built(index, seed, err);
      };
      callbacks.sanitize_pipeline_input = [this](std::size_t index, Sample& next) {
        graph_sanitize_pipeline_input(index, next);
      };
      callbacks.request_stop = [this](const std::string& err) { graph_request_stop(err); };
      callbacks.stop_requested = [this] { return graph_stop_requested(); };

      EdgeRouterDispatchOptions dispatch_options;
      dispatch_options.sanitize_pipeline_input_before_enqueue = true;
      dispatch_options.sink_backpressure_context = static_cast<std::size_t>(node_id);
      return router.dispatch_to_targets(*targets, Sample{sample}, options, callbacks,
                                        dispatch_options);
    }

    if (!pipe.transport.input_queue) {
      if (has_port) {
        std::fprintf(stderr,
                     "[GRAPH] GraphRun::push failed: input queue not found for node %zu port %zu "
                     "(check graph wiring)\n",
                     static_cast<std::size_t>(node_id), static_cast<std::size_t>(port));
      } else {
        std::fprintf(stderr,
                     "[GRAPH] GraphRun::push failed: input queue not found for node %zu (check "
                     "graph wiring)\n",
                     static_cast<std::size_t>(node_id));
      }
      return false;
    }

    std::string build_err;
    if (!ensure_graph_pipeline_built(it_pipe->second, sample, &build_err)) {
      graph_request_stop(build_err.empty() ? "GraphRun: pipeline build failed" : build_err);
      return false;
    }

    Sample copy = sample;
    const bool ok = pipe.transport.input_queue->push(std::move(copy), options.push_timeout_ms);
    if (!ok && !graph_stop_requested()) {
      std::ostringstream msg;
      msg << "GraphRun::push timed out waiting for pipeline input queue (seg="
          << static_cast<std::size_t>(pipe.seg.id) << ", edge_queue=" << options.edge_queue
          << ", push_timeout_ms=" << options.push_timeout_ms << ")."
          << graph_backpressure_timeout_explanation();
      graph_request_stop(msg.str());
    }
    return ok;
  }

  auto it_stage = execution.node_to_stage_group.find(node_id);
  if (it_stage != execution.node_to_stage_group.end()) {
    simaai::neat::graph::PortId in_port = port;
    if (!has_port) {
      in_port = simaai::neat::graph::kInvalidPort;
      const auto& group = execution.stage_groups[it_stage->second];
      if (!group.instances.empty()) {
        const auto& stage = *execution.stages[group.instances.front()];
        if (stage.input_ports.size() == 1) {
          in_port = stage.input_ports[0];
        }
      }
    }
    Sample copy = sample;
    return graph_dispatch_to_stage_group(it_stage->second, in_port, std::move(copy), options);
  }

  std::fprintf(stderr, "[GRAPH] GraphRun::push failed: node %zu not found in any pipeline stage\n",
               static_cast<std::size_t>(node_id));
  return false;
}

void RunCore::graph_sanitize_pipeline_input(std::size_t index, Sample& sample) {
  ExecutionGraphRuntime& execution = graph_execution();
  if (index >= execution.pipelines.size() || !execution.pipelines[index])
    return;
  auto& pipe = *execution.pipelines[index];
  const int64_t prev_pts = sample.pts_ns;
  const int64_t prev_dts = sample.dts_ns;
  const int64_t prev_duration = sample.duration_ns;
  const int64_t prev_frame = sample.frame_id;
  sample = simaai::neat::pipeline_internal::canonicalize_tensor_transport_sample(sample);
  if (simaai::neat::pipeline_internal::env_bool("SIMA_SAMPLE_TIMING_DEBUG", false)) {
    std::fprintf(stderr,
                 "[SAMPLE_TIMING] graph_sanitize seg=%zu before_frame=%lld after_frame=%lld "
                 "before_pts=%lld after_pts=%lld before_dts=%lld after_dts=%lld "
                 "before_dur=%lld after_dur=%lld kind=%d stream_id=%s\n",
                 static_cast<std::size_t>(pipe.seg.id), static_cast<long long>(prev_frame),
                 static_cast<long long>(sample.frame_id), static_cast<long long>(prev_pts),
                 static_cast<long long>(sample.pts_ns), static_cast<long long>(prev_dts),
                 static_cast<long long>(sample.dts_ns), static_cast<long long>(prev_duration),
                 static_cast<long long>(sample.duration_ns), static_cast<int>(sample.kind),
                 sample.stream_id.c_str());
  }
  const int64_t prev_input_seq = sample.input_seq;
  if (sample.orig_input_seq < 0 && prev_input_seq >= 0) {
    sample.orig_input_seq = prev_input_seq;
  }
  sample.input_seq = pipe.transport.next_input_seq.fetch_add(1, std::memory_order_relaxed);
  if (simaai::neat::graph::graph_debug_enabled() && prev_input_seq >= 0 &&
      prev_input_seq != sample.input_seq) {
    std::fprintf(stderr, "[GRAPH] input_seq_override seg=%zu old=%lld new=%lld stream_id=%s\n",
                 static_cast<std::size_t>(pipe.seg.id), static_cast<long long>(prev_input_seq),
                 static_cast<long long>(sample.input_seq), sample.stream_id.c_str());
  }
  const std::size_t max_stream_map = simaai::neat::graph::identity_map_capacity();
  const SampleIdentity identity = capture_sample_identity(sample);
  if (has_sample_identity(identity)) {
    std::lock_guard<std::mutex> lock(pipe.transport.stream_mu);
    pipe.transport.pending_identities.push_back(identity);
    if (max_stream_map > 0) {
      while (pipe.transport.pending_identities.size() > max_stream_map) {
        pipe.transport.pending_identities.pop_front();
      }
    }
    if (sample.input_seq >= 0) {
      pipe.transport.identity_by_input_seq[sample.input_seq] = identity;
      pipe.transport.input_seq_order.push_back(sample.input_seq);
      if (simaai::neat::graph::graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[GRAPH] identity_map_add seg=%zu input_seq=%lld frame_id=%lld "
                     "pts_ns=%lld stream_id=%s map=%zu\n",
                     static_cast<std::size_t>(pipe.seg.id),
                     static_cast<long long>(sample.input_seq),
                     static_cast<long long>(sample.frame_id), static_cast<long long>(sample.pts_ns),
                     sample.stream_id.c_str(), pipe.transport.identity_by_input_seq.size());
      }
      if (max_stream_map > 0) {
        while (pipe.transport.input_seq_order.size() > max_stream_map) {
          const int64_t drop = pipe.transport.input_seq_order.front();
          pipe.transport.input_seq_order.pop_front();
          pipe.transport.identity_by_input_seq.erase(drop);
        }
      }
    }
    if (sample.frame_id >= 0) {
      auto& q = pipe.transport.identity_by_frame[sample.frame_id];
      q.push_back(identity);
      pipe.transport.stream_order.push_back(sample.frame_id);
      if (simaai::neat::graph::graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[GRAPH] identity_map_add seg=%zu frame_id=%lld input_seq=%lld "
                     "pts_ns=%lld stream_id=%s qsize=%zu map=%zu\n",
                     static_cast<std::size_t>(pipe.seg.id), static_cast<long long>(sample.frame_id),
                     static_cast<long long>(sample.input_seq),
                     static_cast<long long>(sample.pts_ns), sample.stream_id.c_str(), q.size(),
                     pipe.transport.identity_by_frame.size());
      }
      if (max_stream_map > 0) {
        while (pipe.transport.stream_order.size() > max_stream_map) {
          const int64_t drop = pipe.transport.stream_order.front();
          pipe.transport.stream_order.pop_front();
          auto it = pipe.transport.identity_by_frame.find(drop);
          if (it == pipe.transport.identity_by_frame.end()) {
            continue;
          }
          if (!it->second.empty()) {
            it->second.pop_front();
          }
          if (it->second.empty()) {
            pipe.transport.identity_by_frame.erase(it);
          }
        }
      }
    }
  }
  const std::string current_label =
      !sample.stream_label.empty() ? sample.stream_label : sample.port_name;
  if (current_label.empty())
    return;
  if (pipe.transport.expected_buffer_names.empty()) {
    if (simaai::neat::graph::graph_debug_enabled()) {
      std::fprintf(stderr, "[GRAPH] clear_port_name seg=%zu got=%s expected=<none>\n",
                   static_cast<std::size_t>(pipe.seg.id), current_label.c_str());
    }
    sample.stream_label.clear();
    sample.port_name.clear();
    return;
  }
  for (const auto& expected : pipe.transport.expected_buffer_names) {
    if (expected == current_label)
      return;
  }
  if (simaai::neat::graph::graph_debug_enabled()) {
    std::string expected_join;
    for (const auto& expected : pipe.transport.expected_buffer_names) {
      if (!expected_join.empty())
        expected_join += ",";
      expected_join += expected;
    }
    std::fprintf(stderr, "[GRAPH] clear_port_name seg=%zu got=%s expected=%s\n",
                 static_cast<std::size_t>(pipe.seg.id), current_label.c_str(),
                 expected_join.c_str());
  }
  sample.stream_label.clear();
  sample.port_name.clear();
}

void RunCore::graph_restore_stream_id_if_needed(std::size_t index, Sample& sample) {
  ExecutionGraphRuntime& execution = graph_execution();
  if (index >= execution.pipelines.size() || !execution.pipelines[index])
    return;
  auto& pipe = *execution.pipelines[index];
  const Sample before = sample;
  const bool prefer_mapped = true;
  const bool looks_internal = is_internal_stream_id(sample.stream_id);
  const bool missing = sample.stream_id.empty();
  bool map_lookup_attempted = false;
  bool map_hit = false;

  auto finalize_identity_diag = [&]() {
    if (sample.stream_id != before.stream_id || sample.frame_id != before.frame_id ||
        sample.pts_ns != before.pts_ns || sample.dts_ns != before.dts_ns ||
        sample.duration_ns != before.duration_ns) {
      pipe.transport.identity_rewrite_count.fetch_add(1, std::memory_order_relaxed);
    }
    const bool required_mapping = missing || looks_internal || prefer_mapped;
    if (required_mapping && map_lookup_attempted && !map_hit) {
      pipe.transport.identity_map_miss_count.fetch_add(1, std::memory_order_relaxed);
    }
  };

  std::lock_guard<std::mutex> lock(pipe.transport.stream_mu);
  if (sample.input_seq >= 0) {
    map_lookup_attempted = true;
    auto it = pipe.transport.identity_by_input_seq.find(sample.input_seq);
    if (it != pipe.transport.identity_by_input_seq.end()) {
      restore_sample_identity_if_needed(sample, it->second, prefer_mapped);
      map_hit = true;
      if (simaai::neat::graph::graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[GRAPH] identity_map_use seg=%zu input_seq=%lld frame_id=%lld pts_ns=%lld "
                     "stream_id=%s map=%zu\n",
                     static_cast<std::size_t>(pipe.seg.id),
                     static_cast<long long>(sample.input_seq),
                     static_cast<long long>(sample.frame_id), static_cast<long long>(sample.pts_ns),
                     sample.stream_id.c_str(), pipe.transport.identity_by_input_seq.size());
      }
      finalize_identity_diag();
      return;
    }
    if (simaai::neat::graph::graph_debug_enabled()) {
      std::fprintf(stderr,
                   "[GRAPH] identity_map_miss seg=%zu input_seq=%lld frame_id=%lld pts_ns=%lld "
                   "stream_id=%s missing=%d internal=%d map=%zu\n",
                   static_cast<std::size_t>(pipe.seg.id), static_cast<long long>(sample.input_seq),
                   static_cast<long long>(sample.frame_id), static_cast<long long>(sample.pts_ns),
                   sample.stream_id.c_str(), static_cast<int>(missing),
                   static_cast<int>(looks_internal), pipe.transport.identity_by_input_seq.size());
    }
  }

  if (sample.frame_id >= 0) {
    map_lookup_attempted = true;
    auto it = pipe.transport.identity_by_frame.find(sample.frame_id);
    if (it != pipe.transport.identity_by_frame.end() && !it->second.empty()) {
      const SampleIdentity identity = it->second.front();
      it->second.pop_front();
      const std::size_t remaining = it->second.size();
      if (remaining == 0) {
        pipe.transport.identity_by_frame.erase(it);
      }
      restore_sample_identity_if_needed(sample, identity, prefer_mapped);
      map_hit = true;
      if (simaai::neat::graph::graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[GRAPH] identity_map_use seg=%zu frame_id=%lld input_seq=%lld pts_ns=%lld "
                     "stream_id=%s qsize=%zu map=%zu\n",
                     static_cast<std::size_t>(pipe.seg.id), static_cast<long long>(sample.frame_id),
                     static_cast<long long>(sample.input_seq),
                     static_cast<long long>(sample.pts_ns), sample.stream_id.c_str(), remaining,
                     pipe.transport.identity_by_frame.size());
      }
      finalize_identity_diag();
      return;
    }
    if (simaai::neat::graph::graph_debug_enabled()) {
      std::fprintf(stderr,
                   "[GRAPH] identity_map_miss seg=%zu frame_id=%lld input_seq=%lld pts_ns=%lld "
                   "stream_id=%s missing=%d internal=%d map=%zu\n",
                   static_cast<std::size_t>(pipe.seg.id), static_cast<long long>(sample.frame_id),
                   static_cast<long long>(sample.input_seq), static_cast<long long>(sample.pts_ns),
                   sample.stream_id.c_str(), static_cast<int>(missing),
                   static_cast<int>(looks_internal), pipe.transport.identity_by_frame.size());
    }
  }

  if (!pipe.transport.pending_identities.empty()) {
    map_lookup_attempted = true;
    const SampleIdentity pending = pipe.transport.pending_identities.front();
    pipe.transport.pending_identities.pop_front();
    restore_sample_identity_if_needed(sample, pending, prefer_mapped);
    map_hit = true;
    if (simaai::neat::graph::graph_debug_enabled()) {
      std::fprintf(stderr,
                   "[GRAPH] identity_map_fallback seg=%zu frame_id=%lld input_seq=%lld "
                   "pts_ns=%lld stream_id=%s pending=%zu\n",
                   static_cast<std::size_t>(pipe.seg.id), static_cast<long long>(sample.frame_id),
                   static_cast<long long>(sample.input_seq), static_cast<long long>(sample.pts_ns),
                   sample.stream_id.c_str(), pipe.transport.pending_identities.size());
    }
  }
  finalize_identity_diag();
}

std::optional<Sample> RunCore::graph_pull(simaai::neat::graph::NodeId node_id, int timeout_ms) {
  ExecutionGraphRuntime& execution = graph_execution();
  const bool has_timeout = (timeout_ms >= 0);
  const bool direct_sink =
      execution.direct_sink_nodes.find(node_id) != execution.direct_sink_nodes.end();
  auto it_pipe = execution.node_to_pipeline.find(node_id);
  if (!direct_sink && it_pipe != execution.node_to_pipeline.end()) {
    auto& pipe = *execution.pipelines[it_pipe->second];
    if (!pipe.transport.built.load(std::memory_order_acquire)) {
      if (simaai::neat::graph::graph_debug_enabled()) {
        std::fprintf(stderr, "[GRAPH] pull_wait_for_build seg=%zu\n",
                     static_cast<std::size_t>(pipe.seg.id));
      }
      const int build_timeout_ms =
          has_timeout ? std::max(timeout_ms,
                                 pipeline_internal::env_int("SIMA_GRAPH_BUILD_TIMEOUT_MS", 5000))
                      : pipeline_internal::env_int("SIMA_GRAPH_BUILD_TIMEOUT_MS", -1);
      std::unique_lock<std::mutex> lock(pipe.transport.mu);
      if (has_timeout && build_timeout_ms >= 0) {
        pipe.transport.cv.wait_until(
            lock, std::chrono::steady_clock::now() + std::chrono::milliseconds(build_timeout_ms),
            [&] {
              return pipe.transport.built.load(std::memory_order_acquire) || graph_stop_requested();
            });
      } else {
        pipe.transport.cv.wait(lock, [&] {
          return pipe.transport.built.load(std::memory_order_acquire) || graph_stop_requested();
        });
      }
      if (!pipe.transport.built.load(std::memory_order_acquire)) {
        if (simaai::neat::graph::graph_debug_enabled()) {
          std::fprintf(stderr, "[GRAPH] pull_wait_for_build_timeout seg=%zu\n",
                       static_cast<std::size_t>(pipe.seg.id));
        }
        return std::nullopt;
      }
    }
  }

  auto it = execution.sinks.find(node_id);
  if (it == execution.sinks.end()) {
    std::fprintf(stderr, "[GRAPH] GraphRun::pull failed: no output sink found for node %zu\n",
                 static_cast<std::size_t>(node_id));
    return std::nullopt;
  }

  Sample out;
  const int wait_ms = has_timeout ? timeout_ms : -1;
  if (!it->second->pop(out, wait_ms)) {
    std::fprintf(stderr,
                 "[GRAPH] GraphRun::pull: queue pop returned empty for node %zu (timeout=%dms or "
                 "queue closed)\n",
                 static_cast<std::size_t>(node_id), wait_ms);
    return std::nullopt;
  }
  return out;
}

std::shared_ptr<RunCore> RunCore::start_pipeline_segment(const PipelineSegmentPlan& segment,
                                                         RunCoreStartOptions opt) {
  const auto total_start = pipeline_internal::build_timing_now();
  gst_init_once();

  const auto materialize_start = pipeline_internal::build_timing_now();
  std::vector<std::shared_ptr<simaai::neat::Node>> nodes = materialize_segment_nodes(segment);
  const auto materialize_us = pipeline_internal::build_timing_us(materialize_start);
  if (nodes.empty()) {
    throw std::runtime_error("RunCore::start_pipeline_segment: empty pipeline segment");
  }

  std::string local_last_pipeline;
  std::string& last_pipeline = opt.last_pipeline ? *opt.last_pipeline : local_last_pipeline;
  GraphOptions route_options = segment.route_options;
  if (segment.boundary_hints.has_value()) {
    if (!opt.input_route_processor && segment.boundary_hints->input_route_processor) {
      opt.input_route_processor = segment.boundary_hints->input_route_processor;
    }
    if (!opt.tensor_input_opt_for_cv.has_value() && segment.boundary_hints->tensor_mode &&
        !segment.boundary_hints->ingress_inputs.empty()) {
      opt.tensor_input_opt_for_cv = segment.boundary_hints->ingress_inputs.front();
    }
  }

  if (opt.image_seed && opt.seed.has_value()) {
    throw std::runtime_error(
        "RunCore::start_pipeline_segment: image_seed and seed are mutually exclusive");
  }

  if (!opt.image_seed && !opt.seed.has_value()) {
    if (segment.boundary.needs_input && !segment.boundary.source_like) {
      throw std::runtime_error(
          "RunCore::start_pipeline_segment: push-style segment requires an input seed");
    }
    const auto source_start = pipeline_internal::build_timing_now();
    const bool public_output_contract = !segment.boundary.graph_internal_output;
    SourceStreamBuildContext source = session_build_source_stream_internal(
        nodes, opt.guard, last_pipeline, route_options, opt.run_options, opt.mode, opt.require_sink,
        public_output_contract, "RunCore::start(plan/source)");
    const auto source_us = pipeline_internal::build_timing_us(source_start);
    const auto start_single_start = pipeline_internal::build_timing_now();
    auto core = RunCore::start_single_pipeline(
        std::move(source.stream), source.merged_opt, source.stream_opt, opt.mode,
        opt.tensor_input_opt_for_cv, std::move(opt.input_route_processor));
    core->push_sample_policy = opt.push_sample_policy;
    const auto start_single_us = pipeline_internal::build_timing_us(start_single_start);
    pipeline_internal::emit_build_timing(
        "RunCore::start_pipeline_segment",
        {{"materialize_nodes", materialize_us},
         {"source_stream_build", source_us},
         {"start_single_pipeline", start_single_us},
         {"total", pipeline_internal::build_timing_us(total_start)}},
        "seg=" + std::to_string(segment.id) + " mode=source nodes=" + std::to_string(nodes.size()));
    return core;
  }

  const auto context_start = pipeline_internal::build_timing_now();
  const bool public_output_contract = !segment.boundary.graph_internal_output;
  const BuildInputContext ctx =
      session_build_prepare_build_input_context(nodes, route_options, opt.mode, opt.run_options,
                                                public_output_contract);
  const auto context_us = pipeline_internal::build_timing_us(context_start);
  InputStreamOptions build_stream_opt = ctx.stream_opt;
  build_stream_opt.startup_preflight =
      opt.allow_startup_preflight && ctx.stream_opt.startup_preflight;
  InputStream stream;
  const auto input_stream_start = pipeline_internal::build_timing_now();
  if (opt.image_seed) {
    stream = session_build_run_input_stream_internal(
        nodes, opt.guard, opt.owner, last_pipeline, *opt.image_seed, route_options,
        build_stream_opt, ctx.name_transform, ctx.insert_queue2, ctx.sync_num_buffers_override,
        ctx.mode == RunMode::Sync);
  } else {
    stream = session_build_run_input_stream_internal(
        nodes, opt.guard, opt.owner, last_pipeline, *opt.seed, route_options, build_stream_opt,
        ctx.name_transform, ctx.insert_queue2, ctx.sync_num_buffers_override,
        ctx.mode == RunMode::Sync);
  }
  const auto input_stream_us = pipeline_internal::build_timing_us(input_stream_start);
  const auto start_single_start = pipeline_internal::build_timing_now();
  auto core = RunCore::start_single_pipeline(std::move(stream), ctx.merged_opt, build_stream_opt,
                                             ctx.mode, opt.tensor_input_opt_for_cv,
                                             std::move(opt.input_route_processor));
  core->push_sample_policy = opt.push_sample_policy;
  const auto start_single_us = pipeline_internal::build_timing_us(start_single_start);
  pipeline_internal::emit_build_timing("RunCore::start_pipeline_segment",
                                       {{"materialize_nodes", materialize_us},
                                        {"prepare_input_context", context_us},
                                        {"input_stream_build", input_stream_us},
                                        {"start_single_pipeline", start_single_us},
                                        {"total", pipeline_internal::build_timing_us(total_start)}},
                                       "seg=" + std::to_string(segment.id) +
                                           " mode=input nodes=" + std::to_string(nodes.size()));
  return core;
}

} // namespace simaai::neat::runtime

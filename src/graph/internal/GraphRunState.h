#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "graph/Compiler.h"
#include "graph/GraphRun.h"
#include "graph/GraphSession.h"
#include "graph/nodes/PipelineNode.h"
#include "graph/runtime/BlockingQueue.h"
#include "graph/runtime/StageMailbox.h"
#include "nodes/io/Input.h"
#include "pipeline/PowerTelemetry.h"
#include "pipeline/Run.h"
#include "pipeline/Session.h"
#include "pipeline/TensorCore.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/PipelineBuild.h"
#include "pipeline/internal/SampleUtil.h"

#include <atomic>
#include <condition_variable>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simaai::neat::graph {

using simaai::neat::pipeline_internal::env_bool;
using simaai::neat::pipeline_internal::env_int;

using PipelineNode = simaai::neat::graph::nodes::PipelineNode;
using StageNode = simaai::neat::graph::nodes::StageNode;
using StageKeyBy = simaai::neat::graph::nodes::StageKeyBy;
using StageNodeOptions = simaai::neat::graph::nodes::StageNodeOptions;
using BlockingQueueSample = simaai::neat::graph::runtime::BlockingQueue<Sample>;
using StageMailbox = simaai::neat::graph::runtime::StageMailbox;

struct DownstreamTarget {
  enum class Kind {
    StageGroup,
    PipelineInput,
    GraphSink,
  };

  Kind kind = Kind::StageGroup;
  std::size_t index = 0; // stage group index or pipeline index
  PortId port = kInvalidPort;
};

bool graph_debug_enabled();
bool graph_push_fail_debug_enabled();
bool graph_serial_pipeline_build_enabled();
std::mutex& graph_pipeline_build_mu();
bool stop_trace_enabled();
bool graph_output_rate_enabled();
bool graph_sched_debug_enabled();

void graph_sched_record(NodeId node_id, const std::string& label, const Sample& sample);
void graph_sched_summary(const std::vector<std::string>* labels);
void graph_log_output_rate(NodeId node_id, const Sample& sample, const char* label);
void graph_output_rate_summary(
    const std::vector<std::string>* labels,
    const std::unordered_map<NodeId, std::shared_ptr<BlockingQueueSample>>* sinks);
void graph_debug_sample(const char* tag, const Sample& sample);

inline std::uint64_t port_key(NodeId id, PortId port) {
  return (static_cast<std::uint64_t>(id) << 32) | static_cast<std::uint64_t>(port);
}

bool has_input_appsrc(const simaai::neat::NodeGroup& group);
bool has_output_appsink(const simaai::neat::NodeGroup& group);
bool has_internal_source(const simaai::neat::NodeGroup& group);
std::size_t identity_map_capacity();
void maybe_force_copy_for_backpressure(Sample& sample, std::size_t qsize, const char* where,
                                       std::size_t seg_id);
InputOptions input_opts_from_spec(const OutputSpec& spec, bool complete);
bool is_encoded_sample(const Sample& sample);
std::optional<Sample> sample_from_input_spec(const OutputSpec& spec, std::string* err);
Sample make_bundle_carrier_sample();
void log_first_decoded_once(const Sample& sample, const CompiledPipelineSegment& seg);

struct GraphRun::State {
  CompiledGraph compiled;
  GraphRunOptions opt;
  std::shared_ptr<void> verbose_guard;
  std::atomic<bool> stop{false};
  std::vector<std::string> node_labels;
  std::atomic<bool> output_rate_reported{false};
  std::atomic<bool> sched_reported{false};
  std::atomic<std::size_t> output_rr{0};
  std::shared_ptr<GraphRunStats> stats;
  std::unique_ptr<simaai::neat::PowerMonitor> power_monitor;

  std::mutex error_mu;
  std::string error;

  struct PipelineRuntime {
    CompiledPipelineSegment seg;
    Session session;
    RunOptions run_options;
    Run run;
    std::atomic<bool> built{false};
    bool building = false;
    bool has_input = false;
    bool has_output = false;
    std::vector<std::string> expected_buffer_names;
    std::mutex stream_mu;
    std::atomic<int64_t> next_input_seq{0};
    std::unordered_map<int64_t, std::deque<std::string>> stream_by_input_seq;
    std::deque<int64_t> input_seq_order;
    std::unordered_map<int64_t, int64_t> frame_by_input_seq;
    std::unordered_map<int64_t, std::deque<std::string>> stream_by_frame;
    std::deque<int64_t> stream_order;
    struct PendingIdentity {
      int64_t frame_id = -1;
      std::string stream_id;
    };
    std::deque<PendingIdentity> pending_identities;
    std::atomic<int64_t> identity_rewrite_count{0};
    std::atomic<int64_t> identity_map_miss_count{0};

    std::shared_ptr<BlockingQueueSample> input_queue;
    std::thread push_thread;
    std::thread pull_thread;
    std::atomic<bool> push_done{false};
    std::atomic<bool> pull_done{false};

    std::mutex mu;
    std::condition_variable cv;
  };

  void signal_stop();
  void request_stop(const std::string& err);
  bool ensure_pipeline_built(std::size_t index, const Sample& sample, std::string* err);
  bool route_stage_output(NodeId node_id, const std::vector<PortId>& output_ports,
                          StageOutMsg&& out_msg);

  struct RuntimeStageEmitter final : StageEmitter {
    bool emit(StageOutMsg msg) override {
      if (state == nullptr || output_ports == nullptr) {
        return false;
      }
      return state->route_stage_output(node_id, *output_ports, std::move(msg));
    }

    bool stop_requested() const override {
      return state == nullptr || state->stop.load(std::memory_order_relaxed);
    }

    GraphRun::State* state = nullptr;
    NodeId node_id = kInvalidNode;
    const std::vector<PortId>* output_ports = nullptr;
  };

  struct StageRuntime {
    explicit StageRuntime(std::size_t capacity = 0) : mailbox(capacity) {}

    NodeId node_id = kInvalidNode;
    std::unique_ptr<StageExecutor> exec;
    RuntimeStageEmitter emitter;
    StageMailbox mailbox;
    std::thread worker;
    std::atomic<bool> worker_done{false};
    std::vector<PortId> input_ports;
    std::vector<PortId> output_ports;
    StagePorts ports;
  };

  struct StageGroup {
    NodeId node_id = kInvalidNode;
    StageNodeOptions options;
    std::vector<std::size_t> instances;
    std::atomic<std::size_t> rr{0};

    StageGroup() = default;
    StageGroup(const StageGroup&) = delete;
    StageGroup& operator=(const StageGroup&) = delete;
    StageGroup(StageGroup&& other) noexcept
        : node_id(other.node_id), options(other.options), instances(std::move(other.instances)),
          rr(other.rr.load()) {}
    StageGroup& operator=(StageGroup&& other) noexcept {
      if (this != &other) {
        node_id = other.node_id;
        options = other.options;
        instances = std::move(other.instances);
        rr.store(other.rr.load());
      }
      return *this;
    }
  };

  std::vector<std::unique_ptr<PipelineRuntime>> pipelines;
  std::vector<std::unique_ptr<StageRuntime>> stages;
  std::vector<StageGroup> stage_groups;

  std::unordered_map<NodeId, std::size_t> node_to_pipeline;
  std::unordered_map<NodeId, std::size_t> node_to_stage_group;
  std::unordered_set<NodeId> direct_sink_nodes;

  std::unordered_map<std::uint64_t, std::vector<DownstreamTarget>> adjacency;
  std::unordered_map<NodeId, std::shared_ptr<BlockingQueueSample>> sinks;

  bool dispatch_to_stage_group(std::size_t group_index, PortId port, Sample&& sample) {
    if (group_index >= stage_groups.size())
      return false;
    auto& group = stage_groups[group_index];
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

    if (graph_debug_enabled()) {
      graph_debug_sample("dispatch_to_stage", sample);
    }
    if (graph_sched_debug_enabled()) {
      const std::string& label =
          (group.node_id < node_labels.size()) ? node_labels[group.node_id] : std::string();
      graph_sched_record(group.node_id, label, sample);
    }
    auto& stage = *stages[pick];
    StageMsg next{.in_port = port, .sample = std::move(sample)};
    const bool ok = stage.mailbox.inbox.push(std::move(next), opt.push_timeout_ms);
    if (!ok && !stop.load(std::memory_order_relaxed)) {
      std::ostringstream msg;
      msg << "GraphRun: stage inbox backpressure timeout (node="
          << static_cast<std::size_t>(group.node_id) << ", edge_queue=" << opt.edge_queue
          << ", push_timeout_ms=" << opt.push_timeout_ms
          << "). Increase GraphRunOptions.edge_queue or pull outputs concurrently.";
      request_stop(msg.str());
    }
    return ok;
  }

  void sanitize_sample_for_pipeline_input(PipelineRuntime& pipe, Sample& sample) {
    sample = simaai::neat::pipeline_internal::canonicalize_tensor_transport_sample(sample);
    const int64_t prev_input_seq = sample.input_seq;
    if (sample.orig_input_seq < 0 && prev_input_seq >= 0) {
      sample.orig_input_seq = prev_input_seq;
    }
    sample.input_seq = pipe.next_input_seq.fetch_add(1, std::memory_order_relaxed);
    if (graph_debug_enabled() && prev_input_seq >= 0 && prev_input_seq != sample.input_seq) {
      std::fprintf(stderr, "[GRAPH] input_seq_override seg=%zu old=%lld new=%lld stream_id=%s\n",
                   static_cast<std::size_t>(pipe.seg.id), static_cast<long long>(prev_input_seq),
                   static_cast<long long>(sample.input_seq), sample.stream_id.c_str());
    }
    const std::size_t max_stream_map = identity_map_capacity();
    const bool track_identity = (sample.frame_id >= 0) || !sample.stream_id.empty();
    if (track_identity) {
      std::lock_guard<std::mutex> lock(pipe.stream_mu);
      if (sample.input_seq >= 0 && sample.frame_id >= 0) {
        pipe.frame_by_input_seq[sample.input_seq] = sample.frame_id;
      }
      pipe.pending_identities.push_back(PipelineRuntime::PendingIdentity{
          .frame_id = sample.frame_id,
          .stream_id = sample.stream_id,
      });
      if (max_stream_map > 0) {
        while (pipe.pending_identities.size() > max_stream_map) {
          pipe.pending_identities.pop_front();
        }
      }
      if (!sample.stream_id.empty()) {
        if (sample.input_seq >= 0) {
          auto& q = pipe.stream_by_input_seq[sample.input_seq];
          q.push_back(sample.stream_id);
          pipe.input_seq_order.push_back(sample.input_seq);
          if (graph_debug_enabled()) {
            std::fprintf(stderr,
                         "[GRAPH] stream_map_add seg=%zu input_seq=%lld frame_id=%lld stream_id=%s "
                         "qsize=%zu map=%zu\n",
                         static_cast<std::size_t>(pipe.seg.id),
                         static_cast<long long>(sample.input_seq),
                         static_cast<long long>(sample.frame_id), sample.stream_id.c_str(),
                         q.size(), pipe.stream_by_input_seq.size());
          }
          if (max_stream_map > 0) {
            while (pipe.input_seq_order.size() > max_stream_map) {
              const int64_t drop = pipe.input_seq_order.front();
              pipe.input_seq_order.pop_front();
              auto it = pipe.stream_by_input_seq.find(drop);
              if (it != pipe.stream_by_input_seq.end()) {
                if (!it->second.empty()) {
                  it->second.pop_front();
                }
                if (it->second.empty()) {
                  pipe.stream_by_input_seq.erase(it);
                }
              }
              pipe.frame_by_input_seq.erase(drop);
            }
          }
        }
        if (sample.frame_id >= 0) {
          auto& q = pipe.stream_by_frame[sample.frame_id];
          q.push_back(sample.stream_id);
          pipe.stream_order.push_back(sample.frame_id);
          if (graph_debug_enabled()) {
            std::fprintf(stderr,
                         "[GRAPH] stream_map_add seg=%zu frame_id=%lld input_seq=%lld stream_id=%s "
                         "qsize=%zu map=%zu\n",
                         static_cast<std::size_t>(pipe.seg.id),
                         static_cast<long long>(sample.frame_id),
                         static_cast<long long>(sample.input_seq), sample.stream_id.c_str(),
                         q.size(), pipe.stream_by_frame.size());
          }
          if (max_stream_map > 0) {
            while (pipe.stream_order.size() > max_stream_map) {
              const int64_t drop = pipe.stream_order.front();
              pipe.stream_order.pop_front();
              auto it = pipe.stream_by_frame.find(drop);
              if (it == pipe.stream_by_frame.end())
                continue;
              if (!it->second.empty()) {
                it->second.pop_front();
              }
              if (it->second.empty()) {
                pipe.stream_by_frame.erase(it);
              }
            }
          }
        }
      }
    }
    const std::string current_label =
        !sample.stream_label.empty() ? sample.stream_label : sample.port_name;
    if (current_label.empty())
      return;
    if (pipe.expected_buffer_names.empty()) {
      if (graph_debug_enabled()) {
        std::fprintf(stderr, "[GRAPH] clear_port_name seg=%zu got=%s expected=<none>\n",
                     static_cast<std::size_t>(pipe.seg.id), current_label.c_str());
      }
      sample.stream_label.clear();
      sample.port_name.clear();
      return;
    }
    for (const auto& expected : pipe.expected_buffer_names) {
      if (expected == current_label)
        return;
    }
    if (graph_debug_enabled()) {
      std::string expected_join;
      for (const auto& expected : pipe.expected_buffer_names) {
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

  void restore_stream_id_if_needed(PipelineRuntime& pipe, Sample& sample) {
    const auto is_internal_stream_id = [](const std::string& id) {
      return !id.empty() && id.find("neatprocess") != std::string::npos;
    };
    const std::string original_stream_id = sample.stream_id;
    const int64_t original_frame_id = sample.frame_id;
    const bool prefer_mapped = true;
    const bool looks_internal = is_internal_stream_id(sample.stream_id);
    const bool missing = sample.stream_id.empty();
    bool map_lookup_attempted = false;
    bool map_hit = false;
    auto finalize_identity_diag = [&]() {
      if (sample.stream_id != original_stream_id || sample.frame_id != original_frame_id) {
        pipe.identity_rewrite_count.fetch_add(1, std::memory_order_relaxed);
      }
      const bool required_mapping = missing || looks_internal || prefer_mapped;
      if (required_mapping && map_lookup_attempted && !map_hit) {
        pipe.identity_map_miss_count.fetch_add(1, std::memory_order_relaxed);
      }
    };

    std::lock_guard<std::mutex> lock(pipe.stream_mu);
    if (sample.input_seq >= 0) {
      auto it_frame = pipe.frame_by_input_seq.find(sample.input_seq);
      if (it_frame != pipe.frame_by_input_seq.end() && it_frame->second >= 0) {
        sample.frame_id = it_frame->second;
      }
    }
    bool needs_stream = missing || looks_internal || prefer_mapped;
    bool used_input_seq = false;
    bool mapped_stream_applied = false;
    if (needs_stream && sample.input_seq >= 0) {
      map_lookup_attempted = true;
      auto it = pipe.stream_by_input_seq.find(sample.input_seq);
      if (it != pipe.stream_by_input_seq.end() && !it->second.empty()) {
        const std::string mapped_stream = it->second.front();
        std::size_t remaining = it->second.size();
        if (!prefer_mapped) {
          it->second.pop_front();
          remaining = it->second.size();
          if (remaining == 0) {
            pipe.stream_by_input_seq.erase(it);
          }
        }
        if (!mapped_stream.empty()) {
          map_hit = true;
          if (prefer_mapped || sample.stream_id.empty() ||
              is_internal_stream_id(sample.stream_id)) {
            sample.stream_id = mapped_stream;
            mapped_stream_applied = true;
          }
        }
        used_input_seq = true;
        if (graph_debug_enabled()) {
          std::fprintf(stderr,
                       "[GRAPH] stream_map_use seg=%zu input_seq=%lld frame_id=%lld stream_id=%s "
                       "qsize=%zu map=%zu\n",
                       static_cast<std::size_t>(pipe.seg.id),
                       static_cast<long long>(sample.input_seq),
                       static_cast<long long>(sample.frame_id), sample.stream_id.c_str(), remaining,
                       pipe.stream_by_input_seq.size());
        }
      } else if (graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[GRAPH] stream_map_miss seg=%zu input_seq=%lld frame_id=%lld stream_id=%s "
                     "missing=%d internal=%d map=%zu\n",
                     static_cast<std::size_t>(pipe.seg.id),
                     static_cast<long long>(sample.input_seq),
                     static_cast<long long>(sample.frame_id), sample.stream_id.c_str(),
                     static_cast<int>(missing), static_cast<int>(looks_internal),
                     pipe.stream_by_input_seq.size());
      }
    }
    if (used_input_seq) {
      if (!prefer_mapped && !sample.stream_id.empty() && !is_internal_stream_id(sample.stream_id)) {
        finalize_identity_diag();
        return;
      }
      if (prefer_mapped && mapped_stream_applied) {
        finalize_identity_diag();
        return;
      }
      needs_stream = true;
    }
    if (needs_stream && sample.frame_id >= 0) {
      map_lookup_attempted = true;
      auto it = pipe.stream_by_frame.find(sample.frame_id);
      if (it != pipe.stream_by_frame.end() && !it->second.empty()) {
        const std::string mapped_stream = it->second.front();
        it->second.pop_front();
        const std::size_t remaining = it->second.size();
        if (remaining == 0) {
          pipe.stream_by_frame.erase(it);
        }
        if (!mapped_stream.empty()) {
          map_hit = true;
          if (prefer_mapped || sample.stream_id.empty() ||
              is_internal_stream_id(sample.stream_id)) {
            sample.stream_id = mapped_stream;
          }
        }
        if (graph_debug_enabled()) {
          std::fprintf(stderr,
                       "[GRAPH] stream_map_use seg=%zu frame_id=%lld input_seq=%lld stream_id=%s "
                       "qsize=%zu map=%zu\n",
                       static_cast<std::size_t>(pipe.seg.id),
                       static_cast<long long>(sample.frame_id),
                       static_cast<long long>(sample.input_seq), sample.stream_id.c_str(),
                       remaining, pipe.stream_by_frame.size());
        }
      } else if (graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[GRAPH] stream_map_miss seg=%zu frame_id=%lld input_seq=%lld stream_id=%s "
                     "missing=%d internal=%d map=%zu\n",
                     static_cast<std::size_t>(pipe.seg.id), static_cast<long long>(sample.frame_id),
                     static_cast<long long>(sample.input_seq), sample.stream_id.c_str(),
                     static_cast<int>(missing), static_cast<int>(looks_internal),
                     pipe.stream_by_frame.size());
      }
    }

    if (!pipe.pending_identities.empty()) {
      map_lookup_attempted = true;
      PipelineRuntime::PendingIdentity pending = pipe.pending_identities.front();
      pipe.pending_identities.pop_front();
      bool applied = false;
      if (sample.frame_id < 0 && pending.frame_id >= 0) {
        sample.frame_id = pending.frame_id;
        applied = true;
      }
      bool should_apply_pending_stream = false;
      if (!pending.stream_id.empty()) {
        if (prefer_mapped) {
          should_apply_pending_stream = sample.stream_id != pending.stream_id;
        } else {
          should_apply_pending_stream =
              sample.stream_id.empty() || is_internal_stream_id(sample.stream_id);
        }
      }
      if (should_apply_pending_stream) {
        map_hit = true;
        sample.stream_id = pending.stream_id;
        applied = true;
      }
      if (applied && graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[GRAPH] stream_map_fallback seg=%zu frame_id=%lld input_seq=%lld "
                     "stream_id=%s pending=%zu\n",
                     static_cast<std::size_t>(pipe.seg.id), static_cast<long long>(sample.frame_id),
                     static_cast<long long>(sample.input_seq), sample.stream_id.c_str(),
                     pipe.pending_identities.size());
      }
    }
    finalize_identity_diag();
  }
};

// =====================================================================================
// Split implementation chunks

} // namespace simaai::neat::graph

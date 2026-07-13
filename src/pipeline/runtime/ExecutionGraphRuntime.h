#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "graph/GraphTypes.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"
#include "graph/runtime/BlockingQueue.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/internal/RealtimeFrameCredit.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "pipeline/runtime/PipelineSegmentRuntime.h"
#include "pipeline/runtime/TraceMessageEvents.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace simaai::neat::pipeline_internal {
struct RealtimeFrameCreditLane;
using RealtimeFrameCreditLanePtr = std::shared_ptr<RealtimeFrameCreditLane>;
} // namespace simaai::neat::pipeline_internal

namespace simaai::neat::runtime {

using BlockingQueueSample = simaai::neat::graph::runtime::BlockingQueue<simaai::neat::Sample>;
using StageKeyBy = simaai::neat::graph::nodes::StageKeyBy;
using StageNodeOptions = simaai::neat::graph::nodes::StageNodeOptions;

struct RuntimeStageQueueMsg {
  simaai::neat::graph::PortId in_port = simaai::neat::graph::kInvalidPort;
  Sample sample;
  std::size_t edge_index = invalid_edge_index();
};

struct RuntimeSinkQueueMsg {
  Sample sample;
  std::size_t edge_index = invalid_edge_index();
};

using GraphSinkQueue = simaai::neat::graph::runtime::BlockingQueue<RuntimeSinkQueueMsg>;

struct DownstreamTarget {
  enum class Kind {
    StageGroup,
    PipelineInput,
    GraphSink,
    RealtimeLatestLink,
  };

  Kind kind = Kind::StageGroup;
  std::size_t index = 0; // stage group index, pipeline index, or sink node id
  simaai::neat::graph::PortId port = simaai::neat::graph::kInvalidPort;
  std::size_t edge_index = invalid_edge_index();
};

class RealtimeLatestLink {
public:
  using DispatchFn =
      std::function<bool(const DownstreamTarget&, simaai::neat::Sample&&, std::size_t)>;
  using StopFn = std::function<bool()>;
  using ErrorFn = std::function<void(const std::string&)>;

  struct Stats {
    std::uint64_t offered = 0;
    std::uint64_t scheduled = 0;
    std::uint64_t overwritten = 0;
    std::uint64_t dispatch_failed = 0;
    std::uint64_t ready_wait_ns = 0;
    std::uint64_t ready_wait_max_ns = 0;
    std::uint64_t dispatch_ns = 0;
    std::uint64_t dispatch_max_ns = 0;
    std::uint64_t no_credit_skips = 0;
    std::uint64_t credit_registered = 0;
    std::uint64_t credit_released_by_output = 0;
    std::uint64_t credit_released_without_output = 0;
    std::uint64_t credit_missing_key = 0;
    std::size_t credit_inflight = 0;
    std::size_t credit_limit = 0;
    std::size_t ready = 0;
  };

  RealtimeLatestLink(DownstreamTarget downstream, GraphLinkOptions options, std::string stream_id);
  RealtimeLatestLink(const RealtimeLatestLink&) = delete;
  RealtimeLatestLink& operator=(const RealtimeLatestLink&) = delete;
  ~RealtimeLatestLink();

  bool offer(simaai::neat::Sample&& sample, std::size_t edge_index);
  void add_edge_stream_id(std::size_t edge_index, const std::string& stream_id);
  void add_edge_stream_id(std::size_t edge_index, const std::string& stream_id,
                          const GraphLinkOptions& options);
  void start(DispatchFn dispatch, StopFn stop, ErrorFn error);
  void close();
  void join();
  Stats stats() const;
  std::string debug_stream_ids() const;
  const DownstreamTarget& downstream() const noexcept {
    return downstream_;
  }
  const GraphLinkOptions& options() const noexcept {
    return options_;
  }

private:
  struct Pending {
    simaai::neat::Sample sample;
    std::chrono::steady_clock::time_point ready_at;
    std::size_t edge_index = invalid_edge_index();
    bool has_sample = false;
    bool queued = false;
  };

  std::string key_for_(const simaai::neat::Sample& sample, std::size_t edge_index) const;
  void recompute_admission_options_locked_();
  pipeline_internal::RealtimeFrameCreditLanePtr credit_lane_for_key_locked_(const std::string& key);
  void configure_global_credit_limit_locked_();
  void run_();

  DownstreamTarget downstream_;
  GraphLinkOptions options_;
  DispatchFn dispatch_;
  StopFn stop_;
  ErrorFn error_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::unordered_map<std::string, Pending> pending_;
  std::unordered_map<std::string, pipeline_internal::RealtimeFrameCreditLanePtr> credit_lanes_;
  pipeline_internal::RealtimeFrameCreditLanePtr global_credit_lane_;
  std::unordered_set<std::size_t> edge_indices_;
  std::unordered_map<std::size_t, std::string> stream_id_by_edge_;
  std::unordered_map<std::size_t, GraphLinkOptions> link_options_by_edge_;
  std::deque<std::string> ready_;
  std::uint64_t credit_namespace_ = 0;
  int credit_limit_per_stream_ = 0;
  int credit_limit_global_ = 0;
  bool closed_ = false;
  std::thread worker_;
  std::atomic<std::uint64_t> offered_{0};
  std::atomic<std::uint64_t> scheduled_{0};
  std::atomic<std::uint64_t> overwritten_{0};
  std::atomic<std::uint64_t> dispatch_failed_{0};
  std::atomic<std::uint64_t> ready_wait_ns_{0};
  std::atomic<std::uint64_t> ready_wait_max_ns_{0};
  std::atomic<std::uint64_t> dispatch_ns_{0};
  std::atomic<std::uint64_t> dispatch_max_ns_{0};
  std::atomic<std::uint64_t> no_credit_skips_{0};
};

struct RuntimeStageEmitter final : simaai::neat::graph::StageEmitter {
  std::function<bool(simaai::neat::graph::StageOutMsg)> emit_fn;
  std::function<bool()> stop_requested_fn;

  bool emit(simaai::neat::graph::StageOutMsg msg) override {
    const auto credits = pipeline_internal::realtime_frame_credits_for_sample(msg.sample);
    const bool ok = emit_fn ? emit_fn(std::move(msg)) : false;
    if (ok && !credits.empty()) {
      std::lock_guard<std::mutex> lock(emitted_credit_mu_);
      if (track_emitted_credits_) {
        emitted_credits_.insert(emitted_credits_.end(), credits.begin(), credits.end());
      }
    }
    return ok;
  }

  bool stop_requested() const override {
    return stop_requested_fn ? stop_requested_fn() : true;
  }

  void begin_input_credit_tracking() {
    std::lock_guard<std::mutex> lock(emitted_credit_mu_);
    emitted_credits_.clear();
    track_emitted_credits_ = true;
  }

  std::vector<pipeline_internal::RealtimeFrameCredit> end_input_credit_tracking() {
    std::lock_guard<std::mutex> lock(emitted_credit_mu_);
    track_emitted_credits_ = false;
    std::vector<pipeline_internal::RealtimeFrameCredit> out;
    out.swap(emitted_credits_);
    return out;
  }

private:
  std::mutex emitted_credit_mu_;
  bool track_emitted_credits_ = false;
  std::vector<pipeline_internal::RealtimeFrameCredit> emitted_credits_;
};

struct StageRuntime {
  explicit StageRuntime(std::size_t capacity = 0) : inbox(capacity) {}

  struct Telemetry {
    std::atomic<std::uint64_t> mailbox_pop_calls{0};
    std::atomic<std::uint64_t> mailbox_pop_miss{0};
    std::atomic<std::uint64_t> mailbox_pop_wait_ns{0};
    std::atomic<std::uint64_t> mailbox_pop_wait_max_ns{0};
    std::atomic<std::uint64_t> on_input_calls{0};
    std::atomic<std::uint64_t> on_input_ns{0};
    std::atomic<std::uint64_t> on_input_max_ns{0};
    std::atomic<std::uint64_t> route_output_calls{0};
    std::atomic<std::uint64_t> route_output_ns{0};
    std::atomic<std::uint64_t> route_output_max_ns{0};
  };

  simaai::neat::graph::NodeId node_id = simaai::neat::graph::kInvalidNode;
  RuntimeStageEmitter emitter;
  std::unique_ptr<simaai::neat::graph::StageExecutor> exec;
  simaai::neat::graph::runtime::BlockingQueue<RuntimeStageQueueMsg> inbox;
  std::thread worker;
  std::atomic<bool> worker_done{false};
  std::vector<simaai::neat::graph::PortId> input_ports;
  std::vector<simaai::neat::graph::PortId> output_ports;
  simaai::neat::graph::StagePorts ports;
  Telemetry telemetry;
};

struct StageGroup {
  simaai::neat::graph::NodeId node_id = simaai::neat::graph::kInvalidNode;
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

struct ExecutionGraphRuntime {
  ExecutionGraphPlan plan;
  std::vector<std::string> node_labels;
  std::atomic<std::size_t> output_rr{0};

  std::vector<std::unique_ptr<PipelineSegmentRuntime>> pipelines;
  std::vector<std::unique_ptr<StageRuntime>> stages;
  std::vector<StageGroup> stage_groups;
  std::vector<std::unique_ptr<RealtimeLatestLink>> realtime_links;

  std::unordered_map<simaai::neat::graph::NodeId, std::size_t> node_to_pipeline;
  std::unordered_map<simaai::neat::graph::NodeId, std::size_t> node_to_stage_group;
  std::unordered_set<simaai::neat::graph::NodeId> direct_sink_nodes;

  std::unordered_map<std::uint64_t, std::vector<DownstreamTarget>> adjacency;
  std::atomic<bool> message_trace_enabled{false};
  std::atomic<std::uint64_t> trace_run_id_hash{0};
  std::atomic<std::uint64_t> trace_graph_id_hash{0};
  std::unordered_map<simaai::neat::graph::NodeId, std::shared_ptr<GraphSinkQueue>> sinks;

  // Graph-wide decoder admission state.  A dense multi-decoder graph reserves
  // and binds decoder daemon leases before any GStreamer fragment is started so
  // each decoder receives the same automatic pool/tuning decision.
  bool decoder_admission_active = false;
  std::array<std::uint8_t, 16> decoder_admission_group_uuid{};
};

} // namespace simaai::neat::runtime

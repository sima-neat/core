#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/RunDiagnostics.h"
#include "ExecutionGraphRuntime.h"
#include "EdgeRouter.h"
#include "PipelineSegmentRuntime.h"
#include "pipeline/Run.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace simaai::neat::graph {
struct GraphRunStats;
}

namespace simaai::neat::runtime {

struct ExecutionGraphPlan;
struct PipelineSegmentPlan;

enum class GraphSampleTimingKeyKind { OrigInputSeq, InputSeq, FrameId };

struct GraphSampleIdentityKey {
  std::string stream_id;
  GraphSampleTimingKeyKind kind = GraphSampleTimingKeyKind::OrigInputSeq;
  std::int64_t value = -1;
  bool operator==(const GraphSampleIdentityKey& other) const noexcept {
    return stream_id == other.stream_id && kind == other.kind && value == other.value;
  }
};

struct GraphSampleIdentityKeyHash {
  std::size_t operator()(const GraphSampleIdentityKey& key) const noexcept {
    std::size_t h = std::hash<std::string>{}(key.stream_id);
    h ^= static_cast<std::size_t>(key.kind) + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
    h ^= std::hash<std::int64_t>{}(key.value) + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
    return h;
  }
};

struct GraphSampleTimingState {
  std::chrono::steady_clock::time_point graph_entry_at{};
  std::string input_endpoint;
  std::uint64_t generation = 0;
  bool ambiguous = false;
};

struct GraphSampleTimingOrderEntry {
  GraphSampleIdentityKey key;
  std::uint64_t generation = 0;
};

struct GraphSampleTimingEvent {
  std::string endpoint;
  std::string stream_id;
  GraphSampleTimingKeyKind key_kind = GraphSampleTimingKeyKind::OrigInputSeq;
  std::int64_t key_value = -1;
  double timestamp_s = 0.0;
};

struct GraphRuntimeOptions {
  std::size_t edge_queue = 256;
  int push_timeout_ms = 5000;
  int pull_timeout_ms = 50;
  VerboseOptions verbose;
  RunOptions pipeline;
  PowerMonitorOptions power_monitor;

  EdgeRouterOptions router_options() const {
    return EdgeRouterOptions{
        .edge_queue = edge_queue,
        .push_timeout_ms = push_timeout_ms,
    };
  }
};

inline GraphRuntimeOptions
graph_runtime_options_from_run_options(const RunOptions& opt, const VerboseOptions& verbose = {}) {
  GraphRuntimeOptions out;
  out.verbose = verbose;
  out.pipeline = opt;
  out.power_monitor = opt.power_monitor;
  // Connected public Graph::build() uses RunOptions as the only customer-facing
  // knob. Treat RunOptions::power_monitor as graph-level board telemetry and do
  // not forward it to each materialized pipeline segment; otherwise a connected
  // graph with N segments would start N+1 board monitors and double-count rails.
  out.pipeline.power_monitor = PowerMonitorOptions{};
  return out;
}

enum class PushSamplePolicy {
  // Public linear Run compatibility: image Samples may use the legacy cv::Mat ingress path
  // when the pipeline is a simple raw-image appsrc pipeline.
  PublicCompatibility,
  // Graph-internal execution: Sample is the only runtime currency. Never downgrade
  // Sample -> cv::Mat, because that drops graph identity/timing metadata.
  PreserveSample,
};

struct RunCoreStartOptions {
  RunOptions run_options{};
  RunMode mode = RunMode::Async;
  std::shared_ptr<const cv::Mat> image_seed;
  std::optional<Sample> seed;
  std::optional<InputOptions> tensor_input_opt_for_cv;
  std::shared_ptr<void> guard;
  pipeline_internal::InputRouteProcessorPtr input_route_processor;
  std::string* last_pipeline = nullptr;
  const void* owner = nullptr;
  bool allow_startup_preflight = false;
  bool require_sink = false;
  GraphRuntimeOptions graph_options;
  std::shared_ptr<void> graph_verbose_guard;
  PushSamplePolicy push_sample_policy = PushSamplePolicy::PublicCompatibility;
};

struct RunCore {
  static std::shared_ptr<RunCore> start(ExecutionGraphPlan plan, RunCoreStartOptions opt);
  static std::shared_ptr<RunCore> create_graph_compat();
  static std::shared_ptr<RunCore> start_pipeline_segment(const PipelineSegmentPlan& segment,
                                                         RunCoreStartOptions opt);

  static std::shared_ptr<RunCore>
  start_single_pipeline(InputStream stream, const RunOptions& opt,
                        const InputStreamOptions& stream_opt, RunMode mode = RunMode::Async,
                        const std::optional<InputOptions>& tensor_input_opt_for_cv = std::nullopt,
                        pipeline_internal::InputRouteProcessorPtr input_route_processor = nullptr);

  ~RunCore();

  bool valid() const noexcept;
  bool can_push() const;
  bool can_pull() const;
  bool running() const;
  std::vector<std::string> input_names() const;
  std::vector<std::string> output_names() const;

  void close_input();
  void stop();
  void stop_graph();
  void close();

  bool push_samples(const Sample& msgs, bool block = true);
  bool push_named_samples(std::string_view input_name, const Sample& msgs, bool block = true);
  PullStatus pull(int timeout_ms, Sample& out, PullError* err = nullptr);
  PullStatus pull_named_output(std::string_view output_name, int timeout_ms, Sample& out,
                               PullError* err = nullptr);
  std::optional<Sample> pull_optional(int timeout_ms = -1);

  RunStats stats() const;
  InputStreamStats input_stats() const;
  RunDiagSnapshot diag_snapshot() const;
  std::string last_error() const;
  std::string diagnostics_summary() const;

  ExecutionGraphRuntime& graph_execution();
  const ExecutionGraphRuntime& graph_execution() const;
  bool graph_stop_requested() const;
  void graph_signal_stop();
  void graph_request_stop(const std::string& err);
  bool ensure_graph_pipeline_built(std::size_t index, const Sample& sample, std::string* err,
                                   bool allow_startup_preflight = false);
  bool graph_dispatch_to_stage_group(std::size_t group_index, simaai::neat::graph::PortId port,
                                     Sample&& sample, std::size_t edge_index,
                                     const EdgeRouterOptions& options);
  bool graph_push(simaai::neat::graph::NodeId node_id, simaai::neat::graph::PortId port,
                  bool has_port, const Sample& sample, const EdgeRouterOptions& options);
  void record_graph_sample_entry(std::string_view endpoint, const Sample& sample,
                                 std::chrono::steady_clock::time_point at);
  void record_graph_sample_output(std::string_view endpoint, const Sample& sample,
                                  std::chrono::steady_clock::time_point at);
  void graph_sanitize_pipeline_input(std::size_t index, Sample& sample);
  void graph_restore_stream_id_if_needed(std::size_t index, Sample& sample);
  std::optional<Sample> graph_pull(simaai::neat::graph::NodeId node_id, int timeout_ms);

  RunOptions opt;
  RunMode mode = RunMode::Async;

  // Phase 3B: the existing linear Run path is represented as one default
  // pipeline segment. Later Phase 3 slices will generalize this to a vector of
  // segments behind one ExecutionGraphPlan.
  PipelineSegmentRuntime pipeline;
  std::unique_ptr<ExecutionGraphRuntime> graph_execution_;
  // When a public Graph::build() compiles to a single linear pipeline, runtime
  // intentionally collapses to the lower-overhead legacy single-pipeline Run
  // path. Keep the compiled plan for reporting/visualization so graph metrics
  // can still show the customer's Graph::add topology instead of falling back
  // to node-metric order.
  std::unique_ptr<ExecutionGraphPlan> graph_export_plan_;
  GraphRuntimeOptions graph_options;
  PushSamplePolicy push_sample_policy = PushSamplePolicy::PublicCompatibility;
  std::shared_ptr<simaai::neat::graph::GraphRunStats> graph_stats;

  std::uint64_t latency_count = 0;
  double latency_mean_ms = 0.0;
  double latency_min_ms = 0.0;
  double latency_max_ms = 0.0;
  std::chrono::steady_clock::time_point created_at{};
  std::chrono::system_clock::time_point created_wall_at{};
  std::chrono::system_clock::time_point closed_wall_at{};
  std::string run_id;
  std::chrono::steady_clock::time_point first_output_at{};
  std::chrono::steady_clock::time_point last_output_at{};
  std::chrono::steady_clock::time_point first_pull_at{};
  std::chrono::steady_clock::time_point last_pull_at{};
  std::chrono::steady_clock::time_point measurement_last_output_at{};
  bool output_timing_init = false;
  bool pull_timing_init = false;
  bool measurement_active = false;
  bool measurement_output_timing_init = false;
  std::chrono::steady_clock::time_point measurement_started_at{};
  std::vector<double> measurement_latencies_ms;
  std::vector<double> measurement_frame_gaps_ms;
  std::vector<GraphSampleTimingEvent> measurement_graph_entries;
  std::vector<GraphSampleTimingEvent> measurement_graph_pulls;

  std::atomic<std::uint64_t> inputs_enqueued{0};
  std::atomic<std::uint64_t> inputs_dropped{0};
  std::atomic<std::uint64_t> inputs_pushed{0};
  std::atomic<std::uint64_t> outputs_ready{0};
  std::atomic<std::uint64_t> outputs_pulled{0};
  std::atomic<std::uint64_t> outputs_dropped{0};

  std::string error;
  std::string diag_sysinfo;
  std::unique_ptr<PowerMonitor> power_monitor;
  std::shared_ptr<void> graph_verbose_guard;
  std::atomic<bool> graph_output_rate_reported{false};
  std::atomic<bool> graph_sched_reported{false};

  mutable std::mutex latency_mu;
  mutable std::mutex graph_sample_timing_mu;
  std::unordered_map<GraphSampleIdentityKey, GraphSampleTimingState, GraphSampleIdentityKeyHash>
      graph_sample_timing_by_key;
  std::deque<GraphSampleTimingOrderEntry> graph_sample_timing_order;
  std::size_t graph_sample_timing_capacity = 4096;
  std::uint64_t graph_sample_timing_generation = 0;
  std::atomic<std::uint64_t> graph_sample_timing_unkeyed{0};
  std::atomic<std::uint64_t> graph_sample_timing_misses{0};
  mutable std::mutex error_mu;
  bool latency_init = false;
  std::atomic<bool> stop_requested{false};
  std::atomic<bool> closed{false};
  bool diag_enabled = false;
  std::atomic<bool> diag_logged{false};
};

void initialize_run_identity(RunCore& core);

} // namespace simaai::neat::runtime

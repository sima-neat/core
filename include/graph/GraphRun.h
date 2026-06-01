/**
 * @file
 * @ingroup graph
 * @brief Internal `GraphRun` — runtime handle for the actor-style graph substrate.
 *
 * Application code should use `simaai::neat::Graph::build() -> simaai::neat::Run`.
 * This type remains available in the source tree for runtime/compiler internals and
 * focused internal tests while the old low-level graph API is absorbed by the public
 * Graph/Run surface.
 *
 * @see simaai::neat::Graph
 * @see simaai::neat::Run
 */
#pragma once

#include "graph/Graph.h"
#include "pipeline/Run.h"

#include <cstdint>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace simaai::neat::graph {

/**
 * @brief Internal tuning knobs for the actor-runtime substrate.
 *
 * Public callers configure graph execution through `RunOptions` on
 * `simaai::neat::Graph::build()`.
 *
 * @ingroup graph
 */
struct GraphRunOptions {
  std::size_t edge_queue =
      256; ///< Bounded capacity for graph edge/stage/sink queues. `0` = unbounded.
  int push_timeout_ms =
      5000; ///< Max wait (ms) on `push()` before failing fast with a backpressure error.
  int pull_timeout_ms = 50; ///< Poll timeout (ms) for internal pop/pull loops.
  VerboseOptions verbose;   ///< Verbosity for graph build/runtime progress logs.
  RunOptions pipeline; ///< Underlying pipeline-runtime options forwarded to GStreamer-side runs.
  PowerMonitorOptions power_monitor; ///< Optional graph-level board power monitor.

  /**
   * @brief Enable one graph-level board power monitor using built-in auto-detect.
   *
   * This is the preferred API for graph-level power because one graph can contain
   * multiple pipeline segments and per-segment monitors would double-count board rails.
   */
  GraphRunOptions& enable_board_power(int sample_interval_ms = 100) {
    power_monitor = board_power_monitor_options(sample_interval_ms);
    return *this;
  }

  /**
   * @brief Enable one graph-level Modalix SOM power monitor.
   *
   * Prefer `enable_board_power()` unless you need to force the SOM profile.
   */
  GraphRunOptions& enable_modalix_som_power(int sample_interval_ms = 100) {
    power_monitor = modalix_som_power_monitor_options(sample_interval_ms);
    return *this;
  }

  /**
   * @brief Enable one graph-level Modalix DVT power monitor.
   */
  GraphRunOptions& enable_modalix_dvt_power(int sample_interval_ms = 100) {
    power_monitor = modalix_dvt_power_monitor_options(sample_interval_ms);
    return *this;
  }

  /// Disable graph-level power monitoring after it was enabled/configured.
  GraphRunOptions& disable_power_monitor() {
    power_monitor = PowerMonitorOptions{};
    return *this;
  }
};

/**
 * @brief Per-node, per-stream telemetry collector for a running `GraphRun`.
 *
 * Records sample counts and first/last timestamps per output stream, per node. Used by
 * `PullLoop` and the various `emit_*_summary()` helpers.
 *
 * @ingroup graph
 */
struct GraphRunStats {
  /// Per-stream counters within a single node.
  struct StreamStat {
    int64_t count = 0;                             ///< Samples seen on this stream.
    std::chrono::steady_clock::time_point first{}; ///< Timestamp of the first sample.
    std::chrono::steady_clock::time_point last{};  ///< Timestamp of the most recent sample.
    bool initialized = false;                      ///< True once any sample has been recorded.
  };

  /// Aggregated counters for a single node, plus per-stream breakdown.
  struct NodeStat {
    int64_t total = 0;                             ///< Samples seen across all streams.
    std::chrono::steady_clock::time_point first{}; ///< Timestamp of the first sample.
    std::chrono::steady_clock::time_point last{};  ///< Timestamp of the most recent sample.
    bool initialized = false;                      ///< True once any sample has been recorded.
    std::unordered_map<std::string, StreamStat> streams; ///< Per-stream counters.
  };

  /// Lock-free snapshot of a single node's stats — safe to copy/return.
  struct Snapshot {
    NodeId node_id = kInvalidNode;                   ///< Node these stats are for.
    int64_t total = 0;                               ///< Total samples seen.
    std::chrono::steady_clock::time_point first{};   ///< First-sample timestamp.
    std::chrono::steady_clock::time_point last{};    ///< Last-sample timestamp.
    std::unordered_map<std::string, int64_t> counts; ///< Per-stream counts.
    std::unordered_map<std::string, std::chrono::steady_clock::time_point>
        last_seen; ///< Per-stream last-seen.
  };

  /// Record a single sample arriving at `node_id`. Thread-safe.
  void record(NodeId node_id, const Sample& sample);
  /// Take a consistent snapshot of all per-node stats.
  std::vector<Snapshot> snapshot() const;
  /// True iff no samples have been recorded yet.
  bool empty() const;
  /// Aggregate per-stream counts across `nodes` (or all nodes if empty).
  std::unordered_map<std::string, int64_t>
  stream_counts(const std::vector<NodeId>& nodes = {}) const;
  /// True iff any of `expected` stream-ids has zero samples on the given nodes.
  bool has_missing_streams(const std::unordered_set<std::string>& expected,
                           const std::vector<NodeId>& nodes = {}) const;
  /// Comma-separated list of expected streams that haven't been seen — for diagnostics.
  std::string missing_streams_list(const std::unordered_set<std::string>& expected,
                                   const std::vector<NodeId>& nodes = {}) const;

private:
  mutable std::mutex mu_;
  std::unordered_map<NodeId, NodeStat> nodes_;
};

/**
 * @brief Tuning knobs for `GraphRun::pull_until()` / `PullLoop::run()`.
 *
 * Controls when a pull loop stops: hit the per-stream sample target, hit the stall
 * deadline (no progress for `stall_ms`), or hit `max_runtime_ms`.
 * @ingroup graph
 */
struct GraphRunPullOptions {
  int per_stream_target =
      0;               ///< Stop when each stream has produced this many samples. `0` = no target.
  int stall_ms = 0;    ///< Stop after this many ms with no progress. `0` disables the stall guard.
  int timeout_ms = 50; ///< Per-poll timeout (ms).
  int max_runtime_ms = -1; ///< Hard wall-clock cap (ms). `-1` = no cap.
  std::vector<std::string>
      stream_ids; ///< Restrict tracking/expectation to these streams (empty = all).
};

/**
 * @brief Live runtime handle for a compiled `Graph`.
 *
 * Returned by `graph::build()`. Move-only (queues, threads, resources are not
 * copyable). Push samples into named-port `Input` handles, pull from `Output` handles,
 * use the fluent `PullLoop` for the common collect-until-done workflow.
 *
 * @ingroup graph
 */
class GraphRun {
private:
  struct State;

public:
  /**
   * @brief Push handle for a node's input port.
   *
   * Obtain via `GraphRun::input(node_id)` or `GraphRun::input(node_id, port)`. Holds a
   * weak reference to the runtime and fails cleanly after the parent `GraphRun` is gone.
   */
  class Input {
  public:
    /// Push samples into the bound port. Returns false on backpressure timeout.
    bool push(const Sample& samples) const;

  private:
    friend class GraphRun;
    Input(std::weak_ptr<State> state, NodeId node, PortId port, bool has_port)
        : state_(std::move(state)), node_(node), port_(port), has_port_(has_port) {}

    std::weak_ptr<State> state_;
    NodeId node_ = kInvalidNode;
    PortId port_ = kInvalidPort;
    bool has_port_ = false;
  };

  /**
   * @brief Pull handle for a node's output stream.
   *
   * Obtained from `GraphRun::output(node_id)`. Optionally records to a `GraphRunStats`
   * collector on every successful pull.
   */
  class Output {
  public:
    /// Pull one sample. `timeout_ms < 0` blocks indefinitely; returns `nullopt` on timeout.
    std::optional<Sample> pull(int timeout_ms = -1, GraphRunStats* stats = nullptr) const;
    /// Like `pull()` but throws `NeatError` on timeout.
    Sample pull_or_throw(int timeout_ms = -1, GraphRunStats* stats = nullptr) const;
    /// Node this output belongs to.
    NodeId node_id() const {
      return node_;
    }

  private:
    friend class GraphRun;
    Output(std::weak_ptr<State> state, NodeId node) : state_(std::move(state)), node_(node) {}

    std::weak_ptr<State> state_;
    NodeId node_ = kInvalidNode;
  };

  /**
   * @brief Tracks progress against a per-stream target and a stall deadline.
   *
   * Used by `pull_until()` to decide when to stop. After each `update()`, query `done()`
   * (target reached) and `stalled()` (no progress for the configured interval).
   */
  class StallGuard {
  public:
    /// Refresh the guard with the current `stats`. Returns true if `done() || stalled()`.
    bool update(const GraphRunStats& stats);
    /// True once every tracked stream has reached the per-stream target.
    bool done() const {
      return done_;
    }
    /// True if no progress has been observed for the configured stall interval.
    bool stalled() const {
      return stalled_;
    }
    /// Current per-stream target — useful for progress display.
    int64_t target_progress() const {
      return target_progress_;
    }

  private:
    friend class GraphRun;
    StallGuard(std::vector<NodeId> nodes, std::vector<std::string> streams, int per_stream_target,
               int stall_ms);

    std::vector<NodeId> nodes_;
    std::vector<std::string> streams_;
    int per_stream_target_ = 0;
    int stall_ms_ = 0;
    bool initialized_ = false;
    bool done_ = false;
    bool stalled_ = false;
    int64_t target_progress_ = 0;
    std::chrono::steady_clock::time_point last_progress{};
  };

  /**
   * @brief Fluent builder for the common "collect samples until target/stall" workflow.
   *
   * Configure with chainable setters, then call `run()`. The pull loop reads from the
   * configured outputs until per-stream targets are met, the runtime stalls, or the
   * max-runtime cap is hit.
   *
   * @code
   * run.collect({run.output(sink_id)})
   *    .per_stream_target(100)
   *    .stall_after_ms(2000)
   *    .on_sample([](const Sample& s, NodeId n) { ... })
   *    .run();
   * @endcode
   */
  class PullLoop {
  public:
    /// Stop once each tracked stream has produced this many samples.
    PullLoop& per_stream_target(int n);
    /// Stop after this many ms with no progress (stall detection).
    PullLoop& stall_after_ms(int ms);
    /// Per-poll timeout (ms).
    PullLoop& timeout_ms(int ms);
    /// Hard wall-clock cap (ms) on the entire pull loop.
    PullLoop& max_runtime_ms(int ms);
    /// Limit tracking to these stream-ids (and treat anything else as unexpected).
    PullLoop& expect_streams(std::vector<std::string> ids);
    /// Callback fired on every successfully pulled sample.
    PullLoop& on_sample(std::function<void(const Sample&, NodeId)> cb);
    /// Live reference to the stats collector this pull loop writes to.
    GraphRunStats& stats();
    /// Execute the configured pull loop. Blocks until done/stall/max-runtime.
    void run();

  private:
    friend class GraphRun;
    PullLoop(std::weak_ptr<State> state, std::vector<Output> outputs,
             std::vector<NodeId> output_nodes, GraphRunStats* stats);

    std::weak_ptr<State> state_;
    std::vector<Output> outputs_;
    std::vector<NodeId> output_nodes_;
    GraphRunStats* stats_ = nullptr;
    GraphRunPullOptions opt_;
    std::unordered_set<std::string> expected_;
    std::unordered_set<std::string> unknown_;
    bool saw_empty_stream_id_ = false;
    std::function<void(const Sample&, NodeId)> on_sample_;
  };

  /// Default-construct an empty (non-running) handle. Useful as a movable placeholder.
  GraphRun() = default;
  /// Copy is disabled — `GraphRun` owns runtime resources and is move-only.
  GraphRun(const GraphRun&) = delete;
  /// Copy assignment is disabled — `GraphRun` is move-only.
  GraphRun& operator=(const GraphRun&) = delete;

  /// Move-construct, transferring ownership of the underlying runtime.
  GraphRun(GraphRun&&) noexcept;
  /// Move-assign, transferring ownership of the underlying runtime.
  GraphRun& operator=(GraphRun&&) noexcept;
  /// Destroy the handle, stopping the runtime if still running.
  ~GraphRun();

  /// True iff this handle owns a live runtime.
  explicit operator bool() const noexcept;
  /// True iff the underlying graph is currently running.
  bool running() const;

  /// Push samples to a node's default input port. Returns false on backpressure timeout.
  bool push(NodeId node_id, const Sample& samples);
  /// Push samples to a specific named port on a node.
  bool push(NodeId node_id, PortId port, const Sample& samples);

  /// Pull from a node's output. `timeout_ms < 0` blocks; returns `nullopt` on timeout.
  std::optional<Sample> pull(NodeId node_id, int timeout_ms = -1);

  /// Get an `Input` push handle bound to a node's default input port.
  Input input(NodeId node_id);
  /// Get an `Input` push handle bound to a specific named port on a node.
  Input input(NodeId node_id, PortId port);
  /// Get an `Output` pull handle for a node.
  Output output(NodeId node_id);

  /// Lazily allocate and return a stats collector tied to this run.
  GraphRunStats& enable_stats();
  /// Read the stats collector, if `enable_stats()` was called; else `nullptr`.
  const GraphRunStats* stats() const;
  /// Begin a fluent `PullLoop` over the given outputs.
  PullLoop collect(const std::vector<Output>& outputs, GraphRunStats* stats = nullptr);

  /// Pull from whichever of `outputs` produces a sample first. `out_node` reports the source.
  std::optional<Sample> pull_any(const std::vector<Output>& outputs, int timeout_ms = -1,
                                 GraphRunStats* stats = nullptr, NodeId* out_node = nullptr);
  /// Drain `warmup_count` samples (one per stream) and discard them.
  bool warmup(const std::vector<Output>& outputs, int warmup_count, int timeout_ms = -1);
  /// Loop `pull_any()` until target/stall/max-runtime, optionally invoking `on_sample` per pull.
  void pull_until(const std::vector<Output>& outputs, GraphRunStats& stats,
                  const GraphRunPullOptions& opt,
                  const std::function<void(const Sample&, NodeId)>& on_sample = {});
  /// Construct a `StallGuard` configured for the given outputs.
  StallGuard stall_guard(const std::vector<Output>& outputs, int per_stream_target, int stall_ms,
                         std::vector<std::string> stream_ids = {});
  /// Print a per-stream rate summary line — for end-of-run logging.
  void emit_rate_summary(const GraphRunStats& stats) const;
  /// Same as above, using the run's internal stats collector.
  void emit_rate_summary() const;
  /// Print a per-stream count summary.
  void emit_stream_summary(const GraphRunStats& stats) const;
  /// Same as above, using the run's internal stats collector.
  void emit_stream_summary() const;
  /// Print both rate and stream summaries.
  void emit_summary(const GraphRunStats& stats) const;
  /// Same as above, using the run's internal stats collector.
  void emit_summary() const;
  /// Return unified runtime metrics for recorded graph stats.
  RuntimeMetrics metrics(const RuntimeMetricsOptions& opt = {}) const;
  /// Render graph metrics in the requested format.
  std::string metrics_report(const RuntimeMetricsOptions& opt = {},
                             RuntimeMetricsFormat format = RuntimeMetricsFormat::Text) const;
  /// Convenience overload for selecting the output format with default options.
  std::string metrics_report(RuntimeMetricsFormat format) const;

  /// Human-readable description of the runtime graph (for diagnostics).
  std::string describe() const;

  /// Stop the underlying runtime. Idempotent.
  void stop();
  /// Most recent error message, or empty string if none.
  std::string last_error() const;
  /// Throw `NeatError` if `last_error()` is non-empty.
  void last_error_or_throw() const;

private:
  std::shared_ptr<State> state_;

  explicit GraphRun(std::shared_ptr<State> state);
  static bool push_state(const std::shared_ptr<State>& state, NodeId node_id, PortId port,
                         bool has_port, const Sample& sample);
  static std::optional<Sample> pull_state(const std::shared_ptr<State>& state, NodeId node_id,
                                          int timeout_ms);
  static std::optional<Sample> pull_any_state(const std::shared_ptr<State>& state,
                                              const std::vector<Output>& outputs, int timeout_ms,
                                              GraphRunStats* stats, NodeId* out_node);
  static void pull_until_state(const std::shared_ptr<State>& state,
                               const std::vector<Output>& outputs, GraphRunStats& stats,
                               const GraphRunPullOptions& opt,
                               const std::function<void(const Sample&, NodeId)>& on_sample);
  friend GraphRun build(Graph graph, const GraphRunOptions& opt);
};

} // namespace simaai::neat::graph

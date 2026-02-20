/**
 * @file
 * @ingroup graph
 * @brief Runtime handle for a compiled hybrid graph.
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
#include <vector>

namespace simaai::neat::graph {

struct GraphRunOptions {
  // Bounded capacity for graph edge/stage/sink queues. 0 means unbounded.
  std::size_t edge_queue = 256;
  // Maximum wait to enqueue before failing fast with a backpressure error.
  int push_timeout_ms = 5000;
  // Poll timeout for pop/pull loops.
  int pull_timeout_ms = 50;
  RunOptions pipeline;
};

struct GraphRunStats {
  struct StreamStat {
    int64_t count = 0;
    std::chrono::steady_clock::time_point first{};
    std::chrono::steady_clock::time_point last{};
    bool initialized = false;
  };

  struct NodeStat {
    int64_t total = 0;
    std::chrono::steady_clock::time_point first{};
    std::chrono::steady_clock::time_point last{};
    bool initialized = false;
    std::unordered_map<std::string, StreamStat> streams;
  };

  struct Snapshot {
    NodeId node_id = kInvalidNode;
    int64_t total = 0;
    std::chrono::steady_clock::time_point first{};
    std::chrono::steady_clock::time_point last{};
    std::unordered_map<std::string, int64_t> counts;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> last_seen;
  };

  void record(NodeId node_id, const Sample& sample);
  std::vector<Snapshot> snapshot() const;
  bool empty() const;
  std::unordered_map<std::string, int64_t>
  stream_counts(const std::vector<NodeId>& nodes = {}) const;
  bool has_missing_streams(const std::unordered_set<std::string>& expected,
                           const std::vector<NodeId>& nodes = {}) const;
  std::string missing_streams_list(const std::unordered_set<std::string>& expected,
                                   const std::vector<NodeId>& nodes = {}) const;

private:
  mutable std::mutex mu_;
  std::unordered_map<NodeId, NodeStat> nodes_;
};

struct GraphRunPullOptions {
  int per_stream_target = 0;
  int stall_ms = 0;
  int timeout_ms = 50;
  int max_runtime_ms = -1;
  std::vector<std::string> stream_ids;
};

class GraphRun {
public:
  class Input {
  public:
    bool push(const Sample& sample) const;

  private:
    friend class GraphRun;
    Input(GraphRun* run, NodeId node, PortId port, bool has_port)
        : run_(run), node_(node), port_(port), has_port_(has_port) {}

    GraphRun* run_ = nullptr;
    NodeId node_ = kInvalidNode;
    PortId port_ = kInvalidPort;
    bool has_port_ = false;
  };

  class Output {
  public:
    std::optional<Sample> pull(int timeout_ms = -1, GraphRunStats* stats = nullptr) const;
    Sample pull_or_throw(int timeout_ms = -1, GraphRunStats* stats = nullptr) const;
    NodeId node_id() const {
      return node_;
    }

  private:
    friend class GraphRun;
    Output(GraphRun* run, NodeId node) : run_(run), node_(node) {}

    GraphRun* run_ = nullptr;
    NodeId node_ = kInvalidNode;
  };

  class StallGuard {
  public:
    bool update(const GraphRunStats& stats);
    bool done() const {
      return done_;
    }
    bool stalled() const {
      return stalled_;
    }
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

  class PullSession {
  public:
    PullSession& per_stream_target(int n);
    PullSession& stall_after_ms(int ms);
    PullSession& timeout_ms(int ms);
    PullSession& max_runtime_ms(int ms);
    PullSession& expect_streams(std::vector<std::string> ids);
    PullSession& on_sample(std::function<void(const Sample&, NodeId)> cb);
    GraphRunStats& stats();
    void run();

  private:
    friend class GraphRun;
    PullSession(GraphRun* run, const std::vector<Output>* outputs, std::vector<NodeId> output_nodes,
                GraphRunStats* stats);

    GraphRun* run_ = nullptr;
    const std::vector<Output>* outputs_ = nullptr;
    std::vector<NodeId> output_nodes_;
    GraphRunStats* stats_ = nullptr;
    GraphRunPullOptions opt_;
    std::unordered_set<std::string> expected_;
    std::unordered_set<std::string> unknown_;
    bool saw_empty_stream_id_ = false;
    std::function<void(const Sample&, NodeId)> on_sample_;
  };

  GraphRun() = default;
  GraphRun(const GraphRun&) = delete;
  GraphRun& operator=(const GraphRun&) = delete;

  GraphRun(GraphRun&&) noexcept;
  GraphRun& operator=(GraphRun&&) noexcept;
  ~GraphRun();

  explicit operator bool() const noexcept;
  bool running() const;

  bool push(NodeId node_id, const Sample& sample);
  bool push(NodeId node_id, PortId port, const Sample& sample);

  std::optional<Sample> pull(NodeId node_id, int timeout_ms = -1);

  Input input(NodeId node_id);
  Input input(NodeId node_id, PortId port);
  Output output(NodeId node_id);

  GraphRunStats& enable_stats();
  const GraphRunStats* stats() const;
  PullSession collect(const std::vector<Output>& outputs, GraphRunStats* stats = nullptr);

  std::optional<Sample> pull_any(const std::vector<Output>& outputs, int timeout_ms = -1,
                                 GraphRunStats* stats = nullptr, NodeId* out_node = nullptr);
  bool warmup(const std::vector<Output>& outputs, int warmup_count, int timeout_ms = -1);
  void pull_until(const std::vector<Output>& outputs, GraphRunStats& stats,
                  const GraphRunPullOptions& opt,
                  const std::function<void(const Sample&, NodeId)>& on_sample = {});
  StallGuard stall_guard(const std::vector<Output>& outputs, int per_stream_target, int stall_ms,
                         std::vector<std::string> stream_ids = {});
  void emit_rate_summary(const GraphRunStats& stats) const;
  void emit_rate_summary() const;
  void emit_stream_summary(const GraphRunStats& stats) const;
  void emit_stream_summary() const;
  void emit_summary(const GraphRunStats& stats) const;
  void emit_summary() const;

  std::string describe() const;

  void stop();
  std::string last_error() const;
  void last_error_or_throw() const;

private:
  struct State;
  std::shared_ptr<State> state_;

  explicit GraphRun(std::shared_ptr<State> state);
  friend class GraphSession;
};

} // namespace simaai::neat::graph

/**
 * @file
 * @ingroup graph_nodes
 * @brief Stage executor for fair multi-stream scheduling.
 */
#pragma once

#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace simaai::neat::graph::nodes {

/**
 * @brief Policy used by `StreamScheduler` when a per-stream queue overflows.
 *
 * Selects which sample is dropped to make room when an incoming sample arrives at a
 * full per-stream queue.
 *
 * @ingroup graph
 */
enum class StreamDropPolicy {
  DropOldest = 0, ///< Drop the oldest queued sample when full.
  DropNewest,     ///< Drop the newly arrived sample when full.
};

/**
 * @brief Configuration for a `StreamScheduler` stage.
 *
 * Sets per-stream queue depth, overflow drop policy, and an upper bound on the per-input
 * batch emitted in one round of scheduling.
 *
 * @ingroup graph
 */
struct StreamSchedulerOptions {
  std::size_t per_stream_queue = 2; ///< Max samples held per stream before drop policy applies.
  StreamDropPolicy drop_policy =
      StreamDropPolicy::DropOldest; ///< Overflow policy when a per-stream queue is full.
  int max_batch = 1;                ///< Scheduling-only batch (emit up to N per input).
  std::vector<std::string> inputs;  ///< Optional named input ports used as stream-id fallbacks.
};

/**
 * @brief Stage executor that round-robin-schedules samples across multiple streams.
 *
 * Maintains a per-stream queue and emits samples in a fair round-robin order across
 * active streams, applying `StreamDropPolicy` when a stream's queue overflows.
 *
 * @see StreamSchedulerNode
 * @see StreamDropPolicy
 * @see StageExecutor
 * @ingroup graph
 */
class StreamScheduler final : public simaai::neat::graph::StageExecutor {
public:
  /// Construct a StreamScheduler from the given options.
  explicit StreamScheduler(StreamSchedulerOptions opt);

  /// Bind the executor to its runtime ports.
  void set_ports(const StagePorts& ports) override;
  /// Enqueue the incoming sample into its stream's queue and emit ready samples.
  void on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) override;

private:
  void ensure_stream_(const std::string& stream_id);
  bool emit_one_(std::vector<StageOutMsg>& out);
  std::string fallback_stream_id_for_port_(PortId port) const;

  StreamSchedulerOptions opt_;
  std::unordered_map<std::string, std::deque<Sample>> queues_;
  std::deque<std::string> rr_order_;
  std::unordered_set<std::string> active_;
  std::unordered_map<PortId, std::string> fallback_stream_id_by_port_;
  PortId out_port_ = kInvalidPort;
};

/**
 * @brief Convenience helper that wraps a `StreamScheduler` executor in a `StageNode`.
 *
 * @param opt    Optional scheduling configuration.
 * @param label  Optional human-readable label.
 * @param input  Name of the single input port (default `"in"`).
 * @param output Name of the single output port (default `"out"`).
 * @return Shared pointer to a `graph::Node` ready for insertion in a runtime graph.
 */
std::shared_ptr<simaai::neat::graph::Node> StreamSchedulerNode(StreamSchedulerOptions opt = {},
                                                               std::string label = {},
                                                               std::string input = "in",
                                                               std::string output = "out");

} // namespace simaai::neat::graph::nodes

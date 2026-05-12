/**
 * @file
 * @ingroup graph_nodes
 * @brief Stage executor that joins multiple inputs into a Bundle sample.
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
 * @brief Selects how the `JoinBundle` stage groups samples coming from different inputs.
 *
 * The chosen policy determines the join key used to decide which samples belong to the
 * same emitted bundle (e.g., match by stream id + frame id, or by stream id + presentation
 * timestamp).
 *
 * @ingroup graph
 */
enum class JoinKeyPolicy {
  StreamFrame = 0, ///< Join on `(stream_id, frame_id)`.
  StreamPts,       ///< Join on `(stream_id, pts)`.
};

/**
 * @brief Configuration for a `JoinBundle` stage executor.
 *
 * Specifies the named input ports to join, which inputs are required for emission, the join
 * key policy, and pending-bookkeeping bounds (max keys, optional timeout-based eviction).
 *
 * @ingroup graph
 */
struct JoinBundleOptions {
  std::vector<std::string> inputs;          ///< Names of input ports participating in the join.
  std::unordered_set<std::string> required; ///< Inputs required for emission; default: all inputs.
  JoinKeyPolicy key_policy = JoinKeyPolicy::StreamFrame; ///< How samples are keyed for joining.
  bool emit_partial = false; ///< If true, emit a bundle even when only required inputs are present.
  std::size_t max_pending_keys =
      4096;           ///< Upper bound on keys held in flight before evicting oldest.
  int timeout_ms = 0; ///< Per-key timeout (ms); 0 disables timeout-based eviction.
};

/**
 * @brief Stage executor that joins samples from multiple input ports into a single bundled output.
 *
 * Buffers samples per join key (selected by `JoinKeyPolicy`) and emits a bundle once the
 * required inputs are present (or once `emit_partial` allows). Unmatched keys are evicted
 * by capacity (`max_pending_keys`) or by timeout (`timeout_ms`).
 *
 * @see JoinBundleNode
 * @see JoinKeyPolicy
 * @see StageExecutor
 * @ingroup graph
 */
class JoinBundle final : public simaai::neat::graph::StageExecutor {
public:
  /// Construct a JoinBundle from the given options.
  explicit JoinBundle(JoinBundleOptions opt);

  /// Bind the executor to its runtime ports.
  void set_ports(const StagePorts& ports) override;
  /// Buffer the incoming sample under its join key, emit a bundle if ready.
  void on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) override;
  /// Periodic tick used to evict timed-out pending keys.
  void on_tick(std::int64_t now_ns, std::vector<StageOutMsg>& out) override;

private:
  /// Per-key in-flight bundle being assembled — one entry per pending join key.
  struct Pending {
    std::unordered_map<PortId, Sample>
        samples;                   ///< Samples received per input port for this bundle.
    std::int64_t last_seen_ns = 0; ///< Timestamp (ns) of the most recent sample on any port.
  };

  std::string make_key_(const Sample& sample) const;
  void touch_key_(const std::string& key);
  void evict_expired_(std::int64_t now_ns);
  void evict_oldest_();
  bool ready_(const Pending& pending) const;
  void erase_key_(const std::string& key);

  JoinBundleOptions opt_;
  std::vector<std::string> input_names_;
  std::unordered_map<PortId, std::string> port_names_;
  std::unordered_map<std::string, PortId> name_to_port_;
  std::unordered_set<PortId> required_ports_;
  std::unordered_map<std::string, Pending> pending_;
  std::deque<std::string> order_;
  PortId out_port_ = kInvalidPort;
};

/**
 * @brief Convenience helper that wraps a `JoinBundle` executor in a `StageNode`.
 *
 * @param inputs Names of the input ports to join.
 * @param label  Optional human-readable label.
 * @param output Name of the bundle output port (default `"bundle"`).
 * @param opt    Optional join configuration overrides.
 * @return Shared pointer to a `graph::Node` ready for insertion in a runtime graph.
 */
std::shared_ptr<simaai::neat::graph::Node> JoinBundleNode(std::vector<std::string> inputs,
                                                          std::string label = {},
                                                          std::string output = "bundle",
                                                          JoinBundleOptions opt = {});

} // namespace simaai::neat::graph::nodes

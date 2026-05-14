/**
 * @file
 * @ingroup graph_nodes
 * @brief Stage executor to join encoded payload with metadata into a Bundle.
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
#include <utility>
#include <vector>

namespace simaai::neat::graph::nodes {

/**
 * @brief Configuration for a `JoinEncodedWithMeta` stage executor.
 *
 * Customises which input port carries the encoded payload, naming for bundle fields, and
 * pending-bookkeeping bounds for the join.
 *
 * @ingroup graph
 */
struct JoinEncodedWithMetaOptions {
  /// Optional: explicitly mark which input port carries encoded data.
  PortId encoded_port = kInvalidPort;
  /// Optional: PortId -> name map for bundle field names.
  std::unordered_map<PortId, std::string> port_names;
  /// Fallback name when encoded detection is implicit.
  std::string encoded_name = "encoded";
  /// Max number of pending keys to hold before eviction.
  std::size_t max_pending = 1024;
  /// Emit a bundle even if some meta inputs are missing.
  bool emit_partial = true;
};

/**
 * @brief Stage executor that bundles an encoded payload (e.g., H.264) with associated metadata.
 *
 * Specialised join used by detection pipelines that overlay model results onto an encoded
 * stream: pairs each encoded buffer with the matching metadata sample (e.g., GstSimaMeta
 * detection results) keyed by stream/frame and emits a bundle.
 *
 * @see JoinEncodedWithMetaNode
 * @see StageExecutor
 * @ingroup graph
 */
class JoinEncodedWithMeta final : public simaai::neat::graph::StageExecutor {
public:
  /// Construct a JoinEncodedWithMeta from the given options.
  explicit JoinEncodedWithMeta(JoinEncodedWithMetaOptions opt);

  /// Bind the executor to its runtime ports.
  void set_ports(const StagePorts& ports) override;
  /// Match the incoming sample against pending entries and emit a bundle when ready.
  void on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) override;
  /// Periodic tick used to evict stale pending entries.
  void on_tick(std::int64_t now_ns, std::vector<StageOutMsg>& out) override;

private:
  std::string make_key_(const Sample& sample) const;
  bool is_encoded_(PortId port, const Sample& sample) const;
  std::string field_name_(PortId port, const Sample& sample, bool encoded) const;
  void evict_if_needed_();

  JoinEncodedWithMetaOptions opt_;
  std::unordered_map<std::string, std::unordered_map<PortId, Sample>> pending_;
  std::deque<std::string> order_;
  PortId out_port_ = kInvalidPort;
};

/**
 * @brief Convenience helper that wraps a `JoinEncodedWithMeta` executor in a `StageNode`.
 *
 * @param inputs Names of the input ports (typically encoded + metadata).
 * @param label  Optional human-readable label.
 * @param output Name of the bundle output port (default `"bundle"`).
 * @return Shared pointer to a `graph::Node` ready for insertion in a runtime graph.
 */
std::shared_ptr<simaai::neat::graph::Node> JoinEncodedWithMetaNode(std::vector<std::string> inputs,
                                                                   std::string label = {},
                                                                   std::string output = "bundle");

} // namespace simaai::neat::graph::nodes

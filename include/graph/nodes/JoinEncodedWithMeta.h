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

struct JoinEncodedWithMetaOptions {
  // Optional: explicitly mark which input port carries encoded data.
  PortId encoded_port = kInvalidPort;
  // Optional: PortId -> name map for bundle field names.
  std::unordered_map<PortId, std::string> port_names;
  // Fallback name when encoded detection is implicit.
  std::string encoded_name = "encoded";
  // Max number of pending keys to hold before eviction.
  std::size_t max_pending = 1024;
  // Emit a bundle even if some meta inputs are missing.
  bool emit_partial = true;
};

class JoinEncodedWithMeta final : public simaai::neat::graph::StageExecutor {
public:
  explicit JoinEncodedWithMeta(JoinEncodedWithMetaOptions opt);

  void set_ports(const StagePorts& ports) override;
  void on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) override;
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

// Convenience: wrap JoinEncodedWithMeta in a StageNode.
std::shared_ptr<simaai::neat::graph::Node> JoinEncodedWithMetaNode(std::vector<std::string> inputs,
                                                                   std::string label = {},
                                                                   std::string output = "bundle");

} // namespace simaai::neat::graph::nodes

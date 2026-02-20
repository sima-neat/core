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

enum class JoinKeyPolicy {
  StreamFrame = 0,
  StreamPts,
};

struct JoinBundleOptions {
  std::vector<std::string> inputs;
  std::unordered_set<std::string> required; // default: all inputs
  JoinKeyPolicy key_policy = JoinKeyPolicy::StreamFrame;
  bool emit_partial = false;
  std::size_t max_pending_keys = 4096;
  int timeout_ms = 0; // 0 => no timeout eviction
};

class JoinBundle final : public simaai::neat::graph::StageExecutor {
public:
  explicit JoinBundle(JoinBundleOptions opt);

  void set_ports(const StagePorts& ports) override;
  void on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) override;
  void on_tick(std::int64_t now_ns, std::vector<StageOutMsg>& out) override;

private:
  struct Pending {
    std::unordered_map<PortId, Sample> samples;
    std::int64_t last_seen_ns = 0;
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

// Convenience: wrap JoinBundle in a StageNode.
std::shared_ptr<simaai::neat::graph::Node> JoinBundleNode(std::vector<std::string> inputs,
                                                          std::string label = {},
                                                          std::string output = "bundle",
                                                          JoinBundleOptions opt = {});

} // namespace simaai::neat::graph::nodes

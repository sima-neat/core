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

enum class StreamDropPolicy {
  DropOldest = 0,
  DropNewest,
};

struct StreamSchedulerOptions {
  std::size_t per_stream_queue = 2;
  StreamDropPolicy drop_policy = StreamDropPolicy::DropOldest;
  int max_batch = 1; // scheduling-only batch (emit up to N per input)
};

class StreamScheduler final : public simaai::neat::graph::StageExecutor {
public:
  explicit StreamScheduler(StreamSchedulerOptions opt);

  void set_ports(const StagePorts& ports) override;
  void on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) override;

private:
  void ensure_stream_(const std::string& stream_id);
  bool emit_one_(std::vector<StageOutMsg>& out);

  StreamSchedulerOptions opt_;
  std::unordered_map<std::string, std::deque<Sample>> queues_;
  std::deque<std::string> rr_order_;
  std::unordered_set<std::string> active_;
  PortId out_port_ = kInvalidPort;
};

// Convenience: wrap StreamScheduler in a StageNode.
std::shared_ptr<simaai::neat::graph::Node> StreamSchedulerNode(StreamSchedulerOptions opt = {},
                                                               std::string label = {},
                                                               std::string input = "in",
                                                               std::string output = "out");

} // namespace simaai::neat::graph::nodes

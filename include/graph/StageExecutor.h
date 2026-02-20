/**
 * @file
 * @ingroup graph
 * @brief Actor-like stage executor interface (multi-port).
 */
#pragma once

#include "graph/GraphTypes.h"
#include "pipeline/SessionOptions.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace simaai::neat::graph {

struct StageMsg {
  PortId in_port = kInvalidPort;
  Sample sample;
};

struct StageOutMsg {
  PortId out_port = kInvalidPort;
  Sample sample;
};

struct StagePorts {
  std::unordered_map<std::string, PortId> in;
  std::unordered_map<std::string, PortId> out;

  PortId in_port(const std::string& name) const {
    auto it = in.find(name);
    return it == in.end() ? kInvalidPort : it->second;
  }

  PortId out_port(const std::string& name) const {
    auto it = out.find(name);
    return it == out.end() ? kInvalidPort : it->second;
  }

  PortId only_input() const {
    return in.size() == 1 ? in.begin()->second : kInvalidPort;
  }

  PortId only_output() const {
    return out.size() == 1 ? out.begin()->second : kInvalidPort;
  }
};

class StageExecutor {
public:
  virtual ~StageExecutor() = default;

  // Optional: provide port IDs for routing (StageExecutor can cache).
  virtual void set_ports(const StagePorts& /*ports*/) {}

  virtual void start() {}
  virtual void stop() {}

  // Called when an input arrives on any port.
  virtual void on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) = 0;

  // Optional periodic tick for timeout-based stages.
  virtual void on_tick(std::int64_t /*now_ns*/, std::vector<StageOutMsg>& /*out*/) {}
};

} // namespace simaai::neat::graph

/**
 * @file
 * @ingroup graph
 * @brief Core types for hybrid graph orchestration (ports, edges, backends).
 */
#pragma once

#include "builder/OutputSpec.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace simaai::neat::graph {

enum class Backend {
  Pipeline,
  Stage,
};

using NodeId = std::size_t;
using PortId = std::uint32_t;

static constexpr NodeId kInvalidNode = static_cast<NodeId>(-1);
static constexpr PortId kInvalidPort = static_cast<PortId>(-1);

struct PortDesc {
  std::string name;
  OutputSpec spec;
  bool optional = false;
  int max_in_edges = 1; // 0 => unlimited
};

struct Edge {
  NodeId from = kInvalidNode;
  PortId from_port = kInvalidPort;
  NodeId to = kInvalidNode;
  PortId to_port = kInvalidPort;
};

} // namespace simaai::neat::graph

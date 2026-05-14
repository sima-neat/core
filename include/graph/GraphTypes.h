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

/// Backend a graph node executes on (legacy GStreamer pipeline vs hybrid Stage runtime).
enum class Backend {
  Pipeline, ///< Node runs as a GStreamer element inside the pipeline backend.
  Stage,    ///< Node runs as a Stage in the hybrid graph runtime.
};

/// Strongly-typed graph node identifier.
using NodeId = std::size_t;
/// Strongly-typed graph port identifier (per-node).
using PortId = std::uint32_t;

/// Sentinel value for an unset/invalid `NodeId`.
static constexpr NodeId kInvalidNode = static_cast<NodeId>(-1);
/// Sentinel value for an unset/invalid `PortId`.
static constexpr PortId kInvalidPort = static_cast<PortId>(-1);

/// Description of one input or output port on a graph node.
struct PortDesc {
  std::string name;      ///< Port name (unique within its node).
  OutputSpec spec;       ///< Tensor/data contract carried on this port.
  bool optional = false; ///< If true, the port may be left unconnected.
  int max_in_edges = 1;  ///< Maximum incoming edges allowed; `0` means unlimited.
};

/// Directed connection from one node's output port to another node's input port.
struct Edge {
  NodeId from = kInvalidNode;      ///< Source node ID.
  PortId from_port = kInvalidPort; ///< Source port index on the `from` node.
  NodeId to = kInvalidNode;        ///< Destination node ID.
  PortId to_port = kInvalidPort;   ///< Destination port index on the `to` node.
};

} // namespace simaai::neat::graph

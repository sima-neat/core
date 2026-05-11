/**
 * @file
 * @ingroup graph
 * @brief Lightweight DSL helpers for wiring graphs.
 */
#pragma once

#include "graph/Graph.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::graph::dsl {

/**
 * @brief Handle to a specific port on a node within a runtime `Graph`, used by the DSL.
 *
 * Holds a non-owning pointer to the `Graph`, the `NodeId` of the node, the `PortId` of the
 * port, and whether the port is an output (otherwise it is an input). Produced by
 * `NodeRef::out()` / `NodeRef::in()` helpers and consumed by `>>` and `connect_ports`.
 *
 * @see NodeRef
 * @ingroup graph
 */
struct PortRef {
  Graph* g = nullptr;            ///< Non-owning pointer to the host graph.
  NodeId node = kInvalidNode;    ///< Node this port belongs to.
  PortId port = kInvalidPort;    ///< Interned port id within the graph.
  bool is_output = true;         ///< True for output ports, false for inputs.
};

/**
 * @brief Handle to a node within a runtime `Graph`, used by the fluent DSL.
 *
 * Wraps a non-owning `Graph*` and a `NodeId`. Implicitly converts to `NodeId` and exposes
 * `in()`/`out()`/`operator[]` to obtain `PortRef` handles for connection.
 *
 * @see PortRef
 * @ingroup graph
 */
struct NodeRef {
  Graph* g = nullptr;          ///< Non-owning pointer to the host graph.
  NodeId id = kInvalidNode;    ///< Underlying node id.

  /// Implicit conversion to the underlying `NodeId`.
  operator NodeId() const {
    return id;
  }

  /// Get the node's sole output port; throws if the node has multiple outputs.
  PortRef out() const;
  /// Get the named output port.
  PortRef out(const std::string& name) const;
  /// Get the node's sole input port; throws if the node has multiple inputs.
  PortRef in() const;
  /// Get the named input port.
  PortRef in(const std::string& name) const;
  /// Resolve a port by name; picks input or output unambiguously.
  PortRef operator[](const std::string& name) const;
  /// `operator[]` overload accepting a C-string for convenience.
  PortRef operator[](const char* name) const {
    return (*this)[std::string(name)];
  }
};

/// Build a `NodeRef` for an existing node id in the given graph.
inline NodeRef ref(Graph& g, NodeId id) {
  return NodeRef{&g, id};
}

/// Add a node to the graph and return a `NodeRef` to the newly inserted node.
inline NodeRef add(Graph& g, Graph::NodePtr node) {
  return NodeRef{&g, g.add(std::move(node))};
}

/// Returns true iff `ports` contains a `PortDesc` whose name matches `name`.
inline bool has_port(const std::vector<PortDesc>& ports, const std::string& name) {
  for (const auto& p : ports) {
    if (p.name == name)
      return true;
  }
  return false;
}

/// Throws if `g` is null; used to validate DSL preconditions.
inline void ensure_graph(const Graph* g, const char* what) {
  if (!g)
    throw std::runtime_error(std::string("GraphDsl: null graph in ") + what);
}

/// Resolve a node handle from a graph + id, throwing on null graph.
inline const std::shared_ptr<Node>& get_node(const Graph* g, NodeId id, const char* what) {
  if (!g)
    throw std::runtime_error(std::string("GraphDsl: null graph in ") + what);
  return g->node(id);
}

inline PortRef NodeRef::out() const {
  const auto& node = get_node(g, id, "out()");
  const auto ports = node->output_ports();
  if (ports.size() != 1) {
    throw std::runtime_error("GraphDsl: out() requires exactly one output port");
  }
  return out(ports.front().name);
}

inline PortRef NodeRef::out(const std::string& name) const {
  const auto& node = get_node(g, id, "out(name)");
  const auto ports = node->output_ports();
  if (!has_port(ports, name)) {
    throw std::runtime_error("GraphDsl: unknown output port: " + name);
  }
  return PortRef{g, id, g->intern_port(name), true};
}

inline PortRef NodeRef::in() const {
  const auto& node = get_node(g, id, "in()");
  const auto ports = node->input_ports();
  if (ports.size() != 1) {
    throw std::runtime_error("GraphDsl: in() requires exactly one input port");
  }
  return in(ports.front().name);
}

inline PortRef NodeRef::in(const std::string& name) const {
  const auto& node = get_node(g, id, "in(name)");
  const auto ports = node->input_ports();
  if (!has_port(ports, name)) {
    throw std::runtime_error("GraphDsl: unknown input port: " + name);
  }
  return PortRef{g, id, g->intern_port(name), false};
}

inline PortRef NodeRef::operator[](const std::string& name) const {
  const auto& node = get_node(g, id, "operator[]");
  const auto in_ports = node->input_ports();
  const auto out_ports = node->output_ports();
  const bool has_in = has_port(in_ports, name);
  const bool has_out = has_port(out_ports, name);

  if (has_out && !has_in)
    return out(name);
  if (has_in && !has_out)
    return in(name);

  if (!has_in && !has_out) {
    throw std::runtime_error("GraphDsl: unknown port: " + name);
  }
  throw std::runtime_error("GraphDsl: ambiguous port name (use .in or .out): " + name);
}

/// Connect an output `PortRef` to an input `PortRef`. Throws on type/graph mismatches.
inline void connect_ports(const PortRef& from, const PortRef& to) {
  ensure_graph(from.g, "connect_ports(from)");
  ensure_graph(to.g, "connect_ports(to)");
  if (from.g != to.g) {
    throw std::runtime_error("GraphDsl: cannot connect ports from different graphs");
  }
  if (!from.is_output) {
    throw std::runtime_error("GraphDsl: left side is not an output port");
  }
  if (to.is_output) {
    throw std::runtime_error("GraphDsl: right side is not an input port");
  }
  from.g->connect(from.node, to.node, from.g->port_name(from.port), to.g->port_name(to.port));
}

/// DSL connector: `a >> b` connects `a`'s sole output to `b`'s sole input and returns `b`.
inline NodeRef operator>>(const NodeRef& from, const NodeRef& to) {
  connect_ports(from.out(), to.in());
  return to;
}

/// DSL connector: connect a specific output port to `to`'s sole input.
inline NodeRef operator>>(const PortRef& from, const NodeRef& to) {
  connect_ports(from, to.in());
  return to;
}

/// DSL connector: connect `from`'s sole output to a specific input port.
inline NodeRef operator>>(const NodeRef& from, const PortRef& to) {
  connect_ports(from.out(), to);
  return NodeRef{to.g, to.node};
}

/// DSL connector: connect a specific output port to a specific input port.
inline NodeRef operator>>(const PortRef& from, const PortRef& to) {
  connect_ports(from, to);
  return NodeRef{to.g, to.node};
}

} // namespace simaai::neat::graph::dsl

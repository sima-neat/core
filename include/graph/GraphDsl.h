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

struct PortRef {
  Graph* g = nullptr;
  NodeId node = kInvalidNode;
  PortId port = kInvalidPort;
  bool is_output = true;
};

struct NodeRef {
  Graph* g = nullptr;
  NodeId id = kInvalidNode;

  operator NodeId() const {
    return id;
  }

  PortRef out() const;
  PortRef out(const std::string& name) const;
  PortRef in() const;
  PortRef in(const std::string& name) const;
  PortRef operator[](const std::string& name) const;
  PortRef operator[](const char* name) const {
    return (*this)[std::string(name)];
  }
};

inline NodeRef ref(Graph& g, NodeId id) {
  return NodeRef{&g, id};
}

inline NodeRef add(Graph& g, Graph::NodePtr node) {
  return NodeRef{&g, g.add(std::move(node))};
}

inline bool has_port(const std::vector<PortDesc>& ports, const std::string& name) {
  for (const auto& p : ports) {
    if (p.name == name)
      return true;
  }
  return false;
}

inline void ensure_graph(const Graph* g, const char* what) {
  if (!g)
    throw std::runtime_error(std::string("GraphDsl: null graph in ") + what);
}

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

inline NodeRef operator>>(const NodeRef& from, const NodeRef& to) {
  connect_ports(from.out(), to.in());
  return to;
}

inline NodeRef operator>>(const PortRef& from, const NodeRef& to) {
  connect_ports(from, to.in());
  return to;
}

inline NodeRef operator>>(const NodeRef& from, const PortRef& to) {
  connect_ports(from.out(), to);
  return NodeRef{to.g, to.node};
}

inline NodeRef operator>>(const PortRef& from, const PortRef& to) {
  connect_ports(from, to);
  return NodeRef{to.g, to.node};
}

} // namespace simaai::neat::graph::dsl

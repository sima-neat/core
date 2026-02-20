/**
 * @file
 * @ingroup graph
 * @brief Hybrid graph DAG with port interning (STL-only).
 */
#pragma once

#include "graph/GraphTypes.h"
#include "graph/Node.h"

#include <cstddef>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simaai::neat::graph {

/**
 * @brief Directed acyclic graph of hybrid nodes with named ports.
 */
class Graph final {
public:
  using NodePtr = std::shared_ptr<Node>;

  Graph() = default;

  // Node management
  NodeId add(NodePtr node) {
    if (!node) {
      throw std::invalid_argument("graph::Graph::add: node is null");
    }
    const NodeId id = nodes_.size();
    nodes_.push_back(std::move(node));
    out_edges_.emplace_back();
    in_edges_.emplace_back();
    return id;
  }

  const NodePtr& node(NodeId id) const {
    validate_id_(id, "graph::Graph::node");
    return nodes_[id];
  }

  std::size_t node_count() const noexcept {
    return nodes_.size();
  }
  bool empty() const noexcept {
    return nodes_.empty();
  }

  // Port interning
  PortId intern_port(std::string_view name) {
    const std::string key(name);
    if (key.empty()) {
      throw std::invalid_argument("graph::Graph::intern_port: empty port name");
    }
    auto it = port_ids_.find(key);
    if (it != port_ids_.end())
      return it->second;
    const PortId id = static_cast<PortId>(port_names_.size());
    port_names_.push_back(key);
    port_ids_.emplace(port_names_.back(), id);
    return id;
  }

  const std::string& port_name(PortId id) const {
    if (id >= port_names_.size()) {
      throw std::out_of_range("graph::Graph::port_name: invalid port id");
    }
    return port_names_[id];
  }

  std::size_t port_count() const noexcept {
    return port_names_.size();
  }

  const std::vector<std::string>& port_names() const noexcept {
    return port_names_;
  }

  // Edge management
  void connect(NodeId from, NodeId to, const std::string& from_port = "out",
               const std::string& to_port = "in") {
    validate_id_(from, "graph::Graph::connect(from)");
    validate_id_(to, "graph::Graph::connect(to)");
    if (from == to) {
      throw std::invalid_argument("graph::Graph::connect: self-loop is not allowed");
    }
    const PortId f = intern_port(from_port);
    const PortId t = intern_port(to_port);

    if (has_edge_(from, to, f, t))
      return;

    edges_.push_back({from, f, to, t});
    const std::size_t idx = edges_.size() - 1;
    out_edges_[from].push_back(idx);
    in_edges_[to].push_back(idx);
  }

  const std::vector<Edge>& edges() const noexcept {
    return edges_;
  }

  const Edge& edge(std::size_t idx) const {
    if (idx >= edges_.size()) {
      throw std::out_of_range("graph::Graph::edge: invalid index");
    }
    return edges_[idx];
  }

  const std::vector<std::size_t>& out_edges(NodeId id) const {
    validate_id_(id, "graph::Graph::out_edges");
    return out_edges_[id];
  }

  const std::vector<std::size_t>& in_edges(NodeId id) const {
    validate_id_(id, "graph::Graph::in_edges");
    return in_edges_[id];
  }

  std::size_t out_degree(NodeId id) const {
    validate_id_(id, "graph::Graph::out_degree");
    return out_edges_[id].size();
  }

  std::size_t in_degree(NodeId id) const {
    validate_id_(id, "graph::Graph::in_degree");
    return in_edges_[id].size();
  }

  // Topological order
  std::vector<NodeId> topo_order() const {
    const std::size_t n = nodes_.size();
    std::vector<std::size_t> indeg(n, 0);
    for (NodeId i = 0; i < n; ++i)
      indeg[i] = in_edges_[i].size();

    std::queue<NodeId> q;
    for (NodeId i = 0; i < n; ++i) {
      if (indeg[i] == 0)
        q.push(i);
    }

    std::vector<NodeId> order;
    order.reserve(n);

    while (!q.empty()) {
      NodeId u = q.front();
      q.pop();
      order.push_back(u);

      for (const std::size_t eidx : out_edges_[u]) {
        const Edge& e = edges_[eidx];
        if (indeg[e.to] == 0) {
          continue;
        }
        indeg[e.to]--;
        if (indeg[e.to] == 0)
          q.push(e.to);
      }
    }

    if (order.size() != n) {
      throw std::runtime_error("graph::Graph::topo_order: cycle detected (graph is not a DAG)");
    }
    return order;
  }

  bool is_dag() const {
    try {
      (void)topo_order();
      return true;
    } catch (...) {
      return false;
    }
  }

private:
  void validate_id_(NodeId id, const char* where) const {
    if (id >= nodes_.size()) {
      throw std::out_of_range(std::string(where) + ": invalid NodeId");
    }
  }

  bool has_edge_(NodeId from, NodeId to, PortId from_port, PortId to_port) const {
    for (const std::size_t eidx : out_edges_[from]) {
      const Edge& e = edges_[eidx];
      if (e.from == from && e.to == to && e.from_port == from_port && e.to_port == to_port) {
        return true;
      }
    }
    return false;
  }

  std::vector<NodePtr> nodes_;
  std::vector<Edge> edges_;
  std::vector<std::vector<std::size_t>> out_edges_;
  std::vector<std::vector<std::size_t>> in_edges_;

  std::vector<std::string> port_names_;
  std::unordered_map<std::string, PortId> port_ids_;
};

} // namespace simaai::neat::graph

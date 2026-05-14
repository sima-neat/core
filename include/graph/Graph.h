/**
 * @file
 * @ingroup graph
 * @brief Runtime `graph::Graph` — actor-style DAG with named ports for the runtime.
 *
 * This is the framework's **runtime** graph — a different type from the builder-side
 * `simaai::neat::Graph`. The runtime graph carries **named ports** (e.g., `"out"`,
 * `"in"`), interned to compact `PortId`s, so a node can have multiple distinct inputs
 * and outputs that the executor can route between. It's the substrate the actor-style
 * runtime uses for stage scheduling and mailbox-based message passing.
 *
 * Use the **builder** `Graph` (in [builder/Graph.h](../builder/Graph.h)) when you're
 * authoring a pipeline shape; use this **runtime** `Graph` only if you're building a
 * runtime-graph-driven Session via `GraphSession`.
 *
 * @see GraphSession, GraphRun, StageExecutor
 * @see "The two graph systems" (§0.14 / §73 of the design deep dive)
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
 *
 * Edges carry **interned port identifiers** at both endpoints (`from_port` and `to_port`),
 * letting a node distinguish multiple inputs (e.g., `"left"` and `"right"`) and multiple
 * outputs (e.g., `"main"` and `"telemetry"`) without inventing new types. Port names are
 * interned to compact `PortId`s so the executor doesn't pay string-compare costs at
 * dispatch time.
 *
 * Construct one of these via `GraphSession`; consume via `GraphRun` / `StageExecutor`.
 *
 * @see GraphSession, GraphRun, StageExecutor
 * @ingroup graph
 */
class Graph final {
public:
  /// Shared-pointer type used to hold a runtime `Node`.
  using NodePtr = std::shared_ptr<Node>;

  /// Construct an empty graph.
  Graph() = default;

  // ---------------------------
  // Node management
  // ---------------------------

  /**
   * @brief Add a node to the graph.
   * @return Stable `NodeId` for the inserted node.
   * @throws std::invalid_argument if `node` is null.
   */
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

  /// Look up a node by id. Throws `std::out_of_range` for an invalid id.
  const NodePtr& node(NodeId id) const {
    validate_id_(id, "graph::Graph::node");
    return nodes_[id];
  }

  /// Number of nodes in the graph.
  std::size_t node_count() const noexcept {
    return nodes_.size();
  }
  /// True iff the graph holds no nodes.
  bool empty() const noexcept {
    return nodes_.empty();
  }

  // ---------------------------
  // Port interning
  // ---------------------------

  /**
   * @brief Intern a port name and return its compact `PortId`.
   *
   * Repeated calls with the same name return the same id. Ids are dense and stable for
   * the lifetime of the graph.
   *
   * @throws std::invalid_argument if `name` is empty.
   */
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

  /// Return the human-readable port name for a `PortId`. Throws on invalid id.
  const std::string& port_name(PortId id) const {
    if (id >= port_names_.size()) {
      throw std::out_of_range("graph::Graph::port_name: invalid port id");
    }
    return port_names_[id];
  }

  /// Number of unique port names interned in this graph.
  std::size_t port_count() const noexcept {
    return port_names_.size();
  }

  /// All interned port names, indexable by `PortId`.
  const std::vector<std::string>& port_names() const noexcept {
    return port_names_;
  }

  // ---------------------------
  // Edge management
  // ---------------------------

  /**
   * @brief Connect `from`'s named output port to `to`'s named input port.
   *
   * Defaults connect `from:"out"` → `to:"in"`. Self-loops are rejected; duplicate edges
   * (same endpoints + same port pair) are silently ignored.
   *
   * @param from      Source node id.
   * @param to        Destination node id.
   * @param from_port Output port name on `from` (default `"out"`).
   * @param to_port   Input port name on `to` (default `"in"`).
   * @throws std::invalid_argument on self-loop or invalid id.
   */
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

  /// All edges in insertion order.
  const std::vector<Edge>& edges() const noexcept {
    return edges_;
  }

  /// Look up an edge by index. Throws on invalid index.
  const Edge& edge(std::size_t idx) const {
    if (idx >= edges_.size()) {
      throw std::out_of_range("graph::Graph::edge: invalid index");
    }
    return edges_[idx];
  }

  /// Indices into `edges()` for outgoing edges of `id`.
  const std::vector<std::size_t>& out_edges(NodeId id) const {
    validate_id_(id, "graph::Graph::out_edges");
    return out_edges_[id];
  }

  /// Indices into `edges()` for incoming edges of `id`.
  const std::vector<std::size_t>& in_edges(NodeId id) const {
    validate_id_(id, "graph::Graph::in_edges");
    return in_edges_[id];
  }

  /// Number of outgoing edges from `id`.
  std::size_t out_degree(NodeId id) const {
    validate_id_(id, "graph::Graph::out_degree");
    return out_edges_[id].size();
  }

  /// Number of incoming edges to `id`.
  std::size_t in_degree(NodeId id) const {
    validate_id_(id, "graph::Graph::in_degree");
    return in_edges_[id].size();
  }

  // ---------------------------
  // Topological order
  // ---------------------------

  /**
   * @brief Topologically sort the graph.
   * @throws std::runtime_error if the graph has a cycle.
   */
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

  /// True iff the graph has no cycles.
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

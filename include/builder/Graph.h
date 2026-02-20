/**
 * @file
 * @ingroup builder
 * @brief STL-only DAG representation for Node pipelines.
 */
// include/builder/Graph.h
#pragma once

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "builder/Node.h"
#include "builder/NodeGroup.h"

namespace simaai::neat {

/**
 * @brief Directed acyclic graph of Nodes (STL-only).
 *
 * This is the "graph face" of the builder layer. It deliberately does NOT know
 * anything about GStreamer. It only manages Node objects + dependencies.
 *
 * Most pipelines are linear chains; Graph exists for:
 *  - expressing fan-in / fan-out structure in a type-safe way
 *  - doing validation (cycle detection, connectivity checks)
 *  - optionally linearizing to a NodeGroup (topological order or strict chain)
 *
 * Session consumes a linear list today; Graph can produce that list via:
 *  - to_node_group_topo(): topological order (DAG)
 *  - to_node_group_chain(): requires a strict single chain (each internal node has in=1,out=1)
 */
class Graph final {
public:
  using NodePtr = std::shared_ptr<Node>;
  using NodeId = std::size_t;

  struct Edge {
    NodeId from{};
    NodeId to{};
  };

  static constexpr NodeId kInvalid = static_cast<NodeId>(-1);

  Graph() = default;

  // ---------------------------
  // Node management
  // ---------------------------

  /// Add a node to the graph. Returns its NodeId.
  NodeId add(NodePtr node) {
    if (!node) {
      throw std::invalid_argument("Graph::add: node is null");
    }
    const NodeId id = nodes_.size();
    nodes_.push_back(std::move(node));
    out_.emplace_back();
    in_.emplace_back();
    return id;
  }

  /// Add all nodes from a NodeGroup, unconnected (no edges). Returns first id (or kInvalid if
  /// empty).
  NodeId add_group(const NodeGroup& group) {
    const auto& gnodes = group.nodes();
    if (gnodes.empty())
      return kInvalid;
    const NodeId first = nodes_.size();
    for (const auto& n : gnodes)
      add(n);
    return first;
  }

  /// Convenience: add nodes from group and connect them as a chain. Returns first id (or kInvalid
  /// if empty).
  NodeId add_group_as_chain(const NodeGroup& group) {
    const auto& gnodes = group.nodes();
    if (gnodes.empty())
      return kInvalid;
    const NodeId first = nodes_.size();
    NodeId prev = kInvalid;
    for (const auto& n : gnodes) {
      NodeId cur = add(n);
      if (prev != kInvalid)
        add_edge(prev, cur);
      prev = cur;
    }
    return first;
  }

  const NodePtr& node(NodeId id) const {
    validate_id_(id, "Graph::node");
    return nodes_[id];
  }

  std::size_t node_count() const noexcept {
    return nodes_.size();
  }
  bool empty() const noexcept {
    return nodes_.empty();
  }

  // ---------------------------
  // Edge management
  // ---------------------------

  /// Add a directed edge `from -> to`.
  void add_edge(NodeId from, NodeId to) {
    validate_id_(from, "Graph::add_edge(from)");
    validate_id_(to, "Graph::add_edge(to)");
    if (from == to) {
      throw std::invalid_argument("Graph::add_edge: self-loop is not allowed");
    }
    // Prevent duplicate edges (keep predictable adjacency).
    if (has_edge_({from, to}))
      return;

    out_[from].push_back(to);
    in_[to].push_back(from);
    edges_.push_back({from, to});
  }

  /// Convenience: add edges to form a chain id0->id1->...->idN.
  void add_chain(std::initializer_list<NodeId> ids) {
    if (ids.size() < 2)
      return;
    auto it = ids.begin();
    NodeId prev = *it++;
    for (; it != ids.end(); ++it) {
      add_edge(prev, *it);
      prev = *it;
    }
  }

  /// Return all edges (in insertion order).
  const std::vector<Edge>& edges() const noexcept {
    return edges_;
  }

  /// Outgoing neighbors for a node.
  const std::vector<NodeId>& out_neighbors(NodeId id) const {
    validate_id_(id, "Graph::out_neighbors");
    return out_[id];
  }

  /// Incoming neighbors for a node.
  const std::vector<NodeId>& in_neighbors(NodeId id) const {
    validate_id_(id, "Graph::in_neighbors");
    return in_[id];
  }

  std::size_t out_degree(NodeId id) const {
    validate_id_(id, "Graph::out_degree");
    return out_[id].size();
  }

  std::size_t in_degree(NodeId id) const {
    validate_id_(id, "Graph::in_degree");
    return in_[id].size();
  }

  // ---------------------------
  // Validation / ordering
  // ---------------------------

  /**
   * @brief Topologically sort the graph.
   * @throws std::runtime_error if the graph has a cycle.
   */
  std::vector<NodeId> topo_order() const {
    const std::size_t n = nodes_.size();
    std::vector<std::size_t> indeg(n, 0);
    for (NodeId i = 0; i < n; ++i)
      indeg[i] = in_[i].size();

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

      for (NodeId v : out_[u]) {
        if (indeg[v] == 0) {
          // Shouldn't happen in a correct indegree tracking, but keep defensive.
          continue;
        }
        indeg[v]--;
        if (indeg[v] == 0)
          q.push(v);
      }
    }

    if (order.size() != n) {
      throw std::runtime_error("Graph::topo_order: cycle detected (graph is not a DAG)");
    }
    return order;
  }

  /// True if the graph has no cycles.
  bool is_dag() const {
    try {
      (void)topo_order();
      return true;
    } catch (...) {
      return false;
    }
  }

  /**
   * @brief Return a NodeGroup in topological order.
   * @throws std::runtime_error if the graph has a cycle.
   */
  NodeGroup to_node_group_topo() const {
    const auto order = topo_order();
    std::vector<NodePtr> out;
    out.reserve(order.size());
    for (NodeId id : order)
      out.push_back(nodes_[id]);
    return NodeGroup(std::move(out));
  }

  /**
   * @brief Return a NodeGroup if (and only if) this DAG is a strict single chain.
   *
   * Requirements:
   *  - If n==0: returns empty NodeGroup.
   *  - If n==1: returns that node.
   *  - Otherwise:
   *      * exactly one start node (in_degree==0) and one end node (out_degree==0)
   *      * all other nodes have in_degree==1 and out_degree==1
   *      * edge count is exactly n-1
   *      * graph is a DAG (no cycles)
   *
   * This is the safe conversion for current Session (linear list semantics).
   */
  NodeGroup to_node_group_chain() const {
    const std::size_t n = nodes_.size();
    if (n == 0)
      return NodeGroup{};
    if (n == 1)
      return NodeGroup(std::vector<NodePtr>{nodes_[0]});

    if (edges_.size() != n - 1) {
      throw std::runtime_error("Graph::to_node_group_chain: expected edge_count == node_count - 1");
    }

    // Identify start/end, validate degrees.
    NodeId start = kInvalid;
    NodeId end = kInvalid;

    for (NodeId i = 0; i < n; ++i) {
      const std::size_t indeg = in_[i].size();
      const std::size_t outdeg = out_[i].size();

      if (indeg == 0) {
        if (start != kInvalid) {
          throw std::runtime_error(
              "Graph::to_node_group_chain: multiple start nodes (in_degree==0)");
        }
        start = i;
      } else if (indeg != 1) {
        throw std::runtime_error(
            "Graph::to_node_group_chain: node has in_degree != 1 (not a chain)");
      }

      if (outdeg == 0) {
        if (end != kInvalid) {
          throw std::runtime_error(
              "Graph::to_node_group_chain: multiple end nodes (out_degree==0)");
        }
        end = i;
      } else if (outdeg != 1) {
        throw std::runtime_error(
            "Graph::to_node_group_chain: node has out_degree != 1 (not a chain)");
      }
    }

    if (start == kInvalid || end == kInvalid) {
      throw std::runtime_error(
          "Graph::to_node_group_chain: expected exactly one start and one end node");
    }

    // Ensure DAG and produce a linear walk from start following unique outgoing edge.
    // (Also implicitly checks connectivity; if disconnected, we won't visit all nodes.)
    (void)topo_order(); // throws on cycle

    std::vector<NodePtr> chain;
    chain.reserve(n);

    std::vector<bool> seen(n, false);
    NodeId cur = start;
    for (std::size_t step = 0; step < n; ++step) {
      if (cur == kInvalid)
        break;
      if (seen[cur]) {
        throw std::runtime_error(
            "Graph::to_node_group_chain: cycle encountered while walking chain");
      }
      seen[cur] = true;
      chain.push_back(nodes_[cur]);

      if (out_[cur].empty()) {
        cur = kInvalid;
      } else {
        cur = out_[cur][0]; // unique by validation
      }
    }

    if (chain.size() != n) {
      throw std::runtime_error("Graph::to_node_group_chain: graph is not a single connected chain");
    }
    return NodeGroup(std::move(chain));
  }

private:
  void validate_id_(NodeId id, const char* where) const {
    if (id >= nodes_.size()) {
      throw std::out_of_range(std::string(where) + ": invalid NodeId");
    }
  }

  bool has_edge_(const Edge& edge) const {
    // out_[from] tends to be tiny; linear scan keeps STL-only + simple.
    for (NodeId v : out_[edge.from]) {
      if (v == edge.to)
        return true;
    }
    return false;
  }

  std::vector<NodePtr> nodes_;
  std::vector<std::vector<NodeId>> out_;
  std::vector<std::vector<NodeId>> in_;
  std::vector<Edge> edges_;
};

} // namespace simaai::neat

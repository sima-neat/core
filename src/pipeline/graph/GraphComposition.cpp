/**
 * @file GraphComposition.cpp
 * @brief Out-of-line implementation of private Graph composition helpers.
 *
 * Keep non-trivial `Graph::CompositionGraph` methods in one translation unit.
 * These helpers used to live inline in GraphDetail.h, which emitted weak COMDAT
 * bodies from any translation unit that touched them. Centralizing them here
 * gives the linker exactly one implementation body and avoids stale inline
 * endpoint-routing behavior after header-only edits.
 */

#include "GraphDetail.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

Graph::CompositionGraph::VertexId Graph::CompositionGraph::append_vertex(NodePtr node) {
  if (!vertices.empty() && tail == kInvalid) {
    throw std::runtime_error("Graph::add after endpoint branching is ambiguous; use connect()");
  }
  const VertexId id = vertices.size();
  vertices.push_back(std::move(node));
  if (tail != kInvalid) {
    edges.push_back(
        CompositionEdge{.from = tail, .to = id, .kind = CompositionEdgeKind::ImplicitLinear});
  }
  tail = id;
  return id;
}

std::pair<Graph::CompositionGraph::VertexId, Graph::CompositionGraph::VertexId>
Graph::CompositionGraph::append_linear_fragment_copy(const std::vector<NodePtr>& nodes) {
  const VertexId start = vertices.size();
  vertices.insert(vertices.end(), nodes.begin(), nodes.end());
  const VertexId end = vertices.size();
  for (VertexId id = start + 1U; id < end; ++id) {
    edges.push_back(
        CompositionEdge{.from = id - 1U, .to = id, .kind = CompositionEdgeKind::ImplicitLinear});
  }
  if (end > start) {
    tail = end - 1U;
  }
  return {start, end};
}

void Graph::CompositionGraph::recompute_unique_tail() noexcept {
  if (vertices.empty()) {
    tail = kInvalid;
    return;
  }
  std::vector<bool> has_out(vertices.size(), false);
  for (const auto& edge : edges) {
    if (edge.from < has_out.size()) {
      has_out[edge.from] = true;
    }
  }
  VertexId candidate = kInvalid;
  for (VertexId id = 0; id < has_out.size(); ++id) {
    if (has_out[id]) {
      continue;
    }
    if (candidate != kInvalid) {
      tail = kInvalid;
      return;
    }
    candidate = id;
  }
  tail = candidate;
}

void Graph::CompositionGraph::connect_runtime_port(VertexId from, VertexId to,
                                                   std::string from_port, std::string to_port) {
  if (from >= vertices.size() || to >= vertices.size()) {
    throw std::runtime_error("Graph::connect: internal vertex id out of range");
  }
  if (from_port.empty()) {
    throw std::runtime_error("Graph::connect: port name must not be empty");
  }
  if (to_port.empty()) {
    throw std::runtime_error("Graph::connect: port name must not be empty");
  }
  edges.push_back(CompositionEdge{.from = from,
                                  .to = to,
                                  .kind = CompositionEdgeKind::RuntimePort,
                                  .from_port = std::move(from_port),
                                  .to_port = std::move(to_port)});
  recompute_unique_tail();
}

void Graph::CompositionGraph::connect_endpoint(VertexId from, VertexId to,
                                               std::string from_endpoint, std::string to_endpoint) {
  if (from >= vertices.size() || to >= vertices.size()) {
    throw std::runtime_error("Graph::connect: internal vertex id out of range");
  }
  if (from_endpoint.empty() || to_endpoint.empty()) {
    throw std::runtime_error("Graph::connect: endpoint name must not be empty");
  }

  const bool destination_is_public_output =
      dynamic_cast<const Output*>(vertices[to].get()) != nullptr;
  if (!destination_is_public_output) {
    for (const auto& edge : edges) {
      if (edge.kind != CompositionEdgeKind::PublicEndpoint || !edge.endpoint.has_value()) {
        continue;
      }
      if (edge.to == to && edge.endpoint->to_endpoint == to_endpoint) {
        throw std::runtime_error("Graph::connect: destination endpoint '" + to_endpoint +
                                 "' is already connected; insert an explicit Combine graph "
                                 "when multiple sources should feed one input");
      }
    }
  }

  endpoint_mode = true;
  edges.push_back(CompositionEdge{
      .from = from,
      .to = to,
      .kind = CompositionEdgeKind::PublicEndpoint,
      .endpoint = EndpointEdgeMeta{.from_endpoint = std::move(from_endpoint),
                                   .to_endpoint = std::move(to_endpoint)},
  });
  recompute_unique_tail();
}

std::pair<Graph::CompositionGraph::VertexId, Graph::CompositionGraph::VertexId>
Graph::CompositionGraph::append_node(NodePtr node) {
  const VertexId start = vertices.size();
  append_vertex(std::move(node));
  return {start, vertices.size()};
}

bool Graph::CompositionGraph::is_linear() const noexcept {
  // Linear compatibility is deliberately strict: only the implicit edges created by add()
  // count as a linear chain. Public endpoint edges and internal runtime-port edges lower
  // through the connected graph compiler instead of being flattened here.
  if (vertices.empty()) {
    return true;
  }
  if (edges.size() + 1U != vertices.size()) {
    return false;
  }
  for (VertexId i = 0; i < edges.size(); ++i) {
    if (edges[i].from != i || edges[i].to != i + 1U ||
        edges[i].kind != CompositionEdgeKind::ImplicitLinear || !edges[i].from_port.empty() ||
        !edges[i].to_port.empty() || edges[i].endpoint.has_value()) {
      return false;
    }
  }
  return true;
}

std::vector<Graph::CompositionGraph::NodePtr>
Graph::CompositionGraph::linear_nodes_or_throw(const char* where) const {
  // Add-only compatibility path: insertion order remains the public linear naming contract.
  // Connected graphs must use the endpoint/runtime compiler so topology, named endpoints,
  // and fragment provenance stay intact.
  if (vertices.empty()) {
    return {};
  }
  if (!is_linear()) {
    throw std::runtime_error(std::string(where ? where : "Graph") +
                             ": internal composition is not linear yet");
  }
  return vertices; // Preserve nulls; existing validation catches them later.
}

} // namespace simaai::neat

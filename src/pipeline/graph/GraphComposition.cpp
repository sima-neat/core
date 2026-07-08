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

#include "nodes/io/HttpSource.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
namespace {

std::string automatic_realtime_stream_id(std::size_t from, std::size_t to,
                                         const std::string& endpoint) {
  return "edge" + std::to_string(from) + "_to_" + std::to_string(to) + "_" + endpoint;
}

bool node_is_live_source(const Node* node) {
  if (!node || node->input_role() != InputRole::Source) {
    return false;
  }
  const std::string kind = node->kind();
  if (kind == "RTSPInput" || kind == "CameraInput" || kind == "PCIeSrc") {
    return true;
  }
  if (kind == "HttpSource") {
    const auto* http = dynamic_cast<const HttpSource*>(node);
    return http && http->options().is_live;
  }
  return false;
}

bool vertex_is_live_source_context(const auto& graph, std::size_t vertex) {
  if (vertex >= graph.vertices.size()) {
    return false;
  }

  bool has_live_source = false;
  std::vector<std::size_t> stack{vertex};
  std::vector<bool> seen(graph.vertices.size(), false);
  while (!stack.empty()) {
    const std::size_t current = stack.back();
    stack.pop_back();
    if (current >= graph.vertices.size() || seen[current]) {
      continue;
    }
    seen[current] = true;

    const Node* node = graph.vertices[current].get();
    if (node && node->input_role() == InputRole::Push) {
      return false;
    }
    has_live_source = has_live_source || node_is_live_source(node);

    for (const auto& edge : graph.edges) {
      if (edge.to == current && edge.from < graph.vertices.size()) {
        stack.push_back(edge.from);
      }
    }
  }
  return has_live_source;
}

} // namespace

Graph::CompositionGraph::VertexId Graph::CompositionGraph::append_vertex(NodePtr node) {
  if (!vertices.empty() && tail == kInvalid) {
    throw std::runtime_error("Graph::add after endpoint branching is ambiguous; use connect()");
  }
  const VertexId id = vertices.size();
  vertices.emplace_back(std::move(node));
  if (tail != kInvalid) {
    edges.push_back(
        CompositionEdge{.from = tail, .to = id, .kind = CompositionEdgeKind::ImplicitLinear});
  }
  tail = id;
  return id;
}

Graph::CompositionGraph::VertexId
Graph::CompositionGraph::append_runtime_vertex(RuntimeNodePtr node) {
  if (!vertices.empty() && tail == kInvalid) {
    throw std::runtime_error("Graph::add after endpoint branching is ambiguous; use connect()");
  }
  const VertexId id = vertices.size();
  vertices.push_back(CompositionVertex::runtime(std::move(node)));
  if (tail != kInvalid) {
    edges.push_back(
        CompositionEdge{.from = tail, .to = id, .kind = CompositionEdgeKind::ImplicitLinear});
  }
  tail = id;
  return id;
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
                                                   std::string from_port, std::string to_port,
                                                   GraphLinkOptions link_options) {
  if (from >= vertices.size() || to >= vertices.size()) {
    throw std::runtime_error("Graph::connect: internal vertex id out of range");
  }
  if (from_port.empty()) {
    throw std::runtime_error("Graph::connect: port name must not be empty");
  }
  if (to_port.empty()) {
    throw std::runtime_error("Graph::connect: port name must not be empty");
  }
  std::string incoming_stream_id = link_options.stream_id;
  for (auto& edge : edges) {
    if (edge.kind != CompositionEdgeKind::RuntimePort) {
      continue;
    }
    if (edge.to == to && edge.to_port == to_port) {
      const bool existing_realtime =
          edge.link_options.policy == GraphLinkPolicy::RealtimeLatestByStream;
      const bool incoming_realtime = link_options.policy == GraphLinkPolicy::RealtimeLatestByStream;
      const bool default_live_fan_in = edge.link_options.policy == GraphLinkPolicy::Default &&
                                       link_options.policy == GraphLinkPolicy::Default &&
                                       vertex_is_live_source_context(*this, edge.from) &&
                                       vertex_is_live_source_context(*this, from);
      if (existing_realtime || incoming_realtime || default_live_fan_in) {
        edge.link_options.policy = GraphLinkPolicy::RealtimeLatestByStream;
        link_options.policy = GraphLinkPolicy::RealtimeLatestByStream;
        edge.link_options.queue_depth =
            std::max(edge.link_options.queue_depth, link_options.queue_depth);
        link_options.queue_depth = edge.link_options.queue_depth;
        if (edge.stream_id.empty()) {
          edge.stream_id = edge.link_options.stream_id.empty()
                               ? automatic_realtime_stream_id(edge.from, edge.to, edge.to_port)
                               : edge.link_options.stream_id;
        }
        if (incoming_stream_id.empty()) {
          incoming_stream_id = automatic_realtime_stream_id(from, to, to_port);
        }
      }
    }
  }
  edges.push_back(CompositionEdge{.from = from,
                                  .to = to,
                                  .kind = CompositionEdgeKind::RuntimePort,
                                  .from_port = std::move(from_port),
                                  .to_port = std::move(to_port),
                                  .link_options = link_options,
                                  .stream_id = std::move(incoming_stream_id)});
  recompute_unique_tail();
}

void Graph::CompositionGraph::connect_endpoint(VertexId from, VertexId to,
                                               std::string from_endpoint, std::string to_endpoint,
                                               GraphLinkOptions link_options) {
  if (from >= vertices.size() || to >= vertices.size()) {
    throw std::runtime_error("Graph::connect: internal vertex id out of range");
  }
  if (from_endpoint.empty() || to_endpoint.empty()) {
    throw std::runtime_error("Graph::connect: endpoint name must not be empty");
  }

  const bool destination_is_public_output =
      dynamic_cast<const Output*>(vertices[to].get()) != nullptr;
  std::string incoming_stream_id = link_options.stream_id;
  if (!destination_is_public_output) {
    for (auto& edge : edges) {
      if (edge.kind != CompositionEdgeKind::PublicEndpoint || !edge.endpoint.has_value()) {
        continue;
      }
      if (edge.to == to && edge.endpoint->to_endpoint == to_endpoint) {
        const bool existing_realtime =
            edge.link_options.policy == GraphLinkPolicy::RealtimeLatestByStream;
        const bool incoming_realtime =
            link_options.policy == GraphLinkPolicy::RealtimeLatestByStream;
        const bool default_live_fan_in = edge.link_options.policy == GraphLinkPolicy::Default &&
                                         link_options.policy == GraphLinkPolicy::Default &&
                                         vertex_is_live_source_context(*this, edge.from) &&
                                         vertex_is_live_source_context(*this, from);
        if (existing_realtime || incoming_realtime || default_live_fan_in) {
          /*
           * Multiple producers feeding one live input should use the framework
           * C++ fair-mux path by default.  This keeps source producers
           * non-blocking, preserves one latest loaned Sample per stream/edge,
           * and schedules into the consumer through RealtimeLatestLink instead
           * of asking users to insert app-local mutex/funnel code.
           */
          edge.link_options.policy = GraphLinkPolicy::RealtimeLatestByStream;
          link_options.policy = GraphLinkPolicy::RealtimeLatestByStream;
          edge.link_options.queue_depth =
              std::max(edge.link_options.queue_depth, link_options.queue_depth);
          link_options.queue_depth = edge.link_options.queue_depth;
          if (edge.stream_id.empty()) {
            edge.stream_id =
                edge.link_options.stream_id.empty()
                    ? automatic_realtime_stream_id(edge.from, edge.to, edge.endpoint->to_endpoint)
                    : edge.link_options.stream_id;
          }
          if (incoming_stream_id.empty()) {
            incoming_stream_id = automatic_realtime_stream_id(from, to, to_endpoint);
          }
          continue;
        }
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
      .link_options = link_options,
      .stream_id = std::move(incoming_stream_id),
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
  for (const auto& vertex : vertices) {
    if (vertex.kind == CompositionVertex::Kind::RuntimeNode) {
      return false;
    }
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
  return pipeline_vertices_snapshot(); // Preserve nulls; existing validation catches them later.
}

std::vector<Graph::CompositionGraph::NodePtr>
Graph::CompositionGraph::pipeline_vertices_snapshot() const {
  std::vector<NodePtr> out;
  out.reserve(vertices.size());
  for (const auto& vertex : vertices) {
    out.push_back(vertex.pipeline_node);
  }
  return out;
}

std::vector<Graph::CompositionGraph::RuntimeNodePtr>
Graph::CompositionGraph::runtime_vertices_snapshot() const {
  std::vector<RuntimeNodePtr> out;
  out.reserve(vertices.size());
  for (const auto& vertex : vertices) {
    out.push_back(vertex.runtime_node);
  }
  return out;
}

Graph::CompositionGraph::RuntimeNodePtr Graph::CompositionGraph::runtime_node(VertexId id) const {
  return id < vertices.size() ? vertices[id].runtime_node : nullptr;
}

bool Graph::CompositionGraph::has_runtime_vertices() const noexcept {
  return std::any_of(vertices.begin(), vertices.end(), [](const auto& vertex) {
    return vertex.kind == CompositionVertex::Kind::RuntimeNode && vertex.runtime_node != nullptr;
  });
}

} // namespace simaai::neat

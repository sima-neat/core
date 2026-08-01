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
#include "pipeline/graph/internal/GraphTestHooks.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
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

void validate_realtime_inflight_option(const char* name, int value) {
  if (value == 0 || value < -1) {
    throw std::runtime_error(std::string("GraphLinkOptions::") + name +
                             " must be -1 or a positive value");
  }
}

void validate_realtime_inflight_options(const GraphLinkOptions& options) {
  validate_realtime_inflight_option("max_inflight_per_stream", options.max_inflight_per_stream);
  validate_realtime_inflight_option("max_inflight_total", options.max_inflight_total);
}

bool is_realtime_stream_policy(GraphLinkPolicy policy) {
  return policy == GraphLinkPolicy::RealtimeLatestByStream;
}

void merge_realtime_link_options(GraphLinkOptions& existing, GraphLinkOptions& incoming) {
  validate_realtime_inflight_options(existing);
  validate_realtime_inflight_options(incoming);

  existing.policy = GraphLinkPolicy::RealtimeLatestByStream;
  incoming.policy = GraphLinkPolicy::RealtimeLatestByStream;
  existing.queue_depth = std::max(existing.queue_depth, incoming.queue_depth);
  incoming.queue_depth = existing.queue_depth;
  // These are admission promises made by each producer edge. Keep them intact
  // while normalizing shared fan-in behavior. Fused lowering configures each
  // branch independently; the non-fused shared link resolves the strictest
  // contract across all of its producer edges.
}

} // namespace

std::optional<Graph::CompositionGraph::VertexId>
Graph::CompositionGraph::find_pipeline_vertex(const Node* node) const noexcept {
  if (!node) {
    return std::nullopt;
  }
  const auto it = pipeline_vertex_by_identity.find(node);
  return it == pipeline_vertex_by_identity.end() ? std::nullopt
                                                 : std::optional<VertexId>(it->second);
}

void Graph::CompositionGraph::preflight_pipeline_vertices(
    std::span<const CompositionVertex> incoming, std::string_view operation) const {
  std::unordered_map<const Node*, std::size_t> incoming_indices;
  incoming_indices.reserve(incoming.size());
  for (std::size_t i = 0; i < incoming.size(); ++i) {
    const auto& vertex = incoming[i];
    if (vertex.kind != CompositionVertex::Kind::PipelineNode || !vertex.pipeline_node) {
      continue;
    }
    const Node* identity = vertex.pipeline_node.get();
    if (const auto existing = find_pipeline_vertex(identity)) {
      std::ostringstream oss;
      oss << (operation.empty() ? "Graph composition" : operation)
          << ": Node object identity already exists at destination vertex " << *existing
          << " (incoming vertex " << i << ", kind='" << vertex.pipeline_node->kind()
          << "'). Construct a separate Node instance to create another logical vertex, or reuse "
             "the existing vertex through connect().";
      throw std::runtime_error(oss.str());
    }
    const auto [it, inserted] = incoming_indices.emplace(identity, i);
    if (!inserted) {
      std::ostringstream oss;
      oss << (operation.empty() ? "Graph composition" : operation)
          << ": incoming fragment contains the same Node object at vertices " << it->second
          << " and " << i << " (kind='" << vertex.pipeline_node->kind()
          << "'). One Node object can represent only one logical vertex.";
      throw std::runtime_error(oss.str());
    }
  }
}

void Graph::CompositionGraph::preflight_pipeline_nodes(std::span<const NodePtr> incoming,
                                                       std::string_view operation) const {
  std::unordered_map<const Node*, std::size_t> incoming_indices;
  incoming_indices.reserve(incoming.size());
  for (std::size_t i = 0; i < incoming.size(); ++i) {
    const auto& node = incoming[i];
    if (!node) {
      continue;
    }
    const Node* identity = node.get();
    if (const auto existing = find_pipeline_vertex(identity)) {
      std::ostringstream oss;
      oss << (operation.empty() ? "Graph composition" : operation)
          << ": Node object identity already exists at destination vertex " << *existing
          << " (incoming vertex " << i << ", kind='" << node->kind()
          << "'). Construct a separate Node instance to create another logical vertex, or reuse "
             "the existing vertex through connect().";
      throw std::runtime_error(oss.str());
    }
    const auto [it, inserted] = incoming_indices.emplace(identity, i);
    if (!inserted) {
      std::ostringstream oss;
      oss << (operation.empty() ? "Graph composition" : operation)
          << ": incoming fragment contains the same Node object at vertices " << it->second
          << " and " << i << " (kind='" << node->kind()
          << "'). One Node object can represent only one logical vertex.";
      throw std::runtime_error(oss.str());
    }
  }
}

Graph::CompositionGraph::VertexId
Graph::CompositionGraph::append_unlinked_pipeline_vertex(NodePtr node, std::string_view operation) {
  if (node) {
    if (const auto existing = find_pipeline_vertex(node.get())) {
      throw std::runtime_error(
          std::string(operation.empty() ? "Graph composition" : operation) +
          ": Node object identity already exists at destination vertex " +
          std::to_string(*existing) + " (kind='" + node->kind() +
          "'). Construct a separate Node instance to create another logical vertex, or reuse "
          "the existing vertex through connect().");
    }
  }
  const VertexId id = vertices.size();
  const Node* identity = node.get();
  if (identity) {
    // Publish the derived-index entry first. If allocation fails, the vertex vector is untouched;
    // if vertex construction then fails, erase the unpublished entry before propagating.
    const auto [it, inserted] = pipeline_vertex_by_identity.emplace(identity, id);
    if (!inserted) {
      throw std::logic_error(std::string(operation.empty() ? "Graph composition" : operation) +
                             ": duplicate Node identity reached indexed insertion");
    }
    try {
      vertices.emplace_back(std::move(node));
    } catch (...) {
      pipeline_vertex_by_identity.erase(it);
      throw;
    }
  } else {
    vertices.emplace_back(std::move(node));
  }
  session_test::maybe_throw_composition_failure_for_test(
      session_test::CompositionFailurePoint::PipelineVertexAppended);
  return id;
}

Graph::CompositionGraph::VertexId Graph::CompositionGraph::append_vertex(NodePtr node) {
  if (!vertices.empty() && tail == kInvalid) {
    throw std::runtime_error("Graph::add after endpoint branching is ambiguous; use connect()");
  }
  const VertexId id = append_unlinked_pipeline_vertex(std::move(node), "Graph::add");
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

Graph::CompositionGraph::VertexId
Graph::CompositionGraph::append_unlinked_runtime_vertex(RuntimeNodePtr node) {
  const VertexId id = vertices.size();
  vertices.push_back(CompositionVertex::runtime(std::move(node)));
  return id;
}

void Graph::CompositionGraph::verify_identity_index_or_throw(std::string_view operation) const {
  std::size_t expected = 0;
  for (VertexId id = 0; id < vertices.size(); ++id) {
    const auto& vertex = vertices[id];
    if (vertex.kind != CompositionVertex::Kind::PipelineNode || !vertex.pipeline_node) {
      continue;
    }
    ++expected;
    const auto it = pipeline_vertex_by_identity.find(vertex.pipeline_node.get());
    if (it == pipeline_vertex_by_identity.end() || it->second != id) {
      throw std::logic_error(std::string(operation.empty() ? "Graph composition" : operation) +
                             ": pipeline Node identity index is inconsistent at vertex " +
                             std::to_string(id));
    }
  }
  if (pipeline_vertex_by_identity.size() != expected) {
    throw std::logic_error(std::string(operation.empty() ? "Graph composition" : operation) +
                           ": pipeline Node identity index contains stale entries");
  }
}

Graph::CompositionMutation::CompositionMutation(Graph& owner)
    : owner_(owner), composition_was_null_(!owner.composition_) {
  if (!owner_.composition_) {
    owner_.composition_ = std::make_unique<CompositionGraph>();
  }
  auto& composition = *owner_.composition_;
  if (composition.active_mutation != nullptr) {
    if (composition_was_null_) {
      owner_.composition_.reset();
    }
    throw std::logic_error("Graph composition does not support nested mutations");
  }
  vertices_size_ = composition.vertices.size();
  edges_size_ = composition.edges.size();
  fragments_size_ = composition.fragments.size();
  named_fragments_size_ = composition.named_fragments.size();
  groups_size_ = owner_.groups_.size();
  tail_ = composition.tail;
  endpoint_mode_ = composition.endpoint_mode;
  input_route_processor_ = owner_.input_route_processor_;
  composition.active_mutation = this;
}

Graph::CompositionMutation::~CompositionMutation() noexcept {
  static_assert(std::is_nothrow_swappable_v<CompositionEdge>);
  static_assert(std::is_nothrow_swappable_v<std::vector<CompositionEdge>>);
  static_assert(std::is_nothrow_move_assignable_v<CompositionEdge>);
  if (committed_ || !owner_.composition_) {
    return;
  }

  auto& composition = *owner_.composition_;
  try {
    if (replaced_edges_) {
      composition.edges.swap(*replaced_edges_);
    } else {
      if (composition.edges.size() > edges_size_) {
        composition.edges.erase(composition.edges.begin() +
                                    static_cast<std::ptrdiff_t>(edges_size_),
                                composition.edges.end());
      }
      for (auto& [index, before] : changed_edges_) {
        if (index < composition.edges.size()) {
          using std::swap;
          swap(composition.edges[index], before);
        }
      }
    }

    for (auto it = composition.imported_fragments.begin();
         it != composition.imported_fragments.end();) {
      if (it->second.start >= vertices_size_) {
        it = composition.imported_fragments.erase(it);
      } else {
        ++it;
      }
    }
    for (auto it = composition.imported_models.begin(); it != composition.imported_models.end();) {
      if (it->second.start >= vertices_size_) {
        it = composition.imported_models.erase(it);
      } else {
        ++it;
      }
    }

    for (std::size_t id = vertices_size_; id < composition.vertices.size(); ++id) {
      const auto& vertex = composition.vertices[id];
      if (vertex.kind == CompositionGraph::CompositionVertex::Kind::PipelineNode &&
          vertex.pipeline_node) {
        composition.pipeline_vertex_by_identity.erase(vertex.pipeline_node.get());
      }
    }
    if (composition.vertices.size() > vertices_size_) {
      composition.vertices.erase(composition.vertices.begin() +
                                     static_cast<std::ptrdiff_t>(vertices_size_),
                                 composition.vertices.end());
    }
    if (composition.fragments.size() > fragments_size_) {
      composition.fragments.erase(composition.fragments.begin() +
                                      static_cast<std::ptrdiff_t>(fragments_size_),
                                  composition.fragments.end());
    }
    if (composition.named_fragments.size() > named_fragments_size_) {
      composition.named_fragments.erase(composition.named_fragments.begin() +
                                            static_cast<std::ptrdiff_t>(named_fragments_size_),
                                        composition.named_fragments.end());
    }
    if (owner_.groups_.size() > groups_size_) {
      owner_.groups_.erase(owner_.groups_.begin() + static_cast<std::ptrdiff_t>(groups_size_),
                           owner_.groups_.end());
    }
    composition.tail = tail_;
    composition.endpoint_mode = endpoint_mode_;
    owner_.input_route_processor_.swap(input_route_processor_);
    composition.active_mutation = nullptr;
    if (composition_was_null_) {
      owner_.composition_.reset();
    }
  } catch (...) {
    // Rollback is required to be nothrow. All operations above are tail erases, default-hash
    // erases, swaps, and destruction of standard-library value types. Terminate rather than leave
    // a partially rolled-back observable Graph if an implementation violates that contract.
    std::terminate();
  }
}

void Graph::CompositionMutation::before_edge_write(std::size_t edge_index) {
  auto& composition = *owner_.composition_;
  if (replaced_edges_ || edge_index >= edges_size_) {
    return;
  }
  const bool inserted = changed_edge_indices_.insert(edge_index).second;
  if (inserted) {
    // Copy before the caller writes so allocation/copy failure leaves the composition untouched.
    try {
      changed_edges_.emplace_back(edge_index, composition.edges.at(edge_index));
    } catch (...) {
      changed_edge_indices_.erase(edge_index);
      throw;
    }
  }
}

void Graph::CompositionMutation::replace_edges(std::vector<CompositionEdge> replacement) {
  auto& composition = *owner_.composition_;
  if (replaced_edges_) {
    composition.edges.swap(replacement);
    return;
  }
  // If a future operation edits an edge and then replaces the vector in the same transaction,
  // retain the true pre-operation version for rollback while leaving the caller's replacement
  // (which was built from the edited vector) unchanged.
  for (auto& [index, before] : changed_edges_) {
    if (index < composition.edges.size()) {
      using std::swap;
      swap(composition.edges[index], before);
    }
  }
  changed_edges_.clear();
  changed_edge_indices_.clear();
  // Endpoint promotion builds `replacement` completely before reaching here. Both swaps are
  // nothrow and preserve the original vector as the rollback before-image.
  replaced_edges_.emplace();
  composition.edges.swap(*replaced_edges_);
  composition.edges.swap(replacement);
}

void Graph::CompositionMutation::commit() noexcept {
  owner_.composition_->active_mutation = nullptr;
  owner_.mark_composition_changed();
  committed_ = true;
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
      const bool existing_realtime = is_realtime_stream_policy(edge.link_options.policy);
      const bool incoming_realtime = is_realtime_stream_policy(link_options.policy);
      const bool default_live_fan_in = edge.link_options.policy == GraphLinkPolicy::Default &&
                                       link_options.policy == GraphLinkPolicy::Default &&
                                       vertex_is_live_source_context(*this, edge.from) &&
                                       vertex_is_live_source_context(*this, from);
      if (existing_realtime || incoming_realtime || default_live_fan_in) {
        if (active_mutation) {
          active_mutation->before_edge_write(static_cast<std::size_t>(&edge - edges.data()));
        }
        merge_realtime_link_options(edge.link_options, link_options);
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
  session_test::maybe_throw_composition_failure_for_test(
      session_test::CompositionFailurePoint::BeforeConnectionEdgeAppend);
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
        const bool existing_realtime = is_realtime_stream_policy(edge.link_options.policy);
        const bool incoming_realtime = is_realtime_stream_policy(link_options.policy);
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
          if (active_mutation) {
            active_mutation->before_edge_write(static_cast<std::size_t>(&edge - edges.data()));
          }
          merge_realtime_link_options(edge.link_options, link_options);
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

  session_test::maybe_throw_composition_failure_for_test(
      session_test::CompositionFailurePoint::BeforeConnectionEdgeAppend);
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

#include "ExecutionGraphPlan.h"

#include "graph/Compiler.h"
#include "graph/Graph.h"
#include "graph/GraphRun.h"
#include "graph/nodes/FanOut.h"
#include "graph/nodes/JoinBundle.h"
#include "graph/nodes/PipelineNode.h"
#include "graph/nodes/StageNode.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelRouteRetarget.h"
#include "nodes/common/Output.h"
#include "nodes/common/Queue.h"
#include "nodes/io/Input.h"
#include "nodes/sima/Preproc.h"
#include "pipeline/Graph.h"
#include "pipeline/internal/BuildTiming.h"
#include "pipeline/internal/InputPolicy.h"
#include "pipeline/internal/InputStreamUtil.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include <utility>
#include <vector>

namespace simaai::neat::runtime {
namespace {

std::string graph_node_label(const graph::Graph& graph, graph::NodeId id) {
  std::string label;
  const auto& node = graph.node(id);
  if (node) {
    label = node->user_label();
    if (label.empty()) {
      label = node->kind();
    }
  }
  if (label.empty()) {
    label = "node" + std::to_string(id);
  }
  return label;
}

bool is_terminal_output_only_segment(const graph::CompiledPipelineSegment& seg) {
  const auto& nodes = seg.nodes;
  if (nodes.size() != 1U || !nodes.front()) {
    return false;
  }
  return dynamic_cast<const simaai::neat::Output*>(nodes.front().get()) != nullptr;
}

bool is_direct_source_input_only_segment(const graph::CompiledPipelineSegment& seg) {
  const auto& nodes = seg.nodes;
  if (nodes.size() != 1U || !nodes.front()) {
    return false;
  }
  return dynamic_cast<const simaai::neat::Input*>(nodes.front().get()) != nullptr;
}

bool is_public_input_only_fragment(std::span<const std::shared_ptr<simaai::neat::Node>> nodes) {
  return nodes.size() == 1U && nodes.front() &&
         dynamic_cast<const simaai::neat::Input*>(nodes.front().get()) != nullptr;
}

BoundaryPolicy decide_boundary_policy(const graph::Graph& graph,
                                      const graph::CompiledPipelineSegment& seg) {
  BoundaryPolicy out;
  out.source_like = seg.source_like;

  const bool is_graph_source = !seg.node_ids.empty() && graph.in_degree(seg.node_ids.front()) == 0U;
  const bool is_graph_sink = !seg.node_ids.empty() && graph.out_degree(seg.node_ids.back()) == 0U;
  const bool need_output = !seg.output_edges.empty() || is_graph_sink;
  const bool direct_graph_source =
      is_graph_source && !seg.output_edges.empty() && is_direct_source_input_only_segment(seg);
  const bool direct_graph_sink = is_graph_sink && is_terminal_output_only_segment(seg);

  out.direct_graph_source = direct_graph_source;
  out.direct_graph_sink = direct_graph_sink;
  out.needs_input = !seg.source_like && !direct_graph_source && !direct_graph_sink;
  out.needs_output = need_output && !direct_graph_source && !direct_graph_sink;
  out.terminal_output = is_graph_sink;
  out.graph_internal_output = !direct_graph_source && !seg.output_edges.empty();
  return out;
}

void resolve_default_endpoints(const graph::Graph& graph, ExecutionGraphPlan* plan) {
  if (!plan) {
    return;
  }
  plan->default_input.reset();
  plan->default_output.reset();

  std::vector<Endpoint> inputs;
  std::vector<Endpoint> outputs;
  std::unordered_set<graph::NodeId> pipeline_nodes;

  for (const auto& seg : plan->pipeline_segments) {
    pipeline_nodes.insert(seg.node_ids.begin(), seg.node_ids.end());
    if (seg.node_ids.empty()) {
      continue;
    }
    if ((seg.boundary.direct_graph_source || seg.boundary.needs_input) && seg.input_edges.empty()) {
      inputs.push_back(Endpoint{
          .kind = Endpoint::Kind::PipelineInput,
          .node = seg.node_ids.front(),
          .port = graph::kInvalidPort,
          .segment = seg.id,
      });
    }
    if (seg.boundary.terminal_output) {
      outputs.push_back(Endpoint{
          .kind = seg.boundary.direct_graph_sink ? Endpoint::Kind::GraphSink
                                                 : Endpoint::Kind::PipelineOutput,
          .node = seg.node_ids.back(),
          .port = graph::kInvalidPort,
          .segment = seg.id,
      });
    }
  }

  for (graph::NodeId id = 0; id < graph.node_count(); ++id) {
    if (graph.out_degree(id) == 0U && pipeline_nodes.find(id) == pipeline_nodes.end()) {
      outputs.push_back(Endpoint{
          .kind = Endpoint::Kind::GraphSink,
          .node = id,
          .port = graph::kInvalidPort,
          .segment = static_cast<std::size_t>(-1),
      });
    }
  }

  if (inputs.size() == 1U) {
    plan->default_input = inputs.front();
  }
  if (outputs.size() == 1U) {
    plan->default_output = outputs.front();
  }
  plan->input_endpoints = std::move(inputs);
  plan->output_endpoints = std::move(outputs);
}

template <typename Edge> bool is_implicit_composition_edge(const Edge& edge) {
  using Kind = std::decay_t<decltype(edge.kind)>;
  return edge.kind == Kind::ImplicitLinear;
}

template <typename Edge> bool is_explicit_composition_edge(const Edge& edge) {
  return !is_implicit_composition_edge(edge);
}

template <typename Edge> bool is_public_endpoint_edge(const Edge& edge) {
  using Kind = std::decay_t<decltype(edge.kind)>;
  return edge.kind == Kind::PublicEndpoint;
}

template <typename Edge> bool is_runtime_port_edge(const Edge& edge) {
  using Kind = std::decay_t<decltype(edge.kind)>;
  return edge.kind == Kind::RuntimePort;
}

struct PublicGraphLowering {
  graph::Graph graph;
  std::unordered_map<graph::NodeId, std::pair<std::size_t, std::size_t>> graph_range_by_node;
  std::vector<graph::NodeId> runtime_node_for_vertex;
};

enum class NormalizedCompositionEdgeKind {
  ImplicitLinear,
  RuntimePort,
  PublicEndpoint,
};

struct NormalizedCompositionEdge {
  std::size_t from = static_cast<std::size_t>(-1);
  std::size_t to = static_cast<std::size_t>(-1);
  NormalizedCompositionEdgeKind kind = NormalizedCompositionEdgeKind::ImplicitLinear;
  std::string from_port;
  std::string to_port;
  std::string from_endpoint;
  std::string to_endpoint;
};

struct NormalizedPublicView {
  static constexpr std::size_t kInvalid = static_cast<std::size_t>(-1);

  std::vector<std::shared_ptr<simaai::neat::Node>> vertices;
  std::vector<std::shared_ptr<simaai::neat::graph::Node>> runtime_vertices;
  std::vector<NormalizedCompositionEdge> edges;
  std::vector<FragmentPlan> fragments;
  std::vector<std::size_t> original_for_vertex;
  std::vector<std::size_t> vertex_for_original;
};

struct LoweredExplicitEdge {
  graph::NodeId from = graph::kInvalidNode;
  graph::NodeId to = graph::kInvalidNode;
  std::string from_port;
  std::string to_port;
};

CombinePolicy output_combine_policy(const std::shared_ptr<simaai::neat::Node>& node) {
  if (const auto* output = dynamic_cast<const simaai::neat::Output*>(node.get())) {
    return output->options().combine_policy;
  }
  return CombinePolicy::None;
}

simaai::neat::graph::nodes::JoinKeyPolicy join_key_policy_for(CombinePolicy policy) {
  switch (policy) {
  case CombinePolicy::ByPts:
    return simaai::neat::graph::nodes::JoinKeyPolicy::StreamPts;
  case CombinePolicy::ByFrame:
  case CombinePolicy::None:
    return simaai::neat::graph::nodes::JoinKeyPolicy::StreamFrame;
  }
  return simaai::neat::graph::nodes::JoinKeyPolicy::StreamFrame;
}

std::string endpoint_from_name(const auto& edge, std::size_t fallback_index) {
  if (edge.endpoint.has_value() && !edge.endpoint->from_endpoint.empty()) {
    return edge.endpoint->from_endpoint;
  }
  if (!edge.from_port.empty()) {
    return edge.from_port;
  }
  return "input_" + std::to_string(fallback_index);
}

std::string endpoint_name_for_vertex(std::span<const std::shared_ptr<simaai::neat::Node>> vertices,
                                     std::size_t vertex, std::string fallback) {
  if (vertex < vertices.size()) {
    if (const auto* input = dynamic_cast<const simaai::neat::Input*>(vertices[vertex].get())) {
      if (!input->endpoint_name().empty()) {
        return input->endpoint_name();
      }
    }
    if (const auto* output = dynamic_cast<const simaai::neat::Output*>(vertices[vertex].get())) {
      if (!output->endpoint_name().empty()) {
        return output->endpoint_name();
      }
    }
  }
  return fallback;
}

std::string normalized_edge_from_name(const NormalizedCompositionEdge& edge,
                                      std::span<const std::shared_ptr<simaai::neat::Node>> vertices,
                                      std::size_t fallback_index) {
  if (!edge.from_endpoint.empty()) {
    return edge.from_endpoint;
  }
  if (!edge.from_port.empty()) {
    return edge.from_port;
  }
  return endpoint_name_for_vertex(vertices, edge.from, "input_" + std::to_string(fallback_index));
}

std::string fanout_branch_port(std::size_t index) {
  return "branch" + std::to_string(index);
}

bool vertex_is_input(std::span<const std::shared_ptr<simaai::neat::Node>> vertices,
                     std::size_t index) {
  return index < vertices.size() && vertices[index] &&
         dynamic_cast<const simaai::neat::Input*>(vertices[index].get()) != nullptr;
}

bool vertex_is_input(const std::vector<std::shared_ptr<simaai::neat::Node>>& vertices,
                     std::size_t index) {
  return index < vertices.size() && vertices[index] &&
         dynamic_cast<const simaai::neat::Input*>(vertices[index].get()) != nullptr;
}

bool vertex_is_output(std::span<const std::shared_ptr<simaai::neat::Node>> vertices,
                      std::size_t index) {
  return index < vertices.size() && vertices[index] &&
         dynamic_cast<const simaai::neat::Output*>(vertices[index].get()) != nullptr;
}

bool vertex_is_runtime_node(
    std::span<const std::shared_ptr<simaai::neat::graph::Node>> runtime_vertices,
    std::size_t index) {
  return index < runtime_vertices.size() && runtime_vertices[index] != nullptr;
}

bool vertex_is_runtime_node(
    const std::vector<std::shared_ptr<simaai::neat::graph::Node>>& runtime_vertices,
    std::size_t index) {
  return index < runtime_vertices.size() && runtime_vertices[index] != nullptr;
}

NormalizedCompositionEdgeKind normalized_kind_from_public_edge_kind(const auto& edge,
                                                                    bool bypassed_boundary) {
  if (is_implicit_composition_edge(edge)) {
    return bypassed_boundary ? NormalizedCompositionEdgeKind::RuntimePort
                             : NormalizedCompositionEdgeKind::ImplicitLinear;
  }
  if (is_public_endpoint_edge(edge)) {
    return NormalizedCompositionEdgeKind::PublicEndpoint;
  }
  return NormalizedCompositionEdgeKind::RuntimePort;
}

template <typename View>
NormalizedPublicView normalize_public_boundaries_for_execution(const View& view) {
  NormalizedPublicView out;
  const std::size_t n = view.vertices.size();
  out.vertex_for_original.assign(n, NormalizedPublicView::kInvalid);
  out.fragments.assign(view.fragments.begin(), view.fragments.end());

  std::vector<std::size_t> explicit_in(n, 0U);
  std::vector<std::size_t> explicit_out(n, 0U);
  std::vector<std::vector<std::size_t>> outgoing(n);
  outgoing.reserve(n);

  for (std::size_t edge_index = 0; edge_index < view.edges.size(); ++edge_index) {
    const auto& edge = view.edges[edge_index];
    if (edge.from >= n || edge.to >= n) {
      throw std::runtime_error("compile_public_graph: composition edge references invalid vertex");
    }
    outgoing[edge.from].push_back(edge_index);
    if (is_explicit_composition_edge(edge)) {
      ++explicit_in[edge.to];
      ++explicit_out[edge.from];
    }
  }

  std::vector<bool> elide(n, false);
  for (std::size_t vertex = 0; vertex < n; ++vertex) {
    // Boundary declaration materialization rule:
    //   - Input with an upstream explicit connection is an internal ingress declaration.
    //   - Output with a downstream explicit connection is an internal egress declaration.
    // Such nodes remain in the public/provenance graph, but they are not executable work and
    // must not become extra appsrc/appsink elements inside the pipeline segment.
    if (vertex_is_input(view.vertices, vertex) && explicit_in[vertex] > 0U) {
      elide[vertex] = true;
    } else if (vertex_is_output(view.vertices, vertex) && explicit_out[vertex] > 0U) {
      elide[vertex] = true;
    }
  }

  out.vertices.reserve(n);
  out.runtime_vertices.reserve(n);
  out.original_for_vertex.reserve(n);
  for (std::size_t original = 0; original < n; ++original) {
    if (elide[original]) {
      continue;
    }
    out.vertex_for_original[original] = out.vertices.size();
    out.original_for_vertex.push_back(original);
    out.vertices.push_back(view.vertices[original]);
    out.runtime_vertices.push_back(
        original < view.runtime_vertices.size() ? view.runtime_vertices[original] : nullptr);
  }

  std::unordered_set<std::string> emitted_edges;
  const auto add_edge = [&](std::size_t from, std::size_t to, NormalizedCompositionEdgeKind kind,
                            std::string from_port, std::string to_port, std::string from_endpoint,
                            std::string to_endpoint) {
    if (from == NormalizedPublicView::kInvalid || to == NormalizedPublicView::kInvalid) {
      return;
    }
    if (from == to) {
      return;
    }
    const std::string key = std::to_string(from) + ":" + std::to_string(to) + ":" +
                            std::to_string(static_cast<int>(kind)) + ":" + from_port + ":" +
                            to_port;
    if (!emitted_edges.insert(key).second) {
      return;
    }
    out.edges.push_back(NormalizedCompositionEdge{.from = from,
                                                  .to = to,
                                                  .kind = kind,
                                                  .from_port = std::move(from_port),
                                                  .to_port = std::move(to_port),
                                                  .from_endpoint = std::move(from_endpoint),
                                                  .to_endpoint = std::move(to_endpoint)});
  };

  std::function<void(std::size_t, std::size_t, std::string, std::string, std::string, bool,
                     std::vector<bool>&)>
      follow_edge;
  follow_edge = [&](std::size_t start_norm, std::size_t edge_index, std::string from_port,
                    std::string from_endpoint, std::string to_endpoint, bool bypassed_boundary,
                    std::vector<bool>& visiting) {
    const auto& edge = view.edges[edge_index];
    if (!from_port.empty() && !edge.from_port.empty() && from_port != edge.from_port) {
      throw std::runtime_error(
          "compile_public_graph: boundary normalization encountered conflicting source ports");
    }
    if (from_port.empty()) {
      from_port = edge.from_port;
    }
    if (edge.endpoint.has_value()) {
      if (from_endpoint.empty()) {
        from_endpoint = edge.endpoint->from_endpoint;
      }
      if (!edge.endpoint->to_endpoint.empty()) {
        to_endpoint = edge.endpoint->to_endpoint;
      }
    }

    const std::size_t to_original = edge.to;
    if (!elide[to_original]) {
      const std::size_t to_norm = out.vertex_for_original[to_original];
      NormalizedCompositionEdgeKind kind =
          normalized_kind_from_public_edge_kind(edge, bypassed_boundary);
      std::string to_port = edge.to_port;
      if (kind == NormalizedCompositionEdgeKind::ImplicitLinear) {
        from_port.clear();
        to_port.clear();
      }
      add_edge(start_norm, to_norm, kind, std::move(from_port), std::move(to_port),
               std::move(from_endpoint), std::move(to_endpoint));
      return;
    }

    if (visiting[to_original]) {
      throw std::runtime_error(
          "compile_public_graph: cycle while normalizing public boundary declarations");
    }
    visiting[to_original] = true;

    if (outgoing[to_original].empty()) {
      throw std::runtime_error(
          "compile_public_graph: internal boundary declaration has no downstream executable path");
    }
    for (const std::size_t next_edge : outgoing[to_original]) {
      follow_edge(start_norm, next_edge, from_port, from_endpoint, to_endpoint, true, visiting);
    }
    visiting[to_original] = false;
  };

  for (std::size_t edge_index = 0; edge_index < view.edges.size(); ++edge_index) {
    const auto& edge = view.edges[edge_index];
    if (elide[edge.from]) {
      continue;
    }
    const std::size_t from_norm = out.vertex_for_original[edge.from];
    std::vector<bool> visiting(n, false);
    follow_edge(from_norm, edge_index, edge.from_port, {}, {}, false, visiting);
  }

  // Pipeline-quality pass: if public composition produced a simple one-to-one connection,
  // keep it as an implicit executable edge so adjacent pipeline work can fuse into one
  // GStreamer pipeline segment. Explicit runtime edges are still preserved for real topology:
  // fan-out, fan-in, non-default ports, and future stage-routing cases remain explicit.
  std::vector<std::size_t> in_degree(out.vertices.size(), 0U);
  std::vector<std::size_t> out_degree(out.vertices.size(), 0U);
  for (const auto& edge : out.edges) {
    if (edge.from < out_degree.size()) {
      ++out_degree[edge.from];
    }
    if (edge.to < in_degree.size()) {
      ++in_degree[edge.to];
    }
  }
  for (auto& edge : out.edges) {
    if (edge.kind == NormalizedCompositionEdgeKind::ImplicitLinear) {
      continue;
    }
    if (vertex_is_runtime_node(out.runtime_vertices, edge.from) ||
        vertex_is_runtime_node(out.runtime_vertices, edge.to)) {
      continue;
    }
    const bool default_ports = (edge.from_port.empty() || edge.from_port == "out") &&
                               (edge.to_port.empty() || edge.to_port == "in");
    if (default_ports && edge.from < out_degree.size() && edge.to < in_degree.size() &&
        out_degree[edge.from] == 1U && in_degree[edge.to] == 1U) {
      edge.kind = NormalizedCompositionEdgeKind::ImplicitLinear;
      edge.from_port.clear();
      edge.to_port.clear();
    }
  }

  return out;
}

const FragmentBoundaryHints* boundary_hints_for_normalized_target(const NormalizedPublicView& view,
                                                                  std::size_t normalized_vertex) {
  if (normalized_vertex >= view.original_for_vertex.size()) {
    return nullptr;
  }
  const std::size_t original = view.original_for_vertex[normalized_vertex];
  for (const auto& fragment : view.fragments) {
    if (!fragment.boundary_hints.has_value() || fragment.graph_end <= fragment.graph_start) {
      continue;
    }
    if (original >= fragment.graph_start && original < fragment.graph_end) {
      return &*fragment.boundary_hints;
    }
  }
  return nullptr;
}

std::string normalized_edge_to_name(const NormalizedCompositionEdge& edge,
                                    std::span<const std::shared_ptr<simaai::neat::Node>> vertices,
                                    std::size_t fallback_index) {
  if (!edge.to_endpoint.empty()) {
    return edge.to_endpoint;
  }
  if (!edge.to_port.empty()) {
    return edge.to_port;
  }
  return endpoint_name_for_vertex(vertices, edge.to, "input_" + std::to_string(fallback_index));
}

void connect_lowered_explicit_edges(graph::Graph* graph,
                                    const std::vector<LoweredExplicitEdge>& edges) {
  if (!graph) {
    return;
  }

  std::vector<bool> emitted(edges.size(), false);
  for (std::size_t i = 0; i < edges.size(); ++i) {
    if (emitted[i]) {
      continue;
    }

    std::vector<std::size_t> group;
    group.push_back(i);
    for (std::size_t j = i + 1U; j < edges.size(); ++j) {
      if (emitted[j]) {
        continue;
      }
      if (edges[j].from == edges[i].from && edges[j].from_port == edges[i].from_port) {
        group.push_back(j);
      }
    }

    if (group.size() == 1U) {
      const auto& edge = edges[i];
      graph->connect(edge.from, edge.to, edge.from_port, edge.to_port);
      emitted[i] = true;
      continue;
    }

    std::vector<std::string> branch_ports;
    branch_ports.reserve(group.size());
    for (std::size_t branch = 0; branch < group.size(); ++branch) {
      branch_ports.push_back(fanout_branch_port(branch));
    }

    const graph::NodeId fanout = graph->add(
        graph::nodes::FanOutNode(branch_ports, "fanout" + std::to_string(edges[i].from), "in"));
    graph->connect(edges[i].from, fanout, edges[i].from_port, "in");
    for (std::size_t branch = 0; branch < group.size(); ++branch) {
      const auto& edge = edges[group[branch]];
      graph->connect(fanout, edge.to, branch_ports[branch], edge.to_port);
      emitted[group[branch]] = true;
    }
  }
}

template <typename View>
PublicGraphLowering
build_runtime_graph_from_connected_public_view(const View& view,
                                               const std::optional<SampleSpec>& seed_spec) {
  // Keep seed completion local to the actual pipeline build. The execution plan must preserve
  // the user's original InputOptions; otherwise a seed frame (for example 64x64) looks like an
  // explicit user width and later becomes max_width=64 through normalize_shape_bounds().
  (void)seed_spec;
  PublicGraphLowering out;
  if (view.vertices.empty()) {
    return out;
  }

  std::vector<std::size_t> implicit_next(view.vertices.size(), static_cast<std::size_t>(-1));
  std::vector<std::size_t> implicit_prev(view.vertices.size(), static_cast<std::size_t>(-1));
  std::vector<std::size_t> public_endpoint_in_degree(view.vertices.size(), 0U);
  for (const auto& edge : view.edges) {
    if (edge.from >= view.vertices.size() || edge.to >= view.vertices.size()) {
      throw std::runtime_error("compile_public_graph: composition edge references invalid vertex");
    }
    if (is_explicit_composition_edge(edge)) {
      if (is_public_endpoint_edge(edge)) {
        ++public_endpoint_in_degree[edge.to];
      }
      continue;
    }
    if (vertex_is_runtime_node(view.runtime_vertices, edge.from) ||
        vertex_is_runtime_node(view.runtime_vertices, edge.to)) {
      // Runtime-stage vertices are executable units on their own.  An add()-style
      // implicit edge crossing a runtime node is semantically a runtime edge with
      // default ports; it must not fuse the stage into a PipelineNode fragment.
      continue;
    }
    if (implicit_next[edge.from] != static_cast<std::size_t>(-1)) {
      throw std::runtime_error("compile_public_graph: ambiguous implicit linear successor");
    }
    if (implicit_prev[edge.to] != static_cast<std::size_t>(-1)) {
      throw std::runtime_error("compile_public_graph: ambiguous implicit linear predecessor");
    }
    implicit_next[edge.from] = edge.to;
    implicit_prev[edge.to] = edge.from;
  }

  std::vector<graph::NodeId> runtime_node_for_vertex(view.vertices.size(), graph::kInvalidNode);
  std::vector<bool> visited(view.vertices.size(), false);
  for (std::size_t id = 0; id < view.vertices.size(); ++id) {
    if (!vertex_is_runtime_node(view.runtime_vertices, id)) {
      continue;
    }
    const auto& runtime_node = view.runtime_vertices[id];
    if (!runtime_node) {
      throw std::runtime_error("compile_public_graph: runtime vertex has no runtime node");
    }
    const graph::NodeId runtime_id = out.graph.add(runtime_node);
    out.graph_range_by_node[runtime_id] = {id, id + 1U};
    runtime_node_for_vertex[id] = runtime_id;
    visited[id] = true;
  }

  for (std::size_t start = 0; start < view.vertices.size(); ++start) {
    if (visited[start] || vertex_is_runtime_node(view.runtime_vertices, start)) {
      continue;
    }
    if (implicit_prev[start] != static_cast<std::size_t>(-1)) {
      continue;
    }

    std::vector<std::shared_ptr<simaai::neat::Node>> nodes;
    std::size_t cur = start;
    while (cur != static_cast<std::size_t>(-1)) {
      if (cur >= view.vertices.size()) {
        throw std::runtime_error("compile_public_graph: implicit chain escaped vertex range");
      }
      if (visited[cur]) {
        throw std::runtime_error("compile_public_graph: cycle in implicit linear chain");
      }
      if (vertex_is_runtime_node(view.runtime_vertices, cur)) {
        throw std::runtime_error(
            "compile_public_graph: runtime node unexpectedly appeared inside a pipeline chain");
      }
      visited[cur] = true;
      nodes.push_back(view.vertices[cur]);
      const std::size_t next = implicit_next[cur];
      if (next == static_cast<std::size_t>(-1)) {
        break;
      }
      cur = next;
    }

    // A connected-graph fanout directly from a bare Input used to lower to a single appsrc
    // pipeline segment feeding a runtime FanOut stage.  GStreamer parses "appsrc ..." as a
    // standalone element rather than a bin in that shape, which made build fail with
    // "parser returned non-bin root element".  Insert the same queue users had to add by hand
    // whenever a public Input-only fragment has downstream graph edges: appsrc ! queue is a real
    // segment boundary, provides a thread/caps handoff point, and keeps tensor/sample semantics
    // unchanged.
    if (is_public_input_only_fragment(nodes)) {
      const bool has_downstream_edge =
          std::any_of(view.edges.begin(), view.edges.end(),
                      [&](const auto& edge) { return edge.from == start; });
      if (has_downstream_edge) {
        nodes.push_back(simaai::neat::nodes::Queue());
      }
    }

    auto pipeline_node = std::make_shared<graph::nodes::PipelineNode>(
        std::move(nodes), "fragment" + std::to_string(start));
    const graph::NodeId runtime_id = out.graph.add(std::move(pipeline_node));
    out.graph_range_by_node[runtime_id] = {start, cur + 1U};
    cur = start;
    while (cur != static_cast<std::size_t>(-1)) {
      runtime_node_for_vertex[cur] = runtime_id;
      const std::size_t next = implicit_next[cur];
      if (next == static_cast<std::size_t>(-1)) {
        break;
      }
      cur = next;
    }
  }

  for (std::size_t id = 0; id < visited.size(); ++id) {
    if (!visited[id]) {
      throw std::runtime_error("compile_public_graph: unvisited vertex in composition");
    }
  }

  std::vector<LoweredExplicitEdge> lowered_edges;
  std::unordered_map<std::size_t, std::vector<std::size_t>> public_endpoint_inputs_by_target;
  for (std::size_t i = 0; i < view.edges.size(); ++i) {
    const auto& edge = view.edges[i];
    if (is_public_endpoint_edge(edge)) {
      public_endpoint_inputs_by_target[edge.to].push_back(i);
    }
  }

  std::unordered_set<std::size_t> combine_handled_edges;
  for (const auto& [target_vertex, incoming_edges] : public_endpoint_inputs_by_target) {
    if (incoming_edges.size() <= 1U) {
      continue;
    }
    if (target_vertex >= view.vertices.size()) {
      throw std::runtime_error("compile_public_graph: public endpoint target out of range");
    }
    const CombinePolicy policy = output_combine_policy(view.vertices[target_vertex]);
    const FragmentBoundaryHints* target_hints =
        boundary_hints_for_normalized_target(view, target_vertex);
    const bool model_ingress_combine = policy == CombinePolicy::None && target_hints != nullptr &&
                                       target_hints->input_route_processor != nullptr &&
                                       target_hints->ingress_endpoint_names.size() > 1U;
    if (policy == CombinePolicy::None && !model_ingress_combine) {
      throw std::runtime_error(
          "compile_public_graph: public endpoint has multiple producers; set an explicit "
          "CombinePolicy::ByFrame or CombinePolicy::ByPts");
    }

    std::vector<std::size_t> ordered_incoming_edges;
    ordered_incoming_edges.reserve(incoming_edges.size());
    std::vector<std::string> input_names;
    input_names.reserve(incoming_edges.size());
    std::unordered_set<std::string> used_input_names;
    if (model_ingress_combine) {
      for (const auto& required_name : target_hints->ingress_endpoint_names) {
        auto match =
            std::find_if(incoming_edges.begin(), incoming_edges.end(), [&](std::size_t edge_index) {
              return view.edges[edge_index].to_endpoint == required_name;
            });
        if (match == incoming_edges.end()) {
          throw std::runtime_error(
              "compile_public_graph: multi-input Model endpoint is missing producer for ingress '" +
              required_name + "'");
        }
        ordered_incoming_edges.push_back(*match);
        if (!used_input_names.insert(required_name).second) {
          throw std::runtime_error("compile_public_graph: duplicate Model ingress endpoint '" +
                                   required_name + "'");
        }
        input_names.push_back(required_name);
      }
      if (ordered_incoming_edges.size() != incoming_edges.size()) {
        throw std::runtime_error(
            "compile_public_graph: multi-input Model endpoint has unexpected extra producer; "
            "connect exactly one producer per Model ingress endpoint");
      }
    } else {
      for (std::size_t i = 0; i < incoming_edges.size(); ++i) {
        const auto& edge = view.edges[incoming_edges[i]];
        std::string name = normalized_edge_from_name(edge, view.vertices, i);
        if (!used_input_names.insert(name).second) {
          throw std::runtime_error("compile_public_graph: duplicate Combine input endpoint '" +
                                   name + "'");
        }
        ordered_incoming_edges.push_back(incoming_edges[i]);
        input_names.push_back(std::move(name));
      }
    }

    simaai::neat::graph::nodes::JoinBundleOptions join_opt;
    join_opt.inputs = input_names;
    join_opt.key_policy = model_ingress_combine
                              ? simaai::neat::graph::nodes::JoinKeyPolicy::StreamFrame
                              : join_key_policy_for(policy);
    join_opt.include_stream_id_in_key = false;
    join_opt.allow_key_fallback = false;
    const graph::NodeId join_node = out.graph.add(simaai::neat::graph::nodes::JoinBundleNode(
        input_names, "combine_" + std::to_string(target_vertex), "bundle", join_opt));

    for (std::size_t i = 0; i < ordered_incoming_edges.size(); ++i) {
      const auto& edge = view.edges[ordered_incoming_edges[i]];
      const graph::NodeId from = runtime_node_for_vertex[edge.from];
      if (from == graph::kInvalidNode) {
        throw std::runtime_error("compile_public_graph: combine edge references unmapped source");
      }
      lowered_edges.push_back(LoweredExplicitEdge{
          .from = from,
          .to = join_node,
          .from_port = edge.from_port.empty() ? "out" : edge.from_port,
          .to_port = input_names[i],
      });
      combine_handled_edges.insert(ordered_incoming_edges[i]);
    }

    const graph::NodeId to = runtime_node_for_vertex[target_vertex];
    if (to == graph::kInvalidNode) {
      throw std::runtime_error("compile_public_graph: combine target is unmapped");
    }
    lowered_edges.push_back(LoweredExplicitEdge{
        .from = join_node,
        .to = to,
        .from_port = "bundle",
        .to_port = "in",
    });
  }

  for (std::size_t edge_index = 0; edge_index < view.edges.size(); ++edge_index) {
    const auto& edge = view.edges[edge_index];
    const bool fused_implicit_pipeline_edge =
        !is_explicit_composition_edge(edge) &&
        !vertex_is_runtime_node(view.runtime_vertices, edge.from) &&
        !vertex_is_runtime_node(view.runtime_vertices, edge.to);
    if (fused_implicit_pipeline_edge) {
      continue;
    }
    if (combine_handled_edges.find(edge_index) != combine_handled_edges.end()) {
      continue;
    }
    if (is_public_endpoint_edge(edge) && public_endpoint_in_degree[edge.to] > 1U) {
      throw std::runtime_error(
          "compile_public_graph: public endpoint has multiple producers; set an explicit "
          "CombinePolicy::ByFrame or CombinePolicy::ByPts");
    }
    const graph::NodeId from = runtime_node_for_vertex[edge.from];
    const graph::NodeId to = runtime_node_for_vertex[edge.to];
    if (from == graph::kInvalidNode || to == graph::kInvalidNode) {
      throw std::runtime_error("compile_public_graph: explicit edge references unmapped vertex");
    }
    if (is_runtime_port_edge(edge) && (edge.from_port.empty() || edge.to_port.empty())) {
      throw std::runtime_error("compile_public_graph: internal runtime-port edge is missing ports");
    }
    const std::string from_port = edge.from_port.empty() ? "out" : edge.from_port;
    const std::string to_port = edge.to_port.empty() ? "in" : edge.to_port;
    lowered_edges.push_back(LoweredExplicitEdge{
        .from = from,
        .to = to,
        .from_port = from_port,
        .to_port = to_port,
    });
  }

  connect_lowered_explicit_edges(&out.graph, lowered_edges);
  out.runtime_node_for_vertex = std::move(runtime_node_for_vertex);
  return out;
}

OutputSpec input_options_to_output_spec(const InputOptions& opt) {
  OutputSpec spec;
  spec.media_type = resolve_input_media_type(opt);
  spec.format = opt.format.str();
  spec.width = opt.width;
  spec.height = opt.height;
  spec.depth = opt.depth;
  if (spec.depth <= 0 && opt.max_depth > 0) {
    spec.depth = opt.max_depth;
  }
  return spec;
}

bool input_options_complete(const InputOptions& opt) {
  const std::string media_type = resolve_input_media_type(opt);
  if (media_type.empty()) {
    return false;
  }
  if (media_type == "video/x-raw") {
    return !opt.format.empty() && opt.width > 0 && opt.height > 0;
  }
  if (media_type == "application/vnd.simaai.tensor") {
    const int depth = (opt.depth > 0) ? opt.depth : opt.max_depth;
    return !opt.format.empty() && opt.width > 0 && opt.height > 0 && depth > 0;
  }
  return !opt.format.empty();
}

bool output_spec_complete(const OutputSpec& spec) {
  const std::string media =
      !spec.media_type.empty() ? spec.media_type : media_type_from_payload_type(spec.payload_type);
  if (media.empty()) {
    return false;
  }
  if (media == "video/x-raw") {
    return !spec.format.empty() && spec.width > 0 && spec.height > 0;
  }
  if (media == "application/vnd.simaai.tensor") {
    return !spec.format.empty() && spec.width > 0 && spec.height > 0 && spec.depth > 0;
  }
  return true;
}

bool raw_video_seed_spec_complete_for_model_route(const SampleSpec& spec) {
  return spec.media_type == "video/x-raw" && !spec.format.empty() && spec.width > 0 &&
         spec.height > 0;
}

std::string raw_video_seed_debug_string(const SampleSpec& spec) {
  std::string out = "media=" + (spec.media_type.empty() ? std::string("<empty>") : spec.media_type);
  out += " format=" + (spec.format.empty() ? std::string("<empty>") : spec.format);
  out += " shape=" + std::to_string(spec.width) + "x" + std::to_string(spec.height);
  if (spec.depth > 0) {
    out += "x" + std::to_string(spec.depth);
  }
  return out;
}

[[noreturn]] void throw_raw_image_seed_without_model_preproc(const FragmentPlan& fragment,
                                                             const SampleSpec& seed_spec) {
  std::string model_ref = fragment.provenance.model_source_path;
  if (model_ref.empty()) {
    model_ref = fragment.provenance.model_id.empty() ? std::string("<unknown model>")
                                                     : fragment.provenance.model_id;
  }
  throw std::invalid_argument(
      "Graph::build(image): received a raw image seed (" + raw_video_seed_debug_string(seed_spec) +
      ") for Model::graph() route '" + model_ref +
      "', but that route has no model-managed Preproc node. If this model is intended to accept "
      "images, construct Model with Model::Options::preprocess.kind = InputKind::Image and set "
      "an explicit preprocess.color_convert.input_format plus "
      "preprocess.input_max_width/input_max_height/input_max_depth before calling "
      "model.graph(...). If the model expects preprocessed tensors, pass tensor input instead "
      "or use Model::Options::preprocess.kind = InputKind::Tensor.");
}

const internal::ModelLineageBinding* model_managed_preproc_lineage_for_fragment(
    std::span<const std::shared_ptr<simaai::neat::Node>> nodes, const FragmentPlan& fragment) {
  if (fragment.graph_end <= fragment.graph_start || fragment.graph_end > nodes.size()) {
    return nullptr;
  }
  for (std::size_t i = fragment.graph_start; i < fragment.graph_end; ++i) {
    const auto& node = nodes[i];
    if (!node) {
      continue;
    }
    const auto* preproc = dynamic_cast<const simaai::neat::Preproc*>(node.get());
    if (!preproc) {
      continue;
    }
    const auto& opt = preproc->options();
    if (opt.model_managed_contract && opt.model_lineage) {
      return opt.model_lineage.get();
    }
  }
  return nullptr;
}

bool transparent_root_seed_node(const std::shared_ptr<simaai::neat::Node>& node) {
  if (!node) {
    return true;
  }
  return dynamic_cast<const simaai::neat::Input*>(node.get()) != nullptr ||
         dynamic_cast<const simaai::neat::Queue*>(node.get()) != nullptr;
}

bool fragment_can_use_root_seed(std::span<const std::shared_ptr<simaai::neat::Node>> nodes,
                                const FragmentPlan& fragment) {
  if (fragment.graph_start > nodes.size()) {
    return false;
  }
  for (std::size_t i = 0; i < fragment.graph_start; ++i) {
    if (!transparent_root_seed_node(nodes[i])) {
      return false;
    }
  }
  return true;
}

Model::RouteOptions route_options_from_model_route_fragment(
    const FragmentPlan& fragment, std::span<const std::shared_ptr<simaai::neat::Node>> nodes) {
  Model::RouteOptions opt;
  const auto& provenance = fragment.provenance.model_route;
  opt.upstream_name = provenance.upstream_name;
  opt.name_suffix = provenance.name_suffix;
  opt.buffer_name = provenance.buffer_name;
  opt.processcvu_requested_run_target = provenance.processcvu_requested_run_target;
  opt.processcvu = provenance.processcvu;
  opt.processmla = provenance.processmla;
  opt.prepared_runner = provenance.prepared_runner;
  opt.async_queue_depth = provenance.async_queue_depth;
  opt.expose_all_outputs = provenance.expose_all_outputs;

  if (fragment.graph_start < fragment.graph_end && fragment.graph_end <= nodes.size()) {
    opt.include_input =
        dynamic_cast<const simaai::neat::Input*>(nodes[fragment.graph_start].get()) != nullptr;
    opt.include_output =
        dynamic_cast<const simaai::neat::Output*>(nodes[fragment.graph_end - 1U].get()) != nullptr;
  }
  return opt;
}

std::vector<std::shared_ptr<simaai::neat::Node>> specialize_linear_model_preproc_route_for_seed(
    std::vector<std::shared_ptr<simaai::neat::Node>> nodes, std::span<const FragmentPlan> fragments,
    const Sample& seed, const SampleSpec& seed_spec) {
  if (nodes.empty() || fragments.empty() ||
      !raw_video_seed_spec_complete_for_model_route(seed_spec)) {
    return nodes;
  }

  for (const auto& fragment : fragments) {
    if (!fragment.provenance.model_route.present || fragment.graph_end <= fragment.graph_start ||
        fragment.graph_end > nodes.size()) {
      continue;
    }
    if (fragment.boundary_hints.has_value() && fragment.boundary_hints->bundled_fan_in) {
      continue;
    }
    if (!fragment_can_use_root_seed(nodes, fragment)) {
      continue;
    }

    const internal::ModelLineageBinding* binding =
        model_managed_preproc_lineage_for_fragment(nodes, fragment);
    if (!binding) {
      throw_raw_image_seed_without_model_preproc(fragment, seed_spec);
    }
    if (binding->source_path.empty()) {
      continue;
    }

    Model model(binding->source_path, binding->base_options);
    Model::RouteOptions route_opt = route_options_from_model_route_fragment(fragment, nodes);
    auto replacement =
        internal::ModelAccess::build_public_route_nodes_for_seed(model, route_opt, seed);
    const std::size_t expected_size = fragment.graph_end - fragment.graph_start;
    if (replacement.size() != expected_size) {
      throw std::runtime_error(
          "compile_public_graph: model route seed specialization changed node count for fragment " +
          std::to_string(fragment.graph_start) + ".." + std::to_string(fragment.graph_end));
    }
    for (std::size_t i = 0; i < expected_size; ++i) {
      nodes[fragment.graph_start + i] = std::move(replacement[i]);
    }
    break;
  }

  return nodes;
}

const char* dtype_token(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt8:
    return "UInt8";
  case TensorDType::Int8:
    return "Int8";
  case TensorDType::UInt16:
    return "UInt16";
  case TensorDType::Int16:
    return "Int16";
  case TensorDType::Int32:
    return "Int32";
  case TensorDType::BFloat16:
    return "BFloat16";
  case TensorDType::Float32:
    return "Float32";
  case TensorDType::Float64:
    return "Float64";
  }
  return "";
}

const char* layout_token(TensorLayout layout) {
  switch (layout) {
  case TensorLayout::HWC:
    return "HWC";
  case TensorLayout::CHW:
    return "CHW";
  case TensorLayout::HW:
    return "HW";
  case TensorLayout::Unknown:
  default:
    return "";
  }
}

OutputSpec output_spec_from_sample_spec(const SampleSpec& sample) {
  OutputSpec spec;
  spec.payload_type = payload_type_from_media_type(sample.media_type);
  spec.media_type = sample.media_type;
  spec.format = sample.format;
  spec.width = sample.width;
  spec.height = sample.height;
  spec.depth = sample.depth;
  spec.dtype = dtype_token(sample.dtype);
  spec.layout = layout_token(sample.layout);
  spec.byte_size = sample.required_bytes_actual;
  spec.certainty = SpecCertainty::Derived;
  spec.note = "Graph build seed";
  return spec;
}

bool nodes_have_input_appsrc(std::span<const std::shared_ptr<simaai::neat::Node>> nodes) {
  for (const auto& node : nodes) {
    if (dynamic_cast<const simaai::neat::Input*>(node.get()) != nullptr) {
      return true;
    }
  }
  return false;
}

bool nodes_have_output_appsink(std::span<const std::shared_ptr<simaai::neat::Node>> nodes) {
  for (const auto& node : nodes) {
    if (dynamic_cast<const simaai::neat::Output*>(node.get()) != nullptr) {
      return true;
    }
  }
  return false;
}

bool nodes_are_source_like(std::span<const std::shared_ptr<simaai::neat::Node>> nodes) {
  bool has_source = false;
  bool has_push = false;
  for (const auto& node : nodes) {
    if (!node) {
      continue;
    }
    const InputRole role = node->input_role();
    if (role == InputRole::Source) {
      has_source = true;
    }
    if (role == InputRole::Push) {
      has_push = true;
    }
  }
  return has_source && !has_push;
}

bool is_terminal_output_only_nodes(std::span<const std::shared_ptr<simaai::neat::Node>> nodes) {
  if (nodes.size() != 1U || !nodes.front()) {
    return false;
  }
  return dynamic_cast<const simaai::neat::Output*>(nodes.front().get()) != nullptr;
}

void resolve_single_pipeline_endpoints(ExecutionGraphPlan* plan) {
  if (!plan) {
    return;
  }
  plan->default_input.reset();
  plan->default_output.reset();
  plan->input_endpoints.clear();
  plan->output_endpoints.clear();
  if (plan->pipeline_segments.empty()) {
    return;
  }

  const auto& seg = plan->pipeline_segments.front();
  if (seg.node_ids.empty()) {
    return;
  }
  if (seg.boundary.needs_input && seg.input_edges.empty()) {
    Endpoint input;
    input.kind = Endpoint::Kind::PipelineInput;
    input.node = seg.node_ids.front();
    input.port = graph::kInvalidPort;
    input.segment = seg.id;
    plan->input_endpoints.push_back(input);
    plan->default_input = input;
  }
  if (seg.boundary.terminal_output) {
    Endpoint output;
    output.kind =
        seg.boundary.direct_graph_sink ? Endpoint::Kind::GraphSink : Endpoint::Kind::PipelineOutput;
    output.node = seg.node_ids.back();
    output.port = graph::kInvalidPort;
    output.segment = seg.id;
    plan->output_endpoints.push_back(output);
    plan->default_output = output;
  }
}

ExecutionGraphPlan
build_single_pipeline_execution_plan(std::vector<std::shared_ptr<simaai::neat::Node>> nodes,
                                     const RunOptions& opt,
                                     const std::optional<OutputSpec>& root_input_spec) {
  ExecutionGraphPlan plan;
  plan.linear_compat = true;
  plan.node_labels.push_back("linear");

  PipelineSegmentPlan segment;
  segment.id = 0;
  segment.node_ids.push_back(0);
  segment.nodes = std::move(nodes);
  segment.run_options = opt;
  segment.boundary.source_like = nodes_are_source_like(segment.nodes);
  segment.boundary.direct_graph_sink = is_terminal_output_only_nodes(segment.nodes);

  const bool has_explicit_input = nodes_have_input_appsrc(segment.nodes);
  const bool has_explicit_output = nodes_have_output_appsink(segment.nodes);
  segment.boundary.needs_input =
      !segment.boundary.source_like && !segment.boundary.direct_graph_sink && has_explicit_input;
  segment.boundary.needs_output = !segment.boundary.direct_graph_sink && has_explicit_output;
  segment.boundary.terminal_output = segment.boundary.direct_graph_sink || has_explicit_output;
  segment.boundary.graph_internal_output = false;

  if (root_input_spec.has_value()) {
    segment.input_spec = *root_input_spec;
    segment.input_complete = output_spec_complete(segment.input_spec);
  }

  Provenance provenance;
  provenance.runtime_node = 0;
  provenance.segment_id = segment.id;
  segment.provenance.push_back(std::move(provenance));

  plan.pipeline_segments.push_back(std::move(segment));
  resolve_single_pipeline_endpoints(&plan);
  return plan;
}

bool ranges_overlap(std::size_t a_start, std::size_t a_end, std::size_t b_start,
                    std::size_t b_end) {
  return a_start < b_end && b_start < a_end;
}

const FragmentPlan* best_fragment_for_range(std::span<const FragmentPlan> fragments,
                                            std::size_t start, std::size_t end) {
  const FragmentPlan* best = nullptr;
  std::size_t best_overlap = 0;
  for (const auto& fragment : fragments) {
    if (!fragment.boundary_hints.has_value()) {
      continue;
    }
    if (!ranges_overlap(start, end, fragment.graph_start, fragment.graph_end)) {
      continue;
    }
    const std::size_t overlap =
        std::min(end, fragment.graph_end) - std::max(start, fragment.graph_start);
    if (overlap > best_overlap) {
      best = &fragment;
      best_overlap = overlap;
    }
  }
  return best;
}

std::string make_unique_auto_name(std::string base, const std::unordered_set<std::string>& used) {
  if (base.empty()) {
    base = "endpoint";
  }
  if (used.find(base) == used.end()) {
    return base;
  }
  for (std::size_t i = 2U;; ++i) {
    std::string candidate = base + "_" + std::to_string(i);
    if (used.find(candidate) == used.end()) {
      return candidate;
    }
  }
}

template <typename Map>
void add_named_endpoint(Map* map, std::unordered_set<std::string>* used, std::string name,
                        bool user_named, const Endpoint& endpoint, const char* kind) {
  if (!map || !used || name.empty()) {
    return;
  }
  if (user_named) {
    if (used->find(name) != used->end()) {
      throw std::runtime_error(std::string("Graph endpoint name '") + name +
                               "' is used by more than one " + kind +
                               "; choose unique names for multi-input/multi-output graphs");
    }
  } else {
    name = make_unique_auto_name(std::move(name), *used);
  }
  used->insert(name);
  map->emplace(std::move(name), endpoint);
}

std::string explicit_public_endpoint_name(const std::shared_ptr<simaai::neat::Node>& node) {
  if (!node) {
    return {};
  }
  if (const auto* input = dynamic_cast<const simaai::neat::Input*>(node.get())) {
    return input->endpoint_name();
  }
  if (const auto* output = dynamic_cast<const simaai::neat::Output*>(node.get())) {
    return output->endpoint_name();
  }
  return {};
}

std::string public_node_label(const std::shared_ptr<simaai::neat::Node>& node, std::size_t index) {
  if (!node) {
    return "node" + std::to_string(index);
  }
  std::string label = node->user_label();
  if (label.empty()) {
    label = explicit_public_endpoint_name(node);
  }
  if (label.empty()) {
    label = node->kind();
  }
  if (label.empty()) {
    label = "node" + std::to_string(index);
  }
  return label;
}

std::string public_node_label(const std::shared_ptr<simaai::neat::Node>& node,
                              const std::shared_ptr<simaai::neat::graph::Node>& runtime_node,
                              std::size_t index) {
  if (node) {
    return public_node_label(node, index);
  }
  if (runtime_node) {
    std::string label = runtime_node->user_label();
    if (label.empty()) {
      label = runtime_node->kind();
    }
    if (!label.empty()) {
      return label;
    }
  }
  return "node" + std::to_string(index);
}

std::string public_node_kind(const std::shared_ptr<simaai::neat::Node>& node,
                             const std::shared_ptr<simaai::neat::graph::Node>& runtime_node) {
  if (node) {
    return node->kind();
  }
  if (runtime_node) {
    return runtime_node->kind();
  }
  return "Unknown";
}

template <typename Edge> std::string public_edge_kind_name(const Edge& edge) {
  if (is_implicit_composition_edge(edge)) {
    return "implicit_linear";
  }
  if (is_public_endpoint_edge(edge)) {
    return "public_endpoint";
  }
  if (is_runtime_port_edge(edge)) {
    return "runtime_port";
  }
  return "unknown";
}

std::vector<std::size_t> runtime_edge_path(std::span<const EdgePlan> edges, graph::NodeId from,
                                           graph::NodeId to) {
  if (from == graph::kInvalidNode || to == graph::kInvalidNode || from == to) {
    return {};
  }

  struct Prev {
    graph::NodeId node = graph::kInvalidNode;
    std::size_t edge = static_cast<std::size_t>(-1);
  };

  std::unordered_map<graph::NodeId, std::vector<std::pair<graph::NodeId, std::size_t>>> adj;
  for (std::size_t i = 0; i < edges.size(); ++i) {
    adj[edges[i].from].push_back({edges[i].to, i});
  }

  std::unordered_map<graph::NodeId, Prev> prev;
  std::vector<graph::NodeId> queue;
  queue.push_back(from);
  prev.emplace(from, Prev{});
  for (std::size_t head = 0; head < queue.size(); ++head) {
    const graph::NodeId cur = queue[head];
    auto it = adj.find(cur);
    if (it == adj.end()) {
      continue;
    }
    for (const auto& [next, edge_index] : it->second) {
      if (prev.find(next) != prev.end()) {
        continue;
      }
      prev.emplace(next, Prev{.node = cur, .edge = edge_index});
      if (next == to) {
        std::vector<std::size_t> path;
        graph::NodeId walk = to;
        while (walk != from) {
          const auto pit = prev.find(walk);
          if (pit == prev.end() || pit->second.edge == static_cast<std::size_t>(-1)) {
            return {};
          }
          path.push_back(pit->second.edge);
          walk = pit->second.node;
        }
        std::reverse(path.begin(), path.end());
        return path;
      }
      queue.push_back(next);
    }
  }
  return {};
}

template <typename View>
void attach_public_graph_view(const View& view,
                              const std::vector<graph::NodeId>& runtime_node_for_vertex,
                              ExecutionGraphPlan* plan) {
  if (!plan) {
    return;
  }

  plan->public_nodes.clear();
  plan->public_nodes.reserve(view.vertices.size());
  for (std::size_t i = 0; i < view.vertices.size(); ++i) {
    const auto& node = view.vertices[i];
    const auto runtime_node = i < view.runtime_vertices.size() ? view.runtime_vertices[i] : nullptr;
    PublicGraphNodePlan n;
    n.id = i;
    n.kind = public_node_kind(node, runtime_node);
    n.label = public_node_label(node, runtime_node, i);
    n.endpoint_name = explicit_public_endpoint_name(node);
    n.input_endpoint = dynamic_cast<const simaai::neat::Input*>(node.get()) != nullptr;
    n.output_endpoint = dynamic_cast<const simaai::neat::Output*>(node.get()) != nullptr;
    n.public_node = node;
    if (i < runtime_node_for_vertex.size()) {
      n.runtime_node = runtime_node_for_vertex[i];
    }
    plan->public_nodes.push_back(std::move(n));
  }

  plan->public_edges.clear();
  plan->public_edges.reserve(view.edges.size());
  for (std::size_t i = 0; i < view.edges.size(); ++i) {
    const auto& edge = view.edges[i];
    PublicGraphEdgePlan e;
    e.id = i;
    e.from = edge.from;
    e.to = edge.to;
    e.kind = public_edge_kind_name(edge);
    if (edge.endpoint.has_value()) {
      e.from_endpoint = edge.endpoint->from_endpoint;
      e.to_endpoint = edge.endpoint->to_endpoint;
    } else {
      e.from_endpoint = edge.from_port;
      e.to_endpoint = edge.to_port;
    }
    if (edge.from < runtime_node_for_vertex.size()) {
      e.runtime_from = runtime_node_for_vertex[edge.from];
    }
    if (edge.to < runtime_node_for_vertex.size()) {
      e.runtime_to = runtime_node_for_vertex[edge.to];
    }
    e.runtime_edge_indices = runtime_edge_path(plan->edges, e.runtime_from, e.runtime_to);
    plan->public_edges.push_back(std::move(e));
  }
}

void normalize_public_graph_boundaries(const graph::Graph& graph, ExecutionGraphPlan* plan) {
  if (!plan) {
    return;
  }

  for (auto& segment : plan->pipeline_segments) {
    const bool is_graph_sink =
        !segment.node_ids.empty() && graph.out_degree(segment.node_ids.back()) == 0U;
    const bool has_explicit_input = nodes_have_input_appsrc(segment.nodes);
    const bool has_explicit_output = nodes_have_output_appsink(segment.nodes);

    segment.boundary.needs_input =
        !segment.boundary.source_like && !segment.boundary.direct_graph_source &&
        !segment.boundary.direct_graph_sink && (!segment.input_edges.empty() || has_explicit_input);
    segment.boundary.needs_output = !segment.boundary.direct_graph_sink &&
                                    !segment.boundary.direct_graph_source &&
                                    (!segment.output_edges.empty() || has_explicit_output);
    segment.boundary.terminal_output =
        is_graph_sink && (segment.boundary.direct_graph_sink || has_explicit_output);
    segment.boundary.graph_internal_output =
        !segment.boundary.direct_graph_source && !segment.output_edges.empty();
  }

  resolve_default_endpoints(graph, plan);
}

struct NamedEndpointCandidate {
  Endpoint endpoint;
  std::size_t vertex = 0;
  std::string explicit_name;
};

std::unordered_map<graph::NodeId, std::pair<std::size_t, std::size_t>>
rebase_runtime_ranges_to_original_vertices(
    const std::unordered_map<graph::NodeId, std::pair<std::size_t, std::size_t>>& normalized_ranges,
    const NormalizedPublicView& normalized) {
  std::unordered_map<graph::NodeId, std::pair<std::size_t, std::size_t>> out;
  for (const auto& [runtime_node, range] : normalized_ranges) {
    std::size_t start = std::numeric_limits<std::size_t>::max();
    std::size_t end = 0;
    for (std::size_t i = range.first; i < range.second && i < normalized.original_for_vertex.size();
         ++i) {
      const std::size_t original = normalized.original_for_vertex[i];
      start = std::min(start, original);
      end = std::max(end, original + 1U);
    }
    if (start != std::numeric_limits<std::size_t>::max() && end > start) {
      out.emplace(runtime_node, std::pair<std::size_t, std::size_t>{start, end});
    }
  }
  return out;
}

std::vector<graph::NodeId> rebase_runtime_nodes_to_original_vertices(
    const std::vector<graph::NodeId>& runtime_node_for_normalized_vertex,
    const NormalizedPublicView& normalized, std::size_t original_vertex_count) {
  std::vector<graph::NodeId> out(original_vertex_count, graph::kInvalidNode);
  for (std::size_t normalized_vertex = 0;
       normalized_vertex < runtime_node_for_normalized_vertex.size() &&
       normalized_vertex < normalized.original_for_vertex.size();
       ++normalized_vertex) {
    const std::size_t original = normalized.original_for_vertex[normalized_vertex];
    if (original < out.size()) {
      out[original] = runtime_node_for_normalized_vertex[normalized_vertex];
    }
  }
  return out;
}

template <typename Vertices>
void add_fragment_named_candidates(std::unordered_map<std::string, Endpoint>* named,
                                   std::unordered_set<std::string>* used,
                                   const std::vector<NamedEndpointCandidate>& candidates,
                                   const Vertices& vertices, std::string base_name,
                                   bool strict_single_fragment_name, const char* kind) {
  if (!named || !used || candidates.empty()) {
    return;
  }
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const auto& candidate = candidates[i];
    std::string name = candidate.explicit_name;
    bool strict = !name.empty();
    if (name.empty() && !base_name.empty()) {
      name = candidates.size() > 1U ? base_name + "_" + std::to_string(i) : base_name;
      strict = strict_single_fragment_name && candidates.size() == 1U;
    }
    if (name.empty() && candidate.vertex < vertices.size() && vertices[candidate.vertex]) {
      const std::string kind_name = vertices[candidate.vertex]->kind();
      if (kind_name == "Input") {
        name = "input";
      } else if (kind_name == "Output") {
        name = "output";
      } else {
        name = kind_name.empty() ? std::string(kind) : kind_name;
      }
    }
    add_named_endpoint(named, used, std::move(name), strict, candidate.endpoint, kind);
  }
}

template <typename NamedFragments>
void map_named_public_endpoints(
    const std::vector<graph::NodeId>& runtime_node_for_vertex,
    const std::unordered_map<graph::NodeId, std::pair<std::size_t, std::size_t>>&
        graph_range_by_node,
    std::span<const std::shared_ptr<simaai::neat::Node>> vertices, const NamedFragments& fragments,
    ExecutionGraphPlan* plan) {
  if (!plan) {
    return;
  }

  std::unordered_set<std::string> used_inputs;
  std::unordered_set<std::string> used_outputs;
  const auto add_pass = [&](bool user_named_pass) {
    for (const auto& fragment : fragments) {
      if (fragment.user_named != user_named_pass || fragment.name.empty() ||
          fragment.end <= fragment.start || fragment.end > runtime_node_for_vertex.size()) {
        continue;
      }

      std::vector<NamedEndpointCandidate> input_candidates;
      for (const auto& endpoint : plan->input_endpoints) {
        auto range = graph_range_by_node.find(endpoint.node);
        if (range == graph_range_by_node.end() || range->second.first < fragment.start ||
            range->second.first >= fragment.end) {
          continue;
        }
        const std::size_t vertex = range->second.first;
        input_candidates.push_back(NamedEndpointCandidate{
            .endpoint = endpoint,
            .vertex = vertex,
            .explicit_name = vertex < vertices.size()
                                 ? explicit_public_endpoint_name(vertices[vertex])
                                 : std::string{},
        });
      }
      add_fragment_named_candidates(&plan->named_inputs, &used_inputs, input_candidates, vertices,
                                    fragment.name, fragment.user_named, "input");

      std::vector<NamedEndpointCandidate> output_candidates;
      for (const auto& endpoint : plan->output_endpoints) {
        auto range = graph_range_by_node.find(endpoint.node);
        if (range == graph_range_by_node.end() || range->second.second <= fragment.start ||
            range->second.second > fragment.end) {
          continue;
        }
        const std::size_t vertex = range->second.second - 1U;
        output_candidates.push_back(NamedEndpointCandidate{
            .endpoint = endpoint,
            .vertex = vertex,
            .explicit_name = vertex < vertices.size()
                                 ? explicit_public_endpoint_name(vertices[vertex])
                                 : std::string{},
        });
      }
      add_fragment_named_candidates(&plan->named_outputs, &used_outputs, output_candidates,
                                    vertices, fragment.name, fragment.user_named, "output");
    }
  };
  add_pass(true);
  add_pass(false);
}

void add_named_endpoint_if_absent(std::unordered_map<std::string, Endpoint>* map, std::string name,
                                  const Endpoint& endpoint) {
  if (!map || name.empty()) {
    return;
  }
  map->emplace(std::move(name), endpoint);
}

void map_explicit_public_vertex_endpoints(
    const std::vector<graph::NodeId>& runtime_node_for_vertex,
    std::span<const std::shared_ptr<simaai::neat::Node>> vertices, ExecutionGraphPlan* plan) {
  if (!plan) {
    return;
  }
  for (std::size_t vertex = 0; vertex < vertices.size() && vertex < runtime_node_for_vertex.size();
       ++vertex) {
    const std::string name = explicit_public_endpoint_name(vertices[vertex]);
    if (name.empty()) {
      continue;
    }
    const graph::NodeId runtime_node = runtime_node_for_vertex[vertex];
    if (runtime_node == graph::kInvalidNode) {
      continue;
    }
    if (dynamic_cast<const simaai::neat::Input*>(vertices[vertex].get()) != nullptr) {
      for (const auto& endpoint : plan->input_endpoints) {
        if (endpoint.node == runtime_node) {
          add_named_endpoint_if_absent(&plan->named_inputs, name, endpoint);
          break;
        }
      }
    }
    if (dynamic_cast<const simaai::neat::Output*>(vertices[vertex].get()) != nullptr) {
      for (const auto& endpoint : plan->output_endpoints) {
        if (endpoint.node == runtime_node) {
          add_named_endpoint_if_absent(&plan->named_outputs, name, endpoint);
          break;
        }
      }
    }
  }
}

template <typename View>
void apply_public_fragment_metadata(
    const View& view,
    const std::unordered_map<graph::NodeId, std::pair<std::size_t, std::size_t>>&
        graph_range_by_node,
    ExecutionGraphPlan* plan) {
  if (!plan) {
    return;
  }

  plan->fragments.assign(view.fragments.begin(), view.fragments.end());

  for (auto& segment : plan->pipeline_segments) {
    std::size_t start = std::numeric_limits<std::size_t>::max();
    std::size_t end = 0;
    for (graph::NodeId node_id : segment.node_ids) {
      auto it = graph_range_by_node.find(node_id);
      if (it == graph_range_by_node.end()) {
        continue;
      }
      start = std::min(start, it->second.first);
      end = std::max(end, it->second.second);
    }
    if (start == std::numeric_limits<std::size_t>::max() || end <= start) {
      continue;
    }

    for (auto& provenance : segment.provenance) {
      provenance.graph_start = start;
      provenance.graph_end = end;
    }

    const FragmentPlan* fragment = best_fragment_for_range(view.fragments, start, end);
    if (!fragment || !fragment->boundary_hints.has_value()) {
      continue;
    }

    segment.boundary_hints = *fragment->boundary_hints;
    if (segment.boundary.needs_input && !segment.boundary_hints->ingress_inputs.empty() &&
        !segment.input_complete) {
      const InputOptions& ingress = segment.boundary_hints->ingress_inputs.front();
      segment.input_spec = input_options_to_output_spec(ingress);
      segment.input_complete = input_options_complete(ingress);
    }
  }
}

} // namespace

ExecutionGraphPlan build_execution_plan_from_compiled(const graph::Graph& graph,
                                                      const graph::CompiledGraph& compiled,
                                                      const RuntimeCompileOptions& opt) {
  const pipeline_internal::ScopedBuildTiming timing(
      "build_execution_plan_from_compiled",
      "nodes=" + std::to_string(graph.node_count()) +
          " pipelines=" + std::to_string(compiled.pipelines.size()) +
          " stages=" + std::to_string(compiled.stages.size()) +
          " edges=" + std::to_string(compiled.edges.size()));
  ExecutionGraphPlan plan;
  plan.port_names = compiled.port_names;
  plan.linear_compat = opt.linear_compat;
  plan.node_labels.reserve(graph.node_count());
  for (graph::NodeId id = 0; id < graph.node_count(); ++id) {
    plan.node_labels.push_back(graph_node_label(graph, id));
  }

  plan.pipeline_segments.reserve(compiled.pipelines.size());
  for (const auto& seg : compiled.pipelines) {
    PipelineSegmentPlan p;
    p.id = static_cast<std::size_t>(seg.id);
    p.node_ids = seg.node_ids;
    p.nodes = seg.nodes;
    p.input_edges = seg.input_edges;
    p.output_edges = seg.output_edges;
    p.boundary = decide_boundary_policy(graph, seg);
    p.input_spec = seg.input_spec;
    p.input_complete = seg.input_complete;
    p.output_spec = seg.output_spec;
    p.output_complete = seg.output_complete;
    p.run_options = opt.run_options;
    p.provenance.reserve(seg.node_ids.size());
    for (graph::NodeId node_id : seg.node_ids) {
      Provenance provenance;
      provenance.runtime_node = node_id;
      provenance.segment_id = p.id;
      p.provenance.push_back(std::move(provenance));
    }
    plan.pipeline_segments.push_back(std::move(p));
  }

  plan.stage_nodes.reserve(compiled.stages.size());
  for (const auto& stage : compiled.stages) {
    StageNodePlan s;
    s.node_id = stage.node_id;
    s.node = stage.node;
    s.provenance.runtime_node = stage.node_id;
    plan.stage_nodes.push_back(std::move(s));
  }

  plan.edges.reserve(compiled.edges.size());
  for (std::size_t i = 0; i < compiled.edges.size(); ++i) {
    const graph::Edge& edge = compiled.edges[i];
    EdgePlan p;
    p.from = edge.from;
    p.from_port = edge.from_port;
    p.to = edge.to;
    p.to_port = edge.to_port;
    if (i < compiled.edge_specs.size()) {
      p.spec = compiled.edge_specs[i].spec;
      p.spec_complete = compiled.edge_specs[i].complete;
    }
    plan.edges.push_back(std::move(p));
  }

  resolve_default_endpoints(graph, &plan);
  return plan;
}

ExecutionGraphPlan compile_public_graph(const simaai::neat::Graph& public_graph,
                                        const RunOptions& opt, std::optional<Sample> seed) {
  const auto total_start = pipeline_internal::build_timing_now();
  const auto view = public_graph.composition_view_for_internal_compile();
  (void)view.groups;

  ExecutionGraphPlan empty_plan;
  empty_plan.linear_compat = true;
  empty_plan.public_graph_id = view.graph_id;
  empty_plan.public_graph_version = view.graph_version;
  if (view.vertices.empty()) {
    return empty_plan;
  }

  std::optional<SampleSpec> seed_spec;
  if (seed.has_value()) {
    seed_spec = derive_sample_spec_or_throw(*seed);
  }

  auto prune_to_public_interface = [&](ExecutionGraphPlan* plan) {
    if (!plan) {
      return;
    }
    auto prune = [](auto& named, const std::vector<std::string>& keep_names) {
      const std::unordered_set<std::string> keep(keep_names.begin(), keep_names.end());
      for (auto it = named.begin(); it != named.end();) {
        if (keep.find(it->first) == keep.end()) {
          it = named.erase(it);
        } else {
          ++it;
        }
      }
    };
    prune(plan->named_inputs, public_graph.inputs());
    prune(plan->named_outputs, public_graph.outputs());
  };

  if (!view.linear) {
    const auto lower_start = pipeline_internal::build_timing_now();
    RuntimeCompileOptions compile_opt;
    compile_opt.run_options = opt;
    compile_opt.seed = std::move(seed);
    compile_opt.linear_compat = false;
    NormalizedPublicView normalized = normalize_public_boundaries_for_execution(view);
    PublicGraphLowering lowering =
        build_runtime_graph_from_connected_public_view(normalized, seed_spec);
    const auto graph_range_by_node =
        rebase_runtime_ranges_to_original_vertices(lowering.graph_range_by_node, normalized);
    const auto runtime_node_for_vertex = rebase_runtime_nodes_to_original_vertices(
        lowering.runtime_node_for_vertex, normalized, view.vertices.size());
    const auto lower_us = pipeline_internal::build_timing_us(lower_start);

    const auto compile_start = pipeline_internal::build_timing_now();
    ExecutionGraphPlan plan = compile_runtime_graph(lowering.graph, compile_opt);
    const auto compile_us = pipeline_internal::build_timing_us(compile_start);

    const auto metadata_start = pipeline_internal::build_timing_now();
    plan.public_graph_id = view.graph_id;
    plan.public_graph_version = view.graph_version;
    for (auto& segment : plan.pipeline_segments) {
      segment.route_options = view.options;
    }
    apply_public_fragment_metadata(view, graph_range_by_node, &plan);
    normalize_public_graph_boundaries(lowering.graph, &plan);
    map_named_public_endpoints(runtime_node_for_vertex, graph_range_by_node, view.vertices,
                               view.named_fragments, &plan);
    map_explicit_public_vertex_endpoints(runtime_node_for_vertex, view.vertices, &plan);
    prune_to_public_interface(&plan);
    attach_public_graph_view(view, runtime_node_for_vertex, &plan);
    const auto metadata_us = pipeline_internal::build_timing_us(metadata_start);
    pipeline_internal::emit_build_timing(
        "compile_public_graph",
        {{"lower", lower_us},
         {"runtime_compile", compile_us},
         {"metadata", metadata_us},
         {"total", pipeline_internal::build_timing_us(total_start)}},
        "mode=connected vertices=" + std::to_string(view.vertices.size()) +
            " edges=" + std::to_string(view.edges.size()) +
            " segments=" + std::to_string(plan.pipeline_segments.size()) +
            " stages=" + std::to_string(plan.stage_nodes.size()));
    return plan;
  }

  const auto lower_start = pipeline_internal::build_timing_now();
  std::vector<std::shared_ptr<simaai::neat::Node>> nodes = view.linear_nodes;
  std::optional<OutputSpec> root_input_spec;
  if (seed_spec.has_value()) {
    root_input_spec = output_spec_from_sample_spec(*seed_spec);
  }
  if (seed.has_value() && seed_spec.has_value()) {
    nodes = specialize_linear_model_preproc_route_for_seed(std::move(nodes), view.fragments, *seed,
                                                           *seed_spec);
  }
  const auto lower_us = pipeline_internal::build_timing_us(lower_start);

  const auto plan_start = pipeline_internal::build_timing_now();
  ExecutionGraphPlan plan =
      build_single_pipeline_execution_plan(std::move(nodes), opt, root_input_spec);
  const auto plan_us = pipeline_internal::build_timing_us(plan_start);

  const auto metadata_start = pipeline_internal::build_timing_now();
  plan.public_graph_id = view.graph_id;
  plan.public_graph_version = view.graph_version;
  for (auto& segment : plan.pipeline_segments) {
    segment.route_options = view.options;
  }
  std::unordered_map<graph::NodeId, std::pair<std::size_t, std::size_t>> graph_range_by_node;
  const graph::NodeId runtime_id = 0;
  graph_range_by_node[runtime_id] = {0U, view.vertices.size()};
  std::vector<graph::NodeId> runtime_node_for_vertex(view.vertices.size(), runtime_id);
  apply_public_fragment_metadata(view, graph_range_by_node, &plan);
  resolve_single_pipeline_endpoints(&plan);
  if (plan.default_input.has_value() && !view.vertices.empty()) {
    std::string name = explicit_public_endpoint_name(view.vertices.front());
    if (!name.empty()) {
      plan.named_inputs.emplace(std::move(name), *plan.default_input);
    }
  }
  if (plan.default_output.has_value() && !view.vertices.empty()) {
    std::string name = explicit_public_endpoint_name(view.vertices.back());
    if (!name.empty()) {
      plan.named_outputs.emplace(std::move(name), *plan.default_output);
    }
  }
  prune_to_public_interface(&plan);
  attach_public_graph_view(view, runtime_node_for_vertex, &plan);
  const auto metadata_us = pipeline_internal::build_timing_us(metadata_start);
  pipeline_internal::emit_build_timing(
      "compile_public_graph",
      {{"lower", lower_us},
       {"plan", plan_us},
       {"metadata", metadata_us},
       {"total", pipeline_internal::build_timing_us(total_start)}},
      "mode=linear vertices=" + std::to_string(view.vertices.size()) +
          " segments=" + std::to_string(plan.pipeline_segments.size()));
  return plan;
}

ExecutionGraphPlan compile_runtime_graph(const graph::Graph& graph,
                                         const RuntimeCompileOptions& opt) {
  const pipeline_internal::ScopedBuildTiming timing(
      "compile_runtime_graph", "nodes=" + std::to_string(graph.node_count()) +
                                   " edges=" + std::to_string(graph.edges().size()));
  graph::Compiler compiler;
  graph::CompilerOptions compiler_opt;
  if (opt.root_input_spec.has_value()) {
    for (graph::NodeId id = 0; id < graph.node_count(); ++id) {
      const auto& node = graph.node(id);
      if (!node || graph.in_degree(id) != 0U || node->input_ports().size() != 1U) {
        continue;
      }
      compiler_opt.root_input_specs.emplace(id, *opt.root_input_spec);
    }
  }
  graph::CompiledGraph compiled = compiler.compile(graph, compiler_opt);
  return build_execution_plan_from_compiled(graph, compiled, opt);
}

ExecutionGraphPlan compile_graph_run_plan(const graph::Graph& graph,
                                          const graph::GraphRunOptions& opt) {
  RuntimeCompileOptions compile_opt;
  compile_opt.run_options = opt.pipeline;
  compile_opt.linear_compat = false;
  return compile_runtime_graph(graph, compile_opt);
}

} // namespace simaai::neat::runtime

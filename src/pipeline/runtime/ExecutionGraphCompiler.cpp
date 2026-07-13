#include "ExecutionGraphPlan.h"

#include "graph/Compiler.h"
#include "graph/Graph.h"
#include "graph/GraphRun.h"
#include "graph/nodes/FanOut.h"
#include "graph/nodes/JoinBundle.h"
#include "graph/nodes/PipelineNode.h"
#include "graph/nodes/StageNode.h"
#include "graph/nodes/StreamScheduler.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "pipeline/internal/BuildTiming.h"
#include "pipeline/internal/CapsStringUtil.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/InputPolicy.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <span>
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
  RealtimeGraphLinkOptions link_options;
  std::string stream_id;
  CombinePolicy combine_policy = CombinePolicy::None;
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

std::string normalized_edge_stream_id(const NormalizedCompositionEdge& edge);

struct LoweredExplicitEdge {
  graph::NodeId from = graph::kInvalidNode;
  graph::NodeId to = graph::kInvalidNode;
  std::string from_port;
  std::string to_port;
  RealtimeGraphLinkOptions link_options;
  std::string stream_id;
};

struct PublicGraphLowering {
  graph::Graph graph;
  std::unordered_map<graph::NodeId, std::pair<std::size_t, std::size_t>> graph_range_by_node;
  std::vector<graph::NodeId> runtime_node_for_vertex;
  std::vector<LoweredExplicitEdge> lowered_edges;
};

bool realtime_latest_link(const RealtimeGraphLinkOptions& opt) {
  return opt.policy == GraphLinkPolicy::RealtimeLatestByStream ||
         opt.policy == GraphLinkPolicy::RealtimeEveryFrameByStream;
}

bool default_link(const RealtimeGraphLinkOptions& opt) {
  return opt.policy == GraphLinkPolicy::Default;
}

void validate_realtime_inflight_option(const char* name, int value) {
  if (value == 0 || value < -1) {
    throw std::runtime_error(std::string("RealtimeGraphLinkOptions::") + name +
                             " must be -1 or a positive value");
  }
}

void validate_non_default_link_options(const RealtimeGraphLinkOptions& opt) {
  if (default_link(opt)) {
    return;
  }
  validate_realtime_inflight_option("max_inflight_per_stream", opt.max_inflight_per_stream);
  validate_realtime_inflight_option("max_inflight_total", opt.max_inflight_total);
}

int merge_inflight_cap(int existing, int incoming) {
  if (existing == -1) {
    return incoming;
  }
  if (incoming == -1) {
    return existing;
  }
  return std::min(existing, incoming);
}

RealtimeGraphLinkOptions merge_link_options(RealtimeGraphLinkOptions a,
                                            const RealtimeGraphLinkOptions& b) {
  validate_non_default_link_options(a);
  validate_non_default_link_options(b);

  if (default_link(a)) {
    return b;
  }
  if (default_link(b)) {
    return a;
  }
  if (a.policy != b.policy) {
    const bool compatible_realtime = realtime_latest_link(a) && realtime_latest_link(b);
    if (!compatible_realtime) {
      throw std::runtime_error("compile_public_graph: conflicting Graph link policies");
    }
    throw std::runtime_error("compile_public_graph: cannot mix RealtimeLatestByStream and "
                             "RealtimeEveryFrameByStream on one runtime path");
  }
  if (b.queue_depth > 0) {
    a.queue_depth = b.queue_depth;
  }
  a.max_inflight_per_stream =
      merge_inflight_cap(a.max_inflight_per_stream, b.max_inflight_per_stream);
  a.max_inflight_total = merge_inflight_cap(a.max_inflight_total, b.max_inflight_total);
  if (!b.stream_id.empty()) {
    a.stream_id = b.stream_id;
  }
  return a;
}

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
  case CombinePolicy::RoundRobin:
    throw std::runtime_error(
        "compile_public_graph: CombinePolicy::RoundRobin is lowered through StreamScheduler, "
        "not JoinBundle");
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
                            std::string to_endpoint, RealtimeGraphLinkOptions link_options,
                            std::string stream_id, CombinePolicy combine_policy) {
    if (from == NormalizedPublicView::kInvalid || to == NormalizedPublicView::kInvalid) {
      return;
    }
    if (from == to) {
      return;
    }
    const std::string key = std::to_string(from) + ":" + std::to_string(to) + ":" +
                            std::to_string(static_cast<int>(kind)) + ":" + from_port + ":" +
                            to_port + ":" + from_endpoint + ":" + to_endpoint + ":" +
                            std::to_string(static_cast<int>(link_options.policy)) + ":" +
                            stream_id + ":" + std::to_string(static_cast<int>(combine_policy));
    if (!emitted_edges.insert(key).second) {
      return;
    }
    out.edges.push_back(NormalizedCompositionEdge{.from = from,
                                                  .to = to,
                                                  .kind = kind,
                                                  .from_port = std::move(from_port),
                                                  .to_port = std::move(to_port),
                                                  .from_endpoint = std::move(from_endpoint),
                                                  .to_endpoint = std::move(to_endpoint),
                                                  .link_options = link_options,
                                                  .stream_id = std::move(stream_id),
                                                  .combine_policy = combine_policy});
  };

  std::function<void(std::size_t, std::size_t, std::string, std::string, std::string, bool,
                     RealtimeGraphLinkOptions, std::string, CombinePolicy, std::vector<bool>&)>
      follow_edge;
  follow_edge = [&](std::size_t start_norm, std::size_t edge_index, std::string from_port,
                    std::string from_endpoint, std::string to_endpoint, bool bypassed_boundary,
                    RealtimeGraphLinkOptions link_options, std::string stream_id,
                    CombinePolicy combine_policy, std::vector<bool>& visiting) {
    const auto& edge = view.edges[edge_index];
    link_options = merge_link_options(link_options, edge.link_options);
    if (!edge.stream_id.empty()) {
      stream_id = edge.stream_id;
    } else if (!edge.link_options.stream_id.empty()) {
      stream_id = edge.link_options.stream_id;
    }
    if (!from_port.empty() && !edge.from_port.empty() && from_port != edge.from_port) {
      throw std::runtime_error(
          "compile_public_graph: boundary normalization encountered conflicting source ports");
    }
    if (from_port.empty()) {
      from_port = edge.from_port;
    }
    if (edge.endpoint.has_value()) {
      // When an explicit public boundary is elided, the logical source endpoint for the
      // downstream edge becomes the boundary edge we are crossing (for example a Branch output or
      // Combine input), not the physical producer that originally entered the boundary chain.  If
      // we keep the oldest upstream endpoint, multi-source live graphs whose sources all end in
      // the same node kind (for example CapsRaw) collapse to duplicate fan-in names.
      if ((bypassed_boundary && combine_policy == CombinePolicy::None) || from_endpoint.empty()) {
        from_endpoint = edge.endpoint->from_endpoint;
      }
      if (!edge.endpoint->to_endpoint.empty()) {
        to_endpoint = edge.endpoint->to_endpoint;
      }
    }

    const std::size_t to_original = edge.to;
    if (elide[to_original]) {
      const CombinePolicy boundary_policy = output_combine_policy(view.vertices[to_original]);
      if (boundary_policy != CombinePolicy::None) {
        if (combine_policy != CombinePolicy::None && combine_policy != boundary_policy) {
          throw std::runtime_error(
              "compile_public_graph: conflicting CombinePolicy while lowering connected "
              "boundary");
        }
        combine_policy = boundary_policy;
      }
    }
    if (!elide[to_original]) {
      const std::size_t to_norm = out.vertex_for_original[to_original];
      NormalizedCompositionEdgeKind kind =
          normalized_kind_from_public_edge_kind(edge, bypassed_boundary);
      std::string to_port = edge.to_port;
      const bool implicit_became_runtime = bypassed_boundary &&
                                           is_implicit_composition_edge(edge) &&
                                           kind == NormalizedCompositionEdgeKind::RuntimePort;
      if (implicit_became_runtime) {
        if (from_port.empty()) {
          from_port = "out";
        }
        if (to_port.empty()) {
          to_port = "in";
        }
      }
      if (kind == NormalizedCompositionEdgeKind::ImplicitLinear) {
        from_port.clear();
        to_port.clear();
      }
      add_edge(start_norm, to_norm, kind, std::move(from_port), std::move(to_port),
               std::move(from_endpoint), std::move(to_endpoint), link_options, std::move(stream_id),
               combine_policy);
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
      follow_edge(start_norm, next_edge, from_port, from_endpoint, to_endpoint, true, link_options,
                  stream_id, combine_policy, visiting);
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
    follow_edge(from_norm, edge_index, edge.from_port, {}, {}, false, {}, {}, CombinePolicy::None,
                visiting);
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
    if (!default_link(edge.link_options)) {
      continue;
    }
    if (!edge.stream_id.empty() || !edge.link_options.stream_id.empty()) {
      // An explicit stream identity is edge semantics, even when the link uses
      // the default buffering policy.  Keep a runtime edge so the router can
      // stamp the requested stream_id instead of fusing the edge away.
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
  std::vector<std::size_t> realtime_in_degree(view.vertices.size(), 0U);
  std::vector<std::size_t> total_in_degree(view.vertices.size(), 0U);
  for (const auto& edge : view.edges) {
    if (edge.from >= view.vertices.size() || edge.to >= view.vertices.size()) {
      throw std::runtime_error("compile_public_graph: composition edge references invalid vertex");
    }
    if (is_explicit_composition_edge(edge)) {
      ++total_in_degree[edge.to];
      if (realtime_latest_link(edge.link_options)) {
        ++realtime_in_degree[edge.to];
      }
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
  std::vector<bool> realtime_fan_in_target(view.vertices.size(), false);
  for (std::size_t i = 0; i < view.vertices.size(); ++i) {
    realtime_fan_in_target[i] =
        total_in_degree[i] > 1U && realtime_in_degree[i] == total_in_degree[i];
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
    bool allow_realtime_fan_in = false;
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
      allow_realtime_fan_in = allow_realtime_fan_in || realtime_fan_in_target[cur];
      nodes.push_back(view.vertices[cur]);
      const std::size_t next = implicit_next[cur];
      if (next == static_cast<std::size_t>(-1)) {
        break;
      }
      cur = next;
    }

    auto pipeline_node = std::make_shared<graph::nodes::PipelineNode>(
        std::move(nodes), "fragment" + std::to_string(start), allow_realtime_fan_in ? 0 : 1);
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
  std::unordered_map<std::size_t, std::vector<std::size_t>> combine_inputs_by_target;
  for (std::size_t i = 0; i < view.edges.size(); ++i) {
    const auto& edge = view.edges[i];
    if (is_public_endpoint_edge(edge) || edge.combine_policy != CombinePolicy::None) {
      combine_inputs_by_target[edge.to].push_back(i);
    }
  }

  std::unordered_set<std::size_t> combine_handled_edges;
  for (const auto& [target_vertex, incoming_edges] : combine_inputs_by_target) {
    if (incoming_edges.size() <= 1U) {
      continue;
    }
    if (target_vertex >= view.vertices.size()) {
      throw std::runtime_error("compile_public_graph: public endpoint target out of range");
    }
    CombinePolicy policy = output_combine_policy(view.vertices[target_vertex]);
    for (const std::size_t edge_index : incoming_edges) {
      const CombinePolicy edge_policy = view.edges[edge_index].combine_policy;
      if (edge_policy == CombinePolicy::None) {
        continue;
      }
      if (policy != CombinePolicy::None && policy != edge_policy) {
        throw std::runtime_error(
            "compile_public_graph: conflicting CombinePolicy on public endpoint fan-in");
      }
      policy = edge_policy;
    }
    const FragmentBoundaryHints* target_hints =
        boundary_hints_for_normalized_target(view, target_vertex);
    const bool model_ingress_combine = policy == CombinePolicy::None && target_hints != nullptr &&
                                       target_hints->input_route_processor != nullptr &&
                                       target_hints->ingress_endpoint_names.size() > 1U;
    const bool realtime_fan_in =
        std::all_of(incoming_edges.begin(), incoming_edges.end(), [&](std::size_t edge_index) {
          return realtime_latest_link(view.edges[edge_index].link_options);
        });
    if (policy == CombinePolicy::None && realtime_fan_in && !model_ingress_combine) {
      continue;
    }
    if (policy == CombinePolicy::None && !model_ingress_combine) {
      throw std::runtime_error(
          "compile_public_graph: public endpoint has multiple producers; set an explicit "
          "CombinePolicy::ByFrame, CombinePolicy::ByPts, or CombinePolicy::RoundRobin");
    }

    std::vector<std::size_t> ordered_incoming_edges;
    ordered_incoming_edges.reserve(incoming_edges.size());
    std::vector<std::string> input_names;
    input_names.reserve(incoming_edges.size());
    std::unordered_set<std::string> used_input_names;
    if (model_ingress_combine) {
      std::unordered_set<std::string> seen_model_ingress_edges;
      for (const auto edge_index : incoming_edges) {
        const auto& endpoint = view.edges[edge_index].to_endpoint;
        if (!seen_model_ingress_edges.insert(endpoint).second) {
          throw std::runtime_error("compile_public_graph: duplicate Model ingress endpoint '" +
                                   endpoint +
                                   "'; connect exactly one producer per Model ingress endpoint");
        }
      }
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

    if (policy == CombinePolicy::RoundRobin && !model_ingress_combine) {
      simaai::neat::graph::nodes::StreamSchedulerOptions scheduler_opt;
      scheduler_opt.max_batch = 1;
      scheduler_opt.inputs = input_names;
      const graph::NodeId scheduler_node =
          out.graph.add(simaai::neat::graph::nodes::StreamSchedulerNode(
              scheduler_opt, "combine_rr_" + std::to_string(target_vertex), "in", "out"));

      for (std::size_t i = 0; i < ordered_incoming_edges.size(); ++i) {
        const auto& edge = view.edges[ordered_incoming_edges[i]];
        const graph::NodeId from = runtime_node_for_vertex[edge.from];
        if (from == graph::kInvalidNode) {
          throw std::runtime_error(
              "compile_public_graph: round-robin combine edge references unmapped source");
        }
        std::string stream_id = normalized_edge_stream_id(edge);
        lowered_edges.push_back(LoweredExplicitEdge{
            .from = from,
            .to = scheduler_node,
            .from_port = edge.from_port.empty() ? "out" : edge.from_port,
            .to_port = input_names[i],
            .link_options = edge.link_options,
            .stream_id = std::move(stream_id),
        });
        combine_handled_edges.insert(ordered_incoming_edges[i]);
      }

      const graph::NodeId to = runtime_node_for_vertex[target_vertex];
      if (to == graph::kInvalidNode) {
        throw std::runtime_error("compile_public_graph: round-robin combine target is unmapped");
      }
      lowered_edges.push_back(LoweredExplicitEdge{
          .from = scheduler_node,
          .to = to,
          .from_port = "out",
          .to_port = "in",
      });
      continue;
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
          .link_options = edge.link_options,
          .stream_id = normalized_edge_stream_id(edge),
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
    if (is_public_endpoint_edge(edge) && public_endpoint_in_degree[edge.to] > 1U &&
        !realtime_latest_link(edge.link_options)) {
      throw std::runtime_error(
          "compile_public_graph: public endpoint has multiple producers; set an explicit "
          "CombinePolicy::ByFrame, CombinePolicy::ByPts, or CombinePolicy::RoundRobin");
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
        .link_options = edge.link_options,
        .stream_id = normalized_edge_stream_id(edge),
    });
  }

  connect_lowered_explicit_edges(&out.graph, lowered_edges);
  out.lowered_edges = std::move(lowered_edges);
  out.runtime_node_for_vertex = std::move(runtime_node_for_vertex);
  return out;
}

OutputSpec input_options_to_output_spec(const InputOptions& opt) {
  OutputSpec spec;
  const std::string caps_media = pipeline_internal::caps_media_type(opt.caps_override);
  const PayloadType caps_payload =
      opt.caps_override.empty()
          ? PayloadType::Auto
          : pipeline_internal::payload_type_from_caps_string(opt.caps_override);
  spec.media_type = !caps_media.empty() ? caps_media : resolve_input_media_type(opt);
  spec.payload_type = opt.payload_type != PayloadType::Auto ? opt.payload_type : caps_payload;
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
  const std::string caps_media = pipeline_internal::caps_media_type(opt.caps_override);
  const PayloadType caps_payload =
      opt.caps_override.empty()
          ? PayloadType::Auto
          : pipeline_internal::payload_type_from_caps_string(opt.caps_override);
  const std::string media_type = !caps_media.empty() ? caps_media : resolve_input_media_type(opt);
  if (media_type.empty()) {
    return false;
  }
  if (!opt.caps_override.empty() && caps_payload == PayloadType::Encoded) {
    return true;
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

bool same_endpoint(const Endpoint& a, const Endpoint& b) {
  return a.kind == b.kind && a.node == b.node && a.port == b.port && a.segment == b.segment;
}

template <typename Map>
void add_named_endpoint(Map* map, std::unordered_set<std::string>* used, std::string name,
                        bool user_named, const Endpoint& endpoint, const char* kind) {
  if (!map || !used || name.empty()) {
    return;
  }
  if (user_named) {
    if (used->find(name) != used->end()) {
      auto existing = map->find(name);
      if (existing != map->end() && same_endpoint(existing->second, endpoint)) {
        return;
      }
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

bool is_generated_fanout_node(const ExecutionGraphPlan& plan, graph::NodeId node) {
  return node < plan.node_labels.size() && plan.node_labels[node].rfind("fanout", 0) == 0;
}

std::string normalized_edge_stream_id(const NormalizedCompositionEdge& edge) {
  if (!edge.stream_id.empty()) {
    return edge.stream_id;
  }
  return edge.link_options.stream_id;
}

template <typename Edges>
void validate_unique_source_buffer_names(
    std::span<const std::shared_ptr<simaai::neat::Node>> vertices, const Edges& edges) {
  struct SourceIdentity {
    std::size_t vertex = 0;
    std::string label;
  };
  std::unordered_map<std::string, std::vector<SourceIdentity>> sources_by_buffer_name;
  for (std::size_t i = 0; i < vertices.size(); ++i) {
    const auto& node = vertices[i];
    if (!node || node->input_role() != InputRole::Source) {
      continue;
    }
    const std::string buffer_name = node->buffer_name_hint(static_cast<int>(i));
    if (buffer_name.empty()) {
      continue;
    }
    std::string label = node->user_label();
    if (label.empty()) {
      label = node->kind() + "@" + std::to_string(i);
    }
    sources_by_buffer_name[buffer_name].push_back(SourceIdentity{.vertex = i, .label = label});
  }

  std::vector<std::vector<std::size_t>> outgoing(vertices.size());
  for (const auto& edge : edges) {
    if (edge.from < outgoing.size() && edge.to < outgoing.size()) {
      outgoing[edge.from].push_back(edge.to);
    }
  }
  const auto reachable_from = [&](std::size_t source) {
    std::vector<bool> reachable(vertices.size(), false);
    std::vector<std::size_t> pending{source};
    while (!pending.empty()) {
      const std::size_t current = pending.back();
      pending.pop_back();
      for (const std::size_t next : outgoing[current]) {
        if (!reachable[next]) {
          reachable[next] = true;
          pending.push_back(next);
        }
      }
    }
    return reachable;
  };

  for (const auto& [buffer_name, sources] : sources_by_buffer_name) {
    std::vector<std::vector<bool>> reachable;
    reachable.reserve(sources.size());
    for (const auto& source : sources) {
      reachable.push_back(reachable_from(source.vertex));
    }
    for (std::size_t left = 0; left < sources.size(); ++left) {
      for (std::size_t right = left + 1U; right < sources.size(); ++right) {
        bool converges = false;
        for (std::size_t vertex = 0; vertex < vertices.size(); ++vertex) {
          if (reachable[left][vertex] && reachable[right][vertex]) {
            converges = true;
            break;
          }
        }
        if (!converges) {
          continue;
        }
        throw std::runtime_error(
            "compile_public_graph: duplicate source buffer_name '" + buffer_name + "' for '" +
            sources[left].label + "' and '" + sources[right].label +
            "' on converging graph paths. Multi-source graphs must stamp a unique buffer_name "
            "before Branch/FanOut so downstream CVU/MLA/preprocess metadata is unambiguous.");
      }
    }
  }
}

void apply_link_options_to_runtime_path(ExecutionGraphPlan* plan, graph::NodeId from,
                                        graph::NodeId to,
                                        const RealtimeGraphLinkOptions& link_options,
                                        const std::string& stream_id) {
  const bool propagate_policy = !default_link(link_options);
  const bool propagate_stream_id = !stream_id.empty();
  if (!plan || (!propagate_policy && !propagate_stream_id)) {
    return;
  }

  const auto path = runtime_edge_path(plan->edges, from, to);
  std::size_t first_policy_edge = 0U;
  if (path.size() > 1U) {
    const auto first = path.front();
    if (first < plan->edges.size() && is_generated_fanout_node(*plan, plan->edges[first].to)) {
      // A public/lowered edge may lower through a generated FanOut when the
      // same producer also feeds another branch. Keep branch-specific
      // realtime/drop policy and stream identity on the selected branch edge;
      // applying either to the shared producer->FanOut trunk would change
      // unrelated default branches.
      first_policy_edge = 1U;
    }
  }
  for (std::size_t path_pos = first_policy_edge; path_pos < path.size(); ++path_pos) {
    const std::size_t edge_index = path[path_pos];
    if (edge_index >= plan->edges.size()) {
      continue;
    }
    RealtimeGraphLinkOptions& dst = plan->edges[edge_index].link_options;
    if (propagate_policy) {
      dst = merge_link_options(dst, link_options);
    }
    if (propagate_stream_id) {
      plan->edges[edge_index].stream_id = stream_id;
      dst.stream_id = stream_id;
    }
  }
}

void apply_lowered_link_policies(const std::vector<LoweredExplicitEdge>& edges,
                                 ExecutionGraphPlan* plan) {
  if (!plan) {
    return;
  }
  for (const auto& edge : edges) {
    const std::string stream_id =
        !edge.stream_id.empty() ? edge.stream_id : edge.link_options.stream_id;
    apply_link_options_to_runtime_path(plan, edge.from, edge.to, edge.link_options, stream_id);
  }
}

void apply_normalized_link_policies(const NormalizedPublicView& view,
                                    const std::vector<graph::NodeId>& runtime_node_for_vertex,
                                    ExecutionGraphPlan* plan) {
  if (!plan) {
    return;
  }
  for (const auto& edge : view.edges) {
    if (edge.combine_policy != CombinePolicy::None) {
      continue;
    }
    const std::string stream_id = normalized_edge_stream_id(edge);
    if (edge.from >= runtime_node_for_vertex.size() || edge.to >= runtime_node_for_vertex.size()) {
      continue;
    }
    apply_link_options_to_runtime_path(plan, runtime_node_for_vertex[edge.from],
                                       runtime_node_for_vertex[edge.to], edge.link_options,
                                       stream_id);
  }
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
    e.link_options = edge.link_options;
    e.stream_id = edge.stream_id;
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

bool segment_has_output_edge(const PipelineSegmentPlan& segment, std::size_t edge_index) {
  return std::find(segment.output_edges.begin(), segment.output_edges.end(), edge_index) !=
         segment.output_edges.end();
}

bool segment_is_private_live_source_for_fusion(const PipelineSegmentPlan& segment,
                                               std::size_t edge_index) {
  // A previously fused consumer is source-like and inputless, but its nested
  // source branches live in fused_realtime_ingress rather than nodes.  Until
  // recursive fusion explicitly flattens/preserves those branches, treating
  // that segment as a leaf source would silently discard the original ingress.
  if (segment.consumed_by_fused_realtime_ingress || segment.nodes.empty() ||
      segment.fused_realtime_ingress.has_value()) {
    return false;
  }
  if (!segment.boundary.source_like || !segment.input_edges.empty()) {
    return false;
  }
  if (!segment_has_output_edge(segment, edge_index)) {
    return false;
  }
  // Keep the first implementation deliberately surgical: only fuse a source segment
  // whose sole graph responsibility is feeding this live fan-in.  More complex shared
  // source fan-out should continue through explicit graph transport until it has its
  // own no-copy route plan.
  return segment.output_edges.size() == 1U;
}

bool fused_realtime_input_edges_share_destination(const ExecutionGraphPlan& plan,
                                                  const std::vector<std::size_t>& input_edges) {
  if (input_edges.empty()) {
    return false;
  }
  std::optional<std::pair<graph::NodeId, graph::PortId>> destination;
  for (const std::size_t edge_index : input_edges) {
    if (edge_index >= plan.edges.size()) {
      return false;
    }
    const auto& edge = plan.edges[edge_index];
    const auto current = std::pair{edge.to, edge.to_port};
    if (!destination.has_value()) {
      destination = current;
    } else if (*destination != current) {
      return false;
    }
  }
  return true;
}

void fuse_realtime_fan_in_segments(const graph::Graph& graph, ExecutionGraphPlan* plan) {
  if (!plan || plan->pipeline_segments.empty() || plan->edges.empty()) {
    return;
  }

  std::unordered_map<graph::NodeId, std::size_t> segment_by_node;
  for (std::size_t i = 0; i < plan->pipeline_segments.size(); ++i) {
    for (graph::NodeId node : plan->pipeline_segments[i].node_ids) {
      segment_by_node[node] = i;
    }
  }

  for (std::size_t target_index = 0; target_index < plan->pipeline_segments.size();
       ++target_index) {
    auto& target = plan->pipeline_segments[target_index];
    if (target.input_edges.size() <= 1U || target.boundary.source_like ||
        target.fused_realtime_ingress.has_value()) {
      continue;
    }
    // One neatlatestbystreammux has one linear src pad. It can replace true
    // fan-in to a single consumer ingress, but not a multi-input segment whose
    // edges intentionally address distinct ports/endpoints.
    if (!fused_realtime_input_edges_share_destination(*plan, target.input_edges)) {
      continue;
    }

    bool all_realtime = true;
    std::vector<std::size_t> source_segment_indices;
    source_segment_indices.reserve(target.input_edges.size());
    for (const std::size_t edge_index : target.input_edges) {
      if (edge_index >= plan->edges.size()) {
        all_realtime = false;
        break;
      }
      const auto& edge = plan->edges[edge_index];
      if (!realtime_latest_link(edge.link_options)) {
        all_realtime = false;
        break;
      }
      auto it = segment_by_node.find(edge.from);
      if (it == segment_by_node.end() || it->second == target_index ||
          it->second >= plan->pipeline_segments.size()) {
        all_realtime = false;
        break;
      }
      const auto& source = plan->pipeline_segments[it->second];
      if (!segment_is_private_live_source_for_fusion(source, edge_index)) {
        all_realtime = false;
        break;
      }
      source_segment_indices.push_back(it->second);
    }
    if (!all_realtime || source_segment_indices.size() != target.input_edges.size()) {
      continue;
    }

    FusedRealtimeIngress fused;
    fused.branches.reserve(target.input_edges.size());
    for (std::size_t ordinal = 0; ordinal < target.input_edges.size(); ++ordinal) {
      const std::size_t edge_index = target.input_edges[ordinal];
      const auto& edge = plan->edges[edge_index];
      const auto& source = plan->pipeline_segments[source_segment_indices[ordinal]];

      FusedRealtimeIngressBranch branch;
      branch.edge_index = edge_index;
      branch.source_node = edge.from;
      branch.stream_id =
          edge.stream_id.empty() ? ("stream" + std::to_string(ordinal)) : edge.stream_id;
      branch.link_options = edge.link_options;
      branch.nodes = source.nodes;
      branch.output_spec = edge.spec_complete ? edge.spec : source.output_spec;
      branch.output_complete = edge.spec_complete || source.output_complete;
      fused.branches.push_back(std::move(branch));
    }

    for (const std::size_t edge_index : target.input_edges) {
      if (edge_index < plan->edges.size()) {
        plan->edges[edge_index].consumed_by_fused_realtime_ingress = true;
      }
    }
    for (const std::size_t source_index : source_segment_indices) {
      plan->pipeline_segments[source_index].consumed_by_fused_realtime_ingress = true;
    }

    target.fused_realtime_ingress = std::move(fused);
    target.input_edges.clear();
    target.boundary.source_like = true;
    target.boundary.needs_input = false;
    target.boundary.direct_graph_source = false;
    target.boundary.direct_graph_sink = false;
    target.boundary.needs_output =
        !target.boundary.direct_graph_sink &&
        (!target.output_edges.empty() || nodes_have_output_appsink(target.nodes));
    target.boundary.graph_internal_output = !target.output_edges.empty();
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
  std::vector<NamedEndpointCandidate> unique_candidates;
  unique_candidates.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    const bool duplicate =
        std::any_of(unique_candidates.begin(), unique_candidates.end(), [&](const auto& existing) {
          return same_endpoint(existing.endpoint, candidate.endpoint) &&
                 existing.explicit_name == candidate.explicit_name;
        });
    const bool duplicate_vertex_endpoint =
        std::any_of(unique_candidates.begin(), unique_candidates.end(), [&](const auto& existing) {
          return !candidate.explicit_name.empty() && existing.vertex == candidate.vertex &&
                 existing.explicit_name == candidate.explicit_name;
        });
    if (!duplicate && !duplicate_vertex_endpoint) {
      unique_candidates.push_back(candidate);
    }
  }
  for (std::size_t i = 0; i < unique_candidates.size(); ++i) {
    const auto& candidate = unique_candidates[i];
    std::string name = candidate.explicit_name;
    bool strict = !name.empty();
    if (name.empty() && !base_name.empty()) {
      name = unique_candidates.size() > 1U ? base_name + "_" + std::to_string(i) : base_name;
      strict = strict_single_fragment_name && unique_candidates.size() == 1U;
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

const InputOptions* ingress_options_for_segment_edge(const ExecutionGraphPlan& plan,
                                                     const PipelineSegmentPlan& segment,
                                                     std::size_t edge_index,
                                                     std::size_t ingress_index) {
  if (segment.boundary_hints.has_value() && !segment.boundary_hints->ingress_inputs.empty()) {
    const auto& hints = *segment.boundary_hints;
    if (edge_index < plan.edges.size()) {
      const graph::PortId to_port = plan.edges[edge_index].to_port;
      if (to_port != graph::kInvalidPort && to_port < plan.port_names.size()) {
        const std::string& port_name = plan.port_names[to_port];
        const auto match = std::find(hints.ingress_endpoint_names.begin(),
                                     hints.ingress_endpoint_names.end(), port_name);
        if (match != hints.ingress_endpoint_names.end()) {
          const std::size_t matched_index =
              static_cast<std::size_t>(match - hints.ingress_endpoint_names.begin());
          if (matched_index < hints.ingress_inputs.size()) {
            return &hints.ingress_inputs[matched_index];
          }
        }
      }
    }
    if (ingress_index < hints.ingress_inputs.size()) {
      return &hints.ingress_inputs[ingress_index];
    }
    return nullptr;
  }
  for (const auto& node : segment.nodes) {
    if (const auto* input = dynamic_cast<const simaai::neat::Input*>(node.get())) {
      return &input->options();
    }
  }
  return nullptr;
}

bool is_shape_passthrough_stage(const ExecutionGraphPlan& plan, graph::NodeId node_id) {
  for (const auto& stage : plan.stage_nodes) {
    if (stage.node_id != node_id || !stage.node) {
      continue;
    }
    const std::string kind = stage.node->kind();
    return kind == "FanOut" || kind == "StreamScheduler";
  }
  return false;
}

void collect_static_ingress_specs(const ExecutionGraphPlan& plan, std::size_t edge_index,
                                  std::unordered_set<std::size_t>* visited,
                                  std::vector<const OutputSpec*>* specs) {
  if (!visited || !specs || edge_index >= plan.edges.size() ||
      !visited->insert(edge_index).second) {
    return;
  }
  const EdgePlan& edge = plan.edges[edge_index];
  if (is_shape_passthrough_stage(plan, edge.from)) {
    for (std::size_t i = 0; i < plan.edges.size(); ++i) {
      if (plan.edges[i].to == edge.from) {
        collect_static_ingress_specs(plan, i, visited, specs);
      }
    }
    return;
  }
  if (edge.spec.width > 0 || edge.spec.height > 0 || edge.spec.depth > 0) {
    specs->push_back(&edge.spec);
  }
}

void validate_static_connected_input_capacities_impl(const ExecutionGraphPlan& plan) {
  for (const auto& segment : plan.pipeline_segments) {
    if (segment.input_edges.empty()) {
      continue;
    }

    const auto validate = [&](const char* dimension, int actual, int configured_max,
                              std::size_t ingress_index) {
      const int capacity = configured_max > 0 ? configured_max : -1;
      if (actual <= 0 || capacity <= 0 || actual <= capacity) {
        return;
      }
      const std::string where = "compile_public_graph: segment " + std::to_string(segment.id) +
                                " ingress " + std::to_string(ingress_index);
      throw std::invalid_argument(
          pipeline_internal::shape_limit_exceeded_message(where, dimension, actual, capacity) +
          ". Fix: " + pipeline_internal::shape_limit_fix_hint(dimension, actual));
    };

    for (std::size_t ingress_index = 0; ingress_index < segment.input_edges.size();
         ++ingress_index) {
      const std::size_t edge_index = segment.input_edges[ingress_index];
      const InputOptions* options =
          ingress_options_for_segment_edge(plan, segment, edge_index, ingress_index);
      if (!options) {
        continue;
      }
      std::unordered_set<std::size_t> visited;
      std::vector<const OutputSpec*> specs;
      collect_static_ingress_specs(plan, edge_index, &visited, &specs);
      for (const OutputSpec* spec : specs) {
        validate("width", spec->width, options->max_width, ingress_index);
        validate("height", spec->height, options->max_height, ingress_index);
        validate("depth", spec->depth, options->max_depth, ingress_index);
      }
    }
  }
}

} // namespace

namespace session_test {

bool fused_realtime_source_segment_eligible_for_test(bool already_fused) {
  PipelineSegmentPlan segment;
  segment.nodes.push_back(nullptr);
  segment.boundary.source_like = true;
  segment.output_edges.push_back(7U);
  if (already_fused) {
    segment.fused_realtime_ingress.emplace();
  }
  return segment_is_private_live_source_for_fusion(segment, 7U);
}

bool fused_realtime_destinations_share_port_for_test(
    const std::vector<std::pair<graph::NodeId, graph::PortId>>& destinations) {
  ExecutionGraphPlan plan;
  std::vector<std::size_t> input_edges;
  plan.edges.reserve(destinations.size());
  input_edges.reserve(destinations.size());
  for (const auto& [node, port] : destinations) {
    EdgePlan edge;
    edge.to = node;
    edge.to_port = port;
    input_edges.push_back(plan.edges.size());
    plan.edges.push_back(std::move(edge));
  }
  return fused_realtime_input_edges_share_destination(plan, input_edges);
}

} // namespace session_test

void validate_static_connected_input_capacities(const ExecutionGraphPlan& plan) {
  validate_static_connected_input_capacities_impl(plan);
}

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
                                        const RunOptions& opt, std::optional<Sample> seed,
                                        bool fuse_realtime_source_branches) {
  const auto total_start = pipeline_internal::build_timing_now();
  const auto view = public_graph.composition_view_for_internal_compile();
  (void)view.groups;
  validate_unique_source_buffer_names(view.vertices, view.edges);

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
    apply_lowered_link_policies(lowering.lowered_edges, &plan);
    apply_normalized_link_policies(normalized, lowering.runtime_node_for_vertex, &plan);
    apply_public_fragment_metadata(view, graph_range_by_node, &plan);
    validate_static_connected_input_capacities(plan);
    normalize_public_graph_boundaries(lowering.graph, &plan);
    if (fuse_realtime_source_branches) {
      fuse_realtime_fan_in_segments(lowering.graph, &plan);
    }
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
  const auto lower_us = pipeline_internal::build_timing_us(lower_start);

  const auto plan_start = pipeline_internal::build_timing_now();
  std::optional<OutputSpec> root_input_spec;
  if (seed_spec.has_value()) {
    root_input_spec = output_spec_from_sample_spec(*seed_spec);
  }
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

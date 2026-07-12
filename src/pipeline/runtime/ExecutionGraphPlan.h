#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "builder/Node.h"
#include "builder/OutputSpec.h"
#include "graph/GraphTypes.h"
#include "internal/InputStream.h"
#include "nodes/io/Input.h"
#include "pipeline/Run.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/internal/InputRouteProcessor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace simaai::neat::graph {
class Graph;
struct CompiledGraph;
struct GraphRunOptions;
namespace nodes {
class StageNode;
}
} // namespace simaai::neat::graph

namespace simaai::neat {
class Graph;
}

namespace simaai::neat::runtime {

struct ModelRouteOptionsProvenance {
  bool present = false;
  std::string upstream_name;
  std::string name_suffix;
  std::string buffer_name;
  std::string processcvu_requested_run_target = "AUTO";
  ProcessCvuOptions processcvu;
  ProcessMlaOptions processmla;
  PreparedRunnerOptions prepared_runner;
  int async_queue_depth = 0;
  bool expose_all_outputs = false;
};

struct Provenance {
  std::string user_label;
  std::string model_id;
  std::string model_source_path;
  std::string model_stage_role;
  std::string model_options_json;
  ModelRouteOptionsProvenance model_route;
  std::string mpk_plugin_id;
  std::size_t graph_start = 0;
  std::size_t graph_end = 0;
  graph::NodeId runtime_node = graph::kInvalidNode;
  std::size_t segment_id = static_cast<std::size_t>(-1);
};

struct FragmentBoundaryHints {
  std::vector<InputOptions> ingress_inputs;
  std::vector<std::string> ingress_endpoint_names;
  std::vector<std::string> egress_endpoint_names;
  pipeline_internal::InputRouteProcessorPtr input_route_processor;
  bool tensor_mode = false;
  bool bundled_fan_in = false;
};

struct FragmentPlan {
  std::size_t graph_start = 0;
  std::size_t graph_end = 0;
  std::optional<FragmentBoundaryHints> boundary_hints;
  Provenance provenance;
};

struct Endpoint {
  enum class Kind {
    PipelineInput,
    PipelineOutput,
    StageInput,
    StageOutput,
    GraphSink,
  };

  Kind kind = Kind::PipelineInput;
  graph::NodeId node = graph::kInvalidNode;
  graph::PortId port = graph::kInvalidPort;
  std::size_t segment = static_cast<std::size_t>(-1);
};

struct BoundaryPolicy {
  bool needs_input = false;
  bool needs_output = false;
  bool source_like = false;
  bool terminal_output = false;
  bool graph_internal_output = false;
  bool direct_graph_source = false;
  bool direct_graph_sink = false;
};

struct FusedRealtimeIngressBranch {
  std::size_t edge_index = static_cast<std::size_t>(-1);
  graph::NodeId source_node = graph::kInvalidNode;
  std::string stream_id;
  /// Exact public options from the source-to-consumer realtime link.  Fused
  /// lowering must retain these because there is no graph-runtime scheduler
  /// left outside the monolithic GStreamer pipeline to enforce them.
  GraphLinkOptions link_options;
  std::vector<std::shared_ptr<Node>> nodes;
  OutputSpec output_spec;
  bool output_complete = false;
};

struct FusedRealtimeIngress {
  std::vector<FusedRealtimeIngressBranch> branches;
};

struct MaterializedNodeAttribution {
  enum class Role {
    SegmentNode,
    InjectedInput,
    InjectedOutput,
  };

  std::size_t materialized_index = 0;
  std::size_t segment_node_index = static_cast<std::size_t>(-1);
  graph::NodeId runtime_node = graph::kInvalidNode;
  Role role = Role::SegmentNode;
};

struct PipelineSegmentPlan {
  std::size_t id = 0;
  std::vector<graph::NodeId> node_ids;
  std::vector<std::shared_ptr<Node>> nodes;

  std::vector<std::size_t> input_edges;
  std::vector<std::size_t> output_edges;

  BoundaryPolicy boundary;
  OutputSpec input_spec;
  bool input_complete = false;
  OutputSpec output_spec;
  bool output_complete = false;
  std::optional<FragmentBoundaryHints> boundary_hints;
  std::optional<FusedRealtimeIngress> fused_realtime_ingress;
  bool consumed_by_fused_realtime_ingress = false;

  GraphOptions route_options;
  RunOptions run_options;
  InputStreamOptions stream_options;
  std::vector<Provenance> provenance;
  std::vector<MaterializedNodeAttribution> materialized_node_attribution;
};

inline graph::NodeId attributed_runtime_node_for_segment_node(const PipelineSegmentPlan& segment,
                                                              std::size_t segment_node_index) {
  if (segment_node_index < segment.provenance.size() &&
      segment.provenance[segment_node_index].runtime_node != graph::kInvalidNode) {
    return segment.provenance[segment_node_index].runtime_node;
  }
  if (segment_node_index < segment.node_ids.size()) {
    return segment.node_ids[segment_node_index];
  }
  if (segment.node_ids.size() == 1U) {
    return segment.node_ids.front();
  }
  return graph::kInvalidNode;
}

inline std::vector<MaterializedNodeAttribution>
make_materialized_node_attribution(const PipelineSegmentPlan& segment, bool injected_input,
                                   bool injected_output) {
  std::vector<MaterializedNodeAttribution> out;
  out.reserve(segment.nodes.size() + (injected_input ? 1U : 0U) + (injected_output ? 1U : 0U));

  if (injected_input) {
    MaterializedNodeAttribution attr;
    attr.materialized_index = out.size();
    attr.role = MaterializedNodeAttribution::Role::InjectedInput;
    out.push_back(attr);
  }

  for (std::size_t i = 0; i < segment.nodes.size(); ++i) {
    MaterializedNodeAttribution attr;
    attr.materialized_index = out.size();
    attr.segment_node_index = i;
    attr.runtime_node = attributed_runtime_node_for_segment_node(segment, i);
    attr.role = MaterializedNodeAttribution::Role::SegmentNode;
    out.push_back(attr);
  }

  if (injected_output) {
    MaterializedNodeAttribution attr;
    attr.materialized_index = out.size();
    attr.role = MaterializedNodeAttribution::Role::InjectedOutput;
    out.push_back(attr);
  }

  return out;
}

struct StageNodePlan {
  graph::NodeId node_id = graph::kInvalidNode;
  std::shared_ptr<graph::nodes::StageNode> node;
  Provenance provenance;
};

struct EdgePlan {
  graph::NodeId from = graph::kInvalidNode;
  graph::PortId from_port = graph::kInvalidPort;
  graph::NodeId to = graph::kInvalidNode;
  graph::PortId to_port = graph::kInvalidPort;
  OutputSpec spec;
  bool spec_complete = false;
  GraphLinkOptions link_options;
  std::string stream_id;
  bool consumed_by_fused_realtime_ingress = false;
};

struct PublicGraphNodePlan {
  std::size_t id = 0;
  std::string kind;
  std::string label;
  std::string endpoint_name;
  bool input_endpoint = false;
  bool output_endpoint = false;
  graph::NodeId runtime_node = graph::kInvalidNode;
  std::shared_ptr<simaai::neat::Node> public_node;
};

struct PublicGraphEdgePlan {
  std::size_t id = 0;
  std::size_t from = 0;
  std::size_t to = 0;
  std::string kind;
  std::string from_endpoint;
  std::string to_endpoint;
  graph::NodeId runtime_from = graph::kInvalidNode;
  graph::NodeId runtime_to = graph::kInvalidNode;
  std::vector<std::size_t> runtime_edge_indices;
  GraphLinkOptions link_options;
  std::string stream_id;
};

struct ExecutionGraphPlan {
  std::vector<PipelineSegmentPlan> pipeline_segments;
  std::vector<StageNodePlan> stage_nodes;
  std::vector<EdgePlan> edges;
  std::vector<PublicGraphNodePlan> public_nodes;
  std::vector<PublicGraphEdgePlan> public_edges;
  std::vector<FragmentPlan> fragments;
  std::vector<std::string> port_names;
  std::vector<std::string> node_labels;

  std::optional<Endpoint> default_input;
  std::optional<Endpoint> default_output;
  std::vector<Endpoint> input_endpoints;
  std::vector<Endpoint> output_endpoints;
  std::unordered_map<std::string, Endpoint> named_inputs;
  std::unordered_map<std::string, Endpoint> named_outputs;
  std::uint64_t public_graph_id = 0;
  std::uint64_t public_graph_version = 0;
  bool linear_compat = false;
};

struct RuntimeCompileOptions {
  RunOptions run_options{};
  std::optional<Sample> seed;
  std::optional<OutputSpec> root_input_spec;
  bool linear_compat = false;
};

ExecutionGraphPlan compile_public_graph(const simaai::neat::Graph& graph, const RunOptions& opt,
                                        std::optional<Sample> seed,
                                        bool fuse_realtime_source_branches);

// Reject statically known connected-source shapes that exceed a downstream ingress capacity.
void validate_static_connected_input_capacities(const ExecutionGraphPlan& plan);

ExecutionGraphPlan build_execution_plan_from_compiled(const graph::Graph& graph,
                                                      const graph::CompiledGraph& compiled,
                                                      const RuntimeCompileOptions& opt);

ExecutionGraphPlan compile_runtime_graph(const graph::Graph& graph,
                                         const RuntimeCompileOptions& opt);

ExecutionGraphPlan compile_graph_run_plan(const graph::Graph& graph,
                                          const graph::GraphRunOptions& opt);

} // namespace simaai::neat::runtime

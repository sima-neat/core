#include "graph/Compiler.h"

#include "builder/OutputSpec.h"
#include "graph/nodes/PipelineNode.h"

#include <stdexcept>
#include <unordered_set>

namespace simaai::neat::graph {
namespace {

using PipelineNode = simaai::neat::graph::nodes::PipelineNode;
using StageNode = simaai::neat::graph::nodes::StageNode;

bool is_pipeline_node(const std::shared_ptr<Node>& n) {
  if (!n)
    return false;
  return n->backend() == Backend::Pipeline;
}

const PipelineNode* as_pipeline_node(const std::shared_ptr<Node>& n) {
  return dynamic_cast<const PipelineNode*>(n.get());
}

const StageNode* as_stage_node(const std::shared_ptr<Node>& n) {
  return dynamic_cast<const StageNode*>(n.get());
}

std::unordered_map<std::string, std::size_t> build_port_index(const std::vector<PortDesc>& ports) {
  std::unordered_map<std::string, std::size_t> out;
  for (std::size_t i = 0; i < ports.size(); ++i) {
    const auto& name = ports[i].name;
    if (name.empty()) {
      throw std::runtime_error("Compiler: empty port name");
    }
    if (out.find(name) != out.end()) {
      throw std::runtime_error("Compiler: duplicate port name: " + name);
    }
    out.emplace(name, i);
  }
  return out;
}

struct SpecMergeInputs {
  const OutputSpec& existing;
  const OutputSpec& incoming;
};

OutputSpec merge_specs(SpecMergeInputs inputs, const std::string& port_name) {
  OutputSpec out = inputs.existing;
  auto merge_str = [&](std::string& dst, const std::string& src, const char* field) {
    if (dst.empty()) {
      dst = src;
      return;
    }
    if (!src.empty() && dst != src) {
      throw std::runtime_error("Compiler: input spec mismatch for port " + port_name + " (" +
                               field + ")");
    }
  };
  auto merge_int = [&](int& dst, int src, const char* field) {
    if (dst <= 0) {
      dst = src;
      return;
    }
    if (src > 0 && dst != src) {
      throw std::runtime_error("Compiler: input spec mismatch for port " + port_name + " (" +
                               field + ")");
    }
  };

  merge_str(out.media_type, inputs.incoming.media_type, "media_type");
  merge_str(out.format, inputs.incoming.format, "format");
  merge_int(out.width, inputs.incoming.width, "width");
  merge_int(out.height, inputs.incoming.height, "height");
  merge_int(out.depth, inputs.incoming.depth, "depth");
  merge_str(out.memory, inputs.incoming.memory, "memory");
  merge_str(out.layout, inputs.incoming.layout, "layout");
  merge_str(out.dtype, inputs.incoming.dtype, "dtype");

  if (out.byte_size == 0) {
    out.byte_size = inputs.incoming.byte_size;
  } else if (inputs.incoming.byte_size > 0 && out.byte_size != inputs.incoming.byte_size) {
    throw std::runtime_error("Compiler: input spec mismatch for port " + port_name +
                             " (byte_size)");
  }
  if (static_cast<int>(inputs.incoming.certainty) > static_cast<int>(out.certainty)) {
    out.certainty = inputs.incoming.certainty;
  }
  if (out.note.empty())
    out.note = inputs.incoming.note;
  return out;
}

simaai::neat::NodeGroup merge_groups(const std::vector<const PipelineNode*>& nodes) {
  std::vector<std::shared_ptr<simaai::neat::Node>> out;
  for (const auto* pn : nodes) {
    if (!pn)
      continue;
    const auto& gnodes = pn->group().nodes();
    out.insert(out.end(), gnodes.begin(), gnodes.end());
  }
  return simaai::neat::NodeGroup(std::move(out));
}

} // namespace

bool Compiler::spec_complete_(const OutputSpec& spec) {
  if (spec.media_type.empty())
    return false;
  if (spec.media_type == "video/x-raw") {
    return !spec.format.empty() && spec.width > 0 && spec.height > 0;
  }
  if (spec.media_type == "application/vnd.simaai.tensor") {
    return !spec.format.empty() && spec.width > 0 && spec.height > 0 && spec.depth > 0;
  }
  // For other media types, rely on media_type + format only.
  return true;
}

CompiledGraph Compiler::compile(const Graph& g) const {
  if (!g.is_dag()) {
    throw std::runtime_error("Compiler: graph must be a DAG");
  }

  const std::size_t n = g.node_count();
  CompiledGraph out;
  out.edges = g.edges();
  out.edge_specs.resize(out.edges.size());
  out.port_names = g.port_names();

  // Validate pipeline node degrees (no fan-in/out inside pipelines).
  for (NodeId id = 0; id < n; ++id) {
    const auto& node = g.node(id);
    if (!node)
      continue;
    if (!is_pipeline_node(node))
      continue;

    if (g.in_degree(id) > 1) {
      throw std::runtime_error("Compiler: pipeline node has multiple inputs (add stage join)");
    }
    if (g.out_degree(id) > 1) {
      throw std::runtime_error("Compiler: pipeline node has multiple outputs (add stage fan-out)");
    }
  }

  // Partition pipeline nodes into linear segments.
  std::vector<int> segment_of(n, -1);
  int next_segment_id = 0;
  const auto topo = g.topo_order();

  for (NodeId id : topo) {
    const auto& node = g.node(id);
    if (!node || !is_pipeline_node(node))
      continue;
    if (segment_of[id] >= 0)
      continue;

    const PipelineNode* pn = as_pipeline_node(node);
    if (!pn) {
      throw std::runtime_error("Compiler: pipeline node cast failed");
    }

    const bool start_segment =
        (g.in_degree(id) != 1) || pn->is_source_like() ||
        (g.in_degree(id) == 1 && !is_pipeline_node(g.node(g.edge(g.in_edges(id)[0]).from)));

    if (!start_segment) {
      continue; // Will be claimed by a previous segment walk.
    }

    std::vector<NodeId> segment_nodes;
    NodeId cur = id;
    while (true) {
      segment_nodes.push_back(cur);
      segment_of[cur] = next_segment_id;

      if (g.out_degree(cur) != 1)
        break;
      const std::size_t eidx = g.out_edges(cur)[0];
      const Edge& e = g.edge(eidx);
      const auto& next_node = g.node(e.to);
      if (!next_node || !is_pipeline_node(next_node))
        break;
      if (g.in_degree(e.to) != 1)
        break;
      cur = e.to;
    }

    CompiledPipelineSegment seg;
    seg.id = next_segment_id++;
    seg.node_ids = segment_nodes;

    std::vector<const PipelineNode*> pnodes;
    pnodes.reserve(segment_nodes.size());
    for (NodeId nid : segment_nodes) {
      const auto& nnode = g.node(nid);
      const PipelineNode* p = as_pipeline_node(nnode);
      if (!p) {
        throw std::runtime_error("Compiler: pipeline node cast failed");
      }
      pnodes.push_back(p);
    }
    seg.group = merge_groups(pnodes);

    seg.source_like = false;
    for (const auto* p : pnodes) {
      if (p && p->is_source_like()) {
        seg.source_like = true;
        break;
      }
    }

    out.pipelines.push_back(std::move(seg));
  }

  // Gather stage nodes.
  for (NodeId id = 0; id < n; ++id) {
    const auto& node = g.node(id);
    if (!node || node->backend() != Backend::Stage)
      continue;
    const StageNode* sn = as_stage_node(node);
    if (!sn) {
      throw std::runtime_error("Compiler: stage node cast failed");
    }
    out.stages.push_back(CompiledStageNode{.node_id = id,
                                           .node = std::const_pointer_cast<StageNode>(
                                               std::static_pointer_cast<const StageNode>(node))});
  }

  // Build segment edge lists and validate source/push semantics.
  for (auto& seg : out.pipelines) {
    std::unordered_set<std::size_t> in_edges;
    std::unordered_set<std::size_t> out_edges;

    for (NodeId nid : seg.node_ids) {
      for (const std::size_t eidx : g.in_edges(nid)) {
        const Edge& e = g.edge(eidx);
        if (segment_of[e.from] != seg.id) {
          in_edges.insert(eidx);
        }
      }
      for (const std::size_t eidx : g.out_edges(nid)) {
        const Edge& e = g.edge(eidx);
        if (segment_of[e.to] != seg.id) {
          out_edges.insert(eidx);
        }
      }
    }

    seg.input_edges.assign(in_edges.begin(), in_edges.end());
    seg.output_edges.assign(out_edges.begin(), out_edges.end());

    if (seg.input_edges.size() > 1) {
      throw std::runtime_error("Compiler: pipeline segment has multiple inputs");
    }
    if (seg.output_edges.size() > 1) {
      throw std::runtime_error("Compiler: pipeline segment has multiple outputs");
    }
    if (!seg.input_edges.empty()) {
      const Edge& e = g.edge(seg.input_edges[0]);
      if (e.to != seg.node_ids.front()) {
        throw std::runtime_error("Compiler: pipeline segment input edge must target first node");
      }
    }
    if (!seg.output_edges.empty()) {
      const Edge& e = g.edge(seg.output_edges[0]);
      if (e.from != seg.node_ids.back()) {
        throw std::runtime_error(
            "Compiler: pipeline segment output edge must originate from last node");
      }
    }

    if (seg.source_like && !seg.input_edges.empty()) {
      throw std::runtime_error("Compiler: source pipeline segment cannot have input edges");
    }
    if (!seg.source_like) {
      for (NodeId nid : seg.node_ids) {
        const PipelineNode* pn = as_pipeline_node(g.node(nid));
        if (pn && pn->is_source_like()) {
          throw std::runtime_error("Compiler: push pipeline segment contains source-like node");
        }
      }
    }
  }

  // OutputSpec propagation.
  std::vector<std::unordered_map<PortId, OutputSpec>> node_outputs(n);

  for (NodeId id : topo) {
    const auto& node = g.node(id);
    if (!node)
      continue;

    const auto in_ports = node->input_ports();
    const auto in_index = build_port_index(in_ports);
    std::vector<OutputSpec> inputs(in_ports.size());
    std::vector<std::size_t> counts(in_ports.size(), 0);
    std::vector<bool> seen(in_ports.size(), false);

    for (const std::size_t eidx : g.in_edges(id)) {
      const Edge& e = g.edge(eidx);
      const std::string port_name = g.port_name(e.to_port);
      auto it = in_index.find(port_name);
      if (it == in_index.end()) {
        throw std::runtime_error("Compiler: edge references unknown input port: " + port_name);
      }
      const std::size_t idx = it->second;
      counts[idx]++;
      if (!seen[idx]) {
        inputs[idx] = out.edge_specs[eidx].spec;
        seen[idx] = true;
      } else {
        inputs[idx] = merge_specs({inputs[idx], out.edge_specs[eidx].spec}, port_name);
      }
    }

    for (std::size_t i = 0; i < in_ports.size(); ++i) {
      const int max_edges = in_ports[i].max_in_edges;
      if (max_edges > 0 && counts[i] > static_cast<std::size_t>(max_edges)) {
        throw std::runtime_error("Compiler: input port '" + in_ports[i].name +
                                 "' exceeds max_in_edges");
      }
    }

    std::unordered_set<PortId> out_port_ids;
    for (const std::size_t eidx : g.out_edges(id)) {
      const Edge& e = g.edge(eidx);
      out_port_ids.insert(e.from_port);
    }

    if (node->backend() == Backend::Pipeline) {
      const PipelineNode* pn = as_pipeline_node(node);
      if (!pn)
        throw std::runtime_error("Compiler: pipeline node cast failed");
      OutputSpec in_spec = inputs.empty() ? OutputSpec{} : inputs[0];
      OutputSpec out_spec = derive_output_spec(pn->group(), in_spec);
      for (const PortId pid : out_port_ids) {
        node_outputs[id][pid] = out_spec;
      }
    } else {
      for (const PortId pid : out_port_ids) {
        OutputSpec spec = node->output_spec(inputs, pid);
        node_outputs[id][pid] = spec;
      }
    }

    for (const std::size_t eidx : g.out_edges(id)) {
      const Edge& e = g.edge(eidx);
      auto it = node_outputs[id].find(e.from_port);
      if (it == node_outputs[id].end()) {
        const std::string pname = g.port_name(e.from_port);
        throw std::runtime_error("Compiler: missing output spec for port: " + pname);
      }
      out.edge_specs[eidx].spec = it->second;
      out.edge_specs[eidx].complete = spec_complete_(it->second);
    }
  }

  // Segment input/output specs based on edges.
  for (auto& seg : out.pipelines) {
    if (!seg.input_edges.empty()) {
      const auto& es = out.edge_specs[seg.input_edges[0]];
      seg.input_spec = es.spec;
      seg.input_complete = es.complete;
    }
    if (!seg.output_edges.empty()) {
      const auto& es = out.edge_specs[seg.output_edges[0]];
      seg.output_spec = es.spec;
      seg.output_complete = es.complete;
    }
  }

  return out;
}

} // namespace simaai::neat::graph

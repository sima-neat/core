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
      throw std::runtime_error("Compiler: empty port name at index " + std::to_string(i) +
                               " (port count=" + std::to_string(ports.size()) + ")");
    }
    if (out.find(name) != out.end()) {
      throw std::runtime_error("Compiler: duplicate port name: '" + name + "' at index " +
                               std::to_string(i));
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
      throw std::runtime_error("Compiler: input spec mismatch for port '" + port_name + "' (" +
                               field + "): existing='" + dst + "' vs incoming='" + src + "'");
    }
  };
  auto merge_int = [&](int& dst, int src, const char* field) {
    if (dst <= 0) {
      dst = src;
      return;
    }
    if (src > 0 && dst != src) {
      throw std::runtime_error("Compiler: input spec mismatch for port '" + port_name + "' (" +
                               field + "): existing=" + std::to_string(dst) +
                               " vs incoming=" + std::to_string(src));
    }
  };

  if (out.payload_type == PayloadType::Auto) {
    out.payload_type = inputs.incoming.payload_type;
  } else if (inputs.incoming.payload_type != PayloadType::Auto &&
             out.payload_type != inputs.incoming.payload_type) {
    throw std::runtime_error(
        "Compiler: input spec mismatch for port '" + port_name +
        "' (payload_type): existing=" + std::to_string(static_cast<int>(out.payload_type)) +
        " vs incoming=" + std::to_string(static_cast<int>(inputs.incoming.payload_type)));
  }
  merge_str(out.media_type, inputs.incoming.media_type, "media_type");
  if (out.payload_type == PayloadType::Auto) {
    out.payload_type = payload_type_from_media_type(out.media_type);
  }
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
    throw std::runtime_error("Compiler: input spec mismatch for port '" + port_name +
                             "' (byte_size): existing=" + std::to_string(out.byte_size) +
                             " vs incoming=" + std::to_string(inputs.incoming.byte_size));
  }
  if (static_cast<int>(inputs.incoming.certainty) > static_cast<int>(out.certainty)) {
    out.certainty = inputs.incoming.certainty;
  }
  if (out.note.empty())
    out.note = inputs.incoming.note;
  return out;
}

std::vector<std::shared_ptr<simaai::neat::Node>>
merge_pipeline_nodes(const std::vector<const PipelineNode*>& nodes) {
  std::vector<std::shared_ptr<simaai::neat::Node>> out;
  for (const auto* pn : nodes) {
    if (!pn)
      continue;
    const auto& gnodes = pn->nodes();
    out.insert(out.end(), gnodes.begin(), gnodes.end());
  }
  return out;
}

} // namespace

bool Compiler::spec_complete_(const OutputSpec& spec) {
  const std::string media =
      !spec.media_type.empty() ? spec.media_type : media_type_from_payload_type(spec.payload_type);
  if (media.empty())
    return false;
  if (media == "video/x-raw") {
    return !spec.format.empty() && spec.width > 0 && spec.height > 0;
  }
  if (media == "application/vnd.simaai.tensor") {
    return !spec.format.empty() && spec.width > 0 && spec.height > 0 && spec.depth > 0;
  }
  // For other media types, rely on media_type + format only.
  return true;
}

CompiledGraph Compiler::compile(const Graph& g) const {
  return compile(g, CompilerOptions{});
}

CompiledGraph Compiler::compile(const Graph& g, const CompilerOptions& opt) const {
  if (!g.is_dag()) {
    throw std::runtime_error(
        "Compiler: graph must be a DAG (node_count=" + std::to_string(g.node_count()) +
        ", edge_count=" + std::to_string(g.edges().size()) + "); check for cycles in the graph");
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
      throw std::runtime_error("Compiler: pipeline node " + std::to_string(id) +
                               " has multiple inputs (in_degree=" +
                               std::to_string(g.in_degree(id)) + "; add stage join)");
    }
    if (g.out_degree(id) > 1) {
      throw std::runtime_error("Compiler: pipeline node " + std::to_string(id) +
                               " has multiple outputs (out_degree=" +
                               std::to_string(g.out_degree(id)) + "; add stage fan-out)");
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
      throw std::runtime_error("Compiler: pipeline node cast failed for node " +
                               std::to_string(id) + " (kind='" + node->kind() +
                               "', backend=Pipeline but dynamic_cast to PipelineNode failed)");
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
        throw std::runtime_error("Compiler: pipeline node cast failed for node " +
                                 std::to_string(nid) + " (kind='" +
                                 (nnode ? nnode->kind() : "null") + "')");
      }
      pnodes.push_back(p);
    }
    seg.nodes = merge_pipeline_nodes(pnodes);

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
      throw std::runtime_error("Compiler: stage node cast failed for node " + std::to_string(id) +
                               " (kind='" + node->kind() +
                               "', backend=Stage but dynamic_cast to StageNode failed)");
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

    if (g.in_edges(id).empty() && in_ports.size() == 1U) {
      auto root_it = opt.root_input_specs.find(id);
      if (root_it != opt.root_input_specs.end()) {
        inputs[0] = root_it->second;
        seen[0] = true;
      }
    }

    for (const std::size_t eidx : g.in_edges(id)) {
      const Edge& e = g.edge(eidx);
      const std::string port_name = g.port_name(e.to_port);
      auto it = in_index.find(port_name);
      if (it == in_index.end()) {
        std::string available;
        for (const auto& kv : in_index) {
          if (!available.empty())
            available += ", ";
          available += "'" + kv.first + "'";
        }
        throw std::runtime_error("Compiler: edge references unknown input port: '" + port_name +
                                 "' on node " + std::to_string(id) + " (available ports: [" +
                                 available + "])");
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
        throw std::runtime_error("Compiler: input port '" + in_ports[i].name + "' on node " +
                                 std::to_string(id) +
                                 " exceeds max_in_edges (actual=" + std::to_string(counts[i]) +
                                 ", max=" + std::to_string(max_edges) + ")");
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
        throw std::runtime_error("Compiler: pipeline node cast failed for node " +
                                 std::to_string(id) + " (kind='" + node->kind() + "')");
      OutputSpec in_spec = inputs.empty() ? OutputSpec{} : inputs[0];
      OutputSpec out_spec = derive_output_spec(pn->nodes(), in_spec);
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
        throw std::runtime_error("Compiler: missing output spec for port '" + pname + "' on node " +
                                 std::to_string(id) + " (node kind='" +
                                 (node ? node->kind() : "null") + "')");
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
    } else if (!seg.node_ids.empty()) {
      auto root_it = opt.root_input_specs.find(seg.node_ids.front());
      if (root_it != opt.root_input_specs.end()) {
        seg.input_spec = root_it->second;
        seg.input_complete = spec_complete_(seg.input_spec);
      }
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

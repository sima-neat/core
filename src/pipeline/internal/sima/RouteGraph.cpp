#include "pipeline/internal/sima/RouteGraph.h"

#include <algorithm>
#include <cctype>
#include <unordered_map>

namespace simaai::neat::pipeline_internal::sima {
namespace {

std::string lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

std::string tensor_name_at_index(const std::vector<MpkTensorContract>& tensors, int index) {
  if (index >= 0 && static_cast<std::size_t>(index) < tensors.size()) {
    const auto& tensor = tensors[static_cast<std::size_t>(index)];
    if (!tensor.name.empty()) {
      return tensor.name;
    }
    if (!tensor.segment_name.empty()) {
      return tensor.segment_name;
    }
  }
  return {};
}

} // namespace

RouteGraphKernelKind canonical_route_graph_kernel_kind(const std::string& raw_kernel) {
  const std::string kernel = lower_copy(raw_kernel);
  if (kernel.empty()) {
    return RouteGraphKernelKind::Unknown;
  }
  if (kernel.find("detesscast") != std::string::npos || kernel.find("detess_cast") != std::string::npos || kernel.find("detessellatecast") != std::string::npos) {
    return RouteGraphKernelKind::DetessCast;
  }
  if (kernel.find("detessdequant") != std::string::npos) {
    return RouteGraphKernelKind::DetessDequant;
  }
  if (kernel.find("detessellate" ) != std::string::npos || kernel.find("detess") != std::string::npos) {
    return RouteGraphKernelKind::Detess;
  }
  if (kernel.find("dequant") != std::string::npos) {
    return RouteGraphKernelKind::Dequantize;
  }
  if (kernel.find("boxdecode") != std::string::npos) {
    return RouteGraphKernelKind::BoxDecode;
  }
  if (kernel.find("casttess") != std::string::npos || kernel.find("cast_tess") != std::string::npos || kernel.find("casttessellate") != std::string::npos) {
    return RouteGraphKernelKind::CastTess;
  }
  if (kernel.find("quanttess") != std::string::npos ||
      (kernel.find("quant") != std::string::npos && kernel.find("tess") != std::string::npos)) {
    return RouteGraphKernelKind::QuantTess;
  }
  if (kernel.find("quant") != std::string::npos) {
    return RouteGraphKernelKind::Quant;
  }
  if (kernel.find("tess") != std::string::npos) {
    return RouteGraphKernelKind::Tess;
  }
  if (kernel.find("cast") != std::string::npos) {
    return RouteGraphKernelKind::Cast;
  }
  if (kernel.find("preproc") != std::string::npos || kernel.find("preprocess") != std::string::npos) {
    return RouteGraphKernelKind::Preproc;
  }
  if (kernel.find("unpack") != std::string::npos) {
    return RouteGraphKernelKind::Unpack;
  }
  if (kernel.find("slice") != std::string::npos) {
    return RouteGraphKernelKind::Slice;
  }
  if (kernel.find("pack_transform") != std::string::npos ||
      kernel.find("buffer_concat") != std::string::npos ||
      kernel.find("concatenat") != std::string::npos) {
    return RouteGraphKernelKind::PassThrough;
  }
  if (kernel.find("pass_through") != std::string::npos || kernel.find("passthrough") != std::string::npos) {
    return RouteGraphKernelKind::PassThrough;
  }
  if (kernel.find("mla") != std::string::npos || kernel.find("infer") != std::string::npos) {
    return RouteGraphKernelKind::Mla;
  }
  return RouteGraphKernelKind::Unknown;
}

const char* route_graph_kernel_name(RouteGraphKernelKind kind) {
  switch (kind) {
  case RouteGraphKernelKind::Unknown:
    return "unknown";
  case RouteGraphKernelKind::Preproc:
    return "preproc";
  case RouteGraphKernelKind::Quant:
    return "quant";
  case RouteGraphKernelKind::Tess:
    return "tess";
  case RouteGraphKernelKind::QuantTess:
    return "quanttess";
  case RouteGraphKernelKind::Cast:
    return "cast";
  case RouteGraphKernelKind::CastTess:
    return "casttess";
  case RouteGraphKernelKind::Detess:
    return "detess";
  case RouteGraphKernelKind::DetessCast:
    return "detesscast";
  case RouteGraphKernelKind::DetessDequant:
    return "detessdequant";
  case RouteGraphKernelKind::Dequantize:
    return "dequantize";
  case RouteGraphKernelKind::BoxDecode:
    return "boxdecode";
  case RouteGraphKernelKind::Unpack:
    return "unpack";
  case RouteGraphKernelKind::Slice:
    return "slice";
  case RouteGraphKernelKind::PassThrough:
    return "pass_through";
  case RouteGraphKernelKind::Mla:
    return "mla";
  }
  return "unknown";
}

RouteGraph build_route_graph(const MpkContract& contract) {
  RouteGraph out;
  out.model_name = contract.model_name;
  out.execution_order = plugins_in_execution_order(contract);
  std::vector<bool> seen(contract.plugins.size(), false);
  for (const std::size_t idx : out.execution_order) {
    if (idx < seen.size()) {
      seen[idx] = true;
    }
  }
  for (std::size_t idx = 0; idx < contract.plugins.size(); ++idx) {
    if (!seen[idx]) {
      out.execution_order.push_back(idx);
    }
  }

  const auto* mla_stage = get_mla_stage_io_contract(contract);
  if (mla_stage) {
    const auto mla_idx = find_plugin_index_by_name_or_id(contract, !mla_stage->name.empty() ? mla_stage->name
                                                                                            : mla_stage->plugin_id);
    if (mla_idx.has_value()) {
      out.mla_plugin_index = static_cast<int>(*mla_idx);
    }
  }

  std::unordered_map<std::size_t, std::size_t> rank_by_index;
  rank_by_index.reserve(out.execution_order.size());
  for (std::size_t rank = 0; rank < out.execution_order.size(); ++rank) {
    rank_by_index.emplace(out.execution_order[rank], rank);
  }
  const auto mla_rank_it = out.mla_plugin_index >= 0
                               ? rank_by_index.find(static_cast<std::size_t>(out.mla_plugin_index))
                               : rank_by_index.end();
  const std::size_t mla_rank =
      mla_rank_it == rank_by_index.end() ? out.execution_order.size() : mla_rank_it->second;

  out.nodes.reserve(contract.plugins.size());
  for (std::size_t plugin_idx = 0; plugin_idx < contract.plugins.size(); ++plugin_idx) {
    const auto& plugin = contract.plugins[plugin_idx];
    std::string kernel_source = plugin.kernel;
    if (kernel_source.empty()) {
      kernel_source = plugin.name;
    }
    const auto rank_it = rank_by_index.find(plugin_idx);
    const std::size_t rank = rank_it == rank_by_index.end() ? out.execution_order.size() : rank_it->second;

    RouteGraphNode node;
    node.plugin_index = plugin_idx;
    node.plugin_name = plugin.name;
    node.plugin_id = plugin.plugin_id;
    node.processor = plugin.processor;
    node.kernel = plugin.kernel;
    node.kind = canonical_route_graph_kernel_kind(kernel_source);
    if (node.kind == RouteGraphKernelKind::Unknown &&
        lower_copy(plugin.processor).find("mla") != std::string::npos) {
      node.kind = RouteGraphKernelKind::Mla;
    }
    node.sequence = plugin.sequence;
    node.before_mla = rank < mla_rank;
    node.after_mla = rank > mla_rank;
    out.nodes.push_back(std::move(node));
  }

  out.edges.reserve(contract.edges.size());
  for (const auto& edge : contract.edges) {
    RouteGraphEdge graph_edge;
    graph_edge.src_plugin_index = edge.src_plugin_index;
    graph_edge.src_output_index = edge.src_output_index;
    graph_edge.dst_plugin_index = edge.dst_plugin_index;
    graph_edge.dst_input_index = edge.dst_input_index;
    graph_edge.src_plugin = edge.src_plugin;
    graph_edge.dst_plugin = edge.dst_plugin;
    graph_edge.tensor_name = edge.tensor_name;
    if (edge.src_plugin_index < contract.plugins.size()) {
      graph_edge.src_tensor_name =
          tensor_name_at_index(contract.plugins[edge.src_plugin_index].output_tensors, edge.src_output_index);
    }
    if (edge.dst_plugin_index < contract.plugins.size()) {
      graph_edge.dst_tensor_name =
          tensor_name_at_index(contract.plugins[edge.dst_plugin_index].input_tensors, edge.dst_input_index);
    }
    out.edges.push_back(std::move(graph_edge));
  }

  return out;
}

const RouteGraphNode* route_graph_node(const RouteGraph& graph, std::size_t plugin_index) {
  if (plugin_index >= graph.nodes.size()) {
    return nullptr;
  }
  return &graph.nodes[plugin_index];
}

std::vector<const RouteGraphEdge*> route_graph_incoming_edges(const RouteGraph& graph,
                                                              std::size_t plugin_index) {
  std::vector<const RouteGraphEdge*> out;
  for (const auto& edge : graph.edges) {
    if (edge.dst_plugin_index == plugin_index) {
      out.push_back(&edge);
    }
  }
  std::stable_sort(out.begin(), out.end(), [](const RouteGraphEdge* a, const RouteGraphEdge* b) {
    if (a->dst_input_index != b->dst_input_index) {
      return a->dst_input_index < b->dst_input_index;
    }
    return a->src_plugin_index < b->src_plugin_index;
  });
  return out;
}

std::vector<const RouteGraphEdge*> route_graph_outgoing_edges(const RouteGraph& graph,
                                                              std::size_t plugin_index) {
  std::vector<const RouteGraphEdge*> out;
  for (const auto& edge : graph.edges) {
    if (edge.src_plugin_index == plugin_index) {
      out.push_back(&edge);
    }
  }
  std::stable_sort(out.begin(), out.end(), [](const RouteGraphEdge* a, const RouteGraphEdge* b) {
    if (a->src_output_index != b->src_output_index) {
      return a->src_output_index < b->src_output_index;
    }
    return a->dst_plugin_index < b->dst_plugin_index;
  });
  return out;
}

std::optional<std::size_t> route_graph_execution_rank(const RouteGraph& graph,
                                                      std::size_t plugin_index) {
  for (std::size_t rank = 0; rank < graph.execution_order.size(); ++rank) {
    if (graph.execution_order[rank] == plugin_index) {
      return rank;
    }
  }
  return std::nullopt;
}

} // namespace simaai::neat::pipeline_internal::sima

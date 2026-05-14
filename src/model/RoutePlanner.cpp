#include "model/internal/RoutePlanner.h"
#include "model/internal/InputPlanner.h"
#include "model/internal/ModelParser.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/RenderedMlaContractQuery.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/BoxDecodeTypeUtils.h"
#include "pipeline/internal/sima/MlaStaticContractExtractor.h"

#include <algorithm>
#include <cstddef>
#include <cctype>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace simaai::neat::internal {
namespace fs = std::filesystem;
namespace rendered_stage_query = pipeline_internal::rendered_stage_query;

namespace {

using pipeline_internal::lower_copy;
using pipeline_internal::upper_copy;

bool strict_model_managed_boxdecode_available(const ModelPack& pack);
bool boxdecode_route_available(const RouteCapability& capability);
bool route_has_post_stage(const RouteCapability& capability);

bool route_debug_enabled() {
  if (pipeline_internal::env_bool("SIMA_ROUTE_DEBUG", false)) {
    return true;
  }
  return pipeline_internal::env_bool("SIMA_MPK_CONTRACT_DEBUG", false);
}

bool dtype_is_bf16_like(std::string raw) {
  raw = upper_copy(std::move(raw));
  return raw.find("BF16") != std::string::npos || raw.find("BFLOAT16") != std::string::npos;
}

bool dtype_is_float_like(std::string raw) {
  raw = upper_copy(std::move(raw));
  if (raw.empty())
    return false;
  return raw.find("FP") != std::string::npos || raw.find("FLOAT") != std::string::npos ||
         raw.find("BF16") != std::string::npos || raw.find("BFLOAT16") != std::string::npos;
}

bool dtype_is_quantized_like(std::string raw) {
  raw = upper_copy(std::move(raw));
  if (raw.empty() || dtype_is_float_like(raw))
    return false;
  return raw.find("INT8") != std::string::npos || raw.find("UINT8") != std::string::npos ||
         raw.find("INT16") != std::string::npos || raw.find("UINT16") != std::string::npos ||
         raw.find("INT32") != std::string::npos || raw.find("UINT32") != std::string::npos;
}

std::string canonical_dtype_for_signal(std::string raw) {
  raw = upper_copy(std::move(raw));
  if (raw.find("BFLOAT16") != std::string::npos || raw.find("BF16") != std::string::npos) {
    return "BF16";
  }
  if (raw.find("FLOAT32") != std::string::npos || raw == "FP32") {
    return "FP32";
  }
  if (raw.find("FLOAT16") != std::string::npos || raw == "FP16") {
    return "FP16";
  }
  if (raw.find("INT8") != std::string::npos) {
    return "INT8";
  }
  if (raw.find("UINT8") != std::string::npos) {
    return "UINT8";
  }
  if (raw.find("INT16") != std::string::npos) {
    return "INT16";
  }
  if (raw.find("UINT16") != std::string::npos) {
    return "UINT16";
  }
  if (raw.find("INT32") != std::string::npos) {
    return "INT32";
  }
  if (raw.find("UINT32") != std::string::npos) {
    return "UINT32";
  }
  return raw;
}

std::string
primary_tensor_dtype(const std::vector<pipeline_internal::sima::MpkTensorContract>& tensors) {
  for (const auto& tensor : tensors) {
    if (!tensor.logical_dtype.empty()) {
      return canonical_dtype_for_signal(tensor.logical_dtype);
    }
    if (!tensor.dtype.empty()) {
      return canonical_dtype_for_signal(tensor.dtype);
    }
  }
  return {};
}

int primary_tensor_rank(const std::vector<pipeline_internal::sima::MpkTensorContract>& tensors) {
  for (const auto& tensor : tensors) {
    if (!tensor.logical_shape.empty()) {
      return static_cast<int>(tensor.logical_shape.size());
    }
    if (!tensor.mpk_shape.empty()) {
      return static_cast<int>(tensor.mpk_shape.size());
    }
  }
  return 0;
}

std::string normalize_tensor_format(std::string raw, const std::string& fallback) {
  raw = upper_copy(std::move(raw));
  if (raw.find("BF16") != std::string::npos || raw.find("BFLOAT16") != std::string::npos)
    return "BF16";
  if (raw.find("UINT8") != std::string::npos)
    return "UINT8";
  if (raw.find("INT8") != std::string::npos)
    return "INT8";
  if (raw.find("UINT16") != std::string::npos)
    return "UINT16";
  if (raw.find("INT16") != std::string::npos)
    return "INT16";
  if (raw.find("INT32") != std::string::npos)
    return "INT32";
  if (raw.find("FP32") != std::string::npos || raw.find("FLOAT32") != std::string::npos)
    return "FP32";
  if (raw.find("FP64") != std::string::npos || raw.find("FLOAT64") != std::string::npos)
    return "FP64";
  return fallback;
}

bool media_type_is_video_raw(std::string media_type) {
  media_type = lower_copy(std::move(media_type));
  return media_type == "video/x-raw";
}

std::string canonical_mpk_kernel_kind(const std::string& raw_kernel) {
  std::string token;
  token.reserve(raw_kernel.size());
  for (const unsigned char c : raw_kernel) {
    if (std::isalnum(c)) {
      token.push_back(static_cast<char>(std::tolower(c)));
    }
  }
  if (token.empty()) {
    return {};
  }
  if (token.find("boxdecode") != std::string::npos ||
      token.find("boxdecoder") != std::string::npos) {
    return "boxdecode";
  }
  if (token.find("detess") != std::string::npos && token.find("dequant") != std::string::npos) {
    return "detessdequant";
  }
  if (token.find("unpack") != std::string::npos) {
    return "unpacktransform";
  }
  if (token.find("packtransform") != std::string::npos) {
    return "packtransform";
  }
  if (token.find("bufferconcat") != std::string::npos ||
      token.find("concatenat") != std::string::npos) {
    return "bufferconcat";
  }
  if (token.find("dequant") != std::string::npos) {
    return "dequantize";
  }
  if (token.find("detess") != std::string::npos) {
    return "detessellate";
  }
  if (token.find("quanttess") != std::string::npos) {
    return "quanttess";
  }
  if (token.find("quant") != std::string::npos) {
    return "quantize";
  }
  if (token.find("tess") != std::string::npos) {
    return "tessellate";
  }
  if (token.find("preproc") != std::string::npos) {
    return "preproc";
  }
  if (token.find("cast") != std::string::npos) {
    return "cast";
  }
  if (token.find("passthrough") != std::string::npos) {
    return "pass_through";
  }
  return token;
}

const char* ordered_route_op_name(OrderedRouteOp::Kind kind) {
  switch (kind) {
  case OrderedRouteOp::Kind::Unknown:
    return "unknown";
  case OrderedRouteOp::Kind::Preproc:
    return "preproc";
  case OrderedRouteOp::Kind::Quant:
    return "quant";
  case OrderedRouteOp::Kind::Tess:
    return "tess";
  case OrderedRouteOp::Kind::QuantTess:
    return "quanttess";
  case OrderedRouteOp::Kind::Cast:
    return "cast";
  case OrderedRouteOp::Kind::CastTess:
    return "casttess";
  case OrderedRouteOp::Kind::Detess:
    return "detess";
  case OrderedRouteOp::Kind::DetessCast:
    return "detesscast";
  case OrderedRouteOp::Kind::DetessDequant:
    return "detessdequant";
  case OrderedRouteOp::Kind::Dequantize:
    return "dequantize";
  case OrderedRouteOp::Kind::BoxDecode:
    return "boxdecode";
  case OrderedRouteOp::Kind::Unpack:
    return "unpack";
  }
  return "unknown";
}

std::string ordered_route_chain_csv(const std::vector<OrderedRouteOp>& ops) {
  if (ops.empty()) {
    return "none";
  }
  std::string out;
  for (std::size_t i = 0; i < ops.size(); ++i) {
    if (!out.empty()) {
      out += ",";
    }
    out += ordered_route_op_name(ops[i].kind);
  }
  return out;
}

struct ParsedTensorShapeContract {
  bool valid = false;
  std::string layout;
  int rank = 0;
  int batch = 1;
  int width = 0;
  int height = 0;
  int depth = 0;
};

int clamp_positive_dim(std::int64_t value) {
  if (value <= 0) {
    return 0;
  }
  if (value > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(value);
}

ParsedTensorShapeContract parse_tensor_shape_contract(const std::vector<std::int64_t>& shape) {
  ParsedTensorShapeContract out;
  out.rank = static_cast<int>(shape.size());
  if (shape.empty()) {
    return out;
  }

  if (shape.size() >= 4U) {
    out.batch = clamp_positive_dim(shape[shape.size() - 4U]);
    out.height = clamp_positive_dim(shape[shape.size() - 3U]);
    out.width = clamp_positive_dim(shape[shape.size() - 2U]);
    out.depth = clamp_positive_dim(shape[shape.size() - 1U]);
  } else if (shape.size() == 3U) {
    out.batch = 1;
    out.height = clamp_positive_dim(shape[0]);
    out.width = clamp_positive_dim(shape[1]);
    out.depth = clamp_positive_dim(shape[2]);
  } else if (shape.size() == 2U) {
    out.batch = 1;
    out.height = clamp_positive_dim(shape[0]);
    out.width = clamp_positive_dim(shape[1]);
    out.depth = 1;
  } else {
    out.batch = 1;
    out.height = 1;
    out.width = clamp_positive_dim(shape[0]);
    out.depth = 1;
  }
  if (out.batch <= 0) {
    out.batch = 1;
  }
  out.valid = out.width > 0 && out.height > 0 && out.depth > 0;
  return out;
}

std::vector<std::int64_t>
preferred_tensor_shape(const pipeline_internal::sima::MpkTensorContract& tensor) {
  return tensor.logical_shape;
}

std::string preferred_tensor_dtype(const pipeline_internal::sima::MpkTensorContract& tensor) {
  if (!tensor.logical_dtype.empty()) {
    return canonical_dtype_for_signal(tensor.logical_dtype);
  }
  return canonical_dtype_for_signal(tensor.dtype);
}

template <typename ContractT>
bool populate_tensor_contract_from_stage_tensor(
    const pipeline_internal::sima::MpkTensorContract& tensor, const std::string& media_type,
    const std::string& source_stage, ContractT* out) {
  if (!out) {
    return false;
  }
  const auto shape = preferred_tensor_shape(tensor);
  const auto parsed_shape = parse_tensor_shape_contract(shape);
  if (!parsed_shape.valid) {
    return false;
  }
  out->valid = true;
  out->media_type = media_type.empty() ? "application/vnd.simaai.tensor" : media_type;
  out->dtype = preferred_tensor_dtype(tensor);
  out->layout = parsed_shape.layout;
  out->rank = parsed_shape.rank;
  out->batch = parsed_shape.batch;
  out->width = parsed_shape.width;
  out->height = parsed_shape.height;
  out->depth = parsed_shape.depth;
  // Phase 4a: preserve the canonical shape vector so tensor-domain consumers
  // can read it directly without reconstructing from the (batch,h,w,d) scalars
  // + a layout token.
  out->logical_shape = shape;
  out->source_stage = source_stage;
  return true;
}

template <typename ContractT>
bool populate_tensor_contract_from_stage_tensors(
    const std::vector<pipeline_internal::sima::MpkTensorContract>& tensors,
    const std::string& media_type, const std::string& source_stage, ContractT* out) {
  if (!out || tensors.empty()) {
    return false;
  }
  return populate_tensor_contract_from_stage_tensor(tensors.front(), media_type, source_stage, out);
}

std::vector<IngressTensorContract>
derive_ingress_contracts_from_mpk_contract(const pipeline_internal::sima::MpkContract& contract,
                                           const pipeline_internal::sima::RouteGraph& graph,
                                           const std::vector<std::size_t>& ordered) {
  const auto ordered_route_kind_from_graph_kind =
      [](pipeline_internal::sima::RouteGraphKernelKind kind) -> OrderedRouteOp::Kind {
    using GraphKind = pipeline_internal::sima::RouteGraphKernelKind;
    switch (kind) {
    case GraphKind::Preproc:
      return OrderedRouteOp::Kind::Preproc;
    case GraphKind::Quant:
      return OrderedRouteOp::Kind::Quant;
    case GraphKind::Tess:
      return OrderedRouteOp::Kind::Tess;
    case GraphKind::QuantTess:
      return OrderedRouteOp::Kind::QuantTess;
    case GraphKind::Cast:
      return OrderedRouteOp::Kind::Cast;
    case GraphKind::CastTess:
      return OrderedRouteOp::Kind::CastTess;
    case GraphKind::Detess:
      return OrderedRouteOp::Kind::Detess;
    case GraphKind::DetessCast:
      return OrderedRouteOp::Kind::DetessCast;
    case GraphKind::DetessDequant:
      return OrderedRouteOp::Kind::DetessDequant;
    case GraphKind::Dequantize:
      return OrderedRouteOp::Kind::Dequantize;
    case GraphKind::BoxDecode:
      return OrderedRouteOp::Kind::BoxDecode;
    case GraphKind::Unpack:
      return OrderedRouteOp::Kind::Unpack;
    case GraphKind::Unknown:
    case GraphKind::Slice:
    case GraphKind::PassThrough:
    case GraphKind::Mla:
      break;
    }
    return OrderedRouteOp::Kind::Unknown;
  };

  const auto collect_branch_ops =
      [&](const std::size_t start_plugin,
          std::optional<std::size_t>* join_plugin_index) -> std::vector<OrderedRouteOp> {
    std::vector<OrderedRouteOp> ops;
    std::unordered_set<std::size_t> visited;
    std::size_t current_plugin = start_plugin;
    while (current_plugin < contract.plugins.size() && visited.insert(current_plugin).second) {
      const auto* node = pipeline_internal::sima::route_graph_node(graph, current_plugin);
      if (!node || node->kind == pipeline_internal::sima::RouteGraphKernelKind::Mla) {
        if (join_plugin_index && node &&
            node->kind == pipeline_internal::sima::RouteGraphKernelKind::Mla) {
          *join_plugin_index = current_plugin;
        }
        break;
      }

      const OrderedRouteOp::Kind branch_kind = ordered_route_kind_from_graph_kind(node->kind);
      if (branch_kind != OrderedRouteOp::Kind::Unknown) {
        const auto& stage = contract.plugins[current_plugin];
        OrderedRouteOp op;
        op.kind = branch_kind;
        op.plugin_name = stage.name;
        op.plugin_id = stage.plugin_id;
        op.kernel = stage.kernel;
        op.sequence = stage.sequence;
        op.before_mla = true;
        ops.push_back(std::move(op));
      }

      std::vector<const pipeline_internal::sima::RouteGraphEdge*> outgoing_edges;
      for (const auto* edge :
           pipeline_internal::sima::route_graph_outgoing_edges(graph, current_plugin)) {
        if (edge) {
          outgoing_edges.push_back(edge);
        }
      }
      if (outgoing_edges.size() != 1U) {
        break;
      }

      const std::size_t next_plugin = outgoing_edges.front()->dst_plugin_index;
      const auto* next_node = pipeline_internal::sima::route_graph_node(graph, next_plugin);
      std::size_t next_incoming_count = 0U;
      for (const auto* edge :
           pipeline_internal::sima::route_graph_incoming_edges(graph, next_plugin)) {
        if (edge) {
          ++next_incoming_count;
        }
      }
      if (join_plugin_index &&
          (next_incoming_count > 1U ||
           (next_node &&
            (next_node->kind == pipeline_internal::sima::RouteGraphKernelKind::PassThrough ||
             next_node->kind == pipeline_internal::sima::RouteGraphKernelKind::Mla)))) {
        *join_plugin_index = next_plugin;
      }
      if (!next_node || next_node->kind == pipeline_internal::sima::RouteGraphKernelKind::Mla ||
          next_node->kind == pipeline_internal::sima::RouteGraphKernelKind::PassThrough ||
          next_incoming_count > 1U) {
        break;
      }
      current_plugin = next_plugin;
    }
    return ops;
  };

  std::unordered_map<std::size_t, std::size_t> rank_by_index;
  rank_by_index.reserve(ordered.size());
  for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
    rank_by_index.emplace(ordered[rank], rank);
  }

  std::vector<IngressTensorContract> out;
  out.reserve(contract.ingress_tensors.size());
  for (std::size_t ingress_index = 0; ingress_index < contract.ingress_tensors.size();
       ++ingress_index) {
    const auto& ingress = contract.ingress_tensors[ingress_index];
    const std::string ingress_name = !ingress.name.empty() ? ingress.name : ingress.segment_name;
    if (ingress_name.empty()) {
      continue;
    }
    const std::string ingress_key = lower_copy(ingress_name);

    const pipeline_internal::sima::MpkPluginIoContract* consumer_stage = nullptr;
    std::size_t consumer_plugin_index = 0U;
    int consumer_input_index = -1;
    std::size_t consumer_rank = std::numeric_limits<std::size_t>::max();
    for (std::size_t plugin_index = 0; plugin_index < contract.plugins.size(); ++plugin_index) {
      const auto rank_it = rank_by_index.find(plugin_index);
      const std::size_t rank = rank_it == rank_by_index.end() ? ordered.size() : rank_it->second;
      const auto& stage = contract.plugins[plugin_index];
      for (std::size_t input_index = 0; input_index < stage.input_tensors.size(); ++input_index) {
        const auto& input_tensor = stage.input_tensors[input_index];
        const std::string input_name =
            !input_tensor.name.empty() ? input_tensor.name : input_tensor.segment_name;
        if (lower_copy(input_name) != ingress_key) {
          continue;
        }
        if (!consumer_stage || rank < consumer_rank) {
          consumer_stage = &stage;
          consumer_plugin_index = plugin_index;
          consumer_input_index = static_cast<int>(input_index);
          consumer_rank = rank;
        }
      }
    }

    IngressTensorContract ingress_contract;
    ingress_contract.ingress_index = static_cast<int>(ingress_index);
    ingress_contract.source_tensor_name = ingress_name;
    ingress_contract.dst_plugin_index = consumer_plugin_index;
    ingress_contract.dst_input_index = consumer_input_index;

    if (consumer_stage && consumer_input_index >= 0 &&
        static_cast<std::size_t>(consumer_input_index) < consumer_stage->input_tensors.size()) {
      const auto& consumer_input =
          consumer_stage->input_tensors[static_cast<std::size_t>(consumer_input_index)];
      const std::string source_stage =
          !consumer_stage->name.empty() ? consumer_stage->name : consumer_stage->plugin_id;
      const std::string kernel_source =
          !consumer_stage->kernel.empty() ? consumer_stage->kernel : consumer_stage->name;
      const std::string kernel = canonical_mpk_kernel_kind(kernel_source);
      const std::string media_type = kernel == "preproc"
                                         ? std::string("video/x-raw")
                                         : std::string("application/vnd.simaai.tensor");
      (void)populate_tensor_contract_from_stage_tensor(consumer_input, media_type, source_stage,
                                                       &ingress_contract);
      ingress_contract.ingress_index = static_cast<int>(ingress_index);
      ingress_contract.source_tensor_name = ingress_name;
      ingress_contract.dst_plugin_index = consumer_plugin_index;
      ingress_contract.dst_input_index = consumer_input_index;
      ingress_contract.branch_ops =
          collect_branch_ops(consumer_plugin_index, &ingress_contract.join_plugin_index);
      if (ingress_contract.branch_ops.empty()) {
        const auto* consumer_node =
            pipeline_internal::sima::route_graph_node(graph, consumer_plugin_index);
        if (consumer_node &&
            consumer_node->kind == pipeline_internal::sima::RouteGraphKernelKind::Mla) {
          ingress_contract.join_plugin_index = consumer_plugin_index;
        }
      }
    } else {
      const auto shape = preferred_tensor_shape(ingress);
      const auto parsed_shape = parse_tensor_shape_contract(shape);
      ingress_contract.valid = parsed_shape.valid;
      ingress_contract.media_type = "application/vnd.simaai.tensor";
      ingress_contract.dtype = preferred_tensor_dtype(ingress);
      ingress_contract.layout = parsed_shape.layout;
      ingress_contract.rank = parsed_shape.rank;
      ingress_contract.batch = parsed_shape.batch;
      ingress_contract.width = parsed_shape.width;
      ingress_contract.height = parsed_shape.height;
      ingress_contract.depth = parsed_shape.depth;
      // Phase 4a: preserve canonical shape vector.
      ingress_contract.logical_shape = shape;
      ingress_contract.source_stage = "session_ingress";
    }

    out.push_back(std::move(ingress_contract));
  }
  return out;
}

std::vector<RouteRegion>
ingress_regions_from_contracts(const std::vector<IngressTensorContract>& ingress_contracts) {
  const auto graph_kind_from_ordered_route_kind =
      [](OrderedRouteOp::Kind kind) -> pipeline_internal::sima::RouteGraphKernelKind {
    using GraphKind = pipeline_internal::sima::RouteGraphKernelKind;
    switch (kind) {
    case OrderedRouteOp::Kind::Preproc:
      return GraphKind::Preproc;
    case OrderedRouteOp::Kind::Quant:
      return GraphKind::Quant;
    case OrderedRouteOp::Kind::Tess:
      return GraphKind::Tess;
    case OrderedRouteOp::Kind::QuantTess:
      return GraphKind::QuantTess;
    case OrderedRouteOp::Kind::Cast:
      return GraphKind::Cast;
    case OrderedRouteOp::Kind::CastTess:
      return GraphKind::CastTess;
    case OrderedRouteOp::Kind::Detess:
      return GraphKind::Detess;
    case OrderedRouteOp::Kind::DetessCast:
      return GraphKind::DetessCast;
    case OrderedRouteOp::Kind::DetessDequant:
      return GraphKind::DetessDequant;
    case OrderedRouteOp::Kind::Dequantize:
      return GraphKind::Dequantize;
    case OrderedRouteOp::Kind::BoxDecode:
      return GraphKind::BoxDecode;
    case OrderedRouteOp::Kind::Unpack:
      return GraphKind::Unpack;
    case OrderedRouteOp::Kind::Unknown:
      break;
    }
    return GraphKind::Unknown;
  };

  std::vector<RouteRegion> regions;
  if (ingress_contracts.empty()) {
    return regions;
  }

  std::size_t max_branch_depth = 0U;
  for (const auto& ingress : ingress_contracts) {
    max_branch_depth = std::max(max_branch_depth, ingress.branch_ops.size());
  }

  for (std::size_t depth = 0U; depth < max_branch_depth; ++depth) {
    std::vector<const IngressTensorContract*> members;
    members.reserve(ingress_contracts.size());
    for (const auto& ingress : ingress_contracts) {
      if (ingress.branch_ops.size() > depth) {
        members.push_back(&ingress);
      }
    }
    if (members.empty()) {
      continue;
    }

    RouteRegion region;
    region.kind = members.size() > 1U ? RouteRegionKind::FanoutMap : RouteRegionKind::Linear;
    region.op_kind = graph_kind_from_ordered_route_kind(members.front()->branch_ops[depth].kind);
    for (const auto* ingress : members) {
      region.member_plugin_indices.push_back(ingress->dst_plugin_index);
      RouteTensorBinding binding;
      binding.dst_plugin_index = ingress->dst_plugin_index;
      binding.dst_tensor_name = ingress->source_tensor_name;
      binding.logical_index = ingress->ingress_index;
      binding.segment_name = ingress->source_tensor_name;
      region.inputs.push_back(std::move(binding));
    }
    if (!region.member_plugin_indices.empty()) {
      region.producer_plugin_index = region.member_plugin_indices.front();
    }
    regions.push_back(std::move(region));
  }

  if (ingress_contracts.size() > 1U) {
    std::optional<std::size_t> common_join_plugin_index;
    bool common_join_valid = true;
    for (const auto& ingress : ingress_contracts) {
      const std::optional<std::size_t> join_candidate =
          ingress.join_plugin_index.has_value()
              ? ingress.join_plugin_index
              : (ingress.branch_ops.empty() ? std::optional<std::size_t>(ingress.dst_plugin_index)
                                            : std::nullopt);
      if (!join_candidate.has_value()) {
        common_join_valid = false;
        break;
      }
      if (!common_join_plugin_index.has_value()) {
        common_join_plugin_index = join_candidate;
      } else if (*common_join_plugin_index != *join_candidate) {
        common_join_valid = false;
        break;
      }
    }

    if (common_join_valid && common_join_plugin_index.has_value()) {
      RouteRegion join_region;
      join_region.kind = RouteRegionKind::FaninJoin;
      join_region.op_kind = pipeline_internal::sima::RouteGraphKernelKind::PassThrough;
      join_region.join_plugin_index = common_join_plugin_index;
      join_region.producer_plugin_index = *common_join_plugin_index;
      for (const auto& ingress : ingress_contracts) {
        RouteTensorBinding binding;
        binding.dst_plugin_index = *common_join_plugin_index;
        binding.dst_tensor_name = ingress.source_tensor_name;
        binding.logical_index = ingress.ingress_index;
        binding.segment_name = ingress.source_tensor_name;
        join_region.inputs.push_back(std::move(binding));
      }
      regions.push_back(std::move(join_region));
    }
  }
  return regions;
}

std::string
ingress_contracts_debug_string(const std::vector<IngressTensorContract>& ingress_contracts) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < ingress_contracts.size(); ++i) {
    if (i) {
      oss << ", ";
    }
    const auto& ingress = ingress_contracts[i];
    oss << "{idx=" << ingress.ingress_index << ",dtype=" << ingress.dtype
        << ",layout=" << ingress.layout << ",shape=" << ingress.height << "x" << ingress.width
        << "x" << ingress.depth << ",source=" << ingress.source_stage
        << ",name=" << ingress.source_tensor_name << "}";
  }
  oss << "]";
  return oss.str();
}

IngressTensorContract make_single_ingress_contract_from_semantics(const ModelSemantics& semantics) {
  IngressTensorContract ingress;
  ingress.valid = true;
  ingress.media_type = "application/vnd.simaai.tensor";
  ingress.dtype = canonical_dtype_for_signal(semantics.mla_input_dtype_raw);
  ingress.rank = 4;
  ingress.batch = 1;
  ingress.width = semantics.mla_input_dims.width;
  ingress.height = semantics.mla_input_dims.height;
  ingress.depth = semantics.mla_input_dims.depth;
  ingress.source_stage = "mla_input";
  ingress.ingress_index = 0;
  ingress.source_tensor_name = "ifm0";
  return ingress;
}

bool populate_route_mla_facts_from_mpk_contract(
    const pipeline_internal::sima::MpkContract& contract, RouteCapability* out,
    std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = "output is null";
    }
    return false;
  }

  const auto* mla_stage = pipeline_internal::sima::get_mla_stage_io_contract(contract);
  if (!mla_stage) {
    if (error_message) {
      *error_message = "MPK contract is missing an MLA stage";
    }
    return false;
  }

  const auto boundary_inputs =
      pipeline_internal::sima::get_mla_boundary_physical_inputs_contract(contract);

  const std::vector<pipeline_internal::sima::MpkTensorContract> published_outputs =
      pipeline_internal::sima::get_mla_published_outputs_contract(contract);
  const std::vector<pipeline_internal::sima::MpkTensorContract> logical_outputs =
      pipeline_internal::sima::get_mla_logical_outputs_contract(contract);
  const std::vector<pipeline_internal::sima::MpkTensorContract> physical_outputs =
      pipeline_internal::sima::get_mla_boundary_physical_outputs_contract(contract);
  const std::vector<pipeline_internal::sima::MpkTensorContract>& planning_outputs =
      !published_outputs.empty()
          ? published_outputs
          : (logical_outputs.empty() ? mla_stage->output_tensors : logical_outputs);
  const std::string node_name_hint =
      !mla_stage->name.empty() ? mla_stage->name : mla_stage->plugin_id;
  const auto mla_contract = pipeline_internal::sima::build_mla_static_contract_from_mpk_stage(
      *mla_stage, planning_outputs,
      physical_outputs.empty() ? mla_stage->output_tensors : physical_outputs, node_name_hint,
      boundary_inputs.empty() ? nullptr : &boundary_inputs);

  if (mla_contract.logical_inputs.empty()) {
    if (error_message) {
      *error_message = "typed MLA contract is missing logical_inputs";
    }
    return false;
  }
  if (mla_contract.logical_outputs.empty()) {
    if (error_message) {
      *error_message = "typed MLA contract is missing logical_outputs";
    }
    return false;
  }

  const auto& logical_input = mla_contract.logical_inputs.front();
  const auto& logical_output = mla_contract.logical_outputs.front();

  out->mla_input_dtype_raw = canonical_dtype_for_signal(logical_input.dtype);
  out->mla_output_dtype_raw = canonical_dtype_for_signal(logical_output.dtype);
  out->mla_input_media_type = "application/vnd.simaai.tensor";

  const ParsedTensorShapeContract parsed_input_shape =
      parse_tensor_shape_contract(logical_input.shape);
  const ParsedTensorShapeContract parsed_output_shape =
      parse_tensor_shape_contract(logical_output.shape);

  out->mla_input_dims.width = parsed_input_shape.width;
  out->mla_input_dims.height = parsed_input_shape.height;
  out->mla_input_dims.depth = parsed_input_shape.depth;
  out->mla_output_dims.width = parsed_output_shape.width;
  out->mla_output_dims.height = parsed_output_shape.height;
  out->mla_output_dims.depth = parsed_output_shape.depth;

  const auto shape_dims_valid = [](const std::vector<std::int64_t>& shape) {
    return !shape.empty() &&
           std::all_of(shape.begin(), shape.end(), [](const std::int64_t dim) { return dim > 0; });
  };

  const bool input_ok = !out->mla_input_dtype_raw.empty() && !out->mla_input_media_type.empty() &&
                        shape_dims_valid(logical_input.shape);
  const bool output_ok =
      !out->mla_output_dtype_raw.empty() && shape_dims_valid(logical_output.shape);
  if (!input_ok || !output_ok) {
    if (error_message) {
      *error_message = "typed MLA contract is missing required logical tensor shape/dtype facts";
    }
    return false;
  }

  return true;
}

void append_ordered_route_op(std::vector<OrderedRouteOp>* dst, OrderedRouteOp::Kind kind,
                             const pipeline_internal::sima::MpkPluginIoContract& plugin,
                             bool before_mla, bool after_mla) {
  if (!dst || kind == OrderedRouteOp::Kind::Unknown) {
    return;
  }
  OrderedRouteOp op;
  op.kind = kind;
  op.plugin_name = plugin.name;
  op.plugin_id = plugin.plugin_id;
  op.kernel = plugin.kernel;
  op.sequence = plugin.sequence;
  op.before_mla = before_mla;
  op.after_mla = after_mla;
  dst->push_back(std::move(op));
}

std::optional<SessionPreStageOp> session_pre_op_from_ordered_kind(OrderedRouteOp::Kind kind) {
  switch (kind) {
  case OrderedRouteOp::Kind::Preproc:
    return SessionPreStageOp::Preproc;
  case OrderedRouteOp::Kind::Quant:
    return SessionPreStageOp::Quant;
  case OrderedRouteOp::Kind::Tess:
    return SessionPreStageOp::Tess;
  case OrderedRouteOp::Kind::QuantTess:
    return SessionPreStageOp::QuantTess;
  case OrderedRouteOp::Kind::Cast:
    return SessionPreStageOp::Cast;
  case OrderedRouteOp::Kind::CastTess:
    return SessionPreStageOp::CastTess;
  case OrderedRouteOp::Kind::Unknown:
  case OrderedRouteOp::Kind::Detess:
  case OrderedRouteOp::Kind::DetessCast:
  case OrderedRouteOp::Kind::DetessDequant:
  case OrderedRouteOp::Kind::Dequantize:
  case OrderedRouteOp::Kind::BoxDecode:
  case OrderedRouteOp::Kind::Unpack:
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<SessionPostStageOp> session_post_op_from_ordered_kind(OrderedRouteOp::Kind kind) {
  switch (kind) {
  case OrderedRouteOp::Kind::Detess:
    return SessionPostStageOp::Detess;
  case OrderedRouteOp::Kind::DetessDequant:
    return SessionPostStageOp::DetessDequant;
  case OrderedRouteOp::Kind::Dequantize:
    return SessionPostStageOp::Dequantize;
  case OrderedRouteOp::Kind::BoxDecode:
    return SessionPostStageOp::BoxDecode;
  case OrderedRouteOp::Kind::Cast:
    return SessionPostStageOp::Cast;
  case OrderedRouteOp::Kind::DetessCast:
    return SessionPostStageOp::DetessCast;
  case OrderedRouteOp::Kind::Unpack:
  case OrderedRouteOp::Kind::Unknown:
  case OrderedRouteOp::Kind::Preproc:
  case OrderedRouteOp::Kind::Quant:
  case OrderedRouteOp::Kind::Tess:
  case OrderedRouteOp::Kind::QuantTess:
  case OrderedRouteOp::Kind::CastTess:
    return std::nullopt;
  }
  return std::nullopt;
}

const char* session_pre_stage_op_name(SessionPreStageOp op) {
  switch (op) {
  case SessionPreStageOp::Preproc:
    return "preproc";
  case SessionPreStageOp::Quant:
    return "quant";
  case SessionPreStageOp::Tess:
    return "tess";
  case SessionPreStageOp::QuantTess:
    return "quanttess";
  case SessionPreStageOp::Cast:
    return "cast";
  case SessionPreStageOp::CastTess:
    return "casttess";
  }
  return "unknown";
}

const char* session_post_stage_op_name(SessionPostStageOp op) {
  switch (op) {
  case SessionPostStageOp::Detess:
    return "detess";
  case SessionPostStageOp::DetessDequant:
    return "detessdequant";
  case SessionPostStageOp::Dequantize:
    return "dequantize";
  case SessionPostStageOp::BoxDecode:
    return "boxdecode";
  case SessionPostStageOp::Cast:
    return "cast";
  case SessionPostStageOp::DetessCast:
    return "detesscast";
  }
  return "unknown";
}

std::string session_pre_chain_csv(const std::vector<SessionPreStageOp>& ops) {
  if (ops.empty()) {
    return "none";
  }
  std::string out;
  for (std::size_t i = 0; i < ops.size(); ++i) {
    if (!out.empty()) {
      out += ",";
    }
    out += session_pre_stage_op_name(ops[i]);
  }
  return out;
}

std::optional<OrderedRouteOp::Kind> ordered_pre_kind_from_session_pre_stage(SessionPreStageOp op) {
  switch (op) {
  case SessionPreStageOp::Preproc:
    return OrderedRouteOp::Kind::Preproc;
  case SessionPreStageOp::Quant:
    return OrderedRouteOp::Kind::Quant;
  case SessionPreStageOp::Tess:
    return OrderedRouteOp::Kind::Tess;
  case SessionPreStageOp::QuantTess:
    return OrderedRouteOp::Kind::QuantTess;
  case SessionPreStageOp::Cast:
    return OrderedRouteOp::Kind::Cast;
  case SessionPreStageOp::CastTess:
    return OrderedRouteOp::Kind::CastTess;
  }
  return std::nullopt;
}

std::vector<OrderedRouteOp>
normalize_ingress_branch_ops_for_pre_chain(const std::vector<OrderedRouteOp>& branch_ops,
                                           const std::vector<SessionPreStageOp>& pre_chain) {
  if (pre_chain.empty()) {
    return {};
  }

  std::unordered_map<OrderedRouteOp::Kind, std::size_t> allowed_counts;
  allowed_counts.reserve(pre_chain.size());
  for (const auto stage : pre_chain) {
    const auto kind = ordered_pre_kind_from_session_pre_stage(stage);
    if (kind.has_value()) {
      ++allowed_counts[*kind];
    }
  }

  std::vector<OrderedRouteOp> normalized;
  normalized.reserve(std::min(branch_ops.size(), pre_chain.size()));
  for (std::size_t cursor = 0U; cursor < branch_ops.size(); ++cursor) {
    const auto& op = branch_ops[cursor];

    const auto fused_it = allowed_counts.find(OrderedRouteOp::Kind::QuantTess);
    const std::size_t fused_used = static_cast<std::size_t>(std::count_if(
        normalized.begin(), normalized.end(), [](const OrderedRouteOp& normalized_op) {
          return normalized_op.kind == OrderedRouteOp::Kind::QuantTess;
        }));
    if (fused_it != allowed_counts.end() && fused_used < fused_it->second &&
        cursor + 1U < branch_ops.size() && op.kind == OrderedRouteOp::Kind::Quant &&
        branch_ops[cursor + 1U].kind == OrderedRouteOp::Kind::Tess) {
      const auto& downstream = branch_ops[cursor + 1U];
      OrderedRouteOp synthetic;
      synthetic.kind = OrderedRouteOp::Kind::QuantTess;
      synthetic.plugin_name =
          !downstream.plugin_name.empty() ? downstream.plugin_name : op.plugin_name;
      synthetic.plugin_id = !downstream.plugin_id.empty() ? downstream.plugin_id : op.plugin_id;
      synthetic.kernel = !downstream.kernel.empty() ? downstream.kernel : op.kernel;
      synthetic.before_mla = op.before_mla || branch_ops[cursor + 1U].before_mla;
      synthetic.after_mla = op.after_mla || branch_ops[cursor + 1U].after_mla;
      synthetic.sequence = downstream.sequence >= 0 ? downstream.sequence : op.sequence;
      normalized.push_back(std::move(synthetic));
      ++cursor;
      continue;
    }

    const auto casttess_it = allowed_counts.find(OrderedRouteOp::Kind::CastTess);
    const std::size_t casttess_used = static_cast<std::size_t>(std::count_if(
        normalized.begin(), normalized.end(), [](const OrderedRouteOp& normalized_op) {
          return normalized_op.kind == OrderedRouteOp::Kind::CastTess;
        }));
    if (casttess_it != allowed_counts.end() && casttess_used < casttess_it->second &&
        cursor + 1U < branch_ops.size() && op.kind == OrderedRouteOp::Kind::Cast &&
        branch_ops[cursor + 1U].kind == OrderedRouteOp::Kind::Tess) {
      const auto& downstream = branch_ops[cursor + 1U];
      OrderedRouteOp synthetic;
      synthetic.kind = OrderedRouteOp::Kind::CastTess;
      synthetic.plugin_name =
          !downstream.plugin_name.empty() ? downstream.plugin_name : op.plugin_name;
      synthetic.plugin_id = !downstream.plugin_id.empty() ? downstream.plugin_id : op.plugin_id;
      synthetic.kernel = !downstream.kernel.empty() ? downstream.kernel : op.kernel;
      synthetic.before_mla = op.before_mla || downstream.before_mla;
      synthetic.after_mla = op.after_mla || downstream.after_mla;
      synthetic.sequence = downstream.sequence >= 0 ? downstream.sequence : op.sequence;
      normalized.push_back(std::move(synthetic));
      ++cursor;
      continue;
    }

    const auto allowed_it = allowed_counts.find(op.kind);
    if (allowed_it == allowed_counts.end()) {
      continue;
    }
    const std::size_t used = static_cast<std::size_t>(std::count_if(
        normalized.begin(), normalized.end(),
        [&](const OrderedRouteOp& normalized_op) { return normalized_op.kind == op.kind; }));
    if (used >= allowed_it->second) {
      continue;
    }
    normalized.push_back(op);
  }
  return normalized;
}

void normalize_ingress_contracts_for_pre_chain(
    std::vector<IngressTensorContract>* ingress_contracts,
    const std::vector<SessionPreStageOp>& pre_chain) {
  if (!ingress_contracts) {
    return;
  }
  for (auto& ingress : *ingress_contracts) {
    ingress.branch_ops = normalize_ingress_branch_ops_for_pre_chain(ingress.branch_ops, pre_chain);
  }
}

std::string session_post_chain_csv(const std::vector<SessionPostStageOp>& ops) {
  if (ops.empty()) {
    return "none";
  }
  std::string out;
  for (std::size_t i = 0; i < ops.size(); ++i) {
    if (!out.empty()) {
      out += ",";
    }
    out += session_post_stage_op_name(ops[i]);
  }
  return out;
}

template <typename OpT>
bool collapse_periodic_chain(std::vector<OpT>* ops, std::size_t* out_period = nullptr) {
  if (!ops || ops->size() < 2U) {
    return false;
  }
  const std::size_t n = ops->size();
  for (std::size_t period = 1U; period <= n / 2U; ++period) {
    if ((n % period) != 0U) {
      continue;
    }
    bool periodic = true;
    for (std::size_t i = period; i < n; ++i) {
      if ((*ops)[i] != (*ops)[i % period]) {
        periodic = false;
        break;
      }
    }
    if (!periodic) {
      continue;
    }
    if (out_period) {
      *out_period = period;
    }
    ops->erase(ops->begin() + static_cast<std::ptrdiff_t>(period), ops->end());
    return true;
  }
  return false;
}

template <typename OpT>
bool collapse_adjacent_duplicates(std::vector<OpT>* ops, std::size_t* out_removed = nullptr) {
  if (!ops || ops->size() < 2U) {
    return false;
  }
  std::vector<OpT> compact;
  compact.reserve(ops->size());
  compact.push_back((*ops)[0]);
  for (std::size_t i = 1; i < ops->size(); ++i) {
    if ((*ops)[i] != compact.back()) {
      compact.push_back((*ops)[i]);
    }
  }
  if (compact.size() == ops->size()) {
    return false;
  }
  if (out_removed) {
    *out_removed = ops->size() - compact.size();
  }
  *ops = std::move(compact);
  return true;
}

template <typename OpT> bool has_duplicate_ops(const std::vector<OpT>& ops) {
  std::unordered_set<int> seen;
  seen.reserve(ops.size());
  for (const auto op : ops) {
    if (!seen.insert(static_cast<int>(op)).second) {
      return true;
    }
  }
  return false;
}

const char* route_graph_op_name(pipeline_internal::sima::RouteGraphKernelKind kind) {
  return pipeline_internal::sima::route_graph_kernel_name(kind);
}

bool is_graph_pre_stage_kind(const pipeline_internal::sima::RouteGraphKernelKind kind) {
  using Kind = pipeline_internal::sima::RouteGraphKernelKind;
  switch (kind) {
  case Kind::Preproc:
  case Kind::Quant:
  case Kind::Tess:
  case Kind::QuantTess:
  case Kind::Cast:
  case Kind::CastTess:
    return true;
  case Kind::Unknown:
  case Kind::Detess:
  case Kind::DetessDequant:
  case Kind::Dequantize:
  case Kind::BoxDecode:
  case Kind::Unpack:
  case Kind::Slice:
  case Kind::PassThrough:
  case Kind::Mla:
    return false;
  }
  return false;
}

bool is_graph_post_stage_kind(const pipeline_internal::sima::RouteGraphKernelKind kind) {
  using Kind = pipeline_internal::sima::RouteGraphKernelKind;
  switch (kind) {
  case Kind::Detess:
  case Kind::DetessCast:
  case Kind::DetessDequant:
  case Kind::Dequantize:
  case Kind::BoxDecode:
  case Kind::Cast:
  case Kind::Slice:
    return true;
  case Kind::Unknown:
  case Kind::Preproc:
  case Kind::Quant:
  case Kind::Tess:
  case Kind::QuantTess:
  case Kind::CastTess:
  case Kind::Unpack:
  case Kind::PassThrough:
  case Kind::Mla:
    return false;
  }
  return false;
}

std::optional<SessionPreStageOp>
session_pre_op_from_graph_kind(const pipeline_internal::sima::RouteGraphKernelKind kind) {
  using Kind = pipeline_internal::sima::RouteGraphKernelKind;
  switch (kind) {
  case Kind::Preproc:
    return SessionPreStageOp::Preproc;
  case Kind::Quant:
    return SessionPreStageOp::Quant;
  case Kind::Tess:
    return SessionPreStageOp::Tess;
  case Kind::QuantTess:
    return SessionPreStageOp::QuantTess;
  case Kind::Cast:
    return SessionPreStageOp::Cast;
  case Kind::CastTess:
    return SessionPreStageOp::CastTess;
  case Kind::Unknown:
  case Kind::Detess:
  case Kind::DetessDequant:
  case Kind::Dequantize:
  case Kind::BoxDecode:
  case Kind::Unpack:
  case Kind::Slice:
  case Kind::PassThrough:
  case Kind::Mla:
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<SessionPostStageOp>
session_post_op_from_graph_kind(const pipeline_internal::sima::RouteGraphKernelKind kind) {
  using Kind = pipeline_internal::sima::RouteGraphKernelKind;
  switch (kind) {
  case Kind::Detess:
    return SessionPostStageOp::Detess;
  case Kind::DetessCast:
    return SessionPostStageOp::DetessCast;
  case Kind::DetessDequant:
    return SessionPostStageOp::DetessDequant;
  case Kind::Dequantize:
    return SessionPostStageOp::Dequantize;
  case Kind::BoxDecode:
    return SessionPostStageOp::BoxDecode;
  case Kind::Cast:
    return SessionPostStageOp::Cast;
  case Kind::Unknown:
  case Kind::Preproc:
  case Kind::Quant:
  case Kind::Tess:
  case Kind::QuantTess:
  case Kind::CastTess:
  case Kind::Unpack:
  case Kind::Slice:
  case Kind::PassThrough:
  case Kind::Mla:
    return std::nullopt;
  }
  return std::nullopt;
}

PostRouteStageKind
post_route_kind_from_graph_kind(const pipeline_internal::sima::RouteGraphKernelKind kind) {
  using Kind = pipeline_internal::sima::RouteGraphKernelKind;
  switch (kind) {
  case Kind::Detess:
    return PostRouteStageKind::Detess;
  case Kind::DetessCast:
    return PostRouteStageKind::DetessCast;
  case Kind::DetessDequant:
    return PostRouteStageKind::DetessDequant;
  case Kind::Dequantize:
    return PostRouteStageKind::Dequantize;
  case Kind::BoxDecode:
    return PostRouteStageKind::BoxDecode;
  case Kind::Cast:
    return PostRouteStageKind::Cast;
  case Kind::Unknown:
  case Kind::Preproc:
  case Kind::Quant:
  case Kind::Tess:
  case Kind::QuantTess:
  case Kind::CastTess:
  case Kind::Unpack:
  case Kind::Slice:
  case Kind::PassThrough:
  case Kind::Mla:
    return PostRouteStageKind::Unknown;
  }
  return PostRouteStageKind::Unknown;
}

SessionPostAdapterKind post_adapter_from_post_route_kind(const PostRouteStageKind kind) {
  switch (kind) {
  case PostRouteStageKind::Detess:
    return SessionPostAdapterKind::Detess;
  case PostRouteStageKind::DetessDequant:
    return SessionPostAdapterKind::DetessDequant;
  case PostRouteStageKind::Dequantize:
    return SessionPostAdapterKind::Dequant;
  case PostRouteStageKind::BoxDecode:
    return SessionPostAdapterKind::BoxDecode;
  case PostRouteStageKind::Cast:
    return SessionPostAdapterKind::Cast;
  case PostRouteStageKind::DetessCast:
    return SessionPostAdapterKind::DetessCast;
  case PostRouteStageKind::None:
  case PostRouteStageKind::Unknown:
    return SessionPostAdapterKind::None;
  }
  return SessionPostAdapterKind::None;
}

RouteTensorBinding
route_tensor_binding_from_edge(const pipeline_internal::sima::RouteGraphEdge& edge) {
  RouteTensorBinding binding;
  binding.src_plugin_index = edge.src_plugin_index;
  binding.dst_plugin_index = edge.dst_plugin_index;
  binding.src_tensor_name = !edge.src_tensor_name.empty() ? edge.src_tensor_name : edge.tensor_name;
  binding.dst_tensor_name = !edge.dst_tensor_name.empty() ? edge.dst_tensor_name : edge.tensor_name;
  binding.logical_index = edge.src_output_index;
  binding.physical_index = edge.src_output_index;
  binding.segment_name = binding.src_tensor_name;
  return binding;
}

std::vector<int>
unique_logical_indices_from_bindings(const std::vector<RouteTensorBinding>& bindings) {
  std::vector<int> logical_indices;
  logical_indices.reserve(bindings.size());
  for (const auto& binding : bindings) {
    if (binding.logical_index < 0) {
      continue;
    }
    if (std::find(logical_indices.begin(), logical_indices.end(), binding.logical_index) !=
        logical_indices.end()) {
      continue;
    }
    logical_indices.push_back(binding.logical_index);
  }
  return logical_indices;
}

void sort_route_bindings_by_logical_index(std::vector<RouteTensorBinding>* bindings) {
  if (!bindings) {
    return;
  }
  std::stable_sort(bindings->begin(), bindings->end(),
                   [](const RouteTensorBinding& lhs, const RouteTensorBinding& rhs) {
                     const bool lhs_has_logical = lhs.logical_index >= 0;
                     const bool rhs_has_logical = rhs.logical_index >= 0;
                     if (lhs_has_logical != rhs_has_logical) {
                       return lhs_has_logical;
                     }
                     if (lhs_has_logical && rhs_has_logical &&
                         lhs.logical_index != rhs.logical_index) {
                       return lhs.logical_index < rhs.logical_index;
                     }
                     if (lhs.src_plugin_index != rhs.src_plugin_index) {
                       return lhs.src_plugin_index < rhs.src_plugin_index;
                     }
                     if (lhs.dst_plugin_index != rhs.dst_plugin_index) {
                       return lhs.dst_plugin_index < rhs.dst_plugin_index;
                     }
                     if (lhs.src_tensor_name != rhs.src_tensor_name) {
                       return lhs.src_tensor_name < rhs.src_tensor_name;
                     }
                     return lhs.dst_tensor_name < rhs.dst_tensor_name;
                   });
}

bool route_region_bindings_are_lineage_compatible(const std::vector<RouteTensorBinding>& lhs,
                                                  const std::vector<RouteTensorBinding>& rhs) {
  const auto lhs_indices = unique_logical_indices_from_bindings(lhs);
  const auto rhs_indices = unique_logical_indices_from_bindings(rhs);
  if (!lhs_indices.empty() && !rhs_indices.empty()) {
    return lhs_indices == rhs_indices;
  }
  return lhs.size() == rhs.size();
}

bool can_fuse_detess_and_dequant_regions(const RouteRegion& detess, const RouteRegion& dequant) {
  using GraphKind = pipeline_internal::sima::RouteGraphKernelKind;
  if (detess.op_kind != GraphKind::Detess || dequant.op_kind != GraphKind::Dequantize) {
    return false;
  }
  if (detess.kind != dequant.kind) {
    return false;
  }
  switch (detess.kind) {
  case RouteRegionKind::Linear:
  case RouteRegionKind::FanoutMap:
    break;
  case RouteRegionKind::FaninJoin:
  case RouteRegionKind::BoxDecodeTerminal:
    return false;
  }
  if (!detess.member_plugin_indices.empty() && !dequant.member_plugin_indices.empty() &&
      detess.member_plugin_indices.size() != dequant.member_plugin_indices.size()) {
    return false;
  }
  return route_region_bindings_are_lineage_compatible(detess.outputs, dequant.inputs);
}

bool can_fuse_detess_and_cast_regions(const RouteRegion& detess, const RouteRegion& cast) {
  using GraphKind = pipeline_internal::sima::RouteGraphKernelKind;
  if (detess.op_kind != GraphKind::Detess || cast.op_kind != GraphKind::Cast) {
    return false;
  }
  if (detess.kind != cast.kind) {
    return false;
  }
  switch (detess.kind) {
  case RouteRegionKind::Linear:
  case RouteRegionKind::FanoutMap:
    break;
  case RouteRegionKind::FaninJoin:
  case RouteRegionKind::BoxDecodeTerminal:
    return false;
  }
  if (!detess.member_plugin_indices.empty() && !cast.member_plugin_indices.empty() &&
      detess.member_plugin_indices.size() != cast.member_plugin_indices.size()) {
    return false;
  }
  return route_region_bindings_are_lineage_compatible(detess.outputs, cast.inputs);
}

RouteRegion
fuse_adjacent_post_regions(const RouteRegion& lhs, const RouteRegion& rhs,
                           const pipeline_internal::sima::RouteGraphKernelKind fused_kind) {
  RouteRegion fused;
  fused.kind = lhs.kind;
  fused.op_kind = fused_kind;
  fused.producer_plugin_index = lhs.producer_plugin_index;
  fused.member_plugin_indices =
      !rhs.member_plugin_indices.empty() ? rhs.member_plugin_indices : lhs.member_plugin_indices;
  fused.join_plugin_index =
      rhs.join_plugin_index.has_value() ? rhs.join_plugin_index : lhs.join_plugin_index;
  fused.inputs = lhs.inputs;
  fused.outputs = rhs.outputs;
  fused.egress_contracts =
      !rhs.egress_contracts.empty() ? rhs.egress_contracts : lhs.egress_contracts;
  return fused;
}

std::vector<EgressTensorContract>
stage_output_contracts_from_plugin(const pipeline_internal::sima::MpkPluginIoContract& plugin) {
  std::vector<EgressTensorContract> outputs;
  const std::string source_stage = !plugin.name.empty() ? plugin.name : plugin.plugin_id;
  for (const auto& tensor : plugin.output_tensors) {
    EgressTensorContract out;
    if (populate_tensor_contract_from_stage_tensor(tensor, "application/vnd.simaai.tensor",
                                                   source_stage, &out)) {
      outputs.push_back(std::move(out));
    }
  }
  return outputs;
}

std::string graph_node_source_stage_name_local(const pipeline_internal::sima::MpkGraphNode& node) {
  if (!node.label.empty()) {
    return node.label;
  }
  if (!node.name.empty()) {
    return node.name;
  }
  if (!node.plugin_id.empty()) {
    return node.plugin_id;
  }
  if (!node.node_id.empty()) {
    return node.node_id;
  }
  return "graph_stage";
}

pipeline_internal::sima::RouteGraphKernelKind
route_graph_kind_from_mpk_node_local(const pipeline_internal::sima::MpkGraphNode& node) {
  std::string kernel_source = !node.canonical_op.empty() ? node.canonical_op : node.kernel;
  if (kernel_source.empty()) {
    kernel_source = node.name;
  }
  auto kind = pipeline_internal::sima::canonical_route_graph_kernel_kind(kernel_source);
  if (kind == pipeline_internal::sima::RouteGraphKernelKind::Unknown &&
      lower_copy(node.processor).find("mla") != std::string::npos) {
    kind = pipeline_internal::sima::RouteGraphKernelKind::Mla;
  }
  return kind;
}

int tensor_name_position_local(const std::vector<std::string>& names, const std::string& name) {
  if (name.empty()) {
    return -1;
  }
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (names[i] == name) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

std::optional<EgressTensorContract>
egress_contract_for_tensor_name_local(const pipeline_internal::sima::MpkContract& contract,
                                      const std::string& tensor_name,
                                      const std::string& source_stage) {
  if (tensor_name.empty()) {
    return std::nullopt;
  }
  for (const auto& plugin : contract.plugins) {
    for (const auto& tensor : plugin.output_tensors) {
      if (tensor.name != tensor_name) {
        continue;
      }
      EgressTensorContract out;
      if (populate_tensor_contract_from_stage_tensor(tensor, "application/vnd.simaai.tensor",
                                                     source_stage, &out)) {
        return out;
      }
    }
  }
  return std::nullopt;
}

std::vector<EgressTensorContract>
stage_output_contracts_from_graph_node_local(const pipeline_internal::sima::MpkContract& contract,
                                             const pipeline_internal::sima::MpkGraphNode& node) {
  if (node.plugin_index != static_cast<std::size_t>(-1) &&
      node.plugin_index < contract.plugins.size()) {
    return stage_output_contracts_from_plugin(contract.plugins[node.plugin_index]);
  }

  std::vector<EgressTensorContract> outputs;
  std::unordered_set<std::string> seen;
  const std::string source_stage = graph_node_source_stage_name_local(node);
  for (const auto& tensor_name : node.output_tensor_names) {
    if (tensor_name.empty() || !seen.insert(tensor_name).second) {
      continue;
    }
    if (const auto output =
            egress_contract_for_tensor_name_local(contract, tensor_name, source_stage);
        output.has_value()) {
      outputs.push_back(*output);
    }
  }
  return outputs;
}

struct MpkPostGraphView {
  std::unordered_map<std::string, std::size_t> node_index_by_id;
  std::vector<std::vector<std::size_t>> incoming_edges;
  std::vector<std::vector<std::size_t>> outgoing_edges;
  std::vector<bool> reachable_from_mla;
  std::size_t mla_node_index = std::numeric_limits<std::size_t>::max();
};

MpkPostGraphView build_mpk_post_graph_view_local(const pipeline_internal::sima::MpkGraph& graph) {
  MpkPostGraphView view;
  view.node_index_by_id.reserve(graph.nodes.size());
  view.incoming_edges.resize(graph.nodes.size());
  view.outgoing_edges.resize(graph.nodes.size());
  view.reachable_from_mla.assign(graph.nodes.size(), false);

  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    view.node_index_by_id.emplace(graph.nodes[i].node_id, i);
    if (view.mla_node_index == std::numeric_limits<std::size_t>::max() &&
        route_graph_kind_from_mpk_node_local(graph.nodes[i]) ==
            pipeline_internal::sima::RouteGraphKernelKind::Mla) {
      view.mla_node_index = i;
    }
  }
  for (std::size_t edge_index = 0; edge_index < graph.edges.size(); ++edge_index) {
    const auto& edge = graph.edges[edge_index];
    const auto src_it = view.node_index_by_id.find(edge.src_node_id);
    const auto dst_it = view.node_index_by_id.find(edge.dst_node_id);
    if (src_it == view.node_index_by_id.end() || dst_it == view.node_index_by_id.end()) {
      continue;
    }
    view.outgoing_edges[src_it->second].push_back(edge_index);
    view.incoming_edges[dst_it->second].push_back(edge_index);
  }

  if (view.mla_node_index == std::numeric_limits<std::size_t>::max()) {
    return view;
  }

  std::vector<std::size_t> stack = {view.mla_node_index};
  view.reachable_from_mla[view.mla_node_index] = true;
  while (!stack.empty()) {
    const auto current = stack.back();
    stack.pop_back();
    for (const auto edge_index : view.outgoing_edges[current]) {
      if (edge_index >= graph.edges.size()) {
        continue;
      }
      const auto dst_it = view.node_index_by_id.find(graph.edges[edge_index].dst_node_id);
      if (dst_it == view.node_index_by_id.end() || view.reachable_from_mla[dst_it->second]) {
        continue;
      }
      view.reachable_from_mla[dst_it->second] = true;
      stack.push_back(dst_it->second);
    }
  }

  return view;
}

RouteTensorBinding
route_tensor_binding_from_mpk_graph_edge_local(const pipeline_internal::sima::MpkGraphEdge& edge,
                                               const MpkPostGraphView& view) {
  RouteTensorBinding binding;
  if (edge.src_plugin_index != static_cast<std::size_t>(-1)) {
    binding.src_plugin_index = edge.src_plugin_index;
  } else if (const auto src_it = view.node_index_by_id.find(edge.src_node_id);
             src_it != view.node_index_by_id.end()) {
    binding.src_plugin_index = src_it->second;
  }
  if (edge.dst_plugin_index != static_cast<std::size_t>(-1)) {
    binding.dst_plugin_index = edge.dst_plugin_index;
  } else if (const auto dst_it = view.node_index_by_id.find(edge.dst_node_id);
             dst_it != view.node_index_by_id.end()) {
    binding.dst_plugin_index = dst_it->second;
  }
  binding.src_tensor_name = edge.tensor_name;
  binding.dst_tensor_name = edge.tensor_name;
  return binding;
}

int graph_node_sequence_sort_key_local(const pipeline_internal::sima::MpkGraphNode& node) {
  return node.sequence >= 0 ? node.sequence : std::numeric_limits<int>::max();
}

std::vector<std::size_t> sorted_unique_post_predecessors_from_mpk_graph_local(
    const pipeline_internal::sima::MpkGraph& graph, const MpkPostGraphView& view,
    const std::unordered_set<std::size_t>& candidate_nodes, const std::size_t node_index) {
  std::vector<std::size_t> out;
  if (node_index >= view.incoming_edges.size()) {
    return out;
  }
  for (const auto edge_index : view.incoming_edges[node_index]) {
    if (edge_index >= graph.edges.size()) {
      continue;
    }
    const auto src_it = view.node_index_by_id.find(graph.edges[edge_index].src_node_id);
    if (src_it == view.node_index_by_id.end() ||
        candidate_nodes.find(src_it->second) == candidate_nodes.end() ||
        !view.reachable_from_mla[src_it->second] ||
        !is_graph_post_stage_kind(
            route_graph_kind_from_mpk_node_local(graph.nodes[src_it->second]))) {
      continue;
    }
    if (std::find(out.begin(), out.end(), src_it->second) == out.end()) {
      out.push_back(src_it->second);
    }
  }
  std::sort(out.begin(), out.end(), [&](const std::size_t lhs, const std::size_t rhs) {
    const int lhs_rank = graph_node_sequence_sort_key_local(graph.nodes[lhs]);
    const int rhs_rank = graph_node_sequence_sort_key_local(graph.nodes[rhs]);
    if (lhs_rank != rhs_rank) {
      return lhs_rank < rhs_rank;
    }
    return lhs < rhs;
  });
  return out;
}

std::vector<std::size_t> sorted_unique_post_successors_from_mpk_graph_local(
    const pipeline_internal::sima::MpkGraph& graph, const MpkPostGraphView& view,
    const std::unordered_set<std::size_t>& candidate_nodes, const std::size_t node_index) {
  std::vector<std::size_t> out;
  if (node_index >= view.outgoing_edges.size()) {
    return out;
  }
  for (const auto edge_index : view.outgoing_edges[node_index]) {
    if (edge_index >= graph.edges.size()) {
      continue;
    }
    const auto dst_it = view.node_index_by_id.find(graph.edges[edge_index].dst_node_id);
    if (dst_it == view.node_index_by_id.end() ||
        candidate_nodes.find(dst_it->second) == candidate_nodes.end() ||
        !view.reachable_from_mla[dst_it->second] ||
        !is_graph_post_stage_kind(
            route_graph_kind_from_mpk_node_local(graph.nodes[dst_it->second]))) {
      continue;
    }
    if (std::find(out.begin(), out.end(), dst_it->second) == out.end()) {
      out.push_back(dst_it->second);
    }
  }
  std::sort(out.begin(), out.end(), [&](const std::size_t lhs, const std::size_t rhs) {
    const int lhs_rank = graph_node_sequence_sort_key_local(graph.nodes[lhs]);
    const int rhs_rank = graph_node_sequence_sort_key_local(graph.nodes[rhs]);
    if (lhs_rank != rhs_rank) {
      return lhs_rank < rhs_rank;
    }
    return lhs < rhs;
  });
  return out;
}

std::optional<std::vector<std::size_t>> trace_post_lineage_from_mpk_graph_terminal_local(
    const pipeline_internal::sima::MpkGraph& graph, const MpkPostGraphView& view,
    const std::unordered_set<std::size_t>& candidate_nodes, const std::size_t terminal_node_index) {
  std::vector<std::size_t> lineage_rev;
  std::unordered_set<std::size_t> seen;
  std::size_t current = terminal_node_index;
  while (true) {
    if (candidate_nodes.find(current) == candidate_nodes.end() || !seen.insert(current).second) {
      return std::nullopt;
    }
    lineage_rev.push_back(current);
    const auto predecessors =
        sorted_unique_post_predecessors_from_mpk_graph_local(graph, view, candidate_nodes, current);
    if (predecessors.empty()) {
      break;
    }
    if (predecessors.size() != 1U) {
      return std::nullopt;
    }
    current = predecessors.front();
  }
  std::reverse(lineage_rev.begin(), lineage_rev.end());
  return lineage_rev;
}

std::optional<int> terminal_post_logical_index_hint_from_mpk_graph_local(
    const pipeline_internal::sima::MpkGraph& graph, const MpkPostGraphView& view,
    const std::unordered_set<std::size_t>& candidate_nodes, const std::size_t node_index) {
  std::optional<int> logical_index;
  if (node_index >= view.outgoing_edges.size()) {
    return logical_index;
  }
  for (const auto edge_index : view.outgoing_edges[node_index]) {
    if (edge_index >= graph.edges.size()) {
      continue;
    }
    const auto& edge = graph.edges[edge_index];
    const auto dst_it = view.node_index_by_id.find(edge.dst_node_id);
    if (dst_it != view.node_index_by_id.end() &&
        candidate_nodes.find(dst_it->second) != candidate_nodes.end()) {
      continue;
    }
    int candidate_index = -1;
    if (dst_it != view.node_index_by_id.end()) {
      candidate_index = tensor_name_position_local(graph.nodes[dst_it->second].input_tensor_names,
                                                   edge.tensor_name);
    }
    if (candidate_index < 0) {
      candidate_index =
          tensor_name_position_local(graph.nodes[node_index].output_tensor_names, edge.tensor_name);
    }
    if (candidate_index >= 0 && (!logical_index.has_value() || candidate_index < *logical_index)) {
      logical_index = candidate_index;
    }
  }
  return logical_index;
}

std::optional<std::vector<RouteRegion>> build_post_regions_from_mpk_graph_lineages_local(
    const pipeline_internal::sima::MpkGraph& graph, const MpkPostGraphView& view,
    const pipeline_internal::sima::MpkContract& contract,
    const std::unordered_set<std::size_t>& candidate_nodes,
    const std::vector<std::vector<std::size_t>>& lineages,
    const std::vector<std::optional<int>>& lineage_logical_indices) {
  if (lineages.empty()) {
    return std::vector<RouteRegion>{};
  }
  if (lineages.size() != lineage_logical_indices.size()) {
    return std::nullopt;
  }

  const std::size_t max_depth =
      std::max_element(lineages.begin(), lineages.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.size() < rhs.size();
      })->size();
  if (max_depth == 0U) {
    return std::vector<RouteRegion>{};
  }

  std::vector<RouteRegion> regions;
  regions.reserve(max_depth);
  for (std::size_t reverse_offset = max_depth; reverse_offset-- > 0U;) {
    std::vector<std::pair<std::size_t, std::size_t>> members;
    members.reserve(lineages.size());
    pipeline_internal::sima::RouteGraphKernelKind kind =
        pipeline_internal::sima::RouteGraphKernelKind::Unknown;
    bool kind_set = false;

    for (std::size_t lineage_index = 0; lineage_index < lineages.size(); ++lineage_index) {
      const auto& lineage = lineages[lineage_index];
      if (lineage.size() <= reverse_offset) {
        continue;
      }
      const std::size_t stage_pos = lineage.size() - 1U - reverse_offset;
      const auto member = lineage[stage_pos];
      if (member >= graph.nodes.size()) {
        return std::nullopt;
      }
      const auto node_kind = route_graph_kind_from_mpk_node_local(graph.nodes[member]);
      if (!kind_set) {
        kind = node_kind;
        kind_set = true;
      } else if (node_kind != kind) {
        return std::nullopt;
      }
      members.emplace_back(lineage_index, stage_pos);
    }
    if (members.empty() || !kind_set) {
      continue;
    }

    RouteRegion region;
    const bool terminal_stage = reverse_offset == 0U;
    if (members.size() == 1U) {
      region.kind =
          terminal_stage && kind == pipeline_internal::sima::RouteGraphKernelKind::BoxDecode
              ? RouteRegionKind::BoxDecodeTerminal
              : RouteRegionKind::Linear;
    } else {
      if (kind == pipeline_internal::sima::RouteGraphKernelKind::BoxDecode) {
        return std::nullopt;
      }
      region.kind = RouteRegionKind::FanoutMap;
    }
    region.op_kind = kind;
    region.member_plugin_indices.reserve(members.size());

    std::optional<std::size_t> common_join;
    bool common_join_valid = true;
    for (const auto& [lineage_index, stage_pos] : members) {
      const auto& lineage = lineages[lineage_index];
      const auto member = lineage[stage_pos];
      region.member_plugin_indices.push_back(member);

      for (const auto edge_index : view.incoming_edges[member]) {
        if (edge_index >= graph.edges.size()) {
          continue;
        }
        const auto& edge = graph.edges[edge_index];
        const auto src_it = view.node_index_by_id.find(edge.src_node_id);
        if (src_it == view.node_index_by_id.end()) {
          continue;
        }
        if (stage_pos > 0U) {
          if (src_it->second != lineage[stage_pos - 1U]) {
            continue;
          }
        } else if (candidate_nodes.find(src_it->second) != candidate_nodes.end()) {
          continue;
        }
        auto binding = route_tensor_binding_from_mpk_graph_edge_local(edge, view);
        if (lineage_logical_indices[lineage_index].has_value()) {
          binding.logical_index = *lineage_logical_indices[lineage_index];
        }
        region.inputs.push_back(std::move(binding));
        if (region.inputs.size() == 1U) {
          region.producer_plugin_index = region.inputs.back().src_plugin_index;
        }
      }

      std::size_t selected_outputs = 0U;
      for (const auto edge_index : view.outgoing_edges[member]) {
        if (edge_index >= graph.edges.size()) {
          continue;
        }
        const auto& edge = graph.edges[edge_index];
        const auto dst_it = view.node_index_by_id.find(edge.dst_node_id);
        if (dst_it == view.node_index_by_id.end()) {
          continue;
        }
        if (!terminal_stage) {
          if (dst_it->second != lineage[stage_pos + 1U]) {
            continue;
          }
        } else if (candidate_nodes.find(dst_it->second) != candidate_nodes.end()) {
          continue;
        }
        auto binding = route_tensor_binding_from_mpk_graph_edge_local(edge, view);
        if (lineage_logical_indices[lineage_index].has_value()) {
          binding.logical_index = *lineage_logical_indices[lineage_index];
        }
        region.outputs.push_back(std::move(binding));
        ++selected_outputs;
        if (terminal_stage) {
          if (!common_join.has_value()) {
            common_join = dst_it->second;
          } else if (*common_join != dst_it->second) {
            common_join_valid = false;
          }
        }
      }
      if (terminal_stage && selected_outputs != 1U) {
        common_join_valid = false;
      }

      if (terminal_stage) {
        auto outputs = stage_output_contracts_from_graph_node_local(contract, graph.nodes[member]);
        if (members.size() == 1U) {
          region.egress_contracts = std::move(outputs);
        } else if (!outputs.empty()) {
          region.egress_contracts.push_back(outputs.front());
        }
      }
    }

    if (!region.inputs.empty()) {
      region.producer_plugin_index = region.inputs.front().src_plugin_index;
    } else if (!region.member_plugin_indices.empty()) {
      region.producer_plugin_index = region.member_plugin_indices.front();
    }
    sort_route_bindings_by_logical_index(&region.inputs);
    sort_route_bindings_by_logical_index(&region.outputs);

    if (terminal_stage && common_join_valid && common_join.has_value() &&
        *common_join < graph.nodes.size() &&
        route_graph_kind_from_mpk_node_local(graph.nodes[*common_join]) ==
            pipeline_internal::sima::RouteGraphKernelKind::PassThrough) {
      region.join_plugin_index = common_join;
    }
    regions.push_back(std::move(region));
  }
  return regions;
}

std::optional<RouteRegion>
build_linear_region_from_graph_node(const pipeline_internal::sima::RouteGraph& graph,
                                    const pipeline_internal::sima::MpkContract& contract,
                                    const std::size_t plugin_index,
                                    const RouteRegionKind region_kind = RouteRegionKind::Linear) {
  const auto* node = pipeline_internal::sima::route_graph_node(graph, plugin_index);
  if (!node || plugin_index >= contract.plugins.size()) {
    return std::nullopt;
  }
  RouteRegion region;
  region.kind = region_kind;
  region.op_kind = node->kind;
  region.producer_plugin_index = plugin_index;
  region.member_plugin_indices = {plugin_index};
  for (const auto* edge :
       pipeline_internal::sima::route_graph_incoming_edges(graph, plugin_index)) {
    if (edge) {
      region.inputs.push_back(route_tensor_binding_from_edge(*edge));
      region.producer_plugin_index = edge->src_plugin_index;
    }
  }
  for (const auto* edge :
       pipeline_internal::sima::route_graph_outgoing_edges(graph, plugin_index)) {
    if (edge) {
      region.outputs.push_back(route_tensor_binding_from_edge(*edge));
    }
  }
  region.egress_contracts = stage_output_contracts_from_plugin(contract.plugins[plugin_index]);
  return region;
}

std::vector<std::size_t>
sorted_unique_post_predecessors(const pipeline_internal::sima::RouteGraph& graph,
                                const std::unordered_set<std::size_t>& candidate_nodes,
                                const std::size_t plugin_index) {
  std::vector<std::size_t> out;
  for (const auto* edge :
       pipeline_internal::sima::route_graph_incoming_edges(graph, plugin_index)) {
    if (!edge || candidate_nodes.find(edge->src_plugin_index) == candidate_nodes.end()) {
      continue;
    }
    const auto* src = pipeline_internal::sima::route_graph_node(graph, edge->src_plugin_index);
    if (!src || !src->after_mla || !is_graph_post_stage_kind(src->kind)) {
      continue;
    }
    if (std::find(out.begin(), out.end(), edge->src_plugin_index) == out.end()) {
      out.push_back(edge->src_plugin_index);
    }
  }
  std::sort(out.begin(), out.end(), [&](const std::size_t lhs, const std::size_t rhs) {
    const auto lhs_rank = pipeline_internal::sima::route_graph_execution_rank(graph, lhs)
                              .value_or(std::numeric_limits<std::size_t>::max());
    const auto rhs_rank = pipeline_internal::sima::route_graph_execution_rank(graph, rhs)
                              .value_or(std::numeric_limits<std::size_t>::max());
    if (lhs_rank != rhs_rank) {
      return lhs_rank < rhs_rank;
    }
    return lhs < rhs;
  });
  return out;
}

std::vector<std::size_t>
sorted_unique_post_successors(const pipeline_internal::sima::RouteGraph& graph,
                              const std::unordered_set<std::size_t>& candidate_nodes,
                              const std::size_t plugin_index) {
  std::vector<std::size_t> out;
  for (const auto* edge :
       pipeline_internal::sima::route_graph_outgoing_edges(graph, plugin_index)) {
    if (!edge || candidate_nodes.find(edge->dst_plugin_index) == candidate_nodes.end()) {
      continue;
    }
    const auto* dst = pipeline_internal::sima::route_graph_node(graph, edge->dst_plugin_index);
    if (!dst || !dst->after_mla || !is_graph_post_stage_kind(dst->kind)) {
      continue;
    }
    if (std::find(out.begin(), out.end(), edge->dst_plugin_index) == out.end()) {
      out.push_back(edge->dst_plugin_index);
    }
  }
  std::sort(out.begin(), out.end(), [&](const std::size_t lhs, const std::size_t rhs) {
    const auto lhs_rank = pipeline_internal::sima::route_graph_execution_rank(graph, lhs)
                              .value_or(std::numeric_limits<std::size_t>::max());
    const auto rhs_rank = pipeline_internal::sima::route_graph_execution_rank(graph, rhs)
                              .value_or(std::numeric_limits<std::size_t>::max());
    if (lhs_rank != rhs_rank) {
      return lhs_rank < rhs_rank;
    }
    return lhs < rhs;
  });
  return out;
}

std::optional<std::vector<std::size_t>>
trace_post_lineage_from_terminal(const pipeline_internal::sima::RouteGraph& graph,
                                 const std::unordered_set<std::size_t>& candidate_nodes,
                                 const std::size_t terminal_plugin_index) {
  std::vector<std::size_t> lineage_rev;
  std::unordered_set<std::size_t> seen;
  std::size_t current = terminal_plugin_index;
  while (true) {
    if (candidate_nodes.find(current) == candidate_nodes.end()) {
      return std::nullopt;
    }
    if (!seen.insert(current).second) {
      return std::nullopt;
    }
    lineage_rev.push_back(current);
    const auto predecessors = sorted_unique_post_predecessors(graph, candidate_nodes, current);
    if (predecessors.empty()) {
      break;
    }
    if (predecessors.size() != 1U) {
      return std::nullopt;
    }
    current = predecessors.front();
  }
  std::reverse(lineage_rev.begin(), lineage_rev.end());
  return lineage_rev;
}

std::optional<int>
terminal_post_logical_index_hint(const pipeline_internal::sima::RouteGraph& graph,
                                 const std::unordered_set<std::size_t>& candidate_nodes,
                                 const std::size_t plugin_index) {
  std::optional<int> logical_index;
  for (const auto* edge :
       pipeline_internal::sima::route_graph_outgoing_edges(graph, plugin_index)) {
    if (!edge || candidate_nodes.find(edge->dst_plugin_index) != candidate_nodes.end()) {
      continue;
    }
    const int candidate_index =
        edge->dst_input_index >= 0 ? edge->dst_input_index : edge->src_output_index;
    if (candidate_index < 0) {
      continue;
    }
    if (!logical_index.has_value() || candidate_index < *logical_index) {
      logical_index = candidate_index;
    }
  }
  return logical_index;
}

std::optional<std::vector<RouteRegion>>
build_post_regions_from_lineages(const pipeline_internal::sima::RouteGraph& graph,
                                 const pipeline_internal::sima::MpkContract& contract,
                                 const std::unordered_set<std::size_t>& candidate_nodes,
                                 const std::vector<std::vector<std::size_t>>& lineages,
                                 const std::vector<std::optional<int>>& lineage_logical_indices) {
  if (lineages.empty()) {
    return std::vector<RouteRegion>{};
  }
  if (lineages.size() != lineage_logical_indices.size()) {
    return std::nullopt;
  }
  const std::size_t max_depth =
      std::max_element(lineages.begin(), lineages.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.size() < rhs.size();
      })->size();
  if (max_depth == 0U) {
    return std::vector<RouteRegion>{};
  }

  std::vector<RouteRegion> regions;
  regions.reserve(max_depth);
  for (std::size_t reverse_offset = max_depth; reverse_offset-- > 0U;) {
    std::vector<std::pair<std::size_t, std::size_t>> members;
    members.reserve(lineages.size());
    const pipeline_internal::sima::RouteGraphNode* first_node = nullptr;
    pipeline_internal::sima::RouteGraphKernelKind kind =
        pipeline_internal::sima::RouteGraphKernelKind::Unknown;
    for (std::size_t lineage_index = 0; lineage_index < lineages.size(); ++lineage_index) {
      const auto& lineage = lineages[lineage_index];
      if (lineage.size() <= reverse_offset) {
        continue;
      }
      const std::size_t stage_pos = lineage.size() - 1U - reverse_offset;
      const auto* node = pipeline_internal::sima::route_graph_node(graph, lineage[stage_pos]);
      if (!node) {
        return std::nullopt;
      }
      if (!first_node) {
        first_node = node;
        kind = node->kind;
      } else if (node->kind != kind) {
        return std::nullopt;
      }
      members.emplace_back(lineage_index, stage_pos);
    }
    if (members.empty() || !first_node) {
      continue;
    }

    RouteRegion region;
    const bool terminal_stage = reverse_offset == 0U;
    if (members.size() == 1U) {
      region.kind =
          terminal_stage && kind == pipeline_internal::sima::RouteGraphKernelKind::BoxDecode
              ? RouteRegionKind::BoxDecodeTerminal
              : RouteRegionKind::Linear;
    } else {
      if (kind == pipeline_internal::sima::RouteGraphKernelKind::BoxDecode) {
        return std::nullopt;
      }
      region.kind = RouteRegionKind::FanoutMap;
    }
    region.op_kind = kind;
    region.member_plugin_indices.reserve(members.size());

    std::optional<std::size_t> common_join;
    bool common_join_valid = true;
    for (const auto& [lineage_index, stage_pos] : members) {
      const auto& lineage = lineages[lineage_index];
      const auto member = lineage[stage_pos];
      region.member_plugin_indices.push_back(member);

      for (const auto* edge : pipeline_internal::sima::route_graph_incoming_edges(graph, member)) {
        if (!edge) {
          continue;
        }
        if (stage_pos > 0U) {
          if (edge->src_plugin_index != lineages[lineage_index][stage_pos - 1U]) {
            continue;
          }
        } else {
          if (candidate_nodes.find(edge->src_plugin_index) != candidate_nodes.end()) {
            continue;
          }
        }
        auto binding = route_tensor_binding_from_edge(*edge);
        if (lineage_logical_indices[lineage_index].has_value()) {
          binding.logical_index = *lineage_logical_indices[lineage_index];
        }
        region.inputs.push_back(std::move(binding));
        if (region.inputs.size() == 1U) {
          region.producer_plugin_index = edge->src_plugin_index;
        }
      }

      std::size_t selected_outputs = 0U;
      for (const auto* edge : pipeline_internal::sima::route_graph_outgoing_edges(graph, member)) {
        if (!edge) {
          continue;
        }
        if (!terminal_stage) {
          if (edge->dst_plugin_index != lineages[lineage_index][stage_pos + 1U]) {
            continue;
          }
        } else if (candidate_nodes.find(edge->dst_plugin_index) != candidate_nodes.end()) {
          continue;
        }
        auto binding = route_tensor_binding_from_edge(*edge);
        if (lineage_logical_indices[lineage_index].has_value()) {
          binding.logical_index = *lineage_logical_indices[lineage_index];
        }
        region.outputs.push_back(std::move(binding));
        ++selected_outputs;
        if (terminal_stage) {
          if (!common_join.has_value()) {
            common_join = edge->dst_plugin_index;
          } else if (*common_join != edge->dst_plugin_index) {
            common_join_valid = false;
          }
        }
      }
      if (terminal_stage && selected_outputs != 1U) {
        common_join_valid = false;
      }

      if (terminal_stage && member < contract.plugins.size()) {
        auto outputs = stage_output_contracts_from_plugin(contract.plugins[member]);
        if (members.size() == 1U) {
          region.egress_contracts = std::move(outputs);
        } else if (!outputs.empty()) {
          region.egress_contracts.push_back(outputs.front());
        }
      }
    }

    if (!region.inputs.empty()) {
      region.producer_plugin_index = region.inputs.front().src_plugin_index;
    } else if (!region.member_plugin_indices.empty()) {
      region.producer_plugin_index = region.member_plugin_indices.front();
    }
    sort_route_bindings_by_logical_index(&region.inputs);
    sort_route_bindings_by_logical_index(&region.outputs);

    if (terminal_stage && common_join_valid && common_join.has_value()) {
      const auto* join_node = pipeline_internal::sima::route_graph_node(graph, *common_join);
      if (join_node &&
          join_node->kind == pipeline_internal::sima::RouteGraphKernelKind::PassThrough) {
        region.join_plugin_index = common_join;
      }
    }
    regions.push_back(std::move(region));
  }
  return regions;
}

// Structural mirror of derive_post_regions_from_graph for the pre side.
// Today the post side derives RouteRegion entries straight from the MPK graph
// (FanoutMap + Linear + FaninJoin + BoxDecodeTerminal) so that the materializer
// can walk a single canonical structure and emit the right element shape per
// region. The pre side's `ingress_regions` already carries the correct
// fan-out / fan-in topology per branch_op depth (one FanoutMap per depth, plus
// an optional FaninJoin when all ingresses converge into a single packer).
//
// `pre_regions` is what build_preprocess_group_impl walks to materialize the
// pre stage. Building it from `ingress_regions` guarantees that:
//   - For monolithic models with a single ingress contract, depth-0 emits a
//     Linear region (one node per op level), exactly what the legacy flat
//     pre_chain produced.
//   - For native multi-IFM / packer-style models with N ingress contracts,
//     each depth becomes a FanoutMap region of size N, which the materializer
//     collapses to ONE multi-input pre node whose compiled contract has
//     num_in_tensor=N (instead of N separate single-IO elements).
//   - Adjacent same-multiplicity Quant+Tess and Cast+Tess regions are fused
//     into QuantTess / CastTess respectively, mirroring the per-op fusion
//     that the flat pre_chain already does.
//   - The FaninJoin region is dropped for materialization: the IFM pack
//     boundary is absorbed by the multi-input pre node's compiled contract
//     (logical_inputs.size() == N) and produces a single packed output, the
//     symmetric counterpart of how the post side's multi-IO detessdequant
//     element produces a single buffer carrying N logical outputs.
std::vector<RouteRegion>
derive_pre_regions_from_ingress_regions(const std::vector<RouteRegion>& ingress_regions,
                                        bool fuse_quant_tess, bool fuse_cast_tess) {
  using GraphKind = pipeline_internal::sima::RouteGraphKernelKind;
  std::vector<RouteRegion> regions;
  regions.reserve(ingress_regions.size());
  for (std::size_t i = 0; i < ingress_regions.size(); ++i) {
    const auto& current = ingress_regions[i];
    if (current.kind == RouteRegionKind::FaninJoin) {
      // The packer / pass-through join is absorbed by the upstream multi-input
      // pre node — do not emit a separate materialization region for it.
      continue;
    }
    if (current.kind == RouteRegionKind::FanoutMap && current.op_kind == GraphKind::Quant &&
        i + 1U < ingress_regions.size() &&
        ingress_regions[i + 1U].kind == RouteRegionKind::FaninJoin) {
      // Same-op ingress branches that immediately join at the MLA boundary are
      // best represented as one multi-input quantize stage. The graph-222 ABI
      // already has tensors[] descriptors; keeping this FanoutMap intact lets
      // processcvu build one JobEVXX with tensor_count=N instead of N branch
      // sessions and N EV74 graph launches.
      regions.push_back(current);
      continue;
    }
    if (i + 1U < ingress_regions.size()) {
      const auto& next = ingress_regions[i + 1U];
      const bool same_shape = current.kind == next.kind && current.member_plugin_indices.size() ==
                                                               next.member_plugin_indices.size();
      if (same_shape) {
        if (fuse_quant_tess && current.op_kind == GraphKind::Quant &&
            next.op_kind == GraphKind::Tess) {
          RouteRegion fused = current;
          fused.op_kind = GraphKind::QuantTess;
          regions.push_back(std::move(fused));
          ++i;
          continue;
        }
        if (fuse_cast_tess && current.op_kind == GraphKind::Cast &&
            next.op_kind == GraphKind::Tess) {
          RouteRegion fused = current;
          fused.op_kind = GraphKind::CastTess;
          regions.push_back(std::move(fused));
          ++i;
          continue;
        }
      }
    }
    regions.push_back(current);
  }
  return regions;
}

std::vector<RouteRegion> pre_regions_from_pre_chain(const std::vector<SessionPreStageOp>& chain) {
  std::vector<RouteRegion> regions;
  regions.reserve(chain.size());
  for (const auto op : chain) {
    RouteRegion region;
    region.kind = RouteRegionKind::Linear;
    switch (op) {
    case SessionPreStageOp::Preproc:
      region.op_kind = pipeline_internal::sima::RouteGraphKernelKind::Preproc;
      break;
    case SessionPreStageOp::Quant:
      region.op_kind = pipeline_internal::sima::RouteGraphKernelKind::Quant;
      break;
    case SessionPreStageOp::Tess:
      region.op_kind = pipeline_internal::sima::RouteGraphKernelKind::Tess;
      break;
    case SessionPreStageOp::QuantTess:
      region.op_kind = pipeline_internal::sima::RouteGraphKernelKind::QuantTess;
      break;
    case SessionPreStageOp::Cast:
      region.op_kind = pipeline_internal::sima::RouteGraphKernelKind::Cast;
      break;
    case SessionPreStageOp::CastTess:
      region.op_kind = pipeline_internal::sima::RouteGraphKernelKind::CastTess;
      break;
    }
    regions.push_back(std::move(region));
  }
  return regions;
}

std::vector<RouteRegion>
post_regions_from_post_chain(const std::vector<SessionPostStageOp>& chain,
                             const std::vector<EgressTensorContract>& egress) {
  std::vector<RouteRegion> regions;
  regions.reserve(chain.size());
  for (std::size_t i = 0; i < chain.size(); ++i) {
    RouteRegion region;
    region.kind = (chain[i] == SessionPostStageOp::BoxDecode) ? RouteRegionKind::BoxDecodeTerminal
                                                              : RouteRegionKind::Linear;
    switch (chain[i]) {
    case SessionPostStageOp::Detess:
      region.op_kind = pipeline_internal::sima::RouteGraphKernelKind::Detess;
      break;
    case SessionPostStageOp::DetessDequant:
      region.op_kind = pipeline_internal::sima::RouteGraphKernelKind::DetessDequant;
      break;
    case SessionPostStageOp::DetessCast:
      region.op_kind = pipeline_internal::sima::RouteGraphKernelKind::DetessCast;
      break;
    case SessionPostStageOp::Dequantize:
      region.op_kind = pipeline_internal::sima::RouteGraphKernelKind::Dequantize;
      break;
    case SessionPostStageOp::BoxDecode:
      region.op_kind = pipeline_internal::sima::RouteGraphKernelKind::BoxDecode;
      break;
    case SessionPostStageOp::Cast:
      region.op_kind = pipeline_internal::sima::RouteGraphKernelKind::Cast;
      break;
    }
    if (i + 1U == chain.size()) {
      region.egress_contracts = egress;
    }
    regions.push_back(std::move(region));
  }
  return regions;
}

std::vector<SessionPostStageOp> post_chain_from_regions(const std::vector<RouteRegion>& regions) {
  std::vector<SessionPostStageOp> chain;
  chain.reserve(regions.size());
  for (const auto& region : regions) {
    if (const auto mapped = session_post_op_from_graph_kind(region.op_kind); mapped.has_value()) {
      chain.push_back(*mapped);
    }
  }
  return chain;
}

std::string graph_chain_csv_from_regions(const std::vector<RouteRegion>& regions) {
  if (regions.empty()) {
    return "none";
  }
  std::ostringstream oss;
  for (std::size_t i = 0; i < regions.size(); ++i) {
    if (i > 0U) {
      oss << ",";
    }
    oss << route_graph_op_name(regions[i].op_kind);
  }
  return oss.str();
}

bool is_materialized_post_graph_kind(const pipeline_internal::sima::RouteGraphKernelKind kind) {
  using Kind = pipeline_internal::sima::RouteGraphKernelKind;
  switch (kind) {
  case Kind::Detess:
  case Kind::DetessCast:
  case Kind::DetessDequant:
  case Kind::Dequantize:
  case Kind::BoxDecode:
  case Kind::Cast:
    return true;
  case Kind::Unknown:
  case Kind::Preproc:
  case Kind::Quant:
  case Kind::Tess:
  case Kind::QuantTess:
  case Kind::CastTess:
  case Kind::Unpack:
  case Kind::Slice:
  case Kind::PassThrough:
  case Kind::Mla:
    return false;
  }
  return false;
}

std::vector<RouteRegion>
filter_non_materialized_post_regions(const std::vector<RouteRegion>& regions) {
  std::vector<RouteRegion> materialized;
  materialized.reserve(regions.size());
  for (const auto& region : regions) {
    if (is_materialized_post_graph_kind(region.op_kind)) {
      materialized.push_back(region);
    }
  }
  std::vector<RouteRegion> filtered;
  filtered.reserve(materialized.size());
  for (std::size_t i = 0; i < materialized.size(); ++i) {
    if (i + 1U < materialized.size()) {
      const auto& lhs = materialized[i];
      const auto& rhs = materialized[i + 1U];
      if (can_fuse_detess_and_dequant_regions(lhs, rhs)) {
        filtered.push_back(fuse_adjacent_post_regions(
            lhs, rhs, pipeline_internal::sima::RouteGraphKernelKind::DetessDequant));
        ++i;
        continue;
      }
      if (can_fuse_detess_and_cast_regions(lhs, rhs)) {
        filtered.push_back(fuse_adjacent_post_regions(
            lhs, rhs, pipeline_internal::sima::RouteGraphKernelKind::DetessCast));
        ++i;
        continue;
      }
    }
    filtered.push_back(materialized[i]);
  }
  return filtered;
}

std::vector<EgressTensorContract>
terminal_egress_contracts_from_regions(const std::vector<RouteRegion>& regions) {
  for (auto it = regions.rbegin(); it != regions.rend(); ++it) {
    if (!it->egress_contracts.empty()) {
      return it->egress_contracts;
    }
  }
  return {};
}

void finalize_post_summary_from_regions(RouteMaterializationPlan* out) {
  if (!out) {
    return;
  }

  out->post_chain = post_chain_from_regions(out->post_regions);
  out->include_post_stage = !out->post_regions.empty();
  out->post_cast_bf16_to_fp32 = std::any_of(
      out->post_regions.begin(), out->post_regions.end(), [](const RouteRegion& region) {
        return region.op_kind == pipeline_internal::sima::RouteGraphKernelKind::Cast;
      });

  const auto graph_egress_contracts = terminal_egress_contracts_from_regions(out->post_regions);
  if (!graph_egress_contracts.empty()) {
    out->egress_contracts = graph_egress_contracts;
    out->egress_contract = out->egress_contracts.front();
  }

  if (out->post_regions.empty()) {
    out->selected_post_kind = PostRouteStageKind::None;
    out->post_adapter = SessionPostAdapterKind::None;
    return;
  }

  const PostRouteStageKind graph_selected_post_kind =
      post_route_kind_from_graph_kind(out->post_regions.back().op_kind);
  out->selected_post_kind = graph_selected_post_kind != PostRouteStageKind::Unknown
                                ? graph_selected_post_kind
                                : PostRouteStageKind::None;
  out->post_adapter = post_adapter_from_post_route_kind(out->selected_post_kind);
}

std::string route_region_csv(const std::vector<RouteRegion>& regions) {
  if (regions.empty()) {
    return "none";
  }
  std::ostringstream oss;
  for (std::size_t i = 0; i < regions.size(); ++i) {
    if (i > 0U) {
      oss << ",";
    }
    const auto& region = regions[i];
    switch (region.kind) {
    case RouteRegionKind::Linear:
      oss << route_graph_op_name(region.op_kind);
      break;
    case RouteRegionKind::FanoutMap:
      oss << "fanout(" << route_graph_op_name(region.op_kind) << "x"
          << region.member_plugin_indices.size() << ")";
      break;
    case RouteRegionKind::FaninJoin:
      oss << "fanin(" << route_graph_op_name(region.op_kind) << ")";
      break;
    case RouteRegionKind::BoxDecodeTerminal:
      oss << "boxdecode";
      break;
    }
  }
  return oss.str();
}

std::vector<RouteRegion> derive_post_regions_from_graph(const ModelPack& pack) {
  if (!pack.mpk_contract().has_value()) {
    return {};
  }
  const auto& contract = *pack.mpk_contract();
  const auto& graph = contract.graph;
  if (graph.nodes.empty()) {
    return {};
  }
  const auto view = build_mpk_post_graph_view_local(graph);
  if (view.mla_node_index == std::numeric_limits<std::size_t>::max()) {
    return {};
  }
  std::vector<std::size_t> candidate_nodes;
  candidate_nodes.reserve(graph.nodes.size());
  for (std::size_t node_index = 0; node_index < graph.nodes.size(); ++node_index) {
    if (node_index == view.mla_node_index || !view.reachable_from_mla[node_index]) {
      continue;
    }
    const auto kind = route_graph_kind_from_mpk_node_local(graph.nodes[node_index]);
    if (!is_graph_post_stage_kind(kind)) {
      continue;
    }
    candidate_nodes.push_back(node_index);
  }
  if (candidate_nodes.empty()) {
    return {};
  }
  std::unordered_set<std::size_t> candidate_set(candidate_nodes.begin(), candidate_nodes.end());

  std::vector<std::size_t> terminal_nodes;
  terminal_nodes.reserve(candidate_nodes.size());
  for (const auto node_index : candidate_nodes) {
    if (sorted_unique_post_successors_from_mpk_graph_local(graph, view, candidate_set, node_index)
            .empty()) {
      terminal_nodes.push_back(node_index);
    }
  }
  std::sort(
      terminal_nodes.begin(), terminal_nodes.end(),
      [&](const std::size_t lhs, const std::size_t rhs) {
        const auto lhs_logical =
            terminal_post_logical_index_hint_from_mpk_graph_local(graph, view, candidate_set, lhs);
        const auto rhs_logical =
            terminal_post_logical_index_hint_from_mpk_graph_local(graph, view, candidate_set, rhs);
        if (lhs_logical.has_value() != rhs_logical.has_value()) {
          return lhs_logical.has_value();
        }
        if (lhs_logical.has_value() && rhs_logical.has_value() && *lhs_logical != *rhs_logical) {
          return *lhs_logical < *rhs_logical;
        }
        const auto lhs_rank = graph_node_sequence_sort_key_local(graph.nodes[lhs]);
        const auto rhs_rank = graph_node_sequence_sort_key_local(graph.nodes[rhs]);
        if (lhs_rank != rhs_rank) {
          return lhs_rank < rhs_rank;
        }
        return lhs < rhs;
      });
  if (terminal_nodes.empty()) {
    return {};
  }

  std::vector<std::vector<std::size_t>> lineages;
  lineages.reserve(terminal_nodes.size());
  std::vector<std::optional<int>> lineage_logical_indices;
  lineage_logical_indices.reserve(terminal_nodes.size());
  std::unordered_set<std::size_t> visited_nodes;
  for (const auto terminal : terminal_nodes) {
    const auto lineage =
        trace_post_lineage_from_mpk_graph_terminal_local(graph, view, candidate_set, terminal);
    if (!lineage.has_value() || lineage->empty()) {
      return {};
    }
    for (const auto member : *lineage) {
      visited_nodes.insert(member);
    }
    lineages.push_back(*lineage);
    lineage_logical_indices.push_back(terminal_post_logical_index_hint_from_mpk_graph_local(
        graph, view, candidate_set, terminal));
  }
  if (visited_nodes.size() != candidate_set.size()) {
    return {};
  }

  if (const auto regions = build_post_regions_from_mpk_graph_lineages_local(
          graph, view, contract, candidate_set, lineages, lineage_logical_indices);
      regions.has_value()) {
    return *regions;
  }
  return {};
}

bool extract_route_capability_from_mpk_graph(const ModelPack& pack, RouteCapability* out) {
  if (!out || !pack.mpk_contract().has_value()) {
    return false;
  }
  const auto& graph = pack.route_graph();
  const auto& contract = *pack.mpk_contract();
  const auto* mla_stage = pipeline_internal::sima::get_mla_stage_io_contract(contract);
  if (!mla_stage) {
    return false;
  }
  const auto mla_idx = pipeline_internal::sima::find_plugin_index_by_name_or_id(
      contract, !mla_stage->name.empty() ? mla_stage->name : mla_stage->plugin_id);
  if (!mla_idx.has_value()) {
    return false;
  }
  const auto ordered = pipeline_internal::sima::plugins_in_execution_order(contract);
  std::unordered_map<std::size_t, std::size_t> rank_by_index;
  rank_by_index.reserve(ordered.size());
  for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
    rank_by_index[ordered[rank]] = rank;
  }
  const auto mla_rank_it = rank_by_index.find(*mla_idx);
  if (mla_rank_it == rank_by_index.end()) {
    return false;
  }
  const std::size_t mla_rank = mla_rank_it->second;
  const auto pre_kind_dbg = [](PreRouteStageKind kind) -> const char* {
    switch (kind) {
    case PreRouteStageKind::None:
      return "none";
    case PreRouteStageKind::Preproc:
      return "preproc";
    case PreRouteStageKind::Quant:
      return "quant";
    case PreRouteStageKind::Tess:
      return "tess";
    case PreRouteStageKind::QuantTess:
      return "quanttess";
    case PreRouteStageKind::Cast:
      return "cast";
    case PreRouteStageKind::Unknown:
      return "unknown";
    }
    return "unknown";
  };
  const auto post_kind_dbg = [](PostRouteStageKind kind) -> const char* {
    switch (kind) {
    case PostRouteStageKind::None:
      return "none";
    case PostRouteStageKind::Detess:
      return "detess";
    case PostRouteStageKind::DetessDequant:
      return "detessdequant";
    case PostRouteStageKind::Dequantize:
      return "dequantize";
    case PostRouteStageKind::BoxDecode:
      return "boxdecode";
    case PostRouteStageKind::Cast:
      return "cast";
    case PostRouteStageKind::Unknown:
      return "unknown";
    }
    return "unknown";
  };
  if (route_debug_enabled()) {
    std::fprintf(stderr,
                 "[route-debug] source=mpk_graph plugins=%zu edges=%zu mla_idx=%zu mla_rank=%zu\n",
                 contract.plugins.size(), contract.edges.size(), *mla_idx, mla_rank);
  }

  out->pre_kind = PreRouteStageKind::None;
  out->post_kind = PostRouteStageKind::None;
  out->has_external_pre = false;
  out->has_external_post = false;
  out->has_external_tess = false;
  out->has_external_pre_cast = false;
  out->has_external_detess = false;
  out->has_external_dequant = false;
  out->has_external_post_cast = false;
  out->has_external_boxdecode = false;
  out->has_strict_boxdecode_route = false;
  out->ordered_pre_ops.clear();
  out->ordered_post_ops.clear();
  out->ingress_contracts = derive_ingress_contracts_from_mpk_contract(contract, graph, ordered);
  out->ingress_regions = ingress_regions_from_contracts(out->ingress_contracts);
  out->egress_contract = EgressTensorContract{};
  out->egress_contracts.clear();

  bool pre_has_quant = false;
  bool pre_has_tess = false;
  bool pre_has_preproc = false;
  bool pre_has_cast = false;
  bool pre_has_unknown = false;
  std::vector<EgressTensorContract> aggregated_post_outputs_preferred;
  std::vector<EgressTensorContract> aggregated_post_outputs_fallback;
  auto bump_post_kind = [&](PostRouteStageKind kind) {
    if (kind == PostRouteStageKind::BoxDecode) {
      out->post_kind = kind;
      return;
    }
    if (out->post_kind == PostRouteStageKind::None ||
        out->post_kind == PostRouteStageKind::Unknown) {
      out->post_kind = kind;
    }
  };

  for (const std::size_t plugin_idx : ordered) {
    if (plugin_idx >= contract.plugins.size() || plugin_idx == *mla_idx) {
      continue;
    }
    const auto rank_it = rank_by_index.find(plugin_idx);
    if (rank_it == rank_by_index.end()) {
      continue;
    }
    const bool before_mla = rank_it->second < mla_rank;
    std::string kernel_source = contract.plugins[plugin_idx].kernel;
    if (kernel_source.empty()) {
      kernel_source = contract.plugins[plugin_idx].name;
    }
    const std::string kernel = canonical_mpk_kernel_kind(kernel_source);
    if (route_debug_enabled()) {
      std::fprintf(stderr,
                   "[route-debug] mpk_plugin idx=%zu rank=%zu before_mla=%d raw_kernel=%s "
                   "raw_name=%s canonical=%s\n",
                   plugin_idx, rank_it->second, before_mla ? 1 : 0,
                   contract.plugins[plugin_idx].kernel.c_str(),
                   contract.plugins[plugin_idx].name.c_str(), kernel.c_str());
    }
    if (kernel.empty()) {
      continue;
    }
    if (before_mla && kernel == "pass_through") {
      continue;
    }

    const std::string input_dtype =
        primary_tensor_dtype(contract.plugins[plugin_idx].input_tensors);
    const std::string output_dtype =
        primary_tensor_dtype(contract.plugins[plugin_idx].output_tensors);
    const int input_rank = primary_tensor_rank(contract.plugins[plugin_idx].input_tensors);
    const int output_rank = primary_tensor_rank(contract.plugins[plugin_idx].output_tensors);
    const bool dtype_transition =
        !input_dtype.empty() && !output_dtype.empty() && input_dtype != output_dtype;
    const bool hint_pre_quant = before_mla && dtype_transition &&
                                !dtype_is_quantized_like(input_dtype) &&
                                dtype_is_quantized_like(output_dtype);
    const bool suppress_post_dequant_hint = (kernel == "unpacktransform");
    const bool hint_post_dequant = !before_mla && !suppress_post_dequant_hint && dtype_transition &&
                                   dtype_is_quantized_like(input_dtype) &&
                                   dtype_is_float_like(output_dtype);
    const bool hint_cast = dtype_transition && !hint_pre_quant && !hint_post_dequant;
    const bool hint_pre_tess =
        before_mla && input_rank >= 3 && output_rank > 0 && output_rank < input_rank;
    const bool hint_post_detess =
        !before_mla && input_rank > 0 && input_rank <= 2 && output_rank >= 3;
    if (route_debug_enabled()) {
      std::fprintf(
          stderr,
          "[route-debug] mpk_plugin_hints idx=%zu in_dtype=%s out_dtype=%s in_rank=%d out_rank=%d "
          "pre_quant=%d pre_tess=%d post_dequant=%d post_detess=%d cast=%d\n",
          plugin_idx, input_dtype.c_str(), output_dtype.c_str(), input_rank, output_rank,
          hint_pre_quant ? 1 : 0, hint_pre_tess ? 1 : 0, hint_post_dequant ? 1 : 0,
          hint_post_detess ? 1 : 0, hint_cast ? 1 : 0);
    }

    if (before_mla) {
      const bool implicit_mla_handoff_kernel =
          kernel == "packtransform" || kernel == "bufferconcat" || kernel == "pass_through";
      if (implicit_mla_handoff_kernel) {
        if (route_debug_enabled()) {
          std::fprintf(
              stderr,
              "[route-debug] mpk_plugin idx=%zu canonical=%s treated_as_implicit_mla_handoff=1\n",
              plugin_idx, kernel.c_str());
        }
        continue;
      }
      out->has_external_pre = true;
      OrderedRouteOp::Kind ordered_kind = OrderedRouteOp::Kind::Unknown;
      const bool explicit_pre_kernel = kernel == "quanttess" || kernel == "quantize" ||
                                       kernel == "tessellate" || kernel == "casttess" ||
                                       kernel == "preproc" || kernel == "cast";
      bool consumed_by_hint = false;
      if (!explicit_pre_kernel) {
        if (hint_pre_quant) {
          pre_has_quant = true;
          consumed_by_hint = true;
        }
        if (hint_pre_tess) {
          pre_has_tess = true;
          out->has_external_tess = true;
          consumed_by_hint = true;
        }
        if (hint_cast) {
          pre_has_cast = true;
          out->has_external_pre_cast = true;
          consumed_by_hint = true;
        }
        if (hint_pre_quant && hint_pre_tess) {
          ordered_kind = OrderedRouteOp::Kind::QuantTess;
        } else if (hint_cast && hint_pre_tess) {
          ordered_kind = OrderedRouteOp::Kind::CastTess;
        } else if (hint_pre_quant) {
          ordered_kind = OrderedRouteOp::Kind::Quant;
        } else if (hint_pre_tess) {
          ordered_kind = OrderedRouteOp::Kind::Tess;
        } else if (hint_cast) {
          ordered_kind = OrderedRouteOp::Kind::Cast;
        }
      }
      if (kernel == "casttess") {
        pre_has_cast = true;
        pre_has_tess = true;
        out->has_external_pre_cast = true;
        out->has_external_tess = true;
        ordered_kind = OrderedRouteOp::Kind::CastTess;
      } else if (kernel == "quanttess") {
        pre_has_quant = true;
        pre_has_tess = true;
        out->has_external_tess = true;
        ordered_kind = OrderedRouteOp::Kind::QuantTess;
      } else if (kernel == "quantize") {
        pre_has_quant = true;
        ordered_kind = OrderedRouteOp::Kind::Quant;
      } else if (kernel == "tessellate") {
        pre_has_tess = true;
        out->has_external_tess = true;
        ordered_kind = OrderedRouteOp::Kind::Tess;
      } else if (kernel == "preproc") {
        pre_has_preproc = true;
        ordered_kind = OrderedRouteOp::Kind::Preproc;
      } else if (kernel == "cast") {
        pre_has_cast = true;
        out->has_external_pre_cast = true;
        ordered_kind = OrderedRouteOp::Kind::Cast;
      } else if (!consumed_by_hint) {
        pre_has_unknown = true;
      }
      append_ordered_route_op(&out->ordered_pre_ops, ordered_kind, contract.plugins[plugin_idx],
                              /*before_mla=*/true, /*after_mla=*/false);
      if (out->ingress_contracts.empty() && ordered_kind != OrderedRouteOp::Kind::Unknown) {
        const std::string source_stage = !contract.plugins[plugin_idx].name.empty()
                                             ? contract.plugins[plugin_idx].name
                                             : contract.plugins[plugin_idx].plugin_id;
        const std::string media_type = ordered_kind == OrderedRouteOp::Kind::Preproc
                                           ? std::string("video/x-raw")
                                           : std::string("application/vnd.simaai.tensor");
        IngressTensorContract ingress_contract;
        populate_tensor_contract_from_stage_tensors(contract.plugins[plugin_idx].input_tensors,
                                                    media_type, source_stage, &ingress_contract);
        if (ingress_contract.valid) {
          ingress_contract.ingress_index = 0;
          ingress_contract.source_tensor_name = ingress_contract.source_tensor_name.empty()
                                                    ? "ifm0"
                                                    : ingress_contract.source_tensor_name;
          out->ingress_contracts.push_back(std::move(ingress_contract));
          out->ingress_regions = ingress_regions_from_contracts(out->ingress_contracts);
        }
      }
    } else {
      out->has_external_post = true;
      OrderedRouteOp::Kind ordered_kind = OrderedRouteOp::Kind::Unknown;
      const bool explicit_post_kernel = kernel == "detessdequant" || kernel == "detesscast" ||
                                        kernel == "detessellate" || kernel == "dequantize" ||
                                        kernel == "cast" || kernel == "boxdecode" ||
                                        kernel == "unpacktransform";
      bool consumed_by_hint = false;
      const bool passthrough_kernel = (kernel == "pass_through");
      if (passthrough_kernel) {
        consumed_by_hint = true;
      }
      if (kernel == "unpacktransform") {
        consumed_by_hint = true;
      }
      if (!explicit_post_kernel) {
        if (hint_post_detess) {
          bump_post_kind(PostRouteStageKind::Detess);
          out->has_external_detess = true;
          consumed_by_hint = true;
        }
        if (hint_post_dequant) {
          bump_post_kind(hint_post_detess ? PostRouteStageKind::DetessDequant
                                          : PostRouteStageKind::Dequantize);
          out->has_external_dequant = true;
          consumed_by_hint = true;
        }
        if (hint_cast) {
          bump_post_kind(PostRouteStageKind::Cast);
          out->has_external_post_cast = true;
          consumed_by_hint = true;
        }
        if (hint_post_detess || hint_post_dequant) {
          ordered_kind = hint_post_detess ? (hint_post_dequant ? OrderedRouteOp::Kind::DetessDequant
                                                               : OrderedRouteOp::Kind::Detess)
                                          : OrderedRouteOp::Kind::Dequantize;
          if (hint_post_detess && hint_cast && !hint_post_dequant) {
            ordered_kind = OrderedRouteOp::Kind::DetessCast;
          }
        } else if (hint_cast) {
          ordered_kind = OrderedRouteOp::Kind::Cast;
        }
      }

      if (kernel == "detesscast") {
        bump_post_kind(PostRouteStageKind::DetessCast);
        out->has_external_detess = true;
        out->has_external_post_cast = true;
        ordered_kind = OrderedRouteOp::Kind::DetessCast;
      } else if (kernel == "detessdequant") {
        bump_post_kind(PostRouteStageKind::DetessDequant);
        out->has_external_detess = true;
        out->has_external_dequant = true;
        ordered_kind = OrderedRouteOp::Kind::DetessDequant;
      } else if (kernel == "detessellate") {
        bump_post_kind(PostRouteStageKind::Detess);
        out->has_external_detess = true;
        ordered_kind = OrderedRouteOp::Kind::Detess;
      } else if (kernel == "dequantize") {
        bump_post_kind(PostRouteStageKind::Dequantize);
        out->has_external_dequant = true;
        ordered_kind = OrderedRouteOp::Kind::Dequantize;
      } else if (kernel == "cast") {
        bump_post_kind(PostRouteStageKind::Cast);
        out->has_external_post_cast = true;
        ordered_kind = OrderedRouteOp::Kind::Cast;
      } else if (kernel == "boxdecode") {
        bump_post_kind(PostRouteStageKind::BoxDecode);
        out->has_external_boxdecode = true;
        ordered_kind = OrderedRouteOp::Kind::BoxDecode;
      } else if (kernel == "unpacktransform") {
        ordered_kind = OrderedRouteOp::Kind::Unpack;
      } else if (!consumed_by_hint) {
        if (out->post_kind == PostRouteStageKind::None) {
          out->post_kind = PostRouteStageKind::Unknown;
        }
      }
      append_ordered_route_op(&out->ordered_post_ops, ordered_kind, contract.plugins[plugin_idx],
                              /*before_mla=*/false, /*after_mla=*/true);
      if (ordered_kind != OrderedRouteOp::Kind::Unknown || passthrough_kernel) {
        const std::string source_stage = !contract.plugins[plugin_idx].name.empty()
                                             ? contract.plugins[plugin_idx].name
                                             : contract.plugins[plugin_idx].plugin_id;
        std::vector<EgressTensorContract> stage_outputs;
        for (const auto& tensor : contract.plugins[plugin_idx].output_tensors) {
          EgressTensorContract tensor_contract;
          if (populate_tensor_contract_from_stage_tensor(tensor, "application/vnd.simaai.tensor",
                                                         source_stage, &tensor_contract)) {
            stage_outputs.push_back(std::move(tensor_contract));
          }
        }
        if (stage_outputs.empty() && ordered_kind != OrderedRouteOp::Kind::Unknown) {
          populate_tensor_contract_from_stage_tensors(contract.plugins[plugin_idx].output_tensors,
                                                      "application/vnd.simaai.tensor", source_stage,
                                                      &out->egress_contract);
        } else if (!stage_outputs.empty()) {
          out->egress_contract = stage_outputs.front();
          if (passthrough_kernel) {
            out->egress_contracts = stage_outputs;
          } else {
            const bool preferred_output_stage =
                ordered_kind == OrderedRouteOp::Kind::Dequantize ||
                ordered_kind == OrderedRouteOp::Kind::DetessDequant ||
                ordered_kind == OrderedRouteOp::Kind::DetessCast ||
                ordered_kind == OrderedRouteOp::Kind::Cast ||
                ordered_kind == OrderedRouteOp::Kind::BoxDecode;
            auto& dst = preferred_output_stage ? aggregated_post_outputs_preferred
                                               : aggregated_post_outputs_fallback;
            dst.insert(dst.end(), stage_outputs.begin(), stage_outputs.end());
          }
        }
      }
    }
  }
  if (out->egress_contracts.empty()) {
    if (!aggregated_post_outputs_preferred.empty()) {
      out->egress_contracts = std::move(aggregated_post_outputs_preferred);
    } else if (!aggregated_post_outputs_fallback.empty()) {
      out->egress_contracts = std::move(aggregated_post_outputs_fallback);
    }
  }
  if (out->egress_contracts.size() == 1U && !out->egress_contract.valid) {
    out->egress_contract = out->egress_contracts.front();
  }
  if (pre_has_cast && pre_has_tess) {
    out->pre_kind = PreRouteStageKind::CastTess;
  } else if (pre_has_quant && pre_has_tess) {
    out->pre_kind = PreRouteStageKind::QuantTess;
  } else if (pre_has_quant) {
    out->pre_kind = PreRouteStageKind::Quant;
  } else if (pre_has_tess) {
    out->pre_kind = PreRouteStageKind::Tess;
  } else if (pre_has_preproc) {
    out->pre_kind = PreRouteStageKind::Preproc;
  } else if (pre_has_cast) {
    out->pre_kind = PreRouteStageKind::Cast;
  } else if (pre_has_unknown) {
    out->pre_kind = PreRouteStageKind::Unknown;
  } else {
    out->pre_kind = PreRouteStageKind::None;
  }
  out->has_strict_boxdecode_route = strict_model_managed_boxdecode_available(pack);
  if (route_debug_enabled()) {
    std::fprintf(stderr,
                 "[route-debug] mpk_summary pre=%s post=%s has_pre=%d has_post=%d tess=%d "
                 "pre_cast=%d detess=%d dequant=%d post_cast=%d boxdecode=%d strict_boxdecode=%d "
                 "ordered_pre=%s ordered_post=%s ingress_contracts=%s\n",
                 pre_kind_dbg(out->pre_kind), post_kind_dbg(out->post_kind),
                 out->has_external_pre ? 1 : 0, out->has_external_post ? 1 : 0,
                 out->has_external_tess ? 1 : 0, out->has_external_pre_cast ? 1 : 0,
                 out->has_external_detess ? 1 : 0, out->has_external_dequant ? 1 : 0,
                 out->has_external_post_cast ? 1 : 0, out->has_external_boxdecode ? 1 : 0,
                 out->has_strict_boxdecode_route ? 1 : 0,
                 ordered_route_chain_csv(out->ordered_pre_ops).c_str(),
                 ordered_route_chain_csv(out->ordered_post_ops).c_str(),
                 ingress_contracts_debug_string(out->ingress_contracts).c_str());
  }
  return true;
}

bool preprocess_auto_mode(const PreprocessOptions& p) {
  if (p.enable != AutoFlag::Auto)
    return false;
  if (p.kind != InputKind::Auto)
    return false;
  if (p.preset != NormalizePreset::None)
    return false;
  if (!p.transforms.empty())
    return false;
  if (p.input_max_width > 0 || p.input_max_height > 0 || p.input_max_depth > 0)
    return false;
  if (p.resize.enable != AutoFlag::Auto || p.resize.width > 0 || p.resize.height > 0)
    return false;
  if (p.color_convert.enable != AutoFlag::Auto ||
      p.color_convert.input_format != PreprocessColorFormat::Auto ||
      p.color_convert.output_format != PreprocessColorFormat::Auto) {
    return false;
  }
  if (p.layout_convert.enable != AutoFlag::Auto || p.layout_convert.has_perm()) {
    return false;
  }
  if (p.normalize.enable != AutoFlag::Auto || p.normalize.has_explicit_stats)
    return false;
  if (p.quantize.enable != AutoFlag::Auto)
    return false;
  if (p.tessellate.enable != AutoFlag::Auto)
    return false;
  return true;
}

bool generic_preproc_requested(const PreprocessOptions& p) {
  if (p.kind == InputKind::Image) {
    return true;
  }

  const auto resize_requested = [&]() {
    return p.resize.enable == AutoFlag::On || p.resize.width > 0 || p.resize.height > 0;
  };
  const auto color_requested = [&]() {
    return p.color_convert.enable == AutoFlag::On ||
           p.color_convert.input_format != PreprocessColorFormat::Auto ||
           p.color_convert.output_format != PreprocessColorFormat::Auto;
  };
  const auto layout_requested = [&]() {
    return p.layout_convert.enable == AutoFlag::On || p.layout_convert.has_perm();
  };
  const auto normalize_requested = [&]() {
    return p.normalize.enable == AutoFlag::On || p.normalize.has_explicit_stats ||
           p.preset != NormalizePreset::None;
  };
  if (resize_requested() || color_requested() || layout_requested() || normalize_requested()) {
    return true;
  }

  for (const auto& t : p.transforms) {
    switch (t.type) {
    case TransformType::Resize:
      if (t.resize.enable == AutoFlag::On || t.resize.width > 0 || t.resize.height > 0) {
        return true;
      }
      break;
    case TransformType::ColorConvert:
      if (t.color_convert.enable == AutoFlag::On ||
          t.color_convert.input_format != PreprocessColorFormat::Auto ||
          t.color_convert.output_format != PreprocessColorFormat::Auto) {
        return true;
      }
      break;
    case TransformType::LayoutConvert:
      if (t.layout_convert.enable == AutoFlag::On || t.layout_convert.has_perm()) {
        return true;
      }
      break;
    case TransformType::Normalize:
      if (t.normalize.enable == AutoFlag::On || t.normalize.has_explicit_stats) {
        return true;
      }
      break;
    case TransformType::Quantize:
    case TransformType::Tessellate:
      break;
    }
  }

  return false;
}

bool postprocess_auto_mode(const Model::Options& options) {
  return options.decode_type == BoxDecodeType::Unspecified && options.score_threshold == 0.0f &&
         options.nms_iou_threshold == 0.0f && options.top_k == 0;
}

bool user_requested_boxdecode(const Model::Options& options) {
  return options.decode_type != BoxDecodeType::Unspecified;
}

bool generic_preproc_fusion_supported(const RouteCapability& capability) {
  if (!capability.has_external_pre) {
    return false;
  }
  if (capability.pre_kind == PreRouteStageKind::Unknown) {
    // MPK chains that start with cast-only pre-adapter are handled by generic preproc cast path.
    return capability.mla_input_bf16 || capability.mla_input_quantized;
  }
  return capability.pre_kind == PreRouteStageKind::Preproc ||
         capability.pre_kind == PreRouteStageKind::Quant ||
         capability.pre_kind == PreRouteStageKind::Tess ||
         capability.pre_kind == PreRouteStageKind::QuantTess ||
         capability.pre_kind == PreRouteStageKind::CastTess;
}

bool strict_model_managed_boxdecode_contract_supported(
    const pipeline_internal::sima::BoxDecodeStaticContract& contract) {
  if (!pipeline_internal::sima::is_box_decode_type_specified(contract.decode_type)) {
    return false;
  }
  // The strict auto-selected model-managed route must resolve to a canonical backend decode type.
  // Legacy ambiguous "yolo" contracts are not stable enough for auto selection because the
  // runtime backend expects an explicit versioned layout family (for example yolov8 grouped or
  // decoupled heads). Keep those on the non-boxdecode post chain unless the model pack exposes a
  // fully canonical decode contract.
  if (contract.decode_type == BoxDecodeType::Yolo) {
    return false;
  }
  return !contract.tensors.empty();
}

bool strict_model_managed_boxdecode_available(const ModelPack& pack) {
  if (!pack.mpk_contract().has_value()) {
    return false;
  }
  std::string route_flags_error;
  const auto route_flags =
      pipeline_internal::sima::resolve_model_managed_boxdecode_route_flags_from_mpk(
          *pack.mpk_contract(), nullptr, &route_flags_error);
  if (!route_flags.has_value()) {
    return false;
  }
  std::string contract_error;
  const auto contract = pipeline_internal::sima::build_boxdecode_static_contract_from_mpk(
      *pack.mpk_contract(), *route_flags, &contract_error);
  return contract.has_value() && strict_model_managed_boxdecode_contract_supported(*contract);
}

bool boxdecode_route_available(const RouteCapability& capability) {
  return capability.has_external_boxdecode || capability.has_strict_boxdecode_route;
}

bool route_has_post_stage(const RouteCapability& capability) {
  return capability.has_external_post || capability.has_strict_boxdecode_route;
}

bool boxdecode_fusion_supported(const RouteCapability& capability) {
  if (boxdecode_route_available(capability)) {
    return true;
  }
  if (!capability.has_external_post) {
    return false;
  }
  return capability.post_kind == PostRouteStageKind::BoxDecode ||
         capability.post_kind == PostRouteStageKind::Detess ||
         capability.post_kind == PostRouteStageKind::DetessCast ||
         capability.post_kind == PostRouteStageKind::DetessDequant ||
         capability.post_kind == PostRouteStageKind::Dequantize;
}

PostRouteStageKind selected_post_kind_for_route(const RouteCapability& capability,
                                                const bool prefer_boxdecode) {
  if (!route_has_post_stage(capability)) {
    return PostRouteStageKind::None;
  }
  if (prefer_boxdecode) {
    if (!boxdecode_route_available(capability)) {
      return PostRouteStageKind::None;
    }
    return PostRouteStageKind::BoxDecode;
  }
  if (capability.has_external_boxdecode && capability.post_kind == PostRouteStageKind::BoxDecode) {
    return PostRouteStageKind::BoxDecode;
  }

  // Decision matrix (single source of policy truth):
  // non-boxdecode: (tess,quant)->{detessdequant,dequantize,detessdequant,dequantize}
  // boxdecode: always terminal boxdecode (kernel handles detess/dequant skips internally).
  if (capability.tess_needed && capability.quant_needed) {
    if (capability.has_external_detess || capability.has_external_dequant ||
        capability.post_kind == PostRouteStageKind::DetessDequant) {
      return PostRouteStageKind::DetessDequant;
    }
    return PostRouteStageKind::None;
  }
  if (capability.tess_needed) {
    if (capability.has_external_detess || capability.has_external_post_cast ||
        capability.post_kind == PostRouteStageKind::Detess ||
        capability.post_kind == PostRouteStageKind::DetessCast ||
        capability.post_kind == PostRouteStageKind::DetessDequant) {
      return capability.has_external_post_cast ||
                     capability.post_kind == PostRouteStageKind::DetessCast
                 ? PostRouteStageKind::DetessCast
                 : PostRouteStageKind::Detess;
    }
    return PostRouteStageKind::None;
  }
  if (capability.quant_needed) {
    if (capability.has_external_dequant || capability.has_external_post_cast ||
        capability.post_kind == PostRouteStageKind::Dequantize) {
      return PostRouteStageKind::Dequantize;
    }
    return PostRouteStageKind::None;
  }

  if (capability.has_external_post_cast || capability.post_kind == PostRouteStageKind::Cast ||
      capability.post_kind == PostRouteStageKind::DetessCast || capability.needs.post_cast) {
    return PostRouteStageKind::Cast;
  }

  return PostRouteStageKind::None;
}

PipelineType pipeline_type_from_pre_kind(PreRouteStageKind kind) {
  switch (kind) {
  case PreRouteStageKind::Preproc:
    return PipelineType::Preproc;
  case PreRouteStageKind::Quant:
    return PipelineType::Quant;
  case PreRouteStageKind::Tess:
    return PipelineType::Tess;
  case PreRouteStageKind::QuantTess:
    return PipelineType::QuantTess;
  case PreRouteStageKind::Cast:
    return PipelineType::Cast;
  case PreRouteStageKind::CastTess:
    return PipelineType::CastTess;
  case PreRouteStageKind::None:
  case PreRouteStageKind::Unknown:
    return PipelineType::Preproc;
  }
  return PipelineType::Preproc;
}

std::string pre_kind_name(PreRouteStageKind kind) {
  switch (kind) {
  case PreRouteStageKind::None:
    return "none";
  case PreRouteStageKind::Preproc:
    return "preproc";
  case PreRouteStageKind::Quant:
    return "quant";
  case PreRouteStageKind::Tess:
    return "tess";
  case PreRouteStageKind::QuantTess:
    return "quanttess";
  case PreRouteStageKind::Cast:
    return "cast";
  case PreRouteStageKind::CastTess:
    return "casttess";
  case PreRouteStageKind::Unknown:
    return "unknown";
  }
  return "unknown";
}

std::string post_kind_name(PostRouteStageKind kind) {
  switch (kind) {
  case PostRouteStageKind::None:
    return "none";
  case PostRouteStageKind::Detess:
    return "detess";
  case PostRouteStageKind::DetessDequant:
    return "detessdequant";
  case PostRouteStageKind::Dequantize:
    return "dequantize";
  case PostRouteStageKind::BoxDecode:
    return "boxdecode";
  case PostRouteStageKind::Cast:
    return "cast";
  case PostRouteStageKind::DetessCast:
    return "detesscast";
  case PostRouteStageKind::Unknown:
    return "unknown";
  }
  return "unknown";
}

std::string tess_location_name(TessellationLocation loc) {
  switch (loc) {
  case TessellationLocation::Unknown:
    return "unknown";
  case TessellationLocation::External:
    return "external";
  case TessellationLocation::MLA:
    return "mla";
  }
  return "unknown";
}

std::string generic_terminal_pre_stage_name(const RouteCapability& capability) {
  std::vector<SessionPreStageOp> mapped;
  mapped.reserve(capability.ordered_pre_ops.size());
  for (const auto& op : capability.ordered_pre_ops) {
    if (const auto pre = session_pre_op_from_ordered_kind(op.kind); pre.has_value()) {
      mapped.push_back(*pre);
    }
  }
  std::vector<SessionPreStageOp> fused;
  fused.reserve(mapped.size());
  for (std::size_t i = 0; i < mapped.size(); ++i) {
    const auto current = mapped[i];
    if (current == SessionPreStageOp::Quant && i + 1U < mapped.size() &&
        mapped[i + 1U] == SessionPreStageOp::Tess) {
      fused.push_back(SessionPreStageOp::QuantTess);
      ++i;
      continue;
    }
    if (current == SessionPreStageOp::Cast && i + 1U < mapped.size() &&
        mapped[i + 1U] == SessionPreStageOp::Tess) {
      fused.push_back(SessionPreStageOp::CastTess);
      ++i;
      continue;
    }
    fused.push_back(current);
  }
  const auto terminal = !fused.empty() ? fused.back() : ([&]() {
    switch (capability.pre_kind) {
    case PreRouteStageKind::Preproc:
      return SessionPreStageOp::Preproc;
    case PreRouteStageKind::Quant:
      return SessionPreStageOp::Quant;
    case PreRouteStageKind::Tess:
      return SessionPreStageOp::Tess;
    case PreRouteStageKind::QuantTess:
      return SessionPreStageOp::QuantTess;
    case PreRouteStageKind::Cast:
      return SessionPreStageOp::Cast;
    case PreRouteStageKind::CastTess:
      return SessionPreStageOp::CastTess;
    case PreRouteStageKind::None:
    case PreRouteStageKind::Unknown:
      return SessionPreStageOp::Preproc;
    }
    return SessionPreStageOp::Preproc;
  })();
  switch (terminal) {
  case SessionPreStageOp::Preproc:
    return "preproc";
  case SessionPreStageOp::Quant:
    return "quant";
  case SessionPreStageOp::Tess:
    return "tess";
  case SessionPreStageOp::QuantTess:
    return "quanttess";
  case SessionPreStageOp::Cast:
    return "cast";
  case SessionPreStageOp::CastTess:
    return "casttess";
  }
  return "preproc";
}

} // namespace

ModelSemantics build_model_semantics(const ModelPack& pack) {
  ModelSemantics out;
  (void)parse_model_semantics_from_pack(pack, &out);
  return out;
}

SessionRoutePlan build_route_plan(const Model::Options& options, const ModelSemantics& semantics,
                                  const RouteCapability* capability, const ModelPack* pack) {
  SessionRoutePlan out;
  const bool have_capability = capability != nullptr;
  if (have_capability && !capability->ingress_contracts.empty()) {
    out.ingress_contracts = capability->ingress_contracts;
    out.ingress_regions = capability->ingress_regions;
  } else {
    out.ingress_contracts = {make_single_ingress_contract_from_semantics(semantics)};
    out.ingress_regions = ingress_regions_from_contracts(out.ingress_contracts);
  }
  if (have_capability && capability->egress_contract.valid) {
    out.egress_contract = capability->egress_contract;
  }
  if (have_capability && !capability->egress_contracts.empty()) {
    out.egress_contracts = capability->egress_contracts;
    if (!out.egress_contract.valid) {
      out.egress_contract = out.egress_contracts.front();
    }
  } else if (out.egress_contract.valid) {
    out.egress_contracts = {out.egress_contract};
  }

  out.preproc_context.quant_needed = false;
  out.preproc_context.tess_needed = false;
  out.preproc_context.pre_cast_needed = semantics.pre_cast_needed;

  const bool user_requested_preproc = generic_preproc_requested(options.preprocess);
  out.use_preproc = user_requested_preproc;
  out.boxdecode_selected = user_requested_boxdecode(options);

  const auto dtype_is_bf16_like = [](const std::string& raw) {
    const std::string token = lower_copy(raw);
    return token.find("bf16") != std::string::npos || token.find("bfloat16") != std::string::npos;
  };
  const bool mla_output_is_bf16 = dtype_is_bf16_like(semantics.mla_output_dtype_raw);

  const bool use_ordered_pre_ops = have_capability && !capability->ordered_pre_ops.empty();
  if (use_ordered_pre_ops) {
    std::vector<SessionPreStageOp> desired_pre_chain;
    desired_pre_chain.reserve(capability->ordered_pre_ops.size());
    bool ordered_pre_has_preproc = false;
    bool ordered_pre_has_quant = false;
    bool ordered_pre_has_tess = false;
    bool ordered_pre_has_cast = false;

    for (const auto& op : capability->ordered_pre_ops) {
      if (const auto mapped = session_pre_op_from_ordered_kind(op.kind); mapped.has_value()) {
        desired_pre_chain.push_back(*mapped);
      }
      switch (op.kind) {
      case OrderedRouteOp::Kind::Preproc:
        ordered_pre_has_preproc = true;
        break;
      case OrderedRouteOp::Kind::Quant:
        ordered_pre_has_quant = true;
        break;
      case OrderedRouteOp::Kind::Tess:
        ordered_pre_has_tess = true;
        break;
      case OrderedRouteOp::Kind::QuantTess:
        ordered_pre_has_quant = true;
        ordered_pre_has_tess = true;
        break;
      case OrderedRouteOp::Kind::Cast:
        ordered_pre_has_cast = true;
        break;
      case OrderedRouteOp::Kind::CastTess:
        ordered_pre_has_cast = true;
        ordered_pre_has_tess = true;
        break;
      case OrderedRouteOp::Kind::Unknown:
      case OrderedRouteOp::Kind::Detess:
      case OrderedRouteOp::Kind::DetessCast:
      case OrderedRouteOp::Kind::DetessDequant:
      case OrderedRouteOp::Kind::Dequantize:
      case OrderedRouteOp::Kind::BoxDecode:
      case OrderedRouteOp::Kind::Unpack:
        break;
      }
    }

    out.diagnostics.push_back("session-route: ordered_pre_chain=" +
                              ordered_route_chain_csv(capability->ordered_pre_ops));
    out.diagnostics.push_back("session-route: desired_pre_chain=" +
                              session_pre_chain_csv(desired_pre_chain));

    if (user_requested_preproc) {
      out.pre_chain = {SessionPreStageOp::Preproc};
      out.use_preproc = true;
      out.diagnostics.push_back("session-route: pre_fusion=user_preproc(cast+quant+tess)->preproc");
    } else {
      std::vector<SessionPreStageOp> fused_pre_chain;
      fused_pre_chain.reserve(desired_pre_chain.size());
      bool fused_quant_tess = false;
      bool fused_cast_tess = false;
      for (std::size_t i = 0; i < desired_pre_chain.size(); ++i) {
        const auto current = desired_pre_chain[i];
        if (current == SessionPreStageOp::Quant && i + 1U < desired_pre_chain.size() &&
            desired_pre_chain[i + 1U] == SessionPreStageOp::Tess) {
          fused_pre_chain.push_back(SessionPreStageOp::QuantTess);
          ++i;
          fused_quant_tess = true;
          continue;
        }
        if (current == SessionPreStageOp::Cast && i + 1U < desired_pre_chain.size() &&
            desired_pre_chain[i + 1U] == SessionPreStageOp::Tess) {
          fused_pre_chain.push_back(SessionPreStageOp::CastTess);
          ++i;
          fused_cast_tess = true;
          continue;
        }
        fused_pre_chain.push_back(current);
      }
      out.pre_chain = std::move(fused_pre_chain);
      out.use_preproc = std::find(out.pre_chain.begin(), out.pre_chain.end(),
                                  SessionPreStageOp::Preproc) != out.pre_chain.end();
      if (fused_quant_tess) {
        out.diagnostics.push_back("session-route: pre_fusion=quant+tess->quanttess");
      }
      if (fused_cast_tess) {
        out.diagnostics.push_back("session-route: pre_fusion=cast+tess->casttess");
      }
      if (!out.use_preproc && ordered_pre_has_preproc) {
        out.use_preproc = true;
      }
    }

    out.preproc_context.pre_quant_needed = ordered_pre_has_quant;
    out.preproc_context.pre_tess_needed = ordered_pre_has_tess;
    out.preproc_context.pre_cast_needed = ordered_pre_has_cast;

    const bool chain_has_quant = std::find(out.pre_chain.begin(), out.pre_chain.end(),
                                           SessionPreStageOp::Quant) != out.pre_chain.end() ||
                                 std::find(out.pre_chain.begin(), out.pre_chain.end(),
                                           SessionPreStageOp::QuantTess) != out.pre_chain.end();
    const bool chain_has_tess = std::find(out.pre_chain.begin(), out.pre_chain.end(),
                                          SessionPreStageOp::Tess) != out.pre_chain.end() ||
                                std::find(out.pre_chain.begin(), out.pre_chain.end(),
                                          SessionPreStageOp::QuantTess) != out.pre_chain.end() ||
                                std::find(out.pre_chain.begin(), out.pre_chain.end(),
                                          SessionPreStageOp::CastTess) != out.pre_chain.end();
    const bool chain_has_cast = std::find(out.pre_chain.begin(), out.pre_chain.end(),
                                          SessionPreStageOp::Cast) != out.pre_chain.end() ||
                                std::find(out.pre_chain.begin(), out.pre_chain.end(),
                                          SessionPreStageOp::CastTess) != out.pre_chain.end();

    out.pre_cast_fp32_to_bf16 = chain_has_cast;
    if (chain_has_cast && chain_has_tess && !chain_has_quant) {
      out.pre_adapter = SessionPreAdapterKind::CastTess;
    } else if (chain_has_quant && chain_has_tess) {
      out.pre_adapter = SessionPreAdapterKind::QuantTess;
    } else if (chain_has_quant) {
      out.pre_adapter = SessionPreAdapterKind::Quant;
    } else if (chain_has_tess) {
      out.pre_adapter = SessionPreAdapterKind::Tess;
    } else {
      out.pre_adapter = SessionPreAdapterKind::None;
    }

    out.include_pre_stage = !out.pre_chain.empty();
    if (out.use_preproc) {
      out.pipeline_type = PipelineType::Preproc;
    } else {
      switch (out.pre_adapter) {
      case SessionPreAdapterKind::None:
        out.pipeline_type = out.pre_cast_fp32_to_bf16 ? PipelineType::Cast : PipelineType::Preproc;
        break;
      case SessionPreAdapterKind::Quant:
        out.pipeline_type = PipelineType::Quant;
        break;
      case SessionPreAdapterKind::Tess:
        out.pipeline_type = PipelineType::Tess;
        break;
      case SessionPreAdapterKind::QuantTess:
        out.pipeline_type = PipelineType::QuantTess;
        break;
      case SessionPreAdapterKind::CastTess:
        out.pipeline_type = PipelineType::CastTess;
        break;
      }
    }
  } else {
    bool quant_needed_for_pre = semantics.pre_quant_needed;
    bool tess_needed_for_pre = semantics.pre_tess_needed;
    if (have_capability) {
      switch (capability->pre_kind) {
      case PreRouteStageKind::Quant:
        quant_needed_for_pre = true;
        tess_needed_for_pre = false;
        break;
      case PreRouteStageKind::Tess:
        quant_needed_for_pre = false;
        tess_needed_for_pre = true;
        break;
      case PreRouteStageKind::QuantTess:
        quant_needed_for_pre = true;
        tess_needed_for_pre = true;
        break;
      case PreRouteStageKind::Cast:
        quant_needed_for_pre = false;
        tess_needed_for_pre = false;
        break;
      case PreRouteStageKind::Preproc:
        if (capability->has_external_pre) {
          quant_needed_for_pre = false;
          tess_needed_for_pre = false;
        }
        break;
      case PreRouteStageKind::CastTess:
        quant_needed_for_pre = false;
        tess_needed_for_pre = true;
        break;
      case PreRouteStageKind::None:
      case PreRouteStageKind::Unknown:
        break;
      }
    }
    out.preproc_context.pre_quant_needed = quant_needed_for_pre;
    out.preproc_context.pre_tess_needed = tess_needed_for_pre;
    if (out.use_preproc) {
      out.pre_adapter = SessionPreAdapterKind::None;
    } else if (tess_needed_for_pre && !quant_needed_for_pre &&
               (semantics.pre_cast_needed ||
                (have_capability && capability->has_external_pre_cast))) {
      out.pre_adapter = SessionPreAdapterKind::CastTess;
    } else if (tess_needed_for_pre && quant_needed_for_pre) {
      out.pre_adapter = SessionPreAdapterKind::QuantTess;
    } else if (quant_needed_for_pre) {
      out.pre_adapter = SessionPreAdapterKind::Quant;
    } else if (tess_needed_for_pre) {
      out.pre_adapter = SessionPreAdapterKind::Tess;
    } else {
      out.pre_adapter = SessionPreAdapterKind::None;
    }

    out.pre_chain.clear();
    if (out.use_preproc) {
      out.pre_chain.push_back(SessionPreStageOp::Preproc);
    }
    out.pre_cast_fp32_to_bf16 = semantics.pre_cast_needed;
    if (!out.pre_cast_fp32_to_bf16 && have_capability && capability->has_external_pre_cast &&
        !out.use_preproc) {
      out.pre_cast_fp32_to_bf16 = true;
    }
    out.preproc_context.pre_cast_needed = out.pre_cast_fp32_to_bf16;
    if (out.pre_cast_fp32_to_bf16 && !out.use_preproc &&
        out.pre_adapter != SessionPreAdapterKind::CastTess) {
      out.pre_chain.push_back(SessionPreStageOp::Cast);
    }
    switch (out.pre_adapter) {
    case SessionPreAdapterKind::None:
      break;
    case SessionPreAdapterKind::Quant:
      out.pre_chain.push_back(SessionPreStageOp::Quant);
      break;
    case SessionPreAdapterKind::Tess:
      out.pre_chain.push_back(SessionPreStageOp::Tess);
      break;
    case SessionPreAdapterKind::QuantTess:
      out.pre_chain.push_back(SessionPreStageOp::QuantTess);
      break;
    case SessionPreAdapterKind::CastTess:
      out.pre_chain.push_back(SessionPreStageOp::CastTess);
      break;
    }
    out.include_pre_stage = !out.pre_chain.empty();
    if (out.use_preproc) {
      out.pipeline_type = PipelineType::Preproc;
    } else {
      switch (out.pre_adapter) {
      case SessionPreAdapterKind::None:
        out.pipeline_type = out.pre_cast_fp32_to_bf16 ? PipelineType::Cast : PipelineType::Preproc;
        break;
      case SessionPreAdapterKind::Quant:
        out.pipeline_type = PipelineType::Quant;
        break;
      case SessionPreAdapterKind::Tess:
        out.pipeline_type = PipelineType::Tess;
        break;
      case SessionPreAdapterKind::QuantTess:
        out.pipeline_type = PipelineType::QuantTess;
        break;
      case SessionPreAdapterKind::CastTess:
        out.pipeline_type = PipelineType::CastTess;
        break;
      }
    }
  }

  out.post_chain.clear();
  const bool needs_tess = have_capability ? capability->tess_needed : semantics.tess_needed;
  const bool needs_quant = have_capability ? capability->quant_needed : semantics.quant_needed;
  out.model_managed_route_flags.tess_needed = needs_tess;
  out.model_managed_route_flags.quant_needed = needs_quant;
  out.model_managed_route_flags.pre_cast_needed = out.preproc_context.pre_cast_needed;
  out.model_managed_route_flags.quant_contract_required = needs_quant;
  out.preproc_context.quant_needed = out.model_managed_route_flags.quant_needed;
  out.preproc_context.tess_needed = out.model_managed_route_flags.tess_needed;

  const bool has_detess_adapter =
      semantics.has_post_detess_adapter || (have_capability && capability->has_external_detess);
  const bool has_dequant_adapter =
      semantics.has_post_dequant_adapter || (have_capability && capability->has_external_dequant);
  const bool has_cast_adapter =
      semantics.has_post_cast_adapter || (have_capability && capability->has_external_post_cast);
  const bool has_boxdecode_adapter =
      semantics.has_post_boxdecode || (have_capability && (capability->has_external_boxdecode ||
                                                           capability->has_strict_boxdecode_route));
  const bool post_auto = postprocess_auto_mode(options);
  (void)post_auto;

  const bool use_ordered_post_ops = have_capability && !capability->ordered_post_ops.empty();
  if (use_ordered_post_ops) {
    std::vector<SessionPostStageOp> desired_post_chain;
    desired_post_chain.reserve(capability->ordered_post_ops.size());
    for (const auto& op : capability->ordered_post_ops) {
      if (const auto mapped = session_post_op_from_ordered_kind(op.kind); mapped.has_value()) {
        desired_post_chain.push_back(*mapped);
      }
    }
    out.diagnostics.push_back("session-route: ordered_post_chain=" +
                              ordered_route_chain_csv(capability->ordered_post_ops));
    out.diagnostics.push_back("session-route: desired_post_chain=" +
                              session_post_chain_csv(desired_post_chain));

    if (out.boxdecode_selected) {
      out.post_chain = {SessionPostStageOp::BoxDecode};
      out.diagnostics.push_back(
          "session-route: post_fusion=user_boxdecode(cast+detess+dequant)->boxdecode");
    } else {
      std::vector<SessionPostStageOp> filtered_post_chain;
      filtered_post_chain.reserve(desired_post_chain.size());
      for (const auto op : desired_post_chain) {
        if (op == SessionPostStageOp::BoxDecode) {
          continue;
        }
        filtered_post_chain.push_back(op);
      }
      std::size_t periodic_prefix = 0U;
      if (collapse_periodic_chain(&filtered_post_chain, &periodic_prefix)) {
        out.diagnostics.push_back("session-route: post_normalize=collapse_periodic(period=" +
                                  std::to_string(periodic_prefix) + ")");
      }

      std::vector<SessionPostStageOp> fused_post_chain;
      fused_post_chain.reserve(filtered_post_chain.size());
      bool fused_detess_dequant = false;
      bool fused_detess_cast = false;
      bool skipped_detess_dequant_fusion = false;
      for (std::size_t i = 0; i < filtered_post_chain.size(); ++i) {
        const auto current = filtered_post_chain[i];
        const bool has_adjacent_dequant =
            (i + 1U < filtered_post_chain.size()) &&
            (filtered_post_chain[i + 1U] == SessionPostStageOp::Dequantize);
        if ((current == SessionPostStageOp::Detess ||
             current == SessionPostStageOp::DetessDequant) &&
            has_adjacent_dequant) {
          if (needs_quant) {
            fused_post_chain.push_back(SessionPostStageOp::DetessDequant);
            ++i;
            fused_detess_dequant = true;
            continue;
          }
          skipped_detess_dequant_fusion = true;
        }
        const bool has_adjacent_cast = (i + 1U < filtered_post_chain.size()) &&
                                       (filtered_post_chain[i + 1U] == SessionPostStageOp::Cast);
        if ((current == SessionPostStageOp::Detess || current == SessionPostStageOp::DetessCast) &&
            has_adjacent_cast && !needs_quant) {
          fused_post_chain.push_back(SessionPostStageOp::DetessCast);
          ++i;
          fused_detess_cast = true;
          continue;
        }
        fused_post_chain.push_back(current);
      }
      periodic_prefix = 0U;
      if (collapse_periodic_chain(&fused_post_chain, &periodic_prefix)) {
        out.diagnostics.push_back("session-route: post_normalize=collapse_periodic_fused(period=" +
                                  std::to_string(periodic_prefix) + ")");
      }
      std::size_t removed_adjacent = 0U;
      if (collapse_adjacent_duplicates(&fused_post_chain, &removed_adjacent)) {
        out.diagnostics.push_back(
            "session-route: post_normalize=collapse_adjacent_duplicates(removed=" +
            std::to_string(removed_adjacent) + ")");
      }
      if (has_duplicate_ops(desired_post_chain)) {
        std::vector<SessionPostStageOp> canonical_post_chain;
        bool needs_post_cast = semantics.post_cast_needed || mla_output_is_bf16;
        if (have_capability) {
          needs_post_cast = needs_post_cast || capability->needs.post_cast;
        }
        if (needs_tess && needs_quant) {
          canonical_post_chain.push_back(SessionPostStageOp::DetessDequant);
          needs_post_cast = false;
        } else if (!needs_tess && needs_quant) {
          canonical_post_chain.push_back(SessionPostStageOp::Dequantize);
          needs_post_cast = false;
        } else if (needs_tess && !needs_quant) {
          canonical_post_chain.push_back(needs_post_cast ? SessionPostStageOp::DetessCast
                                                         : SessionPostStageOp::Detess);
          needs_post_cast = false;
        }
        if (needs_post_cast) {
          canonical_post_chain.push_back(SessionPostStageOp::Cast);
        }
        if (!canonical_post_chain.empty()) {
          fused_post_chain = std::move(canonical_post_chain);
          out.diagnostics.push_back("session-route: post_normalize=canonicalize_fanout_from_needs");
        }
      }
      if (!needs_quant) {
        bool downgraded_detess_only = false;
        for (auto& op : fused_post_chain) {
          if (op == SessionPostStageOp::DetessDequant) {
            op = SessionPostStageOp::Detess;
            downgraded_detess_only = true;
          }
        }
        if (downgraded_detess_only) {
          out.diagnostics.push_back(
              "session-route: post_normalize=demote_detessdequant_to_detess(route_quant_off)");
        }
      }
      out.post_chain = std::move(fused_post_chain);
      if (fused_detess_dequant) {
        out.diagnostics.push_back("session-route: post_fusion=detess+dequant->detessdequant");
      }
      if (fused_detess_cast) {
        out.diagnostics.push_back("session-route: post_fusion=detess+cast->detesscast");
      } else if (skipped_detess_dequant_fusion) {
        out.diagnostics.push_back(
            "session-route: post_fusion=detess+dequant_skipped(route_quant_off)");
      }
    }
  } else {
    if (out.boxdecode_selected) {
      out.post_chain.push_back(SessionPostStageOp::BoxDecode);
      if (!has_boxdecode_adapter) {
        out.diagnostics.push_back("session-route: requested boxdecode forced despite missing "
                                  "advertised boxdecode capability");
      }
    } else {
      bool needs_post_cast = semantics.post_cast_needed || mla_output_is_bf16;
      if (have_capability) {
        needs_post_cast = needs_post_cast || capability->needs.post_cast;
      }
      if (needs_tess && needs_quant) {
        if (has_detess_adapter || has_dequant_adapter) {
          out.post_chain.push_back(SessionPostStageOp::DetessDequant);
        } else {
          out.diagnostics.push_back("session-route: detessdequant required but no post "
                                    "detess/dequant adapter is available");
        }
        needs_post_cast = false;
      } else if (!needs_tess && needs_quant) {
        if (has_dequant_adapter) {
          out.post_chain.push_back(SessionPostStageOp::Dequantize);
        } else {
          out.diagnostics.push_back(
              "session-route: dequant required but no post dequant adapter is available");
        }
        needs_post_cast = false;
      } else if (needs_tess && !needs_quant) {
        if (has_detess_adapter) {
          out.post_chain.push_back(needs_post_cast ? SessionPostStageOp::DetessCast
                                                   : SessionPostStageOp::Detess);
          needs_post_cast = false;
        } else {
          out.diagnostics.push_back(
              "session-route: detess required but no post detess adapter is available");
        }
      }

      if (needs_post_cast) {
        if (!has_cast_adapter && have_capability) {
          out.diagnostics.push_back("session-route: post cast requested without explicit cast "
                                    "adapter; using host cast node");
        }
        out.post_chain.push_back(SessionPostStageOp::Cast);
      }
    }
  }

  out.model_managed_route_flags.pre_cast_needed = out.preproc_context.pre_cast_needed;
  out.model_managed_route_flags.include_pre_stage = out.include_pre_stage;
  out.model_managed_route_flags.boxdecode_selected = out.boxdecode_selected;
  out.cast_symmetry_ok = semantics.cast_symmetry_ok;

  if (!out.ingress_contracts.empty()) {
    normalize_ingress_contracts_for_pre_chain(&out.ingress_contracts, out.pre_chain);
    out.ingress_regions = ingress_regions_from_contracts(out.ingress_contracts);
  }

  // Prefer the structural derivation from ingress_regions when available.
  // ingress_regions encodes per-depth FanoutMap multiplicity and the optional
  // FaninJoin (packer) explicitly, so the resulting pre_regions correctly
  // represents native multi-IFM / packer-style models as a single FanoutMap
  // (which the materializer collapses to a multi-input pre node) instead of
  // N separate Linear regions. For pipelines without ingress_contracts (e.g.
  // legacy monolithic-input flows) we fall back to the chain-derived form to
  // preserve identical behavior. For user-requested preproc we honor the flat
  // chain (single Linear(Preproc) region) since preproc replaces the cast/
  // quant/tess sequence.
  if (out.use_preproc || out.ingress_regions.empty()) {
    out.pre_regions = pre_regions_from_pre_chain(out.pre_chain);
  } else {
    // Always fuse Q+T -> QuantTess and C+T -> CastTess at the region level
    // when adjacent regions have the same multiplicity. This is the canonical
    // pre-MLA family the renderer materializes — separate Cast+Tess (or
    // Quant+Tess) regions never want a separate-element materialization.
    out.pre_regions = derive_pre_regions_from_ingress_regions(out.ingress_regions, true, true);
  }
  if (!out.ingress_regions.empty()) {
    out.diagnostics.push_back("session-route: ingress_regions=" +
                              route_region_csv(out.ingress_regions));
  }
  std::vector<RouteRegion> raw_graph_post_regions;
  if (!out.boxdecode_selected && pack != nullptr) {
    raw_graph_post_regions = derive_post_regions_from_graph(*pack);
    if (!raw_graph_post_regions.empty()) {
      out.diagnostics.push_back("session-route: graph_post_chain_raw=" +
                                graph_chain_csv_from_regions(raw_graph_post_regions));
      out.diagnostics.push_back("session-route: graph_post_regions_raw=" +
                                route_region_csv(raw_graph_post_regions));
    }
  }
  if (!raw_graph_post_regions.empty()) {
    out.post_regions = filter_non_materialized_post_regions(raw_graph_post_regions);
  } else {
    out.post_regions = post_regions_from_post_chain(out.post_chain, out.egress_contracts);
  }
  finalize_post_summary_from_regions(&out);
  out.infer_only = !out.include_pre_stage && !out.include_post_stage;

  out.diagnostics.push_back("session-route: final_pre_chain=" +
                            session_pre_chain_csv(out.pre_chain));
  out.diagnostics.push_back("session-route: final_post_chain=" +
                            session_post_chain_csv(out.post_chain));
  out.diagnostics.push_back("session-route: final_pre_regions=" +
                            route_region_csv(out.pre_regions));
  out.diagnostics.push_back("session-route: final_post_regions=" +
                            route_region_csv(out.post_regions));

  if (!out.ingress_contracts.empty()) {
    out.diagnostics.push_back("session-route: ingress_contracts=" +
                              ingress_contracts_debug_string(out.ingress_contracts));
  }
  return out;
}
RouteCapability extract_route_capability(const ModelPack& pack,
                                         const PreprocessPlannerResult& preprocess_plan) {
  (void)preprocess_plan;
  RouteCapability out;
  const bool used_mpk_graph = extract_route_capability_from_mpk_graph(pack, &out);
  if (!used_mpk_graph) {
    throw std::runtime_error(
        "RoutePlanner: MPK graph route extraction failed; pipeline_sequence fallback is disabled."
        " The MPK manifest must define a valid plugin graph with edges and stages."
        " Verify the model pack was built with a supported MPK version.");
  }
  out.evidence.push_back("route_capability_source=mpk_graph");
  if (pack.mpk_contract().has_value()) {
    out.evidence.push_back("mpk_plugins=" + std::to_string(pack.mpk_contract()->plugins.size()));
    out.evidence.push_back("mpk_edges=" + std::to_string(pack.mpk_contract()->edges.size()));
  }

  if (out.has_external_tess || out.has_external_detess || out.has_external_dequant) {
    out.tessellation_location = TessellationLocation::External;
  } else {
    out.tessellation_location = TessellationLocation::Unknown;
  }

  if (!pack.mpk_contract().has_value()) {
    throw std::runtime_error(
        "RoutePlanner: strict MPK contract is required for MLA planning facts."
        " Ensure the model pack includes an mpk.json manifest with plugin IO contracts.");
  }
  std::string mla_fact_error;
  if (!populate_route_mla_facts_from_mpk_contract(*pack.mpk_contract(), &out, &mla_fact_error)) {
    throw std::runtime_error(
        "RoutePlanner: strict MPK MLA planning facts are missing required tensor contracts"
        " (plugins=" +
        std::to_string(pack.mpk_contract()->plugins.size()) +
        ", edges=" + std::to_string(pack.mpk_contract()->edges.size()) + ")" +
        (mla_fact_error.empty() ? std::string(". No additional detail available.")
                                : ": " + mla_fact_error));
  }
  out.evidence.push_back("mla_planning_source=mpk_static_contract");

  out.mla_input_bf16 = dtype_is_bf16_like(out.mla_input_dtype_raw);
  out.mla_input_quantized = dtype_is_quantized_like(out.mla_input_dtype_raw);
  out.mla_output_bf16 = dtype_is_bf16_like(out.mla_output_dtype_raw);
  out.mla_output_quantized = dtype_is_quantized_like(out.mla_output_dtype_raw);

  out.tess_needed = out.has_external_tess || out.has_external_detess;
  // MPK semantics: quant requirement tracks MLA egress dtype, not whether a
  // post adapter is present. Some BF16-input models still emit INT8 OFMs and
  // rely on post decode/dequant handling.
  out.quant_needed = out.mla_output_quantized || out.has_external_dequant;
  // Detess-only post routes should not force dequant semantics when MPK
  // already emits float logical outputs after detessellation.
  if (out.quant_needed && out.has_external_detess && !out.has_external_dequant) {
    out.quant_needed = false;
    out.evidence.push_back("quant_needed_override=detess_only_post");
  }
  // Cast-only post routes on BF16 ingress should not force quantized
  // post-processing semantics even when MLA egress is packed as INT8.
  if (out.quant_needed && out.mla_input_bf16 && out.has_external_post_cast &&
      !out.has_external_dequant && out.post_kind == PostRouteStageKind::Cast) {
    out.quant_needed = false;
    out.evidence.push_back("quant_needed_override=bf16_cast_only_post");
  }

  out.needs.pre_quantization =
      out.has_external_pre &&
      (out.pre_kind == PreRouteStageKind::Quant || out.pre_kind == PreRouteStageKind::QuantTess);
  out.needs.pre_tessellation =
      out.has_external_tess ||
      (out.pre_kind == PreRouteStageKind::Tess || out.pre_kind == PreRouteStageKind::QuantTess);
  out.needs.pre_cast = out.has_external_pre_cast;
  out.needs.post_detessellation = out.tess_needed;
  out.needs.post_dequantization = out.has_external_dequant;
  out.needs.post_cast = out.has_external_post_cast;
  if (out.needs.pre_cast && !out.needs.post_cast) {
    out.evidence.push_back("cast_symmetry_unresolved=pre_cast_without_post_cast");
  }

  out.adapter_capabilities.has_pre_quantization =
      out.has_external_pre &&
      (out.pre_kind == PreRouteStageKind::Quant || out.pre_kind == PreRouteStageKind::QuantTess);
  out.adapter_capabilities.has_pre_tessellation = out.has_external_tess;
  out.adapter_capabilities.has_pre_cast = out.has_external_pre_cast;
  out.adapter_capabilities.has_post_detessellation = out.has_external_detess;
  out.adapter_capabilities.has_post_dequantization = out.has_external_dequant;
  out.adapter_capabilities.has_post_cast = out.has_external_post_cast;
  out.adapter_capabilities.has_post_boxdecode =
      out.has_external_boxdecode || out.has_strict_boxdecode_route;

  out.evidence.push_back("pre_kind=" + pre_kind_name(out.pre_kind));
  out.evidence.push_back("post_kind=" + post_kind_name(out.post_kind));
  out.evidence.push_back("tess_location=" + tess_location_name(out.tessellation_location));
  out.evidence.push_back("mla_input_dtype=" + out.mla_input_dtype_raw);
  out.evidence.push_back("mla_output_dtype=" + out.mla_output_dtype_raw);
  out.evidence.push_back("mla_input_media_type=" + out.mla_input_media_type);
  out.evidence.push_back(std::string("tess_needed=") + (out.tess_needed ? "1" : "0"));
  out.evidence.push_back(std::string("quant_needed=") + (out.quant_needed ? "1" : "0"));
  out.evidence.push_back("ordered_pre_chain=" + ordered_route_chain_csv(out.ordered_pre_ops));
  out.evidence.push_back("ordered_post_chain=" + ordered_route_chain_csv(out.ordered_post_ops));
  if (!out.ingress_contracts.empty()) {
    out.evidence.push_back("ingress_contracts=" +
                           ingress_contracts_debug_string(out.ingress_contracts));
  }
  out.evidence.push_back(std::string("needs_post_cast=") + (out.needs.post_cast ? "1" : "0"));
  out.evidence.push_back(std::string("cap_post_boxdecode=") +
                         (out.adapter_capabilities.has_post_boxdecode ? "1" : "0"));
  out.evidence.push_back(std::string("strict_boxdecode_route=") +
                         (out.has_strict_boxdecode_route ? "1" : "0"));
  out.evidence.push_back(std::string("fusion_genericpreproc_supported=") +
                         (generic_preproc_fusion_supported(out) ? "1" : "0"));
  out.evidence.push_back(std::string("fusion_boxdecode_supported=") +
                         (boxdecode_fusion_supported(out) ? "1" : "0"));
  if (route_debug_enabled()) {
    std::fprintf(stderr, "[route-debug] capability final pre=%s post=%s has_pre=%d has_post=%d\n",
                 pre_kind_name(out.pre_kind).c_str(), post_kind_name(out.post_kind).c_str(),
                 out.has_external_pre ? 1 : 0, out.has_external_post ? 1 : 0);
  }

  return out;
}

RouteSelection plan_route_selection(const Model::Options& options,
                                    const PreprocessPlannerResult& preprocess_plan,
                                    const RouteCapability& capability) {
  RouteSelection out;
  RouteEffectiveRoute& effective = out.effective;
  RouteUserIntent& intent = out.user_intent;

  effective.pipeline_type = preprocess_plan.pipeline_type;
  effective.include_preprocess_stage = preprocess_plan.include_preprocess_stage;
  effective.include_postprocess_stage = preprocess_plan.include_postprocess_stage;
  out.modelpack_media_type = preprocess_plan.modelpack_media_type;
  out.modelpack_format = preprocess_plan.modelpack_format;
  out.modelpack_input_depth = preprocess_plan.modelpack_input_depth;
  out.modelpack_max_width = preprocess_plan.modelpack_max_width;
  out.modelpack_max_height = preprocess_plan.modelpack_max_height;
  out.modelpack_max_depth = preprocess_plan.modelpack_max_depth;
  effective.mla_tessellation = capability.tessellation_location == TessellationLocation::MLA;

  intent.pre_auto = preprocess_auto_mode(options.preprocess);
  intent.post_auto = postprocess_auto_mode(options);
  intent.requested_boxdecode = user_requested_boxdecode(options);
  const bool pre_auto = intent.pre_auto;
  const bool post_auto = intent.post_auto;
  const bool requested_boxdecode = intent.requested_boxdecode;
  const bool mla_video_ingress = media_type_is_video_raw(capability.mla_input_media_type);

  if (pre_auto) {
    if (capability.has_external_pre) {
      if (capability.pre_kind == PreRouteStageKind::Cast) {
        effective.include_preprocess_stage = true;
        effective.pipeline_type = PipelineType::Cast;
        out.modelpack_media_type = "application/vnd.simaai.tensor";
        out.modelpack_format = "FP32";
        if (out.modelpack_input_depth <= 0)
          out.modelpack_input_depth = 3;
        out.diagnostics.push_back("auto pre-route: cast-only pre stage detected; use tensor "
                                  "ingress with typed cast node");
      } else {
        effective.pipeline_type = pipeline_type_from_pre_kind(capability.pre_kind);
        effective.include_preprocess_stage = true;
        if (effective.pipeline_type == PipelineType::Preproc) {
          out.modelpack_media_type = "video/x-raw";
          if (out.modelpack_format.empty())
            out.modelpack_format = "RGB";
          if (out.modelpack_input_depth <= 0)
            out.modelpack_input_depth = 3;
        } else {
          out.modelpack_media_type = "application/vnd.simaai.tensor";
          out.modelpack_format = "FP32";
          if (out.modelpack_input_depth <= 0)
            out.modelpack_input_depth = 3;
        }
        out.diagnostics.push_back("auto pre-route: selected external pre stage '" +
                                  pre_kind_name(capability.pre_kind) + "'");
      }
    } else {
      effective.include_preprocess_stage = false;
      out.diagnostics.push_back("auto pre-route: no external pre stage found");
    }

    if (effective.mla_tessellation && capability.mla_input_bf16 && !capability.has_external_pre) {
      effective.include_preprocess_stage = false;
      effective.pipeline_type = PipelineType::Preproc;
      const int inferred_w = capability.mla_input_dims.width;
      const int inferred_h = capability.mla_input_dims.height;
      const bool dims_unusable =
          inferred_w <= 0 || inferred_h <= 0 || inferred_w > 1024 || inferred_h > 1024;
      out.modelpack_max_width = dims_unusable ? 640 : inferred_w;
      out.modelpack_max_height = dims_unusable ? 640 : inferred_h;
      const int inferred_d =
          capability.mla_input_dims.depth > 0 ? capability.mla_input_dims.depth : 3;
      out.modelpack_input_depth = inferred_d;
      out.modelpack_max_depth = inferred_d;
      // BF16 MLA input is emitted as tensorized BF16 even when upstream MPK media looks video.
      out.modelpack_media_type = "application/vnd.simaai.tensor";
      out.modelpack_format = "BF16";
      out.diagnostics.push_back(
          "auto pre-route: BF16 MLA ingress with MLA tess -> skip external pre");
    }
  }

  if (post_auto) {
    effective.selected_post_kind = selected_post_kind_for_route(capability, false);
    effective.include_postprocess_stage = route_has_post_stage(capability) &&
                                          effective.selected_post_kind != PostRouteStageKind::None;
    if (!route_has_post_stage(capability)) {
      effective.include_postprocess_stage = false;
      out.diagnostics.push_back("auto post-route: no external post stage found");
    } else if (effective.selected_post_kind == PostRouteStageKind::None) {
      effective.include_postprocess_stage = false;
      out.diagnostics.push_back("auto post-route: matrix resolved to no compatible post stage");
    } else {
      out.diagnostics.push_back("auto post-route: using external post stage '" +
                                post_kind_name(effective.selected_post_kind) + "'");
      if (capability.post_kind != effective.selected_post_kind) {
        out.diagnostics.push_back("auto post-route: effective post stage '" +
                                  post_kind_name(effective.selected_post_kind) +
                                  "' differs from capability signal '" +
                                  post_kind_name(capability.post_kind) + "'");
      }
    }

  } else {
    if (requested_boxdecode) {
      effective.selected_post_kind = PostRouteStageKind::BoxDecode;
      effective.include_postprocess_stage = true;
      out.diagnostics.push_back(
          "explicit post-route: requested boxdecode -> terminal post stage 'boxdecode' "
          "(capability-independent)");
    } else {
      if (!route_has_post_stage(capability)) {
        out.ambiguous = true;
        out.ambiguity_reason =
            "route ambiguity: postprocess explicitly requested but no compatible post route exists";
        return out;
      }
      effective.selected_post_kind = selected_post_kind_for_route(capability, false);
      if (effective.selected_post_kind == PostRouteStageKind::None) {
        out.ambiguous = true;
        out.ambiguity_reason = "route ambiguity: explicit postprocess requested but no compatible "
                               "post stage was selected";
        return out;
      }
      effective.include_postprocess_stage = true;
      out.diagnostics.push_back("explicit post-route: selected external post stage '" +
                                post_kind_name(effective.selected_post_kind) + "'");
    }
  }

  effective.cast_symmetry_ok = !capability.needs.pre_cast || capability.needs.post_cast;
  if (!effective.cast_symmetry_ok) {
    out.ambiguous = true;
    out.ambiguity_reason = "route ambiguity: cast symmetry violated (pre cast requires post cast)";
    return out;
  }

  if (!effective.include_preprocess_stage && capability.mla_input_quantized && !mla_video_ingress) {
    out.modelpack_media_type = "application/vnd.simaai.tensor";
    const bool internal_quant_only = !capability.has_external_pre;
    out.modelpack_format = internal_quant_only
                               ? "FP32"
                               : normalize_tensor_format(capability.mla_input_dtype_raw, "INT8");
    if (capability.mla_input_dims.depth > 0)
      out.modelpack_input_depth = capability.mla_input_dims.depth;
    if (capability.mla_input_dims.width > 0)
      out.modelpack_max_width = capability.mla_input_dims.width;
    if (capability.mla_input_dims.height > 0)
      out.modelpack_max_height = capability.mla_input_dims.height;
    if (capability.mla_input_dims.depth > 0)
      out.modelpack_max_depth = capability.mla_input_dims.depth;
  }

  if (!effective.include_preprocess_stage && effective.mla_tessellation && mla_video_ingress &&
      !capability.mla_input_bf16) {
    out.modelpack_media_type = "video/x-raw";
    if (out.modelpack_format.empty())
      out.modelpack_format = "RGB";
    if (capability.mla_input_dims.depth > 0)
      out.modelpack_input_depth = capability.mla_input_dims.depth;
    if (capability.mla_input_dims.width > 0)
      out.modelpack_max_width = capability.mla_input_dims.width;
    if (capability.mla_input_dims.height > 0)
      out.modelpack_max_height = capability.mla_input_dims.height;
    if (capability.mla_input_dims.depth > 0)
      out.modelpack_max_depth = capability.mla_input_dims.depth;
    out.diagnostics.push_back(
        "auto pre-route: no external pre stage with MLA tess -> direct MLA video ingress");
  }

  effective.infer_only = effective.mla_tessellation && capability.mla_input_bf16 &&
                         !effective.include_preprocess_stage &&
                         !effective.include_postprocess_stage &&
                         out.modelpack_media_type == "application/vnd.simaai.tensor";
  if (effective.infer_only) {
    out.diagnostics.push_back("route selected: infer-only (BF16 + MLA tess + no external post)");
  }

  return out;
}

std::string route_capability_debug_string(const RouteCapability& capability) {
  std::ostringstream oss;
  oss << "RouteCapability{pre=" << pre_kind_name(capability.pre_kind)
      << ", post=" << post_kind_name(capability.post_kind) << ", mla_tess="
      << (capability.tessellation_location == TessellationLocation::MLA ? "1" : "0")
      << ", mla_in=" << capability.mla_input_dtype_raw
      << ", mla_out=" << capability.mla_output_dtype_raw
      << ", has_external_pre=" << (capability.has_external_pre ? "1" : "0")
      << ", has_external_post=" << (capability.has_external_post ? "1" : "0")
      << ", tess_needed=" << (capability.tess_needed ? "1" : "0")
      << ", quant_needed=" << (capability.quant_needed ? "1" : "0")
      << ", pre_cast=" << (capability.needs.pre_cast ? "1" : "0")
      << ", post_cast=" << (capability.needs.post_cast ? "1" : "0")
      << ", ordered_pre=" << ordered_route_chain_csv(capability.ordered_pre_ops)
      << ", ordered_post=" << ordered_route_chain_csv(capability.ordered_post_ops)
      << ", ingress_contracts=" << ingress_contracts_debug_string(capability.ingress_contracts)
      << ", cap_post_boxdecode=" << (capability.adapter_capabilities.has_post_boxdecode ? "1" : "0")
      << "}";
  return oss.str();
}

std::string route_selection_debug_string(const RouteSelection& selection) {
  std::ostringstream oss;
  oss << "RouteSelection{pipeline_type=" << static_cast<int>(selection.effective.pipeline_type)
      << ", include_pre=" << (selection.effective.include_preprocess_stage ? "1" : "0")
      << ", include_post=" << (selection.effective.include_postprocess_stage ? "1" : "0")
      << ", selected_post=" << post_kind_name(selection.effective.selected_post_kind)
      << ", requested_boxdecode=" << (selection.user_intent.requested_boxdecode ? "1" : "0")
      << ", pre_auto=" << (selection.user_intent.pre_auto ? "1" : "0")
      << ", post_auto=" << (selection.user_intent.post_auto ? "1" : "0")
      << ", cast_symmetry_ok=" << (selection.effective.cast_symmetry_ok ? "1" : "0")
      << ", infer_only=" << (selection.effective.infer_only ? "1" : "0")
      << ", media_type=" << selection.modelpack_media_type
      << ", format=" << selection.modelpack_format << "}";
  return oss.str();
}

} // namespace simaai::neat::internal

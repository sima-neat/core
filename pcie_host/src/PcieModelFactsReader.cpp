#include "PcieModelFactsReader.h"
#include "PcieModelFactsReaderInternal.h"

#include "model/internal/ModelArchiveLoader.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/sima/RouteGraph.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;

namespace simaai::neat::pcie::internal {
namespace {

class TempDir {
public:
  TempDir() {
    std::string tmpl = (fs::temp_directory_path() / "sima-neat-pcie-model-XXXXXX").string();
    std::vector<char> chars(tmpl.begin(), tmpl.end());
    chars.push_back('\0');
    char* made = ::mkdtemp(chars.data());
    if (!made) {
      throw std::runtime_error("failed to create temporary extraction directory");
    }
    path_ = made;
  }

  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }

  const std::string& path() const {
    return path_;
  }

private:
  std::string path_;
};

std::vector<std::int64_t>
best_shape(const simaai::neat::pipeline_internal::sima::MpkTensorContract& tensor) {
  if (!tensor.logical_shape.empty()) {
    return tensor.logical_shape;
  }
  return tensor.mpk_shape;
}

std::string best_dtype(const simaai::neat::pipeline_internal::sima::MpkTensorContract& tensor) {
  if (!tensor.logical_dtype.empty()) {
    return tensor.logical_dtype;
  }
  return tensor.dtype;
}

bool is_numbered_route_wrapper(const std::string& component, const std::string& prefix) {
  const std::size_t prefix_len = prefix.size();
  if (component.compare(0, prefix_len, prefix) != 0) {
    return false;
  }
  if (component.size() == prefix_len) {
    return true;
  }
  if (component[prefix_len] != '_') {
    return false;
  }
  for (std::size_t i = prefix_len + 1; i < component.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(component[i]))) {
      return false;
    }
  }
  return component.size() > prefix_len + 1;
}

bool is_public_route_wrapper(const std::string& component) {
  return is_numbered_route_wrapper(component, "cast") ||
         is_numbered_route_wrapper(component, "dequantize");
}

std::string strip_public_route_wrapper_prefix(std::string name) {
  const std::size_t slash = name.find('/');
  if (slash == std::string::npos || slash + 1 >= name.size()) {
    return name;
  }
  const std::string component = name.substr(0, slash);
  if (is_public_route_wrapper(component)) {
    return name.substr(slash + 1);
  }
  return name;
}

PcieTensorFact
convert_tensor(const simaai::neat::pipeline_internal::sima::MpkTensorContract& tensor,
               const bool normalize_public_name = false) {
  PcieTensorFact out;
  out.name = !tensor.name.empty() ? tensor.name : tensor.segment_name;
  if (normalize_public_name) {
    out.name = strip_public_route_wrapper_prefix(std::move(out.name));
  }
  out.dtype = best_dtype(tensor);
  out.shape = best_shape(tensor);
  out.size_bytes = tensor.size_bytes;
  out.tensor_index = tensor.tensor_index;
  out.physical_index = tensor.physical_index;
  out.byte_offset = tensor.byte_offset;
  if (tensor.input_range.size() == 2U) {
    out.input_range = std::make_pair(tensor.input_range[0], tensor.input_range[1]);
  }
  return out;
}

bool has_semantic_tensor_info(
    const simaai::neat::pipeline_internal::sima::MpkTensorContract& tensor) {
  return !best_shape(tensor).empty() || !best_dtype(tensor).empty() || tensor.size_bytes > 0U;
}

simaai::neat::pipeline_internal::sima::MpkTensorContract semantic_view_for_name(
    const simaai::neat::pipeline_internal::sima::MpkTensorContract& public_tensor,
    const simaai::neat::pipeline_internal::sima::MpkTensorContract& semantic_tensor,
    const int index) {
  auto out = public_tensor;
  out.tensor_index = index;
  out.name = !public_tensor.name.empty() ? public_tensor.name : semantic_tensor.name;
  out.segment_name = !public_tensor.segment_name.empty() ? public_tensor.segment_name
                                                         : semantic_tensor.segment_name;
  out.logical_shape = best_shape(semantic_tensor);
  out.logical_dtype = best_dtype(semantic_tensor);
  out.mpk_shape = semantic_tensor.mpk_shape;
  out.dtype = semantic_tensor.dtype;
  out.size_bytes =
      semantic_tensor.size_bytes > 0U ? semantic_tensor.size_bytes : public_tensor.size_bytes;
  out.physical_index = semantic_tensor.physical_index;
  out.byte_offset = semantic_tensor.byte_offset;
  return out;
}

std::string canonical_token(std::string text) {
  std::string out;
  out.reserve(text.size());
  for (const unsigned char c : text) {
    if (std::isalnum(c)) {
      out.push_back(static_cast<char>(std::tolower(c)));
    }
  }
  return out;
}

bool is_pass_through_stage(
    const simaai::neat::pipeline_internal::sima::MpkPluginIoContract& stage) {
  return canonical_token(stage.kernel) == "passthrough" ||
         canonical_token(stage.name) == "passthrough" ||
         canonical_token(stage.plugin_id) == "passthrough";
}

bool input_has_internal_producer(
    const simaai::neat::pipeline_internal::sima::MpkContract& contract,
    const std::size_t plugin_index, const std::size_t input_index,
    const simaai::neat::pipeline_internal::sima::MpkTensorContract& input) {
  for (const auto& edge : contract.edges) {
    if (edge.dst_plugin_index != plugin_index) {
      continue;
    }
    if (edge.dst_input_index >= 0) {
      if (edge.dst_input_index == static_cast<int>(input_index)) {
        return true;
      }
      continue;
    }
    if (!input.name.empty() && edge.tensor_name == input.name) {
      return true;
    }
  }
  return false;
}

std::optional<QuantParams>
public_quant(const std::optional<simaai::neat::pipeline_internal::sima::MpkQuantContract>& quant,
             const std::string& name) {
  if (!quant.has_value()) {
    return std::nullopt;
  }
  QuantParams out;
  out.axis = quant->axis;
  for (const auto scale : quant->scales) {
    if (scale == 0.0) {
      throw std::runtime_error("mla_only tensor '" + name + "' has a zero quantization scale");
    }
    out.scales.push_back(1.0f / static_cast<float>(scale));
  }
  for (const auto zero_point : quant->zero_points) {
    if (zero_point < std::numeric_limits<std::int32_t>::min() ||
        zero_point > std::numeric_limits<std::int32_t>::max()) {
      throw std::runtime_error("mla_only tensor '" + name + "' has an out-of-range zero point");
    }
    out.zero_points.push_back(static_cast<std::int32_t>(zero_point));
  }
  return out;
}

const simaai::neat::pipeline_internal::sima::MpkPluginIoContract&
dequantize_consumer(const simaai::neat::pipeline_internal::sima::MpkContract& contract,
                    const std::string& head_name) {
  const simaai::neat::pipeline_internal::sima::MpkPluginIoContract* consumer = nullptr;
  for (const auto& edge : contract.edges) {
    if (edge.tensor_name != head_name || edge.dst_plugin_index >= contract.plugins.size()) {
      continue;
    }
    const auto& stage = contract.plugins[edge.dst_plugin_index];
    if (canonical_token(stage.kernel).find("dequant") == std::string::npos ||
        stage.output_tensors.size() != 1U || consumer != nullptr) {
      throw std::runtime_error("mla_only output '" + head_name +
                               "' must feed exactly one dequantize stage");
    }
    consumer = &stage;
  }
  if (!consumer) {
    throw std::runtime_error("mla_only output '" + head_name + "' has no dequantize consumer");
  }
  return *consumer;
}

std::size_t dense_element_count(const std::vector<std::int64_t>& shape) {
  std::size_t elements = shape.empty() ? 0U : 1U;
  for (const auto dim : shape) {
    if (dim <= 0 || !simaai::neat::pipeline_internal::safe_mul(
                        elements, static_cast<std::size_t>(dim), &elements)) {
      return 0U;
    }
  }
  return elements;
}

simaai::neat::pipeline_internal::sima::RouteGraph
validate_mla_only_stages(const simaai::neat::pipeline_internal::sima::MpkContract& contract) {
  using simaai::neat::pipeline_internal::sima::RouteGraphKernelKind;
  auto graph = simaai::neat::pipeline_internal::sima::build_route_graph(contract);
  if (graph.mla_plugin_index < 0) {
    throw std::runtime_error("mla_only requires exactly one MLA stage in the MPK contract");
  }
  for (const auto& node : graph.nodes) {
    switch (node.kind) {
    case RouteGraphKernelKind::Mla:
    case RouteGraphKernelKind::Quant:
    case RouteGraphKernelKind::Unpack:
    case RouteGraphKernelKind::Slice:
    case RouteGraphKernelKind::Dequantize:
    case RouteGraphKernelKind::PassThrough:
      continue;
    default:
      throw std::runtime_error(
          "mla_only does not support stage '" + node.plugin_name + "' (" +
          simaai::neat::pipeline_internal::sima::route_graph_kernel_name(node.kind) + ")");
    }
  }
  return graph;
}

std::optional<std::size_t>
ingress_index(const std::vector<simaai::neat::pipeline_internal::sima::MpkTensorContract>& inputs,
              const std::string& name) {
  if (name.empty()) {
    return std::nullopt;
  }
  const auto it = std::find_if(inputs.begin(), inputs.end(),
                               [&](const auto& input) { return input.name == name; });
  return it != inputs.end() ? std::optional<std::size_t>(it - inputs.begin()) : std::nullopt;
}

// The MLA always takes a single input buffer. A model with several inputs quantizes each one
// separately and concatenates the results through an ifm pack transform, which mla_only drops
// along with the quantize stages, so the application submits the dense INT8 output of every
// quantize stage and the channel stages them contiguously - the same bytes the pack produced.
// Facts are published in pack order for that reason: it is the layout of the staged payload.
std::vector<std::size_t>
mla_only_quantize_stages(const simaai::neat::pipeline_internal::sima::RouteGraph& graph,
                         const std::size_t mla_index) {
  using simaai::neat::pipeline_internal::sima::RouteGraphKernelKind;
  // route_graph_incoming_edges is ordered by destination input slot, so pack order - the layout
  // of the staged payload - falls out of the traversal.
  std::vector<std::size_t> stages;
  for (const auto* edge :
       simaai::neat::pipeline_internal::sima::route_graph_incoming_edges(graph, mla_index)) {
    const auto* node =
        simaai::neat::pipeline_internal::sima::route_graph_node(graph, edge->src_plugin_index);
    if (node == nullptr) {
      throw std::runtime_error("mla_only cannot resolve the stage feeding the MLA");
    }
    if (node->kind == RouteGraphKernelKind::Quant) {
      stages.push_back(edge->src_plugin_index);
      continue;
    }
    if (node->kind == RouteGraphKernelKind::PassThrough) {
      for (const auto* packed : simaai::neat::pipeline_internal::sima::route_graph_incoming_edges(
               graph, edge->src_plugin_index)) {
        const auto* producer = simaai::neat::pipeline_internal::sima::route_graph_node(
            graph, packed->src_plugin_index);
        if (producer == nullptr || producer->kind != RouteGraphKernelKind::Quant) {
          throw std::runtime_error(
              "mla_only input '" + packed->tensor_name + "' is produced by stage '" +
              packed->src_plugin +
              "', not a quantize stage; hybrid host/card quantization is out of scope");
        }
        stages.push_back(packed->src_plugin_index);
      }
      continue;
    }
    throw std::runtime_error("mla_only MLA input is produced by stage '" + edge->src_plugin +
                             "', which is neither a quantize nor a pack stage; hybrid host/card "
                             "quantization is out of scope");
  }
  return stages;
}

std::vector<PcieTensorFact>
mla_only_input_facts(const simaai::neat::pipeline_internal::sima::MpkContract& contract,
                     const simaai::neat::pipeline_internal::sima::RouteGraph& graph,
                     const std::size_t mla_index) {
  const auto public_inputs = detail::application_input_contracts(contract);
  const auto stages = mla_only_quantize_stages(graph, mla_index);
  if (stages.empty() || public_inputs.empty()) {
    throw std::runtime_error("mla_only requires at least one model input");
  }
  if (stages.size() != public_inputs.size()) {
    throw std::runtime_error("mla_only requires every model input to reach the MLA through its own "
                             "quantize stage: the model has " +
                             std::to_string(public_inputs.size()) + " input(s) but " +
                             std::to_string(stages.size()) +
                             " quantize stage(s); hybrid host/card quantization is out of scope");
  }

  std::vector<PcieTensorFact> facts;
  facts.reserve(stages.size());
  std::vector<bool> claimed(public_inputs.size(), false);
  for (const std::size_t stage_index : stages) {
    const auto& stage = contract.plugins[stage_index];
    if (stage.output_tensors.size() != 1U) {
      throw std::runtime_error("mla_only quantize stage '" + stage.name +
                               "' must produce exactly one tensor");
    }
    std::optional<std::size_t> ingress;
    for (const auto& candidate : stage.input_tensors) {
      const auto found = ingress_index(public_inputs, candidate.name);
      if (!found.has_value()) {
        continue;
      }
      if (ingress.has_value()) {
        throw std::runtime_error("mla_only quantize stage '" + stage.name +
                                 "' consumes more than one model input");
      }
      ingress = found;
    }
    if (!ingress.has_value()) {
      throw std::runtime_error("mla_only quantize stage '" + stage.name +
                               "' is not fed by a model input; hybrid host/card quantization is "
                               "out of scope");
    }
    if (claimed[*ingress]) {
      throw std::runtime_error("mla_only model input '" + public_inputs[*ingress].name +
                               "' feeds more than one quantize stage");
    }
    claimed[*ingress] = true;

    auto input = stage.output_tensors.front();
    input.name = public_inputs[*ingress].name;
    const std::string dtype = best_dtype(input);
    if (canonical_token(dtype) != "int8") {
      throw std::runtime_error("mla_only input '" + input.name + "' must be INT8, got '" + dtype +
                               "'");
    }
    const std::size_t elements = dense_element_count(best_shape(input));
    if (elements == 0U || elements != input.size_bytes) {
      throw std::runtime_error("mla_only input '" + input.name +
                               "' is not a dense INT8 tensor: shape does not cover " +
                               std::to_string(input.size_bytes) + " bytes");
    }
    facts.push_back(convert_tensor(input));
    facts.back().quant = public_quant(stage.quant, input.name);
    facts.back().input_range = convert_tensor(public_inputs[*ingress]).input_range;
  }
  return facts;
}

std::vector<simaai::neat::pipeline_internal::sima::MpkTensorContract>
application_output_contracts(const simaai::neat::pipeline_internal::sima::MpkContract& contract);

void add_mla_only_outputs(const simaai::neat::pipeline_internal::sima::MpkContract& contract,
                          PcieModelFacts* facts) {
  const auto carrier =
      simaai::neat::pipeline_internal::sima::get_mla_boundary_physical_outputs_contract(contract);
  if (carrier.size() != 1U || carrier.front().size_bytes == 0U) {
    throw std::runtime_error("mla_only supports exactly one MLA output carrier");
  }
  const std::size_t raw = carrier.front().size_bytes;
  const auto logical =
      simaai::neat::pipeline_internal::sima::get_mla_logical_outputs_contract(contract);
  const auto published =
      simaai::neat::pipeline_internal::sima::get_mla_published_outputs_contract(contract);
  if (logical.empty() || logical.size() != published.size()) {
    throw std::runtime_error("MPK contract does not expose consistent MLA output heads");
  }

  std::vector<std::pair<std::size_t, std::size_t>> spans;
  spans.reserve(logical.size());
  for (std::size_t i = 0; i < logical.size(); ++i) {
    const auto& head = logical[i];
    auto fact = convert_tensor(head);
    if (published[i].materialization_kind ==
        simaai::neat::pipeline_internal::sima::MpkTensorMaterializationKind::Bf16LaneSplitRepack) {
      throw std::runtime_error("mla_only output '" + fact.name +
                               "' needs a lane-split repack the host does not perform");
    }
    if (canonical_token(fact.dtype) != "int8") {
      throw std::runtime_error("mla_only output '" + fact.name + "' must be INT8, got '" +
                               fact.dtype + "'");
    }
    if (dense_element_count(fact.shape) != fact.size_bytes ||
        head.stride_bytes.size() != fact.shape.size() || head.stride_bytes.back() != 1 ||
        head.source_byte_offset < 0) {
      throw std::runtime_error("mla_only output '" + fact.name + "' has no usable geometry");
    }
    fact.transport_strides_bytes = head.stride_bytes;
    fact.payload_offset = static_cast<std::size_t>(head.source_byte_offset);
    std::size_t end = 0U;
    if (!simaai::neat::pipeline_internal::safe_mul(
            static_cast<std::size_t>(fact.shape.front()),
            static_cast<std::size_t>(head.stride_bytes.front()), &fact.transport_size_bytes) ||
        !simaai::neat::pipeline_internal::safe_add(fact.payload_offset, fact.transport_size_bytes,
                                                   &end) ||
        end > raw) {
      throw std::runtime_error("mla_only output '" + fact.name +
                               "' lies outside the MLA output carrier");
    }
    const auto& dequantize = dequantize_consumer(contract, head.name);
    if (best_shape(dequantize.output_tensors.front()) != fact.shape) {
      throw std::runtime_error("mla_only output '" + fact.name +
                               "' does not match the shape of its dequantized output");
    }
    fact.name = strip_public_route_wrapper_prefix(dequantize.output_tensors.front().name);
    fact.quant = public_quant(dequantize.quant, fact.name);
    fact.dense_offset = facts->dense_output_bytes;
    if (!simaai::neat::pipeline_internal::safe_add(facts->dense_output_bytes, fact.size_bytes,
                                                   &facts->dense_output_bytes)) {
      throw std::runtime_error("mla_only dense output size overflows");
    }
    spans.emplace_back(fact.payload_offset, fact.transport_size_bytes);
    facts->outputs.push_back(std::move(fact));
  }

  std::sort(spans.begin(), spans.end());
  std::size_t next = 0U;
  for (const auto& [offset, size] : spans) {
    if (offset != next) {
      throw std::runtime_error("mla_only output heads do not tile the MLA output carrier");
    }
    next += size;
  }
  if (next != raw) {
    throw std::runtime_error("mla_only output heads do not tile the MLA output carrier");
  }
  if (facts->outputs.size() != application_output_contracts(contract).size()) {
    throw std::runtime_error("mla_only heads do not cover every model output");
  }
  facts->packed_output_bytes = raw;
}

} // namespace

namespace detail {

std::vector<simaai::neat::pipeline_internal::sima::MpkTensorContract>
application_input_contracts(const simaai::neat::pipeline_internal::sima::MpkContract& contract) {
  std::vector<simaai::neat::pipeline_internal::sima::MpkTensorContract> inputs;
  inputs.reserve(contract.ingress_tensors.size());
  const auto ordered = simaai::neat::pipeline_internal::sima::plugins_in_execution_order(contract);

  for (std::size_t i = 0; i < contract.ingress_tensors.size(); ++i) {
    const auto& ingress = contract.ingress_tensors[i];
    const simaai::neat::pipeline_internal::sima::MpkTensorContract* semantic = nullptr;
    for (const std::size_t plugin_index : ordered) {
      if (plugin_index >= contract.plugins.size()) {
        continue;
      }
      const auto& stage = contract.plugins[plugin_index];
      for (std::size_t input_index = 0; input_index < stage.input_tensors.size(); ++input_index) {
        const auto& candidate = stage.input_tensors[input_index];
        if (!ingress.name.empty() && candidate.name == ingress.name &&
            has_semantic_tensor_info(candidate) &&
            !input_has_internal_producer(contract, plugin_index, input_index, candidate)) {
          if (semantic && (best_shape(*semantic) != best_shape(candidate) ||
                           best_dtype(*semantic) != best_dtype(candidate) ||
                           semantic->size_bytes != candidate.size_bytes)) {
            throw std::runtime_error("MPK ingress '" + ingress.name +
                                     "' has ambiguous root input contracts");
          }
          semantic = &candidate;
        }
      }
    }
    inputs.push_back(semantic ? semantic_view_for_name(ingress, *semantic, static_cast<int>(i))
                              : ingress);
  }

  return inputs;
}

void validate_supported_input_dtype(
    const simaai::neat::pipeline_internal::sima::MpkTensorContract& input) {
  const std::string dtype = best_dtype(input);
  const std::string token = canonical_token(dtype);
  if (token == "uint16" || token == "evxxuint16" || token == "ev74uint16" || token == "fp64" ||
      token == "float64" || token == "evxxfp64" || token == "evxxfloat64" || token == "ev74fp64" ||
      token == "ev74float64") {
    const std::string name = !input.name.empty() ? input.name : input.segment_name;
    throw std::runtime_error("PCIe model input '" + name + "' uses unsupported dtype '" + dtype +
                             "'; supported input dtypes are UINT8, INT8, INT16, INT32, BF16, "
                             "and FP32");
  }
}

PcieModelFacts
read_mla_only_facts(const simaai::neat::pipeline_internal::sima::MpkContract& contract) {
  const auto graph = validate_mla_only_stages(contract);
  const auto mla_index = static_cast<std::size_t>(graph.mla_plugin_index);

  PcieModelFacts facts;
  for (auto& input : mla_only_input_facts(contract, graph, mla_index)) {
    facts.packed_input_bytes += input.size_bytes;
    facts.inputs.push_back(std::move(input));
  }
  if (facts.inputs.size() > 1U) {
    const auto& mla_inputs = contract.plugins[mla_index].input_tensors;
    if (mla_inputs.size() != 1U || mla_inputs.front().size_bytes != facts.packed_input_bytes) {
      throw std::runtime_error("mla_only inputs do not tile the MLA packed ingress");
    }
    facts.packed_input = convert_tensor(mla_inputs.front());
    facts.packed_input->dtype = facts.inputs.front().dtype;
  }
  add_mla_only_outputs(contract, &facts);
  return facts;
}

} // namespace detail

namespace {

std::vector<simaai::neat::pipeline_internal::sima::MpkTensorContract>
application_output_contracts(const simaai::neat::pipeline_internal::sima::MpkContract& contract) {
  const auto ordered = simaai::neat::pipeline_internal::sima::plugins_in_execution_order(contract);
  if (ordered.empty()) {
    throw std::runtime_error("MPK contract has no executable plugin topology");
  }

  std::vector<bool> has_outgoing(contract.plugins.size(), false);
  for (const auto& edge : contract.edges) {
    if (edge.src_plugin_index < has_outgoing.size() &&
        edge.dst_plugin_index < contract.plugins.size() &&
        edge.src_plugin_index != edge.dst_plugin_index) {
      has_outgoing[edge.src_plugin_index] = true;
    }
  }

  std::vector<std::size_t> terminals;
  for (const std::size_t index : ordered) {
    if (index >= contract.plugins.size() || has_outgoing[index]) {
      continue;
    }
    const auto& stage = contract.plugins[index];
    if (!stage.output_tensors.empty() ||
        (is_pass_through_stage(stage) && !stage.input_tensors.empty())) {
      terminals.push_back(index);
    }
  }
  if (terminals.size() != 1U) {
    throw std::runtime_error("PCIe co-processing requires exactly one terminal MPK stage; found " +
                             std::to_string(terminals.size()));
  }

  const auto& terminal = contract.plugins[terminals.front()];
  if (is_pass_through_stage(terminal) && !terminal.input_tensors.empty()) {
    return terminal.input_tensors;
  }
  return terminal.output_tensors;
}

void finalize_output_layout(PcieModelFacts* facts) {
  if (!facts || facts->outputs.empty()) {
    return;
  }

  std::vector<std::size_t> physical_spans(facts->outputs.size(), 0U);
  std::vector<bool> physical_seen(facts->outputs.size(), false);
  int max_physical_index = -1;
  for (std::size_t i = 0; i < facts->outputs.size(); ++i) {
    auto& output = facts->outputs[i];
    if (output.physical_index < 0) {
      output.physical_index = static_cast<int>(i);
    }
    if (output.physical_index < 0 ||
        static_cast<std::size_t>(output.physical_index) >= facts->outputs.size()) {
      throw std::runtime_error("PCIe output '" + output.name +
                               "' has an unsupported physical index " +
                               std::to_string(output.physical_index));
    }
    if (output.byte_offset < 0) {
      throw std::runtime_error("PCIe output '" + output.name + "' has a negative byte offset");
    }
    const auto local_offset = static_cast<std::size_t>(output.byte_offset);
    if (output.size_bytes > std::numeric_limits<std::size_t>::max() - local_offset) {
      throw std::runtime_error("PCIe output '" + output.name + "' byte span overflows");
    }
    const std::size_t physical_index = static_cast<std::size_t>(output.physical_index);
    physical_spans[physical_index] =
        std::max(physical_spans[physical_index], local_offset + output.size_bytes);
    physical_seen[physical_index] = true;
    max_physical_index = std::max(max_physical_index, output.physical_index);
  }

  for (int index = 0; index <= max_physical_index; ++index) {
    if (!physical_seen[static_cast<std::size_t>(index)]) {
      throw std::runtime_error("PCIe output physical indices must be dense from zero");
    }
  }

  std::vector<std::size_t> physical_bases(facts->outputs.size(), 0U);
  std::size_t packed_bytes = 0U;
  for (int index = 0; index <= max_physical_index; ++index) {
    const std::size_t physical_index = static_cast<std::size_t>(index);
    physical_bases[physical_index] = packed_bytes;
    if (physical_spans[physical_index] > std::numeric_limits<std::size_t>::max() - packed_bytes) {
      throw std::runtime_error("PCIe packed output size overflows");
    }
    packed_bytes += physical_spans[physical_index];
  }

  for (auto& output : facts->outputs) {
    const std::size_t physical_index = static_cast<std::size_t>(output.physical_index);
    output.payload_offset =
        physical_bases[physical_index] + static_cast<std::size_t>(output.byte_offset);
  }
  facts->packed_output_bytes = packed_bytes;
}

bool graph_has_preprocess(const simaai::neat::pipeline_internal::sima::MpkContract& contract) {
  for (const auto& node : contract.graph.nodes) {
    if (node.kind == simaai::neat::pipeline_internal::sima::MpkGraphNodeKind::FusedPreproc ||
        node.requirements.preproc || node.canonical_op == "preproc" ||
        node.kernel.find("preproc") != std::string::npos ||
        node.kernel.find("processcvu") != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool graph_has_boxdecode(const simaai::neat::pipeline_internal::sima::MpkContract& contract) {
  for (const auto& node : contract.graph.nodes) {
    if (node.kind == simaai::neat::pipeline_internal::sima::MpkGraphNodeKind::FusedBoxDecode ||
        node.requirements.boxdecode || node.canonical_op == "boxdecode" ||
        node.kernel.find("boxdecode") != std::string::npos) {
      return true;
    }
  }
  return false;
}

} // namespace

PcieModelFacts read_model_facts(const std::string& model_path, const ModelOptions& options) {
  TempDir temp;

  simaai::neat::internal::ModelArchiveLoaderOptions loader_options;
  loader_options.check_output_free_space = false;
  // Runtime model packs can contain provenance/build helper files such as
  // archived_compile_script.*.py. Match ModelPack's runtime extraction policy:
  // keep the loader defaults strict for tests, but tolerate these extras here.
  loader_options.reject_unsupported_file_types = false;
  const auto extracted =
      simaai::neat::internal::ModelArchiveLoader::extract(model_path, temp.path(), loader_options);

  std::string error;
  auto contract = simaai::neat::pipeline_internal::sima::load_mpk_contract_from_pack_root(
      extracted.package_root, &error);
  if (!contract.has_value()) {
    throw std::runtime_error("failed to read MPK contract: " + error);
  }
  if (options.mla_only) {
    return detail::read_mla_only_facts(*contract);
  }

  const auto public_inputs = detail::application_input_contracts(*contract);
  const auto public_outputs = application_output_contracts(*contract);

  PcieModelFacts facts;
  facts.has_preprocess = graph_has_preprocess(*contract);
  facts.has_boxdecode = graph_has_boxdecode(*contract);

  for (const auto& input : public_inputs) {
    detail::validate_supported_input_dtype(input);
    auto converted = convert_tensor(input);
    facts.packed_input_bytes += converted.size_bytes;
    facts.inputs.push_back(std::move(converted));
  }
  for (const auto& output : public_outputs) {
    auto converted = convert_tensor(output, true);
    facts.outputs.push_back(std::move(converted));
  }
  finalize_output_layout(&facts);

  if (facts.inputs.empty()) {
    throw std::runtime_error("MPK contract does not expose MLA input tensors");
  }
  if (facts.outputs.empty()) {
    throw std::runtime_error("MPK contract does not expose MLA output tensors");
  }
  return facts;
}

ModelInfo to_public_model_info(const PcieModelFacts& facts) {
  ModelInfo out;
  out.inputs.reserve(facts.inputs.size());
  out.outputs.reserve(facts.outputs.size());
  for (const auto& input : facts.inputs) {
    out.inputs.push_back(TensorInfo{.name = input.name,
                                    .dtype = input.dtype,
                                    .shape = input.shape,
                                    .size_bytes = input.size_bytes,
                                    .quant = input.quant,
                                    .input_range = input.input_range});
  }
  for (const auto& output : facts.outputs) {
    out.outputs.push_back(TensorInfo{.name = output.name,
                                     .dtype = output.dtype,
                                     .shape = output.shape,
                                     .size_bytes = output.size_bytes,
                                     .quant = output.quant});
  }
  return out;
}

} // namespace simaai::neat::pcie::internal

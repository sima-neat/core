#include "PcieModelFactsReader.h"
#include "PcieModelFactsReaderInternal.h"

#include "model/internal/ModelArchiveLoader.h"
#include "pipeline/internal/TensorMath.h"

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

// The manifest stores the reciprocal scale; the public contract is per-tensor only.
std::optional<QuantParams>
quant_from_mpk(const std::optional<simaai::neat::pipeline_internal::sima::MpkQuantContract>& quant,
               const std::string& name) {
  if (!quant.has_value()) {
    return std::nullopt;
  }
  if (quant->scales.size() != 1U || quant->zero_points.size() != 1U) {
    throw std::runtime_error("mla_only tensor '" + name +
                             "' has per-channel quantization parameters; only one scale and zero "
                             "point per tensor is supported");
  }
  return QuantParams{.scale = 1.0f / static_cast<float>(quant->scales.front()),
                     .zero_point = static_cast<std::int32_t>(quant->zero_points.front())};
}

const simaai::neat::pipeline_internal::sima::MpkPluginIoContract&
public_consumer_for_head(const simaai::neat::pipeline_internal::sima::MpkContract& contract,
                         const std::string& head_name) {
  for (const auto& edge : contract.edges) {
    if (edge.tensor_name != head_name || edge.dst_plugin_index >= contract.plugins.size()) {
      continue;
    }
    const auto& stage = contract.plugins[edge.dst_plugin_index];
    const std::string kernel = canonical_token(stage.kernel);
    if ((kernel.find("dequant") != std::string::npos || kernel.find("cast") != std::string::npos) &&
        !stage.output_tensors.empty()) {
      return stage;
    }
  }
  throw std::runtime_error("mla_only output '" + head_name +
                           "' has no dequantize or cast consumer");
}

// Element size of the activation dtypes the route publishes; 0 for anything else.
std::size_t mla_only_dtype_bytes(const std::string& dtype) {
  const std::string token = canonical_token(dtype);
  return token == "int8" ? 1U : (token == "bf16" || token == "bfloat16") ? 2U : 0U;
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

std::vector<PcieTensorFact>
mla_only_input_facts(const simaai::neat::pipeline_internal::sima::MpkContract& contract,
                     const simaai::neat::pipeline_internal::sima::MpkPluginIoContract& mla,
                     bool* packed) {
  const auto public_inputs = detail::application_input_contracts(contract);
  const auto boundary =
      simaai::neat::pipeline_internal::sima::get_mla_boundary_logical_inputs_contract(contract);
  if (boundary.size() != public_inputs.size()) {
    throw std::runtime_error("mla_only requires every model input to reach the MLA through its own "
                             "quantize stage: the model has " +
                             std::to_string(public_inputs.size()) + " input(s) but " +
                             std::to_string(boundary.size()) +
                             " quantize stage(s); hybrid host/card quantization is out of scope");
  }
  *packed = boundary.size() != 1U || mla.input_tensors.size() != 1U ||
            mla.input_tensors.front().name != boundary.front().name;

  std::vector<PcieTensorFact> facts;
  for (const auto& tensor : boundary) {
    const auto producer =
        std::find_if(contract.plugins.begin(), contract.plugins.end(), [&](const auto& stage) {
          return std::any_of(stage.output_tensors.begin(), stage.output_tensors.end(),
                             [&](const auto& output) { return output.name == tensor.name; });
        });
    if (producer == contract.plugins.end()) {
      throw std::runtime_error("mla_only input '" + tensor.name + "' has no producing stage");
    }
    const std::string kernel = canonical_token(producer->kernel);
    if (kernel.find("quant") == std::string::npos && kernel.find("cast") == std::string::npos) {
      throw std::runtime_error(
          "mla_only input '" + tensor.name + "' is produced by stage '" + producer->name +
          "', not a quantize or cast stage; hybrid host/card quantization is out of scope");
    }
    const auto ingress =
        std::find_if(public_inputs.begin(), public_inputs.end(), [&](const auto& input) {
          return std::any_of(producer->input_tensors.begin(), producer->input_tensors.end(),
                             [&](const auto& candidate) { return candidate.name == input.name; });
        });
    if (ingress == public_inputs.end()) {
      throw std::runtime_error("mla_only quantize stage '" + producer->name +
                               "' is not fed by a model input; hybrid host/card quantization is "
                               "out of scope");
    }

    auto input = tensor;
    input.name = ingress->name;
    const std::string dtype = best_dtype(input);
    const std::size_t elem = mla_only_dtype_bytes(dtype);
    if (elem == 0U) {
      throw std::runtime_error("mla_only input '" + input.name + "' must be INT8 or BF16, got '" +
                               dtype + "'");
    }
    if (elem == 1U && !producer->quant.has_value()) {
      throw std::runtime_error("mla_only INT8 input '" + input.name +
                               "' has no quantization parameters");
    }
    const std::size_t elements = dense_element_count(best_shape(input));
    if (elements == 0U || elements * elem != input.size_bytes) {
      throw std::runtime_error("mla_only input '" + input.name + "' is not a dense " + dtype +
                               " tensor: shape does not cover " + std::to_string(input.size_bytes) +
                               " bytes");
    }
    facts.push_back(convert_tensor(input));
    facts.back().quant = quant_from_mpk(producer->quant, input.name);
  }
  return facts;
}

std::vector<simaai::neat::pipeline_internal::sima::MpkTensorContract>
application_output_contracts(const simaai::neat::pipeline_internal::sima::MpkContract& contract);

void add_mla_only_outputs(const simaai::neat::pipeline_internal::sima::MpkContract& contract,
                          PcieModelFacts* facts) {
  using simaai::neat::pipeline_internal::sima::MpkTensorMaterializationKind;
  const auto carrier =
      simaai::neat::pipeline_internal::sima::get_mla_boundary_physical_outputs_contract(contract);
  if (carrier.size() != 1U) {
    throw std::runtime_error("mla_only supports exactly one MLA output carrier");
  }
  const auto logical =
      simaai::neat::pipeline_internal::sima::get_mla_logical_outputs_contract(contract);
  const auto published =
      simaai::neat::pipeline_internal::sima::get_mla_published_outputs_contract(contract);
  if (logical.size() != published.size()) {
    throw std::runtime_error("MPK contract does not expose consistent MLA output heads");
  }

  bool padded = false;
  for (std::size_t i = 0; i < logical.size(); ++i) {
    const auto& head = logical[i];
    auto fact = convert_tensor(head);
    if (published[i].materialization_kind == MpkTensorMaterializationKind::Bf16LaneSplitRepack) {
      throw std::runtime_error("mla_only output '" + fact.name +
                               "' needs a lane-split repack the host does not perform");
    }
    const std::size_t elem = mla_only_dtype_bytes(fact.dtype);
    if (elem == 0U) {
      throw std::runtime_error("mla_only output '" + fact.name + "' must be INT8 or BF16, got '" +
                               fact.dtype + "'");
    }
    const std::size_t elements = dense_element_count(fact.shape);
    if (elements == 0U || elements * elem != fact.size_bytes ||
        head.stride_bytes.size() != fact.shape.size() || head.source_byte_offset < 0) {
      throw std::runtime_error("mla_only output '" + fact.name + "' has no usable geometry");
    }
    fact.transport_strides_bytes = head.stride_bytes;
    fact.payload_offset = static_cast<std::size_t>(head.source_byte_offset);
    padded =
        padded || head.stride_bytes !=
                      simaai::neat::pipeline_internal::contiguous_strides_bytes(fact.shape, elem);
    const auto& consumer = public_consumer_for_head(contract, head.name);
    fact.name = strip_public_route_wrapper_prefix(consumer.output_tensors.front().name);
    if (elem == 1U && !consumer.quant.has_value()) {
      throw std::runtime_error("mla_only INT8 output '" + fact.name +
                               "' has no quantization parameters");
    }
    fact.quant = quant_from_mpk(consumer.quant, fact.name);
    fact.dense_offset = facts->dense_output_bytes;
    if (!simaai::neat::pipeline_internal::safe_add(facts->dense_output_bytes, fact.size_bytes,
                                                   &facts->dense_output_bytes)) {
      throw std::runtime_error("mla_only dense output size overflows");
    }
    facts->outputs.push_back(std::move(fact));
  }

  std::vector<PcieTensorFact> ordered;
  for (const auto& output : application_output_contracts(contract)) {
    const std::string name = strip_public_route_wrapper_prefix(output.name);
    const auto it = std::find_if(facts->outputs.begin(), facts->outputs.end(),
                                 [&](const PcieTensorFact& fact) { return fact.name == name; });
    if (it == facts->outputs.end()) {
      throw std::runtime_error("mla_only has no head for model output '" + name + "'");
    }
    ordered.push_back(std::move(*it));
  }
  facts->outputs = std::move(ordered);
  if (!padded) {
    facts->dense_output_bytes = 0;
  }
  facts->packed_output_bytes = carrier.front().size_bytes;
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
  const auto* mla = simaai::neat::pipeline_internal::sima::get_mla_stage_io_contract(contract);
  if (mla == nullptr) {
    throw std::runtime_error("mla_only requires exactly one MLA stage in the MPK contract");
  }
  for (const auto& stage : contract.plugins) {
    const std::string kernel = canonical_token(stage.kernel);
    const bool supported =
        &stage == mla || is_pass_through_stage(stage) || kernel.find("pack") != std::string::npos ||
        kernel.find("slice") != std::string::npos || kernel.find("dequant") != std::string::npos ||
        ((kernel.find("quant") != std::string::npos || kernel.find("cast") != std::string::npos) &&
         kernel.find("tess") == std::string::npos);
    if (!supported) {
      throw std::runtime_error("mla_only does not support stage '" + stage.name + "' (" +
                               stage.kernel + ")");
    }
  }

  PcieModelFacts facts;
  bool packed = false;
  for (auto& input : mla_only_input_facts(contract, *mla, &packed)) {
    facts.packed_input_bytes += input.size_bytes;
    facts.inputs.push_back(std::move(input));
  }
  if (packed) {
    if (mla->input_tensors.size() != 1U ||
        mla->input_tensors.front().size_bytes != facts.packed_input_bytes) {
      throw std::runtime_error("mla_only inputs do not tile the MLA packed ingress");
    }
    facts.packed_input = convert_tensor(mla->input_tensors.front());
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
                                    .quant = input.quant});
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

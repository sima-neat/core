#include "PcieModelFactsReader.h"

#include "model/internal/ModelArchiveLoader.h"
#include "pipeline/internal/sima/MpkContract.h"

#include <cctype>
#include <cstdlib>
#include <filesystem>
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

std::vector<simaai::neat::pipeline_internal::sima::MpkTensorContract>
application_input_contracts(const simaai::neat::pipeline_internal::sima::MpkContract& contract) {
  std::vector<simaai::neat::pipeline_internal::sima::MpkTensorContract> inputs;
  inputs.reserve(contract.ingress_tensors.size());

  for (std::size_t i = 0; i < contract.ingress_tensors.size(); ++i) {
    const auto& ingress = contract.ingress_tensors[i];
    const simaai::neat::pipeline_internal::sima::MpkTensorContract* semantic = nullptr;
    for (const auto& stage : contract.plugins) {
      for (const auto& candidate : stage.input_tensors) {
        if (!ingress.name.empty() && candidate.name == ingress.name &&
            has_semantic_tensor_info(candidate)) {
          semantic = &candidate;
          break;
        }
      }
      if (semantic) {
        break;
      }
    }
    inputs.push_back(semantic ? semantic_view_for_name(ingress, *semantic, static_cast<int>(i))
                              : ingress);
  }

  return inputs;
}

std::vector<simaai::neat::pipeline_internal::sima::MpkTensorContract>
application_output_contracts(const simaai::neat::pipeline_internal::sima::MpkContract& contract) {
  for (auto it = contract.plugins.rbegin(); it != contract.plugins.rend(); ++it) {
    const auto& stage = *it;
    if (is_pass_through_stage(stage) && !stage.input_tensors.empty()) {
      return stage.input_tensors;
    }
    if (!stage.output_tensors.empty()) {
      return stage.output_tensors;
    }
  }
  return {};
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

PcieModelFacts read_model_facts(const std::string& model_path) {
  TempDir temp;

  simaai::neat::internal::ModelArchiveLoaderOptions loader_options;
  loader_options.check_output_free_space = false;
  // Runtime model packs can contain provenance/build helper files such as
  // archived_compile_script.*.py. Match ModelPack's runtime extraction policy:
  // keep the loader defaults strict for tests, but tolerate these extras here.
  loader_options.reject_unsupported_file_types = false;
  loader_options.require_pipeline_sequence = false;
  const auto extracted =
      simaai::neat::internal::ModelArchiveLoader::extract(model_path, temp.path(), loader_options);

  std::string error;
  auto contract = simaai::neat::pipeline_internal::sima::load_mpk_contract_from_pack_root(
      extracted.package_root, &error);
  if (!contract.has_value()) {
    throw std::runtime_error("failed to read MPK contract: " + error);
  }

  const auto public_inputs = application_input_contracts(*contract);
  const auto public_outputs = application_output_contracts(*contract);

  PcieModelFacts facts;
  facts.has_preprocess = graph_has_preprocess(*contract);
  facts.has_boxdecode = graph_has_boxdecode(*contract);

  for (const auto& input : public_inputs) {
    auto converted = convert_tensor(input);
    facts.packed_input_bytes += converted.size_bytes;
    facts.inputs.push_back(std::move(converted));
  }
  for (const auto& output : public_outputs) {
    auto converted = convert_tensor(output, true);
    facts.packed_output_bytes += converted.size_bytes;
    facts.outputs.push_back(std::move(converted));
  }

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
  out.has_preprocess = facts.has_preprocess;
  out.has_boxdecode = facts.has_boxdecode;
  out.inputs.reserve(facts.inputs.size());
  out.outputs.reserve(facts.outputs.size());
  for (const auto& input : facts.inputs) {
    out.inputs.push_back(TensorInfo{.name = input.name,
                                    .dtype = input.dtype,
                                    .shape = input.shape,
                                    .size_bytes = input.size_bytes});
  }
  for (const auto& output : facts.outputs) {
    out.outputs.push_back(TensorInfo{.name = output.name,
                                     .dtype = output.dtype,
                                     .shape = output.shape,
                                     .size_bytes = output.size_bytes});
  }
  return out;
}

} // namespace simaai::neat::pcie::internal

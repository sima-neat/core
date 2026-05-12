#include "nodes/sima/CastTess.h"
#include "nodes/sima/NodeConfigHelpers.h"

#include "gst/GstHelpers.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace simaai::neat {
using json = nlohmann::json;

struct CastTess::ConfigHolder {
  json config;
  bool has_config = false;
  std::optional<CompiledProcessCvuContract> compiled_contract;
};

namespace {
std::shared_ptr<CastTess::ConfigHolder> init_config_holder(const CastTessOptions& opt,
                                                           std::string& config_path_out) {
  auto holder = std::make_shared<CastTess::ConfigHolder>();
  if (opt.compiled_contract) {
    holder->compiled_contract = *opt.compiled_contract;
    config_path_out.clear();
    return holder;
  }
  if (opt.config_json.has_value()) {
    holder->config = *opt.config_json;
    holder->has_config = true;
    config_path_out.clear();
  } else if (!opt.config_path.empty()) {
    config_path_out = opt.config_path;
    holder->config = node_helpers::load_json_file(config_path_out, "CastTess");
    holder->has_config = true;
  }
  return holder;
}
} // namespace

CastTessOptions::CastTessOptions(const simaai::neat::Model& model) {
  *this = simaai::neat::internal::ModelAccess::build_casttess_stage_options(model, false);
}

CastTess::CastTess(CastTessOptions opt) : opt_(std::move(opt)) {
  if (opt_.num_buffers_locked) {
    if (opt_.num_buffers != opt_.num_buffers_model) {
      throw std::runtime_error("CastTess: num_buffers override is not allowed; must match model.");
    }
    if (opt_.num_buffers != 4 && opt_.num_buffers != 1) {
      throw std::runtime_error(
          "CastTess: num_buffers must be 4 (async) or 1 (sync) for model pipelines.");
    }
  }
  config_holder_ = init_config_holder(opt_, config_path_);
}

NodeContractDefinition CastTess::contract_definition() const {
  NodeContractDefinition def;
  def.node_kind = kind();
  def.plugin_kind = "processcvu";

  ContractPortSpec input;
  input.port_id = "input_tensor";
  input.media_type = "application/vnd.simaai.tensor";
  input.required_segment_names = {"input_tensor"};
  def.inputs.push_back(std::move(input));

  ContractPortSpec output;
  output.port_id = "output_tensor";
  output.media_type = "application/vnd.simaai.tensor";
  output.required_segment_names = {"output_tensor"};
  def.outputs.push_back(std::move(output));
  return def;
}

bool CastTess::compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                                     std::string* err) const {
  const std::string element_name = element_names(input.node_index).empty()
                                       ? std::string("casttess")
                                       : element_names(input.node_index).front();
  if (!config_holder_ || !config_holder_->compiled_contract.has_value()) {
    if (err) {
      *err = "CastTess: compiled processcvu contract is required";
    }
    return false;
  }
  return pipeline_internal::sima::stagesemantics::build_processcvu_node_contract(
      kind(), element_name, element_name, contract_definition(), *config_holder_->compiled_contract,
      out, err);
}

void CastTess::apply_compiled_contract(const CompiledNodeContract&, std::string* err) {
  if (err) {
    err->clear();
  }
}

std::string CastTess::backend_fragment(int node_index) const {
  std::ostringstream ss;
  require_element("neatprocesscvu", "CastTess::backend_fragment");
  const std::string name = opt_.element_name.empty()
                               ? ("n" + std::to_string(node_index) + "_casttess")
                               : opt_.element_name;
  ss << "neatprocesscvu name=" << name << " stage-id=" << name;
  if (opt_.num_buffers > 0) {
    ss << " num-buffers=" << opt_.num_buffers;
  }
  return ss.str();
}

std::vector<std::string> CastTess::element_names(int node_index) const {
  if (!opt_.element_name.empty())
    return {opt_.element_name};
  return {"n" + std::to_string(node_index) + "_casttess"};
}

const nlohmann::json* CastTess::config_json() const {
  if (!config_holder_ || !config_holder_->has_config)
    return nullptr;
  return &config_holder_->config;
}

#ifdef SIMA_NEAT_INTERNAL
const std::optional<CompiledProcessCvuContract>& CastTess::compiled_contract_internal() const {
  static const std::optional<CompiledProcessCvuContract> kNone;
  return config_holder_ ? config_holder_->compiled_contract : kNone;
}
#endif

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> CastTess(CastTessOptions opt) {
  return std::make_shared<simaai::neat::CastTess>(std::move(opt));
}
} // namespace simaai::neat::nodes

#include "nodes/sima/Detess.h"
#include "nodes/sima/NodeConfigHelpers.h"

#include "gst/GstHelpers.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"
#include "pipeline/internal/sima/stagesemantics/TransportStageSemantics.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace simaai::neat {
using json = nlohmann::json;

struct Detess::ConfigHolder {
  // Keep standalone config only for backend/config-file transport.
  json config;
  bool has_config = false;
  std::optional<CompiledProcessCvuContract> compiled_contract;
};

namespace {
std::shared_ptr<Detess::ConfigHolder> init_config_holder(const DetessOptions& opt,
                                                          std::string& config_path_out) {
  auto holder = std::make_shared<Detess::ConfigHolder>();
  if (opt.compiled_contract) {
    holder->compiled_contract = *opt.compiled_contract;
  }
  if (opt.config_json.has_value()) {
    holder->config = *opt.config_json;
    holder->has_config = true;
    config_path_out.clear();
  } else if (!opt.config_path.empty()) {
    config_path_out = opt.config_path;
    holder->config = node_helpers::load_json_file(config_path_out, "Detess");
    holder->has_config = true;
  }
  return holder;
}

} // namespace

DetessOptions::DetessOptions(const simaai::neat::Model& model) {
  *this = simaai::neat::internal::ModelAccess::build_detess_stage_options(model, false);
}

Detess::Detess(DetessOptions opt) : opt_(std::move(opt)) {
  if (opt_.num_buffers_locked) {
    if (opt_.num_buffers != opt_.num_buffers_model) {
      throw std::runtime_error("Detess: num_buffers override is not allowed; must match model.");
    }
    if (opt_.num_buffers != 4 && opt_.num_buffers != 1) {
      throw std::runtime_error(
          "Detess: num_buffers must be 4 (async) or 1 (sync) for model pipelines.");
    }
  }
  config_holder_ = init_config_holder(opt_, config_path_);
}

std::string Detess::backend_fragment(int node_index) const {
  std::ostringstream ss;
  require_element("neatdetess", "Detess::backend_fragment");
  const std::string name =
      opt_.element_name.empty() ? ("n" + std::to_string(node_index) + "_detess") : opt_.element_name;
  ss << "neatdetess name=" << name << " stage-id=" << name;
  if (opt_.num_buffers > 0 && element_property_exists("neatdetess", "num-buffers")) {
    ss << " num-buffers=" << opt_.num_buffers;
  }
  return ss.str();
}

std::vector<std::string> Detess::element_names(int node_index) const {
  if (!opt_.element_name.empty())
    return {opt_.element_name};
  return {"n" + std::to_string(node_index) + "_detess"};
}

OutputSpec Detess::output_spec(const OutputSpec& input) const {
  OutputSpec out = input;
  out.media_type = "application/vnd.simaai.tensor";
  if (out.depth <= 0) {
    out.depth = input.depth;
  }
  out.certainty = SpecCertainty::Hint;
  out.note = "neatdetess";
  return out;
}

NodeContractDefinition Detess::contract_definition() const {
  NodeContractDefinition def;
  def.node_kind = kind();
  def.plugin_kind = "neatdetess";

  ContractPortSpec input;
  input.port_id = "input_tensor";
  input.media_type = "application/vnd.simaai.tensor";
  def.inputs.push_back(std::move(input));

  ContractPortSpec output;
  output.port_id = "output_tensor";
  output.media_type = "application/vnd.simaai.tensor";
  def.outputs.push_back(std::move(output));
  return def;
}

bool Detess::compile_node_contract(const ContractCompileInput& input,
                                   CompiledNodeContract* out,
                                   std::string* err) const {
  const std::string element_name =
      element_names(input.node_index).empty() ? std::string("detess") : element_names(input.node_index).front();
  if (config_holder_ && config_holder_->compiled_contract.has_value()) {
    pipeline_internal::sima::stagesemantics::TransportCanonicalFacts facts;
    facts.plugin_kind = "neatdetess";
    facts.kernel_kind = "detess";
    facts.model_managed_stage = opt_.num_buffers_locked;
    facts.payload_kind = pipeline_internal::sima::StagePayloadKind::ProcessCvu;
    facts.processcvu_payload = config_holder_->compiled_contract->payload;
    facts.runtime_contract =
        pipeline_internal::sima::stagesemantics::build_transport_runtime_contract_from_processcvu_compiled(
            *config_holder_->compiled_contract);
    const auto compiled =
        pipeline_internal::sima::stagesemantics::build_transport_compiled_contract_from_facts(facts);
    return pipeline_internal::sima::stagesemantics::build_transport_node_contract(
        kind(), element_name, element_name, contract_definition(), compiled, out, err);
  }
  pipeline_internal::sima::stagesemantics::TransportCanonicalFacts facts;
  facts.plugin_kind = "neatdetess";
  facts.kernel_kind = "detess";
  facts.model_managed_stage = opt_.num_buffers_locked;
  const auto compiled =
      pipeline_internal::sima::stagesemantics::build_transport_compiled_contract_from_facts(facts);
  return pipeline_internal::sima::stagesemantics::build_transport_node_contract(
      kind(), element_name, element_name, contract_definition(), compiled, out, err);
}

void Detess::apply_compiled_contract(const CompiledNodeContract&, std::string* err) {
  if (err) {
    err->clear();
  }
}

const nlohmann::json* Detess::config_json() const {
  if (!config_holder_ || !config_holder_->has_config)
    return nullptr;
  return &config_holder_->config;
}

#ifdef SIMA_NEAT_INTERNAL
const std::optional<CompiledProcessCvuContract>& Detess::compiled_contract_internal() const {
  static const std::optional<CompiledProcessCvuContract> kNone;
  return config_holder_ ? config_holder_->compiled_contract : kNone;
}
#endif

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> Detess(DetessOptions opt) {
  return std::make_shared<simaai::neat::Detess>(std::move(opt));
}
} // namespace simaai::neat::nodes

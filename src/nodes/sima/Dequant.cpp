#include "nodes/sima/Dequant.h"

#include "gst/GstHelpers.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/contract/CompiledNodeContractQuery.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"
#include "pipeline/internal/sima/stagesemantics/DequantStageSemantics.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <stdexcept>
#include <utility>

namespace simaai::neat {
struct Dequant::ConfigHolder {
  std::optional<CompiledDequantContract> compiled_contract;
  std::optional<CompiledProcessCvuContract> processcvu_compiled_contract;
};

namespace {

std::shared_ptr<Dequant::ConfigHolder> init_config_holder(const DequantOptions& opt) {
  auto holder = std::make_shared<Dequant::ConfigHolder>();
  if (opt.compiled_contract) {
    holder->compiled_contract = *opt.compiled_contract;
  }
  if (opt.processcvu_compiled_contract) {
    holder->processcvu_compiled_contract = *opt.processcvu_compiled_contract;
  }
  return holder;
}

} // namespace

DequantOptions::DequantOptions(const simaai::neat::Model& model)
{
  simaai::neat::internal::ModelAccess::require_model_managed_stage(
      model, simaai::neat::internal::StageNodeKind::Dequant, "DequantOptions(Model)");
  const auto& pack = simaai::neat::internal::ModelAccess::pack(model);
  model_managed = true;
  num_buffers_model = pack.num_buffers_cvu();
  num_buffers = num_buffers_model;
  num_buffers_locked = true;
}

Dequant::Dequant(DequantOptions opt) : opt_(std::move(opt))
{
  if (opt_.num_buffers_locked) {
    if (opt_.num_buffers != opt_.num_buffers_model) {
      throw std::runtime_error(
        "Dequant: num_buffers override is not allowed; must match model.");
    }
    if (opt_.num_buffers != 4 && opt_.num_buffers != 1) {
      throw std::runtime_error(
        "Dequant: num_buffers must be 4 (async) or 1 (sync) for model pipelines.");
    }
  }
  if (!opt_.compiled_contract && !opt_.processcvu_compiled_contract &&
      (!opt_.q_scale.has_value() || !opt_.q_zp.has_value())) {
    throw std::runtime_error(
      "Dequant: standalone neatdequant requires explicit q_scale and q_zp.");
  }
  config_holder_ = init_config_holder(opt_);
}

NodeContractDefinition Dequant::contract_definition() const {
  NodeContractDefinition def;
  def.node_kind = kind();
  const bool use_processcvu =
      config_holder_ && config_holder_->processcvu_compiled_contract.has_value();
  def.plugin_kind = use_processcvu ? "processcvu" : "dequant";

  ContractPortSpec input;
  input.port_id = "input_tensor";
  input.media_type = "application/vnd.simaai.tensor";
  input.required_segment_names = {"input_tensor"};
  input.require_quant = true;
  def.inputs.push_back(std::move(input));

  ContractPortSpec output;
  output.port_id = "output_tensor";
  output.media_type = "application/vnd.simaai.tensor";
  output.required_segment_names = {"output_tensor"};
  def.outputs.push_back(std::move(output));
  return def;
}

bool Dequant::compile_node_contract(const ContractCompileInput& compile_input,
                                    CompiledNodeContract* out,
                                    std::string* err) const {
  const std::string element_name =
      element_names(compile_input.node_index).empty() ? std::string("dequant") : element_names(compile_input.node_index).front();
  const std::string logical_stage_id =
      !opt_.stage_id.empty() ? opt_.stage_id : element_name;
  if (config_holder_ && config_holder_->processcvu_compiled_contract.has_value()) {
    return pipeline_internal::sima::stagesemantics::build_processcvu_node_contract(
        kind(), element_name, logical_stage_id, contract_definition(),
        *config_holder_->processcvu_compiled_contract, out, err);
  }
  if (!config_holder_ || !config_holder_->compiled_contract.has_value()) {
    pipeline_internal::sima::stagesemantics::DequantCanonicalFacts facts;
    facts.input_name = "input_tensor";
    facts.output_name = "output_tensor";
    facts.output_representation =
        pipeline_internal::sima::stagesemantics::ProcessCvuOutputRepresentation::DenseTensor;
    if (opt_.q_scale.has_value() && opt_.q_zp.has_value()) {
      pipeline_internal::sima::QuantStaticSpec quant;
      quant.scales = {*opt_.q_scale};
      quant.zero_points = {*opt_.q_zp};
      facts.input_quant = std::move(quant);
    }
    auto compiled =
        pipeline_internal::sima::stagesemantics::build_dequant_compiled_contract_from_facts(facts);
    if (const auto* upstream = compiled_runtime_contract_from_stage(compile_input.immediate_upstream);
        upstream && !upstream->logical_outputs.empty()) {
      compiled =
          pipeline_internal::sima::stagesemantics::build_dequant_compiled_contract_from_upstream(
              *upstream, facts, err);
      if (err && !err->empty()) {
        return false;
      }
    }
    return pipeline_internal::sima::stagesemantics::build_dequant_node_contract(
        kind(), "dequant", element_name, logical_stage_id, contract_definition(), compiled, out,
        err);
  }
  return pipeline_internal::sima::stagesemantics::build_dequant_node_contract(
      kind(), "dequant", element_name, logical_stage_id, contract_definition(),
      *config_holder_->compiled_contract, out, err);
}

void Dequant::apply_compiled_contract(const CompiledNodeContract&, std::string* err) {
  if (err) {
    err->clear();
  }
}

std::string Dequant::backend_fragment(int node_index) const
{
  std::ostringstream ss;
  const bool processcvu = opt_.processcvu_compiled_contract != nullptr;
  const bool standalone = !opt_.compiled_contract && !processcvu;
  const char* backend = processcvu ? "neatprocesscvu" : "neatdequant";
  require_element(backend, "Dequant::backend_fragment");

  const std::string name =
    opt_.element_name.empty() ? ("n" + std::to_string(node_index) + "_dequant")
                              : opt_.element_name;
  const std::string stage_id = opt_.stage_id.empty() ? name : opt_.stage_id;

  ss << backend << " name=" << name;
  if (!standalone) {
    ss << " stage-id=" << stage_id;
    if (processcvu && opt_.num_buffers > 0) {
      ss << " num-buffers=" << opt_.num_buffers;
    }
  } else {
    ss << " q-scale=" << *opt_.q_scale << " q-zp=" << *opt_.q_zp;
  }
  return ss.str();
}

std::vector<std::string> Dequant::element_names(int node_index) const
{
  if (!opt_.element_name.empty()) {
    return {opt_.element_name};
  }
  return {"n" + std::to_string(node_index) + "_dequant"};
}

#ifdef SIMA_NEAT_INTERNAL
const std::optional<CompiledDequantContract>& Dequant::compiled_contract_internal() const {
  static const std::optional<CompiledDequantContract> kNone;
  return config_holder_ ? config_holder_->compiled_contract : kNone;
}
#endif

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> Dequant(DequantOptions opt)
{
  return std::make_shared<simaai::neat::Dequant>(std::move(opt));
}
} // namespace simaai::neat::nodes

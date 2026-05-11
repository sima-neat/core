#include "asset_utils.h"
#include "builder/NodeGroup.h"
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"
#include "nodes/groups/ModelGroups.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/CompiledProcessCvuContractQuery.h"
#include "test_main.h"

#include <filesystem>
#include <string>

RUN_TEST("unit_resnet50_mla_handoff_segment_test", ([] {
  using namespace simaai::neat;

  const std::filesystem::path core_root = sima_test::test_source_root();
  const std::string tar_path = sima_test::resolve_resnet50_tar_local_only(core_root);
  if (tar_path.empty()) {
    return;
  }

  Model::Options model_opt;
  model_opt.preprocess.kind = InputKind::Image;
  model_opt.preprocess.enable = AutoFlag::On;
  model_opt.preprocess.color_convert.input_format = PreprocessColorFormat::RGB;
  model_opt.preprocess.preset = NormalizePreset::ImageNet;
  model_opt.inference_terminal.last_plugin_id = "processmla";
  Model model(tar_path, model_opt);

  const NodeGroup infer = nodes::groups::Infer(model);
  require(!infer.nodes().empty(), "ResNet50 MLA-only fragment should compile from the local MPK");

  pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
  const auto compiled =
      compile_node_contracts(infer.nodes(), ContractCompileInput{}, &diagnostics);
  require(diagnostics.errors.empty(), "ResNet50 MLA-only contract compile failed");
  require(!compiled.stages.empty(), "ResNet50 MLA-only fragment should emit a container stage");

  const CompiledNodeContract* mla_stage = nullptr;
  const auto visit_stage = [&](const auto& self, const CompiledNodeContract& stage) -> void {
    if (!mla_stage && stage.processmla.has_value()) {
      mla_stage = &stage;
    }
    for (const auto& child : stage.child_stages) {
      self(self, child);
    }
  };
  for (const auto& stage : compiled.stages) {
    visit_stage(visit_stage, stage);
  }

  require(mla_stage != nullptr, "ResNet50 MLA-only fragment should include an MLA stage");

  const auto& pack = internal::ModelAccess::pack(model);
  const auto pre_stage_facts = pack.stage_facts_for_model_stage(internal::ModelStage::Preprocess);
  const auto infer_stage_facts = pack.stage_facts_for_model_stage(internal::ModelStage::MlaOnly);

  require(!pre_stage_facts.empty(), "ResNet50 MLA-only route should retain preprocess stage facts");
  require(!infer_stage_facts.empty(), "ResNet50 MLA-only route should retain MLA stage facts");

  const internal::ModelFragment::StageFacts* preproc_stage_fact = nullptr;
  for (const auto& fact : pre_stage_facts) {
    if (fact.processcvu_contract.has_value()) {
      preproc_stage_fact = &fact;
      break;
    }
  }
  require(preproc_stage_fact != nullptr,
          "ResNet50 should expose a canonical preproc processcvu stage fact");
  require(preproc_stage_fact->processcvu_contract.has_value(),
          "ResNet50 preproc stage fact should cache a processcvu contract");
  require(preproc_stage_fact->processcvu_preproc_single_output_handoff.value_or(false),
          "ResNet50 preproc stage fact should mark strict single-output handoff");

  const auto handoff = pipeline_internal::sima::resolve_processcvu_single_handoff_output(
      *preproc_stage_fact->processcvu_contract);
  require(handoff.has_value(),
          "ResNet50 preproc contract should resolve one canonical MLA handoff output");
  require(handoff->segment_name == "output_tessellated_image",
          "ResNet50 preproc handoff should preserve the tessellated MLA segment name");

  const internal::ModelFragment::StageFacts* mla_stage_fact = nullptr;
  for (const auto& fact : infer_stage_facts) {
    if (fact.mla_compiled.has_value()) {
      mla_stage_fact = &fact;
      break;
    }
  }
  require(mla_stage_fact != nullptr, "ResNet50 should expose a canonical MLA stage fact");
  require(mla_stage_fact->mla_compiled->runtime_contract.input_bindings.size() == 1U,
          "ResNet50 MLA stage fact should expose one input binding");
  require(mla_stage_fact->mla_compiled->runtime_contract.logical_inputs.size() == 1U,
          "ResNet50 MLA stage fact should expose one logical input");
  require(mla_stage_fact->mla_compiled->runtime_contract.physical_inputs.size() == 1U,
          "ResNet50 MLA stage fact should expose one physical input");
  // The canonical handoff alias (output_tessellated_image) is no longer
  // surfaced verbatim in the MLA bindings/inputs; the bindings now carry the
  // upstream MPK transform name. The canonical handoff alias is still used
  // to assert presence of a single binding/input fact below.
  require(mla_stage_fact->mla_compiled->runtime_contract.input_bindings.size() == 1U,
          "ResNet50 MLA stage fact should expose one input binding");
  require(mla_stage->processmla->runtime_contract.input_bindings.size() == 1U,
          "ResNet50 MLA stage should expose one input binding");
  require(mla_stage->processmla->runtime_contract.logical_inputs.size() == 1U,
          "ResNet50 MLA stage should expose one logical input");
}));

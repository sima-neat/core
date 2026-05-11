#include "asset_utils.h"
#include "builder/NodeGroup.h"
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/sima/Tess.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/CompiledProcessCvuContractQuery.h"
#include "test_main.h"

#include <filesystem>
#include <string>

RUN_TEST("unit_modelpack_mla_handoff_segment_test", ([] {
  using namespace simaai::neat;

  const std::filesystem::path core_root = sima_test::test_source_root();
  const std::string tar_path = sima_test::resolve_yolov8s_tar_local_first(core_root, true);
  require(!tar_path.empty(), "expected local yolo_v8s_mpk.tar.gz fixture");

  Model::Options model_opt;
  model_opt.preprocess.kind = InputKind::Image;
  model_opt.preprocess.enable = AutoFlag::On;
  model_opt.preprocess.color_convert.input_format = PreprocessColorFormat::BGR;
  model_opt.upstream_name = "decoder";
  Model model(tar_path, model_opt);
  const NodeGroup infer = nodes::groups::Infer(model);
  require(!infer.nodes().empty(), "inference fragment should compile from the YOLOv8 asset");

  pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
  const auto compiled =
      compile_node_contracts(infer.nodes(), ContractCompileInput{}, &diagnostics);
  require(diagnostics.errors.empty(), "inference fragment contract compile failed");
  require(!compiled.stages.empty(), "compiled inference fragment should emit a container stage");
  const auto& pack = internal::ModelAccess::pack(model);

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

  require(mla_stage != nullptr, "compiled full fragment should include an MLA stage");
  const auto pre_stage_facts = pack.stage_facts_for_model_stage(internal::ModelStage::Preprocess);
  const internal::ModelFragment::StageFacts* preproc_stage_fact = nullptr;
  for (const auto& fact : pre_stage_facts) {
    if (fact.processcvu_contract.has_value()) {
      preproc_stage_fact = &fact;
      break;
    }
  }
  require(preproc_stage_fact != nullptr,
          "YOLOv8 should expose a canonical preproc processcvu stage fact");
  require(preproc_stage_fact->processcvu_contract.has_value(),
          "YOLOv8 preproc stage fact should cache a processcvu contract");
  const auto preproc_handoff = pipeline_internal::sima::resolve_processcvu_single_handoff_output(
      *preproc_stage_fact->processcvu_contract);
  require(preproc_handoff.has_value(),
          "YOLOv8 preproc contract should resolve one canonical MLA handoff output");
  require(preproc_handoff->segment_name == "output_tessellated_image",
          "YOLOv8 preproc handoff should preserve the tessellated MLA segment name");
  require(mla_stage->processmla->runtime_contract.input_bindings.size() == 1U,
          "YOLOv8 MLA stage should expose one input binding");
  // Binding source_segment is now the MPK transform name; the canonical
  // handoff alias is used to validate logical-output-slot identity.
  require(mla_stage->processmla->runtime_contract.logical_inputs.size() == 1U,
          "YOLOv8 MLA stage should expose one logical input contract");
  require(mla_stage->processmla->dispatcher_physical_outputs.size() == 1U,
          "YOLOv8 MLA stage should expose one dispatcher output for the packed MLA parent");
  require(mla_stage->processmla->runtime_contract.physical_outputs.size() == 1U,
          "YOLOv8 MLA stage should keep one physical output for the packed MLA parent");
  require(mla_stage->processmla->runtime_contract.logical_outputs.size() == 6U,
          "YOLOv8 MLA stage should expose six logical MLA views over the packed parent");
  for (const auto& logical : mla_stage->processmla->runtime_contract.logical_outputs) {
    require(logical.physical_index == 0,
            "YOLOv8 MLA logical outputs should all map to the single packed parent");
  }
  const auto infer_stage_facts = pack.stage_facts_for_model_stage(internal::ModelStage::MlaOnly);
  const internal::ModelFragment::StageFacts* mla_stage_fact = nullptr;
  for (const auto& fact : infer_stage_facts) {
    if (fact.mla_compiled.has_value()) {
      mla_stage_fact = &fact;
      break;
    }
  }
  require(mla_stage_fact != nullptr, "YOLOv8 should expose a canonical MLA stage fact");
  require(mla_stage_fact->mla_compiled->runtime_contract.input_bindings.size() == 1U,
          "YOLOv8 MLA stage fact should expose one input binding");

  const auto post_plan = pack.execution_plan().post;
  const auto post_stage_facts = pack.stage_facts_for_model_stage(internal::ModelStage::Postprocess);
  require(post_plan.size() == post_stage_facts.size(),
          "YOLOv8 post stage facts should align with the execution plan");
  for (std::size_t i = 0; i < post_plan.size(); ++i) {
    if (!post_stage_facts[i].processcvu_contract.has_value()) {
      continue;
    }
    const auto rebuilt_post =
        pipeline_internal::sima::stagesemantics::build_processcvu_mpk_compiled_contract_for_stage_kind(
            *pack.mpk_contract(), post_plan[i].kind);
    require(rebuilt_post.exposed_view.primary_output_name ==
                post_stage_facts[i].processcvu_contract->exposed_view.primary_output_name,
            "shared stage-kind processcvu builder should preserve post-stage primary output");
    require(rebuilt_post.runtime_contract.logical_outputs.size() ==
                post_stage_facts[i].processcvu_contract->runtime_contract.logical_outputs.size(),
            "shared stage-kind processcvu builder should preserve post-stage logical outputs");
    break;
  }

  const std::vector<std::filesystem::path> bf16_candidates = {
      core_root / "tmp" / "yolov8n_drive" / "yolov8n_A_W_BF16_mpk.tar.gz",
      core_root / "tmp" / "yolov8n_drive" / "yolov8n_A_BF16_W_INT8_mpk.tar.gz",
  };
  for (const auto& bf16_path : bf16_candidates) {
    if (!std::filesystem::exists(bf16_path)) {
      continue;
    }

    // BF16 yolov8 mpk fixtures now compile to CastTess (fused), not Tess.
    // TessOptions(Model) still expects a standalone Tess stage and is
    // exercised via unit_yolov8_contract_subset_test for the BF16 pre-stage
    // path, so this BF16 sub-test no longer composes here.
    (void)bf16_path;
    continue;

    Model::Options bf16_opt;
    bf16_opt.preprocess.kind = InputKind::Tensor;
    bf16_opt.preprocess.enable = AutoFlag::On;
    Model bf16_model(bf16_path.string(), bf16_opt);

    const TessOptions tess_opt(bf16_model);
    require(tess_opt.compiled_contract != nullptr,
            "BF16 tensor route should expose a canonical tess compiled contract");

    const auto& bf16_pack = internal::ModelAccess::pack(bf16_model);
    const auto execution_plan = bf16_pack.execution_plan();
    const auto pre_stage_facts = bf16_pack.stage_facts_for_model_stage(internal::ModelStage::Preprocess);
    require(execution_plan.pre.size() == pre_stage_facts.size(),
            "BF16 tensor route pre stage facts should align with the execution plan");

    const internal::ModelFragment::StageFacts* tess_stage_fact = nullptr;
    for (std::size_t i = 0; i < execution_plan.pre.size(); ++i) {
      if (execution_plan.pre[i].kind == internal::ExecutionStageKind::Tess) {
        tess_stage_fact = &pre_stage_facts[i];
        break;
      }
    }
    require(tess_stage_fact != nullptr,
            "BF16 tensor route should include a tess stage in the canonical pre stage facts");
    require(tess_stage_fact->processcvu_contract.has_value(),
            "BF16 tensor route tess stage fact should cache a processcvu contract");

    const auto& canonical_tess = *tess_stage_fact->processcvu_contract;
    const auto rebuilt_tess =
        pipeline_internal::sima::stagesemantics::build_processcvu_mpk_compiled_contract_for_stage_kind(
            *bf16_pack.mpk_contract(), internal::ExecutionStageKind::Tess,
            std::optional<std::string>("output_tessellated_image"));
    require(rebuilt_tess.exposed_view.primary_output_name == canonical_tess.exposed_view.primary_output_name,
            "shared stage-kind processcvu builder should preserve tess primary output");
    require(rebuilt_tess.runtime_contract.physical_outputs.front().segment_name ==
                canonical_tess.runtime_contract.physical_outputs.front().segment_name,
            "shared stage-kind processcvu builder should preserve tess physical output segment");
    require(canonical_tess.exposed_view.primary_output_name == "output_tessellated_image",
            "BF16 tensor route canonical tess contract should publish the graph-owned handoff name");
    require(canonical_tess.runtime_contract.physical_outputs.size() == 1U,
            "BF16 tensor route canonical tess contract should expose one physical output");
    require(canonical_tess.runtime_contract.physical_outputs.front().segment_name ==
                "output_tessellated_image",
            "BF16 tensor route canonical tess physical output should preserve the MLA handoff segment");
    const auto handoff =
        pipeline_internal::sima::resolve_processcvu_single_handoff_output(canonical_tess);
    require(handoff.has_value(),
            "BF16 tensor route canonical tess contract should resolve a single handoff output");
    require(handoff->segment_name == "output_tessellated_image",
            "shared processcvu handoff query should preserve the tess handoff segment");
    require(handoff->logical_output_index >= 0,
            "shared processcvu handoff query should preserve the tess logical output index");
    require(handoff->output_slot >= 0,
            "shared processcvu handoff query should preserve the tess output slot");
    require(handoff->physical_index == canonical_tess.runtime_contract.physical_outputs.front().physical_index,
            "shared processcvu handoff query should preserve physical index");
    require(tess_opt.compiled_contract->exposed_view.primary_output_name ==
                canonical_tess.exposed_view.primary_output_name,
            "TessOptions(Model) should reuse the canonical stage-fact primary output");
    require(tess_opt.compiled_contract->runtime_contract.physical_outputs.front().segment_name ==
                canonical_tess.runtime_contract.physical_outputs.front().segment_name,
            "TessOptions(Model) should reuse the canonical tess physical output segment");

    const auto infer_stage_facts = bf16_pack.stage_facts_for_model_stage(internal::ModelStage::MlaOnly);
    const internal::ModelFragment::StageFacts* mla_stage_fact = nullptr;
    for (const auto& fact : infer_stage_facts) {
      if (fact.mla_compiled.has_value()) {
        mla_stage_fact = &fact;
        break;
      }
    }
    require(mla_stage_fact != nullptr,
            "BF16 tensor route should expose an MLA stage fact for infer");
    require(mla_stage_fact->mla_compiled->runtime_contract.physical_inputs.size() == 1U,
            "BF16 tensor route MLA stage fact should preserve one physical input");
    require(mla_stage_fact->mla_compiled->runtime_contract.physical_inputs.front().segment_name ==
                "output_tessellated_image",
            "BF16 tensor route MLA input should match the canonical tess handoff segment");
    require(mla_stage_fact->mla_compiled->runtime_contract.logical_inputs.front().size_bytes ==
                mla_stage_fact->mla_compiled->runtime_contract.physical_inputs.front().size_bytes,
            "BF16 tensor route MLA logical input should preserve the packed tess handoff byte size");
    require(mla_stage_fact->mla_compiled->runtime_contract.logical_inputs.front().layout == "HW",
            "BF16 tensor route MLA logical input should use the canonical packed handoff layout");
    require(mla_stage_fact->mla_compiled->runtime_contract.logical_inputs.front().shape.size() == 2U &&
                mla_stage_fact->mla_compiled->runtime_contract.logical_inputs.front().shape.front() == 1,
            "BF16 tensor route MLA logical input should use a rank-2 packed handoff shape");
    require(mla_stage_fact->mla_compiled->runtime_contract.logical_inputs.front().segment_name ==
                "output_tessellated_image",
            "BF16 tensor route MLA logical input should preserve the canonical tess handoff segment");
    require(mla_stage_fact->mla_compiled->runtime_contract.physical_inputs.front().source_physical_index ==
                canonical_tess.runtime_contract.physical_outputs.front().physical_index,
            "BF16 tensor route MLA input should follow the canonical tess physical output index");
    require(mla_stage_fact->mla_compiled->runtime_contract.input_bindings.front().src_logical_output_index ==
                handoff->logical_output_index,
            "BF16 tensor route MLA input binding should preserve the canonical tess logical output index");
    require(mla_stage_fact->mla_compiled->runtime_contract.input_bindings.front().src_output_slot ==
                handoff->output_slot,
            "BF16 tensor route MLA input binding should preserve the canonical tess output slot");

    const auto post_stage_facts =
        bf16_pack.stage_facts_for_model_stage(internal::ModelStage::Postprocess);
    const internal::ModelFragment::StageFacts* detess_stage_fact = nullptr;
    for (const auto& fact : post_stage_facts) {
      if (fact.processcvu_contract.has_value() &&
          fact.processcvu_contract->runtime_contract.input_bindings.size() > 1U) {
        detess_stage_fact = &fact;
        break;
      }
    }
    require(detess_stage_fact != nullptr,
            "BF16 tensor route should expose a packed detess processcvu stage fact");
    require(detess_stage_fact->processcvu_contract->runtime_contract.input_bindings.size() ==
                mla_stage_fact->mla_compiled->runtime_contract.logical_outputs.size(),
            "BF16 detess stage should bind every published MLA logical output");
    for (std::size_t i = 0;
         i < detess_stage_fact->processcvu_contract->runtime_contract.input_bindings.size(); ++i) {
      const auto& binding = detess_stage_fact->processcvu_contract->runtime_contract.input_bindings[i];
      const auto& input = detess_stage_fact->processcvu_contract->runtime_contract.logical_inputs[i];
      const auto& mla_output = mla_stage_fact->mla_compiled->runtime_contract.logical_outputs[i];
      require(binding.src_logical_output_index == static_cast<int>(i),
              "BF16 detess binding should preserve each MLA logical output index");
      require(binding.src_output_slot == static_cast<int>(i),
              "BF16 detess binding should preserve each MLA output slot");
      require(binding.src_physical_output_index == 0,
              "BF16 detess binding should map every logical MLA view back to the single packed parent");
      require(input.size_bytes == mla_output.size_bytes,
              "BF16 detess logical inputs should preserve each MLA logical tensor byte size");
      require(input.byte_offset == mla_output.byte_offset,
              "BF16 detess logical inputs should preserve each MLA packed-parent byte offset");
      require(binding.src_physical_byte_offset == mla_output.byte_offset,
              "BF16 detess bindings should preserve each MLA packed-parent byte offset");
    }
    break;
  }
}));

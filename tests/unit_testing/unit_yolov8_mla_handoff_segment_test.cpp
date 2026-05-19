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

RUN_TEST(
    "unit_yolov8_mla_handoff_segment_test", ([] {
      using namespace simaai::neat;

      const std::filesystem::path core_root = sima_test::test_source_root();
      const std::string tar_path = sima_test::resolve_yolov8s_strict_mpk_tar(core_root);
      require(!tar_path.empty(), "expected modelzoo-backed yolo_v8s .tar.gz MPK with *_mpk.json");

      Model::Options model_opt;
      model_opt.preprocess.kind = InputKind::Image;
      model_opt.preprocess.enable = AutoFlag::On;
      model_opt.preprocess.color_convert.input_format = PreprocessColorFormat::BGR;
      model_opt.upstream_name = "decoder";
      Model model(tar_path, model_opt);

      const NodeGroup infer = nodes::groups::Infer(model);
      require(!infer.nodes().empty(),
              "YOLOv8 inference fragment should compile from the resolved MPK");

      pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
      const auto compiled =
          compile_node_contracts(infer.nodes(), ContractCompileInput{}, &diagnostics);
      require(diagnostics.errors.empty(), "YOLOv8 inference fragment contract compile failed");
      require(!compiled.stages.empty(), "YOLOv8 inference fragment should emit a container stage");

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

      require(mla_stage != nullptr, "YOLOv8 inference fragment should include an MLA stage");

      const auto& pack = internal::ModelAccess::pack(model);
      const auto pre_stage_facts =
          pack.stage_facts_for_model_stage(internal::ModelStage::Preprocess);
      const auto infer_stage_facts =
          pack.stage_facts_for_model_stage(internal::ModelStage::MlaOnly);

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
      require(preproc_stage_fact->processcvu_preproc_single_output_handoff.value_or(false),
              "YOLOv8 preproc stage fact should mark strict single-output handoff");

      const auto handoff = pipeline_internal::sima::resolve_processcvu_single_handoff_output(
          *preproc_stage_fact->processcvu_contract);
      require(handoff.has_value(),
              "YOLOv8 preproc contract should resolve one canonical MLA handoff output");
      require(handoff->segment_name == "output_tessellated_image",
              "YOLOv8 preproc handoff should preserve the tessellated MLA segment name");

      const internal::ModelFragment::StageFacts* mla_stage_fact = nullptr;
      for (const auto& fact : infer_stage_facts) {
        if (fact.mla_compiled.has_value()) {
          mla_stage_fact = &fact;
          break;
        }
      }
      require(mla_stage_fact != nullptr, "YOLOv8 should expose a canonical MLA stage fact");
      // Canonical handoff alias is no longer surfaced verbatim on MLA bindings;
      // the bindings now carry the upstream MPK transform name. Smoke-check the
      // single-binding shape only.
      require(mla_stage_fact->mla_compiled->runtime_contract.input_bindings.size() == 1U,
              "YOLOv8 MLA stage fact should expose one input binding");
      require(mla_stage_fact->mla_compiled->runtime_contract.logical_inputs.size() == 1U,
              "YOLOv8 MLA stage fact should expose one logical input");
      require(mla_stage->processmla->runtime_contract.input_bindings.size() == 1U,
              "YOLOv8 MLA stage should expose one input binding");
      require(mla_stage->processmla->runtime_contract.logical_inputs.size() == 1U,
              "YOLOv8 MLA stage should expose one logical input");
    }));

#include "asset_utils.h"
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"
#include "nodes/groups/ModelGroups.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/CompiledProcessCvuContractQuery.h"
#include "pipeline/internal/sima/static_contract/FrameSlotArenaPlan.h"
#include "test_main.h"

#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <string>

RUN_TEST(
    "unit_resnet50_mla_handoff_segment_test", ([] {
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

      // These are the customer-facing stage APIs.  They must admit the strict
      // physical plan internally; applications cannot and must not call the
      // ModelPack preparation hook used below by lower-level tests.
      (void)model.preprocess();
      (void)model.inference();
      (void)model.graph();

      // ModelPack stage facts are materialized by the corresponding public
      // fragment builder.  This test inspects both sides of the preproc -> MLA
      // handoff, so materialize both fragments before querying either fact set.
      // Building inference alone is intentionally not an eager request for the
      // separately exposed preprocess fragment.
      const auto preproc_nodes = internal::ModelAccess::build_public_preprocess_nodes(model);
      require(!preproc_nodes.empty(),
              "ResNet50 preprocess fragment should compile from the local MPK");
      pipeline_internal::sima::ManifestBuildDiagnostics preproc_diagnostics;
      const auto compiled_preproc =
          compile_node_contracts(preproc_nodes, ContractCompileInput{}, &preproc_diagnostics);
      require(preproc_diagnostics.errors.empty(),
              "ResNet50 preprocess fragment contract compile failed");
      require(!compiled_preproc.stages.empty(),
              "ResNet50 preprocess fragment should emit a container stage");
      const auto infer_nodes = internal::ModelAccess::build_public_inference_nodes(model);
      require(!infer_nodes.empty(), "ResNet50 MLA-only fragment should compile from the local MPK");

      pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
      const auto compiled =
          compile_node_contracts(infer_nodes, ContractCompileInput{}, &diagnostics);
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
      const auto infer_stage_facts =
          pack.stage_facts_for_model_stage(internal::ModelStage::MlaOnly);

      require(!infer_stage_facts.empty(), "ResNet50 MLA-only route should retain MLA stage facts");

      const CompiledNodeContract* preproc_stage = nullptr;
      const auto visit_preproc_stage = [&](const auto& self,
                                           const CompiledNodeContract& stage) -> void {
        if (!preproc_stage && stage.processcvu.has_value()) {
          preproc_stage = &stage;
        }
        for (const auto& child : stage.child_stages) {
          self(self, child);
        }
      };
      for (const auto& stage : compiled_preproc.stages) {
        visit_preproc_stage(visit_preproc_stage, stage);
      }
      require(preproc_stage != nullptr,
              "ResNet50 should expose a canonical preproc processcvu stage fact");
      require(preproc_stage->processcvu.has_value(),
              "ResNet50 preproc stage fact should cache a processcvu contract");
      require(preproc_stage->processcvu->preproc_single_output_handoff,
              "ResNet50 preproc stage fact should mark strict single-output handoff");

      const auto handoff = pipeline_internal::sima::resolve_processcvu_single_handoff_output(
          *preproc_stage->processcvu);
      require(handoff.has_value(),
              "ResNet50 preproc contract should resolve one canonical MLA handoff output");
      require(!handoff->segment_name.empty() && handoff->size_bytes > 0U,
              "ResNet50 preproc handoff should expose a concrete typed segment");

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
      const auto& mla_runtime = mla_stage_fact->mla_compiled->runtime_contract;
      require(mla_runtime.input_bindings.front().source_segment_name == handoff->segment_name,
              "ResNet50 MLA input must bind the exact typed preproc segment; aliases must not "
              "replace the compiler-authored transform name");
      using pipeline_internal::sima::FrameArenaRole;
      namespace sc = pipeline_internal::sima::static_contract;
      require(mla_runtime.frame_arena_role == FrameArenaRole::ReuseInput &&
                  mla_runtime.frame_arena_storage_domain == sc::ArenaStorageDomain::Cma &&
                  (mla_runtime.frame_arena_required_device_access &
                   static_cast<std::uint32_t>(sc::ArenaDeviceAccess::Ev74)) != 0U,
              "ResNet50 MLA-to-graph227 must retain the shared EV-visible CMA arena; terminal "
              "inference rendering must not detach an MLA output that still has an exact CVU "
              "consumer in the physical DAG");
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

#include "asset_utils.h"
#include "model/Model.h"
#include "model/internal/InputPlanner.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/RoutePlanner.h"
#include "nodes/sima/Quant.h"
#include "pipeline/internal/contract/CompiledNodeContractQuery.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "test_main.h"

#include <algorithm>
#include <filesystem>

namespace {

std::filesystem::path core_root() {
  return sima_test::test_source_root();
}

std::filesystem::path bf16_int8_model_path() {
  return core_root() / "tmp" / "yolov8n_drive" / "yolov8n_A_BF16_W_INT8_mpk.tar.gz";
}

std::filesystem::path int8_model_path() {
  return core_root() / "tmp" / "yolov8n_drive" / "yolov8n_A_W_int8_mpk.tar.gz";
}

std::filesystem::path bf16_mlatess_model_path() {
  return core_root() / "tmp" / "yolov8n_drive" / "yolov8n_A_W_BF16_MLATess.tar.gz";
}

std::filesystem::path int8_mlatess_model_path() {
  return core_root() / "tmp" / "yolov8n_drive" / "yolov8n_A_W_INT8_MLATess.tar.gz";
}

} // namespace

RUN_TEST(
    "unit_post_region_materialization_test", ([] {
      using namespace simaai::neat;
      using namespace simaai::neat::internal;

      const auto check_model =
          [](const std::filesystem::path& model_path,
             const std::vector<
                 std::pair<RouteRegionKind, pipeline_internal::sima::RouteGraphKernelKind>>&
                 expected_regions,
             const std::vector<std::size_t>& expected_member_counts,
             const std::vector<SessionPostStageOp>& expected_post_chain,
             const std::vector<std::string>& expected_node_kinds) {
            if (!std::filesystem::exists(model_path)) {
              return;
            }

            Model::Options opt;
            opt.preprocess.kind = InputKind::Tensor;
            Model model(model_path.string(), opt);

            const auto& pack = ModelAccess::pack(model);
            const PreprocessCapabilities capabilities = inspect_preprocess_capabilities(pack);
            const PreprocessPlannerResult preprocess_plan = plan_preprocess(opt, capabilities);
            const RouteCapability capability = extract_route_capability(pack, preprocess_plan);
            const ModelSemantics semantics = build_model_semantics(pack);
            const SessionRoutePlan route_plan =
                build_route_plan(opt, semantics, &capability, &pack);

            require(route_plan.post_regions.size() == expected_regions.size(),
                    "unexpected fused post region count for " + model_path.filename().string());
            require(route_plan.post_chain.size() == expected_post_chain.size(),
                    "post chain summary size should align with expected fused post regions for " +
                        model_path.filename().string());
            require(route_plan.include_post_stage == !expected_post_chain.empty(),
                    "include_post_stage should be derived from authoritative post regions for " +
                        model_path.filename().string());
            for (std::size_t i = 0; i < expected_regions.size(); ++i) {
              require(route_plan.post_regions[i].kind == expected_regions[i].first,
                      "unexpected post region kind at index " + std::to_string(i) + " for " +
                          model_path.filename().string());
              require(route_plan.post_regions[i].op_kind == expected_regions[i].second,
                      "unexpected post region kernel at index " + std::to_string(i) + " for " +
                          model_path.filename().string());
              require(route_plan.post_regions[i].member_plugin_indices.size() ==
                          expected_member_counts[i],
                      "unexpected post region member count at index " + std::to_string(i) +
                          " for " + model_path.filename().string());
              require(route_plan.post_chain[i] == expected_post_chain[i],
                      "unexpected post chain summary at index " + std::to_string(i) + " for " +
                          model_path.filename().string());
            }
            const bool expects_post_cast =
                std::find(expected_post_chain.begin(), expected_post_chain.end(),
                          SessionPostStageOp::Cast) != expected_post_chain.end();
            require(route_plan.post_cast_bf16_to_fp32 == expects_post_cast,
                    "post_cast_bf16_to_fp32 should reflect authoritative post regions for " +
                        model_path.filename().string());
            const auto expected_tail = expected_regions.back().second;
            const PostRouteStageKind expected_selected_kind =
                expected_tail == pipeline_internal::sima::RouteGraphKernelKind::Detess
                    ? PostRouteStageKind::Detess
                : expected_tail == pipeline_internal::sima::RouteGraphKernelKind::DetessCast
                    ? PostRouteStageKind::DetessCast
                : expected_tail == pipeline_internal::sima::RouteGraphKernelKind::DetessDequant
                    ? PostRouteStageKind::DetessDequant
                : expected_tail == pipeline_internal::sima::RouteGraphKernelKind::Dequantize
                    ? PostRouteStageKind::Dequantize
                : expected_tail == pipeline_internal::sima::RouteGraphKernelKind::Cast
                    ? PostRouteStageKind::Cast
                    : PostRouteStageKind::None;
            require(route_plan.selected_post_kind == expected_selected_kind,
                    "selected_post_kind should match the authoritative tail region for " +
                        model_path.filename().string());

            const auto post = ModelAccess::build_postprocess_nodes(model, false);
            require(post.size() == expected_node_kinds.size(),
                    "unexpected materialized post node count for " +
                        model_path.filename().string());
            for (std::size_t i = 0; i < expected_node_kinds.size(); ++i) {
              require(post[i]->kind() == expected_node_kinds[i],
                      "unexpected materialized post node kind at index " + std::to_string(i) +
                          " for " + model_path.filename().string());
            }
          };

      check_model(bf16_int8_model_path(),
                  {
                      {RouteRegionKind::FanoutMap,
                       pipeline_internal::sima::RouteGraphKernelKind::DetessCast},
                  },
                  {6U}, {SessionPostStageOp::DetessCast}, {"DetessCast"});

      check_model(int8_model_path(),
                  {
                      {RouteRegionKind::FanoutMap,
                       pipeline_internal::sima::RouteGraphKernelKind::DetessDequant},
                  },
                  {6U}, {SessionPostStageOp::DetessDequant}, {"DetessDequant"});

      check_model(
          bf16_mlatess_model_path(),
          {
              {RouteRegionKind::FanoutMap, pipeline_internal::sima::RouteGraphKernelKind::Cast},
          },
          {6U}, {SessionPostStageOp::Cast}, {"Cast"});

      check_model(int8_mlatess_model_path(),
                  {
                      {RouteRegionKind::FanoutMap,
                       pipeline_internal::sima::RouteGraphKernelKind::Dequantize},
                  },
                  {6U}, {SessionPostStageOp::Dequantize}, {"Dequant"});

      const auto check_packed_processcvu_payload = [](const std::filesystem::path& model_path,
                                                      const std::string& expected_graph_family,
                                                      int expected_num_in_tensor) {
        if (!std::filesystem::exists(model_path)) {
          return;
        }

        Model::Options opt;
        opt.preprocess.kind = InputKind::Tensor;
        opt.preprocess.enable = AutoFlag::On;
        Model model(model_path.string(), opt);
        const auto& pack = ModelAccess::pack(model);
        const auto& mpk_opt = pack.mpk_contract();
        require(mpk_opt.has_value(), "packed routed-input regression requires MPK contract for " +
                                         model_path.filename().string());
        const auto published_inputs =
            pipeline_internal::sima::get_mla_published_outputs_contract(*mpk_opt);
        require(published_inputs.size() >= 6U,
                "packed routed-input regression expects six published upstream tensors for " +
                    model_path.filename().string());

        const auto post = ModelAccess::build_postprocess_nodes(model, false);
        pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
        const auto compiled = compile_node_contracts(post, ContractCompileInput{}, &diagnostics);
        require(diagnostics.errors.empty(), "post group should compile without diagnostics for " +
                                                model_path.filename().string());
        require(compiled.stages.size() == 1U,
                "packed post regression expects one materialized stage for " +
                    model_path.filename().string());
        require(compiled.stages.front().processcvu.has_value(),
                "packed post regression expects processcvu stage for " +
                    model_path.filename().string());
        const auto& payload = compiled.stages.front().processcvu->payload;
        require(payload.graph_family == expected_graph_family,
                "unexpected packed post graph family for " + model_path.filename().string());
        require(payload.default_output_names.size() ==
                    static_cast<std::size_t>(expected_num_in_tensor),
                "packed post regression should preserve every named output view for " +
                    model_path.filename().string());
        const auto& runtime = compiled.stages.front().processcvu->runtime_contract;
        require(runtime.logical_outputs.size() ==
                        static_cast<std::size_t>(expected_num_in_tensor) &&
                    runtime.frame_arena_role == pipeline_internal::sima::FrameArenaRole::ReuseInput,
                "packed post regression must retain all outputs in the imported frame arena for " +
                    model_path.filename().string());
        require(payload.num_in_tensor == expected_num_in_tensor,
                "packed post regression should preserve semantic tensor count for " +
                    model_path.filename().string());
      };

      check_packed_processcvu_payload(int8_mlatess_model_path(), "dequantize", 6);
      if (std::filesystem::exists(int8_model_path())) {
        Model::Options opt;
        opt.preprocess.kind = InputKind::Tensor;
        opt.preprocess.enable = AutoFlag::On;
        Model model(int8_model_path().string(), opt);

        const auto post = ModelAccess::build_postprocess_nodes(model, false);
        pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
        const auto compiled = compile_node_contracts(post, ContractCompileInput{}, &diagnostics);
        require(diagnostics.errors.empty(),
                "INT8 EV74 post group should compile without diagnostics");
        require(compiled.stages.size() == 1U,
                "INT8 EV74 post group should materialize fused detessdequant stage");
        require(compiled.stages.front().processcvu.has_value(),
                "INT8 EV74 post group should use processcvu detessdequant stage");
        require(compiled.stages.front().processcvu->payload.graph_family == "detessdequant",
                "INT8 EV74 post group should preserve fused detessdequant graph family");
        require(compiled.stages.front().processcvu->payload.num_in_tensor == 6,
                "INT8 EV74 detessdequant stage should preserve six routed heads");
      }

      if (std::filesystem::exists(bf16_int8_model_path())) {
        Model::Options opt;
        opt.preprocess.kind = InputKind::Tensor;
        opt.preprocess.enable = AutoFlag::On;
        Model model(bf16_int8_model_path().string(), opt);

        const auto post = ModelAccess::build_postprocess_nodes(model, false);
        pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
        const auto compiled = compile_node_contracts(post, ContractCompileInput{}, &diagnostics);
        require(diagnostics.errors.empty(), "BF16 post group should compile without diagnostics");
        require(!compiled.stages.empty(),
                "BF16 post group should materialize at least one compiled stage");
        const auto* runtime = compiled_runtime_contract_from_stage(&compiled.stages.back());
        require(runtime != nullptr, "BF16 post group should expose a final runtime contract");
        require(runtime->logical_outputs.size() == 6U,
                "BF16 post group should preserve six output heads");

        const std::vector<std::vector<std::int64_t>> expected_shapes = {
            {1, 80, 80, 64}, {1, 40, 40, 64}, {1, 20, 20, 64},
            {1, 80, 80, 80}, {1, 40, 40, 80}, {1, 20, 20, 80},
        };
        for (std::size_t i = 0; i < expected_shapes.size(); ++i) {
          require(runtime->logical_outputs[i].shape == expected_shapes[i],
                  "BF16 post group should preserve YOLO head channel geometry at output " +
                      std::to_string(i));
        }
      }

      const auto check_routed_input_addressing_matches_mla = [](const auto& runtime,
                                                                const ModelPack& pack,
                                                                const std::string& label,
                                                                const std::string& model_name) {
        const auto infer_facts = pack.stage_facts_for_model_stage(ModelStage::MlaOnly);
        const auto mla =
            std::find_if(infer_facts.rbegin(), infer_facts.rend(),
                         [](const auto& fact) { return fact.mla_compiled.has_value(); });
        require(mla != infer_facts.rend(), label + " requires admitted MLA output facts");
        const auto& upstream = mla->mla_compiled->runtime_contract;
        require(upstream.logical_outputs.size() == 6U && runtime.logical_inputs.size() == 6U &&
                    runtime.physical_inputs.size() == 6U && runtime.input_bindings.size() == 6U,
                label + " must retain six exact MLA-to-CVU views for " + model_name);
        require(runtime.frame_arena_role == pipeline_internal::sima::FrameArenaRole::ReuseInput &&
                    runtime.frame_arena_size_bytes == upstream.frame_arena_size_bytes,
                label + " must import the existing MLA frame arena for " + model_name);

        std::vector<std::int64_t> view_offsets;
        for (std::size_t i = 0; i < 6U; ++i) {
          const auto& published = upstream.logical_outputs[i];
          require(published.physical_index >= 0 &&
                      static_cast<std::size_t>(published.physical_index) <
                          upstream.physical_outputs.size(),
                  label + " MLA output must reference its physical carrier for " + model_name);
          const auto& carrier = upstream.physical_outputs[published.physical_index];
          const std::int64_t expected_offset = carrier.source_byte_offset + published.byte_offset;
          view_offsets.push_back(expected_offset);
          const auto& physical = runtime.physical_inputs[i];
          const auto& logical = runtime.logical_inputs[i];
          const auto& binding = runtime.input_bindings[i];
          require(logical.logical_name == published.logical_name &&
                      logical.shape == published.shape &&
                      logical.size_bytes == published.size_bytes,
                  label + " must preserve the exact upstream logical tensor for " + model_name);
          require(logical.physical_index == static_cast<int>(i) && logical.byte_offset == 0 &&
                      physical.physical_index == static_cast<int>(i) &&
                      physical.source_byte_offset == expected_offset,
                  label + " must preserve the independently composed MLA arena view offset for " +
                      model_name);
          require(!physical.segment_name.empty() && logical.segment_name == physical.segment_name &&
                      binding.source_segment_name == physical.segment_name,
                  label + " local descriptor and route must address the same view for " +
                      model_name);
          require(binding.local_logical_input_index == static_cast<int>(i) &&
                      binding.sink_pad_index == 0 &&
                      binding.src_physical_output_index == static_cast<int>(i) &&
                      binding.src_physical_byte_offset == 0 &&
                      binding.src_physical_size_bytes == physical.size_bytes,
                  label + " must select each local view on the one arena input pad for " +
                      model_name);
        }
        std::sort(view_offsets.begin(), view_offsets.end());
        require(std::adjacent_find(view_offsets.begin(), view_offsets.end()) == view_offsets.end(),
                label + " must retain six distinct arena view addresses for " + model_name);
      };

      const auto check_routed_input_addressing = [&](const std::filesystem::path& model_path,
                                                     const std::string& expected_graph_family) {
        if (!std::filesystem::exists(model_path)) {
          return;
        }

        Model::Options opt;
        opt.preprocess.kind = InputKind::Tensor;
        opt.preprocess.enable = AutoFlag::On;
        Model model(model_path.string(), opt);
        const auto& pack = ModelAccess::pack(model);
        const auto& mpk_opt = pack.mpk_contract();
        require(mpk_opt.has_value(), "packed routed-input regression requires MPK contract for " +
                                         model_path.filename().string());
        const auto published_inputs =
            pipeline_internal::sima::get_mla_published_outputs_contract(*mpk_opt);
        require(published_inputs.size() >= 6U,
                "packed routed-input regression expects six published upstream tensors for " +
                    model_path.filename().string());

        const auto post = ModelAccess::build_postprocess_nodes(model, false);
        pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
        const auto compiled = compile_node_contracts(post, ContractCompileInput{}, &diagnostics);
        require(diagnostics.errors.empty(), "post group should compile without diagnostics for " +
                                                model_path.filename().string());
        require(compiled.stages.size() == 1U,
                "packed routed-input regression expects one materialized stage for " +
                    model_path.filename().string());
        require(compiled.stages.front().processcvu.has_value(),
                "packed routed-input regression expects processcvu stage for " +
                    model_path.filename().string());
        const auto& stage = *compiled.stages.front().processcvu;
        require(stage.payload.graph_family == expected_graph_family,
                "unexpected packed routed-input graph family for " +
                    model_path.filename().string());
        check_routed_input_addressing_matches_mla(stage.runtime_contract, pack,
                                                  "routed-input regression",
                                                  model_path.filename().string());
      };

      check_routed_input_addressing(int8_mlatess_model_path(), "dequantize");
      if (std::filesystem::exists(int8_model_path())) {
        Model::Options opt;
        opt.preprocess.kind = InputKind::Tensor;
        opt.preprocess.enable = AutoFlag::On;
        Model model(int8_model_path().string(), opt);
        const auto& pack = ModelAccess::pack(model);
        const auto& mpk_opt = pack.mpk_contract();
        require(mpk_opt.has_value(),
                "INT8 EV74 packed routed-input regression requires MPK contract");
        const auto published_inputs =
            pipeline_internal::sima::get_mla_published_outputs_contract(*mpk_opt);
        require(published_inputs.size() >= 6U,
                "INT8 EV74 packed routed-input regression expects six published upstream tensors");

        const auto post = ModelAccess::build_postprocess_nodes(model, false);
        pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
        const auto compiled = compile_node_contracts(post, ContractCompileInput{}, &diagnostics);
        require(diagnostics.errors.empty(),
                "INT8 EV74 post group should compile without diagnostics");
        require(compiled.stages.size() == 1U,
                "INT8 EV74 packed routed-input regression expects fused detessdequant post stage");
        require(compiled.stages.front().processcvu.has_value(),
                "INT8 EV74 packed routed-input regression expects detessdequant processcvu stage");
        const auto& detess_stage = *compiled.stages.front().processcvu;
        require(
            detess_stage.payload.graph_family == "detessdequant",
            "INT8 EV74 packed routed-input regression should preserve fused detessdequant family");
        require(detess_stage.runtime_contract.logical_inputs.size() == 6U,
                "INT8 EV74 detessdequant stage should preserve six logical inputs");
        require(detess_stage.runtime_contract.input_bindings.size() == 6U,
                "INT8 EV74 detessdequant stage should preserve six input bindings");
        check_routed_input_addressing_matches_mla(detess_stage.runtime_contract, pack,
                                                  "INT8 EV74 detessdequant",
                                                  int8_model_path().filename().string());
      }

      const auto check_model_managed_dequant_routed_input_addressing = [&](const std::filesystem::
                                                                               path& model_path) {
        if (!std::filesystem::exists(model_path)) {
          return;
        }

        Model::Options opt;
        opt.preprocess.kind = InputKind::Tensor;
        opt.preprocess.enable = AutoFlag::On;
        Model model(model_path.string(), opt);

        const auto& pack = ModelAccess::pack(model);
        const auto& mpk_opt = pack.mpk_contract();
        require(mpk_opt.has_value(), "model-managed dequant regression requires MPK contract for " +
                                         model_path.filename().string());
        const auto published_inputs =
            pipeline_internal::sima::get_mla_published_outputs_contract(*mpk_opt);
        require(published_inputs.size() == 6U,
                "model-managed dequant regression expects six published MLA tensors for " +
                    model_path.filename().string());

        const auto post_plan = pack.execution_plan().post;
        const auto post_stage_facts =
            pack.stage_facts_for_model_stage(internal::ModelStage::Postprocess);
        require(post_plan.size() == post_stage_facts.size(),
                "model-managed dequant regression expects stage facts aligned with post plan for " +
                    model_path.filename().string());

        const internal::ModelFragment::StageFacts* dequant_stage_fact = nullptr;
        for (std::size_t i = 0; i < post_plan.size(); ++i) {
          if (post_plan[i].kind == internal::ExecutionStageKind::Dequant &&
              post_stage_facts[i].processcvu_contract.has_value()) {
            dequant_stage_fact = &post_stage_facts[i];
            break;
          }
        }
        require(dequant_stage_fact != nullptr,
                "model-managed dequant regression expects a processcvu dequant stage fact for " +
                    model_path.filename().string());
        require(dequant_stage_fact->processcvu_contract->payload.graph_family == "dequantize",
                "model-managed dequant regression expects dequantize processcvu family for " +
                    model_path.filename().string());
        check_routed_input_addressing_matches_mla(
            dequant_stage_fact->processcvu_contract->runtime_contract, pack, "stage fact",
            model_path.filename().string());

        const auto post = ModelAccess::build_postprocess_nodes(model, false);
        pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
        const auto compiled = compile_node_contracts(post, ContractCompileInput{}, &diagnostics);
        require(diagnostics.errors.empty(), "model-managed dequant regression expects post group "
                                            "to compile without diagnostics for " +
                                                model_path.filename().string());
        require(compiled.stages.size() == 1U,
                "model-managed dequant regression expects one compiled post stage for " +
                    model_path.filename().string());
        require(
            compiled.stages.front().processcvu.has_value(),
            "model-managed dequant regression expects a processcvu dequant compiled stage for " +
                model_path.filename().string());
        require(
            compiled.stages.front().processcvu->payload.graph_family == "dequantize",
            "model-managed dequant regression expects compiled dequantize processcvu family for " +
                model_path.filename().string());

        check_routed_input_addressing_matches_mla(
            compiled.stages.front().processcvu->runtime_contract, pack, "compiled stage",
            model_path.filename().string());
      };

      check_model_managed_dequant_routed_input_addressing(int8_mlatess_model_path());

      const auto check_preadapter_transport_vs_published_names =
          [](const std::filesystem::path& model_path, const std::string& expected_graph_family,
             const std::string& expected_transport_output_name) {
            if (!std::filesystem::exists(model_path)) {
              return;
            }

            Model::Options opt;
            opt.preprocess.kind = InputKind::Tensor;
            opt.preprocess.enable = AutoFlag::On;
            Model model(model_path.string(), opt);

            const auto quant_opt = ModelAccess::build_quant_stage_options(model, false);
            require(static_cast<bool>(quant_opt.compiled_contract),
                    "preadapter regression expects compiled quant contract for " +
                        model_path.filename().string());
            const auto& payload = quant_opt.compiled_contract->payload;
            require(payload.graph_family == expected_graph_family,
                    "unexpected preadapter graph family for " + model_path.filename().string());
            require(payload.default_output_names.size() == 1U,
                    "preadapter regression should preserve one transport output for " +
                        model_path.filename().string());
            require(payload.default_output_names.front() == expected_transport_output_name,
                    "preadapter regression should preserve the canonical transport output name");
            require(!payload.primary_output_name.empty(),
                    "preadapter regression requires an explicit published output for " +
                        model_path.filename().string());
            require(payload.primary_output_name == expected_transport_output_name,
                    "MLATess packed-parent preadapter should publish the transport parent name");
          };

      check_preadapter_transport_vs_published_names(int8_mlatess_model_path(), "quantize",
                                                    "output_tensor");
    }));

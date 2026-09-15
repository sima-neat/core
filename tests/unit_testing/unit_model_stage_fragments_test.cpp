#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "nodes/sima/DetessDequant.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "model_archive_fixture_utils.h"
#include "pipeline/Graph.h"
#include "test_main.h"
#include "test_utils.h"

namespace {

sima_test::ModelArchiveFixture make_stage_fixture(const std::string& tag) {
  return sima_test::make_strict_model_archive_fixture(tag,
                                                      {
                                                          {"etc/pipeline_sequence.json",
                                                           R"json({
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
        "name": "preproc_0",
        "pluginId": "processcvu",
        "configPath": "0_preproc.json",
        "processor": "CVU",
        "kernel": "preproc",
        "input": "decoder"
      },
      {
        "sequence_id": 2,
        "name": "mla_0",
        "pluginId": "processmla",
        "configPath": "0_process_mla.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "preproc_0"
      },
      {
        "sequence_id": 3,
        "name": "detessdequant_0",
        "pluginId": "processcvu",
        "configPath": "0_postproc.json",
        "processor": "CVU",
        "kernel": "detessdequant",
        "input": "mla_0"
      }
    ]
  }]
})json"},
                                                          {"etc/0_preproc.json",
                                                           R"json({
  "node_name": "preproc_0",
  "input_width": 1280,
  "input_height": 720,
  "input_img_type": "RGB",
  "output_width": 640,
  "output_height": 640,
  "output_img_type": "RGB"
})json"},
                                                          {"etc/0_process_mla.json",
                                                           R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "preproc_0"}],
  "data_type": ["INT8"],
  "output_width": [80],
  "output_height": [80],
  "output_depth": [6]
})json"},
                                                          {"etc/0_postproc.json",
                                                           R"json({
  "node_name": "detessdequant_0",
  "num_in_tensor": 1,
  "out_data_type": "FP32",
  "input_width": [80],
  "input_height": [80],
  "input_depth": [6]
})json"},
                                                      },
                                                      true, "yolo_v9c_seg",
                                                      /*include_executable_artifacts=*/true);
}

} // namespace

RUN_TEST(
    "unit_model_stage_fragments_test", ([] {
      using namespace simaai::neat;

      const auto fixture = make_stage_fixture("model_stage_fragments");
      Model model(fixture.tar_path);

      // Exercise the cold route before another helper prepares the model. A
      // copied pack must not independently replan MLA away from its pre/post arena.
      const auto cold_nodes = internal::ModelAccess::build_public_route_nodes(model, {});
      pipeline_internal::sima::ManifestBuildDiagnostics cold_diagnostics;
      const auto cold_contracts = compile_node_contracts(cold_nodes, {}, &cold_diagnostics);
      const auto cold_manifest =
          render_manifest_from_compiled_contracts(cold_contracts, {}, &cold_diagnostics);
      require(cold_manifest && cold_diagnostics.errors.empty() &&
                  cold_manifest->stages.size() == 3U,
              "cold public route must compile preproc, MLA and postprocess together");
      const auto arena_bytes = cold_manifest->stages.front().frame_arena_size_bytes;
      require(arena_bytes > 0U && cold_manifest->stages.front().frame_arena_role ==
                                      pipeline_internal::sima::FrameArenaRole::Allocate,
              "cold public route must allocate its admitted ingress arena");
      for (std::size_t i = 1; i < cold_manifest->stages.size(); ++i) {
        const auto& stage = cold_manifest->stages[i];
        require(stage.frame_arena_size_bytes == arena_bytes &&
                    stage.frame_arena_role == pipeline_internal::sima::FrameArenaRole::ReuseInput,
                "cold public route must retain one placement authority across all stages");
      }

      // Naming overrides create distinct graph vertices, not distinct model
      // sources. Contract planning must preserve each invocation's own arena.
      Model::RouteOptions first_route;
      first_route.name_suffix = "_first";
      Model::RouteOptions second_route;
      second_route.name_suffix = "_second";
      auto repeated_nodes = internal::ModelAccess::build_public_route_nodes(model, first_route);
      const auto second_nodes =
          internal::ModelAccess::build_public_route_nodes(model, second_route);
      repeated_nodes.insert(repeated_nodes.end(), second_nodes.begin(), second_nodes.end());
      pipeline_internal::sima::ManifestBuildDiagnostics repeated_diagnostics;
      const auto repeated_contracts =
          compile_node_contracts(repeated_nodes, {}, &repeated_diagnostics);
      require(repeated_diagnostics.errors.empty(),
              "separate model routes must not share command selection: " +
                  (repeated_diagnostics.errors.empty() ? std::string{}
                                                       : repeated_diagnostics.errors.front()));
      const auto repeated_manifest =
          render_manifest_from_compiled_contracts(repeated_contracts, {}, &repeated_diagnostics);
      require(repeated_manifest && repeated_diagnostics.errors.empty() &&
                  repeated_manifest->stages.size() == 2U * cold_manifest->stages.size(),
              "repeated routes must preserve every stage of both invocations");
      for (std::size_t i = 0; i < repeated_manifest->stages.size(); ++i) {
        const auto& stage = repeated_manifest->stages[i];
        const auto& original = cold_manifest->stages[i % cold_manifest->stages.size()];
        require(stage.frame_arena_role == original.frame_arena_role &&
                    stage.frame_arena_size_bytes == original.frame_arena_size_bytes,
                "each invocation must retain its admitted arena ownership and placement");
      }

      // Whole-region fragments also carry an invocation boundary when there
      // is no typed preprocess node between successive executions.
      const auto& pack = internal::ModelAccess::pack(model);
      for (const auto stage : {internal::ModelStage::MlaOnly, internal::ModelStage::Full}) {
        auto region_nodes = pack.to_nodes(stage);
        const auto next_region = pack.to_nodes(stage);
        region_nodes.insert(region_nodes.end(), next_region.begin(), next_region.end());
        pipeline_internal::sima::ManifestBuildDiagnostics region_diagnostics;
        const auto regions = compile_node_contracts(region_nodes, {}, &region_diagnostics);
        require(regions.fully_renderable && region_diagnostics.errors.empty() &&
                    regions.stages.size() == 2U,
                "repeating a whole region must compile as two independent invocations (stage=" +
                    std::to_string(static_cast<int>(stage)) + "): " +
                    (region_diagnostics.errors.empty() ? std::string("no diagnostic")
                                                       : region_diagnostics.errors.front()));
        const auto manifest =
            render_manifest_from_compiled_contracts(regions, {}, &region_diagnostics);
        require(manifest && region_diagnostics.errors.empty(),
                "repeated whole regions must retain renderable physical contracts");
        if (stage == internal::ModelStage::MlaOnly) {
          for (const auto& invocation : manifest->stages) {
            require(invocation.frame_arena_role ==
                            pipeline_internal::sima::FrameArenaRole::Allocate &&
                        !invocation.physical_inputs.empty() &&
                        invocation.physical_inputs.front().address_source ==
                            pipeline_internal::sima::PhysicalAddressSource::RuntimePhysicalBinding,
                    "each inference-only invocation must import its IFM, not reuse a prior arena");
          }
        }
      }

      // Advancing through the model's regions still describes one invocation.
      auto split_nodes = pack.to_nodes(internal::ModelStage::Preprocess);
      for (const auto stage : {internal::ModelStage::MlaOnly, internal::ModelStage::Postprocess}) {
        const auto region = pack.to_nodes(stage);
        split_nodes.insert(split_nodes.end(), region.begin(), region.end());
      }
      pipeline_internal::sima::ManifestBuildDiagnostics split_diagnostics;
      const auto split = compile_node_contracts(split_nodes, {}, &split_diagnostics);
      const auto split_manifest =
          render_manifest_from_compiled_contracts(split, {}, &split_diagnostics);
      require(split_manifest && split_diagnostics.errors.empty() &&
                  split_manifest->stages.size() == cold_manifest->stages.size(),
              "split regions of one invocation must retain one complete route");
      for (std::size_t i = 0; i < split_manifest->stages.size(); ++i) {
        require(split_manifest->stages[i].frame_arena_role ==
                        cold_manifest->stages[i].frame_arena_role &&
                    split_manifest->stages[i].frame_arena_size_bytes == arena_bytes,
                "split regions must preserve the full route's admitted placement");
      }

      Graph pre = model.preprocess();
      Graph infer = model.inference();
      Graph post = model.postprocess();
      Model::RouteOptions runnable_opt;
      runnable_opt.include_input = true;
      runnable_opt.include_output = true;
      Graph sess = model.graph(runnable_opt);
      Graph full_graph = model.graph();
      Graph direct_model_graph;
      direct_model_graph.add(model);

      require_contains(pre.describe_backend(false), "neatprocesscvu",
                       "Model::preprocess should produce a non-empty Graph fragment");
      require_contains(infer.describe_backend(false), "neatprocessmla",
                       "Model::inference should produce a non-empty Graph fragment");
      require_contains(post.describe_backend(false), "neatprocesscvu",
                       "Model::postprocess should expose Model boundary post stage");
      const std::string route_backend = sess.describe_backend(false);
      require_contains(route_backend, "appsrc",
                       "Model::graph({include_input=true}) should include appsrc");
      require_contains(route_backend, "appsink",
                       "Model::graph({include_output=true}) should include appsink");
      const std::string graph_backend = full_graph.describe_backend(false);
      require_contains(graph_backend, "neatprocessmla",
                       "Model::graph should expose the model route as a Graph fragment");
      require(graph_backend.find("appsrc") == std::string::npos,
              "Model::graph should not bake in a default appsrc boundary");
      require(graph_backend.find("appsink") == std::string::npos,
              "Model::graph should not bake in a default appsink boundary");
      const std::string direct_model_backend = direct_model_graph.describe_backend(false);
      require_contains(direct_model_backend, "neatprocessmla",
                       "Graph::add(Model) should splice the model route directly");
      require(direct_model_backend.find("appsrc") == std::string::npos,
              "Graph::add(Model) should not bake in a default appsrc boundary");
      require(direct_model_backend.find("appsink") == std::string::npos,
              "Graph::add(Model) should not bake in a default appsink boundary");

      const std::string infer_fragment = model.backend_fragment(Model::Stage::Inference);
      require_contains(infer_fragment, "neatprocessmla",
                       "Model::backend_fragment(inference) should include MLA plugin");
      require_contains(infer_fragment, "stage-id=",
                       "Model::backend_fragment(inference) should include stage metadata");

      const std::string full_fragment = model.backend_fragment(Model::Stage::Full);
      require_contains(full_fragment, "neatprocesscvu",
                       "Model::backend_fragment(full) should include CVU preproc plugin");
      require_contains(full_fragment, "neatprocessmla",
                       "Model::backend_fragment(full) should include MLA plugin");
      require_contains(full_fragment,
                       "stage-id=", "Model::backend_fragment(full) should include stage metadata");

      Graph infer_only = model.fragment(Model::Stage::Inference);
      require_contains(infer_only.describe_backend(false), "neatprocessmla",
                       "Model::fragment(inference) should produce non-empty Graph");

      // The public standalone post node carries the same immutable model
      // source as the inference fragment. Both must consume one contextual
      // arena, rather than keeping the post node's original full-model offsets.
      auto direct_nodes = internal::ModelAccess::build_public_inference_nodes(model);
      direct_nodes.push_back(std::make_shared<DetessDequant>(DetessDequantOptions(model)));
      ContractCompileInput compile_input;
      pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
      auto direct_contracts = compile_node_contracts(direct_nodes, compile_input, &diagnostics);
      auto direct_manifest =
          render_manifest_from_compiled_contracts(direct_contracts, compile_input, &diagnostics);
      require(direct_manifest && diagnostics.errors.empty(),
              "public MLA plus typed DetessDequant must compile one contextual route");
      namespace sima = pipeline_internal::sima;
      const sima::StageStaticSpec* mla_stage = nullptr;
      const sima::StageStaticSpec* post_stage = nullptr;
      for (const auto& stage : direct_manifest->stages) {
        if (stage.payload_kind == sima::StagePayloadKind::ProcessMla)
          mla_stage = &stage;
        if (stage.payload_kind == sima::StagePayloadKind::ProcessCvu)
          post_stage = &stage;
      }
      require(mla_stage && post_stage &&
                  mla_stage->frame_arena_role == sima::FrameArenaRole::Allocate &&
                  post_stage->frame_arena_role == sima::FrameArenaRole::ReuseInput &&
                  mla_stage->frame_arena_size_bytes == post_stage->frame_arena_size_bytes &&
                  !mla_stage->physical_inputs.empty() &&
                  mla_stage->physical_inputs.front().address_source ==
                      sima::PhysicalAddressSource::RuntimePhysicalBinding &&
                  !post_stage->physical_inputs.empty() &&
                  post_stage->physical_inputs.front().address_source ==
                      sima::PhysicalAddressSource::FrameArenaSpan,
              "standalone post must reuse the selected output arena, never the input-only IFM");

      // A duplicated typed continuation is not a new whole-region invocation.
      // Keep the exact-command guard rather than treating a collision as a fence.
      direct_nodes.push_back(std::make_shared<DetessDequant>(DetessDequantOptions(model)));
      pipeline_internal::sima::ManifestBuildDiagnostics duplicate_diagnostics;
      const auto duplicate = compile_node_contracts(direct_nodes, {}, &duplicate_diagnostics);
      require(!duplicate.fully_renderable && !duplicate_diagnostics.errors.empty(),
              "duplicate commands within an invocation must still be rejected");
      require_contains(duplicate_diagnostics.errors.front(), "physical command identity",
                       "duplicate continuation must retain the exact command diagnostic");

      const auto legacy = sima_test::make_model_archive_fixture(
          "model_stage_fragments_legacy_missing_mpk", {
                                                          {"etc/pipeline_sequence.json",
                                                           R"json({
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
        "name": "mla_0",
        "pluginId": "processmla",
        "configPath": "0_process_mla.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "decoder"
      }
    ]
  }]
})json"},
                                                          {"etc/0_process_mla.json",
                                                           R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "decoder"}]
})json"},
                                                      });
      bool threw = false;
      try {
        Model legacy_model(legacy.tar_path);
        (void)legacy_model.graph();
      } catch (const std::exception& e) {
        threw = true;
        require_contains(std::string(e.what()), "strict MPK contract required",
                         "legacy missing-mpk fixture should fail with strict contract error");
      }
      require(threw, "legacy missing-mpk fixture must fail under strict contract");
    }));

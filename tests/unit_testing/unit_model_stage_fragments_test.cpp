#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "nodes/sima/DetessDequant.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/graph/internal/GraphTestHooks.h"
#include "pipeline/graph/GraphDetail.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StageMemoryPolicy.h"
#include "pipeline/internal/sima/StageRecipePreparation.h"
#include "model_archive_fixture_utils.h"
#include "pipeline/Graph.h"
#include "test_main.h"
#include "test_utils.h"

#include <simaai/scheduler/client/frame_layout.h>

#include <algorithm>
#include <array>
#include <iterator>
#include <set>
#include <variant>

namespace {

namespace client = simaai::scheduler::client;
namespace si = simaai::neat::pipeline_internal::sima;

void finalize_storage(si::SimaPluginStaticManifest& manifest) {
  const auto failure = si::prepare_kernel_stage_recipes(&manifest, {}, {}, {}, {});
  require(!failure, failure ? failure->field + ": " + failure->message : "");
  si::finalize_pipeline_storage(manifest);
}

bool same_allocations(const client::FrameLayout& lhs, const client::FrameLayout& rhs) {
  return lhs.allocations.size() == rhs.allocations.size() &&
         std::equal(lhs.allocations.begin(), lhs.allocations.end(), rhs.allocations.begin(),
                    [](const auto& left, const auto& right) {
                      return left.size_bytes == right.size_bytes && left.policy == right.policy &&
                             left.required_device_access == right.required_device_access;
                    });
}

bool imports_model_input(const si::StageStaticSpec& stage) {
  if (!stage.frame_layout || stage.frame_layout->inputs.size() != 1U ||
      stage.physical_inputs.size() != 1U)
    return false;
  const auto* input = std::get_if<client::DirectMemory>(&stage.frame_layout->inputs.front().source);
  return input && input->pad == 0 && input->memory_index == 0 && input->byte_offset == 0 &&
         input->size_bytes == stage.physical_inputs.front().size_bytes;
}

// Each runner is configured through its typed context; legacy manifest/pool
// properties must not leak from mixed model fragments onto the new factory.
void require_runner_property_boundary(const std::string& pipeline) {
  std::size_t pos = 0;
  while ((pos = pipeline.find("neatkernelrunner ", pos)) != std::string::npos) {
    const std::size_t end = pipeline.find('!', pos);
    const std::string fragment = pipeline.substr(pos, end - pos);
    for (const char* property : {"stage-id=", "num-buffers=", "config=", "q-scale=", "q-zp="}) {
      require(fragment.find(property) == std::string::npos,
              "model runner fragment contains an unsupported property: " + std::string(property));
    }
    pos += fragment.size();
  }
  require(pipeline.find("neatprocesscvu") == std::string::npos,
          "model fragments must not fall back to the legacy execution factory");
}

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
      // copied pack must retain the selected pre/MLA/post allocation group.
      const auto cold_nodes = internal::ModelAccess::build_public_route_nodes(model, {});
      pipeline_internal::sima::ManifestBuildDiagnostics cold_diagnostics;
      const auto cold_contracts = compile_node_contracts(cold_nodes, {}, &cold_diagnostics);
      auto cold_manifest =
          render_manifest_from_compiled_contracts(cold_contracts, {}, &cold_diagnostics);
      require(cold_manifest && cold_diagnostics.errors.empty() &&
                  cold_manifest->stages.size() == 3U,
              "cold public route must compile preproc, MLA and postprocess together");
      finalize_storage(*cold_manifest);
      const auto& owner = cold_manifest->stages.front().frame_layout;
      require(owner && !owner->allocations.empty() &&
                  owner->ownership == client::FrameOwnership::Allocate,
              "cold public route must finalize its owned ingress group");
      for (std::size_t i = 1; i < cold_manifest->stages.size(); ++i) {
        const auto& layout = cold_manifest->stages[i].frame_layout;
        require(layout && layout->ownership == client::FrameOwnership::ReuseInput &&
                    same_allocations(*layout, *owner),
                "cold public route must retain the finalized shared allocation group");
      }

      // Fused compilation must retain each independently compiled model plan,
      // including its full preproc-owned group and the global stage identities.
      Model::Options live_options;
      live_options.preprocess.kind = InputKind::Image;
      live_options.preprocess.enable = AutoFlag::On;
      live_options.preprocess.color_convert.input_format = PreprocessColorFormat::NV12;
      Model live_model(fixture.tar_path, live_options);
      const auto live_nodes = internal::ModelAccess::build_public_route_nodes(live_model, {});
      GraphOptions live_graph_options;
      live_graph_options.processcvu.post_run_target = "A65";
      OutputSpec live_ingress;
      live_ingress.media_type = "video/x-raw";
      live_ingress.format = "NV12";
      live_ingress.width = 1280;
      live_ingress.height = 720;
      auto fused = session_test::compile_fused_realtime_contracts_for_test(
          live_nodes, {{std::make_shared<Input>(InputOptions{})}, live_nodes}, live_graph_options,
          live_ingress);
      require(fused.rendered_manifest && fused.manifest_diagnostics.errors.empty() &&
                  fused.rendered_manifest->model_storage_scopes,
              "fused compilation must preserve model storage ownership");
      auto& live_manifest = *fused.rendered_manifest;
      const auto live_preparation = pipeline_internal::sima::prepare_kernel_stage_recipes(
          &live_manifest, std::nullopt, {}, {}, {});
      require(!live_preparation,
              live_preparation ? live_preparation->field + ": " + live_preparation->message : "");
      pipeline_internal::sima::finalize_pipeline_storage(live_manifest);
      namespace client = simaai::scheduler::client;
      std::size_t planned_invocations = 0;
      std::set<std::size_t> rendered_indices;
      std::set<std::size_t> model_node_indices;
      for (std::size_t i = 0; i < live_manifest.stages.size(); ++i) {
        const auto& owner = live_manifest.stages[i];
        require(owner.compile_stage_index &&
                    rendered_indices.insert(*owner.compile_stage_index).second,
                "fused rendered stage identities must remain unique");
        if (!owner.native_kernel || !owner.native_kernel->frame_adapter_factory)
          continue;
        require(i + 1 < live_manifest.stages.size() && owner.frame_layout &&
                    owner.frame_layout->ownership == client::FrameOwnership::Allocate &&
                    owner.compile_node_index &&
                    model_node_indices.insert(*owner.compile_node_index).second,
                "each fused model invocation must keep its own planned owner");
        const auto& mla = live_manifest.stages[i + 1];
        require(mla.payload_kind == pipeline_internal::sima::StagePayloadKind::ProcessMla &&
                    mla.frame_layout &&
                    mla.frame_layout->ownership == client::FrameOwnership::ReuseInput,
                "fused MLA must reuse the preprocess-owned allocation group: owner=" +
                    owner.element_name + ", next=" + mla.element_name + ", ownership=" +
                    std::to_string(mla.frame_layout ? static_cast<int>(mla.frame_layout->ownership)
                                                    : -1));
        const auto& allocations = owner.frame_layout->allocations;
        const auto& mla_layout = *mla.frame_layout;
        require(allocations.size() == 2 && allocations.size() == mla_layout.allocations.size(),
                "fused preproc must own the complete CMA/DMS group reused by MLA");
        for (std::size_t allocation = 0; allocation < allocations.size(); ++allocation) {
          require(allocations[allocation].size_bytes ==
                          mla_layout.allocations[allocation].size_bytes &&
                      allocations[allocation].policy == mla_layout.allocations[allocation].policy &&
                      allocations[allocation].required_device_access ==
                          mla_layout.allocations[allocation].required_device_access,
                  "fused producer and reuser must agree on allocations in index order");
        }
        const auto& ifm = std::get<client::KnownFrameSpan>(mla_layout.inputs.front().source).span;
        const auto& ofm = mla_layout.outputs.front().span;
        for (const auto& span : {ifm, ofm}) {
          require(span.allocation_index < allocations.size() &&
                      span.byte_offset <= allocations[span.allocation_index].size_bytes &&
                      span.size_bytes <=
                          allocations[span.allocation_index].size_bytes - span.byte_offset,
                  "fused MLA spans must fit the producer's actual allocation group");
        }
        ++planned_invocations;
      }
      require(planned_invocations == 2,
              "fused consumer and model-bearing branch must retain independent storage plans");

      // Custom preprocessing owns its own output group. A named inference
      // fragment must import that IFM and allocate its selected outputs rather
      // than keeping the original full-model ReuseInput plan.
      Model::Options custom_options;
      custom_options.name_suffix = "_chapter";
      custom_options.preprocess.kind = InputKind::Image;
      custom_options.preprocess.color_convert.input_format = PreprocessColorFormat::BGR;
      custom_options.preprocess.input_max_width = 640;
      custom_options.preprocess.input_max_height = 640;
      custom_options.preprocess.input_max_depth = 3;
      custom_options.preprocess.normalize.enable = AutoFlag::On;
      custom_options.preprocess.normalize.mean = std::array<float, 3>{0.485f, 0.456f, 0.406f};
      custom_options.preprocess.normalize.stddev = std::array<float, 3>{0.229f, 0.224f, 0.225f};
      Model custom_model(fixture.tar_path, custom_options);
      auto preprocess_options = custom_options;
      preprocess_options.name_suffix = "_custom_preproc";
      Model preprocess_model(fixture.tar_path, preprocess_options);
      auto custom_nodes = internal::ModelAccess::build_public_preprocess_nodes(preprocess_model);
      const auto custom_inference =
          internal::ModelAccess::build_public_inference_nodes(custom_model);
      const auto custom_post = internal::ModelAccess::build_public_postprocess_nodes(custom_model);
      custom_nodes.insert(custom_nodes.end(), custom_inference.begin(), custom_inference.end());
      custom_nodes.insert(custom_nodes.end(), custom_post.begin(), custom_post.end());
      pipeline_internal::sima::ManifestBuildDiagnostics custom_diagnostics;
      const auto custom_contracts = compile_node_contracts(custom_nodes, {}, &custom_diagnostics);
      auto custom_manifest =
          render_manifest_from_compiled_contracts(custom_contracts, {}, &custom_diagnostics);
      require(custom_manifest && custom_diagnostics.errors.empty(),
              "custom named model route must compile");
      const auto custom_preparation = pipeline_internal::sima::prepare_kernel_stage_recipes(
          &*custom_manifest, std::nullopt, {}, {}, {});
      require(!custom_preparation,
              custom_preparation ? custom_preparation->field + ": " + custom_preparation->message
                                 : "");
      pipeline_internal::sima::finalize_pipeline_storage(*custom_manifest);
      const auto custom_mla = std::find_if(
          custom_manifest->stages.begin(), custom_manifest->stages.end(), [](const auto& stage) {
            return stage.payload_kind == pipeline_internal::sima::StagePayloadKind::ProcessMla;
          });
      require(custom_mla != custom_manifest->stages.end() && custom_mla->frame_layout &&
                  !custom_mla->physical_inputs.empty() && !custom_mla->frame_layout->inputs.empty(),
              "custom named inference must finalize its IFM and output layout");
      require(custom_mla->frame_layout->ownership == client::FrameOwnership::Allocate,
              "custom named inference must allocate a fresh output group");
      require(custom_mla != custom_manifest->stages.begin(),
              "custom named inference must retain its preceding preprocessor");
      const auto& custom_preproc = *std::prev(custom_mla);
      require(custom_preproc.frame_layout && !custom_preproc.frame_layout->outputs.empty(),
              "custom preprocessor must publish its owned IFM span");
      const auto* custom_ifm =
          std::get_if<client::KnownFrameSpan>(&custom_mla->frame_layout->inputs.front().source);
      const auto& preproc_span = custom_preproc.frame_layout->outputs.front().span;
      require(custom_mla->physical_inputs.front().address_source ==
                      pipeline_internal::sima::PhysicalAddressSource::FrameSpan &&
                  custom_ifm && custom_ifm->pad == 0 &&
                  custom_ifm->span.allocation_index == preproc_span.allocation_index &&
                  custom_ifm->span.byte_offset == preproc_span.byte_offset &&
                  custom_ifm->span.size_bytes == preproc_span.size_bytes,
              "custom named inference must consume the preprocessor's exact incoming span");

      // Naming overrides create distinct graph vertices, not distinct model
      // sources. Contract planning must preserve each invocation's own allocation group.
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
      auto repeated_manifest =
          render_manifest_from_compiled_contracts(repeated_contracts, {}, &repeated_diagnostics);
      require(repeated_manifest && repeated_diagnostics.errors.empty() &&
                  repeated_manifest->stages.size() == 2U * cold_manifest->stages.size(),
              "repeated routes must preserve every stage of both invocations");
      finalize_storage(*repeated_manifest);
      for (std::size_t i = 0; i < repeated_manifest->stages.size(); ++i) {
        const auto& stage = repeated_manifest->stages[i];
        const auto& original = cold_manifest->stages[i % cold_manifest->stages.size()];
        require(
            stage.frame_layout && original.frame_layout &&
                stage.frame_layout != original.frame_layout &&
                stage.frame_layout->ownership == original.frame_layout->ownership &&
                same_allocations(*stage.frame_layout, *original.frame_layout),
            "independent invocations must finalize separate layouts with the same requirements");
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
        auto manifest = render_manifest_from_compiled_contracts(regions, {}, &region_diagnostics);
        require(manifest && region_diagnostics.errors.empty(),
                "repeated whole regions must retain renderable physical contracts");
        if (stage == internal::ModelStage::MlaOnly) {
          pipeline_internal::sima::finalize_pipeline_storage(*manifest);
          for (const auto& invocation : manifest->stages) {
            require(invocation.frame_layout &&
                        invocation.frame_layout->ownership == client::FrameOwnership::Allocate &&
                        imports_model_input(invocation),
                    "each inference-only invocation must import its IFM, not reuse a prior "
                    "allocation group");
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
      auto split_manifest = render_manifest_from_compiled_contracts(split, {}, &split_diagnostics);
      require(split_manifest && split_diagnostics.errors.empty() &&
                  split_manifest->stages.size() == cold_manifest->stages.size(),
              "split regions of one invocation must retain one complete route");
      finalize_storage(*split_manifest);
      for (std::size_t i = 0; i < split_manifest->stages.size(); ++i) {
        const auto& split_layout = split_manifest->stages[i].frame_layout;
        const auto& complete_layout = cold_manifest->stages[i].frame_layout;
        require(split_layout && complete_layout &&
                    split_layout->ownership == complete_layout->ownership &&
                    same_allocations(*split_layout, *complete_layout),
                "split regions must finalize the same ownership and allocation requirements");
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

      require_contains(pre.describe_backend(false), "neatkernelrunner",
                       "Model::preprocess should produce a non-empty Graph fragment");
      require_contains(infer.describe_backend(false), "neatprocessmla",
                       "Model::inference should produce a non-empty Graph fragment");
      require_contains(post.describe_backend(false), "neatkernelrunner",
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
      require_runner_property_boundary(full_fragment);
      require_contains(full_fragment, "neatkernelrunner",
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
      // output group, rather than keeping the post node's original full-model offsets.
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
      const auto preparation =
          sima::prepare_kernel_stage_recipes(&*direct_manifest, std::nullopt, {}, {}, {});
      require(!preparation, preparation ? preparation->field + ": " + preparation->message : "");
      sima::finalize_pipeline_storage(*direct_manifest);
      const sima::StageStaticSpec* mla_stage = nullptr;
      const sima::StageStaticSpec* post_stage = nullptr;
      for (const auto& stage : direct_manifest->stages) {
        if (stage.payload_kind == sima::StagePayloadKind::ProcessMla)
          mla_stage = &stage;
        if (stage.payload_kind == sima::StagePayloadKind::ProcessCvu)
          post_stage = &stage;
      }
      require(mla_stage && post_stage && mla_stage->frame_layout && post_stage->frame_layout &&
                  mla_stage->frame_layout->ownership == client::FrameOwnership::Allocate &&
                  post_stage->frame_layout->ownership == client::FrameOwnership::ReuseInput &&
                  same_allocations(*mla_stage->frame_layout, *post_stage->frame_layout) &&
                  imports_model_input(*mla_stage) && !post_stage->physical_inputs.empty() &&
                  post_stage->physical_inputs.front().address_source ==
                      sima::PhysicalAddressSource::FrameSpan,
              "standalone post must reuse the selected output group, never the input-only IFM");

      // A duplicated typed continuation is not a new whole-region invocation.
      // Distinct element names must not hide the duplicated model command.
      DetessDequantOptions duplicate_options(model);
      duplicate_options.element_name = "duplicate_post";
      direct_nodes.push_back(std::make_shared<DetessDequant>(duplicate_options));
      pipeline_internal::sima::ManifestBuildDiagnostics duplicate_diagnostics;
      const auto duplicate = compile_node_contracts(direct_nodes, {}, &duplicate_diagnostics);
      require(!duplicate.fully_renderable &&
                  std::any_of(
                      duplicate_diagnostics.errors.begin(), duplicate_diagnostics.errors.end(),
                      [](const auto& error) {
                        return error.find("selected route repeats an invalid command identity") !=
                               std::string::npos;
                      }),
              "semantic selection must reject duplicate commands within one invocation: " +
                  (duplicate_diagnostics.errors.empty() ? std::string("no diagnostic")
                                                        : duplicate_diagnostics.errors.front()));

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

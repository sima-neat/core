#include <simaai/scheduler/client/frame_layout.h>
#include "asset_utils.h"
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "model_archive_fixture_utils.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/BoxDecodeType.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/contract/CompiledNodeContractQuery.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StageMemoryPolicy.h"
#include "pipeline/internal/TerminalOutputContractQuery.h"
#include "pipeline/runtime/ExecutionPlanAssembly.h"
#include "test_main.h"
#include "support/cold_contract_gst_init.h"

#include <algorithm>
#include <filesystem>
#include <optional>

namespace {

sima_test::ModelArchiveFixture make_decoded_yolo_fixture() {
  using nlohmann::json;
  json shapes = json::array(), bf16_heads = json::array(), fp32_heads = json::array();
  std::size_t total_bytes = 0;
  for (int channels : {4, 80}) {
    for (int side : {80, 40, 20}) {
      const auto bytes = side * side * channels * 2;
      const auto name =
          std::string(channels == 4 ? "bbox_" : "class_prob_") + std::to_string(shapes.size() % 3);
      shapes.push_back({1, side, side, channels});
      bf16_heads.push_back({{"name", name + "_bf16"}, {"type", "buffer"}, {"size", bytes}});
      fp32_heads.push_back({{"name", name}, {"type", "buffer"}, {"size", bytes * 2}});
      total_bytes += bytes;
    }
  }
  auto mpk = json::parse(R"json({
    "name": "decoded_yolo", "model_sdk_version": "3.0.0",
    "input_nodes": [{"name": "input", "type": "buffer", "size": 4915200}],
    "plugins": [
      {"name": "cast_input", "processor": "EV74", "sequence": 1, "type": "sgpProcess",
       "config_params": {"kernel": "cast_transform", "params": {
         "in_dtype": "float32", "out_dtype": "bfloat16",
         "input_shapes": [[1,640,640,3]], "output_shapes": [[1,640,640,3]]}},
       "input_nodes": [{"name": "input", "size": 4915200}],
       "output_nodes": [{"name": "cast_input", "type": "buffer", "size": 2457600}]},
      {"name": "MLA_0", "processor": "MLA", "sequence": 2, "type": "sgpProcess",
       "config_params": {"number_of_quads_to_user": 4},
       "input_nodes": [{"name": "cast_input", "size": 2457600}],
       "output_nodes": [{"name": "MLA_0", "type": "buffer", "size": 1411200}],
       "resources": {"executable": "model.elf"}},
      {"name": "unpack", "processor": "EV74", "sequence": 3, "type": "sgpProcess",
       "config_params": {"kernel": "unpack_transform", "params": {}},
       "input_nodes": [{"name": "MLA_0", "size": 1411200}]}
    ]
  })json");
  auto& plugins = mpk["plugins"];
  plugins[1]["output_nodes"][0]["size"] = total_bytes;
  plugins[2]["input_nodes"][0]["size"] = total_bytes;
  plugins[2]["output_nodes"] = bf16_heads;
  plugins[2]["config_params"]["params"] = {
      {"tensor_types", std::vector<std::string>(6, "bfloat16")},
      {"tensor_shapes", shapes},
      {"input_shapes", {{1, total_bytes}}},
      {"output_shapes", shapes}};
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    auto cast = plugins.front();
    cast["name"] = "cast_" + std::to_string(i);
    cast["sequence"] = 4 + i;
    cast["input_nodes"] = {bf16_heads[i]};
    cast["output_nodes"] = {fp32_heads[i]};
    cast["config_params"]["params"] = {{"in_dtype", "bfloat16"},
                                       {"out_dtype", "float32"},
                                       {"input_shapes", {shapes[i]}},
                                       {"output_shapes", {shapes[i]}}};
    plugins.push_back(std::move(cast));
  }
  auto published = fp32_heads;
  for (std::size_t i = 0; i < published.size(); ++i)
    published[i]["name"] = "pass_through_out_" + std::to_string(i);
  plugins.push_back({{"name", "PassThrough"},
                     {"processor", "EV74"},
                     {"sequence", 10},
                     {"type", "sgpProcess"},
                     {"config_params", {{"kernel", "pass_through"}, {"params", json::object()}}},
                     {"input_nodes", fp32_heads},
                     {"output_nodes", published}});
  for (auto& plugin : plugins) {
    plugin["config_params"]["desired_batch_size"] = 1;
    plugin["config_params"]["actual_batch_size"] = 1;
  }
  const auto fixture = sima_test::make_model_archive_fixture(
      "decoded_yolo", {{"etc/decoded_mpk.json", mpk.dump()}}, false);
  std::filesystem::create_directories(std::filesystem::path(fixture.root_dir) / "share");
  sima_test::write_topology_elf(std::filesystem::path(fixture.root_dir) / "share/model.elf",
                                "data.ifm.b0", 640 * 640 * 3 * 2, "data.ofm.b0", total_bytes);
  const auto archive = "tar -czf " + sima_test::model_archive_shell_quote(fixture.tar_path) +
                       " -C " + sima_test::model_archive_shell_quote(fixture.root_dir) + " .";
  require(std::system(archive.c_str()) == 0, "failed to archive decoded YOLO topology fixture");
  return fixture;
}

sima_test::ModelArchiveFixture make_fixture() {
  return sima_test::make_strict_model_archive_fixture("boxdecode_render_manifest_model",
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
        "name": "boxdecode_0",
        "pluginId": "processcvu",
        "configPath": "0_boxdecoder.json",
        "processor": "CVU",
        "kernel": "boxdecode",
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
  "output_depth": [84],
  "q_scale": [0.125],
  "q_zp": [-7]
})json"},
                                                          {"etc/0_boxdecoder.json",
                                                           R"json({
  "node_name": "boxdecode_0",
  "decode_type": "yolov8",
  "topk": 100,
  "detection_threshold": 0.25,
  "nms_iou_threshold": 0.45,
  "input_width": [80],
  "input_height": [80],
  "input_depth": [84],
  "slice_width": [80],
  "slice_height": [80],
  "slice_depth": [84],
  "data_type": ["INT8"],
  "dq_scale": [0.5],
  "dq_zp": [1]
})json"},
                                                      },
                                                      true, "yolo_v9c_seg",
                                                      /*include_topology_artifacts=*/true);
}

const simaai::neat::CompiledNodeContract*
find_first_mla_stage(const simaai::neat::CompiledPipelineContracts& compiled) {
  const simaai::neat::CompiledNodeContract* found = nullptr;
  const auto visit = [&](const auto& self,
                         const simaai::neat::CompiledNodeContract& stage) -> void {
    if (!found && stage.processmla.has_value()) {
      found = &stage;
    }
    for (const auto& child : stage.child_stages) {
      self(self, child);
    }
  };
  for (const auto& stage : compiled.stages) {
    visit(visit, stage);
  }
  return found;
}

const simaai::neat::CompiledNodeContract*
find_first_boxdecode_stage(const simaai::neat::CompiledPipelineContracts& compiled) {
  const simaai::neat::CompiledNodeContract* found = nullptr;
  const auto visit = [&](const auto& self,
                         const simaai::neat::CompiledNodeContract& stage) -> void {
    if (!found && stage.node_kind == "SimaBoxDecode") {
      found = &stage;
    }
    for (const auto& child : stage.child_stages) {
      self(self, child);
    }
  };
  for (const auto& stage : compiled.stages) {
    visit(visit, stage);
  }
  return found;
}

const simaai::neat::pipeline_internal::sima::StageStaticSpec* find_first_rendered_boxdecode_stage(
    const simaai::neat::pipeline_internal::sima::SimaPluginStaticManifest& manifest,
    const std::string& element_name) {
  for (const auto& stage : manifest.stages) {
    if (stage.element_name == element_name) {
      return &stage;
    }
  }
  return nullptr;
}

} // namespace

RUN_TEST(
    "unit_boxdecode_render_manifest_from_model_test", ([] {
      using namespace simaai::neat;
      sima_test::initialize_cold_contract_gstreamer();

      {
        const auto decoded = make_decoded_yolo_fixture();
        Model::Options options;
        options.decode_type = BoxDecodeType::Yolo;
        options.decode_type_option = BoxDecodeTypeOption::GroupedByRoleProbability;
        options.num_classes = 80;
        Model model(decoded.tar_path, options);
        for (bool sync : {false, true}) {
          const auto contract = internal::ModelAccess::build_boxdecode_stage_contract(model, sync);
          require(contract.payload.decode_type == BoxDecodeType::Yolo &&
                      contract.payload.score_activation ==
                          pipeline_internal::sima::BoxDecodeScoreActivation::Identity &&
                      contract.payload.num_classes == 80 &&
                      contract.runtime_contract.logical_inputs.size() == 6,
                  "decoded YOLO model heads must retain pixel coordinates and probability scores");
        }
      }

      const auto fixture = make_fixture();
      const std::string tar_path = fixture.tar_path;

      Model::Options base_opt;
      Model default_model(tar_path, base_opt);
      sima_test::require_exact_yolo_v9c_seg_parsed_contract(
          internal::ModelAccess::pack(default_model));
      require(!default_model.resolved_preprocess_plan().enabled,
              "tensor/infer-only render fixture must keep user preprocess disabled");
      require(!internal::ModelAccess::has_model_managed_stage(default_model,
                                                              internal::StageNodeKind::BoxDecode),
              "default Model route must not auto-select BoxDecode from inferred MPK topology");
      require(internal::ModelAccess::resolved_post_kind(default_model) !=
                  internal::PostRouteStageKind::BoxDecode,
              "default Model route should expose raw tensor postprocess, not BoxDecode");

      Model::Options mismatched_opt = base_opt;
      mismatched_opt.decode_type = BoxDecodeType::YoloV8;
      Model mismatched_model(tar_path, mismatched_opt);
      bool boxdecode_mismatch_rejected = false;
      try {
        (void)simaai::neat::nodes::SimaBoxDecode(mismatched_model, BoxDecodeType::YoloV8, 0.25,
                                                 0.45, 100);
      } catch (const std::exception&) {
        boxdecode_mismatch_rejected = true;
      }
      require(boxdecode_mismatch_rejected,
              "explicit BoxDecode must reject a detection decoder for a segmentation MPK contract");

      Model::Options model_opt = base_opt;
      model_opt.decode_type = BoxDecodeType::YoloV8Seg;
      Model model(tar_path, model_opt);

      // A non-SSD model route with external preprocessing has no Preproc element that can emit
      // preproc metadata. Complete dimensions plus the explicit resize policy are the established
      // provenance contract for this route and must discharge that bucket.
      Model::Options external_opt = model_opt;
      Model external_model(tar_path, external_opt);
      auto external_node = simaai::neat::nodes::SimaBoxDecode(
          external_model, BoxDecodeType::YoloV8Seg, 0.25, 0.45, 100, "", std::nullopt, std::nullopt,
          1280, 720, 640, 640, ResizeMode::Letterbox);
      const auto* external_box = dynamic_cast<const SimaBoxDecode*>(external_node.get());
      require(external_box != nullptr,
              "external-preprocessing boxdecode should return a concrete node");
      const auto external_req = external_box->preprocess_meta_requirement();
      require(!external_req.has_value() || external_req->required_fields.empty(),
              "complete external YOLO preprocessing provenance must not require metadata from an "
              "absent Preproc element");

      auto nodes = internal::ModelAccess::build_public_inference_nodes(model);
      require(!nodes.empty(), "inference fragment should contain renderable nodes");

      nodes.push_back(
          simaai::neat::nodes::SimaBoxDecode(model, BoxDecodeType::YoloV8Seg, 0.25, 0.45, 100));

      pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
      const auto compiled = compile_node_contracts(nodes, ContractCompileInput{}, &diagnostics);
      require(diagnostics.errors.empty(), "inference + boxdecode contract compile failed");

      const auto* mla_stage = find_first_mla_stage(compiled);
      require(mla_stage != nullptr, "compiled inference fragment should include an MLA stage");

      const auto* boxdecode_stage = find_first_boxdecode_stage(compiled);
      require(boxdecode_stage != nullptr,
              "compiled inference fragment should include a boxdecode stage");
      require(boxdecode_stage->renderable,
              "compiled boxdecode stage should remain renderable for manifest export");

      const auto manifest_opt =
          render_manifest_from_compiled_contracts(compiled, ContractCompileInput{}, &diagnostics);
      require(diagnostics.errors.empty(), "rendered manifest should not emit diagnostics errors");
      require(manifest_opt.has_value(), "compiled inference fragment should render a manifest");

      const auto* rendered_box =
          find_first_rendered_boxdecode_stage(*manifest_opt, boxdecode_stage->element_name);
      require(rendered_box != nullptr, "rendered manifest should contain a boxdecode stage");
      pipeline_internal::SegmentBuildPlan build;
      build.rendered_manifest = *manifest_opt;
      for (std::size_t i = 0; i < nodes.size(); ++i) {
        NodeReport report;
        report.index = static_cast<int>(i);
        report.kind = nodes[i]->kind();
        report.elements = nodes[i]->element_names(static_cast<int>(i));
        build.node_reports.push_back(std::move(report));
      }
      const auto plan = nlohmann::json::parse(
          internal::ExecutionPlanAccess::make(build, model.info().model_name).to_json());
      const auto box =
          std::find_if(plan.at("nodes").begin(), plan.at("nodes").end(),
                       [](const auto& node) { return node.at("kind") == "SimaBoxDecode"; });
      require(box != plan.at("nodes").end() && box->contains("settings"),
              "execution-plan JSON must retain native BoxDecode settings");
      require(box->at("settings") == nlohmann::json{{"decode_type", "yolov8-seg"},
                                                    {"num_classes", 80},
                                                    {"detection_threshold", 0.25},
                                                    {"nms_iou_threshold", 0.45},
                                                    {"top_k", 100}},
              "execution-plan JSON must report the resolved BoxDecode settings exactly");
      {
        // Exercise omission in the report serializer without changing numerical requests.
        auto defaults = build;
        auto& stage = defaults.rendered_manifest->stages.back();
        auto recipe =
            std::make_shared<pipeline_internal::sima::KernelStageRecipe>(*stage.native_kernel);
        recipe->boxdecode_settings->detection_threshold = 0;
        recipe->boxdecode_settings->nms_iou_threshold = 0;
        recipe->boxdecode_settings->topk = 0;
        stage.native_kernel = std::move(recipe);
        const auto omitted = nlohmann::json::parse(
            internal::ExecutionPlanAccess::make(defaults, model.info().model_name).to_json());
        require(omitted.at("nodes").back().at("settings") ==
                    nlohmann::json{{"decode_type", "yolov8-seg"}, {"num_classes", 80}},
                "execution-plan JSON must preserve default-field omission");
      }
      CompiledPipelineContracts nested;
      nested.fully_renderable = true;
      nested.model_storage_scopes = compiled.model_storage_scopes;
      CompiledNodeContract fragment;
      fragment.node_kind = "ModelFragment";
      fragment.element_name = fragment.logical_stage_id = "infer";
      fragment.child_stages = compiled.stages;
      nested.stages.push_back(std::move(fragment));
      auto nested_manifest =
          render_manifest_from_compiled_contracts(nested, ContractCompileInput{}, &diagnostics);
      require(nested_manifest && diagnostics.errors.empty(),
              "model-backed native child container must render");
      pipeline_internal::terminal_output_contract::PublicOutputEndpointSelector endpoint;
      endpoint.terminal_stage_key = "infer";
      endpoint.terminal_node_kind = "ModelFragment";
      endpoint.allow_unresolved_terminal_stage_fallback = true;
      pipeline_internal::sima::finalize_pipeline_storage(*nested_manifest, &endpoint);
      const auto* terminal =
          find_first_rendered_boxdecode_stage(*nested_manifest, rendered_box->element_name);
      require(terminal && terminal->native_kernel && terminal->frame_layout &&
                  terminal->frame_layout->allocations.front().policy ==
                      ::simaai::scheduler::client::AllocationPolicy::RequireHostCpu,
              "model-backed native BoxDecode child must retain final HostCpu placement");
      const auto* boxdecode_runtime = compiled_runtime_contract_from_stage(boxdecode_stage);
      require(boxdecode_runtime != nullptr, "BoxDecode must expose its tensor contract");
      require(!rendered_box->element_name.empty(),
              "rendered boxdecode stage should preserve element identity");
      require(!rendered_box->logical_stage_id.empty(),
              "rendered boxdecode stage should preserve logical stage identity");
      require(rendered_box->element_name == boxdecode_stage->element_name,
              "rendered boxdecode stage should preserve compiled element name");
      require(rendered_box->logical_stage_id == boxdecode_stage->logical_stage_id,
              "rendered boxdecode stage should preserve compiled logical stage id");
      require(boxdecode_runtime->logical_inputs.size() == 10U &&
                  boxdecode_runtime->input_bindings.size() == 10U,
              "compiled boxdecode stage must preserve all ten segmentation heads and bindings");
      require(rendered_box->logical_inputs.size() == 10U &&
                  rendered_box->input_bindings.size() == 10U,
              "rendered boxdecode stage must preserve all ten segmentation heads and bindings");
      for (std::size_t index = 0; index < 10U; ++index) {
        const auto& compiled_logical = boxdecode_runtime->logical_inputs[index];
        const auto& rendered_logical = rendered_box->logical_inputs[index];
        const auto& compiled_binding = boxdecode_runtime->input_bindings[index];
        const auto& rendered_binding = rendered_box->input_bindings[index];
        require(compiled_logical.logical_index == static_cast<int>(index) &&
                    compiled_logical.backend_input_index == static_cast<int>(index) &&
                    rendered_logical.logical_index == compiled_logical.logical_index &&
                    rendered_logical.backend_input_index == compiled_logical.backend_input_index &&
                    rendered_logical.logical_name == compiled_logical.logical_name &&
                    rendered_logical.dtype == compiled_logical.dtype &&
                    rendered_logical.shape == compiled_logical.shape,
                "compiled and rendered segmentation logical inputs must preserve exact 0..9 "
                "order and identity");
        // All ten logical members travel in one grouped TensorBuffer/TensorList
        // carrier on consumer sink pad 0. The exact member order is carried by
        // local_logical_input_index and the producer logical output/slot.
        require(compiled_binding.sink_pad_index == 0 && rendered_binding.sink_pad_index == 0 &&
                    compiled_binding.local_logical_input_index == static_cast<int>(index) &&
                    rendered_binding.sink_pad_index == compiled_binding.sink_pad_index &&
                    rendered_binding.local_logical_input_index ==
                        compiled_binding.local_logical_input_index &&
                    compiled_binding.src_logical_output_index == static_cast<int>(index) &&
                    rendered_binding.src_logical_output_index ==
                        compiled_binding.src_logical_output_index &&
                    compiled_binding.src_output_slot == static_cast<int>(index) &&
                    rendered_binding.src_output_slot == compiled_binding.src_output_slot &&
                    rendered_binding.src_physical_output_index ==
                        compiled_binding.src_physical_output_index &&
                    rendered_binding.src_physical_byte_offset ==
                        compiled_binding.src_physical_byte_offset &&
                    rendered_binding.src_physical_size_bytes ==
                        compiled_binding.src_physical_size_bytes &&
                    rendered_binding.source_segment_name == compiled_binding.source_segment_name,
                "compiled and rendered segmentation bindings must preserve exact 0..9 order "
                "and source identity");
      }
      require(!rendered_box->input_bindings.empty(),
              "rendered boxdecode stage should expose upstream MLA bindings");
      require(!rendered_box->input_bindings.front().source_segment_name.empty(),
              "rendered boxdecode stage should preserve a concrete upstream segment identity");
      require(rendered_box->input_bindings.front().source_segment_name ==
                  boxdecode_runtime->input_bindings.front().source_segment_name,
              "rendered boxdecode stage should preserve upstream packed segment binding");
    }));

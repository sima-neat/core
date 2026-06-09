#include "asset_utils.h"
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "model_archive_fixture_utils.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/BoxDecodeType.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "test_main.h"

#include <filesystem>

namespace {

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
                                                      true);
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
    if (!found && stage.boxdecode.has_value()) {
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
    const simaai::neat::pipeline_internal::sima::SimaPluginStaticManifest& manifest) {
  for (const auto& stage : manifest.stages) {
    if (stage.payload_kind == simaai::neat::pipeline_internal::sima::StagePayloadKind::BoxDecode) {
      return &stage;
    }
  }
  return nullptr;
}

} // namespace

RUN_TEST(
    "unit_boxdecode_render_manifest_from_model_test", ([] {
      using namespace simaai::neat;

      const auto fixture = make_fixture();
      const std::string tar_path = fixture.tar_path;

      Model::Options base_opt;
      base_opt.preprocess.kind = InputKind::Image;
      base_opt.preprocess.enable = AutoFlag::On;
      base_opt.preprocess.color_convert.input_format = PreprocessColorFormat::BGR;

      Model default_model(tar_path, base_opt);
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

      const auto* rendered_box = find_first_rendered_boxdecode_stage(*manifest_opt);
      require(rendered_box != nullptr, "rendered manifest should contain a boxdecode stage");
      require(rendered_box->payload_kind == pipeline_internal::sima::StagePayloadKind::BoxDecode,
              "rendered stage payload kind should remain boxdecode");
      require(!rendered_box->element_name.empty(),
              "rendered boxdecode stage should preserve element identity");
      require(!rendered_box->logical_stage_id.empty(),
              "rendered boxdecode stage should preserve logical stage identity");
      require(rendered_box->element_name == boxdecode_stage->element_name,
              "rendered boxdecode stage should preserve compiled element name");
      require(rendered_box->logical_stage_id == boxdecode_stage->logical_stage_id,
              "rendered boxdecode stage should preserve compiled logical stage id");
      require(rendered_box->logical_inputs.size() ==
                  boxdecode_stage->boxdecode->runtime_contract.logical_inputs.size(),
              "rendered boxdecode stage should preserve logical input count");
      require(rendered_box->input_bindings.size() ==
                  boxdecode_stage->boxdecode->runtime_contract.input_bindings.size(),
              "rendered boxdecode stage should preserve binding count");
      require(!rendered_box->input_bindings.empty(),
              "rendered boxdecode stage should expose upstream MLA bindings");
      require(!rendered_box->input_bindings.front().source_segment_name.empty(),
              "rendered boxdecode stage should preserve a concrete upstream segment identity");
      require(rendered_box->input_bindings.front().source_segment_name ==
                  boxdecode_stage->boxdecode->runtime_contract.input_bindings.front()
                      .source_segment_name,
              "rendered boxdecode stage should preserve upstream packed segment binding");
    }));

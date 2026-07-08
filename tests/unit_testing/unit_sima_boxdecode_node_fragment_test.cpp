#include "asset_utils.h"
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "model_archive_fixture_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <filesystem>

namespace {

sima_test::ModelArchiveFixture make_fixture() {
  return sima_test::make_strict_model_archive_fixture("boxdecode_node_fragment",
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
  "original_width": 1280,
  "original_height": 720,
  "model_width": 640,
  "model_height": 640,
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

sima_test::ModelArchiveFixture make_quanttess_boxdecode_fixture() {
  return sima_test::make_strict_model_archive_fixture("boxdecode_node_fragment_quanttess",
                                                      {
                                                          {"etc/pipeline_sequence.json",
                                                           R"json({
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
        "name": "quanttess_0",
        "pluginId": "processcvu",
        "configPath": "0_quanttess.json",
        "processor": "CVU",
        "kernel": "quanttess",
        "input": "decoder"
      },
      {
        "sequence_id": 2,
        "name": "mla_0",
        "pluginId": "processmla",
        "configPath": "0_process_mla.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "quanttess_0"
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
                                                          {"etc/0_quanttess.json",
                                                           R"json({
  "node_name": "quanttess_0",
  "input_width": 640,
  "input_height": 640,
  "input_depth": 3
})json"},
                                                          {"etc/0_process_mla.json",
                                                           R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "quanttess_0"}],
  "input_format": ["EV81_INT8"],
  "data_type": ["EV81_INT8"],
  "input_width": [640],
  "input_height": [640],
  "input_depth": [3],
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

} // namespace

RUN_TEST("unit_sima_boxdecode_node_fragment_test", ([] {
           const auto fixture = make_fixture();
           const std::string tar_path = fixture.tar_path;

           simaai::neat::Model::Options model_opt;
           model_opt.preprocess.kind = simaai::neat::InputKind::Image;
           model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
           model_opt.preprocess.color_convert.input_format =
               simaai::neat::PreprocessColorFormat::BGR;

           simaai::neat::Model default_model(tar_path, model_opt);
           require(!simaai::neat::internal::ModelAccess::has_model_managed_stage(
                       default_model, simaai::neat::internal::StageNodeKind::BoxDecode),
                   "default Model route must not auto-select BoxDecode from inferred MPK "
                   "topology");

           simaai::neat::Model::Options mismatched_opt = model_opt;
           mismatched_opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
           simaai::neat::Model mismatched_model(tar_path, mismatched_opt);
           bool boxdecode_mismatch_rejected = false;
           try {
             (void)simaai::neat::internal::ModelAccess::build_boxdecode_stage_contract(
                 mismatched_model, false);
           } catch (const std::exception&) {
             boxdecode_mismatch_rejected = true;
           }
           require(boxdecode_mismatch_rejected,
                   "explicit BoxDecode must reject a detection decoder for a segmentation MPK "
                   "contract");

           simaai::neat::Model::Options managed_opt = model_opt;
           managed_opt.decode_type = simaai::neat::BoxDecodeType::YoloV8Seg;
           simaai::neat::Model managed_model(tar_path, managed_opt);
           auto managed_node = simaai::neat::nodes::SimaBoxDecode(
               managed_model, simaai::neat::BoxDecodeType::YoloV8Seg, 0.25, 0.45, 100);
           const auto* managed_box =
               dynamic_cast<const simaai::neat::SimaBoxDecode*>(managed_node.get());
           require(managed_box != nullptr,
                   "model-managed boxdecode factory should return a concrete SimaBoxDecode node");
           const std::string managed_fragment = managed_box->backend_fragment(0);
           require(managed_fragment.find("original-width=") == std::string::npos,
                   "model-managed boxdecode should let metadata drive original width");
           require(managed_fragment.find("original-height=") == std::string::npos,
                   "model-managed boxdecode should let metadata drive original height");

           auto standalone_node =
               simaai::neat::nodes::SimaBoxDecode(simaai::neat::BoxDecodeType::YoloV8, 0.25, 0.45,
                                                  100, "manual_boxdecode", 1280, 720, 640, 640);
           const auto* standalone_box =
               dynamic_cast<const simaai::neat::SimaBoxDecode*>(standalone_node.get());
           require(standalone_box != nullptr,
                   "manual boxdecode factory should return a concrete SimaBoxDecode node");
           const auto standalone_req = standalone_box->preprocess_meta_requirement();
           require(standalone_req.has_value(),
                   "manual boxdecode should still expose non-geometry preprocess requirements");
           require_contains(standalone_box->backend_fragment(0), "model-width=640",
                            "boxdecode node fragment should emit explicit model-width");
           require_contains(standalone_box->backend_fragment(0), "model-height=640",
                            "boxdecode node fragment should emit explicit model-height");
           require(std::find(standalone_req->required_fields.begin(),
                             standalone_req->required_fields.end(),
                             "preproc_original_width") == standalone_req->required_fields.end(),
                   "manual boxdecode should drop original-width meta requirement when overridden");
           require(std::find(standalone_req->required_fields.begin(),
                             standalone_req->required_fields.end(),
                             "preproc_original_height") == standalone_req->required_fields.end(),
                   "manual boxdecode should drop original-height meta requirement when overridden");
           require(std::find(standalone_req->required_fields.begin(),
                             standalone_req->required_fields.end(),
                             "preproc_resized_width") == standalone_req->required_fields.end(),
                   "manual boxdecode should drop resized-width meta requirement when model dims "
                   "are overridden");
           require(std::find(standalone_req->required_fields.begin(),
                             standalone_req->required_fields.end(),
                             "preproc_scaled_height") == standalone_req->required_fields.end(),
                   "manual boxdecode should drop scaled-height meta requirement when model dims "
                   "are overridden");
           require(std::find(standalone_req->required_fields.begin(),
                             standalone_req->required_fields.end(),
                             "preproc_resize_mode") != standalone_req->required_fields.end(),
                   "manual boxdecode should preserve non-geometry preprocess requirements");

           bool threw_partial_model_dims = false;
           try {
             (void)simaai::neat::nodes::SimaBoxDecode(simaai::neat::BoxDecodeType::YoloV8, 0.25,
                                                      0.45, 100, "bad_manual_boxdecode", 1280, 720,
                                                      640, 0);
           } catch (const std::exception& e) {
             threw_partial_model_dims = true;
             require_contains(std::string(e.what()),
                              "explicit model dimensions requires both width and height",
                              "partial-model-dims error text mismatch");
           }
           require(threw_partial_model_dims,
                   "manual boxdecode must fail when only one explicit model dimension is provided");

           const auto legacy = sima_test::make_model_archive_fixture(
               "boxdecode_node_fragment_legacy_missing_mpk", {
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
             simaai::neat::Model legacy_model(legacy.tar_path);
             (void)simaai::neat::nodes::SimaBoxDecode(
                 legacy_model, simaai::neat::BoxDecodeType::YoloV8, 0.35, 0.5, 120);
           } catch (const std::exception& e) {
             threw = true;
             require_contains(std::string(e.what()), "strict MPK contract required",
                              "legacy missing-mpk fixture should fail with strict contract error");
           }
           require(threw, "legacy missing-mpk fixture must fail under strict contract");
         }));

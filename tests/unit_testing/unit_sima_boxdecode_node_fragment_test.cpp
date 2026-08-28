#include "asset_utils.h"
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"
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
           const auto poisoned_payload = sima_test::make_model_archive_fixture(
               "poisoned_six_head_yolo_cache_payload",
               {{"etc/poisoned_six_head_mpk.json",
                 R"json({
  "name": "yolo_v9c_seg",
  "plugins": [{
    "name": "MLA_0_ofm_unpack_transform",
    "output_nodes": [
      {"name":"out_0"}, {"name":"out_1"}, {"name":"out_2"},
      {"name":"out_3"}, {"name":"out_4"}, {"name":"out_5"}
    ]
  }]
})json"}},
               false);
           const std::filesystem::path poison_root =
               sima_test::make_fixture_temp_dir("poisoned_yolo_v9c_seg_cache");
           std::filesystem::create_directories(poison_root / "tmp");
           std::filesystem::copy_file(
               poisoned_payload.tar_path, poison_root / "tmp" / "yolo_v9c_seg_mpk.tar.gz");
           const auto original_cwd = std::filesystem::current_path();
           std::pair<std::string, std::string> isolated_seed;
           try {
             std::filesystem::current_path(poison_root);
             isolated_seed =
                 sima_test::strict_contract_json_entry_from_modelzoo("yolo_v9c_seg");
             std::filesystem::current_path(original_cwd);
           } catch (...) {
             std::filesystem::current_path(original_cwd);
             throw;
           }
           require(isolated_seed.second.size() == 47224U &&
                       isolated_seed.second.find("dequantize_11/mask") != std::string::npos,
                   "strict fixture authority must ignore a poisoned six-head modelzoo cache");

           const auto fixture = make_fixture();
           const std::string tar_path = fixture.tar_path;

           simaai::neat::Model::Options model_opt;
           model_opt.preprocess.kind = simaai::neat::InputKind::Image;
           model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
           model_opt.preprocess.color_convert.input_format =
               simaai::neat::PreprocessColorFormat::BGR;

           simaai::neat::Model default_model(tar_path, model_opt);
           sima_test::require_exact_yolo_v9c_seg_parsed_contract(
               simaai::neat::internal::ModelAccess::pack(default_model));
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
           const auto& managed_pack =
               simaai::neat::internal::ModelAccess::pack(managed_model);
           const auto managed_post_plan = managed_pack.execution_plan().post;
           const auto managed_post_facts = managed_pack.stage_facts_for_model_stage(
               simaai::neat::internal::ModelStage::Postprocess);
           require(managed_post_plan.size() == managed_post_facts.size(),
                   "synthetic BoxDecode fixture must keep its post stage facts aligned");
           bool saw_synthetic_boxdecode_without_packaged_contract = false;
           for (std::size_t index = 0; index < managed_post_plan.size(); ++index) {
             if (managed_post_plan[index].kind !=
                 simaai::neat::internal::ExecutionStageKind::BoxDecode) {
               continue;
             }
             require(!managed_post_facts[index].boxdecode_compiled.has_value(),
                     "a synthetic BoxDecode stage without an authored MPK terminal must defer "
                     "contract compilation to ModelAccess");
             saw_synthetic_boxdecode_without_packaged_contract = true;
           }
           require(saw_synthetic_boxdecode_without_packaged_contract,
                   "fixture must exercise a synthetic BoxDecode stage without a packaged "
                   "terminal contract");

           const auto managed_async_contract =
               simaai::neat::internal::ModelAccess::build_boxdecode_stage_contract(
                   managed_model, false);
           const auto managed_sync_contract =
               simaai::neat::internal::ModelAccess::build_boxdecode_stage_contract(
                   managed_model, true);
           require(managed_async_contract.payload.decode_type ==
                       simaai::neat::BoxDecodeType::YoloV8Seg &&
                       managed_sync_contract.payload.decode_type ==
                           managed_async_contract.payload.decode_type,
                   "ModelAccess must derive identical external true-leaf contracts for "
                   "synthetic sync and async BoxDecode routes");
           require(managed_async_contract.runtime_contract.logical_inputs.size() == 10U &&
                       managed_sync_contract.runtime_contract.logical_inputs.size() == 10U &&
                       managed_async_contract.runtime_contract.input_bindings.size() == 10U &&
                       managed_sync_contract.runtime_contract.input_bindings.size() == 10U,
                   "synthetic sync and async BoxDecode contracts must preserve all ten "
                   "segmentation heads and bindings");
           for (std::size_t index = 0; index < 10U; ++index) {
             const auto& async_logical =
                 managed_async_contract.runtime_contract.logical_inputs[index];
             const auto& sync_logical =
                 managed_sync_contract.runtime_contract.logical_inputs[index];
             const auto& async_binding =
                 managed_async_contract.runtime_contract.input_bindings[index];
             const auto& sync_binding =
                 managed_sync_contract.runtime_contract.input_bindings[index];
             const std::string prefix = "segmentation head " + std::to_string(index) + ": ";
             require(async_logical.logical_index == static_cast<int>(index) &&
                         sync_logical.logical_index == static_cast<int>(index),
                     prefix + "logical_index mismatch async=" +
                         std::to_string(async_logical.logical_index) + " sync=" +
                         std::to_string(sync_logical.logical_index) + " expected=" +
                         std::to_string(index));
             require(async_logical.backend_input_index == static_cast<int>(index) &&
                         sync_logical.backend_input_index == static_cast<int>(index),
                     prefix + "backend_input_index mismatch async=" +
                         std::to_string(async_logical.backend_input_index) + " sync=" +
                         std::to_string(sync_logical.backend_input_index) + " expected=" +
                         std::to_string(index));
             require(async_logical.logical_name == sync_logical.logical_name &&
                         async_logical.backend_name == sync_logical.backend_name,
                     prefix + "logical/backend name parity mismatch async_logical='" +
                         async_logical.logical_name + "' sync_logical='" +
                         sync_logical.logical_name + "' async_backend='" +
                         async_logical.backend_name + "' sync_backend='" +
                         sync_logical.backend_name + "'");
             require(async_logical.physical_index == sync_logical.physical_index &&
                         async_logical.segment_name == sync_logical.segment_name &&
                         async_logical.byte_offset == sync_logical.byte_offset &&
                         async_logical.size_bytes == sync_logical.size_bytes,
                     prefix + "logical source parity mismatch async_phys=" +
                         std::to_string(async_logical.physical_index) + " sync_phys=" +
                         std::to_string(sync_logical.physical_index) + " async_segment='" +
                         async_logical.segment_name + "' sync_segment='" +
                         sync_logical.segment_name + "' async_offset=" +
                         std::to_string(async_logical.byte_offset) + " sync_offset=" +
                         std::to_string(sync_logical.byte_offset) + " async_span=" +
                         std::to_string(async_logical.size_bytes) + " sync_span=" +
                         std::to_string(sync_logical.size_bytes));
             // BoxDecode receives one grouped TensorBuffer/TensorList carrier on
             // consumer sink pad 0. Member identity is expressed by the exact
             // local logical index and producer logical output/slot below, not by
             // N separate GStreamer sink pads.
             require(async_binding.sink_pad_index == 0 && sync_binding.sink_pad_index == 0,
                     prefix + "sink_pad_index mismatch async=" +
                         std::to_string(async_binding.sink_pad_index) + " sync=" +
                         std::to_string(sync_binding.sink_pad_index) + " expected=0");
             require(async_binding.local_logical_input_index == static_cast<int>(index) &&
                         sync_binding.local_logical_input_index == static_cast<int>(index),
                     prefix + "local_logical_input_index mismatch async=" +
                         std::to_string(async_binding.local_logical_input_index) + " sync=" +
                         std::to_string(sync_binding.local_logical_input_index) + " expected=" +
                         std::to_string(index));
             require(async_binding.src_logical_output_index == static_cast<int>(index) &&
                         sync_binding.src_logical_output_index == static_cast<int>(index),
                     prefix + "src_logical_output_index mismatch async=" +
                         std::to_string(async_binding.src_logical_output_index) + " sync=" +
                         std::to_string(sync_binding.src_logical_output_index) + " expected=" +
                         std::to_string(index));
             require(async_binding.src_output_slot == static_cast<int>(index) &&
                         sync_binding.src_output_slot == static_cast<int>(index),
                     prefix + "src_output_slot mismatch async=" +
                         std::to_string(async_binding.src_output_slot) + " sync=" +
                         std::to_string(sync_binding.src_output_slot) + " expected=" +
                         std::to_string(index));
             require(async_binding.src_physical_output_index ==
                             sync_binding.src_physical_output_index &&
                         async_binding.src_physical_byte_offset ==
                             sync_binding.src_physical_byte_offset &&
                         async_binding.src_physical_size_bytes ==
                             sync_binding.src_physical_size_bytes,
                     prefix + "physical source parity mismatch async_phys=" +
                         std::to_string(async_binding.src_physical_output_index) + " sync_phys=" +
                         std::to_string(sync_binding.src_physical_output_index) +
                         " async_offset=" +
                         std::to_string(async_binding.src_physical_byte_offset) + " sync_offset=" +
                         std::to_string(sync_binding.src_physical_byte_offset) + " async_span=" +
                         std::to_string(async_binding.src_physical_size_bytes) + " sync_span=" +
                         std::to_string(sync_binding.src_physical_size_bytes));
             require(async_binding.cm_input_name == sync_binding.cm_input_name &&
                         async_binding.source_segment_name == sync_binding.source_segment_name,
                     prefix + "binding name/segment parity mismatch async_input='" +
                         async_binding.cm_input_name + "' sync_input='" +
                         sync_binding.cm_input_name + "' async_segment='" +
                         async_binding.source_segment_name + "' sync_segment='" +
                         sync_binding.source_segment_name + "'");
           }
           auto managed_node = simaai::neat::nodes::SimaBoxDecode(
               managed_model, simaai::neat::BoxDecodeType::YoloV8Seg, 0.25, 0.45, 100);
           const auto* managed_box =
               dynamic_cast<const simaai::neat::SimaBoxDecode*>(managed_node.get());
           require(managed_box != nullptr,
                   "model-managed boxdecode factory should return a concrete SimaBoxDecode node");
           const std::string managed_fragment = managed_box->backend_fragment(0);
           require_contains(managed_fragment, "num-buffers=4",
                            "model-managed terminal must use the model-authored MLA lane window");
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
           require_contains(standalone_box->backend_fragment(0), "num-buffers=2",
                            "standalone terminal must declare its bounded compatibility window");
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

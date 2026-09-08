#include "asset_utils.h"
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "model_archive_fixture_utils.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <stdexcept>

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

sima_test::ModelArchiveFixture
make_rfdetr_feature_geometry_fixture(simaai::neat::BoxDecodeType type) {
  // RF transformer ingress is a feature map; image geometry comes from upstream metadata.

  auto mpk = nlohmann::json::parse(R"json({
    "name": "rfdetr_feature_geometry",
    "model_sdk_version": "2.1.0",
    "input_nodes": [{"name":"features","type":"buffer","size":1327104,
                     "dtype":"float32","shape":[1,36,36,256]}],
    "plugins": [{
      "name": "MLA_0", "sequence": 1, "processor": "MLA",
      "config_params": {
        "desired_batch_size": 1, "actual_batch_size": 1,
        "number_of_quads_to_user": 1,
        "input_types": [{"scalar":"float32","shape":[1,36,36,256]}],
        "output_types": [
          {"scalar":"float32","shape":[1,1,7,4]},
          {"scalar":"float32","shape":[1,1,7,5]},
          {"scalar":"float32","shape":[1,3,6,7]}
        ]
      },
      "input_nodes": [{"name":"features","type":"buffer","size":1327104,
                       "dtype":"float32","shape":[1,36,36,256]}],
      "output_nodes": [
        {"name":"output_0","type":"buffer","size":112,
         "dtype":"float32","shape":[1,1,7,4],"layout":"normal"},
        {"name":"output_1","type":"buffer","size":140,
         "dtype":"float32","shape":[1,1,7,5],"layout":"normal"},
        {"name":"output_2","type":"buffer","size":504,
         "dtype":"float32","shape":[1,3,6,7],"layout":"normal"}
      ],
      "type": "sgpProcess", "resources": {"executable":"placeholder.elf"}
    }, {
      "name": "boxdecode_rf", "sequence": 2, "processor": "A65",
      "config_params": {"kernel":"boxdecode","params":{"decode_type":"rfdetr_seg"}},
      "input_nodes": [],
      "output_nodes": [{"name":"detections","type":"buffer","size":4096}]
    }]
  })json");
  auto& mla = mpk["plugins"][0];
  auto& terminal = mpk["plugins"][1];
  if (type == simaai::neat::BoxDecodeType::RfDetr) {
    mla["output_nodes"].erase(2);
    mla["config_params"]["output_types"].erase(2);
    terminal["config_params"]["params"]["decode_type"] = "rfdetr";
  }
  terminal["input_nodes"] = mla["output_nodes"];
  return sima_test::make_model_archive_fixture(
      "rfdetr_feature_geometry", {{"etc/rfdetr_feature_geometry_mpk.json", mpk.dump()}});
}

} // namespace

RUN_TEST(
    "unit_sima_boxdecode_node_fragment_test", ([] {
      for (const auto type :
           {simaai::neat::BoxDecodeType::RfDetr, simaai::neat::BoxDecodeType::RfDetrSeg}) {
        const auto rf_fixture = make_rfdetr_feature_geometry_fixture(type);
        for (const auto requested_type : {type, simaai::neat::BoxDecodeType::Unspecified}) {
          simaai::neat::Model::Options rf_options;
          rf_options.decode_type = requested_type;
          rf_options.preprocess.kind = simaai::neat::InputKind::Tensor;
          rf_options.preprocess.enable = simaai::neat::AutoFlag::Off;
          const simaai::neat::Model rf_model(rf_fixture.tar_path, rf_options);
          const auto resolved = rf_model.resolved_preprocess_plan();
          require(!resolved.enabled && resolved.mla_contract.width == 36 &&
                      resolved.mla_contract.height == 36,
                  "RF geometry fixture must retain a nonzero feature-map ingress with preproc off");
          const auto post =
              simaai::neat::internal::ModelAccess::build_public_postprocess_nodes(rf_model);
          const simaai::neat::SimaBoxDecode* box = nullptr;
          for (const auto& node : post) {
            if (const auto* candidate =
                    dynamic_cast<const simaai::neat::SimaBoxDecode*>(node.get())) {
              require(box == nullptr, "RF model route must materialize exactly one BoxDecode");
              box = candidate;
            }
          }
          require(box != nullptr, "explicit and MPK-selected RF routes must materialize BoxDecode");
          const auto fragment = box->backend_fragment(0);
          require(
              fragment.find("model-width=") == std::string::npos &&
                  fragment.find("model-height=") == std::string::npos,
              "explicit and MPK-selected RF routes must leave image geometry to runtime metadata");
        }
      }

      {
        using namespace simaai::neat;
        const auto rf_fixture = make_rfdetr_feature_geometry_fixture(BoxDecodeType::RfDetrSeg);
        Model::Options model_options;
        model_options.preprocess.kind = InputKind::Tensor;
        model_options.preprocess.enable = AutoFlag::Off;
        model_options.score_threshold = 0.7f;
        model_options.top_k = 1;
        const Model rf_model(rf_fixture.tar_path, model_options);

        const auto require_options = [](const Node& node, const BoxDecodeOptions& expected) {
          ContractCompileInput input;
          CompiledNodeContract compiled;
          std::string error;
          require(dynamic_cast<const SimaBoxDecode&>(node).compile_node_contract(input, &compiled,
                                                                                 &error),
                  "named RF node must compile its model-backed contract: " + error);
          require(compiled.boxdecode.has_value(), "named RF node must emit a BoxDecode payload");
          const auto& payload = compiled.boxdecode->payload;
          require(payload.decode_type == BoxDecodeType::RfDetrSeg,
                  "named Unspecified decoder must retain MPK-selected RF segmentation");
          require(payload.detection_threshold == expected.detection_threshold &&
                      payload.nms_iou_threshold == expected.nms_iou_threshold &&
                      payload.topk == expected.top_k,
                  "named RF controls must replace model defaults, including explicit zeros");
          const auto& masks = payload.rfdetr.masks;
          require(masks.output == expected.masks.output && masks.size == expected.masks.size &&
                      masks.threshold == expected.masks.threshold &&
                      masks.width == expected.masks.width && masks.height == expected.masks.height,
                  "named RF mask controls must reach the compiled payload");
        };

        BoxDecodeOptions probabilities(BoxDecodeType::Unspecified);
        probabilities.masks.output = MaskOutput::Probabilities;
        probabilities.masks.threshold = 0.25;
        BoxDecodeOptions fixed(BoxDecodeType::Unspecified);
        fixed.masks.size = MaskSize::Fixed;
        fixed.masks.width = 31;
        fixed.masks.height = 17;
        fixed.masks.threshold = 0.75;
        for (const auto& requested : {probabilities, fixed}) {
          const auto node = nodes::SimaBoxDecode(rf_model, requested);
          require_options(*node, requested);
          const auto retargeted =
              dynamic_cast<const SimaBoxDecode&>(*node).retargeted_for_model_internal(rf_model);
          const auto* box = dynamic_cast<const SimaBoxDecode*>(retargeted.get());
          require(box != nullptr, "retargeted RF node must remain a BoxDecode node");
          require_options(*box, requested);
        }

        BoxDecodeOptions invalid(BoxDecodeType::Unspecified);
        invalid.masks.threshold = -0.1;
        bool rejected = false;
        try {
          (void)nodes::SimaBoxDecode(rf_model, invalid);
        } catch (const std::invalid_argument& error) {
          require_contains(error.what(), "RF-DETR",
                           "auto-selected RF mask validation must identify the decoder");
          rejected = true;
        }
        require(rejected, "MPK auto-selection must validate named RF mask controls");
      }

      const auto fixture = make_fixture();
      const std::string tar_path = fixture.tar_path;

      simaai::neat::Model::Options model_opt;
      model_opt.preprocess.kind = simaai::neat::InputKind::Image;
      model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
      model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;

      simaai::neat::Model default_model(tar_path, model_opt);
      require(!simaai::neat::internal::ModelAccess::has_model_managed_stage(
                  default_model, simaai::neat::internal::StageNodeKind::BoxDecode),
              "default Model route must not auto-select BoxDecode from inferred MPK "
              "topology");

      const auto requires_model = [](auto construct) {
        try {
          (void)construct();
        } catch (const std::invalid_argument& error) {
          require_contains(error.what(), "RF-DETR requires a Model-backed MPK contract",
                           "standalone RF decoder must explain the supported construction");
          return;
        }
        throw std::runtime_error("standalone RF decoder must reject before compilation");
      };
      for (const auto type :
           {simaai::neat::BoxDecodeType::RfDetr, simaai::neat::BoxDecodeType::RfDetrSeg}) {
        requires_model([&] {
          return simaai::neat::nodes::SimaBoxDecode(simaai::neat::BoxDecodeOptions{type});
        });
        requires_model([&] { return simaai::neat::nodes::SimaBoxDecode(type); });
        requires_model([&] {
          return simaai::neat::nodes::SimaBoxDecode(
              type, 0.3, 0.0, 100, "", 1280, 720, 640, 640, simaai::neat::BoxDecodeTypeOption::Auto,
              std::nullopt, std::nullopt, std::nullopt, simaai::neat::ResizeMode::Stretch);
        });
      }

      // YOLO defines the established preprocessing-metadata contract. SSD and
      // SuperPoint must consume geometry through the same neatobjectdecode path rather
      // than inventing family-specific width/height sources.
      auto metadata_yolo = simaai::neat::nodes::SimaBoxDecode(simaai::neat::BoxDecodeType::YoloV8,
                                                              0.25, 0.45, 100, "metadata_yolo");
      auto metadata_ssd = simaai::neat::nodes::SimaBoxDecode(simaai::neat::BoxDecodeType::Ssd, 0.30,
                                                             0.60, 100, "metadata_ssd");
      simaai::neat::BoxDecodeOptions superpoint_options(simaai::neat::BoxDecodeType::SuperPoint);
      superpoint_options.superpoint.profile = simaai::neat::SuperPointProfile::A65V1;
      auto metadata_superpoint =
          simaai::neat::nodes::SimaBoxDecode(superpoint_options, "metadata_superpoint");

      const auto* metadata_yolo_box =
          dynamic_cast<const simaai::neat::SimaBoxDecode*>(metadata_yolo.get());
      const auto* metadata_ssd_box =
          dynamic_cast<const simaai::neat::SimaBoxDecode*>(metadata_ssd.get());
      const auto* metadata_superpoint_box =
          dynamic_cast<const simaai::neat::SimaBoxDecode*>(metadata_superpoint.get());
      require(metadata_yolo_box && metadata_ssd_box && metadata_superpoint_box,
              "all BoxDecode families must use concrete SimaBoxDecode nodes");
      require_contains(metadata_yolo_box->backend_fragment(0), "neatobjectdecode",
                       "YOLO must use the shared ObjectDecode plugin");
      require_contains(metadata_ssd_box->backend_fragment(0), "neatobjectdecode",
                       "SSD must use the shared ObjectDecode plugin");
      require_contains(metadata_superpoint_box->backend_fragment(0), "neatobjectdecode",
                       "SuperPoint must use the shared ObjectDecode plugin");

      const auto yolo_geometry_req = metadata_yolo_box->preprocess_meta_requirement();
      const auto ssd_geometry_req = metadata_ssd_box->preprocess_meta_requirement();
      const auto superpoint_geometry_req = metadata_superpoint_box->preprocess_meta_requirement();
      require(yolo_geometry_req && ssd_geometry_req && superpoint_geometry_req,
              "metadata-driven BoxDecode families must require preprocessing metadata");
      require(yolo_geometry_req->required_fields == ssd_geometry_req->required_fields,
              "SSD must preserve YOLO's preprocessing metadata contract");
      require(yolo_geometry_req->required_fields == superpoint_geometry_req->required_fields,
              "SuperPoint must preserve YOLO's preprocessing metadata contract");
      for (const char* geometry_field :
           {"preproc_original_width", "preproc_original_height", "preproc_resized_width",
            "preproc_resized_height", "preproc_scaled_width",    "preproc_scaled_height",
            "preproc_pad_left",       "preproc_pad_right",       "preproc_pad_top",
            "preproc_pad_bottom",     "preproc_resize_mode",     "preproc_affine_m00",
            "preproc_affine_m01",     "preproc_affine_m02",      "preproc_affine_m10",
            "preproc_affine_m11",     "preproc_affine_m12",      "preproc_affine_scale_x",
            "preproc_affine_scale_y", "preproc_affine_offset_x", "preproc_affine_offset_y"}) {
        require(std::find(yolo_geometry_req->required_fields.begin(),
                          yolo_geometry_req->required_fields.end(),
                          geometry_field) != yolo_geometry_req->required_fields.end(),
                std::string("shared BoxDecode preprocessing contract is missing '") +
                    geometry_field + "'");
      }

      simaai::neat::Model::Options mismatched_opt = model_opt;
      mismatched_opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
      simaai::neat::Model mismatched_model(tar_path, mismatched_opt);
      bool boxdecode_mismatch_rejected = false;
      try {
        (void)simaai::neat::internal::ModelAccess::build_boxdecode_stage_contract(mismatched_model,
                                                                                  false);
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

      // A resize assertion may fill in missing external provenance, but it must not relabel
      // an active model transform. Otherwise the decoder would invert the wrong geometry.
      simaai::neat::Model::Options letterbox_opt = managed_opt;
      letterbox_opt.preprocess.resize.enable = simaai::neat::AutoFlag::On;
      letterbox_opt.preprocess.resize.mode = simaai::neat::ResizeMode::Letterbox;
      letterbox_opt.preprocess.resize.width = 640;
      letterbox_opt.preprocess.resize.height = 640;
      simaai::neat::Model letterbox_model(tar_path, letterbox_opt);
      bool conflicting_resize_override_rejected = false;
      try {
        (void)simaai::neat::nodes::SimaBoxDecode(
            letterbox_model, simaai::neat::BoxDecodeType::YoloV8Seg, 0.25, 0.45, 100, "",
            std::nullopt, std::nullopt, 0, 0, 0, 0, simaai::neat::ResizeMode::Stretch);
      } catch (const std::exception& e) {
        conflicting_resize_override_rejected = true;
        require_contains(e.what(), "conflicts with the active preprocess resize mode",
                         "resize-plan conflict should have an actionable diagnostic");
      }
      require(conflicting_resize_override_rejected,
              "an explicit Stretch override must not mask an active Letterbox resize");

      auto standalone_node =
          simaai::neat::nodes::SimaBoxDecode(simaai::neat::BoxDecodeType::YoloV8, 0.25, 0.45, 100,
                                             "manual_boxdecode", 1280, 720, 640, 640);
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

      // A raw SSD node must not manufacture preprocessing evidence. Without an explicit
      // assertion it consumes resize mode from upstream metadata.
      auto ssd_node = simaai::neat::nodes::SimaBoxDecode(
          simaai::neat::BoxDecodeType::Ssd, 0.30, 0.60, 100, "ssd_manual", 1280, 720, 300, 300);
      const auto* ssd_box = dynamic_cast<const simaai::neat::SimaBoxDecode*>(ssd_node.get());
      require(ssd_box != nullptr, "raw SSD boxdecode factory should return a concrete node");
      require(ssd_box->backend_fragment(0).find("resize-mode=") == std::string::npos,
              "raw SSD boxdecode must not invent a resize-mode override");
      const auto ssd_req = ssd_box->preprocess_meta_requirement();
      require(ssd_req.has_value() &&
                  std::find(ssd_req->required_fields.begin(), ssd_req->required_fields.end(),
                            "preproc_resize_mode") != ssd_req->required_fields.end(),
              "raw SSD boxdecode must require upstream resize-mode metadata");

      // External preprocessing may be asserted explicitly, but SSD accepts only Stretch.
      auto asserted_ssd_node = simaai::neat::nodes::SimaBoxDecode(
          simaai::neat::BoxDecodeType::Ssd, 0.30, 0.60, 100, "ssd_manual_asserted", 1280, 720, 300,
          300, simaai::neat::BoxDecodeTypeOption::Auto, std::nullopt, std::nullopt, std::nullopt,
          simaai::neat::ResizeMode::Stretch);
      const auto* asserted_ssd_box =
          dynamic_cast<const simaai::neat::SimaBoxDecode*>(asserted_ssd_node.get());
      require(asserted_ssd_box != nullptr, "asserted raw SSD node must be concrete");
      require_contains(asserted_ssd_box->backend_fragment(0), "resize-mode=stretch",
                       "explicit SSD Stretch assertion must reach the backend fragment");
      const auto asserted_ssd_req = asserted_ssd_box->preprocess_meta_requirement();
      require(asserted_ssd_req.has_value(),
              "an SSD geometry assertion must preserve unrelated preprocess requirements");
      require(std::find(asserted_ssd_req->required_fields.begin(),
                        asserted_ssd_req->required_fields.end(),
                        "preproc_resize_mode") == asserted_ssd_req->required_fields.end(),
              "the explicit SSD Stretch assertion should discharge resize-mode metadata");
      require(std::find(asserted_ssd_req->required_fields.begin(),
                        asserted_ssd_req->required_fields.end(),
                        "preproc_color_in") != asserted_ssd_req->required_fields.end(),
              "a resize assertion must not discharge color metadata");
      require(std::find(asserted_ssd_req->required_fields.begin(),
                        asserted_ssd_req->required_fields.end(),
                        "preproc_normalize") != asserted_ssd_req->required_fields.end(),
              "a resize assertion must not discharge normalization metadata");
      require(std::find(asserted_ssd_req->required_fields.begin(),
                        asserted_ssd_req->required_fields.end(),
                        "preproc_quantize") != asserted_ssd_req->required_fields.end(),
              "a resize assertion must not discharge quantization metadata");

      bool rejected_letterbox = false;
      try {
        (void)simaai::neat::nodes::SimaBoxDecode(
            simaai::neat::BoxDecodeType::Ssd, 0.30, 0.60, 100, "ssd_manual_letterbox", 1280, 720,
            300, 300, simaai::neat::BoxDecodeTypeOption::Auto, std::nullopt, std::nullopt,
            std::nullopt, simaai::neat::ResizeMode::Letterbox);
      } catch (const std::exception& e) {
        rejected_letterbox = true;
        require_contains(e.what(), "requires a stretch",
                         "raw SSD letterbox rejection should explain the requirement");
      }
      require(rejected_letterbox, "raw SSD must reject an explicit Letterbox assertion");

      bool threw_partial_model_dims = false;
      try {
        (void)simaai::neat::nodes::SimaBoxDecode(simaai::neat::BoxDecodeType::YoloV8, 0.25, 0.45,
                                                 100, "bad_manual_boxdecode", 1280, 720, 640, 0);
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
        (void)simaai::neat::nodes::SimaBoxDecode(legacy_model, simaai::neat::BoxDecodeType::YoloV8,
                                                 0.35, 0.5, 120);
      } catch (const std::exception& e) {
        threw = true;
        require_contains(std::string(e.what()), "strict MPK contract required",
                         "legacy missing-mpk fixture should fail with strict contract error");
      }
      require(threw, "legacy missing-mpk fixture must fail under strict contract");
    }));

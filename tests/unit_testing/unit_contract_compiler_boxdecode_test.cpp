#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "pipeline/internal/sima/BoxDecodeTypeUtils.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "test_main.h"

#include <array>
#include <cstdlib>

RUN_TEST(
    "unit_contract_compiler_boxdecode_test", ([] {
      using namespace simaai::neat;
      using namespace simaai::neat::pipeline_internal::sima;
      using namespace simaai::neat::pipeline_internal::sima::stagesemantics;

      auto mark_storage = [](BoxDecodeStaticContract& contract,
                             BoxDecodeSourceStorageKind storage_kind) {
        for (auto& tensor : contract.tensors) {
          tensor.source_storage_kind = storage_kind;
        }
      };

      BoxDecodeStaticContract contract;
      contract.decode_type = BoxDecodeType::YoloV8;
      contract.input_dtype = "INT8";
      contract.tess_needed = true;
      contract.quant_needed = true;
      contract.tensors = {
          BoxDecodeTensorStaticContract{{80, 80, 64},
                                        {80, 80, 64},
                                        "INT8",
                                        "HWC",
                                        "reg_head0",
                                        "reg_head0",
                                        "reg_head0",
                                        0,
                                        0,
                                        -1,
                                        0,
                                        409600},
          BoxDecodeTensorStaticContract{{80, 80, 80},
                                        {80, 80, 80},
                                        "INT8",
                                        "HWC",
                                        "cls_head0",
                                        "cls_head0",
                                        "cls_head0",
                                        1,
                                        1,
                                        -1,
                                        0,
                                        512000},
      };
      contract.tensor_names = {"reg_head0", "cls_head0"};
      contract.physical_inputs = {
          BoxDecodePhysicalInputStaticContract{"reg_head0", -1, 0, 409600},
          BoxDecodePhysicalInputStaticContract{"cls_head0", -1, 0, 512000},
      };
      contract.dq_scale = {0.25, 0.125};
      contract.dq_zp = {4, 5};

      contract.quant_contract_required = true;
      contract.topk = 100;
      contract.detection_threshold = 0.25;
      contract.nms_iou_threshold = 0.55;
      contract.required_preprocess_meta_fields = {"orig_width", "orig_height"};
      mark_storage(contract, BoxDecodeSourceStorageKind::DenseHwcPhysical);

      const auto compiled = build_boxdecode_compiled_contract(contract);
      require(compiled.runtime_contract.logical_inputs.size() == 2U,
              "boxdecode contract should expose both logical inputs");
      require(compiled.payload.tensor_storage_kind.size() == 2U &&
                  compiled.payload.tensor_storage_kind[0] ==
                      static_cast<int>(BoxDecodeSourceStorageKind::DenseHwcPhysical),
              "boxdecode payload should preserve dense-HWC storage kind");
      require(compiled.payload.num_classes == 80,
              "boxdecode compiled contract should infer grouped YOLO class count from class-head "
              "depth when user num_classes is unset");
      require(compiled.runtime_contract.input_bindings.size() == 2U,
              "boxdecode contract should expose both input bindings");
      require(compiled.runtime_contract.required_preprocess_meta_fields.size() == 2U,
              "boxdecode contract should preserve required preprocess fields");
      require(compiled.runtime_contract.logical_inputs.front().quant.has_value(),
              "boxdecode logical input quant should be preserved");
      require(compiled.runtime_contract.output_quant.empty(),
              "boxdecode quant contract should stay on logical inputs");
      require(compiled.runtime_contract.logical_inputs.front().logical_name == "reg_head0",
              "boxdecode logical input name should be preserved");
      require(compiled.runtime_contract.logical_inputs.front().segment_name == "reg_head0",
              "boxdecode logical input segment should be preserved");
      require(compiled.runtime_contract.input_bindings.front().src_logical_output_index == 0,
              "boxdecode input binding should preserve upstream logical index");
      require(compiled.runtime_contract.input_bindings.front().src_output_slot == 0,
              "boxdecode input binding should preserve upstream output slot");
      require(compiled.runtime_contract.input_bindings.front().source_segment_name == "reg_head0",
              "boxdecode input binding should preserve upstream segment name");
      require(compiled.runtime_contract.input_bindings.front().src_physical_size_bytes == 409600U,
              "boxdecode physical input size should be preserved");
      const auto subset =
          plugin_contracts::extract_boxdecode_contract_subset_from_static_contract(contract);
      const auto subset_compiled = build_boxdecode_compiled_contract_from_subset(subset);
      require(subset_compiled.payload.decode_type == BoxDecodeType::YoloV8,
              "subset-based compiled contract should preserve decode_type");

      const auto finalized_user_classes =
          finalize_boxdecode_static_contract(contract, BoxDecodeType::YoloV8, std::nullopt,
                                             std::nullopt, BoxDecodeTypeOption::GroupedByRoleLogit,
                                             0.25, 0.55, 100, 42, {"orig_width", "orig_height"});
      require(finalized_user_classes.num_classes == 42,
              "explicit user num_classes should override MPK-inferred class depth");
      const auto compiled_user_classes = build_boxdecode_compiled_contract(finalized_user_classes);
      require(compiled_user_classes.payload.num_classes == 42,
              "compiled payload should preserve explicit user num_classes override");

      BoxDecodeStaticContract probability_domain = contract;
      probability_domain.decode_type_option = BoxDecodeTypeOption::GroupedByRole;
      probability_domain.score_activation = BoxDecodeScoreActivation::Identity;
      resolve_grouped_yolo_dfl_score_domain(&probability_domain);
      require(probability_domain.decode_type_option ==
                      BoxDecodeTypeOption::GroupedByRoleProbability &&
                  probability_domain.score_activation == BoxDecodeScoreActivation::Identity,
              "grouped YOLO DFL probability heads must not be forced through sigmoid");

      BoxDecodeStaticContract logit_domain = contract;
      logit_domain.decode_type_option = BoxDecodeTypeOption::GroupedByRole;
      logit_domain.score_activation = BoxDecodeScoreActivation::Sigmoid;
      resolve_grouped_yolo_dfl_score_domain(&logit_domain);
      require(logit_domain.decode_type_option == BoxDecodeTypeOption::GroupedByRoleLogit &&
                  logit_domain.score_activation == BoxDecodeScoreActivation::Sigmoid,
              "grouped YOLO DFL logit heads must preserve sigmoid activation");

      BoxDecodeStaticContract explicit_probability = contract;
      explicit_probability.decode_type_option = BoxDecodeTypeOption::GroupedByRoleProbability;
      explicit_probability.score_activation = BoxDecodeScoreActivation::Sigmoid;
      resolve_grouped_yolo_dfl_score_domain(&explicit_probability);
      require(explicit_probability.score_activation == BoxDecodeScoreActivation::Identity,
              "explicit grouped probability option must override inferred activation");

      BoxDecodeStaticContract packed_yolov5_contract;
      packed_yolov5_contract.decode_type = BoxDecodeType::YoloV5;
      packed_yolov5_contract.input_dtype = "INT8";
      packed_yolov5_contract.tensors = {
          BoxDecodeTensorStaticContract{{80, 80, 255},
                                        {80, 80, 255},
                                        "INT8",
                                        "HWC",
                                        "packed_head0",
                                        "packed_head0",
                                        "packed_head0",
                                        0,
                                        0,
                                        0,
                                        0,
                                        80U * 80U * 255U},
      };
      packed_yolov5_contract.tensor_names = {"packed_head0"};
      packed_yolov5_contract.physical_inputs = {
          BoxDecodePhysicalInputStaticContract{"packed_head0", 0, 0, 80U * 80U * 255U},
      };
      mark_storage(packed_yolov5_contract, BoxDecodeSourceStorageKind::DenseHwcPhysical);
      const auto packed_yolov5_finalized = finalize_boxdecode_static_contract(
          packed_yolov5_contract, BoxDecodeType::YoloV5, std::nullopt, std::nullopt,
          BoxDecodeTypeOption::PackedPerHead, 0.25, 0.55, 100, 0, {});
      require(packed_yolov5_finalized.num_classes == 80,
              "packed YOLO class count should be inferred from depth=3*(classes+5)");

      BoxDecodeStaticContract packed_contract;
      packed_contract.decode_type = BoxDecodeType::YoloV8;
      packed_contract.input_dtype = "INT8";
      packed_contract.tess_needed = true;
      packed_contract.quant_needed = true;
      packed_contract.tensors = {
          BoxDecodeTensorStaticContract{{80, 80, 64},
                                        {80, 80, 64},
                                        "INT8",
                                        "HWC",
                                        "reg_head0",
                                        "reg_head0",
                                        "reg_head0",
                                        0,
                                        0,
                                        0,
                                        128,
                                        409600},
      };
      packed_contract.tensor_names = {"reg_head0"};
      packed_contract.physical_inputs = {
          BoxDecodePhysicalInputStaticContract{"output_tensor", 0, 128, 409600},
      };
      packed_contract.dq_scale = {0.25};
      packed_contract.dq_zp = {4};
      packed_contract.quant_contract_required = true;
      packed_contract.topk = 100;
      packed_contract.detection_threshold = 0.25;
      packed_contract.nms_iou_threshold = 0.55;
      packed_contract.required_preprocess_meta_fields = {"orig_width", "orig_height"};
      mark_storage(packed_contract, BoxDecodeSourceStorageKind::PackedCBlock);

      const auto packed_compiled = build_boxdecode_compiled_contract(packed_contract);
      require(packed_compiled.runtime_contract.logical_inputs.size() == 1U,
              "packed boxdecode contract should expose one logical input");
      require(packed_compiled.payload.tensor_storage_kind.size() == 1U &&
                  packed_compiled.payload.tensor_storage_kind[0] ==
                      static_cast<int>(BoxDecodeSourceStorageKind::PackedCBlock),
              "packed boxdecode payload should preserve packed/cblock storage kind");
      require(packed_compiled.runtime_contract.logical_inputs.front().segment_name ==
                  "output_tensor",
              "packed boxdecode logical input should use the physical TensorBuffer parent segment");
      require(packed_compiled.runtime_contract.input_bindings.front().source_segment_name ==
                  "output_tensor",
              "packed boxdecode input binding should source the packed parent segment");
      require(packed_compiled.runtime_contract.input_bindings.front().src_physical_output_index ==
                  0,
              "packed boxdecode input binding should preserve physical output index");
      require(packed_compiled.runtime_contract.input_bindings.front().src_physical_byte_offset ==
                  128,
              "packed boxdecode input binding should preserve physical byte offset");

      setenv("SIMA_BOXDECODE_BYPASS_MLA_UNPACK", "1", 1);
      ModelManagedRouteFlags route_flags;
      route_flags.quant_needed = true;
      route_flags.tess_needed = false;
      route_flags.quant_contract_required = true;
      route_flags.boxdecode_selected = true;
      const auto finalized_contract =
          finalize_boxdecode_static_contract(packed_contract, BoxDecodeType::YoloV8, std::nullopt,
                                             route_flags, BoxDecodeTypeOption::GroupedByRoleLogit,
                                             0.25, 0.55, 100, 0, {"orig_width", "orig_height"});
      unsetenv("SIMA_BOXDECODE_BYPASS_MLA_UNPACK");
      require(finalized_contract.tess_needed,
              "raw-parent bypass should preserve contract-local tess semantics over route flags");
      require(finalized_contract.quant_needed,
              "raw-parent bypass should preserve contract-local quant semantics");
      require(finalized_contract.tensors.size() == 1U &&
                  finalized_contract.tensors.front().slice_shape == std::vector<int>{80, 80, 64},
              "raw-parent bypass should preserve contract slice geometry");
      require(finalized_contract.decode_type_option == BoxDecodeTypeOption::GroupedByRoleLogit,
              "finalized contract should preserve decode_type_option");
      const auto finalized_compiled = build_boxdecode_compiled_contract(finalized_contract);
      require(finalized_compiled.payload.decode_type_option ==
                  BoxDecodeTypeOption::GroupedByRoleLogit,
              "compiled payload should preserve decode_type_option");
      require(box_decode_type_option_token_string(BoxDecodeTypeOption::GroupedByRoleLogit) ==
                  "grouped-by-role-logit",
              "decode_type_option token string mismatch");
      const auto parsed_decode_type_option =
          parse_box_decode_type_option_token("grouped-by-role-logit");
      require(parsed_decode_type_option.has_value() &&
                  *parsed_decode_type_option == BoxDecodeTypeOption::GroupedByRoleLogit,
              "decode_type_option parser should round-trip grouped-by-role-logit");
      require(box_decode_type_token_string(BoxDecodeType::YoloV26) == "yolo26",
              "YoloV26 token string mismatch");
      const auto parsed_yolo26 = parse_box_decode_type_token("yolo26");
      require(parsed_yolo26.has_value() && *parsed_yolo26 == BoxDecodeType::YoloV26,
              "decode_type parser should round-trip yolo26");
      require(box_decode_type_token_string(BoxDecodeType::YoloV26Pose) == "yolo26-pose",
              "YoloV26Pose token string mismatch");
      const auto parsed_yolo26_pose = parse_box_decode_type_token("yolov26-pose");
      require(parsed_yolo26_pose.has_value() && *parsed_yolo26_pose == BoxDecodeType::YoloV26Pose,
              "decode_type parser should round-trip yolov26-pose alias");

      BoxDecodeStaticContract yolo26_contract;
      yolo26_contract.decode_type = BoxDecodeType::YoloV8;
      yolo26_contract.decode_type_option = BoxDecodeTypeOption::GroupedByRoleProbability;
      yolo26_contract.score_activation = BoxDecodeScoreActivation::Identity;
      yolo26_contract.input_dtype = "INT8";
      yolo26_contract.quant_needed = true;
      yolo26_contract.quant_contract_required = true;
      yolo26_contract.topk = 100;
      yolo26_contract.detection_threshold = 0.25;
      yolo26_contract.nms_iou_threshold = 0.55;
      for (int i = 0; i < 3; ++i) {
        const int width = i == 0 ? 80 : (i == 1 ? 40 : 20);
        yolo26_contract.tensors.push_back(BoxDecodeTensorStaticContract{
            {width, width, 4},
            {width, width, 4},
            "INT8",
            "HWC",
            "opaque_bbox_" + std::to_string(i),
            "opaque_bbox_" + std::to_string(i),
            "opaque_bbox_" + std::to_string(i),
            i,
            i,
            i,
            0,
            static_cast<std::uint64_t>(width * width * 4),
        });
        yolo26_contract.tensor_names.push_back("opaque_bbox_" + std::to_string(i));
      }
      for (int i = 0; i < 3; ++i) {
        const int width = i == 0 ? 80 : (i == 1 ? 40 : 20);
        yolo26_contract.tensors.push_back(BoxDecodeTensorStaticContract{
            {width, width, 80},
            {width, width, 80},
            "INT8",
            "HWC",
            "opaque_class_" + std::to_string(i),
            "opaque_class_" + std::to_string(i),
            "opaque_class_" + std::to_string(i),
            i + 3,
            i + 3,
            i + 3,
            0,
            static_cast<std::uint64_t>(width * width * 80),
        });
        yolo26_contract.tensor_names.push_back("opaque_class_" + std::to_string(i));
      }
      yolo26_contract.dq_scale.assign(6U, 0.125);
      yolo26_contract.dq_zp.assign(6U, 0);
      mark_storage(yolo26_contract, BoxDecodeSourceStorageKind::DenseHwcPhysical);

      const auto finalized_yolo26_inferred = finalize_boxdecode_static_contract(
          yolo26_contract, BoxDecodeType::YoloV26, std::nullopt, std::nullopt,
          BoxDecodeTypeOption::Auto, 0.25, 0.55, 100, 0, {"orig_width", "orig_height"});
      require(finalized_yolo26_inferred.num_classes == 80,
              "YOLO26 detection should infer class count from grouped class-logit heads");

      const auto finalized_yolo26 = finalize_boxdecode_static_contract(
          yolo26_contract, BoxDecodeType::YoloV26, std::nullopt, std::nullopt,
          BoxDecodeTypeOption::Auto, 0.25, 0.55, 100, 80, {"orig_width", "orig_height"});
      require(finalized_yolo26.decode_type == BoxDecodeType::YoloV26,
              "YOLO26 finalize should preserve the requested decode type");
      require(finalized_yolo26.decode_type_option == BoxDecodeTypeOption::GroupedByRoleLogit,
              "YOLO26 finalize should force grouped-by-role-logit");
      require(finalized_yolo26.score_activation == BoxDecodeScoreActivation::Sigmoid,
              "YOLO26 finalize should force sigmoid score activation");
      require(finalized_yolo26.tensors[0].logical_name == "bbox_0" &&
                  finalized_yolo26.tensors[3].logical_name == "class_logit_0",
              "YOLO26 finalize should synthesize canonical bbox/class-logit names");
      const auto compiled_yolo26 = build_boxdecode_compiled_contract(finalized_yolo26);
      require(compiled_yolo26.payload.decode_type == BoxDecodeType::YoloV26,
              "YOLO26 compiled payload should preserve decode type");
      require(compiled_yolo26.payload.decode_type_option == BoxDecodeTypeOption::GroupedByRoleLogit,
              "YOLO26 compiled payload should preserve grouped-by-role-logit");
      require(compiled_yolo26.payload.score_activation == BoxDecodeScoreActivation::Sigmoid,
              "YOLO26 compiled payload should preserve sigmoid activation");

      BoxDecodeStaticContract yolo26_pose_contract = yolo26_contract;
      yolo26_pose_contract.tensors.clear();
      yolo26_pose_contract.tensor_names.clear();
      yolo26_pose_contract.num_classes = 0;
      for (int i = 0; i < 3; ++i) {
        const int width = i == 0 ? 80 : (i == 1 ? 40 : 20);
        yolo26_pose_contract.tensors.push_back(BoxDecodeTensorStaticContract{
            {width, width, 16},
            {width, width, 4},
            "BF16",
            "HWC",
            "opaque_pose_bbox_" + std::to_string(i),
            "opaque_pose_bbox_" + std::to_string(i),
            "opaque_pose_bbox_" + std::to_string(i),
            i,
            i,
            i,
            0,
            static_cast<std::uint64_t>(width * width * 16 * 2),
        });
        yolo26_pose_contract.tensor_names.push_back("opaque_pose_bbox_" + std::to_string(i));
      }
      for (int i = 0; i < 3; ++i) {
        const int width = i == 0 ? 80 : (i == 1 ? 40 : 20);
        yolo26_pose_contract.tensors.push_back(BoxDecodeTensorStaticContract{
            {width, width, 16},
            {width, width, 1},
            "BF16",
            "HWC",
            "opaque_pose_score_" + std::to_string(i),
            "opaque_pose_score_" + std::to_string(i),
            "opaque_pose_score_" + std::to_string(i),
            i + 3,
            i + 3,
            i + 3,
            0,
            static_cast<std::uint64_t>(width * width * 16 * 2),
        });
        yolo26_pose_contract.tensor_names.push_back("opaque_pose_score_" + std::to_string(i));
      }
      for (int i = 0; i < 3; ++i) {
        const int width = i == 0 ? 80 : (i == 1 ? 40 : 20);
        yolo26_pose_contract.tensors.push_back(BoxDecodeTensorStaticContract{
            {width, width, 64},
            {width, width, 51},
            "BF16",
            "HWC",
            "opaque_pose_keypoint_" + std::to_string(i),
            "opaque_pose_keypoint_" + std::to_string(i),
            "opaque_pose_keypoint_" + std::to_string(i),
            i + 6,
            i + 6,
            i + 6,
            0,
            static_cast<std::uint64_t>(width * width * 64 * 2),
        });
        yolo26_pose_contract.tensor_names.push_back("opaque_pose_keypoint_" + std::to_string(i));
      }
      yolo26_pose_contract.dq_scale.assign(9U, 1.0);
      yolo26_pose_contract.dq_zp.assign(9U, 0);
      mark_storage(yolo26_pose_contract, BoxDecodeSourceStorageKind::DenseHwcPhysical);
      const auto finalized_yolo26_pose = finalize_boxdecode_static_contract(
          yolo26_pose_contract, BoxDecodeType::YoloV26Pose, std::nullopt, std::nullopt,
          BoxDecodeTypeOption::Auto, 0.25, 0.55, 100, 0, {"orig_width", "orig_height"});
      require(finalized_yolo26_pose.decode_type == BoxDecodeType::YoloV26Pose,
              "YOLO26-pose finalize should preserve the requested decode type");
      require(finalized_yolo26_pose.num_classes == 1,
              "YOLO26-pose finalize should default to one pose score class");
      require(finalized_yolo26_pose.decode_type_option == BoxDecodeTypeOption::GroupedByRoleLogit,
              "YOLO26-pose finalize should force grouped-by-role-logit");
      require(finalized_yolo26_pose.score_activation == BoxDecodeScoreActivation::Sigmoid,
              "YOLO26-pose finalize should force sigmoid score activation");
      require(finalized_yolo26_pose.tensors[0].logical_name == "bbox_0" &&
                  finalized_yolo26_pose.tensors[3].logical_name == "class_logit_0" &&
                  finalized_yolo26_pose.tensors[6].logical_name == "keypoint_0",
              "YOLO26-pose finalize should synthesize canonical bbox/score/keypoint names");
      const auto compiled_yolo26_pose = build_boxdecode_compiled_contract(finalized_yolo26_pose);
      require(compiled_yolo26_pose.payload.decode_type == BoxDecodeType::YoloV26Pose,
              "YOLO26-pose compiled payload should preserve decode type");
      require(compiled_yolo26_pose.payload.num_classes == 1,
              "YOLO26-pose compiled payload should preserve one pose score class");

      BoxDecodeStaticContract yolo26_seg_contract = yolo26_contract;
      yolo26_seg_contract.tensors.clear();
      yolo26_seg_contract.tensor_names.clear();
      yolo26_seg_contract.num_classes = 0;
      for (int i = 0; i < 3; ++i) {
        const int width = i == 0 ? 80 : (i == 1 ? 40 : 20);
        yolo26_seg_contract.tensors.push_back(BoxDecodeTensorStaticContract{
            {width, width, 16},
            {width, width, 4},
            "BF16",
            "HWC",
            "opaque_seg_bbox_" + std::to_string(i),
            "opaque_seg_bbox_" + std::to_string(i),
            "opaque_seg_bbox_" + std::to_string(i),
            i,
            i,
            i,
            0,
            static_cast<std::uint64_t>(width * width * 16 * 2),
        });
        yolo26_seg_contract.tensor_names.push_back("opaque_seg_bbox_" + std::to_string(i));
      }
      for (int i = 0; i < 3; ++i) {
        const int width = i == 0 ? 80 : (i == 1 ? 40 : 20);
        yolo26_seg_contract.tensors.push_back(BoxDecodeTensorStaticContract{
            {width, width, 80},
            {width, width, 80},
            "BF16",
            "HWC",
            "opaque_seg_class_" + std::to_string(i),
            "opaque_seg_class_" + std::to_string(i),
            "opaque_seg_class_" + std::to_string(i),
            i + 3,
            i + 3,
            i + 3,
            0,
            static_cast<std::uint64_t>(width * width * 80 * 2),
        });
        yolo26_seg_contract.tensor_names.push_back("opaque_seg_class_" + std::to_string(i));
      }
      for (int i = 0; i < 3; ++i) {
        const int width = i == 0 ? 80 : (i == 1 ? 40 : 20);
        yolo26_seg_contract.tensors.push_back(BoxDecodeTensorStaticContract{
            {width, width, 32},
            {width, width, 32},
            "BF16",
            "HWC",
            "opaque_seg_mask_" + std::to_string(i),
            "opaque_seg_mask_" + std::to_string(i),
            "opaque_seg_mask_" + std::to_string(i),
            i + 6,
            i + 6,
            i + 6,
            0,
            static_cast<std::uint64_t>(width * width * 32 * 2),
        });
        yolo26_seg_contract.tensor_names.push_back("opaque_seg_mask_" + std::to_string(i));
      }
      yolo26_seg_contract.tensors.push_back(BoxDecodeTensorStaticContract{
          {160, 160, 32},
          {160, 160, 32},
          "BF16",
          "HWC",
          "opaque_seg_proto",
          "opaque_seg_proto",
          "opaque_seg_proto",
          9,
          9,
          9,
          0,
          static_cast<std::uint64_t>(160 * 160 * 32 * 2),
      });
      yolo26_seg_contract.tensor_names.push_back("opaque_seg_proto");
      yolo26_seg_contract.dq_scale.assign(10U, 1.0);
      yolo26_seg_contract.dq_zp.assign(10U, 0);
      mark_storage(yolo26_seg_contract, BoxDecodeSourceStorageKind::DenseHwcPhysical);
      const auto finalized_yolo26_seg = finalize_boxdecode_static_contract(
          yolo26_seg_contract, BoxDecodeType::YoloV26Seg, std::nullopt, std::nullopt,
          BoxDecodeTypeOption::Auto, 0.25, 0.55, 100, 0, {"orig_width", "orig_height"});
      require(finalized_yolo26_seg.num_classes == 80,
              "YOLO26-seg should infer class count from class heads and ignore mask/proto heads");

      BoxDecodeStaticContract yolox_contract;
      yolox_contract.decode_type = BoxDecodeType::YoloX;
      yolox_contract.input_dtype = "BF16";
      for (int i = 0; i < 3; ++i) {
        const int width = i == 0 ? 80 : (i == 1 ? 40 : 20);
        yolox_contract.tensors.push_back(
            BoxDecodeTensorStaticContract{{width, width, 8},
                                          {width, width, 4},
                                          "BF16",
                                          "HWC",
                                          "yolox_bbox_" + std::to_string(i),
                                          "yolox_bbox_" + std::to_string(i),
                                          "yolox_bbox_" + std::to_string(i),
                                          i * 3,
                                          i * 3,
                                          i * 3,
                                          0,
                                          static_cast<std::uint64_t>(width * width * 8 * 2)});
        yolox_contract.tensors.push_back(
            BoxDecodeTensorStaticContract{{width, width, 8},
                                          {width, width, 1},
                                          "BF16",
                                          "HWC",
                                          "yolox_obj_logit_" + std::to_string(i),
                                          "yolox_obj_logit_" + std::to_string(i),
                                          "yolox_obj_logit_" + std::to_string(i),
                                          (i * 3) + 1,
                                          (i * 3) + 1,
                                          (i * 3) + 1,
                                          0,
                                          static_cast<std::uint64_t>(width * width * 8 * 2)});
        yolox_contract.tensors.push_back(
            BoxDecodeTensorStaticContract{{width, width, 80},
                                          {width, width, 80},
                                          "BF16",
                                          "HWC",
                                          "yolox_class_logit_" + std::to_string(i),
                                          "yolox_class_logit_" + std::to_string(i),
                                          "yolox_class_logit_" + std::to_string(i),
                                          (i * 3) + 2,
                                          (i * 3) + 2,
                                          (i * 3) + 2,
                                          0,
                                          static_cast<std::uint64_t>(width * width * 80 * 2)});
      }
      yolox_contract.dq_scale.assign(9U, 1.0);
      yolox_contract.dq_zp.assign(9U, 0);
      mark_storage(yolox_contract, BoxDecodeSourceStorageKind::DenseHwcPhysical);
      const auto finalized_yolox = finalize_boxdecode_static_contract(
          yolox_contract, BoxDecodeType::YoloX, std::nullopt, std::nullopt,
          BoxDecodeTypeOption::Auto, 0.25, 0.55, 100, 0, {});
      require(finalized_yolox.num_classes == 80,
              "YOLOX should infer class count from class heads while ignoring objectness");

      // Named-node precedence is sentinel-aware: Auto/-1 inherit Model choices, while concrete
      // wire-format selections remain node-owned.
      SuperPointOptions inherited_options;
      inherited_options.profile = SuperPointProfile::MagicLeapDemoV1;
      inherited_options.nms_radius = 2;
      inherited_options.border_margin = 0;
      inherited_options.descriptor_output_dtype = TensorDType::Int8;
      inherited_options.output_format = SuperPointOutputFormat::LegacyA65InterleavedV0;
      const SuperPointOptions default_node_options;
      require(default_node_options.profile == SuperPointProfile::Auto,
              "public Auto sentinel must remain available for Model/MPK inheritance");
      require(parse_superpoint_profile_token("a65-v1") == SuperPointProfile::A65V1 &&
                  !parse_superpoint_profile_token("legacy-a65-v1").has_value() &&
                  std::string(superpoint_profile_token(SuperPointProfile::A65V1)) == "a65-v1",
              "A65V1 token/name contract changed or deprecated profile alias leaked");
      const auto inherited_merge =
          merge_superpoint_node_options(inherited_options, default_node_options);
      require(inherited_merge.profile == SuperPointProfile::MagicLeapDemoV1 &&
                  inherited_merge.nms_radius == 2 && inherited_merge.border_margin == 0,
              "node Auto/-1 options must not erase explicit Model profile/spatial options");
      require(inherited_merge.descriptor_output_dtype == TensorDType::Float32 &&
                  inherited_merge.output_format == SuperPointOutputFormat::FeaturePointsV1,
              "node descriptor dtype/output format are concrete selections, not sentinels");
      SuperPointOptions explicit_node_options;
      explicit_node_options.profile = SuperPointProfile::A65V1;
      explicit_node_options.nms_radius = 0;
      explicit_node_options.border_margin = 1;
      const auto explicit_merge =
          merge_superpoint_node_options(inherited_options, explicit_node_options);
      require(explicit_merge.profile == SuperPointProfile::A65V1 &&
                  explicit_merge.nms_radius == 0 && explicit_merge.border_margin == 1,
              "explicit node profile/spatial options must override Model options and preserve "
              "radius zero");

      auto make_superpoint_contract = [] {
        BoxDecodeStaticContract sp;
        sp.decode_type = BoxDecodeType::SuperPoint;
        sp.input_dtype = "FP32";
        sp.topk = 600;
        sp.superpoint.schema_version = 1;
        sp.superpoint.profile = SuperPointProfile::LightGlueV1;
        sp.superpoint.profile_from_mpk = true;
        sp.superpoint.fingerprint_profile = SuperPointProfile::LightGlueV1;
        sp.superpoint.profile_fingerprint =
            "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        sp.superpoint.detector_tensor_id = "semi";
        sp.superpoint.descriptor_tensor_id = "desc";
        sp.superpoint.detector_representation = "raw-logits-65";
        sp.superpoint.descriptor_representation = "coarse-pre-l2";
        sp.superpoint.descriptor_dim = 256;
        BoxDecodeTensorStaticContract detector;
        detector.input_shape = {60, 80, 65};
        detector.slice_shape = detector.input_shape;
        detector.data_type = "FP32";
        detector.layout = "HWC";
        detector.logical_name = "semi";
        detector.backend_name = "semi";
        detector.source_segment_name = "semi";
        detector.source_logical_output_index = 0;
        detector.source_output_slot = 0;
        detector.source_size_bytes = 60U * 80U * 65U * 4U;
        detector.source_storage_kind = BoxDecodeSourceStorageKind::DenseHwcPhysical;
        detector.role = BoxDecodeTensorRole::DetectorLogits;
        BoxDecodeTensorStaticContract descriptor = detector;
        descriptor.input_shape = {60, 80, 256};
        descriptor.slice_shape = descriptor.input_shape;
        descriptor.logical_name = "desc";
        descriptor.backend_name = "desc";
        descriptor.source_segment_name = "desc";
        descriptor.source_logical_output_index = 1;
        descriptor.source_output_slot = 1;
        descriptor.source_size_bytes = 60U * 80U * 256U * 4U;
        descriptor.role = BoxDecodeTensorRole::DescriptorGrid;
        sp.tensors = {detector, descriptor};
        sp.tensor_names = {"semi", "desc"};
        sp.dq_scale = {1.0, 1.0};
        sp.dq_zp = {0, 0};
        return sp;
      };

      const auto valid_superpoint = build_boxdecode_compiled_contract(make_superpoint_contract());
      require(valid_superpoint.payload.superpoint.profile == SuperPointProfile::LightGlueV1 &&
                  valid_superpoint.payload.superpoint.nms_radius == 4 &&
                  valid_superpoint.payload.superpoint.border_margin == 4 &&
                  valid_superpoint.payload.detection_threshold == 5.0e-4 &&
                  valid_superpoint.payload.topk == 600 && valid_superpoint.payload.num_classes == 0,
              "valid schema-v1 SuperPoint contract must preserve profile and apply defaults");

      auto valid_packed_geometry = make_superpoint_contract();
      valid_packed_geometry.tensors[0].slice_shape = {12, 4, 65};
      valid_packed_geometry.tensors[1].slice_shape = {12, 16, 64};
      valid_packed_geometry.tensors[0].source_storage_kind =
          BoxDecodeSourceStorageKind::PackedCBlock;
      valid_packed_geometry.tensors[1].source_storage_kind =
          BoxDecodeSourceStorageKind::PackedCBlock;
      const auto compiled_packed_geometry =
          build_boxdecode_compiled_contract(valid_packed_geometry);
      require(compiled_packed_geometry.runtime_contract.logical_inputs[1].shape ==
                  std::vector<std::int64_t>({60, 80, 256}),
              "packed SuperPoint descriptor must keep its full C256 logical frame when the "
              "physical tile is C64");
      require(compiled_packed_geometry.payload.slice_shapes[1].sizes[2] == 64,
              "packed SuperPoint descriptor must retain the authoritative C64 tile geometry");

      struct SuperPointDTypeCase {
        const char* token;
        std::size_t bytes;
        TensorDType output_dtype;
      };
      const std::array dtype_cases = {
          SuperPointDTypeCase{"INT8", 1U, TensorDType::Int8},
          SuperPointDTypeCase{"BF16", 2U, TensorDType::BFloat16},
          SuperPointDTypeCase{"FP32", 4U, TensorDType::Float32},
      };
      const std::array storage_cases = {
          BoxDecodeSourceStorageKind::DenseHwcPhysical,
          BoxDecodeSourceStorageKind::PackedCBlock,
          BoxDecodeSourceStorageKind::PackedHwcC16,
      };
      for (const auto& input_dtype : dtype_cases) {
        for (const auto storage : storage_cases) {
          for (const auto& output_dtype : dtype_cases) {
            auto matrix_contract = make_superpoint_contract();
            matrix_contract.input_dtype = input_dtype.token;
            matrix_contract.superpoint.descriptor_output_dtype = output_dtype.output_dtype;
            for (auto& tensor : matrix_contract.tensors) {
              tensor.data_type = input_dtype.token;
              tensor.source_storage_kind = storage;
              tensor.source_size_bytes = static_cast<std::size_t>(tensor.input_shape[0]) *
                                         static_cast<std::size_t>(tensor.input_shape[1]) *
                                         static_cast<std::size_t>(tensor.input_shape[2]) *
                                         input_dtype.bytes;
            }
            if (storage != BoxDecodeSourceStorageKind::DenseHwcPhysical) {
              matrix_contract.tensors[0].slice_shape = {12, 4, 65};
              matrix_contract.tensors[1].slice_shape = {12, 16, 64};
            }

            const auto matrix_compiled = build_boxdecode_compiled_contract(matrix_contract);
            require(
                matrix_compiled.payload.input_dtype == input_dtype.token &&
                    matrix_compiled.payload.tensor_storage_kind.size() == 2U &&
                    matrix_compiled.payload.tensor_storage_kind[0] == static_cast<int>(storage) &&
                    matrix_compiled.payload.tensor_storage_kind[1] == static_cast<int>(storage) &&
                    matrix_compiled.payload.superpoint.descriptor_output_dtype ==
                        output_dtype.output_dtype,
                "SuperPoint INT8/BF16/FP32 dense/tessellated contract matrix changed");
          }
        }
      }

      auto manual_schema0 = make_superpoint_contract();
      manual_schema0.superpoint.schema_version = 0;
      manual_schema0.superpoint.profile_from_mpk = false;
      manual_schema0.superpoint.fingerprint_profile = SuperPointProfile::Auto;
      manual_schema0.superpoint.profile_fingerprint.clear();
      manual_schema0.superpoint.detector_tensor_id.clear();
      manual_schema0.superpoint.descriptor_tensor_id.clear();
      manual_schema0.superpoint.detector_representation.clear();
      manual_schema0.superpoint.descriptor_representation.clear();
      const auto compiled_manual_schema0 = build_boxdecode_compiled_contract(manual_schema0);
      require(compiled_manual_schema0.payload.superpoint.detector_representation ==
                      "raw-logits-65" &&
                  compiled_manual_schema0.payload.superpoint.descriptor_representation ==
                      "coarse-pre-l2" &&
                  compiled_manual_schema0.payload.superpoint.representations_defaulted,
              "manual/schema-0 SuperPoint contracts must publish canonical runtime "
              "representations with default provenance");

      auto default_profile_schema0 = manual_schema0;
      default_profile_schema0.superpoint.profile = SuperPointProfile::Auto;
      default_profile_schema0.topk = 0;
      default_profile_schema0.detection_threshold = 0.0;
      const auto compiled_default_profile =
          build_boxdecode_compiled_contract(default_profile_schema0);
      require(compiled_default_profile.payload.superpoint.profile == SuperPointProfile::A65V1 &&
                  compiled_default_profile.payload.superpoint.nms_radius == 4 &&
                  compiled_default_profile.payload.superpoint.border_margin == 0 &&
                  compiled_default_profile.payload.topk == 600 &&
                  compiled_default_profile.payload.detection_threshold == 0.1,
              "unresolved schema-0 SuperPoint contract must select the complete A65V1 "
              "default policy");

      auto magic_defaults_schema0 = manual_schema0;
      magic_defaults_schema0.superpoint.profile = SuperPointProfile::MagicLeapDemoV1;
      magic_defaults_schema0.topk = 0;
      magic_defaults_schema0.detection_threshold = 0.0;
      const auto compiled_magic_defaults =
          build_boxdecode_compiled_contract(magic_defaults_schema0);
      require(compiled_magic_defaults.payload.detection_threshold == 0.015 &&
                  compiled_magic_defaults.payload.topk == 600 &&
                  compiled_magic_defaults.payload.superpoint.nms_radius == 4 &&
                  compiled_magic_defaults.payload.superpoint.border_margin == 4,
              "MagicLeapDemoV1 profile defaults changed");

      auto profile_override = make_superpoint_contract().superpoint;
      profile_override.nms_radius = 4;
      profile_override.border_margin = 4;
      profile_override.nms_radius_from_mpk = false;
      profile_override.border_margin_from_mpk = false;
      require(apply_superpoint_profile_override(&profile_override, SuperPointProfile::A65V1) &&
                  profile_override.profile == SuperPointProfile::A65V1 &&
                  profile_override.nms_radius == -1 && profile_override.border_margin == -1,
              "profile overrides must clear spatial defaults materialized for the old profile");
      require(rebase_superpoint_detection_threshold(true, SuperPointProfile::A65V1, 0.0, 5.0e-4) ==
                  0.1,
              "profile overrides must rebase a defaulted detection threshold");

      auto authored_spatial_override = make_superpoint_contract().superpoint;
      authored_spatial_override.nms_radius = 7;
      authored_spatial_override.border_margin = 3;
      authored_spatial_override.nms_radius_from_mpk = true;
      authored_spatial_override.border_margin_from_mpk = true;
      require(apply_superpoint_profile_override(&authored_spatial_override,
                                                SuperPointProfile::MagicLeapDemoV1) &&
                  authored_spatial_override.nms_radius == 7 &&
                  authored_spatial_override.border_margin == 3,
              "profile overrides must preserve explicitly authored MPK spatial controls");

      auto cached_lightglue_contract = make_superpoint_contract();
      cached_lightglue_contract.superpoint.schema_version = 0;
      cached_lightglue_contract.superpoint.profile_fingerprint.clear();
      cached_lightglue_contract.superpoint.fingerprint_profile = SuperPointProfile::Auto;
      cached_lightglue_contract.superpoint.nms_radius = 4;
      cached_lightglue_contract.superpoint.border_margin = 4;
      BoxDecodeCompiledContractOptions a65_override_options;
      a65_override_options.decode_type = BoxDecodeType::SuperPoint;
      a65_override_options.superpoint = SuperPointStaticContract{};
      a65_override_options.superpoint->profile = SuperPointProfile::A65V1;
      const auto rebased_a65 = build_boxdecode_compiled_contract_from_subset(
          plugin_contracts::extract_boxdecode_contract_subset_from_static_contract(
              cached_lightglue_contract),
          a65_override_options);
      require(rebased_a65.payload.superpoint.profile == SuperPointProfile::A65V1 &&
                  rebased_a65.payload.superpoint.nms_radius == 4 &&
                  rebased_a65.payload.superpoint.border_margin == 0 &&
                  rebased_a65.payload.detection_threshold == 0.1,
              "compiled subset profile overrides must replace cached LightGlue defaults with "
              "A65 defaults");

      auto require_superpoint_rejection = [&](BoxDecodeStaticContract invalid,
                                              const std::string& diagnostic) {
        bool rejected = false;
        try {
          (void)build_boxdecode_compiled_contract(invalid);
        } catch (const std::invalid_argument& e) {
          rejected = std::string(e.what()).find(diagnostic) != std::string::npos;
        }
        require(rejected, "malformed SuperPoint contract did not fail with expected diagnostic: " +
                              diagnostic);
      };
      auto bad_fingerprint = make_superpoint_contract();
      bad_fingerprint.superpoint.profile_fingerprint = "sha256:not-a-digest";
      require_superpoint_rejection(std::move(bad_fingerprint), "64 hexadecimal");
      auto missing_fingerprint = make_superpoint_contract();
      missing_fingerprint.superpoint.profile_fingerprint.clear();
      require_superpoint_rejection(std::move(missing_fingerprint), "requires profile_fingerprint");
      auto bad_representation = make_superpoint_contract();
      bad_representation.superpoint.detector_representation = "probabilities-64";
      require_superpoint_rejection(std::move(bad_representation),
                                   "unsupported detector_representation");
      auto mismatched_fingerprint = make_superpoint_contract();
      mismatched_fingerprint.superpoint.profile = SuperPointProfile::MagicLeapDemoV1;
      mismatched_fingerprint.superpoint.profile_from_mpk = false;
      require_superpoint_rejection(std::move(mismatched_fingerprint), "re-stamp the MPK");
      auto unknown_schema = make_superpoint_contract();
      unknown_schema.superpoint.schema_version = 2;
      require_superpoint_rejection(std::move(unknown_schema), "supported versions");
      auto bad_tensor_id = make_superpoint_contract();
      bad_tensor_id.superpoint.detector_tensor_id = "missing-semi";
      require_superpoint_rejection(std::move(bad_tensor_id), "does not match an input tensor");
      auto bad_geometry = make_superpoint_contract();
      bad_geometry.tensors[1].input_shape[0] = 59;
      require_superpoint_rejection(std::move(bad_geometry), "coarse H/W geometry");
      auto bad_input_dtype = make_superpoint_contract();
      bad_input_dtype.tensors[1].data_type = "UINT8";
      require_superpoint_rejection(std::move(bad_input_dtype), "input dtypes");
      auto duplicate_roles = manual_schema0;
      duplicate_roles.tensors[1].role = BoxDecodeTensorRole::DetectorLogits;
      require_superpoint_rejection(std::move(duplicate_roles), "duplicate explicit roles");

      BoxDecodeStaticContract paper_superpoint;
      paper_superpoint.decode_type = BoxDecodeType::SuperPoint;
      paper_superpoint.superpoint.profile = SuperPointProfile::PaperBicubicV1;
      bool paper_profile_rejected = false;
      try {
        (void)build_boxdecode_compiled_contract(paper_superpoint);
      } catch (const std::invalid_argument& e) {
        paper_profile_rejected = std::string(e.what()).find("reserved") != std::string::npos;
      }
      require(paper_profile_rejected,
              "PaperBicubicV1 must fail contract compilation until production-defined");
    }));

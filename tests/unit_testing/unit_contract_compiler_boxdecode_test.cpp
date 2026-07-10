#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "pipeline/internal/sima/BoxDecodeTypeUtils.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "test_main.h"

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
    }));

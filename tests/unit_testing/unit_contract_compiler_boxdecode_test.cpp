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

      const auto compiled = build_boxdecode_compiled_contract(contract);
      require(compiled.runtime_contract.logical_inputs.size() == 2U,
              "boxdecode contract should expose both logical inputs");
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

      const auto packed_compiled = build_boxdecode_compiled_contract(packed_contract);
      require(packed_compiled.runtime_contract.logical_inputs.size() == 1U,
              "packed boxdecode contract should expose one logical input");
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
    }));

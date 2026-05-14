#include "pipeline/internal/sima/stagesemantics/DequantStageSemantics.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapter.h"
#include "test_main.h"

RUN_TEST(
    "unit_dequant_processcvu_contract_equivalence_test", ([] {
      using namespace simaai::neat::pipeline_internal::sima::stagesemantics;
      DequantCanonicalFacts dequant_facts;
      dequant_facts.input_dtype = "INT8";
      dequant_facts.output_dtype = "FP32";
      dequant_facts.layout = "HWC";
      dequant_facts.output_representation = ProcessCvuOutputRepresentation::PackedBlob;
      dequant_facts.input_quant = simaai::neat::pipeline_internal::sima::QuantStaticSpec{
          simaai::neat::pipeline_internal::sima::QuantGranularity::PerTensor, -1, {0.125}, {-7}};

      CompiledProcessCvuRuntimeConfig runtime;
      runtime.graph_family = "dequantize";
      runtime.graph_name = "dequantize";
      runtime.graph_id = 6;
      runtime.default_input_name = "input_tensor";
      runtime.runtime_input_names = {"input_tensor"};
      runtime.runtime_output_names = {"output_tensor"};
      runtime.published_output_names = {"output_tensor"};
      runtime.physical_input_names = {"input_tensor"};
      runtime.physical_output_names = {"output_tensor"};
      runtime.primary_output_name = "output_tensor";
      runtime.input_shapes = {{1, 1, 1}};
      runtime.output_shapes = {{1, 1, 1}};
      runtime.input_dtype = "INT8";
      runtime.output_dtype = "FP32";
      runtime.out_dtype = "FP32";
      {
        sima_ev_tensor_desc input_desc{};
        sima_ev_tensor_desc output_desc{};
        std::string error_detail;
        require(simaai::neat::pipeline_internal::sima::tensorsemantics::build_dense_tensor_desc(
                    runtime.input_shapes.front(), runtime.input_dtype, "HWC", &input_desc,
                    &error_detail, "test_input_tensor_output_missing",
                    "test_input_shape_rank_invalid", "test_input_shape_dim_invalid",
                    "test_input_dtype_invalid", "test_input_stride_output_missing"),
                "dequant runtime test should synthesize typed input tensor");
        require(simaai::neat::pipeline_internal::sima::tensorsemantics::build_dense_tensor_desc(
                    runtime.output_shapes.front(), runtime.output_dtype, "HWC", &output_desc,
                    &error_detail, "test_output_tensor_output_missing",
                    "test_output_shape_rank_invalid", "test_output_shape_dim_invalid",
                    "test_output_dtype_invalid", "test_output_stride_output_missing"),
                "dequant runtime test should synthesize typed output tensor");
        runtime.input_tensors = {input_desc};
        runtime.output_tensors = {output_desc};
        runtime.runtime_output_logical_layout_list = {"HWC"};
      }
      runtime.has_q_scale = true;
      runtime.q_scale = 0.125;
      runtime.has_q_zp = true;
      runtime.q_zp = -7;
      runtime.q_scale_list = {0.125};
      runtime.q_zp_list = {-7};

      const auto expected = build_processcvu_compiled_contract_from_runtime_config(runtime);
      const auto adapted = build_dequant_compiled_contract_from_facts(dequant_facts);

      require(adapted.runtime_contract.logical_inputs.size() ==
                  expected.runtime_contract.logical_inputs.size(),
              "dequant facts should preserve logical input count across family builders");
      require(adapted.runtime_contract.input_bindings.size() ==
                  expected.runtime_contract.input_bindings.size(),
              "dequant facts should preserve binding count across family builders");
      require(adapted.runtime_contract.logical_outputs.size() ==
                  expected.runtime_contract.logical_outputs.size(),
              "dequant facts should preserve logical output count across family builders");
      require(adapted.runtime_contract.output_order.size() ==
                  expected.runtime_contract.output_order.size(),
              "dequant facts should preserve output route count across family builders");
      require(adapted.runtime_contract.logical_inputs.front().dtype ==
                  expected.runtime_contract.logical_inputs.front().dtype,
              "dequant facts should preserve logical input dtype");
      require(adapted.runtime_contract.logical_outputs.front().dtype ==
                  expected.runtime_contract.logical_outputs.front().dtype,
              "dequant facts should preserve logical output dtype");
      require(adapted.runtime_contract.logical_inputs.front().quant.has_value(),
              "dequant facts should preserve input quant facts");
      require(adapted.runtime_contract.logical_inputs.front().quant->scales ==
                  expected.runtime_contract.logical_inputs.front().quant->scales,
              "dequant facts should preserve quant scales");
      require(adapted.runtime_contract.logical_inputs.front().quant->zero_points ==
                  expected.runtime_contract.logical_inputs.front().quant->zero_points,
              "dequant facts should preserve quant zero-points");
      require(adapted.runtime_contract.input_bindings.front().source_segment_name ==
                  expected.runtime_contract.input_bindings.front().source_segment_name,
              "dequant facts should preserve input binding segment");
      require(adapted.runtime_contract.output_order.front().segment_name ==
                  expected.runtime_contract.output_order.front().segment_name,
              "dequant facts should preserve output route segment");
    }));

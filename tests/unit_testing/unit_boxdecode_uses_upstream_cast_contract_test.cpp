#include "nodes/sima/Cast.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/SessionOptions.h"
#include "pipeline/Tensor.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/TensorMath.h"
#include "test_main.h"
#include "test_utils.h"

namespace {

simaai::neat::Tensor make_fp32_head(int logical_index, const std::vector<int64_t>& shape,
                                    int64_t physical_offset, const std::string& name,
                                    float fill_value = 0.0f) {
  using namespace simaai::neat;
  Tensor tensor;
  tensor.dtype = TensorDType::Float32;
  tensor.layout = TensorLayout::CHW;
  tensor.shape = shape;
  tensor.strides_bytes = simaai::neat::pipeline_internal::contiguous_strides_bytes(shape, 4U);

  std::size_t bytes = 4U;
  for (const auto dim : shape) {
    bytes *= static_cast<std::size_t>(dim);
  }
  tensor.storage = make_cpu_owned_storage(bytes);
  {
    auto map = tensor.storage->map(MapMode::Write);
    const auto storage_ptr = static_cast<float*>(map.data);
    const std::size_t elems = bytes / sizeof(float);
    for (std::size_t i = 0; i < elems; ++i) {
      storage_ptr[i] = fill_value;
    }
  }
  tensor.route.logical_index = logical_index;
  tensor.route.route_slot = logical_index;
  tensor.route.physical_index = 0;
  tensor.route.memory_index = 0;
  tensor.route.physical_byte_offset = physical_offset;
  tensor.route.name = name;
  tensor.route.backend_name = name;
  tensor.route.segment_name = name;
  return tensor;
}

} // namespace

RUN_TEST(
    "unit_boxdecode_uses_upstream_cast_contract_test", ([] {
      using namespace simaai::neat;

      Tensor reg0 = make_fp32_head(0, {64, 80, 80}, 0, "reg_head_0");
      Tensor reg1 = make_fp32_head(1, {64, 40, 40}, 1638400, "reg_head_1");
      Tensor reg2 = make_fp32_head(2, {64, 20, 20}, 2048000, "reg_head_2");
      Tensor cls0 = make_fp32_head(3, {80, 80, 80}, 2150400, "cls_head_0", 8.0f);
      Tensor cls1 = make_fp32_head(4, {80, 40, 40}, 4198400, "cls_head_1", 8.0f);
      Tensor cls2 = make_fp32_head(5, {80, 20, 20}, 4710400, "cls_head_2", 8.0f);
      const Sample ingress = sample_from_tensors(TensorList{reg0, reg1, reg2, cls0, cls1, cls2});

      std::vector<std::shared_ptr<Node>> nodes;
      nodes.push_back(simaai::neat::nodes::Cast(
          CastOptions{.direction = CastDirection::Fp32ToBf16, .element_name = "pre_box_cast"}));
      nodes.push_back(simaai::neat::nodes::SimaBoxDecode(BoxDecodeType::YoloV8, 0.25, 0.45, 100,
                                                         "manual_boxdecode", 1280, 720, 640, 640));

      ContractCompileInput input;
      input.ingress.ingress_sample = ingress;
      pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
      const auto compiled = compile_node_contracts(nodes, input, &diagnostics);

      require(diagnostics.errors.empty(),
              "compile_node_contracts should succeed for Cast -> SimaBoxDecode");
      require(compiled.stages.size() == 2U, "expected cast and boxdecode stages");
      require(compiled.stages[0].transport.has_value(),
              "first stage should compile as cast transport");
      require(compiled.stages[1].boxdecode.has_value(), "second stage should compile as boxdecode");

      const auto& cast_runtime = compiled.stages[0].transport->runtime_contract;
      require(cast_runtime.logical_outputs.size() == 6U,
              "cast contract should publish six logical outputs");
      require(cast_runtime.logical_outputs[0].size_bytes == 819200U,
              "cast contract should convert first logical size to BF16 bytes");
      require(cast_runtime.logical_outputs[1].byte_offset == 819200,
              "cast contract should convert second logical byte offset to BF16 bytes");
      require(cast_runtime.logical_outputs[1].size_bytes == 204800U,
              "cast contract should convert second logical size to BF16 bytes");
      require(cast_runtime.logical_outputs[5].byte_offset == 2355200,
              "cast contract should preserve grouped-by-role BF16 head offsets");
      require(cast_runtime.logical_outputs[5].size_bytes == 64000U,
              "cast contract should convert final logical size to BF16 bytes");

      const auto& boxdecode_runtime = compiled.stages[1].boxdecode->runtime_contract;
      require(boxdecode_runtime.logical_inputs.size() == 6U,
              "boxdecode contract should inherit six logical inputs from cast");
      require(boxdecode_runtime.logical_inputs[3].byte_offset == 1075200,
              "boxdecode logical inputs should preserve grouped-by-role BF16 offsets");
      require(boxdecode_runtime.input_bindings.size() == 6U,
              "boxdecode contract should build six input bindings");
      require(boxdecode_runtime.input_bindings[3].src_physical_byte_offset == 1075200,
              "boxdecode binding should come from upstream cast-adjusted physical byte offset");
      require(boxdecode_runtime.input_bindings[3].src_physical_size_bytes == 1024000U,
              "boxdecode binding should come from upstream cast-adjusted physical size");
      require(
          compiled.stages[1].boxdecode->payload.decode_type_option ==
              BoxDecodeTypeOption::GroupedByRoleLogit,
          "standalone cast->boxdecode should infer grouped-by-role-logit for BF16 YOLOv8 heads");

      std::vector<std::shared_ptr<Node>> standalone_nodes;
      standalone_nodes.push_back(simaai::neat::nodes::SimaBoxDecode(
          BoxDecodeType::YoloV8, 0.25, 0.45, 100, "fp32_boxdecode", 1280, 720, 640, 640));

      ContractCompileInput standalone_input;
      standalone_input.ingress.ingress_sample = ingress;
      pipeline_internal::sima::ManifestBuildDiagnostics standalone_diagnostics;
      const auto standalone_compiled =
          compile_node_contracts(standalone_nodes, standalone_input, &standalone_diagnostics);

      require(standalone_diagnostics.errors.empty(),
              "compile_node_contracts should succeed for standalone FP32 SimaBoxDecode");
      require(standalone_compiled.stages.size() == 1U, "expected one standalone boxdecode stage");
      require(standalone_compiled.stages[0].boxdecode.has_value(),
              "standalone stage should compile as boxdecode");

      const auto& standalone_runtime = standalone_compiled.stages[0].boxdecode->runtime_contract;
      require(standalone_compiled.stages[0].boxdecode->payload.input_dtype == "FP32",
              "standalone boxdecode should preserve FP32 input dtype");
      require(standalone_runtime.logical_inputs.size() == 6U,
              "standalone boxdecode contract should preserve six logical inputs");
      require(standalone_runtime.logical_inputs[0].size_bytes == 1638400U,
              "standalone boxdecode should preserve FP32 logical size");
      require(standalone_runtime.logical_inputs[3].byte_offset == 2150400,
              "standalone boxdecode should preserve FP32 grouped-by-role offsets");
      require(standalone_runtime.input_bindings.size() == 6U,
              "standalone boxdecode contract should build six input bindings");
      require(standalone_runtime.input_bindings[3].src_physical_size_bytes == 2048000U,
              "standalone boxdecode binding should preserve FP32 physical size");

      Tensor wrap_reg0 = make_fp32_head(0, {64, 80, 80}, 0, "cast_2/bbox_0");
      Tensor wrap_reg1 = make_fp32_head(1, {64, 40, 40}, 1638400, "cast_3/bbox_1");
      Tensor wrap_reg2 = make_fp32_head(2, {64, 20, 20}, 2048000, "cast_4/bbox_2");
      Tensor wrap_cls0 = make_fp32_head(3, {80, 80, 80}, 2150400, "cast_5/class_prob_0", 8.0f);
      Tensor wrap_cls1 = make_fp32_head(4, {80, 40, 40}, 4198400, "cast_6/class_prob_1", 8.0f);
      Tensor wrap_cls2 = make_fp32_head(5, {80, 20, 20}, 4710400, "cast_7/class_prob_2", 8.0f);
      const Sample wrapped_ingress = sample_from_tensors(
          TensorList{wrap_reg0, wrap_reg1, wrap_reg2, wrap_cls0, wrap_cls1, wrap_cls2});

      std::vector<std::shared_ptr<Node>> wrapped_nodes;
      wrapped_nodes.push_back(simaai::neat::nodes::SimaBoxDecode(
          BoxDecodeType::YoloV8, 0.25, 0.45, 100, "wrapped_boxdecode", 1280, 720, 640, 640));

      ContractCompileInput wrapped_input;
      wrapped_input.ingress.ingress_sample = wrapped_ingress;
      pipeline_internal::sima::ManifestBuildDiagnostics wrapped_diagnostics;
      const auto wrapped_compiled =
          compile_node_contracts(wrapped_nodes, wrapped_input, &wrapped_diagnostics);

      require(wrapped_diagnostics.errors.empty(),
              "compile_node_contracts should succeed for cast-wrapped FP32 SimaBoxDecode");
      require(wrapped_compiled.stages.size() == 1U,
              "expected one standalone wrapped boxdecode stage");
      require(wrapped_compiled.stages[0].boxdecode.has_value(),
              "wrapped stage should compile as boxdecode");
      require(wrapped_compiled.stages[0].boxdecode->payload.decode_type_option ==
                  BoxDecodeTypeOption::GroupedByRoleProbability,
              "cast-wrapped class_prob names should preserve probability-domain semantics");
    }));

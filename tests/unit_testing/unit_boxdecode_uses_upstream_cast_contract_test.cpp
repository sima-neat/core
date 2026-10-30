#include "nodes/sima/Cast.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/Tensor.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/contract/CompiledNodeContractQuery.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/TensorMath.h"
#include "test_main.h"
#include "test_utils.h"

#include <array>

namespace {

simaai::neat::Tensor
make_fp32_head(int logical_index, const std::vector<int64_t>& shape, int64_t physical_offset,
               const std::string& name, float fill_value = 0.0f,
               simaai::neat::TensorLayout layout = simaai::neat::TensorLayout::CHW) {
  using namespace simaai::neat;
  Tensor tensor;
  tensor.dtype = TensorDType::Float32;
  tensor.layout = layout;
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

std::string describe_errors(
    const simaai::neat::pipeline_internal::sima::ManifestBuildDiagnostics& diagnostics) {
  std::string text;
  for (const auto& error : diagnostics.errors)
    text += "\n  " + error;
  return text;
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

      // Cast is dtype-only: use genuinely HWC-authored storage for the positive
      // BF16 path. Keep the original planar FP32 sample for native FP32 checks
      // and the explicit unsupported-planar-BF16 admission check below.
      const TensorList heads{reg0, reg1, reg2, cls0, cls1, cls2};
      TensorList hwc_heads;
      for (std::size_t i = 0; i < heads.size(); ++i) {
        const auto& head = heads[i];
        hwc_heads.push_back(make_fp32_head(static_cast<int>(i),
                                           {head.shape[1], head.shape[2], head.shape[0]},
                                           head.route.physical_byte_offset, head.route.name,
                                           i < 3 ? 0.f : 8.f, TensorLayout::HWC));
      }

      std::vector<std::shared_ptr<Node>> nodes;
      nodes.push_back(simaai::neat::nodes::Cast(
          CastOptions{.direction = CastDirection::Fp32ToBf16, .element_name = "pre_box_cast"}));
      nodes.push_back(simaai::neat::nodes::SimaBoxDecode(
          BoxDecodeType::YoloV8, 0.25, 0.45, 100, "manual_boxdecode", 1280, 720, 640, 640,
          BoxDecodeTypeOption::Auto, BoxDecodeSourceStorage::DenseHwc));

      ContractCompileInput input;
      input.ingress.ingress_sample = sample_from_tensors(hwc_heads);
      pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
      const auto compiled = compile_node_contracts(nodes, input, &diagnostics);

      require(diagnostics.errors.empty(),
              "compile_node_contracts should succeed for HWC Cast -> SimaBoxDecode" +
                  describe_errors(diagnostics));
      require(compiled.stages.size() == 2U, "expected cast and boxdecode stages");
      require(compiled.stages[0].node_kind == nodes[0]->kind() &&
                  compiled.stages[1].node_kind == nodes[1]->kind(),
              "compiled stages must preserve Cast and BoxDecode Node identity");
      const auto* cast_runtime = compiled_runtime_contract_from_stage(&compiled.stages[0]);
      const auto* boxdecode_runtime = compiled_runtime_contract_from_stage(&compiled.stages[1]);
      require(cast_runtime && boxdecode_runtime, "both Nodes must expose tensor contracts");
      require(cast_runtime->logical_outputs.size() == 6U &&
                  boxdecode_runtime->logical_inputs.size() == 6U &&
                  boxdecode_runtime->input_bindings.size() == 6U,
              "Cast -> BoxDecode must preserve six heads and bindings");
      constexpr std::array<std::int64_t, 6> offsets{0, 819200, 1024000, 1075200, 2099200, 2355200};
      constexpr std::array<std::uint64_t, 6> sizes{819200, 204800, 51200, 1024000, 256000, 64000};
      constexpr std::uint64_t carrier_bytes = 2419200;
      const std::array<std::string, 6> names{"reg_head_0", "reg_head_1", "reg_head_2",
                                             "cls_head_0", "cls_head_1", "cls_head_2"};
      require(cast_runtime->physical_outputs.size() == 1U &&
                  cast_runtime->physical_outputs[0].physical_index == 0 &&
                  cast_runtime->physical_outputs[0].size_bytes == carrier_bytes,
              "Cast must publish the fixture's complete packed BF16 carrier");
      require(boxdecode_runtime->physical_inputs.size() == 1U &&
                  boxdecode_runtime->physical_inputs[0].size_bytes == carrier_bytes,
              "BoxDecode must retain the complete producer carrier extent");
      // Both producer and consumer retain actual HWC storage. Dtype conversion
      // scales its byte geometry without changing the logical head region.
      const auto hwc_shape = [&](std::size_t i) {
        return std::vector<std::int64_t>{heads[i].shape[1], heads[i].shape[2], heads[i].shape[0]};
      };
      for (std::size_t i = 0; i < heads.size(); ++i) {
        const auto& produced = cast_runtime->logical_outputs[i];
        const auto& consumed = boxdecode_runtime->logical_inputs[i];
        const auto& binding = boxdecode_runtime->input_bindings[i];
        require(produced.dtype == "BF16" && consumed.dtype == "BF16",
                "BoxDecode must consume Cast output dtype, not FP32 ingress dtype");
        require(produced.logical_index == static_cast<int>(i) &&
                    consumed.logical_index == static_cast<int>(i) &&
                    binding.local_logical_input_index == static_cast<int>(i) &&
                    binding.src_logical_output_index == static_cast<int>(i) &&
                    binding.src_output_slot == static_cast<int>(i),
                "Cast -> BoxDecode must preserve head order and source identity");
        require(produced.logical_name == names[i] && consumed.logical_name == names[i] &&
                    produced.layout == "HWC" && produced.shape == hwc_shape(i) &&
                    consumed.layout == "HWC" && consumed.shape == hwc_shape(i),
                "head names and logical shapes must match the fixture");
        const auto expected_strides = pipeline_internal::contiguous_strides_bytes(hwc_shape(i), 2U);
        require(produced.stride_bytes == expected_strides &&
                    consumed.stride_bytes == expected_strides,
                "Cast -> BoxDecode must retain actual HWC BF16 byte strides");
        require(produced.byte_offset == offsets[i] && produced.size_bytes == sizes[i] &&
                    consumed.byte_offset == offsets[i],
                "Cast must scale every head region from FP32 to BF16 exactly");
        require(produced.physical_index == 0 && consumed.physical_index == 0 &&
                    binding.src_physical_output_index == 0 && binding.sink_pad_index == 0,
                "every head must reference the single upstream carrier");
        // The binding admits the carrier; the logical view supplies the offset.
        // Consumed size is available capacity, not necessarily the head's tight size.
        require(binding.src_physical_byte_offset == 0 &&
                    binding.src_physical_size_bytes == carrier_bytes &&
                    binding.src_physical_byte_offset + consumed.byte_offset == offsets[i] &&
                    sizes[i] <= carrier_bytes - static_cast<std::uint64_t>(offsets[i]) &&
                    consumed.size_bytes >= sizes[i] &&
                    consumed.size_bytes <= carrier_bytes - static_cast<std::uint64_t>(offsets[i]),
                "each head must address its exact region once, within the producer extent");
      }
      // Domain inference is a semantic-contract assertion, not an executable
      // recipe assertion. Exercise the same extractor used by the Node.
      namespace si = pipeline_internal::sima;
      const si::BoxDecodeStandaloneContractOverrides overrides{
          si::BoxDecodeSourceStorageKind::DenseHwcPhysical, std::nullopt, std::nullopt};
      std::string semantic_error;
      const auto cast_semantics = si::build_boxdecode_static_contract_from_compiled_upstream(
          compiled.stages[0], BoxDecodeType::YoloV8, overrides, &semantic_error);
      require(cast_semantics && semantic_error.empty() &&
                  cast_semantics->decode_type_option == BoxDecodeTypeOption::GroupedByRoleLogit,
              "grouped BF16 regression/class heads must infer logit-domain semantics");

      // A CHW Cast really emits planar BF16, which the current numerical
      // provider does not support. The former metadata-only pass silently
      // discarded those strides and reinterpreted its bytes as HWC.
      ContractCompileInput planar_cast_input;
      planar_cast_input.ingress.ingress_sample = ingress;
      pipeline_internal::sima::ManifestBuildDiagnostics planar_cast_diagnostics;
      const auto planar_cast_compiled =
          compile_node_contracts(nodes, planar_cast_input, &planar_cast_diagnostics);
      const auto planar_error = describe_errors(planar_cast_diagnostics);
      require(!planar_cast_diagnostics.errors.empty() &&
                  planar_error.find("input.layout") != std::string::npos &&
                  planar_error.find("requires tight HWC rows with bounded channel padding") !=
                      std::string::npos,
              "planar BF16 must fail at numerical layout admission, not be reinterpreted" +
                  planar_error);
      require(planar_cast_compiled.stages.size() == 2U &&
                  planar_cast_compiled.stages[0].node_kind == nodes[0]->kind() &&
                  planar_cast_compiled.stages[0].renderable &&
                  !planar_cast_compiled.stages[1].renderable &&
                  !planar_cast_compiled.fully_renderable,
              "planar BF16 refusal must retain the valid Cast and mark BoxDecode unrenderable");
      const auto* planar_cast_runtime =
          compiled_runtime_contract_from_stage(&planar_cast_compiled.stages[0]);
      require(planar_cast_runtime && planar_cast_runtime->logical_outputs.size() == heads.size(),
              "planar Cast must retain its complete producer contract before refusal");
      for (std::size_t i = 0; i < heads.size(); ++i) {
        const auto& produced = planar_cast_runtime->logical_outputs[i];
        const auto planar_strides = pipeline_internal::contiguous_strides_bytes(heads[i].shape, 2U);
        require(produced.dtype == "BF16" && produced.layout == "CHW" &&
                    produced.shape == heads[i].shape && produced.stride_bytes == planar_strides &&
                    produced.byte_offset == offsets[i] && produced.size_bytes == sizes[i],
                "refused planar BF16 producer bytes must not be relabeled or reordered");
      }

      std::vector<std::shared_ptr<Node>> standalone_nodes;
      standalone_nodes.push_back(simaai::neat::nodes::SimaBoxDecode(
          BoxDecodeType::YoloV8, 0.25, 0.45, 100, "fp32_boxdecode", 1280, 720, 640, 640,
          BoxDecodeTypeOption::Auto, BoxDecodeSourceStorage::DenseHwc));

      ContractCompileInput standalone_input;
      standalone_input.ingress.ingress_sample = ingress;
      pipeline_internal::sima::ManifestBuildDiagnostics standalone_diagnostics;
      const auto standalone_compiled =
          compile_node_contracts(standalone_nodes, standalone_input, &standalone_diagnostics);

      require(standalone_diagnostics.errors.empty(),
              "compile_node_contracts should succeed for standalone FP32 SimaBoxDecode" +
                  describe_errors(standalone_diagnostics));
      require(standalone_compiled.stages.size() == 1U, "expected one standalone boxdecode stage");
      require(standalone_compiled.stages[0].node_kind == standalone_nodes[0]->kind(),
              "standalone stage must retain BoxDecode Node identity");
      const auto* standalone_runtime =
          compiled_runtime_contract_from_stage(&standalone_compiled.stages[0]);
      require(standalone_runtime && standalone_runtime->logical_inputs.size() == 6U &&
                  standalone_runtime->physical_inputs.size() == 6U &&
                  standalone_runtime->input_bindings.size() == 6U,
              "standalone BoxDecode must preserve six independently owned heads");
      // make_fp32_head allocates each tensor independently. Producer route
      // offsets describe the old packed layout, not these CPU-owned carriers.
      for (std::size_t i = 0; i < heads.size(); ++i) {
        const auto& logical = standalone_runtime->logical_inputs[i];
        const auto& physical = standalone_runtime->physical_inputs[i];
        const auto& binding = standalone_runtime->input_bindings[i];
        require(logical.dtype == "FP32" && logical.logical_name == names[i] &&
                    logical.logical_index == static_cast<int>(i) && logical.layout == "HWC" &&
                    logical.shape == hwc_shape(i),
                "standalone BoxDecode must preserve FP32 head identity and shape");
        require(logical.physical_index == static_cast<int>(i) && logical.byte_offset == 0 &&
                    logical.size_bytes == 2 * sizes[i] &&
                    physical.physical_index == static_cast<int>(i) &&
                    physical.size_bytes == 2 * sizes[i],
                "owned FP32 head must address its own complete carrier at offset zero");
        require(binding.local_logical_input_index == static_cast<int>(i) &&
                    binding.src_stage_index == -1 && binding.src_physical_output_index == -1 &&
                    binding.src_physical_byte_offset == 0,
                "standalone ingress must not retain an upstream producer binding");
      }

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
          BoxDecodeType::YoloV8, 0.25, 0.45, 100, "wrapped_boxdecode", 1280, 720, 640, 640,
          BoxDecodeTypeOption::Auto, BoxDecodeSourceStorage::DenseHwc));

      ContractCompileInput wrapped_input;
      wrapped_input.ingress.ingress_sample = wrapped_ingress;
      pipeline_internal::sima::ManifestBuildDiagnostics wrapped_diagnostics;
      const auto wrapped_compiled =
          compile_node_contracts(wrapped_nodes, wrapped_input, &wrapped_diagnostics);

      require(wrapped_diagnostics.errors.empty(),
              "compile_node_contracts should succeed for cast-wrapped FP32 SimaBoxDecode" +
                  describe_errors(wrapped_diagnostics));
      require(wrapped_compiled.stages.size() == 1U,
              "expected one standalone wrapped boxdecode stage");
      require(wrapped_compiled.stages[0].node_kind == wrapped_nodes[0]->kind(),
              "wrapped stage must retain BoxDecode Node identity");
      const auto* wrapped_runtime =
          compiled_runtime_contract_from_stage(&wrapped_compiled.stages[0]);
      require(wrapped_runtime && wrapped_runtime->logical_inputs.size() == 6U,
              "wrapped BoxDecode must retain all six input heads");
      const auto wrapped_semantics = si::build_boxdecode_static_contract_from_sample(
          wrapped_ingress, BoxDecodeType::YoloV8, std::nullopt, overrides, &semantic_error);
      require(wrapped_semantics && semantic_error.empty() &&
                  wrapped_semantics->decode_type_option ==
                      BoxDecodeTypeOption::GroupedByRoleProbability,
              "cast-wrapped class_prob names must preserve probability-domain semantics");
    }));

#include "nodes/sima/Cast.h"
#include "pipeline/Tensor.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/TensorMath.h"
#include "test_main.h"
#include "test_utils.h"

namespace {

simaai::neat::Tensor make_bf16_head(int logical_index, const std::vector<int64_t>& shape,
                                    int64_t physical_offset, const std::string& name) {
  using namespace simaai::neat;
  Tensor tensor;
  tensor.dtype = TensorDType::BFloat16;
  tensor.layout = TensorLayout::CHW;
  tensor.shape = shape;
  tensor.strides_bytes = simaai::neat::pipeline_internal::contiguous_strides_bytes(shape, 2U);

  std::size_t bytes = 2U;
  for (const auto dim : shape) {
    bytes *= static_cast<std::size_t>(dim);
  }
  tensor.storage = make_cpu_owned_storage(bytes);
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

RUN_TEST("unit_cast_fanout_contract_test", ([] {
           using namespace simaai::neat;

           Tensor head0 = make_bf16_head(0, {4, 2, 2}, 0, "head_0");
           Tensor head1 = make_bf16_head(1, {4, 1, 2}, 32, "head_1");
           Tensor head2 = make_bf16_head(2, {2, 1, 2}, 48, "head_2");
           Tensor head3 = make_bf16_head(3, {5, 2, 2}, 56, "head_3");
           Tensor head4 = make_bf16_head(4, {5, 1, 2}, 96, "head_4");
           Tensor head5 = make_bf16_head(5, {3, 1, 2}, 116, "head_5");
           const Sample ingress =
               sample_from_tensors(TensorList{head0, head1, head2, head3, head4, head5});

           auto cast = simaai::neat::nodes::Cast(
               CastOptions{.direction = CastDirection::Bf16ToFp32, .element_name = "post_cast"});

           ContractCompileInput input;
           input.ingress.ingress_sample = ingress;
           pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
           const auto compiled = compile_node_contracts({cast}, input, &diagnostics);

           require(diagnostics.errors.empty(),
                   "compile_node_contracts should succeed for multi-output Cast");
           require(compiled.stages.size() == 1U,
                   "multi-output Cast should compile to exactly one stage");
           require(compiled.stages.front().transport.has_value(),
                   "Cast should compile as a transport contract");

           const auto& runtime = compiled.stages.front().transport->runtime_contract;
           require(runtime.logical_outputs.size() == 6U,
                   "Cast should publish all six logical outputs from the upstream contract");
           require(runtime.logical_outputs[0].logical_index == 0 &&
                       runtime.logical_outputs[0].byte_offset == 0 &&
                       runtime.logical_outputs[0].size_bytes == 64U &&
                       runtime.logical_outputs[0].dtype == "FP32",
                   "Cast should convert the first BF16 head to FP32 bytes and dtype");
           require(runtime.logical_outputs[1].logical_index == 1 &&
                       runtime.logical_outputs[1].byte_offset == 64 &&
                       runtime.logical_outputs[1].size_bytes == 32U,
                   "Cast should preserve logical ordering and double the second head byte facts");
           require(runtime.logical_outputs[2].logical_index == 2 &&
                       runtime.logical_outputs[2].byte_offset == 96 &&
                       runtime.logical_outputs[2].size_bytes == 16U,
                   "Cast should preserve the third logical head");
           require(runtime.logical_outputs[3].logical_index == 3 &&
                       runtime.logical_outputs[3].byte_offset == 112 &&
                       runtime.logical_outputs[3].size_bytes == 80U,
                   "Cast should preserve the fourth logical head");
           require(runtime.logical_outputs[4].logical_index == 4 &&
                       runtime.logical_outputs[4].byte_offset == 192 &&
                       runtime.logical_outputs[4].size_bytes == 40U,
                   "Cast should preserve the fifth logical head");
           require(runtime.logical_outputs[5].logical_index == 5 &&
                       runtime.logical_outputs[5].byte_offset == 232 &&
                       runtime.logical_outputs[5].size_bytes == 24U,
                   "Cast should preserve the sixth logical head");
           require(runtime.output_order.size() == 6U,
                   "Cast should publish output ordering for every logical output");
         }));

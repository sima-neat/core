#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/contract/PluginCompiledContracts.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "test_main.h"

#include <array>
#include <limits>

namespace {
using namespace simaai::neat;
namespace pi = pipeline_internal;
namespace si = pi::sima;
namespace specs = si::specbuilders;

struct IngressFixture {
  pi::TensorBufferView view;
  std::array<Tensor, 2> tensors;
  IngressFixture() {
    view.segments = {{"unselected", 128}, {"shared", 256}};
    pi::TensorBufferTensorDescriptor first;
    first.memory_index = 1;
    first.logical_name = "features";
    first.dtype = SIMA_TENSOR_SET_DTYPE_FP32_V1;
    first.layout = SIMA_TENSOR_SET_LAYOUT_NCHW_V1;
    first.shape = {1, 2, 1, 3};
    first.stride_bytes = {48, 24, 12, 4};
    first.byte_offset = 32;
    first.size_bytes = 24;
    auto second = first;
    second.logical_name = "indices";
    second.dtype = SIMA_TENSOR_SET_DTYPE_INT32_V1;
    second.layout = SIMA_TENSOR_SET_LAYOUT_HW_V1;
    second.shape = {1, 3};
    second.stride_bytes = {12, 4};
    second.byte_offset = 96;
    second.size_bytes = 12;
    view.tensors = {first, second};
    tensors[0].dtype = TensorDType::Float32;
    tensors[0].shape = first.shape;
    tensors[0].axis_semantics = {TensorAxisSemantic::N, TensorAxisSemantic::C,
                                 TensorAxisSemantic::H, TensorAxisSemantic::W};
    auto& quant = tensors[0].semantic.quant.emplace();
    quant.axis = 1;
    quant.scales = {0.125f, 0.25f};
    quant.zero_points = {7, 9};
    tensors[1].dtype = TensorDType::Int32;
    tensors[1].shape = second.shape;
    tensors[1].axis_semantics = {TensorAxisSemantic::H, TensorAxisSemantic::W};
    auto& scalar = tensors[1].semantic.quant.emplace();
    scalar.scale = 0.5f;
    scalar.zero_point = 3;
    for (auto& tensor : tensors) {
      // Historical producer coordinates are not authority for current storage.
      tensor.route.memory_index = 9;
      tensor.route.physical_byte_offset = 4096;
      tensor.byte_offset = 2048;
      tensor.storage = std::make_shared<TensorBuffer>();
      tensor.storage->size_bytes = 4096;
      tensor.storage->map_fn = [](MapMode) -> Mapping {
        throw std::runtime_error("contract projection must not map payloads");
      };
    }
  }
};

void verify_ingress_projection() {
  IngressFixture fixture;
  CompiledRuntimeContract ingress;
  std::string error;
  require(specs::build_ingress_runtime_contract(fixture.view, fixture.tensors, &ingress, &error),
          "ingress projection: " + error);
  require(ingress.physical_outputs.size() == 1 &&
              ingress.physical_outputs.front().physical_index == 1 &&
              ingress.physical_outputs.front().size_bytes == 256,
          "shared views must keep one authoritative current carrier");
  const auto& first = ingress.logical_outputs[0];
  const auto& second = ingress.logical_outputs[1];
  require(first.byte_offset == 32 && first.size_bytes == 24 && second.byte_offset == 96 &&
              second.size_bytes == 12 && first.stride_bytes == fixture.view.tensors[0].stride_bytes,
          "logical bytes/strides must not become carrier remainder or producer provenance");
  require(first.layout == "CHW" && first.quant &&
              first.quant->granularity == si::QuantGranularity::PerAxis && first.quant->axis == 1 &&
              first.quant->scales == std::vector<double>{0.125, 0.25} &&
              first.quant->zero_points == std::vector<std::int64_t>{7, 9},
          "logical axes and every per-channel quantization value must survive projection");
  require(second.layout == "HW" && second.quant &&
              second.quant->granularity == si::QuantGranularity::PerTensor &&
              second.quant->scales == std::vector<double>{0.5} &&
              second.quant->zero_points == std::vector<std::int64_t>{3},
          "scalar quantization must survive projection");

  const auto reject = [&](auto mutate) {
    auto malformed = fixture;
    mutate(malformed);
    CompiledRuntimeContract unchanged;
    unchanged.plugin_kind = "unchanged";
    require(!specs::build_ingress_runtime_contract(malformed.view, malformed.tensors, &unchanged,
                                                   &error) &&
                !error.empty() && unchanged.plugin_kind == "unchanged" &&
                unchanged.logical_outputs.empty() && unchanged.physical_outputs.empty(),
            "bad ingress metadata must fail without partial contract mutation");
  };
  reject([](auto& f) { f.view.tensors.pop_back(); });
  reject([](auto& f) { f.view.tensors[1].dtype = SIMA_TENSOR_SET_DTYPE_UNKNOWN_V1; });
  reject([](auto& f) { f.view.tensors[1].memory_index = 2; });
  reject([](auto& f) { f.view.tensors[1].memory_index = -1; });
  reject([](auto& f) { f.view.tensors[1].byte_offset = -1; });
  reject([](auto& f) { f.view.tensors[1].byte_offset = 250; });
  reject([](auto& f) { f.view.tensors[1].size_bytes = 0; });
  reject([](auto& f) { f.view.tensors[1].size_bytes = std::numeric_limits<std::uint64_t>::max(); });
  reject([](auto& f) { f.view.tensors[0].layout = SIMA_TENSOR_SET_LAYOUT_NHWC_V1; });
  reject([](auto& f) { f.tensors[0].axis_semantics.pop_back(); });
}

void verify_checked_bindings() {
  IngressFixture fixture;
  CompiledRuntimeContract ingress, runtime;
  std::string error;
  require(specs::build_ingress_runtime_contract(fixture.view, fixture.tensors, &ingress, &error),
          "binding fixture projection");
  const auto& physical = ingress.physical_outputs.front();
  for (std::size_t i = 0; i < ingress.logical_outputs.size(); ++i) {
    const auto& source = ingress.logical_outputs[i];
    const std::uint64_t storage_bytes = i == 0 ? 36 : 12;
    require(specs::append_direct_kernel_input(&runtime, source, physical, i, storage_bytes, true,
                                              &error),
            "root binding: " + error);
    const auto& logical = runtime.logical_inputs.back();
    const auto& binding = runtime.input_bindings.back();
    require(logical.byte_offset == source.byte_offset && logical.size_bytes == source.size_bytes &&
                logical.stride_bytes == source.stride_bytes && logical.layout == source.layout &&
                logical.quant->scales == source.quant->scales &&
                logical.quant->zero_points == source.quant->zero_points &&
                logical.quant->axis == source.quant->axis,
            "binding must retain the complete logical view");
    require(binding.src_logical_output_index == -1 && binding.src_output_slot == -1 &&
                binding.src_physical_output_index == -1 && binding.src_physical_size_bytes == 0 &&
                binding.src_physical_byte_offset == 0,
            "root ingress must not acquire invented upstream coordinates");
  }
  require(runtime.physical_inputs.size() == 1, "shared inputs duplicated their carrier");

  const auto reject = [&](auto source, auto carrier, std::size_t index, std::uint64_t bytes) {
    const auto inputs = runtime.logical_inputs.size();
    const auto bindings = runtime.input_bindings.size();
    const auto carriers = runtime.physical_inputs.size();
    require(!specs::append_direct_kernel_input(&runtime, source, carrier, index, bytes, false,
                                               &error) &&
                !error.empty() && runtime.logical_inputs.size() == inputs &&
                runtime.input_bindings.size() == bindings &&
                runtime.physical_inputs.size() == carriers,
            "invalid checked input must not partially mutate the runtime contract");
  };
  auto source = ingress.logical_outputs.front();
  source.size_bytes = 23; // A truncated logical view, although its carrier is large.
  reject(source, physical, 2, 36);
  source = ingress.logical_outputs.front();
  reject(source, physical, 2, 225);               // Crosses carrier end.
  reject(source, physical, 0, source.size_bytes); // Duplicate logical input.
  auto different = physical;
  different.address_source = si::PhysicalAddressSource::RuntimePhysicalBinding;
  reject(source, different, 2, source.size_bytes);
  different = physical;
  different.physical_index = 2;
  reject(source, different, 2, source.size_bytes);
  source.byte_offset = -1;
  reject(source, physical, 2, source.size_bytes);
  source = ingress.logical_outputs.front();
  source.shape = {2, 3};
  source.stride_bytes = {16, 4};
  source.size_bytes = 24;
  CompiledRuntimeContract strided;
  require(specs::append_direct_kernel_input(&strided, source, physical, 0, 28, false, &error) &&
              strided.logical_inputs.front().size_bytes == 24 &&
              strided.logical_inputs.front().stride_bytes == source.stride_bytes,
          "checked strided storage must not replace logical publication bytes");
  source.size_bytes = 0;
  require(specs::append_direct_kernel_input(&strided, source, physical, 1, 28, false, &error) &&
              strided.logical_inputs.back().size_bytes == 24,
          "unknown logical extent retains the existing shape-based builder convention");
  source.shape = {std::numeric_limits<std::int64_t>::max(), 3};
  reject(source, physical, 2, 28);
  source = ingress.logical_outputs.front();
  source.logical_index = 7;
  source.output_slot = 5;
  require(specs::append_direct_kernel_input(&runtime, source, physical, 2, source.size_bytes, false,
                                            &error),
          "producer binding: " + error);
  const auto& bound = runtime.input_bindings.back();
  require(
      bound.src_logical_output_index == 7 && bound.src_output_slot == 5 &&
          bound.src_physical_output_index == 1 && bound.src_physical_size_bytes == 256 &&
          bound.src_physical_byte_offset == 0,
      "producer binding must retain identity and bind its carrier without repeating view offset");
}
} // namespace

RUN_TEST("unit_kernel_input_contract_builders_test", ([] {
           verify_ingress_projection();
           verify_checked_bindings();
         }));

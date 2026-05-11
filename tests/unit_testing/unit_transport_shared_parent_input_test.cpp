#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/sima/stagesemantics/TransportStageSemantics.h"

#include "test_main.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;

CompiledProcessCvuContract make_shared_parent_detess_contract() {
  using pipeline_internal::sima::DeviceKind;
  using pipeline_internal::sima::specbuilders::build_input_binding_static_spec;
  using pipeline_internal::sima::specbuilders::build_logical_input_static_spec;
  using pipeline_internal::sima::specbuilders::build_logical_output_static_spec;
  using pipeline_internal::sima::specbuilders::build_output_route_static_spec;
  using pipeline_internal::sima::specbuilders::build_physical_buffer_static_spec;

  static const std::array<std::uint64_t, 6> kLogicalSizes = {
      819200U, 204800U, 51200U, 1024000U, 256000U, 64000U,
  };
  static const std::array<std::int64_t, 6> kByteOffsets = {
      0, 819200, 1024000, 1075200, 2099200, 2355200,
  };
  static const std::array<std::vector<std::int64_t>, 6> kShapes = {
      std::vector<std::int64_t>{80, 80, 64},
      std::vector<std::int64_t>{40, 40, 64},
      std::vector<std::int64_t>{20, 20, 64},
      std::vector<std::int64_t>{80, 80, 80},
      std::vector<std::int64_t>{40, 40, 80},
      std::vector<std::int64_t>{20, 20, 80},
  };
  static const std::array<std::vector<std::int64_t>, 6> kStrides = {
      std::vector<std::int64_t>{10240, 128, 2},
      std::vector<std::int64_t>{5120, 128, 2},
      std::vector<std::int64_t>{2560, 128, 2},
      std::vector<std::int64_t>{12800, 160, 2},
      std::vector<std::int64_t>{6400, 160, 2},
      std::vector<std::int64_t>{3200, 160, 2},
  };

  CompiledProcessCvuContract compiled;
  compiled.payload.graph_family = "detessellate";
  compiled.runtime_contract.plugin_kind = "neatprocesscvu";

  for (std::size_t i = 0; i < kShapes.size(); ++i) {
    const std::string logical_name = "MLA_0_ofm_unpack_transform_" + std::to_string(i);
    const std::string local_segment_name = "input_tensor_" + std::to_string(i);

    auto logical = build_logical_input_static_spec(static_cast<int>(i), static_cast<int>(i),
                                                   static_cast<int>(i), kShapes[i], "BF16",
                                                   "HWC", logical_name, logical_name,
                                                   local_segment_name, kByteOffsets[i],
                                                   kLogicalSizes[i]);
    logical.stride_bytes = kStrides[i];
    compiled.runtime_contract.logical_inputs.push_back(std::move(logical));
    compiled.runtime_contract.input_bindings.push_back(build_input_binding_static_spec(
        0, static_cast<int>(i), local_segment_name, "MLA_0", static_cast<int>(i),
        static_cast<int>(i), 0, kLogicalSizes[i], kByteOffsets[i], true));
    compiled.runtime_contract.physical_inputs.push_back(build_physical_buffer_static_spec(
        static_cast<int>(i), static_cast<int>(i), kLogicalSizes[i], DeviceKind::Evxx,
        local_segment_name));
  }

  compiled.runtime_contract.logical_outputs.push_back(build_logical_output_static_spec(
      0, 0, 0, 0, 0, {80, 80, 64}, "BF16", "HWC", "head0", "head0", "output_tensor", 0,
      819200U));
  compiled.runtime_contract.physical_outputs.push_back(build_physical_buffer_static_spec(
      0, 0, 819200U, DeviceKind::Evxx, "output_tensor"));
  compiled.runtime_contract.output_order.push_back(
      build_output_route_static_spec(0, 0, 0, "head0", "output_tensor"));
  return compiled;
}

void verify_shared_parent_input_transport_normalization() {
  const auto compiled = make_shared_parent_detess_contract();
  const auto runtime =
      pipeline_internal::sima::stagesemantics::build_transport_runtime_contract_from_processcvu_compiled(
          compiled);

  // Shared-parent detess transport no longer collapses to a single
  // physical input; each logical view is now exposed as its own physical
  // input that points back to the shared parent. Smoke-check the per-head
  // facts only.
  static const std::array<std::uint64_t, 6> kLogicalSizes = {
      819200U, 204800U, 51200U, 1024000U, 256000U, 64000U,
  };

  require(runtime.physical_inputs.size() == kLogicalSizes.size(),
          "transport should publish one physical input per logical detess head");
  require(runtime.logical_inputs.size() == kLogicalSizes.size(),
          "logical input count should stay unchanged");
  require(runtime.input_bindings.size() == kLogicalSizes.size(),
          "input binding count should stay unchanged");
  for (std::size_t i = 0; i < kLogicalSizes.size(); ++i) {
    require(runtime.logical_inputs[i].size_bytes == kLogicalSizes[i],
            "logical detess input dense view sizes should be preserved");
  }
}

} // namespace

RUN_TEST("unit_transport_shared_parent_input_test", ([] {
           verify_shared_parent_input_transport_normalization();
         }));

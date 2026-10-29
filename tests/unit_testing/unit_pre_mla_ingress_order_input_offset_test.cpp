#include "support/quanttess_fanin_fixture.h"
#include "test_main.h"

#include <array>
#include <exception>
#include <string>

namespace {

void check_order(bool reordered, bool packed_ingress) {
  const auto fixture = sima_test::compile_processcvu_fixture(
      sima_test::quanttess_fanin_plan_data(
          {4, 2}, reordered ? std::vector<std::size_t>{1, 0} : std::vector<std::size_t>{0, 1},
          packed_ingress),
      "quanttess");
  const auto& compiled = fixture.contract;
  require(compiled.payload.num_in_tensor == 2 &&
              compiled.runtime_contract.physical_inputs.size() == 2U,
          "fan-in must preserve both input branches");
  for (std::size_t position = 0; position < 2U; ++position) {
    const auto lane = reordered ? 1U - position : position;
    const auto& input = compiled.runtime_contract.physical_inputs[position];
    require(input.source_physical_index == static_cast<int>(packed_ingress ? 0U : lane),
            "MLA boundary order must not change public input identity");
    require(input.source_byte_offset ==
                static_cast<std::int64_t>(packed_ingress && lane == 1U ? 64U : 0U),
            "read offset must come from the actual input carrier, not member order");
    require(input.size_bytes == (lane == 0U ? 64U : 16U),
            "each branch retains its own physical span");
    require(compiled.runtime_contract.logical_inputs[position].logical_name ==
                "input_" + std::to_string(lane),
            "logical identity must be reordered with its physical binding");
  }
}

void reject_invalid_carrier() {
  auto data = sima_test::quanttess_fanin_plan_data({4, 2}, {1, 0}, true);
  // The small view starts at byte 64 in an 80-byte public carrier. A declared
  // 17-byte span is not authority to read byte 80, regardless of tensor shape.
  data.values[2].storage_binding->physical_span = 17U;
  bool rejected = false;
  try {
    (void)sima_test::compile_processcvu_fixture(std::move(data), "quanttess");
  } catch (const std::invalid_argument& error) {
    require_contains(error.what(), "exceeds its public source span",
                     "invalid carrier must fail for its precise bounds violation");
    rejected = true;
  }
  require(rejected, "out-of-range ingress storage must fail before executable preparation");
}

} // namespace

// Issue #538: public ingress order and MLA-boundary order are independent.
// Current TensorList inputs retain distinct physical bindings; an explicitly
// packed source retains its authored offsets. Neither path invents a repack or
// infers a missing byte span merely because branch order happens to match.
RUN_TEST("unit_pre_mla_ingress_order_input_offset_test", ([] {
           check_order(false, false);
           check_order(true, false);
           check_order(false, true);
           check_order(true, true);
           reject_invalid_carrier();
         }));

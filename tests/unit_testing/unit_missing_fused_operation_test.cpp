#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/ProcessCvuRunTargetPolicy.h"
#include "pipeline/internal/sima/static_contract/PhysicalExecutionPlan.h"
#include <simaai/scheduler/kernel_preparation.h>

#include <array>
#include <iostream>
#include <stdexcept>

// This executable deliberately links an immutable restricted catalogue provider,
// not the production request resolver. No registry mutation or production test
// hook is involved. Both standalone implementations exist; their fusion does not.
namespace simaai::kernels {
std::span<const OperationInfo> operations() noexcept {
  static constexpr std::array targets{Target::A65, Target::EV74};
  static const std::array entries{OperationInfo{"detess", targets}, OperationInfo{"cast", targets}};
  return entries;
}
const OperationInfo* find_operation(std::string_view name) noexcept {
  for (const auto& operation : operations())
    if (operation.name == name)
      return &operation;
  return nullptr;
}
} // namespace simaai::kernels

namespace sc = simaai::neat::pipeline_internal::sima::static_contract;
namespace sima = simaai::neat::pipeline_internal::sima;

int main() {
  try {
    if (!simaai::scheduler::find_kernel_operation("detess") ||
        !simaai::scheduler::find_kernel_operation("cast") ||
        simaai::scheduler::find_kernel_operation("detesscast"))
      throw std::runtime_error("restricted catalogue fixture is not exact");

    sc::ModelExecutionPlanData data;
    data.contract_version = "2.1.0";
    for (sc::ValueId id = 0; id != 3; ++id) {
      sc::ValueSpec value;
      value.id = id;
      value.name = "value_" + std::to_string(id);
      value.required_bytes = id == 2 ? 64U : 32U;
      value.logical_dtype = id == 2 ? "FP32" : "BF16";
      value.logical_shape = sc::TensorShape{1, 1, 1, 16};
      value.logical_layout = "HWC";
      value.representation =
          id == 0 ? sc::ValueRepresentation::Tessellated : sc::ValueRepresentation::Dense;
      data.values.push_back(std::move(value));
    }
    data.model_inputs = {0};
    data.model_outputs.push_back({0, "result", 2});
    for (sc::OpId id = 0; id != 2; ++id) {
      sc::OpSpec op;
      op.id = id;
      op.sequence = id + 1U;
      op.name = id == 0 ? "detess" : "cast";
      op.kind = id == 0 ? sc::OpKind::Detessellate : sc::OpKind::Cast;
      op.processor = "EV74";
      op.inputs = {id};
      op.outputs = {id + 1U};
      op.input_shapes = {{1, 1, 1, 16}};
      op.output_shapes = {{1, 1, 1, 16}};
      if (id == 0)
        op.config = sc::DetessellateOpConfig{{1, 1, 1, 16}, {1, 1, 1, 16}, false, false, "BF16"};
      else
        op.config = sc::CastOpConfig{"FP32"};
      data.ops.push_back(std::move(op));
    }
    std::string error;
    auto semantic = sc::ModelExecutionPlan::create(std::move(data), &error);
    if (!semantic)
      throw std::runtime_error(error);
    auto physical = sc::PhysicalExecutionLowerer::lower(*semantic, &error);
    if (!physical)
      throw std::runtime_error(error);
    if (physical->commands.size() != 1 ||
        physical->commands.front().implementation_id != "detesscast" ||
        physical->commands.front().members.front().semantic_chain != std::vector<sc::OpId>{0, 1})
      throw std::runtime_error("missing implementation silently unfused the mandatory pair");

    sima::ProcessCvuStagePayload payload;
    payload.graph_name = physical->commands.front().implementation_id;
    payload.graph_family_enum = sima::ProcessCvuGraphFamily::DetessCast;
    simaai::neat::ContractCompileInput input;
    for (const auto* target : {"AUTO", "A65", "EV74"}) {
      payload.requested_run_target = target;
      bool rejected = false;
      try {
        (void)sima::resolve_processcvu_backend_decision(payload, input, "missing_fused",
                                                        sc::PhysicalCommandRole::Egress);
      } catch (const std::invalid_argument& failure) {
        const std::string diagnostic = failure.what();
        rejected =
            diagnostic.find("unavailable Kernels operation 'detesscast'") != std::string::npos;
      }
      if (!rejected)
        throw std::runtime_error("missing fused operation was not rejected");
    }
    std::cout << "unit_missing_fused_operation_test: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}

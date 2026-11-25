// SimaBEVPool must refuse, while its contract is compiled, a plan whose header
// records a different layout than SimaBEVPoolOptions.plan_format, and the
// message must name both layouts and the fix. The kernel rejects such a plan
// only at execution, where the failure is reported as an accelerator fault.
//
// Contract compilation needs the Scheduler kernel check but no accelerator.
#include "nodes/sima/SimaBEVPool.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/CompiledKernelTensor.h"
#include "pipeline/internal/contract/ContractFacts.h"

#include "support/bevfusion_kernel_reference.h"
#include "support/cold_contract_gst_init.h"
#include "test_main.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace simaai::neat;
namespace ref = ::bevfusion_test::reference;

struct CompileOutcome {
  bool ok = false;
  std::string error;
};

// Compiles a SimaBEVPool contract from an ingress sample holding the fixture's
// depth and context and the given plan bytes.
CompileOutcome compile_pool(const ref::BEVPoolCase& fixture, const std::vector<std::uint8_t>& plan,
                            SimaBEVPoolPlanFormat plan_format) {
  auto options = fixture.options;
  options.plan_format = plan_format;
  const auto depth = Tensor::from_vector(ref::to_channels_last(fixture.depth, 2, 2, 2, 3),
                                         {2, 2, 3, 2}, TensorMemory::CPU);
  const auto context = Tensor::from_vector(ref::to_channels_last(fixture.context, 2, 16, 2, 3),
                                           {2, 2, 3, 16}, TensorMemory::CPU);
  const auto plan_tensor =
      Tensor::from_vector(plan, {static_cast<std::int64_t>(plan.size())}, TensorMemory::CPU);
  ContractCompileInput input;
  input.node_index = 3;
  input.ingress.ingress_sample = sample_from_tensors({depth, context, plan_tensor});
  CompiledNodeContract compiled;
  CompileOutcome outcome;
  outcome.ok = SimaBEVPool(options).compile_node_contract(input, &compiled, &outcome.error);
  return outcome;
}

void set_header_version(std::vector<std::uint8_t>& plan, std::uint32_t version) {
  ref::detail::put_u32(plan, 4, version);
}

// The message must name the node, both layouts and the remedy.
void require_format_refusal(const CompileOutcome& outcome, const std::string& plan_layout,
                            const std::string& option_layout, const std::string& label) {
  require(!outcome.ok, label + ": a plan whose header disagrees with plan_format was accepted");
  require_contains(outcome.error, "SimaBEVPool", label + ": node");
  require_contains(outcome.error, plan_layout, label + ": plan layout");
  require_contains(outcome.error, "plan_format", label + ": option name");
  require_contains(outcome.error, option_layout, label + ": option layout");
  require_contains(outcome.error, "SimaBEVPoolPlanOptions.format", label + ": remedy");
}

// Runs every case so that one failure does not hide the others.
void run() {
  // Ingress samples carry GStreamer caps and tensor metadata.
  sima_test::initialize_cold_contract_gstreamer();
  const auto fixture = ref::bev_pool_case();
  const auto v1 = SimaBEVPoolPlanFormat::V1;
  const auto v2 = SimaBEVPoolPlanFormat::V2;
  std::string failures;
  const auto check = [&](const std::string& name, const std::function<void()>& body) {
    try {
      body();
      std::cout << "[CASE OK] " << name << "\n";
    } catch (const std::exception& error) {
      std::cout << "[CASE FAIL] " << name << ": " << error.what() << "\n";
      failures += "\n  " + name + ": " + error.what();
    }
  };

  // The oracle plan has a version-1 header and holds the version-1 worst case.
  require(fixture.plan.size() > 8 && fixture.plan[4] == 1, "fixture plan must be version 1");

  check("v1_plan_with_v2_option_is_refused", [&] {
    require_format_refusal(compile_pool(fixture, fixture.plan, v2), "V1", "V2", "V1 plan");
  });

  // A version-2 header on a plan as large as the version-1 worst case (the shape
  // of a large-geometry plan) with plan_format V1 is refused.
  check("large_v2_plan_with_v1_option_is_refused", [&] {
    auto plan = fixture.plan;
    set_header_version(plan, 2);
    require_format_refusal(compile_pool(fixture, plan, v1), "V2", "V1", "large V2 plan");
  });

  // A small version-2 plan with plan_format V1 reports the format mismatch, not
  // a worst-case size shortfall.
  check("small_v2_plan_reports_format_not_size", [&] {
    // Below the 228-byte worst case of this fixture, above the 72-byte minimum.
    auto plan = std::vector<std::uint8_t>(100, 0);
    ref::detail::put_u32(plan, 0, 0x4C505642U);
    set_header_version(plan, 2);
    const auto outcome = compile_pool(fixture, plan, v1);
    require_format_refusal(outcome, "V2", "V1", "small V2 plan");
    require(outcome.error.find("smaller than a worst-case plan") == std::string::npos,
            "the message must name the format mismatch, got: " + outcome.error);
  });

  // A plan without the header magic is not checked.
  check("plan_without_magic_is_not_checked", [&] {
    const auto zero = compile_pool(fixture, std::vector<std::uint8_t>(228, 0), v2);
    require(zero.ok, "all-zero plan with plan_format V2 must still compile: " + zero.error);
  });

  // A recognised magic with an unknown version is not checked here; the kernel
  // stays the authority on versions it supports.
  check("unknown_plan_version_is_not_checked", [&] {
    auto plan = fixture.plan;
    set_header_version(plan, 3);
    for (const auto format : {v1, v2}) {
      const auto unknown = compile_pool(fixture, plan, format);
      require(unknown.ok, "version-3 plan must still compile: " + unknown.error);
    }
  });

  if (!failures.empty())
    throw std::runtime_error("plan format checks failed:" + failures);
}
} // namespace

RUN_TEST("unit_bev_pool_plan_format_regression_test", run)

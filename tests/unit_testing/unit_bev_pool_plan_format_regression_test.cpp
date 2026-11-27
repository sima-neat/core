// SimaBEVPool's plan format while its contract is compiled (core#1528):
// - an unset plan_format takes the format the plan's header records, as
//   SimaBEVPoolPlan wrote it, and V2 without a readable plan;
// - an explicit plan_format that the header contradicts is refused with
//   misconfig.kernel_request, naming both formats and the fix;
// - SimaBEVPoolPlan's default V2 without context_channels is refused the same way.
// A plan joined from a connected SimaBEVPoolPlan is covered on device by
// sima_bev_pool_joined_plan_a65_test.
// The kernel rejects a mismatched plan only at execution, where the failure used
// to read as an accelerator fault.
//
// Contract compilation needs the Scheduler kernel check but no accelerator.
#include "nodes/sima/SimaBEVPool.h"
#include "nodes/sima/SimaBEVPoolPlan.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/CompiledKernelTensor.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include "support/bevfusion_kernel_reference.h"
#include "support/cold_contract_gst_init.h"
#include "test_main.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace simaai::neat;
namespace ref = ::bevfusion_test::reference;

struct CompileOutcome {
  bool ok = false;
  std::string error;
  std::string error_code;
};

Sample pool_sample(const ref::BEVPoolCase& fixture, const std::vector<std::uint8_t>& plan) {
  const auto depth = Tensor::from_vector(ref::to_channels_last(fixture.depth, 2, 2, 2, 3),
                                         {2, 2, 3, 2}, TensorMemory::CPU);
  const auto context = Tensor::from_vector(ref::to_channels_last(fixture.context, 2, 16, 2, 3),
                                           {2, 2, 3, 16}, TensorMemory::CPU);
  const auto plan_tensor =
      Tensor::from_vector(plan, {static_cast<std::int64_t>(plan.size())}, TensorMemory::CPU);
  return sample_from_tensors({depth, context, plan_tensor});
}

// Compiles a SimaBEVPool contract from an ingress sample holding the fixture's
// depth and context and the given plan bytes.
CompileOutcome compile_pool(const ref::BEVPoolCase& fixture, const std::vector<std::uint8_t>& plan,
                            std::optional<SimaBEVPoolPlanFormat> plan_format) {
  auto options = fixture.options;
  options.plan_format = plan_format;
  ContractCompileInput input;
  input.node_index = 3;
  input.ingress.ingress_sample = pool_sample(fixture, plan);
  CompiledNodeContract compiled;
  CompileOutcome outcome;
  outcome.ok = SimaBEVPool(options).compile_node_contract(input, &compiled, &outcome.error);
  outcome.error_code = compiled.failure_error_code;
  return outcome;
}

void set_header_version(std::vector<std::uint8_t>& plan, std::uint32_t version) {
  ref::detail::put_u32(plan, 4, version);
}

// A plan of `bytes` bytes whose header records `version`.
std::vector<std::uint8_t> headed_plan(std::size_t bytes, std::uint32_t version) {
  auto plan = std::vector<std::uint8_t>(bytes, 0);
  ref::detail::put_u32(plan, 0, 0x4C505642U);
  set_header_version(plan, version);
  return plan;
}

// The message must name the node, both layouts and the remedy, and the
// refusal must carry misconfig.kernel_request.
void require_format_refusal(const CompileOutcome& outcome, const std::string& plan_layout,
                            const std::string& option_layout, const std::string& label) {
  require(!outcome.ok, label + ": a plan whose header disagrees with plan_format was accepted");
  require_contains(outcome.error, "SimaBEVPool", label + ": node");
  require_contains(outcome.error, "plan_format is " + option_layout, label + ": option layout");
  require_contains(outcome.error, "plan header records format " + plan_layout,
                   label + ": plan layout");
  require_contains(outcome.error, "SimaBEVPoolPlanOptions.format", label + ": remedy");
  require(outcome.error_code == error_codes::kKernelRequest,
          label + ": expected misconfig.kernel_request, got '" + outcome.error_code + "'");
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
    const auto outcome = compile_pool(fixture, headed_plan(100, 2), v1);
    require_format_refusal(outcome, "V2", "V1", "small V2 plan");
    require(outcome.error.find("smaller than a worst-case plan") == std::string::npos,
            "the message must name the format mismatch, got: " + outcome.error);
  });

  // An unset plan_format takes the plan's own format: SimaBEVPoolPlan's V1
  // plan pools as V1 without restating it on the consumer.
  check("unset_plan_format_inherits_v1_from_the_plan", [&] {
    const auto outcome = compile_pool(fixture, fixture.plan, std::nullopt);
    require(outcome.ok, "a V1 plan with plan_format unset must compile: " + outcome.error);
    // Proof that V1 was requested: a 100-byte V1-headed plan is below V1's
    // 228-byte worst case, which only a V1 request checks.
    const auto small = compile_pool(fixture, headed_plan(100, 1), std::nullopt);
    require(!small.ok && small.error.find("smaller than a worst-case plan") != std::string::npos,
            "an inherited V1 format must apply V1's plan-size rule, got: " + small.error);
  });

  check("unset_plan_format_inherits_v2_from_the_plan", [&] {
    const auto outcome = compile_pool(fixture, headed_plan(100, 2), std::nullopt);
    require(outcome.ok, "a V2 plan with plan_format unset must compile: " + outcome.error);
  });

  // Without a readable header the consumer is built for V2.
  check("unset_plan_format_without_a_header_requests_v2", [&] {
    const auto outcome = compile_pool(fixture, std::vector<std::uint8_t>(100, 0), std::nullopt);
    require(outcome.ok, "a headerless plan with plan_format unset is prepared for V2 (whose "
                        "72-byte floor it meets): " +
                            outcome.error);
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

  // The contract compiler hands the provider's code to Graph::build, which
  // reports misconfig.kernel_request instead of misconfig.pipeline_shape.
  check("contract_compiler_carries_the_kernel_request_code", [&] {
    auto options = fixture.options;
    options.plan_format = v2;
    ContractCompileInput input;
    input.ingress.ingress_sample = pool_sample(fixture, fixture.plan);
    pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
    const std::vector<std::shared_ptr<Node>> nodes{nodes::SimaBEVPool(options)};
    (void)compile_node_contracts(nodes, input, &diagnostics);
    require(!diagnostics.errors.empty(), "the conflict must fail contract compilation");
    require(diagnostics.error_code == error_codes::kKernelRequest,
            "expected misconfig.kernel_request, got '" + diagnostics.error_code + "'");
  });

  // SimaBEVPoolPlan's default V2 needs the consumer's context channel count.
  check("v2_plan_without_context_channels_is_a_kernel_request_error", [&] {
    auto plan_case = ref::bev_pool_plan_case();
    plan_case.options.format = v2;
    plan_case.options.context_channels = 0;
    std::vector<float> c2l, k, aug;
    for (std::size_t i = 0; i < 2; ++i) {
      const auto& cal = plan_case.calibration;
      c2l.insert(c2l.end(), cal.camera2lidar[i].begin(), cal.camera2lidar[i].end());
      k.insert(k.end(), cal.intrinsics[i].begin(), cal.intrinsics[i].end());
      aug.insert(aug.end(), cal.img_aug[i].begin(), cal.img_aug[i].end());
    }
    const std::vector<float> lidar(plan_case.calibration.lidar_aug.begin(),
                                   plan_case.calibration.lidar_aug.end());
    ContractCompileInput input;
    input.node_index = 0;
    input.ingress.ingress_sample =
        sample_from_tensors({Tensor::from_vector(c2l, {2, 4, 4}, TensorMemory::CPU),
                             Tensor::from_vector(k, {2, 4, 4}, TensorMemory::CPU),
                             Tensor::from_vector(aug, {2, 4, 4}, TensorMemory::CPU),
                             Tensor::from_vector(lidar, {4, 4}, TensorMemory::CPU)});
    CompiledNodeContract compiled;
    std::string error;
    require(!SimaBEVPoolPlan(plan_case.options).compile_node_contract(input, &compiled, &error),
            "a V2 plan without context_channels must be refused");
    require_contains(error, "needs context_channels", "the refusal names the missing option");
    require(compiled.failure_error_code == error_codes::kKernelRequest,
            "expected misconfig.kernel_request, got '" + compiled.failure_error_code + "'");
    plan_case.options.context_channels = 16;
    require(SimaBEVPoolPlan(plan_case.options).compile_node_contract(input, &compiled, &error),
            "a V2 plan with context_channels must compile: " + error);
  });

  if (!failures.empty())
    throw std::runtime_error("plan format checks failed:" + failures);
}
} // namespace

RUN_TEST("unit_bev_pool_plan_format_regression_test", run)

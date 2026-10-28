// SimaVoxelize, SimaPillarScatter, SimaBEVPoolPlan and SimaBEVPool Nodes: deterministic
// fragments, option validation, cold Scheduler admission of the production
// requests on A65 and EV74, and the stage contract each Node compiles from an
// ingress Sample. The request builders themselves are covered without any
// runtime by unit_bevfusion_kernel_requests_test.
//
// Admission needs a Scheduler set built with Kernels that register voxelize,
// pillar_scatter, bev_pool_plan and bev_pool (sima-neat/kernels#10, #29, #31).
#include "nodes/sima/SimaBEVPool.h"
#include "nodes/sima/SimaArgMax.h"
#include "nodes/sima/SimaBEVPoolPlan.h"
#include "nodes/sima/SimaPillarScatter.h"
#include "nodes/sima/SimaVoxelize.h"
#include "nodes/sima/internal/LidarBevKernelRequests.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include <simaai/scheduler/client/kernel.h>

#include "support/bevfusion_kernel_reference.h"
#include "test_utils.h"

#include <array>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {
using namespace simaai::neat;
namespace kr = nodes_internal::kernel_requests;
namespace rt = ::simaai::scheduler::client;
namespace ref = ::bevfusion_test::reference;

kr::TensorFact fact(std::vector<std::int64_t> shape, rt::DType dtype,
                    std::vector<std::int64_t> strides = {}) {
  return {std::move(shape), dtype, std::move(strides), "", {}};
}

void require_throws(const std::function<void()>& fn, const std::string& needle,
                    const std::string& what) {
  try {
    fn();
  } catch (const std::invalid_argument& error) {
    require_contains(error.what(), needle, what);
    return;
  }
  throw std::runtime_error(what + ": expected std::invalid_argument");
}

// --------------------------------------------------------------------------
// Fragments, element names, contract definitions and option validation
// --------------------------------------------------------------------------
void fragments_and_definitions() {
  const auto vox = ref::voxelize_case().options;
  const auto ps = ref::pillar_scatter_case().options;
  const auto plan = ref::bev_pool_plan_case().options;
  const auto pool = ref::bev_pool_case().options;
  struct Expected {
    std::shared_ptr<Node> node;
    std::string kind, suffix;
    std::vector<std::string> inputs, outputs;
  };
  const std::vector<Expected> nodes{
      {nodes::SimaVoxelize(vox),
       "SimaVoxelize",
       "voxelize",
       {"points", "point_count"},
       {"decorated", "coordinates", "count"}},
      {nodes::SimaPillarScatter(ps),
       "SimaPillarScatter",
       "pillar_scatter",
       {"features", "coordinates", "count"},
       {"canvas"}},
      {nodes::SimaBEVPoolPlan(plan),
       "SimaBEVPoolPlan",
       "bev_pool_plan",
       {"camera2lidar", "camera_intrinsics", "img_aug_matrix", "lidar_aug_matrix"},
       {"plan"}},
      {nodes::SimaBEVPool(pool), "SimaBEVPool", "bev_pool", {"depth", "context", "plan"}, {"bev"}},
  };
  for (const auto& e : nodes) {
    require(e.node->kind() == e.kind, e.kind + " kind");
    const auto names = e.node->element_names(7);
    require(names == std::vector<std::string>{"n7_" + e.suffix}, e.kind + " element names");
    require(e.node->backend_fragment(7) == "neatkernelrunner name=n7_" + e.suffix,
            e.kind + " fragment");
    require(e.node->backend_fragment(7) == e.node->backend_fragment(7),
            e.kind + " fragment is not deterministic");
    const auto* provider = dynamic_cast<const NodeContractProvider*>(e.node.get());
    require(provider != nullptr, e.kind + " must own its contract");
    const auto definition = provider->contract_definition();
    require(definition.node_kind == e.kind && definition.plugin_kind == "kernel",
            e.kind + " contract identity");
    require(definition.inputs.size() == e.inputs.size() &&
                definition.outputs.size() == e.outputs.size(),
            e.kind + " contract arity");
    for (std::size_t i = 0; i < e.inputs.size(); ++i)
      require(definition.inputs[i].port_id == e.inputs[i] &&
                  definition.inputs[i].media_type == "application/vnd.simaai.tensor",
              e.kind + " input port " + e.inputs[i]);
    for (std::size_t i = 0; i < e.outputs.size(); ++i)
      require(definition.outputs[i].port_id == e.outputs[i], e.kind + " output " + e.outputs[i]);
    const auto* spec = dynamic_cast<const OutputSpecProvider*>(e.node.get());
    require(spec && spec->output_spec({}).media_type == "application/vnd.simaai.tensor",
            e.kind + " output spec");
  }

  require_throws(
      [&] {
        auto bad = vox;
        bad.max_pillars = 0;
        SimaVoxelize node(bad);
      },
      "SimaVoxelize: max_pillars", "SimaVoxelize rejects a zero pillar capacity");
  require_throws(
      [&] {
        auto bad = ps;
        bad.columns = -1;
        SimaPillarScatter node(bad);
      },
      "SimaPillarScatter: rows and columns", "SimaPillarScatter rejects an empty canvas");
  require_throws(
      [&] {
        auto bad = plan;
        bad.ds.clear();
        SimaBEVPoolPlan node(bad);
      },
      "SimaBEVPoolPlan: xs, ys and ds", "SimaBEVPoolPlan rejects an empty frustum");
  require_throws(
      [&] {
        auto bad = plan;
        bad.format = static_cast<SimaBEVPoolPlanFormat>(3);
        SimaBEVPoolPlan node(bad);
      },
      "SimaBEVPoolPlan: format", "SimaBEVPoolPlan rejects an unknown format");
  require_throws(
      [&] {
        auto bad = pool;
        bad.cells[2] = 0;
        SimaBEVPool node(bad);
      },
      "SimaBEVPool: every cell count", "SimaBEVPool rejects an empty grid");
  require_throws(
      [&] {
        auto bad = pool;
        bad.output_dtype = TensorDType::Float32;
        SimaBEVPool node(bad);
      },
      "SimaBEVPool: output_dtype", "SimaBEVPool rejects an FP32 bev");
}

// --------------------------------------------------------------------------
// Cold Scheduler admission (needs the pinned Kernels in the Scheduler set)
// --------------------------------------------------------------------------
void admit(const kr::KernelRequestPlan& plan, const std::string& label) {
  require(rt::find_kernel_operation(plan.operation) != nullptr,
          "Scheduler set has no '" + plan.operation +
              "' kernel; build Core against the Scheduler/Runner cohort that bundles "
              "sima-neat/kernels feature/bev-pool-ev74 (deps/manifest.json)");
  auto checked =
      rt::check_kernel({plan.operation, plan.target, plan.inputs, plan.outputs, plan.parameters});
  if (const auto* error = std::get_if<rt::Error>(&checked))
    throw std::runtime_error(label + " admission: " + error->field + ": " + error->message);
  auto inspected = rt::inspect_kernel(std::get<rt::KernelExecutable>(checked));
  require(std::holds_alternative<rt::KernelInspection>(inspected), label + " inspection");
  const auto& view = std::get<rt::KernelInspection>(inspected);
  require(view.target() == plan.target && view.outputs().size() == plan.outputs.size(),
          label + " checked target and arity");
}

void scheduler_admission() {
  for (const auto target : {SimaKernelTarget::A65, SimaKernelTarget::EV74}) {
    const std::string t = target == SimaKernelTarget::A65 ? " A65" : " EV74";
    // PointPillars production geometry.
    SimaVoxelizeOptions vox;
    vox.voxel_size = {0.25f, 0.25f, 8.0f};
    vox.point_cloud_range = {-50.0f, -50.0f, -5.0f, 50.0f, 50.0f, 3.0f};
    vox.quantization = {2.720007f, 8, SimaQuantRounding::NearestEven};
    vox.max_pillars = 40000;
    vox.max_points_per_pillar = 64;
    vox.target = target;
    const std::array vf{fact({600000, 4}, rt::DType::Float32), fact({1}, rt::DType::Int32)};
    admit(kr::voxelize_request(vox, vf), "voxelize" + t);
    const SimaPillarScatterOptions ps{400, 400, -128, target};
    const std::array pf{fact({1, 40000, 1, 64}, rt::DType::Int8),
                        fact({40000, 2}, rt::DType::Int32), fact({1}, rt::DType::Int32)};
    admit(kr::pillar_scatter_request(ps, pf), "pillar_scatter" + t);
    // BEVFusion camera-only production geometry.
    SimaBEVPoolOptions pool;
    pool.cells = {256, 256, 1};
    pool.depth_zero_point = -128;
    pool.context_zero_point = -5;
    pool.plan_format = SimaBEVPoolPlanFormat::V2;
    pool.quantization = {0.0024844530f, 0, SimaQuantRounding::NearestEven};
    pool.target = target;
    const std::array bf{fact({6, 32, 88, 118}, rt::DType::Int8),
                        fact({6, 32, 88, 80}, rt::DType::Int8),
                        fact({100000000}, rt::DType::UInt8)};
    admit(kr::bev_pool_request(pool, bf), "bev_pool" + t);

    // BF16 variants of the same production requests.
    vox.output_dtype = TensorDType::BFloat16;
    vox.quantization = {};
    admit(kr::voxelize_request(vox, vf), "voxelize BF16" + t);
    const SimaPillarScatterOptions ps16{400, 400, 0, target};
    const std::array pf16{fact({1, 40000, 1, 64}, rt::DType::BFloat16),
                          fact({40000, 2}, rt::DType::Int32), fact({1}, rt::DType::Int32)};
    admit(kr::pillar_scatter_request(ps16, pf16), "pillar_scatter BF16" + t);
    const std::array bf16{fact({6, 32, 88, 118}, rt::DType::BFloat16),
                          fact({6, 32, 88, 80}, rt::DType::BFloat16),
                          fact({100000000}, rt::DType::UInt8)};
    pool.depth_zero_point = pool.context_zero_point = 0;
    pool.quantization = {8.13f, 0, SimaQuantRounding::NearestEven};
    admit(kr::bev_pool_request(pool, bf16), "bev_pool BF16->INT8" + t);
    pool.output_dtype = TensorDType::BFloat16;
    pool.quantization = {};
    admit(kr::bev_pool_request(pool, bf16), "bev_pool BF16->BF16" + t);
    // INT8 inputs cannot produce a BF16 bev, on either target.
    const auto mixed = kr::bev_pool_request(pool, bf);
    require(std::holds_alternative<rt::Error>(rt::check_kernel(
                {mixed.operation, mixed.target, mixed.inputs, mixed.outputs, mixed.parameters})),
            "bev_pool INT8->BF16 must be rejected" + t);
  }
  SimaBEVPoolPlanOptions plan;
  for (int w = 0; w < 88; ++w)
    plan.xs.push_back(static_cast<float>(w) * 8.0f);
  for (int h = 0; h < 32; ++h)
    plan.ys.push_back(static_cast<float>(h) * 8.0f);
  for (int d = 0; d < 118; ++d)
    plan.ds.push_back(1.0f + 0.5f * static_cast<float>(d));
  plan.bx = {-51.0f, -51.0f, 0.0f};
  plan.dx = {0.4f, 0.4f, 20.0f};
  plan.nx = {256, 256, 1};
  plan.format = SimaBEVPoolPlanFormat::V2;
  plan.context_channels = 80;
  plan.cache_plan = true;
  const std::array cf{fact({6, 4, 4}, rt::DType::Float32), fact({6, 4, 4}, rt::DType::Float32),
                      fact({6, 4, 4}, rt::DType::Float32), fact({4, 4}, rt::DType::Float32)};
  admit(kr::bev_pool_plan_request(plan, cf), "bev_pool_plan A65 format 2");
  plan.target = SimaKernelTarget::EV74;
  const auto ev = kr::bev_pool_plan_request(plan, cf);
  auto checked = rt::check_kernel({ev.operation, ev.target, ev.inputs, ev.outputs, ev.parameters});
  require(std::holds_alternative<rt::Error>(checked),
          "bev_pool_plan has no EV74 implementation and must not fall back");
  // A capacity below the kernels' worst case is the kernels' rejection, not Core's.
  plan.target = SimaKernelTarget::A65;
  plan.plan_capacity_bytes =
      static_cast<std::int64_t>(kr::bev_pool_plan_capacity_bytes(plan, 6)) - 1;
  const auto small = kr::bev_pool_plan_request(plan, cf);
  require(std::holds_alternative<rt::Error>(rt::check_kernel(
              {small.operation, small.target, small.inputs, small.outputs, small.parameters})),
          "bev_pool_plan capacity formula must match the kernels' format 2 bound");
  plan.input_dtype = TensorDType::BFloat16;
  plan.plan_capacity_bytes = 0;
  admit(kr::bev_pool_plan_request(plan, cf), "bev_pool_plan BF16 A65 format 2");
  plan.plan_capacity_bytes =
      static_cast<std::int64_t>(kr::bev_pool_plan_capacity_bytes(plan, 6)) - 1;
  const auto small16 = kr::bev_pool_plan_request(plan, cf);
  require(std::holds_alternative<rt::Error>(
              rt::check_kernel({small16.operation, small16.target, small16.inputs, small16.outputs,
                                small16.parameters})),
          "bev_pool_plan capacity formula must match the kernels' BF16 format 2 bound");
}

// --------------------------------------------------------------------------
// Stage contracts compiled from a graph-ingress Sample
// --------------------------------------------------------------------------
// BF16 bits as a BF16 tensor (from_vector has no BF16 overload).
Tensor bf16_tensor(const std::vector<std::uint16_t>& bits, std::vector<std::int64_t> shape) {
  auto tensor = Tensor::from_vector(bits, std::move(shape), TensorMemory::CPU);
  tensor.dtype = TensorDType::BFloat16;
  return tensor;
}

CompiledNodeContract compile(const Node& node, const TensorList& tensors) {
  ContractCompileInput input;
  input.node_index = 3;
  input.ingress.ingress_sample = sample_from_tensors(tensors);
  CompiledNodeContract compiled;
  std::string error;
  const auto* provider = dynamic_cast<const NodeContractProvider*>(&node);
  require(provider && provider->compile_node_contract(input, &compiled, &error),
          node.kind() + " contract: " + error);
  require(compiled.kernel && compiled.kernel->recipe && compiled.renderable &&
              compiled.plugin_kind == "kernel" &&
              compiled.element_name == node.element_names(3).front(),
          node.kind() + " compiled kernel stage identity");
  require(compiled.kernel->recipe->members.size() == 1, node.kind() + " runs one member");
  const auto& member = compiled.kernel->recipe->members.front();
  const auto& runtime = compiled.kernel->runtime_contract;
  require(member.inputs.size() == tensors.size() &&
              runtime.logical_inputs.size() == tensors.size() &&
              runtime.input_bindings.size() == tensors.size(),
          node.kind() + " binds every ingress tensor");
  require(member.outputs.size() == runtime.logical_outputs.size() &&
              runtime.physical_outputs.size() == runtime.logical_outputs.size(),
          node.kind() + " publishes every checked output");
  return compiled;
}

void ingress_contracts() {
  using ref::to_channels_last;
  const auto v = ref::voxelize_case();
  const auto n = static_cast<std::int64_t>(v.points.size() / 4);
  auto compiled = compile(
      SimaVoxelize(v.options),
      {Tensor::from_vector(v.points, {n, 4}, TensorMemory::CPU),
       Tensor::from_vector(std::vector<std::int32_t>{v.point_count}, {1}, TensorMemory::CPU)});
  auto& outputs = compiled.kernel->runtime_contract.logical_outputs;
  require(outputs[0].shape == std::vector<std::int64_t>{1, 3, 8, 10} &&
              outputs[0].dtype == "INT8" && outputs[1].dtype == "INT32" &&
              outputs[2].shape == std::vector<std::int64_t>{1},
          "SimaVoxelize stage outputs");

  // Two views into one carrier. The logical input places each view and the
  // member port is relative to it: the Scheduler adds both, so a port that
  // repeated the view's offset would read past the carrier.
  {
    namespace specs = pipeline_internal::sima::specbuilders;
    const std::uint64_t points_at = 64, points_bytes = v.points.size() * sizeof(float);
    const std::uint64_t count_at = points_at + points_bytes + 64, carrier = count_at + 4 + 64;
    CompiledNodeContract upstream;
    auto& produced = upstream.kernel.emplace().runtime_contract;
    produced.physical_outputs.push_back(specs::build_physical_buffer_static_spec(
        0, 0, carrier, pipeline_internal::sima::DeviceKind::Cpu, "shared"));
    produced.logical_outputs.push_back(specs::build_logical_output_static_spec(
        0, 0, 0, 0, 0, {n, 4}, "FP32", "", "points", "points", "shared",
        static_cast<std::int64_t>(points_at), points_bytes));
    produced.logical_outputs.push_back(specs::build_logical_output_static_spec(
        1, 1, 0, 1, 1, {1}, "INT32", "", "point_count", "point_count", "shared",
        static_cast<std::int64_t>(count_at), sizeof(std::int32_t)));
    ContractCompileInput input;
    input.node_index = 3;
    input.immediate_upstream = &upstream;
    CompiledNodeContract shared;
    std::string error;
    require(SimaVoxelize(v.options).compile_node_contract(input, &shared, &error),
            "SimaVoxelize shared-carrier contract: " + error);
    const auto& member = shared.kernel->recipe->members.front();
    const auto& logical = shared.kernel->runtime_contract.logical_inputs;
    const std::array<std::uint64_t, 2> at{points_at, count_at};
    require(member.inputs.size() == at.size() && logical.size() == at.size(),
            "SimaVoxelize shared-carrier arity");
    for (std::size_t i = 0; i < at.size(); ++i)
      require(member.inputs[i].byte_offset == 0 &&
                  logical[i].byte_offset == static_cast<std::int64_t>(at[i]),
              "SimaVoxelize shared-carrier input " + std::to_string(i) +
                  " is placed once, by its logical view");

    // A view shorter than its shape implies must be refused even though the
    // carrier holds the bytes: they belong to the neighbouring tensor.
    produced.logical_outputs[0].size_bytes = points_bytes - 16;
    require_throws(
        [&] {
          CompiledNodeContract truncated;
          if (!SimaVoxelize(v.options).compile_node_contract(input, &truncated, &error))
            throw std::invalid_argument(error);
        },
        "the checked request reads", "SimaVoxelize view shorter than its tensor");
  }

  const auto p = ref::pillar_scatter_case();
  const auto m = static_cast<std::int64_t>(p.pillars), ch = static_cast<std::int64_t>(p.channels);
  compiled =
      compile(SimaPillarScatter(p.options),
              {Tensor::from_vector(p.features, {m, ch}, TensorMemory::CPU),
               Tensor::from_vector(p.coordinates, {m, 2}, TensorMemory::CPU),
               Tensor::from_vector(std::vector<std::int32_t>{p.count}, {1}, TensorMemory::CPU)});
  const auto& canvas = compiled.kernel->runtime_contract.logical_outputs.front();
  require(canvas.shape == std::vector<std::int64_t>{1, 4, 6, 16} &&
              canvas.publication_relation ==
                  pipeline_internal::sima::PublicationViewRelation::PackedStorage &&
              compiled.kernel->runtime_contract.physical_outputs.front().size_bytes >= 4 * 6 * 16,
          "SimaPillarScatter publishes the tiled canvas through an explicit physical view");

  const auto c = ref::bev_pool_plan_case();
  std::vector<float> c2l, k, aug;
  for (std::size_t i = 0; i < 2; ++i) {
    c2l.insert(c2l.end(), c.calibration.camera2lidar[i].begin(),
               c.calibration.camera2lidar[i].end());
    k.insert(k.end(), c.calibration.intrinsics[i].begin(), c.calibration.intrinsics[i].end());
    aug.insert(aug.end(), c.calibration.img_aug[i].begin(), c.calibration.img_aug[i].end());
  }
  const std::vector<float> lidar(c.calibration.lidar_aug.begin(), c.calibration.lidar_aug.end());
  compiled =
      compile(SimaBEVPoolPlan(c.options), {Tensor::from_vector(c2l, {2, 4, 4}, TensorMemory::CPU),
                                           Tensor::from_vector(k, {2, 4, 4}, TensorMemory::CPU),
                                           Tensor::from_vector(aug, {2, 4, 4}, TensorMemory::CPU),
                                           Tensor::from_vector(lidar, {4, 4}, TensorMemory::CPU)});
  require(compiled.kernel->runtime_contract.logical_outputs.front().shape ==
                  std::vector<std::int64_t>{228} &&
              compiled.kernel->recipe->target == rt::Target::A65,
          "SimaBEVPoolPlan stage output");

  for (const auto target : {SimaKernelTarget::A65, SimaKernelTarget::EV74}) {
    auto b = ref::bev_pool_case(target);
    compiled = compile(SimaBEVPool(b.options),
                       {Tensor::from_vector(to_channels_last(b.depth, 2, 2, 2, 3), {2, 2, 3, 2},
                                            TensorMemory::CPU),
                        Tensor::from_vector(to_channels_last(b.context, 2, 16, 2, 3), {2, 2, 3, 16},
                                            TensorMemory::CPU),
                        Tensor::from_vector(b.plan, {static_cast<std::int64_t>(b.plan.size())},
                                            TensorMemory::CPU)});
    const auto& bev = compiled.kernel->runtime_contract.logical_outputs.front();
    require(bev.shape == std::vector<std::int64_t>{1, 16, 4, 4} &&
                bev.stride_bytes == std::vector<std::int64_t>{256, 1, 64, 16} &&
                compiled.kernel->recipe->target == kr::kernel_target(target),
            "SimaBEVPool stage publishes channels-last bev on the selected target");
  }

  // BF16 stage contracts.
  const auto v16 = ref::voxelize_case_bf16();
  compiled = compile(
      SimaVoxelize(v16.options),
      {Tensor::from_vector(v16.points, {n, 4}, TensorMemory::CPU),
       Tensor::from_vector(std::vector<std::int32_t>{v16.point_count}, {1}, TensorMemory::CPU)});
  require(compiled.kernel->runtime_contract.logical_outputs[0].dtype == "BF16" &&
              compiled.kernel->runtime_contract.physical_outputs[0].size_bytes >= 3 * 8 * 10 * 2,
          "SimaVoxelize BF16 stage output");
  const auto p16 = ref::pillar_scatter_case_bf16();
  compiled =
      compile(SimaPillarScatter(p16.options),
              {bf16_tensor(p16.features_bf16, {1, m, 1, ch}),
               Tensor::from_vector(p16.coordinates, {m, 2}, TensorMemory::CPU),
               Tensor::from_vector(std::vector<std::int32_t>{p16.count}, {1}, TensorMemory::CPU)});
  const auto& canvas16 = compiled.kernel->runtime_contract.logical_outputs.front();
  require(canvas16.dtype == "BF16" && canvas16.shape == std::vector<std::int64_t>{1, 4, 6, 16} &&
              canvas16.publication_relation ==
                  pipeline_internal::sima::PublicationViewRelation::PackedStorage,
          "SimaPillarScatter BF16 canvas keeps its tiled physical view");
  {
    ContractCompileInput downstream;
    downstream.immediate_upstream = &compiled;
    SimaArgMaxOptions options;
    options.axis = 3;
    CompiledNodeContract argmax;
    std::string diagnostic;
    require(SimaArgMax(options).compile_node_contract(downstream, &argmax, &diagnostic),
            "ArgMax must consume checked BF16 tile layout: " + diagnostic);
    auto inspected = rt::inspect_kernel(argmax.kernel->recipe->members.front().executable);
    require(std::holds_alternative<rt::KernelInspection>(inspected) &&
                std::holds_alternative<rt::MlaTiled>(
                    std::get<rt::KernelInspection>(inspected).inputs().front().layout),
            "ArgMax lost the producer's physical BF16 layout");
  }
  // Existing dense producers can legally remove singleton axes at publication.
  // Forwarding their raw execution shape would regress model->ArgMax paths.
  {
    rt::TensorSpec cast_input{{1, 1, 1, 32}, rt::DType::Int8};
    rt::TensorSpec cast_output{{1, 1, 1, 32}, rt::DType::Float32};
    auto checked =
        rt::check_kernel({"cast", rt::Hardware::A65, {&cast_input, 1}, {&cast_output, 1}, {}});
    require(std::holds_alternative<rt::KernelExecutable>(checked),
            "singleton projection fixture cast must admit");
    auto producer = compiled;
    auto recipe = std::make_shared<pipeline_internal::sima::KernelStageRecipe>();
    rt::TensorStageMember member;
    member.executable = std::get<rt::KernelExecutable>(std::move(checked));
    member.outputs.push_back({0, 0, rt::TensorStagePortSource::Output});
    recipe->members.push_back(std::move(member));
    producer.kernel->recipe = recipe;
    auto& logical = producer.kernel->runtime_contract.logical_outputs.front();
    logical.dtype = "FP32";
    logical.shape = {1, 32};
    logical.stride_bytes.clear();
    logical.size_bytes = 128;
    logical.publication_relation = pipeline_internal::sima::PublicationViewRelation::SingletonAxes;
    producer.kernel->runtime_contract.physical_outputs.front().size_bytes = 128;
    ContractCompileInput downstream;
    downstream.immediate_upstream = &producer;
    SimaArgMaxOptions options;
    options.axis = 1;
    CompiledNodeContract argmax;
    std::string diagnostic;
    require(SimaArgMax(options).compile_node_contract(downstream, &argmax, &diagnostic),
            "ArgMax must retain dense published singleton projection: " + diagnostic);
    const auto inspected = rt::inspect_kernel(argmax.kernel->recipe->members.front().executable);
    require(std::holds_alternative<rt::KernelInspection>(inspected) &&
                std::get<rt::KernelInspection>(inspected).inputs().front().shape ==
                    std::vector<std::int64_t>{1, 32},
            "ArgMax replaced the published shape with raw execution dimensions");
  }
  for (const auto output : {TensorDType::BFloat16, TensorDType::Int8}) {
    auto b = ref::bev_pool_case_bf16(SimaKernelTarget::EV74, output);
    compiled = compile(SimaBEVPool(b.options),
                       {bf16_tensor(to_channels_last(b.depth_bf16, 2, 2, 2, 3), {2, 2, 3, 2}),
                        bf16_tensor(to_channels_last(b.context_bf16, 2, 16, 2, 3), {2, 2, 3, 16}),
                        Tensor::from_vector(b.plan, {static_cast<std::int64_t>(b.plan.size())},
                                            TensorMemory::CPU)});
    const auto& bev = compiled.kernel->runtime_contract.logical_outputs.front();
    const bool wide = output == TensorDType::BFloat16;
    require(bev.dtype == (wide ? "BF16" : "INT8") &&
                bev.stride_bytes == (wide ? std::vector<std::int64_t>{512, 2, 128, 32}
                                          : std::vector<std::int64_t>{256, 1, 64, 16}),
            "SimaBEVPool BF16 stage publishes its bev dtype and strides");
  }

  // Structural failures surface the Node's own diagnostic, before admission.
  ContractCompileInput input;
  input.ingress.ingress_sample = sample_from_tensors(
      {Tensor::from_vector(std::vector<float>(8, 0.0f), {2, 4}, TensorMemory::CPU)});
  CompiledNodeContract out;
  std::string error;
  require(!SimaVoxelize(v.options).compile_node_contract(input, &out, &error),
          "SimaVoxelize accepted a Sample without point_count");
  require_contains(error, "SimaVoxelize: expects 2 input tensors", "SimaVoxelize arity diagnostic");
  Tensor retained_tile = bf16_tensor(std::vector<std::uint16_t>(16, 0x3F80), {1, 1, 1, 16});
  retained_tile.semantic.tess = TessSpec{{1, 1, 1, 16}, "MLA_TILED_BF16_COMPACT", {0, 1, 2, 3}, 32};
  ContractCompileInput retained;
  retained.ingress.ingress_sample = sample_from_tensors({retained_tile});
  SimaArgMaxOptions argmax_options;
  argmax_options.axis = 3;
  require(!SimaArgMax(argmax_options).compile_node_contract(retained, &out, &error),
          "ArgMax treated unsupported explicit tiled graph ingress as dense");
  require_contains(error, "detessellate", "tiled ingress requires explicit layout admission");
  ContractCompileInput empty;
  require(!SimaBEVPool(ref::bev_pool_case().options).compile_node_contract(empty, &out, &error),
          "SimaBEVPool compiled without an upstream contract");
  require_contains(error, "SimaBEVPool requires a typed upstream tensor contract",
                   "SimaBEVPool missing upstream diagnostic");
}
} // namespace

int main() {
  try {
    fragments_and_definitions();
    scheduler_admission();
    ingress_contracts();
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] unit_bevfusion_kernel_nodes_test: " << error.what() << "\n";
    return 1;
  }
  std::cout << "[OK] unit_bevfusion_kernel_nodes_test\n";
  return 0;
}

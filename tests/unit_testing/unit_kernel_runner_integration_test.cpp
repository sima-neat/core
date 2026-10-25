#include "pipeline/internal/sima/ContractRender.h"
#include "gst/GstInit.h"
#include "pipeline/internal/sima/SchedulerStageIntegration.h"
#include "pipeline/internal/sima/BoxDecodeGeometry.h"
#include "pipeline/internal/sima/BoxDecodeKernelAdmission.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StageMemoryPolicy.h"
#include "pipeline/internal/TerminalOutputContractQuery.h"
#include "pipeline/internal/sima/StageRecipePreparation.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "nodes/sima/Preproc.h"
#include <simaai/scheduler/testing/probe.h>
#include <gstsimaaitensorbuffer_meta.h>
#include <simaai/scheduler/testing/route_probe.h>

#include "tensor_output_contract_fixtures.h"
#include "support/cold_contract_gst_init.h"
#include "pipeline/internal/HolderLoanGate.h"
#include "pipeline/internal/OutputQueueMemory.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/RealtimeFrameCredit.h"
#include "graph/internal/GraphRunState.h"
#include <simaai/scheduler/client/memory_access.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <thread>
#include <atomic>
#include <stdexcept>
#include <utility>

namespace {
namespace integration = simaai::neat::pipeline_internal::sima;
namespace rt = simaai::scheduler::client;
namespace probe = simaai::scheduler::testing;
namespace gs = simaai::scheduler::client;
gs::KernelProgram& kernel(gs::StageRequest& request) {
  return std::get<gs::KernelProgram>(request.program);
}
const gs::KernelProgram& kernel(const gs::StageRequest& request) {
  return std::get<gs::KernelProgram>(request.program);
}
void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
void require_fixture(bool condition, const char* operation, const std::string& error) {
  if (!condition)
    throw std::runtime_error(std::string(operation) + ": " + error);
}
simaai::gst::TensorBufferPublicationOwner compact_geometry_publication() {
  using namespace simaai::gst;
  TensorBufferPublishContract contract;
  contract.stage_key = "compact-geometry-fixture";
  contract.physical_outputs.push_back({0, 4, "arena"});
  TensorBufferPublishLogicalOutput tensor;
  tensor.logical_index = tensor.physical_index = tensor.memory_index = 0;
  tensor.backend_output_index = tensor.route_slot = 0;
  tensor.logical_name = tensor.backend_name = "output";
  tensor.segment_name = "arena";
  tensor.size_bytes = 4;
  tensor.dtype = SIMA_TENSOR_SET_DTYPE_UINT8_V1;
  tensor.layout = SIMA_TENSOR_SET_LAYOUT_HW_V1;
  tensor.shape = {2, 2};
  tensor.stride_bytes = {2, 1};
  contract.logical_outputs.push_back(std::move(tensor));
  TensorBufferPreparedMetaTemplate prepared;
  std::string error;
  const bool ready = tensor_buffer_prepare_meta_template_from_contract(contract, &prepared, &error);
  require_fixture(ready, "prepare compact geometry publication", error);
  auto owner = tensor_buffer_freeze_publication(prepared, &error);
  require_fixture(bool(owner), "freeze compact geometry publication", error);
  return owner;
}
void terminal_host_placement_proof() {
  namespace domain = integration::static_contract;
  namespace specs = integration::specbuilders;
  const std::array inputs{rt::TensorSpec{{2}, rt::DType::Float32}};
  const std::array outputs{rt::TensorSpec{{2}, rt::DType::BFloat16}};
  auto checked = rt::check_kernel({"cast", rt::Hardware::A65, inputs, outputs, {}});
  require(std::holds_alternative<rt::KernelExecutable>(checked), "host proof Cast admission");
  auto recipe = std::make_shared<integration::KernelStageRecipe>();
  recipe->members.push_back({std::get<rt::KernelExecutable>(checked), {{0, 0}}, {{0, 0}}, {}});
  integration::StageStaticSpec stage;
  stage.element_name = stage.logical_stage_id = "unrelated-cast";
  stage.plugin_kind = "kernel";
  stage.native_kernel = recipe;
  stage.core_fresh_default_arena = true;
  stage.frame_arena_escape_policy = domain::ArenaEscapePolicy::CpuMappablePublic;
  stage.logical_inputs.push_back(
      specs::build_logical_input_static_spec(0, 0, 0, {2}, "FP32", "C", "input", "input", "input"));
  stage.input_bindings.push_back(specs::build_input_binding_static_spec(0, 0, "input", "input"));
  stage.physical_inputs.push_back(specs::build_physical_buffer_static_spec(
      0, 0, 8, integration::DeviceKind::Cpu, "input", -1, 0, 64));
  stage.physical_outputs.push_back(specs::build_physical_buffer_static_spec(
      0, 0, 4, integration::DeviceKind::Cpu, "output", -1, 0, 64));
  stage.logical_outputs.push_back(specs::build_logical_output_static_spec(
      0, 0, 0, 0, 0, {2}, "BF16", "C", "output", "output", "output", 0, 4));
  stage.output_order.push_back(specs::build_output_route_static_spec(0, 0, 0, "output", "output"));
  simaai::neat::CompiledNodeContract compiled;
  compiled.node_kind = "Cast";
  compiled.element_name = compiled.logical_stage_id = stage.element_name;
  compiled.renderable = true;
  compiled.kernel.emplace();
  compiled.kernel->recipe = recipe;
  auto& contract = compiled.kernel->runtime_contract;
  contract.physical_inputs = stage.physical_inputs;
  contract.logical_inputs = stage.logical_inputs;
  contract.input_bindings = stage.input_bindings;
  contract.physical_outputs = stage.physical_outputs;
  contract.logical_outputs = stage.logical_outputs;
  contract.output_order = stage.output_order;
  const auto render = [&](const simaai::neat::CompiledNodeContract& node) {
    simaai::neat::CompiledPipelineContracts contracts;
    contracts.fully_renderable = true;
    contracts.stages.push_back(node);
    integration::ManifestBuildDiagnostics diagnostics;
    auto value = simaai::neat::render_manifest_from_compiled_contracts(contracts, {}, &diagnostics);
    require(value && diagnostics.errors.empty() && value->stages.size() == 1,
            "terminal host origin renderer failed");
    return value->stages.front();
  };
  using Endpoint =
      simaai::neat::pipeline_internal::terminal_output_contract::PublicOutputEndpointSelector;
  const auto place = [&](integration::StageStaticSpec value, const Endpoint* endpoint = nullptr) {
    integration::SimaPluginStaticManifest manifest;
    manifest.stages.push_back(std::move(value));
    integration::finalize_pipeline_storage(manifest, endpoint);
    return manifest;
  };
  const auto policy = [](const integration::SimaPluginStaticManifest& manifest) {
    const auto& layout = manifest.stages.front().frame_layout;
    require(layout && layout->allocations.size() == 1, "host proof final allocation absent");
    return layout->allocations.front().policy;
  };
  const auto fresh = render(compiled);
  require(fresh.core_fresh_default_arena, "fresh Cast origin was lost");
  Endpoint direct;
  direct.terminal_stage_key = fresh.element_name;
  direct.terminal_stage_key_required = true;
  simaai::neat::CompiledNodeContract fragment;
  fragment.node_kind = "ModelFragment";
  fragment.element_name = fragment.logical_stage_id = "infer";
  fragment.child_stages.push_back(compiled);
  const auto nested = render(fragment);
  Endpoint endpoint;
  endpoint.terminal_stage_key = "infer";
  endpoint.terminal_node_kind = "ModelFragment";
  endpoint.allow_unresolved_terminal_stage_fallback = true;
  require(policy(place(nested, &endpoint)) == gs::AllocationPolicy::RequireHostCpu,
          "a unique checked ModelFragment child did not select HostCpu");
  auto explicit_child = endpoint;
  explicit_child.terminal_stage_key = nested.element_name;
  require(policy(place(nested, &explicit_child)) == gs::AllocationPolicy::RequireHostCpu,
          "an exact ModelFragment child identity did not select HostCpu");
  const auto reject_endpoint = [&](auto change) {
    auto invalid = endpoint;
    change(invalid);
    require(policy(place(nested, &invalid)) != gs::AllocationPolicy::RequireHostCpu,
            "unauthorized container endpoint selected HostCpu");
  };
  reject_endpoint([](auto& value) { value.terminal_node_kind = "SimaArgMax"; });
  reject_endpoint([](auto& value) { value.terminal_stage_key_required = true; });
  reject_endpoint([](auto& value) { value.allow_unresolved_terminal_stage_fallback = false; });
  reject_endpoint([](auto& value) { value.output_segment_name = "missing-output"; });
  auto private_output = nested;
  private_output.physical_outputs.front().segment_name = "private-only";
  auto private_selector = endpoint;
  private_selector.output_segment_name = "private-only";
  require(policy(place(private_output, &private_selector)) != gs::AllocationPolicy::RequireHostCpu,
          "a private carrier matched the public terminal selector");
  auto disconnected = nested;
  disconnected.element_name = disconnected.logical_stage_id = "disconnected-child";
  integration::SimaPluginStaticManifest ambiguous;
  ambiguous.stages = {nested, disconnected};
  integration::finalize_pipeline_storage(ambiguous, &endpoint);
  require(std::none_of(ambiguous.stages.begin(), ambiguous.stages.end(),
                       [](const auto& value) {
                         return value.frame_layout &&
                                value.frame_layout->allocations.front().policy ==
                                    gs::AllocationPolicy::RequireHostCpu;
                       }),
          "ambiguous disconnected container leaves selected HostCpu");
  auto host = place(fresh, &direct);
  require(policy(host) == gs::AllocationPolicy::RequireHostCpu,
          "fresh explicit A65 terminal did not select HostCpu");
  require(policy(place(fresh)) == gs::AllocationPolicy::RequireDms,
          "ordinary A65 output implicitly selected HostCpu");
  require(!integration::finalize_stage_publications(&host), "host final publication failed");
  std::string error;
  const auto roundtrip =
      integration::parse_manifest_json(integration::serialize_manifest_json(host), &error);
  require_fixture(roundtrip && roundtrip->stages.front().frame_arena_allocations.size() == 1 &&
                      roundtrip->stages.front().frame_arena_allocations.front().placement.domain ==
                          domain::ArenaStorageDomain::HostCpu &&
                      roundtrip->stages.front().frame_arena_escape_policy ==
                          domain::ArenaEscapePolicy::CpuOnlyPublic,
                  "final Host allocation JSON roundtrip", error);
  const auto authored = [&](auto set) {
    auto node = compiled;
    set(node.kernel->runtime_contract);
    const auto rendered = render(node);
    require(!rendered.core_fresh_default_arena &&
                policy(place(rendered, &direct)) != gs::AllocationPolicy::RequireHostCpu,
            "authored placement acquired fresh HostCpu authority");
  };
  authored([](auto& r) { r.frame_arena_storage_domain = domain::ArenaStorageDomain::Cma; });
  authored([](auto& r) { r.frame_arena_storage_domain = domain::ArenaStorageDomain::Dms; });
  authored([](auto& r) { r.frame_arena_required_device_access = 2U; });
  authored(
      [](auto& r) { r.frame_arena_escape_policy = domain::ArenaEscapePolicy::CpuMappablePublic; });
  authored([](auto& r) {
    r.frame_arena_allocation_policy = domain::ArenaAllocationPolicy::PreferDmsThenCma;
  });
  const auto ineligible = [&](auto change) {
    auto value = fresh;
    change(value);
    require(policy(place(value, &direct)) != gs::AllocationPolicy::RequireHostCpu,
            "ineligible native carrier acquired HostCpu authority");
  };
  ineligible([](auto& v) { v.core_fresh_default_arena = false; });
  ineligible([](auto& v) {
    v.frame_arena_allocation_policy = domain::ArenaAllocationPolicy::PreferDmsThenCma;
  });
  ineligible([](auto& v) { v.physical_outputs[0].memory_flags = 1; });
  ineligible([](auto& v) { v.physical_outputs[0].source_physical_index = 0; });
  ineligible([](auto& v) { v.physical_outputs[0].required_alignment_bytes = 128; });
  ineligible([](auto& v) { v.physical_outputs[0].device_kind = integration::DeviceKind::Mla; });
  ineligible([](auto& v) { v.logical_outputs[0].stride_bytes = {4}; });
  ineligible([](auto& v) { v.logical_outputs.clear(); });
  integration::SimaPluginStaticManifest consumed;
  consumed.stages = {fresh, fresh};
  auto& consumer = consumed.stages.back();
  consumer.element_name = consumer.logical_stage_id = "later-consumer";
  // A consumer only needs the original result's BF16 contract for this cold edge proof.
  consumer.native_kernel.reset();
  consumer.frame_arena_required_device_access = gs::kAccessCpuA65;
  consumer.logical_inputs[0].dtype = "BF16";
  consumer.physical_inputs[0].size_bytes = 4;
  consumer.input_bindings[0].src_stage_id = fresh.logical_stage_id;
  consumer.input_bindings[0].src_logical_output_index = 0;
  consumer.input_bindings[0].src_physical_output_index = 0;
  consumer.input_bindings[0].src_output_slot = 0;
  integration::finalize_pipeline_storage(consumed, &direct);
  require(policy(consumed) == gs::AllocationPolicy::RequireDms,
          "intermediate allocation acquired HostCpu authority");
  std::shared_ptr<GstElement> pipeline(gst_pipeline_new("host-proof"), gst_object_unref);
  auto* element = gst_element_factory_make("neatkernelrunner", fresh.element_name.c_str());
  auto* sink = gst_element_factory_make("appsink", "terminal");
  require(element && sink, "host proof topology factories");
  gst_bin_add_many(GST_BIN(pipeline.get()), element, sink, nullptr);
  auto* pad = gst_element_request_pad_simple(element, "sink_0");
  require(pad && gst_element_link(element, sink), "host proof direct terminal edge");
  gst_object_unref(pad);
  integration::SchedulerStageRequest request{fresh.element_name, 1, 2, {}, &host.stages.front(),
                                             fresh.element_name};
  require(bool(integration::prepare_and_attach_stages(pipeline.get(), {&request, 1})) &&
              !probe::inspect_stage(element).valid,
          "wrong terminal peer acquired host authority");
  request.terminal_host_sink = "terminal";
  const auto admitted = integration::prepare_and_attach_stages(pipeline.get(), {&request, 1});
  const auto attached_layout = gs::stage_frame_layout(element);
  const auto& expected_span = host.stages.front().frame_layout->outputs.front().span;
  require_fixture(
      !admitted && probe::inspect_stage(element).valid && attached_layout &&
          attached_layout != host.stages.front().frame_layout &&
          attached_layout->allocations.size() == 1 &&
          attached_layout->allocations.front().policy == gs::AllocationPolicy::RequireHostCpu &&
          attached_layout->outputs.size() == 1 &&
          attached_layout->outputs.front().span.byte_offset == expected_span.byte_offset &&
          attached_layout->outputs.front().span.size_bytes == expected_span.size_bytes,
      "Core finalized Host endpoint attachment", admitted ? admitted->message : "");
}

void output_carrier_budget() {
  const auto check = [](std::size_t executing, std::size_t held, std::size_t expected) {
    const auto result = integration::kernel_runner_output_pool_capacity(executing, held);
    const auto* count = std::get_if<std::size_t>(&result);
    require(count && *count == expected, "incorrect output carrier budget");
  };
  check(1, 0, 2); // Copy-only: active frame plus appsink's current carrier.
  check(1, 4, 6); // Public queue2 permits four held samples, not four executing jobs.
  check(8, 10, 19);
  check(64, 4032, 4097);
  constexpr std::array<std::pair<std::size_t, std::size_t>, 4> invalid_budgets{
      {{0, 0}, {65, 0}, {64, 4033}, {1, std::numeric_limits<std::size_t>::max()}}};
  for (const auto [executing, held] : invalid_budgets) {
    require(std::holds_alternative<rt::Error>(
                integration::kernel_runner_output_pool_capacity(executing, held)),
            "invalid or overflowing output budget accepted");
  }
}
void authored_stage_budget() {
  const auto check = [](std::optional<std::size_t> authored, std::size_t graph_execution,
                        std::size_t graph_pool, std::size_t execution, std::size_t pool) {
    const auto result =
        integration::kernel_runner_stage_budget(authored, graph_execution, graph_pool);
    const auto* budget = std::get_if<integration::KernelRunnerBudget>(&result);
    require(budget && budget->execution_capacity == execution &&
                budget->output_pool_capacity == pool,
            "authored stage must preserve graph retention and output-pool floors");
  };
  check(std::nullopt, 1, 20, 1, 20); // Unauthored BoxDecode does not acquire Preproc's window.
  check(4, 1, 20, 4, 23);            // PR1130 retention plus the model's async window.
  check(4, 1, 2, 4, 5);
  check(4, 1, 0, 4, 5);   // Automatic output budget preserves the spare.
  check(1, 4, 23, 1, 23); // A shallower window cannot shrink an independent floor.
  check(4, 1, 4094, 4, 4097);
  check(1, 64, 4097, 1, 4097);
  check(std::nullopt, 64, 0, 64, 65);
  for (const auto authored : {0U, 65U}) {
    require(
        std::holds_alternative<rt::Error>(integration::kernel_runner_stage_budget(authored, 1, 20)),
        "invalid authored execution capacity accepted");
  }
  for (const auto [execution, pool] : std::array<std::pair<std::size_t, std::size_t>, 6>{
           {{0, 2},
            {65, 66},
            {4, 4},
            {1, 4095},
            {1, 4098},
            {1, std::numeric_limits<std::size_t>::max()}}}) {
    require(std::holds_alternative<rt::Error>(
                integration::kernel_runner_stage_budget(4, execution, pool)),
            "invalid or overflowing graph/stage carrier budget accepted");
  }
}

void authored_preproc_contract_policy() {
  using namespace simaai::neat;
  const auto options = [] {
    PreprocOptions opt;
    opt.model_managed_contract = true;
    opt.num_buffers = opt.num_buffers_model = 4;
    opt.num_buffers_locked = true;
    opt.set_input_shape({720, 1280, 3});
    opt.input_img_type = "RGB";
    opt.set_output_shape({640, 640, 3});
    opt.scaled_width = opt.scaled_height = 640;
    opt.output_img_type = "RGB";
    opt.output_dtype = "EVXX_INT8";
    opt.tessellate = false;
    opt.single_output_handoff = true;
    opt.q_scale = 0.25;
    opt.q_zp = 7;
    return opt;
  };
  const auto compile = [](PreprocOptions opt) {
    CompiledNodeContract result;
    std::string error;
    require_fixture(Preproc(std::move(opt)).compile_node_contract({}, &result, &error),
                    "compile authored Preproc", error);
    return result;
  };
  const auto original = compile(options());
  require(original.native_execution_capacity == 4U, "model async capacity was discarded");
  auto cached = std::make_shared<const CompiledProcessCvuContract>(*original.processcvu);
  // Numerical contracts may be cached across async/sync/async builds. The
  // execution window must always come from the current node, never the cache.
  for (const auto count : {1, 4, 1, 4}) {
    auto opt = options();
    opt.num_buffers = opt.num_buffers_model = count;
    opt.compiled_contract = cached;
    require(compile(opt).native_execution_capacity == static_cast<std::size_t>(count),
            "cached numerical contract overrode the current async/sync author");
  }
  auto unlocked = options();
  unlocked.num_buffers_locked = false;
  unlocked.num_buffers = 0;
  require(!compile(unlocked).native_execution_capacity,
          "unlocked Preproc pool hints must not become execution authority");
  auto reused = original;
  std::string reuse_error;
  require_fixture(Preproc(unlocked).compile_node_contract({}, &reused, &reuse_error),
                  "recompile unlocked Preproc", reuse_error);
  require(!reused.native_execution_capacity,
          "reused contract output retained a previous node's authored capacity");
  auto standalone = options();
  standalone.model_managed_contract = false;
  standalone.num_buffers_locked = false;
  standalone.num_buffers = 0;
  require(!compile(standalone).native_execution_capacity,
          "standalone defaults must not acquire a model execution window");

  CompiledNodeContract child = original;
  child.element_name = "fused_branch_preproc_7";
  child.logical_stage_id = "model_preproc_7";
  CompiledNodeContract inner, outer;
  inner.child_stages.push_back(child);
  outer.child_stages.push_back(inner);
  CompiledPipelineContracts contracts;
  contracts.stages.push_back(outer);
  const auto render = [&] {
    integration::ManifestBuildDiagnostics diagnostics;
    auto manifest = render_manifest_from_compiled_contracts(contracts, {}, &diagnostics);
    require(manifest && diagnostics.errors.empty() && manifest->stages.size() == 1,
            "nested model child must render without changing the authored policy");
    return *manifest;
  };
  const auto manifest = render();
  require(manifest.stages.front().native_execution_capacity == 4U &&
              manifest.stages.front().element_name == child.element_name &&
              manifest.stages.front().logical_stage_id == child.logical_stage_id,
          "nested/renamed child lost its execution author");
  require(integration::serialize_manifest_json(manifest).find("native_execution_capacity") ==
              std::string::npos,
          "private capacity must not extend the serialized plugin manifest ABI");

  BuildResult build;
  build.kernel_runner_capacity = 1;
  build.kernel_runner_output_pool_capacity = 20;
  build.rendered_manifest = manifest;
  build.compiled_contracts = std::make_shared<CompiledPipelineContracts>(contracts);
  build.pipeline_string = "identity name=producer ! appsink name=mysink";
  build.appsink_name = "mysink";
  const BuildResult async_build = build;
  // Exercise the existing sync entry, including its early legacy return, and
  // the same serial helper used by ordinary/fused source construction.
  (void)session_build_clamp_sync_build_result(build, -1);
  require(build.kernel_runner_capacity == 1 &&
              build.rendered_manifest->stages.front().native_execution_capacity == 1U &&
              build.kernel_runner_output_pool_capacity == 20,
          "sync build must clamp authored execution without shrinking retention");
  require(async_build.rendered_manifest->stages.front().native_execution_capacity == 4U &&
              build.compiled_contracts == async_build.compiled_contracts &&
              async_build.compiled_contracts->stages.front()
                      .child_stages.front()
                      .child_stages.front()
                      .native_execution_capacity == 4U,
          "sync copy must not mutate the async manifest or shared compiled child");
  auto rebuilt = render();
  require(rebuilt.stages.front().native_execution_capacity == 4U,
          "sync clamp mutated cached child/numerical contract state");
  build.rendered_manifest = rebuilt;
  build.kernel_runner_capacity = 8;
  session_build_clamp_sync_kernel_runner_capacity(build);
  require(build.kernel_runner_capacity == 1 &&
              build.rendered_manifest->stages.front().native_execution_capacity == 1U,
          "source-mode sync clamp must cover graph and authored execution");
  integration::StageStaticSpec unauthored;
  build.rendered_manifest->stages.push_back(unauthored);
  session_build_clamp_sync_kernel_runner_capacity(build);
  require(!build.rendered_manifest->stages.back().native_execution_capacity,
          "sync clamping must not invent an authored policy for other stages");
}

void ev_service_capacity() {
  const auto request = [](rt::Hardware target, std::size_t capacity) {
    gs::StageAttachment result;
    kernel(result.request).target = target;
    result.request.options.capacity = capacity;
    return result;
  };
  const auto check = [](std::initializer_list<gs::StageAttachment> stages, std::size_t expected) {
    require(integration::kernel_runner_ev_device_lanes({stages.begin(), stages.size()}) == expected,
            "incorrect graph EV service capacity");
  };
  check({}, 1); // Unused when Runtime finds no EV stages.
  check({request(rt::Hardware::A65, 64), request(rt::Hardware::A65, 64)}, 1);
  check({request(rt::Hardware::EV74, 1)}, 1);
  check({request(rt::Hardware::EV74, 1), request(rt::Hardware::A65, 64)}, 1);
  check({request(rt::Hardware::A65, 64), request(rt::Hardware::EV74, 1)}, 1);
  check({request(rt::Hardware::EV74, 2)}, 2);
  check({request(rt::Hardware::EV74, 64)}, 2);
  check({request(rt::Hardware::EV74, 1), request(rt::Hardware::A65, 1),
         request(rt::Hardware::EV74, 1)},
        2);
}

gs::StageRequest cast_request(std::size_t input_pads = 1, bool terminal_host = false) {
  namespace specs = integration::specbuilders;
  const std::array inputs{rt::TensorSpec{{2}, rt::DType::Float32}};
  const std::array outputs{rt::TensorSpec{{2}, rt::DType::BFloat16}};
  auto checked = rt::check_kernel({"cast", rt::Hardware::A65, inputs, outputs, {}});
  require(std::holds_alternative<rt::KernelExecutable>(checked), "real Cast admission failed");
  auto executable = std::get<rt::KernelExecutable>(std::move(checked));
  auto inspected = rt::inspect_kernel(executable);
  require(std::holds_alternative<rt::KernelInspection>(inspected), "real Cast inspection failed");
  const auto& inspection = std::get<rt::KernelInspection>(inspected);
  const auto in = inspection.input_requirements().front();
  const auto out = inspection.output_requirements().front();
  gs::StageRequest request;
  auto layout = std::make_shared<gs::FrameLayout>();
  layout->allocations.push_back(
      {4096, gs::kAccessCpuA65,
       terminal_host ? gs::AllocationPolicy::RequireHostCpu : gs::AllocationPolicy::RequireCma});
  auto publication = std::make_shared<simaai::gst::TensorBufferPublishContract>();
  publication->stage_key = "cast-fixture";
  for (std::size_t pad = 0; pad < input_pads; ++pad) {
    const auto index = static_cast<int>(pad);
    const auto input = "input" + std::to_string(pad), output = "output" + std::to_string(pad);
    layout->inputs.push_back({gs::DirectMemory{pad, 0, 0, in.storage_bytes}, 0});
    layout->outputs.push_back({pad, {0, pad * 64, out.storage_bytes}});
    kernel(request).members.push_back({executable, {{pad, 0}}, {{pad, 0}}, {}});
    if (pad == 0)
      publication->physical_outputs.push_back({0, 4096, "arena"});
    simaai::gst::TensorBufferPublishLogicalOutput tensor;
    tensor.logical_index = index;
    tensor.physical_index = tensor.memory_index = 0;
    tensor.byte_offset = pad * 64;
    tensor.backend_output_index = tensor.route_slot = index;
    tensor.logical_name = tensor.backend_name = tensor.segment_name = output;
    tensor.size_bytes = out.storage_bytes;
    tensor.dtype = SIMA_TENSOR_SET_DTYPE_BF16_V1;
    tensor.shape = {2};
    tensor.stride_bytes = {2};
    publication->logical_outputs.push_back(std::move(tensor));
  }
  request.layout = std::move(layout);
  request.publication.contract = std::move(publication);
  request.publication.output_name = "output0";
  request.publication.input_caps = {gst_caps_from_string("application/x-sima-tensor"),
                                    gst_caps_unref};
  request.publication.output_caps = {gst_caps_from_string("application/x-sima-tensor"),
                                     gst_caps_unref};
  return request;
}
#include "terminal_host_output_test.inc"

void native_stage_admission_transaction() {
  namespace specs = integration::specbuilders;
  using integration::DeviceKind;
  std::shared_ptr<GstElement> pipeline(gst_pipeline_new("stage-admission"),
                                       [](auto* value) { gst_object_unref(value); });
  auto* first = gst_element_factory_make("neatkernelrunner", "first");
  auto* second = gst_element_factory_make("neatkernelrunner", "second");
  require(pipeline && first && second, "installed kernel-runner provider is required");
  gst_bin_add_many(GST_BIN(pipeline.get()), first, second, nullptr);
  for (auto* element : {first, second}) {
    auto* pad = gst_element_request_pad_simple(element, "sink_0");
    require(pad, "request fixture input pad unavailable");
    gst_object_unref(pad);
  }
  const auto check_stopped = [&]() {
    for (auto* element : {pipeline.get(), first, second}) {
      GstState state = GST_STATE_VOID_PENDING;
      gst_element_get_state(element, &state, nullptr, 0);
      require(state == GST_STATE_NULL, "cold policy integration started execution");
    }
  };
  const auto make_spec = [](const char* name) {
    integration::StageStaticSpec stage;
    stage.element_name = stage.logical_stage_id = name;
    stage.plugin_kind = "kernel";
    stage.frame_arena_size_bytes = 4096;
    stage.frame_arena_role = integration::FrameArenaRole::Allocate;
    stage.frame_arena_storage_domain = integration::static_contract::ArenaStorageDomain::Cma;
    stage.frame_arena_provenance =
        integration::static_contract::ArenaAllocationProvenance::CoreAllocated;
    stage.frame_arena_required_device_access =
        static_cast<std::uint32_t>(integration::static_contract::ArenaDeviceAccess::CpuA65);
    stage.logical_inputs.push_back(specs::build_logical_input_static_spec(
        0, 0, 0, {2}, "FP32", "C", "input", "input", "input"));
    stage.input_bindings.push_back(specs::build_input_binding_static_spec(0, 0, "input", "input"));
    auto recipe = std::make_shared<integration::KernelStageRecipe>();
    const std::array inputs{rt::TensorSpec{{2}, rt::DType::Float32}};
    const std::array outputs{rt::TensorSpec{{2}, rt::DType::BFloat16}};
    auto executable = rt::check_kernel({"cast", rt::Hardware::A65, inputs, outputs, {}});
    require(std::holds_alternative<rt::KernelExecutable>(executable),
            "canonical request fixture admission failed");
    auto checked = std::get<rt::KernelExecutable>(std::move(executable));
    auto inspected = rt::inspect_kernel(checked);
    require(std::holds_alternative<rt::KernelInspection>(inspected),
            "canonical request fixture inspection failed");
    const auto& inspection = std::get<rt::KernelInspection>(inspected);
    require(inspection.input_requirements().size() == 1 &&
                inspection.output_requirements().size() == 1,
            "canonical cast fixture must have exactly one input and output");
    const auto input_requirement = inspection.input_requirements().front();
    const auto output_requirement = inspection.output_requirements().front();
    // Core authors storage placement, but Kernels owns the executable's exact
    // extent/alignment contract. A generic physical spec's zero alignment is
    // not a placed arena output and must not make this transaction fail early.
    stage.physical_inputs.push_back(specs::build_physical_buffer_static_spec(
        0, 0, input_requirement.storage_bytes, DeviceKind::Cpu, "input", -1, 0,
        input_requirement.alignment_bytes));
    stage.physical_outputs.push_back(specs::build_physical_buffer_static_spec(
        0, 0, output_requirement.storage_bytes, DeviceKind::Cpu, "output", -1, 0,
        output_requirement.alignment_bytes));
    recipe->members.push_back({std::move(checked), {{0, 0}}, {{0, 0}}, {}});
    stage.logical_outputs.push_back(specs::build_logical_output_static_spec(
        0, 0, 0, 0, 0, {2}, "BF16", "C", "logical_alias", "backend_alias", "source_segment", 0,
        output_requirement.storage_bytes));
    stage.output_order.push_back(
        specs::build_output_route_static_spec(0, 0, 0, "public_alias", "source_segment"));
    stage.native_kernel = std::move(recipe);
    integration::SimaPluginStaticManifest planned;
    planned.stages.push_back(std::move(stage));
    integration::finalize_pipeline_storage(planned);
    stage = std::move(planned.stages.front());
    gs::TensorStagePublication publication;
    std::string error;
    require_fixture(integration::build_kernel_stage_publication(stage, &publication, &error),
                    "finalize native publication", error);
    publication.input_caps = {gst_caps_from_string("application/vnd.simaai.tensor,format=FP32"),
                              gst_caps_unref};
    publication.output_caps = {gst_caps_from_string("application/vnd.simaai.tensor,format=BF16"),
                               gst_caps_unref};
    stage.tensor_publication =
        std::make_shared<const gs::TensorStagePublication>(std::move(publication));
    return stage;
  };
  integration::SimaPluginStaticManifest manifest;
  manifest.stages = {make_spec("first"), make_spec("second")};
  // Admission uses the exact finalized descriptor, without a manifest lookup.
  {
    const auto& contract = *manifest.stages.front().tensor_publication->contract;
    const auto& logical = contract.logical_outputs.front();
    const auto& physical = contract.physical_outputs.at(logical.memory_index);
    require(logical.segment_name == physical.segment_name &&
                contract.output_order.front().segment_name == physical.segment_name &&
                logical.logical_name == "logical_alias" &&
                logical.backend_name == "backend_alias" &&
                contract.output_order.front().cm_output_name == "public_alias",
            "final publication must retain aliases on the exact allocation segment");
    simaai::gst::TensorBufferPreparedMetaTemplate metadata;
    std::string error;
    require_fixture(
        simaai::gst::tensor_buffer_prepare_meta_template_from_contract(contract, &metadata, &error),
        "prepare final canonical publication", error);
    const auto compact = simaai::gst::tensor_buffer_freeze_publication(metadata, &error);
    require_fixture(bool(compact), "freeze final canonical publication", error);
    const auto& stage = manifest.stages[0];
    const auto& native = *stage.native_kernel;
    gs::StageRequest request;
    kernel(request).target = native.target;
    kernel(request).members = native.members;
    request.publication = *stage.tensor_publication;
    request.layout = stage.frame_layout;
    probe::RouteProbe prepared;
    const auto failure = prepared.prepare(request);
    if (failure)
      throw std::runtime_error("valid first request rejected: " + failure->message);
    require(prepared.member_count() == 1 && prepared.input_pads() == 1 && prepared.has_bridge(),
            "valid first request did not retain its canonical prepared stage");
  }
  check_stopped();
  // A complete first request must move into its attachment, not get copied into
  // a second vector just to inspect the EV service budget. Reject the following
  // request at Core's existing capacity boundary, before Scheduler can copy or
  // instantiate any factory during its own transaction.
  struct CopyWitness {
    unsigned* copies;
    explicit CopyWitness(unsigned& count) : copies(&count) {}
    CopyWitness(const CopyWitness& other) : copies(other.copies) {
      ++*copies;
    }
    CopyWitness(CopyWitness&&) = default;
  };
  unsigned factory_copies = 0, factory_calls = 0;
  auto observed = manifest.stages[0];
  auto observed_recipe = std::make_shared<integration::KernelStageRecipe>(*observed.native_kernel);
  observed_recipe->resolver_factories.push_back(
      {0, [witness = CopyWitness(factory_copies), &factory_calls] {
         ++factory_calls;
         return integration::KernelStageRecipe::Resolver{
             [](const rt::VariantPreparer& preparer,
                std::span<GstBuffer* const>) -> rt::Result<rt::PreparedSelection> {
               return preparer.base();
             }};
       }});
  observed.native_kernel = std::move(observed_recipe);
  for (const auto invalid_capacity : {std::size_t{0}, std::size_t{65}}) {
    const std::array copy_requests{
        integration::SchedulerStageRequest{"first", 1, 0, {}, &observed},
        integration::SchedulerStageRequest{"second", invalid_capacity, 0, {}, &manifest.stages[1]}};
    factory_copies = 0;
    const auto failure = integration::prepare_and_attach_stages(pipeline.get(), copy_requests);
    require(failure && failure->field == "kernel-runner.graph" &&
                failure->message == "Core execution capacity must be between 1 and 64",
            "invalid capacity bypassed Core cold admission");
    require(factory_copies == 1 && factory_calls == 0,
            "Core copied complete cold requests beyond the single recipe-to-request transfer");
    for (auto* element : {first, second})
      require(!probe::inspect_stage(element).valid,
              "rejected capacity attached a partial prepared stage");
  }
  check_stopped();
  auto malformed = manifest.stages[1];
  auto malformed_metadata =
      std::make_shared<gs::TensorStagePublication>(*malformed.tensor_publication);
  auto malformed_publication =
      std::make_shared<simaai::gst::TensorBufferPublishContract>(*malformed_metadata->contract);
  require(malformed_publication->logical_outputs[0].size_bytes > 0,
          "canonical output fixture must have a nonempty logical extent");
  --malformed_publication->logical_outputs[0].size_bytes;
  malformed_metadata->contract = std::move(malformed_publication);
  malformed.tensor_publication = std::move(malformed_metadata);
  std::array requests{integration::SchedulerStageRequest{"first", 1, 0, {}, &manifest.stages[0]},
                      integration::SchedulerStageRequest{"second", 1, 0, {}, &malformed}};
  const auto rejected = integration::prepare_and_attach_stages(pipeline.get(), requests);
  require(rejected && rejected->code == rt::ErrorCode::InvalidBinding &&
              rejected->field == "output.span",
          "malformed second request did not reach its checked logical output extent");
  for (auto* element : {first, second}) {
    require(!probe::inspect_stage(element).valid,
            "a rejected cohort partially attached a prepared stage");
  }
  check_stopped();
  requests[1].stage = &manifest.stages[1];
  const auto accepted = integration::prepare_and_attach_stages(pipeline.get(), requests);
  if (accepted)
    throw std::runtime_error("real cohort rejected: " + accepted->field + ": " + accepted->message);
  for (auto* element : {first, second}) {
    const auto stage = probe::inspect_stage(element);
    require(stage.valid && stage.members.size() == 1 && stage.input_pads == 1 && stage.bridge,
            "real canonical request was not prepared and retained by its element");
  }
  check_stopped();
}

void empty_candidate_bindings() {
  auto* pipeline = gst_pipeline_new("empty-candidate");
  require(pipeline, "candidate pipeline is unavailable");
  require(!integration::prepare_and_attach_stages(pipeline, {}),
          "plain candidate bin required numerical preparation");

  auto* runner = gst_element_factory_make("neatkernelrunner", "unbound");
  require(runner && gst_bin_add(GST_BIN(pipeline), runner), "candidate runner is unavailable");
  require(g_object_class_find_property(G_OBJECT_GET_CLASS(runner), "config") == nullptr,
          "kernel runner must not expose a legacy standalone JSON config property");
  require(bool(integration::prepare_and_attach_stages(pipeline, {})),
          "empty preparation requests bypassed unbound runner validation");
  require(!probe::inspect_stage(runner).valid,
          "rejected unbound runner received a prepared context");
  GstState state = GST_STATE_VOID_PENDING;
  gst_element_get_state(runner, &state, nullptr, 0);
  require(state == GST_STATE_NULL, "cold validation started the unbound runner");
  gst_object_unref(pipeline);
}
void multi_input_candidate_pads() {
  auto* pipeline = gst_pipeline_new("multi-input-candidate");
  auto* runner = gst_element_factory_make("neatkernelrunner", "multi-input");
  require(pipeline && runner && gst_bin_add(GST_BIN(pipeline), runner),
          "multi-input candidate is unavailable");
  const std::array bindings{gs::StageAttachment{runner, cast_request(2)}};
  // Request order is not port identity. Both sink_0 and sink_1 are required,
  // even when sink_1 is created first.
  auto* second = gst_element_request_pad_simple(runner, "sink_1");
  require(second, "second input pad request failed");
  require(bool(gs::prepare_and_attach_stages(pipeline, bindings)),
          "multi-input candidate accepted a missing first pad");
  require(!probe::inspect_stage(runner).valid, "missing input partly attached a context");
  auto* first = gst_element_request_pad_simple(runner, "sink_0");
  auto* extra = gst_element_request_pad_simple(runner, "sink_2");
  require(first && extra, "input pad requests failed");
  require(bool(gs::prepare_and_attach_stages(pipeline, bindings)),
          "multi-input candidate accepted an undeclared input pad");
  require(!probe::inspect_stage(runner).valid, "undeclared input partly attached a context");
  gst_element_release_request_pad(runner, extra);
  gst_object_unref(extra);
  require(!gs::prepare_and_attach_stages(pipeline, bindings),
          "complete multi-input candidate rejected out-of-order pad creation");
  const auto admitted = probe::inspect_stage(runner);
  require(admitted.valid && admitted.bridge && admitted.input_pads == 2 &&
              admitted.members.size() == 2 && admitted.members[0].prepared &&
              admitted.members[1].prepared,
          "multi-input candidate retained the wrong stage");
  gst_object_unref(first);
  gst_object_unref(second);
  gst_object_unref(pipeline);
}
void candidate_bin_pad_coverage() {
  auto* pipeline = gst_pipeline_new("candidate");
  auto* first = gst_element_factory_make("neatkernelrunner", "first");
  auto* second = gst_element_factory_make("neatkernelrunner", "second");
  require(pipeline && first && second, "candidate factory is unavailable");
  gst_bin_add_many(GST_BIN(pipeline), first, second, nullptr);
  auto* first_pad = gst_element_request_pad_simple(first, "sink_%u");
  auto* second_pad = gst_element_request_pad_simple(second, "sink_%u");
  std::array bindings{gs::StageAttachment{first, cast_request()},
                      gs::StageAttachment{second, cast_request()}};
  require(bool(gs::prepare_and_attach_stages(pipeline, std::span{bindings}.first(1))),
          "unbound candidate element accepted");
  require(!probe::inspect_stage(first).valid, "whole-bin preflight was not atomic");
  gst_element_release_request_pad(second, second_pad);
  gst_object_unref(second_pad);
  require(bool(gs::prepare_and_attach_stages(pipeline, bindings)), "missing required pad accepted");
  second_pad = gst_element_request_pad_simple(second, "sink_%u");
  require(!gs::prepare_and_attach_stages(pipeline, bindings), "candidate attachment failed");
  gst_object_unref(first_pad);
  gst_object_unref(second_pad);
  gst_object_unref(pipeline);
}
void immutable_recipe_instances() {
  // Core wiring is pure immutable recipe authoring. Actual instance creation
  // belongs to Scheduler's whole-cohort transaction, after complete checking.
  integration::KernelStageRecipe recipe;
  auto request = cast_request(2);
  recipe.members = kernel(request).members;
  unsigned constructions = 0;
  bool fail = true;
  std::vector<std::weak_ptr<std::byte>> instances;
  recipe.resolver_factories.push_back(
      {0, [&] {
         ++constructions;
         auto state = std::make_shared<std::byte>();
         instances.push_back(state);
         return integration::KernelStageRecipe::Resolver{
             [state](const rt::VariantPreparer& preparer,
                     std::span<GstBuffer* const>) -> rt::Result<rt::PreparedSelection> {
               return preparer.base();
             }};
       }});
  recipe.resolver_factories.push_back(
      {1, [&]() -> integration::KernelStageRecipe::Resolver {
         if (fail)
           throw std::runtime_error("injected cold instance failure");
         return [](const rt::VariantPreparer& preparer,
                   std::span<GstBuffer* const>) -> rt::Result<rt::PreparedSelection> {
           return preparer.base();
         };
       }});
  require(!integration::wire_kernel_stage_factories(recipe, kernel(request)) && constructions == 0,
          "cold wiring instantiated live mutable state");
  const auto rejected = probe::StageProbe::create(request);
  const auto* failure = std::get_if<rt::Error>(&rejected);
  require(failure && failure->origin == rt::ErrorOrigin::Application &&
              failure->message == "injected cold instance failure" && instances[0].expired(),
          ("factory exception lost provenance or retained partial instance state; actual=" +
           (failure ? std::string(rt::error_code_name(failure->code)) + ":" +
                          rt::error_phase_name(failure->phase) + ":" + failure->field + ":" +
                          failure->message +
                          "; origin=" + std::to_string(static_cast<int>(failure->origin))
                    : std::string("unexpected successful preparation")) +
           "; constructed=" + std::to_string(constructions) +
           "; retained=" + std::to_string(!instances.empty() && !instances[0].expired()))
              .c_str());
  fail = false;
  auto prepared = probe::StageProbe::create(request);
  require(std::holds_alternative<probe::StageProbe>(prepared) && constructions == 2,
          "fresh retry failed to instantiate exactly once");
  auto copy = request;
  require(bool(integration::wire_kernel_stage_factories(recipe, kernel(copy))) &&
              constructions == 2,
          "already-wired request was silently rewritten or instantiated twice");
  // Copying immutable authoring starts a new cold admission, never shares a cache.
  auto independent = probe::StageProbe::create(copy);
  require(std::holds_alternative<probe::StageProbe>(independent) && constructions == 3,
          "independent request did not instantiate new state");
  const auto source_recipe = recipe;
  request = {};
  copy = {};
  require(!instances[1].expired(), "clearing authoring released admitted state");
  std::get<probe::StageProbe>(prepared).reset();
  require(instances[1].expired() && !instances[2].expired(),
          "reusable recipe retained live state or releasing A invalidated B");
  std::get<probe::StageProbe>(independent).reset();
  require(instances[2].expired() && !source_recipe.members[0].resolve_factory,
          "copied reusable recipe acquired execution state");
  auto invalid = recipe;
  invalid.resolver_factories.push_back(invalid.resolver_factories.front());
  auto malformed = cast_request(2);
  require(bool(integration::wire_kernel_stage_factories(invalid, kernel(malformed))) &&
              constructions == 3,
          "invalid factory list constructed state before validation");

  auto image = cast_request();
  kernel(image).members.front().inputs.clear();
  kernel(image).frame_adapter_factory =
      []() -> rt::Result<std::shared_ptr<gs::TensorStageFrameAdapter>> {
    return rt::Error{rt::ErrorCode::ResourceBusy,
                     "test.factory",
                     "declared construction failure",
                     {},
                     {},
                     rt::ErrorPhase::ImplementationAdmission,
                     "test.rule",
                     rt::ErrorOrigin::Application};
  };
  const auto declared = probe::StageProbe::create(image);
  const auto* error = std::get_if<rt::Error>(&declared);
  require(error && error->code == rt::ErrorCode::ResourceBusy && error->field == "test.factory" &&
              error->message == "declared construction failure" && error->rule_id == "test.rule" &&
              error->origin == rt::ErrorOrigin::Application,
          "declared factory failure was reclassified or committed partial state");
  std::cout << "immutable recipe: independent admission, retry and ownership PASS\n";
}

void boxdecode_geometry_variants() {
  namespace k = simaai::scheduler::client;
  const std::array inputs{k::TensorSpec{{1, 1, 4}, k::DType::Int8},
                          k::TensorSpec{{1, 1, 2}, k::DType::Int8}};
  const std::array outputs{
      k::TensorSpec{{4, 4}, k::DType::Int32}, k::TensorSpec{{4}, k::DType::Float32},
      k::TensorSpec{{4}, k::DType::Int32}, k::TensorSpec{{1}, k::DType::Int32}};
  const std::array parameters{
      k::Parameter{"profile", "yolo26_v1"},
      k::Parameter{"heads", k::ParameterRecords{{{"boxes_input", 0U}, {"scores_input", 1U}}}},
      k::Parameter{
          "quantization",
          k::ParameterRecords{{{"input", 0U}, {"dequant_scale", .25f}, {"zero_point", 0}},
                              {{"input", 1U}, {"dequant_scale", .25f}, {"zero_point", 0}}}},
      k::Parameter{"model_input_size", k::ParameterArray{8, 8}},
      k::Parameter{"original_image_size", k::ParameterArray{8, 8}},
      k::Parameter{"num_classes", 2},
      k::Parameter{"max_detections", 4},
      k::Parameter{"detection_threshold", .5f},
      k::Parameter{"nms_iou_threshold", .5f},
      k::Parameter{"score_activation", "sigmoid"}};
  auto checked = rt::check_kernel({"boxdecode", k::Target::A65, inputs, outputs, parameters});
  require(std::holds_alternative<rt::KernelExecutable>(checked),
          "BoxDecode geometry fixture rejected");
  probe::OperationProbe executor;
  require(!executor.prepare(std::get<rt::KernelExecutable>(std::move(checked))),
          "BoxDecode geometry admission failed");
  const auto base = executor.operation();
  integration::KernelStageRecipe recipe;
  recipe.members.resize(1);
  recipe.members.front().executable = base.executable();
  integration::attach_boxdecode_geometry_resolver(
      recipe, 0, {8, 8, 8, 8, simaai::neat::ResizeMode::Letterbox}, true, true);
  struct Instance {
    struct Member {
      rt::KernelExecutable executable;
      rt::VariantResolver resolve;
    };
    rt::KernelProgram request;
    std::vector<Member> members;
  };
  const auto instantiate = [&] {
    Instance result;
    result.request.members = recipe.members;
    require(!integration::wire_kernel_stage_factories(recipe, result.request),
            "fresh checked BoxDecode factory wiring failed");
    for (const auto& member : result.request.members)
      result.members.push_back({member.executable, member.resolve_factory()});
    return result;
  };
  auto instance = instantiate();
  auto& member = instance.members.front();
  require(!recipe.members.front().resolve_factory && recipe.resolver_factories.size() == 1,
          "reusable recipe acquired a live resolver");
  require(bool(integration::wire_kernel_stage_factories(recipe, instance.request)),
          "active resolver instance was reset by a second instantiation");
  if (!gst_meta_get_info("GstSimaMeta")) {
    const gchar* tags[] = {nullptr};
    require(gst_meta_register_custom("GstSimaMeta", tags, nullptr, nullptr, nullptr),
            "register preprocess fixture metadata");
  }
  probe::GstBufferOwner first(gst_buffer_new()), second(gst_buffer_new());
  auto* meta =
      gst_custom_meta_get_structure(gst_buffer_add_custom_meta(first.get(), "GstSimaMeta"));
  auto* other =
      gst_custom_meta_get_structure(gst_buffer_add_custom_meta(second.get(), "GstSimaMeta"));
  const auto geometry = [](GstStructure* target, int width, const char* resize) {
    gst_structure_set(target, "preproc_original_width", G_TYPE_INT, width,
                      "preproc_original_height", G_TYPE_INT, width, "preproc_resize_mode",
                      G_TYPE_STRING, resize, nullptr);
  };
  const std::array<GstBuffer*, 2> buffers{first.get(), second.get()};
  geometry(meta, 16, "letterbox");
  geometry(other, 16, "letterbox");
  auto selected = member.resolve(executor.preparer(), buffers);
  require(std::holds_alternative<rt::PreparedSelection>(selected),
          "checked geometry variant rejected");
  const auto variant = std::get<rt::PreparedSelection>(selected);
  require(variant != base &&
              rt::same_kernel_contract_resources(variant.executable(), base.executable()),
          "geometry rebuilt tensor contract or numerical scratch");
  auto repeated = member.resolve(executor.preparer(), buffers);
  require(std::get<rt::PreparedSelection>(repeated) == variant,
          "warm geometry did not reuse immutable variant");
  // Actual cold factory path: one reusable recipe must not share caches even
  // when preparations use the same Runtime domain or execute concurrently.
  auto sibling = instantiate();
  probe::OperationProbe sibling_executor;
  require(!sibling_executor.prepare(member.executable, {.domain = executor.context()}),
          "sibling preparation in shared domain failed");
  const auto sibling_base = sibling_executor.operation();
  const auto sibling_variant = std::get<rt::PreparedSelection>(
      sibling.members.front().resolve(sibling_executor.preparer(), buffers));
  require(sibling_variant != variant && std::get<rt::PreparedSelection>(member.resolve(
                                            executor.preparer(), buffers)) == variant,
          "A/B/A in one Runtime domain discarded A's warmed variant");
  probe::OperationProbe independent_executor;
  require(!independent_executor.prepare(member.executable), "independent preparation failed");
  const auto independent_base = independent_executor.operation();
  auto independent = instantiate();
  const auto independent_variant = std::get<rt::PreparedSelection>(
      independent.members.front().resolve(independent_executor.preparer(), buffers));
  std::atomic<bool> isolated{true};
  const auto warm = [&](Instance& request, const probe::OperationProbe& exec,
                        const rt::PreparedSelection&, const rt::PreparedSelection& expected) {
    for (unsigned i = 0; i < 64; ++i) {
      auto result = request.members.front().resolve(exec.preparer(), buffers);
      const auto* value = std::get_if<rt::PreparedSelection>(&result);
      if (!value || *value != expected)
        isolated = false;
    }
  };
  std::thread a([&] { warm(instance, executor, base, variant); });
  std::thread b(
      [&] { warm(independent, independent_executor, independent_base, independent_variant); });
  a.join();
  b.join();
  require(isolated, "independent execution domains shared or rebuilt warmed cache state");
  auto copied_admission = instance;
  require(std::get<rt::PreparedSelection>(
              copied_admission.members.front().resolve(executor.preparer(), buffers)) == variant,
          "copies within one admission unexpectedly cloned live state");
  copied_admission = {};
  geometry(other, 32, "letterbox");
  require(std::holds_alternative<rt::Error>(
              independent.members.front().resolve(independent_executor.preparer(), buffers)),
          "independent instance accepted disagreeing split-head geometry");
  geometry(other, 16, "letterbox");
  require(std::get<rt::PreparedSelection>(member.resolve(executor.preparer(), buffers)) ==
                  variant &&
              std::get<rt::PreparedSelection>(independent.members.front().resolve(
                  independent_executor.preparer(), buffers)) == independent_variant,
          "failed B frame evicted A or B's warmed immutable variant");
  sibling = {};
  independent = {};
  require(!recipe.members.front().resolve_factory,
          "reusable recipe acquired execution state through its factory");
  for (const int width : {24, 40, 48, 56, 64}) {
    geometry(meta, width, "letterbox");
    geometry(other, width, "letterbox");
    auto evicting = member.resolve(executor.preparer(), buffers);
    require(std::holds_alternative<rt::PreparedSelection>(evicting),
            "bounded geometry cache rejected a valid replacement");
    const auto& operation = std::get<rt::PreparedSelection>(evicting);
    require(rt::same_kernel_contract_resources(operation.executable(), base.executable()),
            "geometry eviction changed checked storage or scratch authority");
  }
  geometry(meta, 16, "letterbox");
  geometry(other, 16, "letterbox");
  const auto rebuilt =
      std::get<rt::PreparedSelection>(member.resolve(executor.preparer(), buffers));
  require(rebuilt != variant &&
              rt::same_kernel_contract_resources(variant.executable(), base.executable()),
          "eviction invalidated a retained geometry snapshot or grew the bounded cache");
  probe::OperationProbe other_executor;
  require(!other_executor.prepare(member.executable),
          "second geometry execution domain failed admission");
  const auto other_base = other_executor.operation();
  const auto other_variant =
      std::get<rt::PreparedSelection>(member.resolve(other_executor.preparer(), buffers));
  require(other_variant != rebuilt && other_variant != variant && other_variant != other_base,
          "geometry cache reused another execution domain's admitted operation");
  const auto returned =
      std::get<rt::PreparedSelection>(member.resolve(executor.preparer(), buffers));
  require(returned != other_variant,
          "geometry cache retained the wrong execution domain after returning to the first base");
  geometry(other, 32, "letterbox");
  require(std::holds_alternative<rt::Error>(member.resolve(executor.preparer(), buffers)),
          "inconsistent split-head geometry accepted");
  geometry(meta, 32, "letterbox");
  auto changed = member.resolve(executor.preparer(), buffers);
  require(std::holds_alternative<rt::PreparedSelection>(changed) &&
              std::get<rt::PreparedSelection>(changed) != variant,
          "new image size mutated a retained variant");
  geometry(meta, 8, "stretch");
  geometry(other, 8, "stretch");
  require(std::holds_alternative<rt::Error>(member.resolve(executor.preparer(), buffers)),
          "changed resize contract accepted");
  geometry(meta, 8, "letterbox");
  geometry(other, 8, "letterbox");
  require(std::get<rt::PreparedSelection>(member.resolve(executor.preparer(), buffers)) == base,
          "base geometry not reused");
  gst_structure_remove_field(meta, "preproc_original_width");
  require(std::holds_alternative<rt::Error>(member.resolve(executor.preparer(), buffers)),
          "partial image-size metadata accepted");
  integration::KernelStageRecipe explicit_recipe;
  explicit_recipe.members = recipe.members;
  integration::attach_boxdecode_geometry_resolver(
      explicit_recipe, 0, {8, 8, 8, 8, simaai::neat::ResizeMode::Letterbox}, false, true);
  gs::KernelProgram explicit_request;
  explicit_request.members = explicit_recipe.members;
  require(!integration::wire_kernel_stage_factories(explicit_recipe, explicit_request),
          "explicit-size instance failed");
  auto explicit_size = explicit_request.members.front().resolve_factory();
  geometry(meta, 16, "letterbox");
  geometry(other, 32, "letterbox");
  require(std::get<rt::PreparedSelection>(explicit_size(executor.preparer(), buffers)) == base,
          "explicit original dimensions lost precedence over conflicting frame dimensions");
  geometry(other, 32, "stretch");
  require(std::holds_alternative<rt::Error>(explicit_size(executor.preparer(), buffers)),
          "explicit image size bypassed resize provenance validation");
  probe::GstBufferOwner bare(gst_buffer_new_allocate(nullptr, 4, nullptr));
  std::string bare_error;
  const bool bare_attached = simaai::gst::tensor_buffer_attach_frame_metadata(
      bare.get(), simaai::gst::TensorBufferFrameMetadata{}, &bare_error);
  require(!bare_attached &&
              bare_error.find("writable canonical compact envelope") != std::string::npos,
          "bare geometry carrier accepted frame facts without a static publication");
  const auto compact_owner = compact_geometry_publication();
  probe::GstBufferOwner compact_first(gst_buffer_new_allocate(nullptr, 4, nullptr));
  probe::GstBufferOwner compact_second(gst_buffer_new_allocate(nullptr, 4, nullptr));
  const auto attach_compact = [&compact_owner](GstBuffer* buffer, int width) {
    std::string error;
    const bool published =
        simaai::gst::tensor_buffer_attach_publication(buffer, compact_owner, &error);
    require_fixture(published, "attach compact geometry publication", error);
    simaai::gst::TensorBufferPreprocessFacts facts;
    facts.original_width = facts.original_height = width;
    facts.resized_width = facts.resized_height = 8;
    facts.primary = {0, 0, width, width};
    facts.placed = {0, 0, 8, 8};
    facts.affine = {8. / width, 0., 0., 0., 8. / width, 0.};
    facts.source_stride = width * 3;
    facts.resize_mode = "letterbox";
    facts.color_in = facts.color_out = "RGB";
    simaai::gst::TensorBufferFrameMetadata frame;
    frame.preprocess = simaai::gst::tensor_buffer_freeze_preprocess_facts(std::move(facts), &error);
    require_fixture(bool(frame.preprocess), "freeze compact geometry facts", error);
    const bool attached =
        simaai::gst::tensor_buffer_attach_frame_metadata(buffer, std::move(frame), &error);
    require_fixture(attached, "attach compact geometry frame facts", error);
  };
  attach_compact(compact_first.get(), 16);
  attach_compact(compact_second.get(), 32);
  const std::array<GstBuffer*, 2> compact_buffers{compact_first.get(), compact_second.get()};
  require(std::get<rt::PreparedSelection>(explicit_size(executor.preparer(), compact_buffers)) ==
              base,
          "compact frame facts overrode explicit original dimensions");
  require(std::holds_alternative<rt::Error>(member.resolve(executor.preparer(), compact_buffers)),
          "metadata-driven geometry ignored compact carrier disagreement");
  integration::KernelStageRecipe fixed;
  fixed.members = recipe.members;
  integration::attach_boxdecode_geometry_resolver(
      fixed, 0, {8, 8, 8, 8, simaai::neat::ResizeMode::Letterbox}, false, false);
  require(fixed.resolver_factories.empty(), "fixed explicit geometry added per-frame work");
  integration::KernelStageRecipe missing;
  missing.members.resize(1);
  bool rejected = false;
  try {
    integration::attach_boxdecode_geometry_resolver(
        missing, 0, {8, 8, 8, 8, simaai::neat::ResizeMode::Letterbox}, true, false);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected && missing.resolver_factories.empty(),
          "missing geometry capability did not fail before installing frame work");
}
namespace boxdecode_semantics {
using namespace simaai::neat;
using namespace simaai::neat::pipeline_internal::sima;
struct NumericalResult {
  int count;
  std::array<std::int32_t, 4> first;
  float score;
};
NumericalResult invoke(const CompiledKernelContract& compiled, float logit = 0.f) {
  namespace k = simaai::scheduler::client;
  const auto& checked = compiled.recipe->members.front().executable;
  auto inspected = rt::inspect_kernel(checked);
  require(std::holds_alternative<rt::KernelInspection>(inspected), "numerical inspection failed");
  const auto& executable = std::get<rt::KernelInspection>(inspected);
  require(executable.inputs().size() == 2 && executable.input_requirements().size() == 2 &&
              executable.output_requirements().size() == 4,
          "numerical fixture must retain its two checked heads and context");
  probe::OperationProbe operation;
  require(!operation.prepare(checked), "numerical context rejected");
  // Typed words guarantee eight-byte alignment; publish only each checked
  // byte capacity, not the allocator's rounded-up storage extent.
  std::vector<std::vector<std::uint64_t>> storage, results;
  std::vector<k::ConstBytes> inputs;
  std::vector<k::MutableBytes> outputs;
  for (auto requirement : executable.input_requirements()) {
    require(requirement.alignment_bytes <= alignof(std::uint64_t),
            "fixture input alignment changed");
    storage.emplace_back((requirement.storage_bytes + 7) / 8);
  }
  for (auto requirement : executable.output_requirements()) {
    require(requirement.alignment_bytes <= alignof(std::uint64_t),
            "fixture output alignment changed");
    results.emplace_back((requirement.storage_bytes + 7) / 8);
  }
  // Independent exact BF16 fixture: four .25 distances around the only cell;
  // sigmoid(logit) wins over sigmoid(-1000), with class0 winning equal scores.
  const auto put = [](std::byte* to, float value) {
    const auto bits = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16);
    std::memcpy(to, &bits, sizeof(bits));
  };
  for (int axis = 0; axis < 4; ++axis)
    put(reinterpret_cast<std::byte*>(storage[0].data()) + axis * 2, .25f);
  put(reinterpret_cast<std::byte*>(storage[1].data()), logit);
  put(reinterpret_cast<std::byte*>(storage[1].data()) + 2, -1000.f);
  for (std::size_t i = 0; i < storage.size(); ++i)
    inputs.push_back({reinterpret_cast<const std::byte*>(storage[i].data()),
                      executable.input_requirements()[i].storage_bytes});
  for (std::size_t i = 0; i < results.size(); ++i)
    outputs.push_back({reinterpret_cast<std::byte*>(results[i].data()),
                       executable.output_requirements()[i].storage_bytes});
  require(!operation.execute_host(operation.operation(), inputs, outputs),
          "lowered numerical fixture failed execution");
  NumericalResult result{};
  std::memcpy(&result.count, results[3].data(), sizeof(result.count));
  std::memcpy(result.first.data(), results[0].data(), sizeof(result.first));
  std::memcpy(&result.score, results[1].data(), sizeof(result.score));
  return result;
}
void controls_and_admission_geometry() {
  CompiledBoxDecodeContract s;
  s.payload.decode_type = BoxDecodeType::YoloV26;
  s.payload.num_classes = 2;
  s.payload.score_activation = BoxDecodeScoreActivation::Sigmoid;
  s.payload.decode_type_option = BoxDecodeTypeOption::InterleavedByHead;
  s.payload.topk = 120;
  s.payload.detection_threshold = .25;
  s.payload.nms_iou_threshold = .55;
  for (int port = 0; port < 2; ++port) {
    const int channels = port == 0 ? 4 : 2;
    const auto name = "input" + std::to_string(port);
    s.runtime_contract.logical_inputs.push_back(specbuilders::build_logical_input_static_spec(
        port, port, port, {1, 1, channels}, "BF16", "HWC", name, name, name));
    s.payload.tensor_storage_kind.push_back(1); // Dense HWC physical.
    sima_ev_shape_desc slice{};
    slice.rank = 3;
    slice.sizes[0] = slice.sizes[1] = 1;
    slice.sizes[2] = channels;
    s.payload.slice_shapes.push_back(slice);
    // These are the exact independently allocated BF16 head capacities used
    // by invoke, not capacities inferred from upstream logical-only metadata.
    const std::uint64_t bytes = port == 0 ? 8 : 4;
    s.runtime_contract.physical_inputs.push_back(specbuilders::build_physical_buffer_static_spec(
        port, port, bytes, DeviceKind::Cpu, "input" + std::to_string(port)));
    s.runtime_contract.input_bindings.push_back(specbuilders::build_input_binding_static_spec(
        port, port, "input" + std::to_string(port), "input" + std::to_string(port)));
  }
  const BoxDecodeGeometryOptions geometry{640, 640, 640, 360, ResizeMode::Letterbox};
  const auto admitted =
      compile_boxdecode_kernel(s, ContractCompileInput{}, geometry, ResizeMode::Letterbox);
  require(admitted.recipe->members.size() == 2 && admitted.recipe->resolver_factories.empty(),
          "complete explicit geometry added conversion or per-frame resolution");
  require(admitted.runtime_contract.logical_outputs.front().size_bytes == 4 + 120 * 24,
          "explicit topK120 did not determine the public BBOX capacity");
  CompiledNodeContract child;
  child.node_kind = "ModelFragment";
  child.element_name = child.logical_stage_id = "internal-boxdecode";
  child.renderable = true;
  child.kernel = admitted;
  CompiledNodeContract fragment;
  fragment.node_kind = "ModelFragment";
  fragment.element_name = fragment.logical_stage_id = "infer";
  fragment.child_stages.push_back(std::move(child));
  CompiledPipelineContracts contracts;
  contracts.fully_renderable = true;
  contracts.stages.push_back(std::move(fragment));
  ManifestBuildDiagnostics diagnostics;
  auto manifest = render_manifest_from_compiled_contracts(contracts, {}, &diagnostics);
  require(manifest && diagnostics.errors.empty() && manifest->stages.size() == 1,
          "checked internal BoxDecode child failed rendering");
  pipeline_internal::terminal_output_contract::PublicOutputEndpointSelector endpoint;
  endpoint.terminal_stage_key = "infer";
  endpoint.terminal_node_kind = "ModelFragment";
  endpoint.allow_unresolved_terminal_stage_fallback = true;
  finalize_pipeline_storage(*manifest, &endpoint);
  require(manifest->stages.front().frame_layout &&
              manifest->stages.front().frame_layout->allocations.front().policy ==
                  gs::AllocationPolicy::RequireHostCpu,
          "checked native BoxDecode child lost terminal HostCpu placement");
  const auto result = invoke(admitted);
  // Model640: center(320,320), extent(320,320). Independent letterbox to
  // 640x360 subtracts140 in Y, giving [160,20,320,320] after clipping/rounding.
  require(result.count == 1 && result.score == .5f &&
              result.first == std::array<std::int32_t, 4>{160, 20, 320, 320},
          "actual640x360 geometry or positive controls were lost in cold lowering");
  s.payload.detection_threshold = .75;
  require(
      invoke(compile_boxdecode_kernel(s, ContractCompileInput{}, geometry, ResizeMode::Letterbox))
              .count == 0,
      "explicit positive threshold did not reach numerical execution");
  s.payload.detection_threshold = s.payload.nms_iou_threshold = 0.;
  const auto zero =
      invoke(compile_boxdecode_kernel(s, ContractCompileInput{}, geometry, ResizeMode::Letterbox),
             -1000.f);
  require(zero.count == 1 && zero.score == 0.f,
          "zero controls must lower unchanged and retain a genuinely zero-score candidate");
  s.runtime_contract.physical_inputs.clear();
  try {
    (void)compile_boxdecode_kernel(s, ContractCompileInput{}, geometry, ResizeMode::Letterbox);
    throw std::runtime_error("missing authoritative physical heads accepted");
  } catch (const std::invalid_argument&) {
  }
}
} // namespace boxdecode_semantics
} // namespace
int main(int argc, char** argv) {
  try {
    const auto capacity_policy = [] {
      output_carrier_budget();
      authored_stage_budget();
      authored_preproc_contract_policy();
      ev_service_capacity();
    };
    if (argc == 2 && std::string_view(argv[1]) == "--capacity-policy-only") {
      capacity_policy();
      std::cout << "PASS: authored capacity, retained carriers, nested contracts and sync policy\n";
      return 0;
    }
    const bool host_cpu_only = argc == 2 && std::string_view(argv[1]) == "--terminal-host-cpu-only";
    // This explicit CPU-only profile requires the real factories used below,
    // not unrelated optional MLA/EV factories from Core's full installation.
    if (host_cpu_only) {
      sima_test::initialize_cold_contract_gstreamer();
      require(probe::register_element(), "register native candidate plugin");
      terminal_host_placement_proof();
      terminal_host_output_lifetime(true);
      return 0;
    }
    simaai::neat::gst_init_once();
    immutable_recipe_instances();
    if (argc == 2 && std::string_view(argv[1]) == "--instance-only")
      return 0;
    require(probe::register_element(), "register native candidate plugin");
    const bool cold_admission_only =
        argc == 2 && std::string_view(argv[1]) == "--cold-admission-only";
    terminal_host_placement_proof();
    if (!cold_admission_only)
      terminal_host_output_lifetime(false);
    if (argc == 2 && std::string_view(argv[1]) == "--terminal-host-only") {
      boxdecode_semantics::controls_and_admission_geometry();
      return 0;
    }
    capacity_policy();
    ev_service_capacity();
    native_stage_admission_transaction();
    empty_candidate_bindings();
    multi_input_candidate_pads();
    candidate_bin_pad_coverage();
    boxdecode_geometry_variants();
    boxdecode_semantics::controls_and_admission_geometry();
    std::cout << "PASS: cold whole-bin preflight, immutable recipes and typed stage attachment\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}

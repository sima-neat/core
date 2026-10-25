#include "pipeline/internal/sima/TerminalHostPlacement.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "gst/GstInit.h"
#include "pipeline/internal/sima/KernelRunnerIntegration.h"
#include "pipeline/internal/sima/BoxDecodeGeometry.h"
#include "pipeline/internal/sima/BoxDecodeKernelAdmission.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "nodes/sima/Preproc.h"
#include <simaai/scheduler/testing/probe.h>
#include <gstsimaaitensorbuffer_meta.h>
#include <simaai/scheduler/testing/route_probe.h>

#include "tensor_output_contract_fixtures.h"
#include "pipeline/internal/HolderLoanGate.h"
#include "pipeline/internal/OutputQueueMemory.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/RealtimeFrameCredit.h"
#include "graph/internal/GraphRunState.h"
#include <simaai/scheduler/client/memory_access.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
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
  stage.native_kernel = recipe;
  stage.core_fresh_default_arena = true;
  stage.frame_arena_role = integration::FrameArenaRole::Allocate;
  stage.frame_arena_size_bytes = 4096;
  stage.frame_arena_storage_domain = domain::ArenaStorageDomain::Cma;
  stage.frame_arena_provenance = domain::ArenaAllocationProvenance::CoreAllocated;
  stage.frame_arena_required_device_access =
      static_cast<std::uint32_t>(domain::ArenaDeviceAccess::CpuA65);
  stage.frame_arena_escape_policy = domain::ArenaEscapePolicy::CpuMappablePublic;
  stage.physical_outputs.push_back(specs::build_physical_buffer_static_spec(
      0, 0, 4, integration::DeviceKind::Cpu, "output", -1, 0, 64));
  stage.logical_outputs.push_back(specs::build_logical_output_static_spec(
      0, 0, 0, 0, 0, {2}, "BF16", "", "output", "output", "output", 0, 4));
  stage.output_order.push_back(specs::build_output_route_static_spec(0, 0, 0, "output", "output"));
  // Exercise the real renderer: identical rendered CMA fields do not erase
  // whether the caller authored a placement requirement before rendering.
  simaai::neat::CompiledNodeContract compiled;
  compiled.node_kind = "Cast";
  compiled.element_name = compiled.logical_stage_id = stage.element_name;
  compiled.renderable = true;
  compiled.kernel.emplace();
  compiled.kernel->recipe = recipe;
  compiled.kernel->runtime_contract.physical_outputs = stage.physical_outputs;
  compiled.kernel->runtime_contract.logical_outputs = stage.logical_outputs;
  compiled.kernel->runtime_contract.output_order = stage.output_order;
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
  require(render(compiled).core_fresh_default_arena, "fresh Cast origin was lost");
  const auto authored = [&](auto set) {
    auto node = compiled;
    set(node.kernel->runtime_contract);
    require(!render(node).core_fresh_default_arena, "authored placement acquired fresh origin");
  };
  authored([](auto& r) { r.frame_arena_storage_domain = domain::ArenaStorageDomain::Cma; });
  authored([](auto& r) { r.frame_arena_storage_domain = domain::ArenaStorageDomain::Dms; });
  authored([](auto& r) { r.frame_arena_required_device_access = 2U; });
  authored(
      [](auto& r) { r.frame_arena_escape_policy = domain::ArenaEscapePolicy::CpuMappablePublic; });
  authored([](auto& r) {
    r.frame_arena_allocation_policy = domain::ArenaAllocationPolicy::PreferDmsThenCma;
  });
  integration::SimaPluginStaticManifest manifest;
  manifest.stages.push_back(stage);
  require(integration::terminal_host_arena_eligible(stage, manifest),
          "generic Cast host eligibility");
  const auto reject = [&](auto change) {
    auto invalid = stage;
    change(invalid);
    require(!integration::terminal_host_arena_eligible(invalid, manifest),
            "unsafe host placement accepted");
  };
  reject([](auto& v) { v.core_fresh_default_arena = false; });
  reject([](auto& v) { v.frame_arena_required_device_access |= 2U; });
  reject([](auto& v) { v.frame_arena_storage_domain = domain::ArenaStorageDomain::Dms; });
  reject([](auto& v) {
    v.frame_arena_allocation_policy = domain::ArenaAllocationPolicy::PreferDmsThenCma;
  });
  reject([](auto& v) { v.frame_arena_role = integration::FrameArenaRole::ReuseInput; });
  reject([](auto& v) { v.physical_outputs[0].memory_flags = 1; });
  reject([](auto& v) { v.physical_outputs[0].source_physical_index = 0; });
  reject([](auto& v) { v.physical_outputs[0].size_bytes = 4097; });
  reject([](auto& v) { v.physical_outputs[0].device_kind = integration::DeviceKind::Mla; });
  reject([](auto& v) { v.physical_outputs.push_back(v.physical_outputs[0]); });
  reject([](auto& v) { v.logical_outputs.clear(); });
  auto consumer = stage;
  consumer.element_name = consumer.logical_stage_id = "later-consumer";
  integration::InputBindingStaticSpec binding;
  binding.src_stage_id = stage.logical_stage_id;
  consumer.input_bindings.push_back(binding);
  manifest.stages.push_back(consumer);
  require(!integration::terminal_host_arena_eligible(stage, manifest),
          "intermediate arena accepted");
  std::shared_ptr<GstElement> pipeline(gst_pipeline_new(nullptr), gst_object_unref);
  auto* source = gst_element_factory_make("identity", nullptr);
  auto* sink = gst_element_factory_make("appsink", nullptr);
  require(source && sink, "host proof topology factories");
  gst_bin_add_many(GST_BIN(pipeline.get()), source, sink, nullptr);
  require(gst_element_link(source, sink), "host proof direct edge");
  require(integration::terminal_host_direct_peer(pipeline.get(), source, sink),
          "direct stopped proof");
  gst_element_unlink(source, sink);
  require(!integration::terminal_host_direct_peer(pipeline.get(), source, sink),
          "missing peer accepted");
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
    result.request.target = target;
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

gs::TensorStageRequest cast_request(std::size_t input_pads = 1, bool terminal_host = false) {
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
  integration::StageStaticSpec stage;
  stage.element_name = stage.logical_stage_id = "cast-fixture";
  stage.plugin_kind = "kernel";
  stage.frame_arena_size_bytes = 4096;
  stage.frame_arena_role = integration::FrameArenaRole::Allocate;
  stage.frame_arena_storage_domain = integration::static_contract::ArenaStorageDomain::Cma;
  stage.frame_arena_provenance =
      integration::static_contract::ArenaAllocationProvenance::CoreAllocated;
  stage.frame_arena_required_device_access =
      static_cast<std::uint32_t>(integration::static_contract::ArenaDeviceAccess::CpuA65);
  if (terminal_host) {
    stage.frame_arena_storage_domain = integration::static_contract::ArenaStorageDomain::HostCpu;
    stage.frame_arena_escape_policy =
        integration::static_contract::ArenaEscapePolicy::CpuOnlyPublic;
  }
  gs::TensorStageRequest request;
  request.target = rt::Hardware::A65;
  auto publication = std::make_shared<simaai::gst::TensorBufferPublishContract>();
  publication->stage_key = stage.logical_stage_id;
  for (std::size_t pad = 0; pad < input_pads; ++pad) {
    const auto index = static_cast<int>(pad);
    const auto input = "input" + std::to_string(pad), output = "output" + std::to_string(pad);
    stage.logical_inputs.push_back(specs::build_logical_input_static_spec(
        index, index, index, {2}, "FP32", "C", input, input, input));
    stage.input_bindings.push_back(
        specs::build_input_binding_static_spec(index, index, input, input));
    stage.physical_inputs.push_back(specs::build_physical_buffer_static_spec(
        index, index, in.storage_bytes, integration::DeviceKind::Cpu, input, -1, 0,
        in.alignment_bytes));
    stage.physical_outputs.push_back(specs::build_physical_buffer_static_spec(
        index, index, out.storage_bytes, integration::DeviceKind::Cpu, output, -1,
        static_cast<std::int64_t>(pad * 64), out.alignment_bytes));
    request.members.push_back({executable, {{pad, 0}}, {{pad, 0}}, {}});
    publication->physical_outputs.push_back({index, out.storage_bytes, output});
    simaai::gst::TensorBufferPublishLogicalOutput tensor;
    tensor.logical_index = tensor.physical_index = tensor.memory_index = index;
    tensor.backend_output_index = tensor.route_slot = index;
    tensor.logical_name = tensor.backend_name = tensor.segment_name = output;
    tensor.size_bytes = out.storage_bytes;
    tensor.dtype = SIMA_TENSOR_SET_DTYPE_BF16_V1;
    tensor.shape = {2};
    tensor.stride_bytes = {2};
    publication->logical_outputs.push_back(std::move(tensor));
  }
  std::shared_ptr<GstElement> owner(gst_pipeline_new(nullptr), gst_object_unref);
  integration::SimaPluginStaticManifest manifest;
  manifest.stages.push_back(std::move(stage));
  std::string error;
  require_fixture(integration::attach_manifest_context(owner.get(), manifest, &error),
                  "serialize Cast placement", error);
  auto context = std::shared_ptr<GstContext>(
      gst_element_get_context(owner.get(), SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE),
      gst_context_unref);
  const auto* accessor = sima_plugin_manifest_context_accessor(context.get());
  require(accessor && accessor->stage_by_index, "Cast placement accessor unavailable");
  const auto* placement = accessor->stage_by_index(accessor->user_data, 0);
  require(placement, "Cast placement missing");
  request.placement = {std::move(context), placement};
  request.publication.contract = std::move(publication);
  request.publication.output_name = "output0";
  request.publication.input_caps = {gst_caps_from_string("application/x-sima-tensor"),
                                    gst_caps_unref};
  request.publication.output_caps = {gst_caps_from_string("application/x-sima-tensor"),
                                     gst_caps_unref};
  return request;
}
#include "terminal_host_output_test.inc"

void memory_coherency_candidate_policy() {
  for (const char* launch :
       {"appsrc ! queue ! appsink", "appsrc ! tee name=t t. ! queue ! appsink t. ! queue ! appsink",
        "appsrc ! identity signal-handoffs=false ! appsink",
        "appsrc ! identity signal-handoffs=true ! appsink", "appsrc ! fakesink"}) {
    GError* error = nullptr;
    auto* pipeline = gst_parse_launch(launch, &error);
    require(pipeline && !error, "coherency transport fixture could not be parsed");
    require(!integration::prepare_and_attach_kernel_runners(pipeline, {}),
            "transport-only cohort rejected");
    require(!g_object_get_qdata(G_OBJECT(pipeline), g_quark_from_static_string(
                                                        "sima-core-admitted-memory-coherency-v1")),
            "Core must not install graph-wide coherency selection");
    GstState state = GST_STATE_VOID_PENDING;
    gst_element_get_state(pipeline, &state, nullptr, 0);
    require(state == GST_STATE_NULL, "cold admission started pipeline execution");
    gst_object_unref(pipeline);
  }
}
// Exercise Core's real request -> installed Scheduler/Runner path, not an
// empty request vector or the name of a helper in a source file. The MLA
// producer remains stopped: this tests policy commit, not accelerator jobs.
void real_request_memory_policy_transaction(bool opaque_consumer) {
  namespace specs = integration::specbuilders;
  using integration::DeviceKind;
  std::shared_ptr<GstElement> pipeline(gst_pipeline_new("coherency-transaction"),
                                       [](auto* value) { gst_object_unref(value); });
  auto* mla = gst_element_factory_make("neatprocessmla", "producer");
  auto* first = gst_element_factory_make("neatkernelrunner", "first");
  auto* second = gst_element_factory_make("neatkernelrunner", "second");
  require(pipeline && mla && first && second,
          "installed MLA and kernel-runner providers are required for policy integration");
  gst_bin_add_many(GST_BIN(pipeline.get()), mla, first, second, nullptr);
  if (opaque_consumer) {
    auto* opaque = gst_element_factory_make("fakesink", "opaque");
    require(opaque && gst_bin_add(GST_BIN(pipeline.get()), opaque),
            "opaque policy fixture unavailable");
  }
  for (auto* element : {first, second}) {
    auto* pad = gst_element_request_pad_simple(element, "sink_0");
    require(pad, "request fixture input pad unavailable");
    gst_object_unref(pad);
  }
  const auto check_policy = [&]() {
    gchar* mode = nullptr;
    gboolean deferred = FALSE, handoff = TRUE;
    g_object_get(mla, "memory-coherency-mode", &mode, "defer-output-invalidate", &deferred,
                 "output-read-handoff", &handoff, nullptr);
    const bool matches = g_strcmp0(mode, "demand-local-v1") == 0 && deferred && !handoff;
    g_free(mode);
    require(matches, "MLA default prepared-access contract changed");
    for (const char* name :
         {"memory-coherency-mode", "defer-output-invalidate", "output-read-handoff"}) {
      auto* prop = g_object_class_find_property(G_OBJECT_GET_CLASS(mla), name);
      require(prop && !(prop->flags & G_PARAM_WRITABLE), "obsolete coherency selector is writable");
    }
    for (auto* element : {pipeline.get(), mla, first, second}) {
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
    auto contract = std::make_shared<simaai::gst::TensorBufferPublishContract>();
    contract->stage_key = name;
    contract->physical_outputs.push_back({0, output_requirement.storage_bytes, "output"});
    simaai::gst::TensorBufferPublishLogicalOutput output;
    output.logical_index = output.physical_index = output.memory_index = 0;
    output.backend_output_index = output.route_slot = 0;
    output.logical_name = output.backend_name = output.segment_name = "output";
    output.size_bytes = output_requirement.storage_bytes;
    output.dtype = SIMA_TENSOR_SET_DTYPE_BF16_V1;
    output.shape = {2};
    output.stride_bytes = {2};
    contract->logical_outputs.push_back(std::move(output));
    gs::TensorStagePublication publication;
    publication.input_caps = {gst_caps_from_string("application/vnd.simaai.tensor,format=FP32"),
                              gst_caps_unref};
    publication.output_caps = {gst_caps_from_string("application/vnd.simaai.tensor,format=BF16"),
                               gst_caps_unref};
    publication.contract = std::move(contract);
    publication.output_name = "output";
    recipe->publication = std::move(publication);
    stage.native_kernel = std::move(recipe);
    return stage;
  };
  integration::SimaPluginStaticManifest manifest;
  manifest.stages = {make_spec("first"), make_spec("second")};
  std::string error;
  require_fixture(integration::attach_manifest_context(pipeline.get(), manifest, &error),
                  "attach real policy manifest", error);
  // Prove the first request is fully admissible on the installed stack before
  // using an invalid second request to test all-or-nothing Core attachment.
  // Prepare a separate cold stage: no element context or MLA policy is changed.
  {
    auto context = std::shared_ptr<GstContext>(
        gst_element_get_context(pipeline.get(), SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE),
        gst_context_unref);
    SimaPluginManifestLookupStatus status{};
    const auto* placement = sima_plugin_manifest_context_stage_lookup_typed_checked(
        context.get(), "first", "first", SIMA_PLUGIN_STAGE_PAYLOAD_NONE, &status);
    require(placement, "first request placement is absent from the attached manifest");
    const auto& native = *manifest.stages[0].native_kernel;
    gs::TensorStageRequest request;
    request.target = native.target;
    request.members = native.members;
    request.publication = *native.publication;
    request.placement = {context, placement};
    probe::RouteProbe prepared;
    const auto failure = prepared.prepare(request);
    if (failure)
      throw std::runtime_error("valid first request rejected: " + failure->field + ": " +
                               failure->message);
    require(prepared.member_count() == 1 && prepared.input_pads() == 1 && prepared.has_bridge(),
            "valid first request did not retain its canonical prepared stage");
  }
  check_policy();
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
        integration::KernelRunnerStageRequest{"first", "first", 1, 0, {}, &observed},
        integration::KernelRunnerStageRequest{
            "second", "second", invalid_capacity, 0, {}, &manifest.stages[1]}};
    factory_copies = 0;
    const auto failure =
        integration::prepare_and_attach_kernel_runners(pipeline.get(), copy_requests);
    require(failure && failure->field == "kernel-runner.graph" &&
                failure->message == "Core execution capacity must be between 1 and 64",
            "invalid capacity bypassed Core cold admission");
    require(factory_copies == 1 && factory_calls == 0,
            "Core copied complete cold requests beyond the single recipe-to-request transfer");
    for (auto* element : {first, second})
      require(!probe::inspect_stage(element).valid,
              "rejected capacity attached a partial prepared stage");
  }
  check_policy();
  auto malformed = manifest.stages[1];
  auto malformed_recipe =
      std::make_shared<integration::KernelStageRecipe>(*malformed.native_kernel);
  auto malformed_publication = std::make_shared<simaai::gst::TensorBufferPublishContract>(
      *malformed_recipe->publication->contract);
  require(malformed_publication->logical_outputs[0].size_bytes > 0,
          "canonical output fixture must have a nonempty logical extent");
  --malformed_publication->logical_outputs[0].size_bytes;
  malformed_recipe->publication->contract = std::move(malformed_publication);
  malformed.native_kernel = std::move(malformed_recipe);
  std::array requests{
      integration::KernelRunnerStageRequest{"first", "first", 1, 0, {}, &manifest.stages[0]},
      integration::KernelRunnerStageRequest{"second", "second", 1, 0, {}, &malformed}};
  const auto rejected = integration::prepare_and_attach_kernel_runners(pipeline.get(), requests);
  require(rejected && rejected->field == "output.span" &&
              rejected->message == "kernel output exceeds its logical execution descriptor",
          "malformed second request did not reach its checked logical output extent");
  for (auto* element : {first, second}) {
    require(!probe::inspect_stage(element).valid,
            "a rejected cohort partially attached a prepared stage");
  }
  check_policy();
  requests[1].native_stage = &manifest.stages[1];
  const auto accepted = integration::prepare_and_attach_kernel_runners(pipeline.get(), requests);
  if (accepted)
    throw std::runtime_error("real cohort rejected: " + accepted->field + ": " + accepted->message);
  for (auto* element : {first, second}) {
    const auto stage = probe::inspect_stage(element);
    require(stage.valid && stage.members.size() == 1 && stage.input_pads == 1 && stage.bridge,
            "real canonical request was not prepared and retained by its element");
  }
  check_policy();
}

void empty_candidate_bindings() {
  auto* pipeline = gst_pipeline_new("empty-candidate");
  require(pipeline, "candidate pipeline is unavailable");
  require(!integration::prepare_and_attach_kernel_runners(pipeline, {}),
          "plain candidate bin required numerical preparation");

  auto* runner = gst_element_factory_make("neatkernelrunner", "unbound");
  require(runner && gst_bin_add(GST_BIN(pipeline), runner), "candidate runner is unavailable");
  require(g_object_class_find_property(G_OBJECT_GET_CLASS(runner), "config") == nullptr,
          "kernel runner must not expose a legacy standalone JSON config property");
  require(bool(integration::prepare_and_attach_kernel_runners(pipeline, {})),
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
  require(bool(gs::prepare_and_attach_tensor_stages(pipeline, bindings)),
          "multi-input candidate accepted a missing first pad");
  require(!probe::inspect_stage(runner).valid, "missing input partly attached a context");
  auto* first = gst_element_request_pad_simple(runner, "sink_0");
  auto* extra = gst_element_request_pad_simple(runner, "sink_2");
  require(first && extra, "input pad requests failed");
  require(bool(gs::prepare_and_attach_tensor_stages(pipeline, bindings)),
          "multi-input candidate accepted an undeclared input pad");
  require(!probe::inspect_stage(runner).valid, "undeclared input partly attached a context");
  gst_element_release_request_pad(runner, extra);
  gst_object_unref(extra);
  require(!gs::prepare_and_attach_tensor_stages(pipeline, bindings),
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
  require(bool(gs::prepare_and_attach_tensor_stages(pipeline, std::span{bindings}.first(1))),
          "unbound candidate element accepted");
  require(!probe::inspect_stage(first).valid, "whole-bin preflight was not atomic");
  gst_element_release_request_pad(second, second_pad);
  gst_object_unref(second_pad);
  require(bool(gs::prepare_and_attach_tensor_stages(pipeline, bindings)),
          "missing required pad accepted");
  second_pad = gst_element_request_pad_simple(second, "sink_%u");
  require(!gs::prepare_and_attach_tensor_stages(pipeline, bindings), "candidate attachment failed");
  gst_object_unref(first_pad);
  gst_object_unref(second_pad);
  gst_object_unref(pipeline);
}
void immutable_recipe_instances() {
  // Core wiring is pure immutable recipe authoring. Actual instance creation
  // belongs to Scheduler's whole-cohort transaction, after complete checking.
  integration::KernelStageRecipe recipe;
  auto request = cast_request(2);
  recipe.members = request.members;
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
  require(!integration::wire_kernel_stage_factories(recipe, request) && constructions == 0,
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
  require(bool(integration::wire_kernel_stage_factories(recipe, copy)) && constructions == 2,
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
  require(bool(integration::wire_kernel_stage_factories(invalid, malformed)) && constructions == 3,
          "invalid factory list constructed state before validation");

  auto image = cast_request();
  image.members.front().inputs.clear();
  image.frame_adapter_factory = []() -> rt::Result<std::shared_ptr<gs::TensorStageFrameAdapter>> {
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
    rt::TensorStageRequest request;
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
  gs::TensorStageRequest explicit_request;
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
    simaai::neat::gst_init_once();
    immutable_recipe_instances();
    if (argc == 2 && std::string_view(argv[1]) == "--instance-only")
      return 0;
    require(probe::register_element(), "register native candidate plugin");
    terminal_host_placement_proof();
    const bool host_cpu_only = argc == 2 && std::string_view(argv[1]) == "--terminal-host-cpu-only";
    terminal_host_output_lifetime(host_cpu_only);
    if (host_cpu_only || (argc == 2 && std::string_view(argv[1]) == "--terminal-host-only"))
      return 0;
    capacity_policy();
    ev_service_capacity();
    memory_coherency_candidate_policy();
    real_request_memory_policy_transaction(false);
    real_request_memory_policy_transaction(true);
    empty_candidate_bindings();
    multi_input_candidate_pads();
    candidate_bin_pad_coverage();
    boxdecode_geometry_variants();
    boxdecode_semantics::controls_and_admission_geometry();
    std::cout
        << "PASS: retained ABI4 source, cold whole-bin preflight, multi-input stage attachment\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}

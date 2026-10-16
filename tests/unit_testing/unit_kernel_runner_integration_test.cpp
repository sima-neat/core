#include "pipeline/internal/sima/KernelRunnerIntegration.h"
#include "pipeline/internal/MemoryCoherencyPolicy.h"
#include "pipeline/internal/sima/BoxDecodeGeometry.h"
#include "pipeline/internal/sima/BoxDecodeKernelAdmission.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include <simaai/kernels/authoring.h>
#include <gstsimaaitensorbuffer.h>

#include <simaai/runtime/gstreamer/prepared_tensor_stage.h>
#include <simaai/runtime/gstreamer/memory_coherency.h>
#include <simaai/runtime/gstreamer/tensor_bridge.h>
#include <simaai/runtime/kernel_preparation.h>
#include <simaai/scheduler/kernel_preparation.h>
#include <simaai/kernels/ops/cast.h>

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
namespace rt = simaai::runtime;
namespace gs = rt::gstreamer;
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
void ev_service_capacity() {
  const auto request = [](rt::Hardware target, std::size_t capacity) {
    gs::TensorStageRequest result;
    result.target = target;
    result.options.capacity = capacity;
    return result;
  };
  const auto check = [](std::initializer_list<gs::TensorStageRequest> stages,
                        std::size_t expected) {
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

std::shared_ptr<const gs::Stage> cast_stage(std::size_t input_pads = 1) {
  const std::array inputs{rt::TensorSpec{{2}, rt::DType::Float32, simaai::kernels::Contiguous{}}};
  const std::array outputs{rt::TensorSpec{{2}, rt::DType::BFloat16, simaai::kernels::Contiguous{}}};
  auto checked = simaai::kernels::ops::cast::make_executable({inputs, outputs, {}},
                                                             simaai::kernels::Target::A65);
  require(std::holds_alternative<rt::KernelExecutable>(checked), "real Cast admission failed");
  const std::array executables{std::get<rt::KernelExecutable>(std::move(checked))};
  auto result = rt::prepare_kernels(executables);
  require(std::holds_alternative<rt::KernelSession>(result), "real Cast preparation failed");
  auto session = std::get<rt::KernelSession>(std::move(result));
  auto stage = std::make_shared<gs::Stage>();
  stage->input_pads = input_pads;
  stage->output_caps = {gst_caps_from_string("application/x-sima-tensor"), gst_caps_unref};
  gs::TensorBridgeOptions bridge;
  for (std::size_t pad = 0; pad < input_pads; ++pad) {
    stage->members.push_back({session.executor(0), session.operation(0), {}});
    bridge.input_caps.push_back(stage->output_caps);
    bridge.members.push_back({{{pad, 0}}, {pad}, {}});
  }
  // This test validates cold integration only; no numerical work is submitted.
  stage->bridge = gs::make_tensor_bridge(std::move(bridge));
  return stage;
}
void memory_coherency_candidate_policy() {
  namespace policy = simaai::neat::pipeline_internal;
  const auto check = [](const char* launch, bool participant) {
    GError* error = nullptr;
    auto* pipeline = gst_parse_launch(launch, &error);
    require(pipeline && !error, "coherency transport fixture could not be parsed");
    require(policy::graph_memory_coherency_mode(pipeline) == policy::MemoryCoherencyMode::Legacy,
            "candidate inherited an unadmitted memory mode");
    require(policy::graph_participates_in_demand_coherency(pipeline) == participant,
            "actual graph participation gate changed");
    require(!integration::prepare_and_attach_kernel_runners(pipeline, {}),
            "transport-only cohort rejected");
    const auto expected = participant && policy::demand_coherency_available()
        ? policy::MemoryCoherencyMode::DemandLocalV1 : policy::MemoryCoherencyMode::Legacy;
    require(policy::graph_memory_coherency_mode(pipeline) == expected,
            "whole-bin participation selection differs from explicit build mode");
    GstState state = GST_STATE_VOID_PENDING;
    gst_element_get_state(pipeline, &state, nullptr, 0);
    require(state == GST_STATE_NULL, "coherency admission started pipeline execution");
    gst_object_unref(pipeline);
  };
  check("appsrc ! queue ! appsink", true);
  check("appsrc ! tee name=t t. ! queue ! appsink t. ! queue ! appsink", true);
  check("appsrc ! identity signal-handoffs=false ! appsink", true);
  check("appsrc ! identity signal-handoffs=true ! appsink", false);
  check("appsrc ! fakesink", false); // Unknown callback/writer ownership: entirely Legacy.
  require(!policy::participates_in_demand_coherency("neatdecoder") &&
              !policy::participates_in_demand_coherency("neatboxrender") &&
              !policy::participates_in_demand_coherency("custom-writer"),
          "opaque consumers acquired demand policy");
}
// Exercise Core's real request -> installed Scheduler/Runner path, not an
// empty request vector or the name of a helper in a source file. The MLA
// producer remains stopped: this tests policy commit, not accelerator jobs.
void real_request_memory_policy_transaction(bool opaque_consumer) {
  namespace policy = simaai::neat::pipeline_internal;
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
  // Non-default producer flags make partial policy mutation observable.
  g_object_set(mla, "memory-coherency-mode", "legacy", "defer-output-invalidate", FALSE,
               "output-read-handoff", TRUE, nullptr);
  const auto check_policy = [&](bool committed) {
    gchar* mode = nullptr;
    gboolean deferred = FALSE, handoff = FALSE;
    g_object_get(mla, "memory-coherency-mode", &mode, "defer-output-invalidate", &deferred,
                 "output-read-handoff", &handoff, nullptr);
    const bool matches = g_strcmp0(mode, committed ? "demand-local-v1" : "legacy") == 0 &&
                         deferred == committed && handoff == !committed;
    g_free(mode);
    require(matches, "MLA flags were partially committed or selected the wrong graph mode");
    require(policy::graph_memory_coherency_mode(pipeline.get()) ==
                (committed ? policy::MemoryCoherencyMode::DemandLocalV1
                           : policy::MemoryCoherencyMode::Legacy),
            "Core ingress mode disagrees with the committed producer mode");
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
    stage.frame_arena_provenance = integration::static_contract::ArenaAllocationProvenance::CoreAllocated;
    stage.frame_arena_required_device_access =
        static_cast<std::uint32_t>(integration::static_contract::ArenaDeviceAccess::CpuA65);
    stage.logical_inputs.push_back(specs::build_logical_input_static_spec(
        0, 0, 0, {2}, "FP32", "C", "input", "input", "input"));
    stage.input_bindings.push_back(
        specs::build_input_binding_static_spec(0, 0, "input", "input"));
    auto recipe = std::make_shared<integration::KernelStageRecipe>();
    const std::array inputs{rt::TensorSpec{{2}, rt::DType::Float32}};
    const std::array outputs{rt::TensorSpec{{2}, rt::DType::BFloat16}};
    auto executable = simaai::kernels::ops::cast::make_executable(
        {inputs, outputs, {}}, simaai::kernels::Target::A65);
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
    request.options.storage = gs::make_storage_with_memory_v1(gs::MemoryCoherencyMode::Legacy);
    const std::array first_only{request};
    auto prepared = gs::prepare_tensor_stages_with_memory_v1(
        first_only, {}, gs::MemoryCoherencyMode::Legacy);
    if (const auto* failure = std::get_if<rt::Error>(&prepared))
      throw std::runtime_error("valid first request rejected: " + failure->field + ": " +
                               failure->message);
    const auto& stages =
        std::get<std::vector<std::shared_ptr<const gs::Stage>>>(prepared);
    require(stages.size() == 1 && stages[0] && stages[0]->members.size() == 1 &&
                stages[0]->input_pads == 1 && stages[0]->bridge,
            "valid first request did not retain its canonical prepared stage");
  }
  check_policy(false);
  auto malformed = manifest.stages[1];
  auto malformed_recipe = std::make_shared<integration::KernelStageRecipe>(*malformed.native_kernel);
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
    auto* context = gst_element_get_context(element, gs::context_type);
    if (context) gst_context_unref(context);
    require(!context, "a rejected cohort partially attached a prepared stage");
  }
  check_policy(false);
  requests[1].native_stage = &manifest.stages[1];
  const auto accepted = integration::prepare_and_attach_kernel_runners(pipeline.get(), requests);
  if (accepted)
    throw std::runtime_error("real cohort rejected: " + accepted->field + ": " + accepted->message);
  for (auto* element : {first, second}) {
    auto* context = gst_element_get_context(element, gs::context_type);
    auto stage = context ? gs::stage_from_context(context) : nullptr;
    if (context) gst_context_unref(context);
    require(stage && stage->members.size() == 1 && stage->input_pads == 1 && stage->bridge,
            "real canonical request was not prepared and retained by its element");
  }
  check_policy(policy::demand_coherency_available() && !opaque_consumer);
}

void empty_candidate_bindings() {
  auto* pipeline = gst_pipeline_new("empty-candidate");
  require(pipeline, "candidate pipeline is unavailable");
  require(!integration::attach_kernel_runner_stages(pipeline, {}),
          "plain candidate bin rejected without kernel bindings");
  require(!integration::prepare_and_attach_kernel_runners(pipeline, {}),
          "plain candidate bin required numerical preparation");

  auto* runner = gst_element_factory_make("neatkernelrunner", "unbound");
  require(runner && gst_bin_add(GST_BIN(pipeline), runner), "candidate runner is unavailable");
  require(g_object_class_find_property(G_OBJECT_GET_CLASS(runner), "config") == nullptr,
          "kernel runner must not expose a legacy standalone JSON config property");
  require(bool(integration::attach_kernel_runner_stages(pipeline, {})),
          "empty bindings bypassed unbound runner validation");
  require(bool(integration::prepare_and_attach_kernel_runners(pipeline, {})),
          "empty preparation requests bypassed unbound runner validation");
  require(!gst_element_get_context(runner, gs::context_type),
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
  const std::array bindings{integration::KernelRunnerStageBinding{runner, cast_stage(2)}};
  // Request order is not port identity. Both sink_0 and sink_1 are required,
  // even when sink_1 is created first.
  auto* second = gst_element_request_pad_simple(runner, "sink_1");
  require(second, "second input pad request failed");
  require(bool(integration::attach_kernel_runner_stages(pipeline, bindings)),
          "multi-input candidate accepted a missing first pad");
  require(!gst_element_get_context(runner, gs::context_type),
          "missing input partly attached a context");
  auto* first = gst_element_request_pad_simple(runner, "sink_0");
  auto* extra = gst_element_request_pad_simple(runner, "sink_2");
  require(first && extra, "input pad requests failed");
  require(bool(integration::attach_kernel_runner_stages(pipeline, bindings)),
          "multi-input candidate accepted an undeclared input pad");
  require(!gst_element_get_context(runner, gs::context_type),
          "undeclared input partly attached a context");
  gst_element_release_request_pad(runner, extra);
  gst_object_unref(extra);
  require(!integration::attach_kernel_runner_stages(pipeline, bindings),
          "complete multi-input candidate rejected out-of-order pad creation");
  auto* context = gst_element_get_context(runner, gs::context_type);
  require(context && gs::stage_from_context(context) == bindings[0].stage,
          "multi-input candidate retained the wrong stage");
  gst_context_unref(context);
  gst_object_unref(first);
  gst_object_unref(second);
  gst_object_unref(pipeline);
}
void candidate_attachment() {
  auto* pipeline = gst_pipeline_new("candidate");
  auto* first = gst_element_factory_make("neatkernelrunner", "first");
  auto* second = gst_element_factory_make("neatkernelrunner", "second");
  require(pipeline && first && second, "candidate factory is unavailable");
  gst_bin_add_many(GST_BIN(pipeline), first, second, nullptr);
  auto* first_pad = gst_element_request_pad_simple(first, "sink_%u");
  auto* second_pad = gst_element_request_pad_simple(second, "sink_%u");
  auto stage = cast_stage();
  std::weak_ptr<const gs::Stage> retained = stage;
  std::array bindings{integration::KernelRunnerStageBinding{first, stage},
                      integration::KernelRunnerStageBinding{second, stage}};
  bindings[1].element = first;
  require(bool(integration::attach_kernel_runner_stages(pipeline, bindings)),
          "duplicate attachment accepted");
  require(!gst_element_get_context(first, gs::context_type),
          "failed validation partly attached a context");
  bindings[1].element = second;
  require(bool(integration::attach_kernel_runner_stages(pipeline, std::span{bindings}.first(1))),
          "unbound candidate element accepted");
  require(!gst_element_get_context(first, gs::context_type), "whole-bin preflight was not atomic");
  gst_element_release_request_pad(second, second_pad);
  gst_object_unref(second_pad);
  require(bool(integration::attach_kernel_runner_stages(pipeline, bindings)),
          "missing required pad accepted");
  second_pad = gst_element_request_pad_simple(second, "sink_%u");
  require(!integration::attach_kernel_runner_stages(pipeline, bindings),
          "candidate attachment failed");
  for (auto* element : {first, second}) {
    auto* context = gst_element_get_context(element, gs::context_type);
    require(context && gs::stage_from_context(context) == stage,
            "element retained the wrong Runtime stage");
    gst_context_unref(context);
  }
  bindings = {};
  stage.reset();
  require(!retained.expired(), "element context failed to retain Runtime stage");
  gst_object_unref(first_pad);
  gst_object_unref(second_pad);
  gst_object_unref(pipeline);
  require(retained.expired(), "candidate Runtime stage leaked after destruction");
}
void immutable_recipe_instances() {
  // Factory observation/failure controls only; no numerical executable or
  // transport is admitted by this cold ownership test.
  integration::KernelStageRecipe recipe;
  recipe.members.resize(2);
  unsigned constructions = 0;
  bool fail = true;
  std::vector<std::weak_ptr<std::byte>> instances;
  recipe.resolver_factories.push_back(
      {0, [&] {
         ++constructions;
         auto state = std::make_shared<std::byte>();
         instances.push_back(state);
         return integration::KernelStageRecipe::Resolver{
             [state](const rt::Executor&, const std::shared_ptr<const rt::PreparedOperation>& base,
                     std::span<GstBuffer* const>)
                 -> rt::Result<std::shared_ptr<const rt::PreparedOperation>> { return base; }};
       }});
  recipe.resolver_factories.push_back(
      {1, [&]() -> integration::KernelStageRecipe::Resolver {
         if (fail)
           throw std::runtime_error("injected cold instance failure");
         return [](const rt::Executor&, const std::shared_ptr<const rt::PreparedOperation>& base,
                   std::span<GstBuffer* const>)
                    -> rt::Result<std::shared_ptr<const rt::PreparedOperation>> { return base; };
       }});
  gs::TensorStageRequest request;
  request.members = recipe.members;
  bool threw = false;
  try {
    (void)integration::instantiate_kernel_stage_state(recipe, request);
  } catch (const std::runtime_error& error) {
    threw = std::string_view(error.what()) == "injected cold instance failure";
  }
  require(threw && !request.members[0].resolve && !request.members[1].resolve &&
              instances[0].expired(),
          "factory exception was reclassified or committed/retained partial instance state");
  fail = false;
  require(!integration::instantiate_kernel_stage_state(recipe, request) && constructions == 2,
          "fresh retry failed to instantiate exactly once");
  auto copy = request;
  require(bool(integration::instantiate_kernel_stage_state(recipe, copy)) && constructions == 2,
          "a copied admission was silently reset or instantiated twice");
  auto independent = gs::TensorStageRequest{};
  independent.members = recipe.members;
  require(!integration::instantiate_kernel_stage_state(recipe, independent) && constructions == 3,
          "independent request did not instantiate new state");
  const auto source_recipe = recipe;
  request = {};
  require(!instances[1].expired(), "request copy did not retain its own admission state");
  copy = {};
  require(instances[1].expired() && !instances[2].expired(),
          "reusable recipe retained live state or releasing A invalidated B");
  independent = {};
  require(instances[2].expired() && !source_recipe.members[0].resolve,
          "copied reusable recipe acquired execution state");
  auto invalid = recipe;
  invalid.resolver_factories.push_back(invalid.resolver_factories.front());
  gs::TensorStageRequest rejected;
  rejected.members = invalid.members;
  require(bool(integration::instantiate_kernel_stage_state(invalid, rejected)) &&
              constructions == 3,
          "invalid factory list constructed state before validation");
  integration::KernelStageRecipe image;
  image.members.resize(1);
  image.frame_adapter_factory = []() -> rt::Result<std::shared_ptr<gs::TensorStageFrameAdapter>> {
    return rt::Error{rt::ErrorCode::ResourceBusy, "test.factory", "declared construction failure"};
  };
  gs::TensorStageRequest image_request;
  image_request.members = image.members;
  const auto declared = integration::instantiate_kernel_stage_state(image, image_request);
  require(declared && declared->code == rt::ErrorCode::ResourceBusy &&
              declared->field == "test.factory" &&
              declared->message == "declared construction failure" && !image_request.frame_adapter,
          "declared factory failure was reclassified or committed partial state");
  std::cout << "immutable recipe: independent admission, retry and ownership PASS\n";
}

void boxdecode_geometry_variants() {
  namespace k = simaai::kernels;
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
  auto checked =
      simaai::scheduler::check_kernel({"boxdecode", k::Target::A65, inputs, outputs, parameters});
  require(std::holds_alternative<k::Executable>(checked), "BoxDecode geometry fixture rejected");
  rt::Executor executor(rt::Hardware::A65);
  auto admitted = executor.prepare(std::get<k::Executable>(std::move(checked)));
  require(std::holds_alternative<rt::PreparedOperation>(admitted),
          "BoxDecode geometry admission failed");
  auto base = std::make_shared<const rt::PreparedOperation>(
      std::get<rt::PreparedOperation>(std::move(admitted)));
  integration::KernelStageRecipe recipe;
  recipe.members.resize(1);
  recipe.members.front().executable = base->executable();
  integration::attach_boxdecode_geometry_resolver(
      recipe, 0, {8, 8, 8, 8, simaai::neat::ResizeMode::Letterbox}, true, true);
  const auto instantiate = [&] {
    gs::TensorStageRequest request;
    request.members = recipe.members;
    require(!integration::instantiate_kernel_stage_state(recipe, request),
            "fresh checked BoxDecode instance failed");
    return request;
  };
  auto instance = instantiate();
  auto& member = instance.members.front();
  require(!recipe.members.front().resolve && recipe.resolver_factories.size() == 1,
          "reusable recipe acquired a live resolver");
  require(bool(integration::instantiate_kernel_stage_state(recipe, instance)),
          "active resolver instance was reset by a second instantiation");
  if (!gst_meta_get_info("GstSimaMeta")) {
    const gchar* tags[] = {nullptr};
    require(gst_meta_register_custom("GstSimaMeta", tags, nullptr, nullptr, nullptr),
            "register preprocess fixture metadata");
  }
  gs::Buffer first(gst_buffer_new()), second(gst_buffer_new());
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
  auto selected = member.resolve(executor, base, buffers);
  require(std::holds_alternative<std::shared_ptr<const rt::PreparedOperation>>(selected),
          "checked geometry variant rejected");
  const auto variant = std::get<std::shared_ptr<const rt::PreparedOperation>>(selected);
  require(variant != base &&
              variant->executable().state().contract == base->executable().state().contract &&
              variant->executable().state().resource == base->executable().state().resource,
          "geometry rebuilt tensor contract or numerical scratch");
  auto repeated = member.resolve(executor, base, buffers);
  require(std::get<std::shared_ptr<const rt::PreparedOperation>>(repeated) == variant,
          "warm geometry did not reuse immutable variant");
  // Actual cold factory path: one reusable recipe must not share caches even
  // when preparations use the same Runtime domain or execute concurrently.
  auto sibling = instantiate();
  const auto sibling_base = std::make_shared<const rt::PreparedOperation>(
      std::get<rt::PreparedOperation>(executor.prepare(member.executable)));
  const auto sibling_variant = std::get<std::shared_ptr<const rt::PreparedOperation>>(
      sibling.members.front().resolve(executor, sibling_base, buffers));
  require(sibling_variant != variant && std::get<std::shared_ptr<const rt::PreparedOperation>>(
                                            member.resolve(executor, base, buffers)) == variant,
          "A/B/A in one Runtime domain discarded A's warmed variant");
  rt::Executor independent_executor(rt::Hardware::A65);
  const auto independent_base = std::make_shared<const rt::PreparedOperation>(
      std::get<rt::PreparedOperation>(independent_executor.prepare(member.executable)));
  auto independent = instantiate();
  const auto independent_variant = std::get<std::shared_ptr<const rt::PreparedOperation>>(
      independent.members.front().resolve(independent_executor, independent_base, buffers));
  std::atomic<bool> isolated{true};
  const auto warm = [&](gs::TensorStageRequest& request, const rt::Executor& exec,
                        const std::shared_ptr<const rt::PreparedOperation>& original,
                        const std::shared_ptr<const rt::PreparedOperation>& expected) {
    for (unsigned i = 0; i < 64; ++i) {
      auto result = request.members.front().resolve(exec, original, buffers);
      const auto* value = std::get_if<std::shared_ptr<const rt::PreparedOperation>>(&result);
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
  require(std::get<std::shared_ptr<const rt::PreparedOperation>>(
              copied_admission.members.front().resolve(executor, base, buffers)) == variant,
          "copies within one admission unexpectedly cloned live state");
  copied_admission = {};
  geometry(other, 32, "letterbox");
  require(std::holds_alternative<rt::Error>(
              independent.members.front().resolve(independent_executor, independent_base, buffers)),
          "independent instance accepted disagreeing split-head geometry");
  geometry(other, 16, "letterbox");
  require(std::get<std::shared_ptr<const rt::PreparedOperation>>(
              member.resolve(executor, base, buffers)) == variant &&
              std::get<std::shared_ptr<const rt::PreparedOperation>>(
                  independent.members.front().resolve(independent_executor, independent_base,
                                                      buffers)) == independent_variant,
          "failed B frame evicted A or B's warmed immutable variant");
  sibling = {};
  independent = {};
  require(!recipe.members.front().resolve,
          "reusable recipe acquired execution state through its factory");
  for (const int width : {24, 40, 48, 56, 64}) {
    geometry(meta, width, "letterbox");
    geometry(other, width, "letterbox");
    auto evicting = member.resolve(executor, base, buffers);
    require(std::holds_alternative<std::shared_ptr<const rt::PreparedOperation>>(evicting),
            "bounded geometry cache rejected a valid replacement");
    const auto& operation = std::get<std::shared_ptr<const rt::PreparedOperation>>(evicting);
    require(operation->executable().state().contract == base->executable().state().contract &&
                operation->executable().state().resource == base->executable().state().resource,
            "geometry eviction changed checked storage or scratch authority");
  }
  geometry(meta, 16, "letterbox");
  geometry(other, 16, "letterbox");
  const auto rebuilt = std::get<std::shared_ptr<const rt::PreparedOperation>>(
      member.resolve(executor, base, buffers));
  require(rebuilt != variant &&
              variant->executable().state().contract == base->executable().state().contract &&
              variant->executable().state().resource == base->executable().state().resource,
          "eviction invalidated a retained geometry snapshot or grew the bounded cache");
  rt::Executor other_executor(rt::Hardware::A65);
  auto other_admitted = other_executor.prepare(member.executable);
  require(std::holds_alternative<rt::PreparedOperation>(other_admitted),
          "second geometry execution domain failed admission");
  const auto other_base = std::make_shared<const rt::PreparedOperation>(
      std::get<rt::PreparedOperation>(std::move(other_admitted)));
  const auto other_variant = std::get<std::shared_ptr<const rt::PreparedOperation>>(
      member.resolve(other_executor, other_base, buffers));
  require(other_variant != rebuilt && other_variant != variant && other_variant != other_base,
          "geometry cache reused another execution domain's admitted operation");
  const auto returned = std::get<std::shared_ptr<const rt::PreparedOperation>>(
      member.resolve(executor, base, buffers));
  require(returned != other_variant,
          "geometry cache retained the wrong execution domain after returning to the first base");
  geometry(other, 32, "letterbox");
  require(std::holds_alternative<rt::Error>(member.resolve(executor, base, buffers)),
          "inconsistent split-head geometry accepted");
  geometry(meta, 32, "letterbox");
  auto changed = member.resolve(executor, base, buffers);
  require(std::holds_alternative<std::shared_ptr<const rt::PreparedOperation>>(changed) &&
              std::get<std::shared_ptr<const rt::PreparedOperation>>(changed) != variant,
          "new image size mutated a retained variant");
  geometry(meta, 8, "stretch");
  geometry(other, 8, "stretch");
  require(std::holds_alternative<rt::Error>(member.resolve(executor, base, buffers)),
          "changed resize contract accepted");
  geometry(meta, 8, "letterbox");
  geometry(other, 8, "letterbox");
  require(std::get<std::shared_ptr<const rt::PreparedOperation>>(
              member.resolve(executor, base, buffers)) == base,
          "base geometry not reused");
  gst_structure_remove_field(meta, "preproc_original_width");
  require(std::holds_alternative<rt::Error>(member.resolve(executor, base, buffers)),
          "partial image-size metadata accepted");
  integration::KernelStageRecipe explicit_recipe;
  explicit_recipe.members = recipe.members;
  integration::attach_boxdecode_geometry_resolver(
      explicit_recipe, 0, {8, 8, 8, 8, simaai::neat::ResizeMode::Letterbox}, false, true);
  gs::TensorStageRequest explicit_request;
  explicit_request.members = explicit_recipe.members;
  require(!integration::instantiate_kernel_stage_state(explicit_recipe, explicit_request),
          "explicit-size instance failed");
  auto& explicit_size = explicit_request.members.front();
  geometry(meta, 16, "letterbox");
  geometry(other, 32, "letterbox");
  require(std::get<std::shared_ptr<const rt::PreparedOperation>>(
              explicit_size.resolve(executor, base, buffers)) == base,
          "explicit original dimensions lost precedence over conflicting frame dimensions");
  geometry(other, 32, "stretch");
  require(std::holds_alternative<rt::Error>(explicit_size.resolve(executor, base, buffers)),
          "explicit image size bypassed resize provenance validation");
  gs::Buffer bare(gst_buffer_new_allocate(nullptr, 4, nullptr));
  std::string bare_error;
  const bool bare_attached = simaai::gst::tensor_buffer_attach_frame_metadata(
      bare.get(), simaai::gst::TensorBufferFrameMetadata{}, &bare_error);
  require(!bare_attached &&
              bare_error.find("writable canonical compact envelope") != std::string::npos,
          "bare geometry carrier accepted frame facts without a static publication");
  const auto compact_owner = compact_geometry_publication();
  gs::Buffer compact_first(gst_buffer_new_allocate(nullptr, 4, nullptr));
  gs::Buffer compact_second(gst_buffer_new_allocate(nullptr, 4, nullptr));
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
  require(std::get<std::shared_ptr<const rt::PreparedOperation>>(
              explicit_size.resolve(executor, base, compact_buffers)) == base,
          "compact frame facts overrode explicit original dimensions");
  require(std::holds_alternative<rt::Error>(member.resolve(executor, base, compact_buffers)),
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
  namespace k = simaai::kernels;
  const auto& executable = compiled.recipe->members.front().executable.state();
  require(executable.contract->input_specs.size() == 2 &&
              executable.contract->input_requirements.size() == 2 &&
              executable.contract->output_requirements.size() == 4 && executable.resource &&
              executable.execute_host,
          "numerical fixture must retain its two checked heads and context");
  auto created = executable.resource->create(executable.resource->owner.get(), {});
  require(std::holds_alternative<std::shared_ptr<void>>(created), "numerical context rejected");
  const auto resource = std::get<std::shared_ptr<void>>(std::move(created));
  // Typed words guarantee eight-byte alignment; publish only each checked
  // byte capacity, not the allocator's rounded-up storage extent.
  std::vector<std::vector<std::uint64_t>> storage, results;
  std::vector<k::ConstHostBuffer> inputs;
  std::vector<k::HostBuffer> outputs;
  for (auto requirement : executable.contract->input_requirements) {
    require(requirement.alignment_bytes <= alignof(std::uint64_t),
            "fixture input alignment changed");
    storage.emplace_back((requirement.storage_bytes + 7) / 8);
  }
  for (auto requirement : executable.contract->output_requirements) {
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
                      executable.contract->input_requirements[i].storage_bytes});
  for (std::size_t i = 0; i < results.size(); ++i)
    outputs.push_back({reinterpret_cast<std::byte*>(results[i].data()),
                       executable.contract->output_requirements[i].storage_bytes});
  require(!executable.execute_host(executable.payload.get(), resource.get(), inputs, outputs),
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
  gst_init(&argc, &argv);
  try {
    immutable_recipe_instances();
    if (argc == 2 && std::string_view(argv[1]) == "--instance-only")
      return 0;
    require(gs::register_element(), "register native candidate plugin");
    output_carrier_budget();
    ev_service_capacity();
    memory_coherency_candidate_policy();
    real_request_memory_policy_transaction(false);
    real_request_memory_policy_transaction(true);
    empty_candidate_bindings();
    multi_input_candidate_pads();
    candidate_attachment();
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

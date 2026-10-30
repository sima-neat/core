#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StageMemoryPolicy.h"
#include "test_main.h"

#include <array>
#include <stdexcept>

namespace si = simaai::neat::pipeline_internal::sima;
namespace sc = si::static_contract;
namespace client = simaai::scheduler::client;

namespace {
si::StageStaticSpec stage(std::string name, client::Hardware target) {
  si::StageStaticSpec result;
  result.logical_stage_id = name;
  result.element_name = name;
  result.payload_kind = si::StagePayloadKind::ProcessCvu;
  result.processcvu.resolved_exec_backend = target == client::Hardware::EV74 ? "EV74" : "A65";
  si::PhysicalBufferStaticSpec physical;
  physical.physical_index = 0;
  physical.source_physical_index = 0;
  physical.size_bytes = 64;
  result.physical_outputs = {physical};
  si::LogicalTensorStaticSpec output;
  output.logical_index = 0;
  output.physical_index = 0;
  output.backend_output_index = 0;
  output.size_bytes = 64;
  result.logical_outputs = {output};
  return result;
}

void connect(si::StageStaticSpec& consumer, const std::string& producer) {
  si::LogicalInputStaticSpec input;
  input.logical_index = input.backend_input_index = 0;
  input.physical_index = 0;
  input.size_bytes = 64;
  consumer.logical_inputs = {input};
  auto physical = consumer.physical_outputs.front();
  physical.address_source = si::PhysicalAddressSource::FrameSpan;
  consumer.physical_inputs = {physical};
  si::InputBindingStaticSpec binding;
  binding.sink_pad_index = 0;
  binding.local_logical_input_index = 0;
  binding.src_stage_id = producer;
  binding.src_logical_output_index = 0;
  binding.src_physical_output_index = 0;
  binding.src_physical_size_bytes = 64;
  binding.required = true;
  consumer.input_bindings = {binding};
}

void check_independent_carriers(bool prefer_dms) {
  const std::array inputs{client::TensorSpec{{16}, client::DType::Float32}};
  const std::array outputs{client::TensorSpec{{16}, client::DType::BFloat16}};
  auto checked = client::check_kernel({"cast", client::Hardware::A65, inputs, outputs, {}});
  require(std::holds_alternative<client::KernelExecutable>(checked), "Cast admission failed");
  auto executable = std::get<client::KernelExecutable>(std::move(checked));
  auto recipe = std::make_shared<si::KernelStageRecipe>();
  recipe->members = {{executable, {{0, 0}}, {{0, 0}}, {}}, {executable, {{0, 0}}, {{1, 0}}, {}}};

  auto producer = stage("producer", client::Hardware::A65);
  producer.native_kernel = std::move(recipe);
  if (prefer_dms)
    producer.allocation_policy = client::AllocationPolicy::PreferDmsThenCma;
  producer.physical_outputs.front().size_bytes = 32;
  producer.logical_outputs.front().size_bytes = 32;
  auto second = producer.physical_outputs.front();
  second.physical_index = second.source_physical_index = 1;
  producer.physical_outputs.push_back(second);
  auto second_view = producer.logical_outputs.front();
  second_view.logical_index = second_view.physical_index = second_view.backend_output_index = 1;
  producer.logical_outputs.push_back(second_view);
  si::LogicalInputStaticSpec input;
  input.logical_index = input.physical_index = input.backend_input_index = 0;
  input.size_bytes = 64;
  producer.logical_inputs = {input};
  auto input_carrier = producer.physical_outputs.front();
  input_carrier.size_bytes = 64;
  input_carrier.address_source = si::PhysicalAddressSource::DirectCarrierSpan;
  producer.physical_inputs = {input_carrier};
  si::InputBindingStaticSpec ingress;
  ingress.sink_pad_index = 0;
  ingress.local_logical_input_index = 0;
  producer.input_bindings = {ingress};

  auto consumer = stage("ev", client::Hardware::EV74);
  connect(consumer, "producer");
  consumer.logical_inputs.front().backend_input_index = 0;
  consumer.logical_inputs.front().size_bytes = 32;
  consumer.physical_inputs.front().size_bytes = 32;
  consumer.input_bindings.front().src_physical_size_bytes = 32;
  si::SimaPluginStaticManifest manifest;
  manifest.stages = {std::move(producer), std::move(consumer)};
  si::finalize_pipeline_storage(manifest);

  const auto& layout = *manifest.stages.front().frame_layout;
  const auto& ev_output = layout.outputs.at(0).span;
  const auto& cpu_output = layout.outputs.at(1).span;
  const auto& ev_allocation = layout.allocations.at(ev_output.allocation_index);
  const auto& cpu_allocation = layout.allocations.at(cpu_output.allocation_index);
  require(ev_output.allocation_index != cpu_output.allocation_index && ev_output.byte_offset == 0 &&
              cpu_output.byte_offset == 0 &&
              ev_allocation.policy == client::AllocationPolicy::RequireCma &&
              (ev_allocation.required_device_access & client::kAccessEv74) &&
              cpu_allocation.policy == (prefer_dms ? client::AllocationPolicy::PreferDmsThenCma
                                                   : client::AllocationPolicy::RequireDms) &&
              !(cpu_allocation.required_device_access & client::kAccessEv74),
          "an EV74 consumer must affect only its selected producer carrier");
}

si::StageStaticSpec checked_tensor_stage() {
  const std::array inputs{client::TensorSpec{{16}, client::DType::Float32}};
  const std::array outputs{client::TensorSpec{{16}, client::DType::BFloat16}};
  auto checked = client::check_kernel({"cast", client::Hardware::A65, inputs, outputs, {}});
  require(std::holds_alternative<client::KernelExecutable>(checked), "Cast admission failed");
  auto recipe = std::make_shared<si::KernelStageRecipe>();
  recipe->members = {
      {std::get<client::KernelExecutable>(std::move(checked)), {{0, 0}}, {{0, 0}}, {}}};
  auto result = stage("tensor", client::Hardware::A65);
  result.native_kernel = std::move(recipe);
  result.physical_outputs.front().size_bytes = 32;
  result.logical_outputs.front().size_bytes = 32;
  connect(result, "");
  result.physical_inputs.front().size_bytes = 64;
  result.physical_inputs.front().address_source = si::PhysicalAddressSource::DirectCarrierSpan;
  result.input_bindings.front().cm_input_name = "public_tensor";
  return result;
}

void check_public_tensor_routes() {
  const auto prepare = [](si::StageStaticSpec input) {
    si::SimaPluginStaticManifest manifest;
    manifest.stages.push_back(std::move(input));
    si::finalize_pipeline_storage(manifest);
    return manifest.stages.front();
  };
  const auto expect_direct = [&](si::StageStaticSpec input, const char* message) {
    const auto prepared = prepare(std::move(input));
    require(
        std::holds_alternative<client::DirectMemory>(prepared.frame_layout->inputs.at(0).source),
        message);
  };

  auto external = checked_tensor_stage();
  // Public binding ordinals are not stage-local physical/memory ordinals.
  external.input_bindings.front().sink_pad_index = 2;
  external.input_bindings.front().src_physical_output_index = 3;
  external.logical_inputs.front().byte_offset = 4;
  external.physical_inputs.front().size_bytes = 68;
  const auto prepared = prepare(external);
  const auto& input = prepared.frame_layout->inputs.at(0);
  const auto* binding = std::get_if<client::PhysicalBinding>(&input.source);
  require(binding && binding->pad == 2 && binding->binding_index == 3 &&
              binding->byte_offset == 0 && binding->size_bytes == 68 && input.view_offset == 4,
          "public tensor must retain its pad, binding ordinal and relative logical view");
  require(prepared.physical_inputs.front().address_source ==
              si::PhysicalAddressSource::DirectCarrierSpan,
          "final placement must not rewrite earlier provenance contracts");

  auto consumer = checked_tensor_stage();
  consumer.input_bindings.front().src_stage_id = "producer";
  si::SimaPluginStaticManifest internal;
  internal.stages = {stage("producer", client::Hardware::A65), std::move(consumer)};
  si::finalize_pipeline_storage(internal);
  const auto* known = std::get_if<client::KnownFrameSpan>(
      &internal.stages.back().frame_layout->inputs.at(0).source);
  require(known && known->pad == 0 && known->span.size_bytes == 64,
          "an exact internal producer must keep its known allocation span");

  // The renderer sets this legacy flag even for standalone ProcessCVU nodes.
  auto standalone = checked_tensor_stage();
  standalone.model_managed_stage = true;
  require(std::holds_alternative<client::PhysicalBinding>(
              prepare(standalone).frame_layout->inputs.at(0).source),
          "the legacy model-managed flag is not model storage provenance");
  auto model = checked_tensor_stage();
  model.physical_inputs.front().model_carrier_id = 17;
  expect_direct(std::move(model), "model-derived carriers must keep their authored direct binding");
  auto nonnative = checked_tensor_stage();
  nonnative.native_kernel.reset();
  expect_direct(std::move(nonnative), "non-native direct ingress must not require tensor metadata");
  auto other_payload = checked_tensor_stage();
  other_payload.payload_kind = si::StagePayloadKind::None;
  expect_direct(std::move(other_payload), "other native payloads must keep their binding contract");

  auto imported = checked_tensor_stage();
  auto pinned = std::make_shared<client::FrameLayout>();
  pinned->ownership = client::FrameOwnership::ReuseInput;
  pinned->allocations = {{256, client::kAccessCpuA65, client::AllocationPolicy::PreserveImported}};
  pinned->outputs = {{0, {0, 64, 32}}};
  pinned->inputs = {{client::KnownFrameSpan{0, {0, 128, 64}}, 0}};
  imported.frame_layout = pinned;
  const auto retained = prepare(std::move(imported));
  const auto* preserved =
      std::get_if<client::KnownFrameSpan>(&retained.frame_layout->inputs.at(0).source);
  require(preserved && preserved->span.allocation_index == 0 &&
              preserved->span.byte_offset == 128 && preserved->span.size_bytes == 64 &&
              retained.frame_layout->ownership == client::FrameOwnership::ReuseInput,
          "pinned imported views must keep their exact source span");

  auto image = checked_tensor_stage();
  auto image_recipe = std::make_shared<si::KernelStageRecipe>(*image.native_kernel);
  bool adapter_called = false;
  image_recipe->frame_adapter_factory =
      [&]() -> client::Result<std::shared_ptr<client::TensorStageFrameAdapter>> {
    adapter_called = true;
    return client::Error{client::ErrorCode::InvalidRequest, "test", "not an execution test"};
  };
  image.native_kernel = std::move(image_recipe);
  const auto adapted = prepare(std::move(image));
  require(adapted.frame_layout->inputs.empty() && !adapter_called,
          "image adapters must retain authority over frame-dependent input binding");

  auto missing_producer = checked_tensor_stage();
  missing_producer.input_bindings.front().src_stage_id = "missing";
  bool rejected = false;
  try {
    (void)prepare(std::move(missing_producer));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "an invalid explicit producer must not become public ingress");
}

void check_memory_placement() {
  check_public_tensor_routes();
  check_independent_carriers(false);
  check_independent_carriers(true);
  si::SimaPluginStaticManifest fanout;
  fanout.stages = {stage("producer", client::Hardware::A65), stage("cpu", client::Hardware::A65),
                   stage("ev", client::Hardware::EV74)};
  connect(fanout.stages[1], "producer");
  connect(fanout.stages[2], "producer");
  si::finalize_pipeline_storage(fanout);
  require(fanout.stages[0].frame_layout->allocations.front().policy ==
              simaai::scheduler::client::AllocationPolicy::RequireCma,
          "an EV74 sibling must make the shared allocation CMA");

  si::SimaPluginStaticManifest aliases;
  aliases.stages = {stage("producer", client::Hardware::A65), stage("view", client::Hardware::A65),
                    stage("ev", client::Hardware::EV74)};
  connect(aliases.stages[1], "producer");
  aliases.stages[1].payload_kind = si::StagePayloadKind::None;
  aliases.stages[1].reuse_input_pad = 0;
  connect(aliases.stages[2], "view");
  si::finalize_pipeline_storage(aliases);
  require(aliases.stages[0].frame_layout->allocations.front().policy ==
                  simaai::scheduler::client::AllocationPolicy::RequireCma &&
              aliases.stages[1].frame_layout->ownership ==
                  simaai::scheduler::client::FrameOwnership::ReuseInput &&
              aliases.stages[1].frame_layout->allocations.front().policy ==
                  simaai::scheduler::client::AllocationPolicy::RequireCma,
          "views must propagate the EV74 requirement to the original allocation");

  connect(fanout.stages[0], "");
  auto& imported = fanout.stages[0].input_bindings.front();
  imported.src_logical_output_index = imported.src_physical_output_index = -1;
  imported.src_physical_size_bytes = 0;
  auto pinned = std::make_shared<client::FrameLayout>();
  pinned->ownership = client::FrameOwnership::ReuseInput;
  pinned->allocations = {{256, client::kAccessCpuA65, client::AllocationPolicy::PreserveImported},
                         {256, client::kAccessCpuA65, client::AllocationPolicy::PreserveImported}};
  pinned->outputs = {{0, {1, 64, 64}}};
  pinned->inputs = {{client::KnownFrameSpan{0, {1, 64, 64}}, 0}};
  fanout.stages[0].frame_layout = pinned;
  si::finalize_pipeline_storage(fanout);
  const auto& adopted = *fanout.stages[0].frame_layout;
  require(adopted.ownership == client::FrameOwnership::ReuseInput &&
              adopted.allocations.size() == 2 &&
              adopted.allocations[1].policy == client::AllocationPolicy::PreserveImported &&
              adopted.outputs.front().span.allocation_index == 1 &&
              adopted.outputs.front().span.byte_offset == 64 &&
              adopted.outputs.front().span.size_bytes == 64,
          "imported allocation identity and view origin must not be relabelled or relocated");
  si::SimaPluginStaticManifest public_input;
  public_input.stages = {stage("first", client::Hardware::A65),
                         stage("later", client::Hardware::A65)};
  connect(public_input.stages[0], "");
  public_input.stages[0].input_bindings[0].cm_input_name = "reused_name";
  public_input.stages[1].logical_outputs[0].logical_name = "reused_name";
  public_input.stages[0].physical_inputs[0].address_source =
      si::PhysicalAddressSource::DirectCarrierSpan;
  si::finalize_pipeline_storage(public_input);
  require(std::holds_alternative<simaai::scheduler::client::DirectMemory>(
              public_input.stages[0].frame_layout->inputs.front().source),
          "a later output name must not turn public ingress into a forward edge");
}
} // namespace

RUN_TEST("unit_stage_memory_policy_test", check_memory_placement)

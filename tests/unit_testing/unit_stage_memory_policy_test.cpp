#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StageMemoryPolicy.h"
#include "test_main.h"

#include <array>

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

void check_memory_placement() {
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

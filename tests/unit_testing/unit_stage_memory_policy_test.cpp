#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/StageMemoryPolicy.h"
#include "pipeline/internal/sima/static_contract/PhysicalExecutionPlan.h"
#include "test_main.h"

namespace si = simaai::neat::pipeline_internal::sima;
namespace sc = si::static_contract;

namespace {
si::StageStaticSpec stage(std::string name, sc::PhysicalEngine engine) {
  si::StageStaticSpec result;
  result.logical_stage_id = name;
  result.element_name = name;
  result.payload_kind = engine == sc::PhysicalEngine::Mla ? si::StagePayloadKind::ProcessMla
                                                          : si::StagePayloadKind::ProcessCvu;
  result.processcvu.resolved_exec_backend = engine == sc::PhysicalEngine::Cvu ? "EV74" : "A65";
  result.frame_arena_role = si::FrameArenaRole::Allocate;
  result.frame_arena_size_bytes = 4096;
  result.frame_arena_provenance = sc::ArenaAllocationProvenance::CoreAllocated;
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
  input.logical_index = 0;
  input.physical_index = 0;
  input.size_bytes = 64;
  consumer.logical_inputs = {input};
  auto physical = consumer.physical_outputs.front();
  physical.address_source = si::PhysicalAddressSource::FrameArenaSpan;
  consumer.physical_inputs = {physical};
  si::InputBindingStaticSpec binding;
  binding.local_logical_input_index = 0;
  binding.src_stage_id = producer;
  binding.src_logical_output_index = 0;
  binding.src_physical_output_index = 0;
  binding.required = true;
  consumer.input_bindings = {binding};
}
} // namespace

namespace {
void check_memory_placement() {
  for (const auto producer :
       {sc::PhysicalEngine::A65, sc::PhysicalEngine::Mla, sc::PhysicalEngine::Cvu}) {
    for (const auto consumer :
         {sc::PhysicalEngine::A65, sc::PhysicalEngine::Mla, sc::PhysicalEngine::Cvu}) {
      si::SimaPluginStaticManifest manifest;
      manifest.stages = {stage("producer", producer), stage("consumer", consumer)};
      connect(manifest.stages[1], "producer");
      si::resolve_stage_memory_placement(manifest);
      require(manifest.stages[0].frame_arena_storage_domain ==
                  (producer == sc::PhysicalEngine::Cvu || consumer == sc::PhysicalEngine::Cvu
                       ? sc::ArenaStorageDomain::Cma
                       : sc::ArenaStorageDomain::Dms),
              "allocation must account for both producer and consumer");
      require(manifest.stages[0].frame_arena_allocation_policy ==
                  sc::ArenaAllocationPolicy::RequiredDomain,
              "boundary domain must not permit allocation fallback");
    }
  }
  si::SimaPluginStaticManifest fanout;
  fanout.stages = {stage("producer", sc::PhysicalEngine::A65),
                   stage("cpu", sc::PhysicalEngine::A65), stage("ev", sc::PhysicalEngine::Cvu)};
  connect(fanout.stages[1], "producer");
  connect(fanout.stages[2], "producer");
  si::resolve_stage_memory_placement(fanout);
  require(fanout.stages[0].frame_arena_storage_domain == sc::ArenaStorageDomain::Cma,
          "an EV74 sibling must make the shared allocation CMA");

  si::SimaPluginStaticManifest aliases;
  aliases.stages = {stage("producer", sc::PhysicalEngine::A65),
                    stage("view", sc::PhysicalEngine::A65), stage("ev", sc::PhysicalEngine::Cvu)};
  connect(aliases.stages[1], "producer");
  aliases.stages[1].frame_arena_role = si::FrameArenaRole::ReuseInput;
  connect(aliases.stages[2], "view");
  si::resolve_stage_memory_placement(aliases);
  require(aliases.stages[0].frame_arena_storage_domain == sc::ArenaStorageDomain::Cma &&
              aliases.stages[1].frame_arena_storage_domain == sc::ArenaStorageDomain::Cma,
          "views must propagate the EV74 requirement to the original allocation");

  fanout.stages[0].frame_arena_provenance = sc::ArenaAllocationProvenance::ExternalAdopted;
  fanout.stages[0].frame_arena_storage_domain = sc::ArenaStorageDomain::Unknown;
  si::resolve_stage_memory_placement(fanout);
  require(fanout.stages[0].frame_arena_storage_domain == sc::ArenaStorageDomain::Unknown,
          "imported DMA-BUF provenance must not be relabelled or relocated");
  si::SimaPluginStaticManifest public_input;
  public_input.stages = {stage("first", sc::PhysicalEngine::A65),
                         stage("later", sc::PhysicalEngine::A65)};
  connect(public_input.stages[0], "");
  public_input.stages[0].input_bindings[0].cm_input_name = "reused_name";
  public_input.stages[1].logical_outputs[0].logical_name = "reused_name";
  si::resolve_stage_memory_placement(public_input);
  require(public_input.stages[0].frame_arena_storage_domain == sc::ArenaStorageDomain::Dms,
          "a later output name must not turn public ingress into a forward edge");

  auto preferred = stage("preferred", sc::PhysicalEngine::A65);
  preferred.frame_arena_allocation_policy = sc::ArenaAllocationPolicy::PreferDmsThenCma;
  si::SimaPluginStaticManifest explicit_preference;
  explicit_preference.stages = {preferred};
  si::resolve_stage_memory_placement(explicit_preference);
  require(explicit_preference.stages[0].frame_arena_allocation_policy ==
              sc::ArenaAllocationPolicy::PreferDmsThenCma,
          "default placement must preserve an explicitly authored fallback policy");
  explicit_preference.stages.push_back(stage("ev_reader", sc::PhysicalEngine::Cvu));
  connect(explicit_preference.stages.back(), "preferred");
  bool rejected = false;
  try {
    si::resolve_stage_memory_placement(explicit_preference);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "an EV74 consumer must reject the DMS fallback opt-in");
}
} // namespace

RUN_TEST("unit_stage_memory_policy_test", check_memory_placement)

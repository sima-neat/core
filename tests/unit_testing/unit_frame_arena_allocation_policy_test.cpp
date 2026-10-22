#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/sima/CompiledProcessCvuContractQuery.h"
#include "pipeline/internal/sima/static_contract/DmabufPlanContractProjection.h"
#include "pipeline/internal/sima/stagesemantics/ProcessMlaStageSemantics.h"
#include "gst/SimaPluginStaticManifestAbi.h"

#include "test_main.h"

#include <array>

namespace sc = simaai::neat::pipeline_internal::sima::static_contract;
namespace sima = simaai::neat::pipeline_internal::sima;

RUN_TEST(
    "unit_frame_arena_allocation_policy_test", ([] {
      sc::ModelExecutionPlanData data;
      data.contract_version = "allocation-policy-test";
      data.values = {
          sc::ValueSpec{0U, "ifm", 64U, "uint8", sc::TensorShape{64}},
          sc::ValueSpec{1U, "mla_ofm", 64U, "uint8", sc::TensorShape{64}},
          sc::ValueSpec{2U, "host_ofm", 64U, "uint8", sc::TensorShape{64}},
      };
      data.model_inputs = {0U};
      sc::OpSpec mla;
      mla.id = 0;
      mla.sequence = 1;
      mla.name = "MLA_0";
      mla.kind = sc::OpKind::Mla;
      mla.processor = "MLA";
      mla.inputs = {0};
      mla.outputs = {1};
      mla.config = sc::MlaOpConfig{"model.elf", 4};
      sc::OpSpec host;
      host.id = 1;
      host.sequence = 2;
      host.name = "APU_0";
      host.kind = sc::OpKind::HostTvm;
      host.processor = "A65";
      host.inputs = {1};
      host.outputs = {2};
      host.config = sc::HostTvmOpConfig{"post.so",         {"mla_ofm"}, {{"uint8", {64}}},
                                        {{"uint8", {64}}}, {-1},        {}};
      data.ops = {mla, host};
      data.backend_ports = {
          {0U, sc::BackendPortDirection::Input, 0U, "data.ifm.b0", 0U, 64U, 64U,
           sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::ReadOnly},
          {0U, sc::BackendPortDirection::Output, 0U, "data.ofm.b0", 1U, 64U, 64U,
           sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::WriteOnly},
      };
      data.model_outputs = {{0U, "host_ofm", 2U}};
      std::string error;
      auto two_mla_data = data;
      const auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
      require(plan.has_value(), error);
      const auto physical = sc::PhysicalExecutionLowerer::lower(*plan, &error);
      require(physical.has_value(), error);
      const auto sources = sc::resolve_mla_input_physical_sources(*plan, 0U, {}, &error);
      require(sources.has_value(), error);
      for (const auto policy :
           {sc::ArenaDmsPolicy::PreferDmsForEligible, sc::ArenaDmsPolicy::PreferDmsThenCma}) {
        const auto arena = sc::FrameSlotArenaPlan::compile(
            *plan, *physical, sc::FrameSlotArenaReuse::DisjointLifetimes, 4096, &error, policy);
        require(arena.has_value(), error);
        sima::MlaStaticContract mla_contract;
        mla_contract.stage_id = "MLA_0";
        mla_contract.node_name = "MLA_0";
        mla_contract.model_path = "model.elf";
        mla_contract.batch_size = 4;
        mla_contract.batch_sz_model = 4;
        sima::PhysicalBufferStaticSpec input;
        input.physical_index = 0;
        input.size_bytes = 64;
        input.segment_name = "ifm";
        mla_contract.physical_inputs.push_back(input);
        sima::TensorStaticSpec logical;
        logical.tensor_index = 0;
        mla_contract.logical_inputs.push_back(logical);
        sima::PhysicalBufferStaticSpec output;
        output.physical_index = 0;
        output.size_bytes = 64;
        output.segment_name = "mla_ofm";
        mla_contract.dispatcher_physical_outputs.push_back(output);
        require(sc::apply_dmabuf_plan_contract_projection(*plan, 0U, *arena, &mla_contract,
                                                          *sources, &error),
                error);
        const auto compiled = sima::stagesemantics::build_mla_compiled_contract(mla_contract);
        const auto stage = simaai::neat::project_mla_stage_contract(compiled);
        const auto expected = policy == sc::ArenaDmsPolicy::PreferDmsThenCma
                                  ? sc::ArenaAllocationPolicy::PreferDmsThenCma
                                  : sc::ArenaAllocationPolicy::RequiredDomain;
        require(stage.frame_arena_allocation_policy == expected,
                "projection lost allocation policy");
        require(stage.frame_arena_storage_domain == sc::ArenaStorageDomain::Dms,
                "policy changed placement instead of allocation preference");
        sima::SimaPluginStaticManifest manifest;
        manifest.stages.push_back(stage);
        const auto encoded = sima::serialize_manifest_json(manifest);
        const auto decoded = sima::parse_manifest_json(encoded, &error);
        require(decoded && decoded->stages[0].frame_arena_allocation_policy == expected, error);
        require(decoded->stages[0].physical_outputs[0].size_bytes ==
                        stage.physical_outputs[0].size_bytes &&
                    decoded->stages[0].physical_outputs[0].source_byte_offset ==
                        stage.physical_outputs[0].source_byte_offset,
                "allocation policy changed physical geometry");
      }
      // A detached terminal carrier owns a separate strict placement. Reusing
      // a projection object must not retain the previous shared preference.
      two_mla_data.ops[1].kind = sc::OpKind::Mla;
      two_mla_data.ops[1].processor = "MLA";
      two_mla_data.ops[1].config = sc::MlaOpConfig{"decoder.elf", 4};
      two_mla_data.backend_ports.push_back(
          {1U, sc::BackendPortDirection::Input, 0U, "data.ifm.b0", 1U, 64U, 64U,
           sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::ReadOnly});
      two_mla_data.backend_ports.push_back(
          {1U, sc::BackendPortDirection::Output, 0U, "data.ofm.b0", 2U, 64U, 64U,
           sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::WriteOnly});
      const auto two_mla = sc::ModelExecutionPlan::create(std::move(two_mla_data), &error);
      require(two_mla.has_value(), error);
      const auto two_physical = sc::PhysicalExecutionLowerer::lower(*two_mla, &error);
      require(two_physical.has_value(), error);
      const auto detached = sc::detached_mla_output_roots(*two_mla, *two_physical);
      require(detached == std::vector<sc::ValueId>{2U}, "fixture must detach terminal MLA output");
      const auto two_arena = sc::FrameSlotArenaPlan::compile(
          *two_mla, *two_physical, sc::FrameSlotArenaReuse::DisjointLifetimes, 4096, &error,
          sc::ArenaDmsPolicy::PreferDmsThenCma, detached);
      require(two_arena.has_value(), error);
      sima::MlaStaticContract stale;
      stale.frame_arena_allocation_policy = sc::ArenaAllocationPolicy::PreferDmsThenCma;
      sima::PhysicalBufferStaticSpec previous_input;
      previous_input.physical_index = 0;
      previous_input.size_bytes = 64;
      previous_input.segment_name = "mla_ofm";
      stale.physical_inputs.push_back(previous_input);
      sima::TensorStaticSpec previous_logical;
      previous_logical.tensor_index = 0;
      stale.logical_inputs.push_back(previous_logical);
      sima::PhysicalBufferStaticSpec previous_output;
      previous_output.physical_index = 0;
      previous_output.size_bytes = 64;
      previous_output.segment_name = "host_ofm";
      stale.dispatcher_physical_outputs.push_back(previous_output);
      const std::array detached_sources{sc::PhysicalPortSource{1U, 0}};
      require(sc::apply_dmabuf_plan_contract_projection(*two_mla, 1U, *two_arena, &stale,
                                                        detached_sources, &error),
              error);
      require(stale.frame_arena_allocation_policy == sc::ArenaAllocationPolicy::RequiredDomain,
              "detached projection retained stale shared fallback preference");

      // Standalone ROI specialization owns new placement authority; a copied
      // model preference must not survive when its model-wide arena is cleared.
      simaai::neat::CompiledProcessCvuContract inherited;
      inherited.preproc_single_output_handoff = true;
      inherited.payload.graph_name = "preproc";
      inherited.payload.batch_size = 1;
      sima_ev_tensor_desc output_descriptor{};
      require(sima::tensorsemantics::build_dense_tensor_desc(
                  {1, 2, 2, 3}, "UINT8", "NHWC", &output_descriptor, &error, "empty", "rank",
                  "dimension", "dtype", "stride"),
              error);
      output_descriptor.storage.nbytes = 12;
      inherited.payload.input_tensors.emplace_back(output_descriptor, "UINT8");
      inherited.payload.output_tensors.emplace_back(output_descriptor, "UINT8");
      auto& inherited_runtime = inherited.runtime_contract;
      inherited_runtime.logical_inputs.resize(1);
      inherited_runtime.physical_inputs.resize(1);
      inherited_runtime.input_bindings.resize(1);
      inherited_runtime.physical_outputs.resize(1);
      inherited_runtime.physical_outputs[0].size_bytes = 12;
      inherited_runtime.physical_outputs[0].physical_index = 0;
      inherited_runtime.logical_outputs.resize(1);
      inherited_runtime.logical_outputs[0].backend_output_index = 0;
      inherited_runtime.logical_outputs[0].physical_index = 0;
      inherited_runtime.logical_outputs[0].shape = {1, 2, 2, 3};
      inherited_runtime.logical_outputs[0].layout = "NHWC";
      inherited_runtime.logical_outputs[0].size_bytes = 12;
      inherited.exposed_view.exposed_logical_outputs = inherited_runtime.logical_outputs;
      inherited_runtime.frame_arena_size_bytes = 4096;
      inherited_runtime.frame_arena_allocation_policy = sc::ArenaAllocationPolicy::PreferDmsThenCma;
      const auto standalone = sima::specialize_preproc_roi_contract(inherited, 2, 2, 2, 3, 3);
      require(standalone.runtime_contract.frame_arena_size_bytes == 0 &&
                  standalone.runtime_contract.frame_arena_allocation_policy ==
                      sc::ArenaAllocationPolicy::RequiredDomain,
              "standalone specialization retained stale allocation preference");
      require(standalone.runtime_contract.physical_inputs[0].size_bytes == 24 &&
                  standalone.runtime_contract.physical_outputs[0].size_bytes == 36 &&
                  standalone.runtime_contract.logical_outputs[0].shape ==
                      std::vector<std::int64_t>({3, 2, 2, 3}),
              "standalone authority reset changed dynamic geometry");

      SimaPluginStageSpec stage{};
      stage.frame_arena_size_bytes = 4096;
      stage.frame_arena_role = SIMA_PLUGIN_FRAME_ARENA_ALLOCATE;
      stage.frame_arena_storage_domain = SIMA_PLUGIN_FRAME_ARENA_STORAGE_DMS;
      stage.frame_arena_provenance = SIMA_PLUGIN_FRAME_ARENA_PROVENANCE_CORE_ALLOCATED;
      stage.frame_arena_required_device_access = SIMA_PLUGIN_FRAME_ARENA_ACCESS_MLA;
      stage.frame_arena_allocation_policy = SIMA_PLUGIN_FRAME_ARENA_PREFER_DMS_THEN_CMA;
      require(sima_plugin_frame_arena_allocation_policy_valid(&stage), "eligible policy rejected");
      stage.frame_arena_required_device_access |= SIMA_PLUGIN_FRAME_ARENA_ACCESS_EV74;
      require(!sima_plugin_frame_arena_allocation_policy_valid(&stage), "EV policy accepted");
      stage.frame_arena_required_device_access = SIMA_PLUGIN_FRAME_ARENA_ACCESS_MLA;
      stage.frame_arena_provenance = SIMA_PLUGIN_FRAME_ARENA_PROVENANCE_EXTERNAL_ADOPTED;
      require(!sima_plugin_frame_arena_allocation_policy_valid(&stage), "external policy accepted");
      gst_init(nullptr, nullptr);
      SimaPluginStaticManifestHandle legacy{};
      legacy.abi_version = 30;
      legacy.ref = +[](gpointer) {};
      legacy.unref = +[](gpointer) {};
      auto* context = gst_context_new(SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE, TRUE);
      gst_structure_set(gst_context_writable_structure(context),
                        SIMA_PLUGIN_STATIC_MANIFEST_KEY_HANDLE,
                        sima_plugin_static_manifest_handle_get_type(), &legacy, nullptr);
      SimaPluginManifestLookupStatus status = SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_OK;
      require(!sima_plugin_manifest_context_accessor_checked(context, &status) &&
                  status == SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_HANDLE_ABI_MISMATCH,
              "old accessor ABI accepted");
      gst_context_unref(context);
    }));

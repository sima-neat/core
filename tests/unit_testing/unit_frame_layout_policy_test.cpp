#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/CompiledProcessCvuContractQuery.h"
#include "pipeline/internal/sima/static_contract/CarrierPacking.h"
#include "pipeline/runtime/ExecutionPlanAssembly.h"
#include "test_main.h"

#include <nlohmann/json.hpp>
#include <array>
#include <memory>

namespace sc = simaai::neat::pipeline_internal::sima::static_contract;
namespace sima = simaai::neat::pipeline_internal::sima;

RUN_TEST("unit_frame_layout_policy_test", ([] {
           namespace client = ::simaai::scheduler::client;
           for (std::uint32_t access = 1; access < 8; ++access) {
             for (const auto eligible : {client::AllocationPolicy::RequireDms,
                                         client::AllocationPolicy::PreferDmsThenCma}) {
               const auto expected =
                   access & client::kAccessEv74 ? client::AllocationPolicy::RequireCma : eligible;
               require(sc::resolve_allocation_policy(access, eligible) == expected,
                       "an EV74 access must require CMA; CPU/MLA access preserves DMS policy");
             }
           }
           require(sc::resolve_allocation_policy(client::kAccessCpuA65,
                                                 client::AllocationPolicy::RequireHostCpu) ==
                       client::AllocationPolicy::RequireHostCpu,
                   "an eligible CPU-only terminal preserves explicit HostCpu");
           std::string error;
           // Public diagnostics describe final owned pools once, including detached outputs.
           const auto layout = [](client::FrameOwnership ownership,
                                  std::vector<client::FrameAllocation> allocations) {
             auto result = std::make_shared<client::FrameLayout>();
             result->ownership = ownership;
             result->allocations = std::move(allocations);
             return result;
           };
           const auto memory_report = [](const auto& layouts) {
             simaai::neat::pipeline_internal::SegmentBuildPlan build;
             build.rendered_manifest.emplace();
             for (const auto& frame : layouts) {
               sima::StageStaticSpec stage;
               stage.element_name = "stage_" + std::to_string(build.node_reports.size());
               stage.payload_kind = sima::StagePayloadKind::ProcessMla;
               stage.frame_layout = frame;
               simaai::neat::NodeReport node;
               node.index = static_cast<int>(build.node_reports.size());
               node.kind = "ModelFragment";
               node.elements = {stage.element_name};
               build.node_reports.push_back(std::move(node));
               build.rendered_manifest->stages.push_back(std::move(stage));
             }
             return nlohmann::json::parse(
                        simaai::neat::internal::ExecutionPlanAccess::make(build, "memory-report")
                            .to_json())
                 .at("model")
                 .at("memory");
           };
           const std::vector<client::FrameAllocation> parts{
               {4096U, 0U, client::AllocationPolicy::RequireCma},
               {8192U, 0U, client::AllocationPolicy::RequireDms}};
           const std::vector mixed_layouts{
               layout(client::FrameOwnership::Allocate, parts),
               layout(client::FrameOwnership::ReuseInput, parts),
               layout(client::FrameOwnership::Allocate,
                      {{2048U, 0U, client::AllocationPolicy::RequireDms}})};
           const auto mixed_memory = memory_report(mixed_layouts);
           require(mixed_memory.at("type") == "Mixed" &&
                       mixed_memory.at("per_frame_bytes") == 14336U,
                   "public plan must sum owned mixed and detached pools without counting reusers");
           for (const auto policy :
                {client::AllocationPolicy::RequireCma, client::AllocationPolicy::RequireDms,
                 client::AllocationPolicy::PreferDmsThenCma}) {
             const std::vector single_layout{
                 layout(client::FrameOwnership::Allocate, {{4096U, 0U, policy}})};
             const auto memory = memory_report(single_layout);
             require(memory.size() == 2 && memory.at("per_frame_bytes") == 4096U &&
                         memory.at("type") ==
                             (policy == client::AllocationPolicy::RequireCma ? "CMA" : "DMS"),
                     "homogeneous public memory shape must retain its planned heap");
           }

           // Specializing image geometry discards the source model's storage plan.
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
           inherited_runtime.frame_layout =
               layout(client::FrameOwnership::Allocate,
                      {{4096, client::kAccessCpuA65, client::AllocationPolicy::PreferDmsThenCma}});
           inherited_runtime.allocation_policy = client::AllocationPolicy::PreferDmsThenCma;
           inherited_runtime.reuse_input_pad = 0;
           const auto standalone = sima::specialize_preproc_roi_contract(inherited, 2, 2, 2, 3, 3);
           require(!standalone.runtime_contract.frame_layout &&
                       !standalone.runtime_contract.allocation_policy &&
                       !standalone.runtime_contract.reuse_input_pad,
                   "standalone specialization retained stale allocation preference");
           require(standalone.runtime_contract.physical_inputs[0].size_bytes == 24 &&
                       standalone.runtime_contract.physical_outputs[0].size_bytes == 36 &&
                       standalone.runtime_contract.logical_outputs[0].shape ==
                           std::vector<std::int64_t>({3, 2, 2, 3}),
                   "standalone authority reset changed dynamic geometry");
         }));

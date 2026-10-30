#include "../../tools/pcie_output_capacity.h"
#include "test_main.h"

RUN_TEST("unit_pcie_output_capacity_test", ([] {
           using simaai::neat::pcie_builder::output_capacity;
           namespace scheduler = simaai::scheduler::client;
           scheduler::FrameLayout layout;
           layout.allocations = {
               {1024, scheduler::kAccessEv74, scheduler::AllocationPolicy::RequireCma},
               {256, scheduler::kAccessMla, scheduler::AllocationPolicy::RequireDms}};
           require(output_capacity(layout) == 1280, "must cover each final backing allocation");
           // The same compiled envelope includes output gaps, private scratch and any
           // checked wire serializer output; PCIe need not reconstruct backend formats.
           layout.allocations = {
               {4096, scheduler::kAccessCpuA65, scheduler::AllocationPolicy::RequireDms}};
           require(output_capacity(layout) == 4096, "must retain the final allocation envelope");
           const auto rejects = [](const scheduler::FrameLayout& invalid) {
             try {
               (void)output_capacity(invalid);
               return false;
             } catch (const std::runtime_error&) {
               return true;
             }
           };
           layout.allocations[0].size_bytes = 0;
           require(rejects(layout), "must reject an empty output allocation");
           layout.allocations = {{std::numeric_limits<std::size_t>::max()}, {1}};
           require(rejects(layout), "must reject an overflowing receive-capacity total");
           require(rejects(scheduler::FrameLayout{}), "must reject an unknown output capacity");
         }));

#include "support/quanttess_fanin_fixture.h"
#include "test_main.h"

// Preserve the three-branch pre-MLA fan-in regression at the production typed
// plan boundary. No test-only MPK reconstruction or firmware graph ID is used.
RUN_TEST(
    "unit_mpk_fuse_sibling_pre_mla_test", ([] {
      namespace sc = simaai::neat::pipeline_internal::sima::static_contract;
      const auto fixture = sima_test::compile_processcvu_fixture(
          sima_test::quanttess_fanin_plan_data({2, 2, 2}, {0, 1, 2}), "quanttess");
      const auto& compiled = fixture.contract;
      require(compiled.payload.graph_name == "quanttess" && compiled.payload.graph_id == 0,
              "fan-in must retain canonical fused QuantTess identity");
      require(compiled.payload.num_in_tensor == 3 && compiled.payload.input_tensors.size() == 3U &&
                  compiled.payload.output_tensors.size() == 3U,
              "all three independent sibling branches must be retained");
      require(fixture.physical.commands.size() == 2U,
              "direct placement must need only one fused CVU command and MLA");
      const auto& cvu = fixture.physical.commands.front();
      require(cvu.engine == sc::PhysicalEngine::Cvu && cvu.members.size() == 3U,
              "siblings must share one three-member fused command");
      for (std::size_t lane = 0; lane < 3U; ++lane) {
        const auto& member = cvu.members[lane];
        require(member.semantic_chain.size() == 2U &&
                    fixture.semantic.ops()[member.semantic_chain[0]].kind == sc::OpKind::Quantize &&
                    fixture.semantic.ops()[member.semantic_chain[1]].kind == sc::OpKind::Tessellate,
                "every member must retain the connected Quantize + Tessellate fusion");
        const auto intermediate = fixture.semantic.ops()[member.semantic_chain[0]].outputs.front();
        require(!fixture.arena.region(intermediate),
                "fused scalar intermediate must not allocate storage");
        const auto& input = compiled.runtime_contract.physical_inputs.at(lane);
        require(input.source_physical_index == static_cast<int>(lane) &&
                    input.source_byte_offset == 0 && input.size_bytes == 16U,
                "public source identity and span must survive fan-in without repacking");
      }
    }));

#include "pipeline/internal/sima/stagesemantics/ProcessMlaStageSemantics.h"
#include "test_main.h"

RUN_TEST("unit_contract_compiler_mla_test", ([] {
           using namespace simaai::neat::pipeline_internal::sima;
           using namespace simaai::neat::pipeline_internal::sima::stagesemantics;

           MlaStaticContract contract;
           contract.stage_id = "mla_0";
           contract.node_name = "mla_0";
           TensorStaticSpec input;
           input.tensor_index = 0;
           input.shape = {1, 1228800};
           input.dtype = "INT8";
           input.layout = "HW";
           input.semantic_tag = "mla_ifm_parent";
           contract.inputs.push_back(input);

           TensorStaticSpec logical_input;
           logical_input.tensor_index = 0;
           logical_input.shape = {1, 1228800};
           logical_input.dtype = "INT8";
           logical_input.layout = "HW";
           logical_input.semantic_tag = "output_tessellated_image";
           contract.logical_inputs.push_back(logical_input);
           const std::uint64_t expected_logical_input_bytes = 1228800U;

           PhysicalBufferStaticSpec physical_input;
           physical_input.physical_index = 0;
           physical_input.allocator_index = 0;
           physical_input.source_physical_index = 0;
           physical_input.source_byte_offset = 0;
           physical_input.size_bytes = 1228800;
           physical_input.device_kind = DeviceKind::Evxx;
           physical_input.segment_name = "output_tessellated_image";
           contract.physical_inputs.push_back(physical_input);

           PhysicalBufferStaticSpec physical;
           physical.physical_index = 0;
           physical.allocator_index = 0;
           physical.size_bytes = 1209600;
           physical.device_kind = DeviceKind::Mla;
           physical.segment_name = "MLA_0";
           contract.dispatcher_physical_outputs.push_back(physical);
           contract.physical_outputs.push_back(physical);

           LogicalTensorStaticSpec logical;
           logical.logical_index = 0;
           logical.backend_output_index = 0;
           logical.physical_index = 0;
           logical.output_slot = 0;
           logical.tensor_index = 0;
           logical.byte_offset = 0;
           logical.size_bytes = 409600;
           logical.shape = {80, 80, 64};
           logical.dtype = "INT8";
           logical.layout = "HWC";
           logical.logical_name = "head0";
           logical.backend_name = "head0";
           logical.segment_name = "MLA_0";
           contract.logical_outputs.push_back(logical);

           contract.model_path = "/tmp/fake_model.bin";
           contract.batch_size = 1;
           contract.batch_sz_model = 1;

           const auto compiled = build_mla_compiled_contract(contract);
           require(compiled.runtime_contract.logical_inputs.size() == 1U,
                   "MLA contract should expose one logical input");
           require(compiled.runtime_contract.input_bindings.size() == 1U,
                   "MLA contract should expose one input binding");
           require(compiled.runtime_contract.physical_inputs.size() == 1U,
                   "MLA contract should expose one physical input");
           require(compiled.runtime_contract.physical_outputs.size() == 1U,
                   "MLA contract should expose one physical output");
           require(compiled.runtime_contract.logical_outputs.size() == 1U,
                   "MLA contract should expose one logical output");
           require(compiled.runtime_contract.output_order.size() == 1U,
                   "MLA contract should expose one output route");
           require(compiled.runtime_contract.logical_inputs.front().size_bytes ==
                       expected_logical_input_bytes,
                   "MLA logical input should preserve the packed ingress byte size");
           require(compiled.runtime_contract.physical_inputs.front().size_bytes == 1228800U,
                   "MLA physical input should preserve the packed parent byte size");
           require(compiled.runtime_contract.input_bindings.front().src_physical_size_bytes ==
                       1228800U,
                   "MLA input binding should preserve the packed parent byte size");
           require(compiled.runtime_contract.input_bindings.front().source_segment_name ==
                       "output_tessellated_image",
                   "MLA input binding should preserve the upstream runtime segment name");
           require(compiled.runtime_contract.output_order.front().cm_output_name == "head0",
                   "MLA output route should preserve backend output name");
         }));

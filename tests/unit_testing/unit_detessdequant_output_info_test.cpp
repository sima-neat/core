#include "nodes/sima/DetessDequant.h"
#include "pipeline/internal/RenderedMlaContractQuery.h"
#include "test_main.h"

#include <memory>
#include <vector>

namespace {
using namespace simaai::neat;
namespace query = pipeline_internal::rendered_stage_query;
namespace contract = pipeline_internal::sima;

std::vector<std::shared_ptr<Node>> compiled_outputs(std::int64_t batch) {
  auto compiled = std::make_shared<CompiledProcessCvuContract>();
  compiled->payload.graph_name = compiled->payload.graph_family = "detessdequant";
  compiled->payload.batch_size = static_cast<int>(batch);
  compiled->physical_command_role = contract::static_contract::PhysicalCommandRole::Egress;
  compiled->runtime_contract.plugin_kind = "processcvu";

  contract::LogicalTensorStaticSpec first;
  first.logical_index = first.output_slot = first.backend_output_index = 0;
  first.physical_index = 0;
  first.logical_name = first.backend_name = "boxes";
  first.segment_name = "packed_output";
  first.dtype = "FP32";
  first.layout = "NCHW";
  first.shape = {batch, 6, 3, 4};
  first.stride_bytes = {512, 64, 16, 4};
  first.byte_offset = 32;
  first.size_bytes = 512 * batch;
  auto second = first;
  second.logical_index = second.output_slot = second.backend_output_index = 1;
  second.logical_name = second.backend_name = "scores";
  second.shape = {batch, 7, 5, 2};
  second.stride_bytes = {768, 96, 16, 4};
  second.byte_offset = 4096;
  second.size_bytes = 768 * batch;
  compiled->runtime_contract.logical_outputs = {first, second};
  contract::PhysicalBufferStaticSpec carrier;
  carrier.physical_index = 0;
  carrier.segment_name = "packed_output";
  carrier.size_bytes = 8192;
  compiled->runtime_contract.physical_outputs = {carrier};
  for (const auto& logical : compiled->runtime_contract.logical_outputs) {
    contract::StageOutputRoute route;
    route.logical_output_index = logical.logical_index;
    route.output_slot = logical.output_slot;
    route.cm_output_name = logical.logical_name;
    route.segment_name = logical.segment_name;
    compiled->runtime_contract.output_order.push_back(std::move(route));
  }
  compiled->exposed_view.exposed_logical_outputs = compiled->runtime_contract.logical_outputs;
  compiled->exposed_view.exposed_output_order = compiled->runtime_contract.output_order;

  DetessDequantOptions options;
  options.num_buffers = options.num_buffers_model = 4;
  options.num_buffers_locked = true;
  options.compiled_contract = std::move(compiled);
  return {nodes::DetessDequant(std::move(options))};
}

void check_outputs(const std::vector<std::shared_ptr<Node>>& nodes, bool include_batch,
                   const std::vector<std::vector<std::int64_t>>& shapes,
                   const std::vector<std::vector<std::int64_t>>& strides) {
  const auto info = query::detessdequant_output_info(nodes, include_batch);
  require(info.dtype == TensorDType::Float32 && info.layout == TensorLayout::CHW &&
              info.output_format == "NCHW" && info.outputs.size() == 2,
          "compiled DetessDequant query lost dtype, layout or output count");
  for (std::size_t i = 0; i < info.outputs.size(); ++i) {
    require(info.outputs[i].shape == shapes[i] && info.outputs[i].stride_bytes == strides[i] &&
                info.outputs[i].byte_offset == (i == 0 ? 32 : 4096),
            "compiled DetessDequant query changed a head's shape, strides or packed offset");
  }
}
} // namespace

RUN_TEST("unit_detessdequant_output_info_test", ([] {
           const auto single = compiled_outputs(1);
           check_outputs(single, false, {{6, 3, 4}, {7, 5, 2}}, {{64, 16, 4}, {96, 16, 4}});
           check_outputs(single, true, {{1, 6, 3, 4}, {1, 7, 5, 2}},
                         {{512, 64, 16, 4}, {768, 96, 16, 4}});

           const auto batched = compiled_outputs(3);
           for (const bool include_batch : {false, true})
             check_outputs(batched, include_batch, {{3, 6, 3, 4}, {3, 7, 5, 2}},
                           {{512, 64, 16, 4}, {768, 96, 16, 4}});
         }));

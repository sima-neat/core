#include <gstsimaaitensorbuffer.h>

#include "test_main.h"

#include <array>
#include <string>

namespace {

struct OverlayFixture {
  std::array<gint64, 3> head_shape = {12, 1, 1};
  std::array<gint64, 3> body_shape = {12, 52, 45};
  std::array<gint64, 3> head_stride = {4, 4, 4};
  std::array<gint64, 3> body_stride = {9360, 180, 4};
  std::array<SimaPluginPhysicalBuffer, 2> physical_outputs{};
  std::array<SimaPluginLogicalTensor, 2> logical_outputs{};
  std::array<SimaPluginOutputRoute, 2> output_order{};
  SimaPluginStageSpec stage{};

  OverlayFixture() {
    physical_outputs[0].physical_index = 0;
    physical_outputs[0].size_bytes = 384;
    physical_outputs[0].segment_name = "output_head";

    physical_outputs[1].physical_index = 1;
    physical_outputs[1].size_bytes = 119808;
    physical_outputs[1].segment_name = "output_body";

    logical_outputs[0].logical_index = 0;
    logical_outputs[0].physical_index = 0;
    logical_outputs[0].backend_output_index = 0;
    logical_outputs[0].output_slot = 0;
    logical_outputs[0].tensor_index = 0;
    logical_outputs[0].byte_offset = 0;
    logical_outputs[0].size_bytes = 384;
    logical_outputs[0].shape = head_shape.data();
    logical_outputs[0].shape_len = static_cast<guint>(head_shape.size());
    logical_outputs[0].stride_bytes = head_stride.data();
    logical_outputs[0].stride_bytes_len = static_cast<guint>(head_stride.size());
    logical_outputs[0].dtype = "FP32";
    logical_outputs[0].logical_name = "semseg_head";
    logical_outputs[0].backend_name = "semseg_head";
    logical_outputs[0].segment_name = "output_head";

    logical_outputs[1] = logical_outputs[0];
    logical_outputs[1].logical_index = 1;
    logical_outputs[1].physical_index = 1;
    logical_outputs[1].backend_output_index = 1;
    logical_outputs[1].output_slot = 1;
    logical_outputs[1].size_bytes = 119808;
    logical_outputs[1].shape = body_shape.data();
    logical_outputs[1].shape_len = static_cast<guint>(body_shape.size());
    logical_outputs[1].stride_bytes = body_stride.data();
    logical_outputs[1].stride_bytes_len = static_cast<guint>(body_stride.size());
    logical_outputs[1].logical_name = "semseg_body";
    logical_outputs[1].backend_name = "semseg_body";
    logical_outputs[1].segment_name = "output_body";

    output_order[0].output_slot = 0;
    output_order[0].cm_output_name = "semseg_head";
    output_order[0].segment_name = "output_head";
    output_order[1].output_slot = 1;
    output_order[1].cm_output_name = "semseg_body";
    output_order[1].segment_name = "output_body";

    stage.element_name = "neatcast";
    stage.logical_stage_id = "post_cast";
    stage.plugin_kind = "cast";
    stage.kernel_kind = "cast";
    stage.physical_outputs = physical_outputs.data();
    stage.physical_outputs_len = static_cast<guint>(physical_outputs.size());
    stage.logical_outputs = logical_outputs.data();
    stage.logical_outputs_len = static_cast<guint>(logical_outputs.size());
    stage.output_order = output_order.data();
    stage.output_order_len = static_cast<guint>(output_order.size());
  }
};

} // namespace

RUN_TEST("unit_tensorbuffer_runtime_contract_identity_overlay_test", ([] {
           OverlayFixture fixture;

           simaai::gst::TensorBufferPublishContract runtime_contract;
           runtime_contract.stage_key = "upstream_parent";
           runtime_contract.physical_outputs.push_back({0, 24U, "upstream_head"});
           runtime_contract.physical_outputs.push_back({1, 56160U, "upstream_body"});

           simaai::gst::TensorBufferPublishLogicalOutput head;
           head.logical_index = 0;
           head.physical_index = 0;
           head.memory_index = 0;
           head.backend_output_index = 0;
           head.route_slot = 0;
           head.logical_name = "upstream_head";
           head.backend_name = "upstream_head";
           head.segment_name = "upstream_head";
           head.byte_offset = 0;
           head.size_bytes = 24U;
           head.dtype = SIMA_TENSOR_SET_DTYPE_FP32_V1;
           head.layout = SIMA_TENSOR_SET_LAYOUT_HWC_V1;
           head.shape = {12, 1, 1};
           head.stride_bytes = {4, 4, 4};
           runtime_contract.logical_outputs.push_back(head);
           runtime_contract.output_order.push_back({0, 0, "upstream_head", "upstream_head"});

           simaai::gst::TensorBufferPublishLogicalOutput body = head;
           body.logical_index = 1;
           body.physical_index = 1;
           body.memory_index = 1;
           body.backend_output_index = 1;
           body.route_slot = 1;
           body.logical_name = "upstream_body";
           body.backend_name = "upstream_body";
           body.segment_name = "upstream_body";
           body.size_bytes = 56160U;
           body.shape = {12, 52, 45};
           body.stride_bytes = {4680, 90, 2};
           runtime_contract.logical_outputs.push_back(body);
           runtime_contract.output_order.push_back({1, 1, "upstream_body", "upstream_body"});

           simaai::gst::TensorBufferPublishContract merged;
           std::string err;
           require(simaai::gst::tensor_buffer_overlay_stage_identity_onto_runtime_publish_contract(
                       runtime_contract, fixture.stage,
                       simaai::gst::TensorBufferProducerKind::Transport, &merged, &err),
                   std::string("failed to overlay stage identity onto runtime contract: ") + err);

           require(merged.stage_key == "post_cast",
                   "merged contract should publish the consumer stage key");
           require(merged.logical_outputs.size() == 2U,
                   "merged contract should preserve runtime logical output count");
           require(merged.logical_outputs[0].size_bytes == 24U,
                   "merged contract should preserve runtime logical bytes for head output");
           require(merged.logical_outputs[1].size_bytes == 56160U,
                   "merged contract should preserve runtime logical bytes for body output");
           require(merged.physical_outputs[0].size_bytes == 24U &&
                       merged.physical_outputs[1].size_bytes == 56160U,
                   "merged contract should preserve runtime physical span bytes");
           require(merged.logical_outputs[0].logical_name == "semseg_head" &&
                       merged.logical_outputs[1].logical_name == "semseg_body",
                   "merged contract should adopt stage logical names");
           require(merged.output_order[0].cm_output_name == "semseg_head" &&
                       merged.output_order[1].cm_output_name == "semseg_body",
                   "merged contract should adopt stage route names");
         }));

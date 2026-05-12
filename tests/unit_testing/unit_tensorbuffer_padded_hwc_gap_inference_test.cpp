#include <gstsimaaitensorbuffer.h>

#include "test_main.h"

#include <gst/gst.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

using GstBufferPtr = std::unique_ptr<GstBuffer, decltype(&gst_buffer_unref)>;

void ensure_gst_ready() {
  int argc = 0;
  char** argv = nullptr;
  gst_init(&argc, &argv);
}

GstBufferPtr make_parent_buffer(const std::vector<std::uint8_t>& bytes) {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, bytes.size(), nullptr);
  require(buffer != nullptr, "failed to allocate parent buffer");
  GstMapInfo map{};
  require(gst_buffer_map(buffer, &map, GST_MAP_WRITE), "failed to map parent buffer");
  std::copy(bytes.begin(), bytes.end(), map.data);
  gst_buffer_unmap(buffer, &map);
  return GstBufferPtr(buffer, &gst_buffer_unref);
}

} // namespace

RUN_TEST("unit_tensorbuffer_padded_hwc_gap_inference_test", ([] {
           ensure_gst_ready();

           const std::vector<std::uint8_t> parent_bytes = {
               1, 2, 3, 99, 4, 5, 6, 7, 8, 9, 10, 11,
           };
           auto parent = make_parent_buffer(parent_bytes);

           GstBuffer* raw = nullptr;
           std::string err;
           require(simaai::gst::tensor_buffer_build_segmented_buffer(
                       {{"packed_parent", parent.get(), parent_bytes.size()}}, &raw, &err),
                   std::string("failed to build parent-segment tensor buffer: ") + err);
           GstBufferPtr buffer(raw, &gst_buffer_unref);

           simaai::gst::TensorBufferPublishContract contract;
           contract.stage_key = "padded_gap_test";
           contract.physical_outputs.push_back({0, parent_bytes.size(), "packed_parent"});

           simaai::gst::TensorBufferPublishLogicalOutput first;
           first.logical_index = 0;
           first.physical_index = 0;
           first.memory_index = 0;
           first.backend_output_index = 0;
           first.route_slot = 0;
           first.logical_name = "first";
           first.backend_name = "first";
           first.segment_name = "packed_parent";
           first.byte_offset = 0;
           first.size_bytes = 6U;
           first.dtype = SIMA_TENSOR_SET_DTYPE_INT8_V1;
           first.layout = SIMA_TENSOR_SET_LAYOUT_HWC_V1;
           first.shape = {1, 2, 3};
           first.stride_bytes = {6, 3, 1};
           contract.logical_outputs.push_back(first);
           contract.output_order.push_back({0, 0, "first", "packed_parent"});

           simaai::gst::TensorBufferPublishLogicalOutput second;
           second.logical_index = 1;
           second.physical_index = 0;
           second.memory_index = 0;
           second.backend_output_index = 1;
           second.route_slot = 1;
           second.logical_name = "second";
           second.backend_name = "second";
           second.segment_name = "packed_parent";
           second.byte_offset = 8;
           second.size_bytes = 4U;
           second.dtype = SIMA_TENSOR_SET_DTYPE_INT8_V1;
           second.layout = SIMA_TENSOR_SET_LAYOUT_HWC_V1;
           second.shape = {1, 1, 4};
           second.stride_bytes = {4, 4, 1};
           contract.logical_outputs.push_back(second);
           contract.output_order.push_back({1, 1, "second", "packed_parent"});

           simaai::gst::TensorBufferView view;
           err.clear();
           require(simaai::gst::tensor_buffer_build_publish_view_from_contract(
                       buffer.get(), contract, simaai::gst::TensorBufferProducerKind::Transport,
                       &view, &err),
                   std::string("failed to build padded publish view: ") + err);
           require(view.tensors.size() == 2U, "publish view should expose two logical tensors");
           require(view.tensors[0].stride_bytes == std::vector<std::int64_t>({8, 4, 1}),
                   "first tensor should infer padded HWC strides from the physical gap");
           // Physical span now includes the trailing pad gap (8 bytes
           // including padding) instead of the dense-only 7-byte span.
           require(view.tensors[0].physical_span_bytes == 8U,
                   "first tensor physical_span_bytes should include the padded HWC gap");

           std::vector<std::uint8_t> materialized;
           err.clear();
           require(simaai::gst::tensor_buffer_materialize(view, std::vector<std::size_t>{6U, 4U},
                                                          &materialized, &err),
                   std::string("failed to materialize padded publish view: ") + err);
           require(materialized == std::vector<std::uint8_t>({1, 2, 3, 4, 5, 6, 8, 9, 10, 11}),
                   "materialized bytes should skip padded channel slots and preserve tensor order");
         }));

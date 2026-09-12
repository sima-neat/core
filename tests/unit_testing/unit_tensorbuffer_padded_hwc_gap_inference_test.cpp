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
           // The producer owns tensor geometry.  This logical C3 view is
           // stored in C4 rows, so publish the padded row stride explicitly
           // rather than asking TensorBuffer to guess it from the next
           // tensor's offset.
           first.stride_bytes = {8, 4, 1};
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
                   "first tensor should preserve its explicit padded HWC strides");
           // physical_span_bytes is the byte range the logical view may
           // address.  The final padding byte belongs to the 8-byte carrier
           // partition but is not touched by this C3 view.
           require(view.tensors[0].physical_span_bytes == 7U,
                   "first tensor physical_span_bytes should describe its addressed range");
           require(view.segments.size() == 1U && view.segments[0].size_bytes == 12U,
                   "the packed carrier extent should remain independent of the logical span");

           std::vector<std::uint8_t> materialized;
           err.clear();
           require(simaai::gst::tensor_buffer_materialize(view, std::vector<std::size_t>{6U, 4U},
                                                          &materialized, &err),
                   std::string("failed to materialize padded publish view: ") + err);
           require(materialized == std::vector<std::uint8_t>({1, 2, 3, 4, 5, 6, 8, 9, 10, 11}),
                   "materialized bytes should skip padded channel slots and preserve tensor order");

           // A larger carrier is not evidence of padded tensor geometry.
           // Pool reuse may place a dense tensor in storage sized for a
           // larger sibling, and the declared dense view must remain dense.
           const std::vector<std::uint8_t> oversized_bytes = {
               1, 2, 3, 4, 5, 6, 77, 88,
           };
           auto oversized_parent = make_parent_buffer(oversized_bytes);
           GstBuffer* oversized_raw = nullptr;
           err.clear();
           require(simaai::gst::tensor_buffer_build_segmented_buffer(
                       {{"oversized_parent", oversized_parent.get(), oversized_bytes.size()}},
                       &oversized_raw, &err),
                   std::string("failed to build oversized carrier: ") + err);
           GstBufferPtr oversized_buffer(oversized_raw, &gst_buffer_unref);

           simaai::gst::TensorBufferPublishContract dense_contract;
           dense_contract.stage_key = "dense_oversized_carrier_test";
           dense_contract.physical_outputs.push_back(
               {0, oversized_bytes.size(), "oversized_parent"});
           simaai::gst::TensorBufferPublishLogicalOutput dense = first;
           dense.logical_name = "dense";
           dense.backend_name = "dense";
           dense.segment_name = "oversized_parent";
           dense.stride_bytes = {6, 3, 1};
           dense_contract.logical_outputs.push_back(dense);
           dense_contract.output_order.push_back(
               {0, dense.logical_index, dense.backend_name, dense.segment_name});

           simaai::gst::TensorBufferView dense_view;
           err.clear();
           require(simaai::gst::tensor_buffer_build_publish_view_from_contract(
                       oversized_buffer.get(), dense_contract,
                       simaai::gst::TensorBufferProducerKind::Transport, &dense_view, &err),
                   std::string("failed to build dense oversized-carrier view: ") + err);
           require(dense_view.tensors.size() == 1U,
                   "dense oversized-carrier view should expose one logical tensor");
           require(dense_view.tensors[0].stride_bytes ==
                       std::vector<std::int64_t>({6, 3, 1}),
                   "carrier capacity must not rewrite declared dense strides");
           require(dense_view.tensors[0].physical_span_bytes == 6U,
                   "dense logical span must not expand to carrier capacity");
         }));

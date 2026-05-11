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

RUN_TEST("unit_tensorbuffer_bind_preserve_source_view_test", ([] {
           ensure_gst_ready();

           const std::vector<std::uint8_t> parent_bytes = {
               0, 1, 2, 3,
               4, 5, 6, 7,
               8, 9, 10, 11,
               12, 13, 14, 15,
           };
           auto parent = make_parent_buffer(parent_bytes);

           GstBuffer* raw = nullptr;
           std::string err;
           require(simaai::gst::tensor_buffer_build_segmented_buffer(
                       {{"MLA_0", parent.get(), parent_bytes.size()}}, &raw, &err),
                   std::string("failed to build parent-segment tensor buffer: ") + err);
           GstBufferPtr buffer(raw, &gst_buffer_unref);

           simaai::gst::TensorBufferPublishContract contract;
           contract.stage_key = "MLA_0_test";
           contract.physical_outputs.push_back({0, parent_bytes.size(), "MLA_0"});

           simaai::gst::TensorBufferPublishLogicalOutput logical;
           logical.logical_index = 0;
           logical.physical_index = 0;
           logical.backend_output_index = 0;
           logical.route_slot = 0;
           logical.logical_name = "slice_head";
           logical.backend_name = "slice_head";
           logical.segment_name = "MLA_0";
           logical.byte_offset = 0;
           logical.size_bytes = 12;
           logical.dtype = SIMA_TENSOR_SET_DTYPE_INT8_V1;
           logical.layout = SIMA_TENSOR_SET_LAYOUT_HW_V1;
           logical.shape = {4, 3};
           logical.stride_bytes = {4, 1};
           contract.logical_outputs.push_back(logical);
           contract.output_order.push_back({0, 0, "slice_head", "MLA_0"});

           simaai::gst::TensorBufferView view;
           err.clear();
           require(simaai::gst::tensor_buffer_build_publish_view_from_contract(
                       buffer.get(), contract, simaai::gst::TensorBufferProducerKind::ProcessMla,
                       &view, &err),
                   std::string("failed to build strided publish view: ") + err);
           require(view.tensors.size() == 1U, "publish view should expose one logical tensor");
           require(view.tensors.front().physical_span_bytes == 16U,
                   "publish view should preserve non-dense physical span");

           simaai::gst::TensorBufferReadRequest request;
           request.stage_key = "detesscast_test";
           simaai::gst::TensorBufferReadRequestEntry entry;
           entry.request_index = 0U;
           entry.logical_index = 0;
           entry.logical_name = "slice_head";
           entry.expected_size_bytes = 12U;
           entry.preserve_source_view = true;
           request.entries.push_back(entry);

           simaai::gst::TensorBufferBindingResult bindings;
           err.clear();
           require(simaai::gst::tensor_buffer_bind_inputs(view, request, false, false, &bindings,
                                                          &err),
                   std::string("failed to bind preserve-source-view request: ") + err);
           require(bindings.bindings.size() == 1U,
                   "preserve-source-view bind should expose one binding");
           const auto& binding = bindings.bindings.front();
           require(binding.data != nullptr,
                   "preserve-source-view binding should materialize source-view bytes");
           require(binding.size_bytes == 16U,
                   "binding bytes should match the physical source span");
           require(binding.tensor.size_bytes == 12U,
                   "binding tensor logical bytes should remain semantic/logical");
           require(binding.span.logical_size_bytes == 12U,
                   "resolved span should preserve logical byte size");
           require(binding.span.physical_span_bytes == 16U,
                   "resolved span should preserve physical source span");

           const std::vector<std::uint8_t> bound_bytes(binding.data,
                                                       binding.data + binding.size_bytes);
           require(bound_bytes == parent_bytes,
                   "preserve-source-view binding should copy the authoritative parent span bytes");
         }));

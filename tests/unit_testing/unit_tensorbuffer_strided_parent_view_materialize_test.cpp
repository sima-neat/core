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

RUN_TEST("unit_tensorbuffer_strided_parent_view_materialize_test", ([] {
           ensure_gst_ready();

           const std::vector<std::uint8_t> parent_bytes = {
               0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
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
           require(view.tensors.front().byte_offset == 0,
                   "strided parent view should preserve parent-relative byte offset");
           require(view.tensors.front().physical_span_bytes == 16U,
                   "strided parent view physical_span_bytes should include the trailing pad");

           simaai::gst::TensorBufferContiguousSpanView span;
           err.clear();
           require(simaai::gst::tensor_buffer_resolve_span(view, view.tensors.front(), &span, &err),
                   std::string("failed to resolve strided span: ") + err);
           require(span.segment != nullptr && span.segment->name == "MLA_0",
                   "resolved strided span should stay bound to the raw parent segment");
           require(span.physical_span_bytes == 16U,
                   "resolved span physical_span_bytes should include the trailing pad");

           std::vector<std::uint8_t> materialized;
           err.clear();
           require(simaai::gst::tensor_buffer_materialize(view, std::vector<std::size_t>{12U},
                                                          &materialized, &err),
                   std::string("failed to materialize strided parent view: ") + err);
           require(materialized ==
                       std::vector<std::uint8_t>({0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14}),
                   "materialized bytes should match a dense CPU slice of the parent segment");
         }));

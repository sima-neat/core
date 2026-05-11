#include <gstsimaaitensorbuffer.h>

#include "test_main.h"

#include <gst/gst.h>

#include <cstring>
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

GstBufferPtr make_parent_buffer(std::size_t size, std::uint8_t fill) {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
  require(buffer != nullptr, "failed to allocate parent buffer");
  GstMapInfo map{};
  require(gst_buffer_map(buffer, &map, GST_MAP_WRITE), "failed to map parent buffer");
  std::memset(map.data, fill, map.size);
  gst_buffer_unmap(buffer, &map);
  return GstBufferPtr(buffer, &gst_buffer_unref);
}

} // namespace

RUN_TEST("unit_tensorbuffer_packed_runtime_parent_test", ([] {
           ensure_gst_ready();

           auto buffer = make_parent_buffer(12U, 0x5A);

           simaai::gst::TensorBufferPublishContract contract;
           contract.stage_key = "packed_parent_test";
           contract.physical_outputs.push_back({0, 4U, "ofm0"});
           contract.physical_outputs.push_back({1, 8U, "ofm1"});

           simaai::gst::TensorBufferPublishLogicalOutput lhs;
           lhs.logical_index = 0;
           lhs.physical_index = 0;
           lhs.memory_index = 0;
           lhs.backend_output_index = 0;
           lhs.route_slot = 0;
           lhs.logical_name = "lhs";
           lhs.backend_name = "lhs";
           lhs.segment_name = "ofm0";
           lhs.byte_offset = 0;
           lhs.size_bytes = 4U;
           lhs.dtype = SIMA_TENSOR_SET_DTYPE_INT8_V1;
           lhs.layout = SIMA_TENSOR_SET_LAYOUT_HW_V1;
           lhs.shape = {1, 4};
           lhs.stride_bytes = {4, 1};
           contract.logical_outputs.push_back(lhs);
           contract.output_order.push_back({0, 0, "lhs", "ofm0"});

           simaai::gst::TensorBufferPublishLogicalOutput rhs = lhs;
           rhs.logical_index = 1;
           rhs.physical_index = 1;
           rhs.memory_index = 1;
           rhs.backend_output_index = 1;
           rhs.route_slot = 1;
           rhs.logical_name = "rhs";
           rhs.backend_name = "rhs";
           rhs.segment_name = "ofm1";
           rhs.size_bytes = 8U;
           rhs.shape = {1, 8};
           contract.logical_outputs.push_back(rhs);
           contract.output_order.push_back({1, 1, "rhs", "ofm1"});

           simaai::gst::TensorBufferView view;
           std::string err;
           require(simaai::gst::tensor_buffer_build_publish_view_from_contract(
                       buffer.get(), contract, simaai::gst::TensorBufferProducerKind::Transport,
                       &view, &err),
                   std::string("failed to build packed publish view: ") + err);
           require(view.segments.size() == 1U,
                   "packed runtime publish view should expose one runtime segment");
           require(view.segments.front().name == "__tensorbuffer_packed_parent__",
                   "packed runtime publish view should expose the canonical packed parent name");
           require(view.tensors.size() == 2U,
                   "packed runtime publish view should preserve both logical outputs");
           require(view.tensors[0].memory_index == 0 && view.tensors[1].memory_index == 0,
                   "packed runtime publish view should bind all logical outputs to one runtime memory");
           require(view.tensors[0].byte_offset == 0,
                   "first packed logical output should start at the head of the runtime buffer");
           require(view.tensors[1].byte_offset == 4,
                   "second packed logical output should be rebased after the first physical span");
           require(view.tensors[0].physical_index == 0 && view.tensors[1].physical_index == 1,
                   "packed runtime publish view should preserve logical physical identities");
         }));

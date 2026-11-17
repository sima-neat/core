#pragma once

#include <gstsimaaitensorbuffer_meta.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace simaai::neat::test_support {
// A real byte carrier for metadata-only resolver tests. Frame facts require a
// writable canonical publication, even when the test does not execute its payload.
inline void attach_compact_frame_carrier(GstBuffer* buffer) {
  using namespace simaai::gst;
  const auto bytes = gst_buffer_get_size(buffer);
  if (bytes == 0 || bytes > G_MAXINT)
    throw std::runtime_error("compact fixture requires a bounded nonempty carrier");
  TensorBufferPublishContract contract;
  contract.stage_key = "compact-frame-fixture";
  contract.physical_outputs.push_back({0, bytes, "arena"});
  TensorBufferPublishLogicalOutput tensor;
  tensor.logical_index = tensor.physical_index = tensor.memory_index = 0;
  tensor.backend_output_index = tensor.route_slot = 0;
  tensor.logical_name = tensor.backend_name = "bytes";
  tensor.segment_name = "arena";
  tensor.size_bytes = bytes;
  tensor.dtype = SIMA_TENSOR_SET_DTYPE_UINT8_V1;
  tensor.layout = SIMA_TENSOR_SET_LAYOUT_HW_V1;
  tensor.shape = {1, static_cast<std::int64_t>(bytes)};
  tensor.stride_bytes = {static_cast<std::int64_t>(bytes), 1};
  contract.logical_outputs.push_back(std::move(tensor));
  TensorBufferPreparedMetaTemplate prepared;
  std::string error;
  if (!tensor_buffer_prepare_meta_template_from_contract(contract, &prepared, &error))
    throw std::runtime_error(error);
  const auto publication = tensor_buffer_freeze_publication(prepared, &error);
  if (!publication || !tensor_buffer_attach_publication(buffer, publication, &error))
    throw std::runtime_error(error);
}
} // namespace simaai::neat::test_support

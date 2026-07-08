#pragma once

#include "simaai/neat/pcie/Model.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gst/gst.h>

namespace simaai::neat::pcie::internal {

struct TensorMetaSpan {
  const Tensor* tensor = nullptr;
  std::int64_t byte_offset = 0;
  std::size_t size_bytes = 0;
};

struct PreparedPayload {
  std::shared_ptr<void> owner;
  std::uint8_t* data = nullptr;
  std::size_t size_bytes = 0;
  GstMemoryFlags flags = static_cast<GstMemoryFlags>(0);
  std::vector<TensorMetaSpan> spans;
};

std::size_t tensor_dtype_bytes(TensorDType dtype);
std::vector<std::int64_t> contiguous_tensor_strides(const std::vector<std::int64_t>& shape,
                                                    std::size_t elem_size);
PreparedPayload prepare_tensor_payload(const TensorList& tensors);

} // namespace simaai::neat::pcie::internal

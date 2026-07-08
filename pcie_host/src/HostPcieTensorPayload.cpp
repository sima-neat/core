#include "HostPcieTensorPayload.h"

#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>

namespace simaai::neat::pcie::internal {
namespace {

const std::uint8_t* tensor_data(const Tensor& tensor, std::size_t* size_out) {
  if (!tensor.data) {
    throw std::runtime_error("tensor has no data pointer");
  }
  if (tensor.byte_offset < 0 || static_cast<std::size_t>(tensor.byte_offset) > tensor.size_bytes) {
    throw std::runtime_error("tensor byte offset is outside tensor payload");
  }
  const auto offset = static_cast<std::size_t>(tensor.byte_offset);
  if (size_out) {
    *size_out = tensor.size_bytes - offset;
  }
  return static_cast<const std::uint8_t*>(tensor.data) + offset;
}

std::size_t dense_size_bytes(const Tensor& tensor) {
  const std::size_t elem = tensor_dtype_bytes(tensor.dtype);
  if (elem == 0) {
    return 0;
  }
  std::size_t bytes = elem;
  for (const auto dim : tensor.shape) {
    if (dim < 0) {
      return 0;
    }
    const auto udim = static_cast<std::size_t>(dim);
    if (udim != 0 && bytes > std::numeric_limits<std::size_t>::max() / udim) {
      return 0;
    }
    bytes *= udim;
  }
  return bytes;
}

bool copy_dense_rows(const std::uint8_t* src, const std::size_t src_size,
                     const std::vector<std::int64_t>& shape,
                     const std::vector<std::int64_t>& strides, const std::size_t elem_size,
                     const std::size_t dim, std::uint8_t** dst) {
  if (dim + 1U == shape.size()) {
    const auto elements = static_cast<std::size_t>(shape[dim]);
    if (strides[dim] < static_cast<std::int64_t>(elem_size)) {
      return false;
    }
    for (std::size_t i = 0; i < elements; ++i) {
      const auto src_offset = static_cast<std::size_t>(static_cast<std::int64_t>(i) * strides[dim]);
      if (src_offset + elem_size > src_size) {
        return false;
      }
      std::memcpy(*dst, src + src_offset, elem_size);
      *dst += elem_size;
    }
    return true;
  }

  const auto count = static_cast<std::size_t>(shape[dim]);
  for (std::size_t i = 0; i < count; ++i) {
    const auto src_offset = static_cast<std::size_t>(static_cast<std::int64_t>(i) * strides[dim]);
    if (src_offset > src_size) {
      return false;
    }
    if (!copy_dense_rows(src + src_offset, src_size - src_offset, shape, strides, elem_size,
                         dim + 1U, dst)) {
      return false;
    }
  }
  return true;
}

std::vector<std::uint8_t> copy_dense_tensor_payload(const Tensor& tensor) {
  const std::size_t elem = tensor_dtype_bytes(tensor.dtype);
  const std::size_t bytes = dense_size_bytes(tensor);
  if (bytes == 0) {
    throw std::runtime_error("dense tensor has unknown or empty byte size");
  }

  std::size_t mapped_size = 0;
  const auto* src = tensor_data(tensor, &mapped_size);
  if (mapped_size < bytes && tensor.strides_bytes.empty()) {
    throw std::runtime_error("dense tensor mapping is smaller than required payload");
  }

  std::vector<std::uint8_t> out(bytes);
  if (tensor.strides_bytes.empty()) {
    std::memcpy(out.data(), src, bytes);
    return out;
  }

  std::vector<std::int64_t> strides = tensor.strides_bytes;
  if (strides.size() != tensor.shape.size()) {
    throw std::runtime_error("dense tensor strides must match shape rank");
  }
  std::uint8_t* dst = out.data();
  if (!copy_dense_rows(src, mapped_size, tensor.shape, strides, elem, 0U, &dst)) {
    throw std::runtime_error("dense tensor strided copy failed");
  }
  return out;
}

std::vector<std::uint8_t> copy_raw_tensor_payload(const Tensor& tensor) {
  std::size_t mapped_size = 0;
  const auto* src = tensor_data(tensor, &mapped_size);
  return std::vector<std::uint8_t>(src, src + mapped_size);
}

std::size_t tensor_payload_size(const Tensor& tensor) {
  if (!tensor.planes.empty()) {
    if (tensor.byte_offset < 0 ||
        static_cast<std::size_t>(tensor.byte_offset) > tensor.size_bytes) {
      return 0;
    }
    return tensor.size_bytes - static_cast<std::size_t>(tensor.byte_offset);
  }
  if (!tensor.shape.empty()) {
    return dense_size_bytes(tensor);
  }
  if (tensor.byte_offset < 0 || static_cast<std::size_t>(tensor.byte_offset) > tensor.size_bytes) {
    return 0;
  }
  return tensor.size_bytes - static_cast<std::size_t>(tensor.byte_offset);
}

std::vector<std::uint8_t> copy_tensor_payload(const Tensor& tensor) {
  if (!tensor.planes.empty()) {
    return copy_raw_tensor_payload(tensor);
  }
  if (!tensor.shape.empty()) {
    return copy_dense_tensor_payload(tensor);
  }
  return copy_raw_tensor_payload(tensor);
}

std::vector<std::uint8_t> pack_tensor_payloads(const TensorList& tensors) {
  std::vector<std::uint8_t> out;
  std::size_t total = 0;
  for (const auto& tensor : tensors) {
    total += tensor_payload_size(tensor);
  }
  out.reserve(total);
  for (const auto& tensor : tensors) {
    std::vector<std::uint8_t> bytes = copy_tensor_payload(tensor);
    out.insert(out.end(), bytes.begin(), bytes.end());
  }
  return out;
}

bool is_direct_contiguous_payload(const Tensor& tensor, const std::uint8_t** data_out,
                                  std::size_t* size_out) {
  if (!tensor.planes.empty()) {
    return false;
  }

  std::size_t mapped_size = 0;
  const std::uint8_t* data = nullptr;
  try {
    data = tensor_data(tensor, &mapped_size);
  } catch (const std::runtime_error&) {
    return false;
  }

  const std::size_t payload_size = tensor_payload_size(tensor);
  if (payload_size == 0 || mapped_size < payload_size) {
    return false;
  }

  if (!tensor.shape.empty()) {
    const std::size_t elem = tensor_dtype_bytes(tensor.dtype);
    if (elem == 0) {
      return false;
    }
    if (!tensor.strides_bytes.empty()) {
      if (tensor.strides_bytes.size() != tensor.shape.size()) {
        return false;
      }
      if (tensor.strides_bytes != contiguous_tensor_strides(tensor.shape, elem)) {
        return false;
      }
    }
  } else if (!tensor.strides_bytes.empty()) {
    return false;
  }

  if (data_out) {
    *data_out = data;
  }
  if (size_out) {
    *size_out = payload_size;
  }
  return true;
}

std::vector<TensorMetaSpan> packed_spans_for_tensors(const TensorList& tensors) {
  std::vector<TensorMetaSpan> spans;
  spans.reserve(tensors.size());
  std::int64_t running_offset = 0;
  for (const auto& tensor : tensors) {
    const std::size_t size_bytes = tensor_payload_size(tensor);
    spans.push_back(TensorMetaSpan{
        .tensor = &tensor,
        .byte_offset = running_offset,
        .size_bytes = size_bytes,
    });
    if (size_bytes >
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max() - running_offset)) {
      throw std::runtime_error("tensor-set payload is too large");
    }
    running_offset += static_cast<std::int64_t>(size_bytes);
  }
  return spans;
}

std::optional<PreparedPayload> prepare_single_contiguous_payload(const TensorList& tensors) {
  if (tensors.size() != 1U) {
    return std::nullopt;
  }
  const Tensor& tensor = tensors.front();
  if (!tensor.owner) {
    return std::nullopt;
  }

  const std::uint8_t* data = nullptr;
  std::size_t size_bytes = 0;
  if (!is_direct_contiguous_payload(tensor, &data, &size_bytes)) {
    return std::nullopt;
  }

  PreparedPayload payload;
  payload.owner = tensor.owner;
  payload.data = const_cast<std::uint8_t*>(data);
  payload.size_bytes = size_bytes;
  payload.flags = static_cast<GstMemoryFlags>(GST_MEMORY_FLAG_READONLY);
  payload.spans.push_back(TensorMetaSpan{
      .tensor = &tensor,
      .byte_offset = 0,
      .size_bytes = size_bytes,
  });
  return payload;
}

std::optional<PreparedPayload> prepare_shared_packed_payload(const TensorList& tensors) {
  if (tensors.size() <= 1U) {
    return std::nullopt;
  }

  const auto& owner = tensors.front().owner;
  if (!owner || !tensors.front().data) {
    return std::nullopt;
  }
  const auto* base = static_cast<const std::uint8_t*>(tensors.front().data);

  std::vector<TensorMetaSpan> spans;
  spans.reserve(tensors.size());
  std::int64_t running_offset = 0;
  for (const auto& tensor : tensors) {
    if (tensor.owner != owner || !tensor.data ||
        static_cast<const std::uint8_t*>(tensor.data) != base || tensor.byte_offset < 0 ||
        tensor.byte_offset != running_offset) {
      return std::nullopt;
    }

    const std::uint8_t* payload_data = nullptr;
    std::size_t size_bytes = 0;
    if (!is_direct_contiguous_payload(tensor, &payload_data, &size_bytes) ||
        payload_data != base + running_offset) {
      return std::nullopt;
    }

    spans.push_back(TensorMetaSpan{
        .tensor = &tensor,
        .byte_offset = running_offset,
        .size_bytes = size_bytes,
    });
    if (size_bytes >
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max() - running_offset)) {
      return std::nullopt;
    }
    running_offset += static_cast<std::int64_t>(size_bytes);
  }
  if (running_offset <= 0) {
    return std::nullopt;
  }

  PreparedPayload payload;
  payload.owner = owner;
  payload.data = const_cast<std::uint8_t*>(base);
  payload.size_bytes = static_cast<std::size_t>(running_offset);
  payload.flags = static_cast<GstMemoryFlags>(GST_MEMORY_FLAG_READONLY);
  payload.spans = std::move(spans);
  return payload;
}

PreparedPayload prepare_staging_payload(const TensorList& tensors) {
  std::vector<std::uint8_t> bytes = pack_tensor_payloads(tensors);
  if (bytes.empty()) {
    throw std::runtime_error("PCIe payload is empty");
  }

  auto owner = std::make_shared<std::vector<std::uint8_t>>(std::move(bytes));
  PreparedPayload payload;
  payload.data = owner->data();
  payload.size_bytes = owner->size();
  payload.owner = std::move(owner);
  payload.spans = packed_spans_for_tensors(tensors);
  return payload;
}

} // namespace

std::size_t tensor_dtype_bytes(const TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt8:
  case TensorDType::Int8:
    return 1;
  case TensorDType::UInt16:
  case TensorDType::Int16:
  case TensorDType::BFloat16:
    return 2;
  case TensorDType::Int32:
  case TensorDType::Float32:
    return 4;
  case TensorDType::Float64:
    return 8;
  }
  return 0;
}

std::vector<std::int64_t> contiguous_tensor_strides(const std::vector<std::int64_t>& shape,
                                                    const std::size_t elem_size) {
  std::vector<std::int64_t> strides(shape.size(), 0);
  std::int64_t stride = static_cast<std::int64_t>(elem_size);
  for (std::size_t index = shape.size(); index > 0; --index) {
    const std::size_t dim = index - 1;
    strides[dim] = stride;
    stride *= shape[dim];
  }
  return strides;
}

PreparedPayload prepare_tensor_payload(const TensorList& tensors) {
  if (std::optional<PreparedPayload> payload = prepare_single_contiguous_payload(tensors)) {
    return std::move(*payload);
  }
  if (std::optional<PreparedPayload> payload = prepare_shared_packed_payload(tensors)) {
    return std::move(*payload);
  }
  return prepare_staging_payload(tensors);
}

} // namespace simaai::neat::pcie::internal

#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <ev/ev_tensor_abi.h>
#include <simaai/scheduler/client/tensor.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace simaai::neat::pipeline_internal::sima {

// Only the legacy boundary interprets string/wire pairs. In particular, wire 1
// cannot distinguish BF16 from INT16; transport byte types are not annotations.
inline std::optional<::simaai::scheduler::client::DType>
scalar_from_model_descriptor(std::string_view meaning, std::uint32_t wire) {
  using DType = ::simaai::scheduler::client::DType;
  const auto matches = [meaning](std::string_view expected) {
    if (meaning.size() != expected.size())
      return false;
    for (std::size_t i = 0; i < meaning.size(); ++i) {
      const char c = meaning[i];
      if ((c >= 'a' && c <= 'z' ? char(c - 'a' + 'A') : c) != expected[i])
        return false;
    }
    return true;
  };
  if (matches("BF16") || matches("BFLOAT16") || matches("EVXX_BFLOAT16")) {
    if (wire == SIMA_EV_DTYPE_BF16)
      return DType::BFloat16;
  } else if (matches("INT16") || matches("EVXX_INT16")) {
    if (wire == SIMA_EV_DTYPE_INT16)
      return DType::Int16;
  } else if (matches("FP16") || matches("FLOAT16") || matches("EVXX_FLOAT16")) {
    if (wire == SIMA_EV_DTYPE_FP16)
      return DType::Float16;
  } else if (matches("FP32") || matches("FLOAT32") || matches("EVXX_FLOAT32")) {
    if (wire == SIMA_EV_DTYPE_FP32)
      return DType::Float32;
  } else if (matches("INT8") || matches("EVXX_INT8")) {
    if (wire == SIMA_EV_DTYPE_INT8)
      return DType::Int8;
  } else if (matches("UINT8") || matches("EVXX_UINT8")) {
    if (wire == SIMA_EV_DTYPE_INT8)
      return DType::UInt8;
  } else if (matches("UINT16") || matches("EVXX_UINT16")) {
    if (wire == SIMA_EV_DTYPE_INT16)
      return DType::UInt16;
  } else if (matches("INT32") || matches("EVXX_INT32")) {
    if (wire == SIMA_EV_DTYPE_INT32)
      return DType::Int32;
  } else if (meaning.empty()) {
    switch (wire) {
    case SIMA_EV_DTYPE_INT8:
      return DType::Int8;
    case SIMA_EV_DTYPE_INT32:
      return DType::Int32;
    case SIMA_EV_DTYPE_FP32:
      return DType::Float32;
    case SIMA_EV_DTYPE_FP16:
      return DType::Float16;
    }
  }
  return std::nullopt;
}

// Cold Core-owned numerical authority. Copy/reordering retains scalar meaning
// with geometry; neither publication dtype nor physical byte carriers can
// rewrite it. The wire descriptor and its public ABI remain unchanged.
class AuthoredTensorDescriptor {
public:
  using DType = ::simaai::scheduler::client::DType;
  AuthoredTensorDescriptor(sima_ev_tensor_desc descriptor, std::string_view meaning)
      : descriptor_(descriptor), scalar_(checked_scalar(descriptor, meaning)) {}
  const sima_ev_tensor_desc& descriptor() const noexcept {
    return descriptor_;
  }
  DType scalar() const noexcept {
    return scalar_;
  }
  const char* scalar_token() const noexcept {
    switch (scalar_) {
    case DType::Float32:
      return "FP32";
    case DType::BFloat16:
      return "BF16";
    case DType::Int8:
      return "INT8";
    case DType::UInt8:
      return "UINT8";
    case DType::Int16:
      return "INT16";
    case DType::Int32:
      return "INT32";
    case DType::Float16:
      return "FP16";
    case DType::UInt16:
      return "UINT16";
    }
    return ""; // Unreachable for checked construction.
  }
  // Existing cold geometry helpers consume the wire projection read-only.
  operator const sima_ev_tensor_desc&() const noexcept {
    return descriptor_;
  }
  [[nodiscard]] AuthoredTensorDescriptor with_geometry(sima_ev_tensor_desc descriptor) const {
    if (descriptor.dtype != descriptor_.dtype)
      throw std::invalid_argument("geometry replacement cannot change numerical scalar encoding");
    auto result = *this;
    result.descriptor_ = descriptor;
    return result;
  }
  void set_storage_address(std::uint64_t address) noexcept {
    descriptor_.storage.addr = address;
  }

private:
  static DType checked_scalar(const sima_ev_tensor_desc& descriptor, std::string_view meaning) {
    const auto scalar = scalar_from_model_descriptor(meaning, descriptor.dtype);
    if (meaning.empty() || !scalar)
      throw std::invalid_argument("authored tensor requires an explicit compatible scalar: '" +
                                  std::string(meaning) +
                                  "', wire=" + std::to_string(descriptor.dtype));
    return *scalar;
  }
  sima_ev_tensor_desc descriptor_;
  DType scalar_;
};
} // namespace simaai::neat::pipeline_internal::sima

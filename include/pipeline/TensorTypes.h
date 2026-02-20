/**
 * @file
 * @ingroup tensors
 * @brief Tensor dtype/layout and minimal DLPack-like structs.
 */
#pragma once

#include <cstdint>

namespace simaai::neat {

// Minimal DLPack-like structs (CPU only). This keeps a zero-copy bridge path
// for future bindings without adding a hard dependency today.
namespace dlpack {

enum class DLDeviceType : int {
  kDLCPU = 1,
};

struct DLDevice {
  DLDeviceType device_type;
  int device_id;
};

enum DLDataTypeCode : uint8_t {
  kDLInt = 0,
  kDLUInt = 1,
  kDLFloat = 2,
};

struct DLDataType {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

struct DLTensor {
  void* data;
  DLDevice device;
  int ndim;
  DLDataType dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
};

struct DLManagedTensor {
  DLTensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(DLManagedTensor* self);
};

} // namespace dlpack

enum class TensorDType {
  UInt8,
  Int8,
  UInt16,
  Int16,
  Int32,
  BFloat16,
  Float32,
  Float64,
};

enum class TensorLayout {
  Unknown = 0,
  HWC,
  CHW,
  HW,
  Planar,
};

} // namespace simaai::neat

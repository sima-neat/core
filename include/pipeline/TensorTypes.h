/**
 * @file
 * @ingroup tensors
 * @brief Tensor dtype, axis semantics, layout token, and minimal DLPack-like structs.
 *
 * This header defines the small enum vocabulary every tensor consumer needs: data type
 * (`TensorDType`), per-axis semantic tag (`TensorAxisSemantic`, the long-term direction),
 * legacy layout token (`TensorLayout`, transitional), plus minimal DLPack-shaped structs
 * for zero-copy bridges to NumPy/PyTorch via the standard DLPack interchange format.
 *
 * @see TensorCore.h for the Tensor struct that uses these
 * @see "TensorAxisSemantic" rationale (§0.10 of the design deep dive)
 */
#pragma once

#include <cstdint>

namespace simaai::neat {

/**
 * @brief Minimal DLPack-shaped structs for zero-copy interop with NumPy/PyTorch.
 *
 * Mirrors the standard DLPack interchange types without taking a dependency on the dlpack
 * header itself. Used by the Python binding layer (`pyneat`) to expose tensors via the
 * `__dlpack__` protocol. CPU-only today; device variants can be added without breaking the ABI.
 * @ingroup tensors
 */
namespace dlpack {

/// DLPack device type. Currently CPU-only; GPU/accelerator variants can be added later.
enum class DLDeviceType : int {
  kDLCPU = 1,
};

/// DLPack device descriptor: type + numeric ID.
struct DLDevice {
  DLDeviceType device_type; ///< Device family; one of the DLPack DeviceType codes.
  int device_id;            ///< Device-local index; meaningful only for accelerators with multiple instances.
};

/// DLPack data-type category (signed int / unsigned int / floating-point).
enum DLDataTypeCode : uint8_t {
  kDLInt = 0,
  kDLUInt = 1,
  kDLFloat = 2,
};

/// DLPack scalar dtype: code + bit width + SIMD lane count.
struct DLDataType {
  uint8_t code;   ///< One of `DLDataTypeCode`.
  uint8_t bits;   ///< Number of bits per scalar (e.g., 8, 16, 32, 64).
  uint16_t lanes; ///< SIMD lane count (1 for scalar values).
};

/// DLPack non-owning tensor view: pointer + shape + strides + dtype + device.
struct DLTensor {
  void* data;          ///< Raw data pointer (aligned per DLPack spec).
  DLDevice device;     ///< Where the data lives.
  int ndim;            ///< Number of dimensions.
  DLDataType dtype;    ///< Element dtype.
  int64_t* shape;      ///< Per-dimension sizes (length `ndim`).
  int64_t* strides;    ///< Per-dimension strides in elements (may be null for compact row-major).
  uint64_t byte_offset; ///< Offset from `data` in bytes to the first element.
};

/// DLPack managed tensor: owning wrapper that calls `deleter` on destruction.
struct DLManagedTensor {
  DLTensor dl_tensor;                              ///< The non-owning view.
  void* manager_ctx;                               ///< Opaque handle for the producer.
  void (*deleter)(DLManagedTensor* self);          ///< Function to destroy the managed view.
};

} // namespace dlpack

/**
 * @brief Element data type of a tensor.
 *
 * Note: floating-point variants include `BFloat16` (16-bit "brain float", same dynamic range
 * as Float32), `Float32`, and `Float64`. `Float16` (IEEE half) is intentionally omitted —
 * the framework targets BF16 for the MLA's floating-point path; FP16 has no first-class path.
 * @ingroup tensors
 */
enum class TensorDType {
  UInt8,    ///< 8-bit unsigned integer (typical pixel values).
  Int8,     ///< 8-bit signed integer (typical quantized weights/activations).
  UInt16,   ///< 16-bit unsigned integer.
  Int16,    ///< 16-bit signed integer.
  Int32,    ///< 32-bit signed integer.
  BFloat16, ///< 16-bit "brain float" (1 sign + 8 exponent + 7 mantissa bits) — MLA's FP path.
  Float32,  ///< IEEE 754 single-precision (the user-facing default).
  Float64,  ///< IEEE 754 double-precision.
};

/**
 * @brief Per-axis semantic tag — the long-term layout vocabulary.
 *
 * Each axis of a tensor is tagged with a role: batch (`N`), depth (`D`), height (`H`),
 * width (`W`), or channel (`C`). Together with `shape` and `strides`, this is the
 * authoritative description of tensor layout and supersedes the coarse `TensorLayout`
 * token for generic 3D/4D/5D tensors.
 * @ingroup tensors
 */
enum class TensorAxisSemantic : uint8_t {
  Unknown = 0, ///< Axis role not annotated.
  N,           ///< Batch axis.
  D,           ///< Depth axis (5D tensors: batch + depth + spatial).
  H,           ///< Height (spatial).
  W,           ///< Width (spatial).
  C,           ///< Channel axis.
};

/**
 * @brief Transition-only coarse layout token (HWC / CHW / HW).
 *
 * This is **not** the long-term semantic truth for generic tensors. Internal tensor semantics
 * should move toward `shape + strides + explicit axis_semantics`, with layout tokens remaining
 * only at boundary/compatibility surfaces during the migration. Prefer `TensorAxisSemantic` for
 * new code.
 * @see TensorAxisSemantic
 * @ingroup tensors
 */
enum class TensorLayout {
  Unknown = 0,
  HWC, ///< Height × Width × Channels (image-natural layout).
  CHW, ///< Channels × Height × Width (PyTorch-natural layout).
  HW,  ///< Height × Width (single-channel / grayscale).
};

} // namespace simaai::neat

/**
 * @file
 * @ingroup pipeline
 * @brief Mirrored EV-side tensor ABI used to communicate with EV74 kernels.
 *
 * This header captures a candidate common ABI for EV kernels so host and
 * firmware can converge on a single tensor/storage contract instead of
 * per-graph ad hoc structs. It is currently a *draft proposal* — not yet wired
 * into the live EV runtime — but it documents the shape of the per-op config
 * blocks (tess, detess, quantize/dequantize, detessdequant, preproc) and the
 * shared tensor/storage descriptor types they all reference.
 *
 * The header is C-callable so firmware-side code can include it directly. All
 * values are POD; layout is fixed by the structs themselves.
 */
#pragma once

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif

/*
 * Draft proposal only.
 *
 * This header is not wired into the live EV runtime yet. It captures a
 * candidate common ABI for EV kernels so host and firmware can converge on one
 * tensor/storage contract instead of per-graph ad hoc structs.
 */

/// @brief ABI-wide capacity constants and version identifier.
enum {
  SIMA_EV_ABI_VERSION_1 = 1, ///< Current ABI version tag (carried in `sima_ev_abi_header`).
  SIMA_EV_MAX_RANK = 8,      ///< Maximum tensor rank supported by descriptors.
  SIMA_EV_MAX_IO = 32,       ///< Maximum number of inputs/outputs per multi-IO kernel.
};

/// @brief Operation-type tag carried in `sima_ev_abi_header::op_type`.
enum sima_ev_op_type {
  SIMA_EV_OP_INVALID = 0,       ///< Invalid/unset.
  SIMA_EV_OP_TESS = 1,          ///< Tessellate.
  SIMA_EV_OP_DETESS = 2,        ///< Detessellate.
  SIMA_EV_OP_QUANTIZE = 3,      ///< Quantize.
  SIMA_EV_OP_DEQUANTIZE = 4,    ///< Dequantize.
  SIMA_EV_OP_QUANTTESS = 5,     ///< Fused quantize + tessellate.
  SIMA_EV_OP_DETESSDEQUANT = 6, ///< Fused detessellate + dequantize.
  SIMA_EV_OP_PREPROC = 7,       ///< Image preprocessing.
  SIMA_EV_OP_CAST = 8,          ///< Element-type cast (e.g., BF16 ↔ FP32).
};

/// @brief Address-space selector for `sima_ev_storage_desc::addr`.
enum sima_ev_addr_space {
  SIMA_EV_ADDR_BUS = 0,  ///< Bus address (canonical long-term).
  SIMA_EV_ADDR_PHYS = 1, ///< Physical address (transitional).
};

/// @brief Element types the ABI can describe.
enum sima_ev_dtype {
  SIMA_EV_DTYPE_INVALID = 0,
  SIMA_EV_DTYPE_INT8 = 1,
  SIMA_EV_DTYPE_INT16 = 2,
  SIMA_EV_DTYPE_INT32 = 3,
  SIMA_EV_DTYPE_BF16 = 4,
  SIMA_EV_DTYPE_FP16 = 5,
  SIMA_EV_DTYPE_FP32 = 6,
  SIMA_EV_DTYPE_UINT8 = 7,
  SIMA_EV_DTYPE_UINT16 = 8,
  SIMA_EV_DTYPE_UINT32 = 9,
};

/// @brief Layout family carried in `sima_ev_tensor_desc::logical_kind`/`physical_kind`.
enum sima_ev_layout_kind {
  SIMA_EV_LAYOUT_STRIDED = 0, ///< Standard rank/sizes/strides view.
  SIMA_EV_LAYOUT_TILED = 1,   ///< Tiled view with explicit tile geometry.
  SIMA_EV_LAYOUT_OPAQUE = 2,  ///< Opaque blob (kernel-private layout).
};

/// @brief Optional per-axis semantic tag (advisory metadata).
enum sima_ev_axis_semantic {
  SIMA_EV_AXIS_UNKNOWN = 0,
  SIMA_EV_AXIS_N = 1, ///< Batch.
  SIMA_EV_AXIS_C = 2, ///< Channel.
  SIMA_EV_AXIS_D = 3, ///< Depth.
  SIMA_EV_AXIS_H = 4, ///< Height.
  SIMA_EV_AXIS_W = 5, ///< Width.
};

/// @brief Quantization scheme tag for `sima_ev_quant_desc::scheme`.
enum sima_ev_quant_scheme {
  SIMA_EV_QUANT_NONE = 0,              ///< No quantization (other fields zeroed).
  SIMA_EV_QUANT_PER_TENSOR_AFFINE = 1, ///< Single (scale, zero-point) for the whole tensor.
  SIMA_EV_QUANT_PER_AXIS_AFFINE = 2,   ///< Per-axis (scale, zero-point) arrays.
};

/// @brief Rounding mode used by quantize/cast kernels.
enum sima_ev_round_mode {
  SIMA_EV_ROUND_DEFAULT = 0,
  SIMA_EV_ROUND_TO_ZERO = 1,
  SIMA_EV_ROUND_TO_EVEN = 2,
  SIMA_EV_ROUND_TO_POS_INF = 3,
  SIMA_EV_ROUND_TO_NEG_INF = 4,
};

/**
 * @brief Common header at the start of every per-op config block.
 *
 * Identifies the ABI version, op type, and IO arity so firmware can dispatch
 * payloads generically before touching op-specific fields.
 */
struct sima_ev_abi_header {
  uint16_t abi_version; ///< One of `SIMA_EV_ABI_VERSION_*`.
  uint16_t op_type;     ///< `sima_ev_op_type` value.
  uint32_t flags;       ///< Op-specific flags.
  uint32_t num_inputs;  ///< Number of input tensors in this dispatch.
  uint32_t num_outputs; ///< Number of output tensors in this dispatch.
  uint32_t reserved[2]; ///< Reserved for future use; zero on write.
};

/*
 * Device-visible storage. For long-term consistency, BUS should be the
 * canonical address space, but the field remains explicit during migration.
 */
/**
 * @brief Device-visible storage backing a tensor descriptor.
 *
 * For long-term consistency, BUS should be the canonical address space; the
 * `addr_space` field remains explicit during migration so firmware can detect
 * legacy callers.
 */
struct sima_ev_storage_desc {
  uint64_t addr;                ///< Bus or physical address (per `addr_space`).
  uint64_t nbytes;              ///< Size of the backing region in bytes.
  uint32_t addr_space;          ///< `sima_ev_addr_space` selector.
  uint32_t alignment_bytes;     ///< Required alignment of `addr`.
  int64_t storage_offset_bytes; ///< Signed offset from `addr` to logical origin.
  uint64_t reserved0;           ///< Reserved for future use; zero on write.
};

/*
 * Logical tensor view, intentionally close to PyTorch semantics: rank, sizes,
 * and byte strides are authoritative. Axis semantics are advisory metadata for
 * validation, debugging, and kernels that want explicit N/C/D/H/W naming.
 */
/**
 * @brief Logical strided tensor view (rank + sizes + byte strides).
 *
 * Intentionally close to PyTorch semantics: rank, sizes, and byte strides are
 * authoritative. `axis_semantics` is advisory metadata for validation,
 * debugging, and kernels that want explicit N/C/D/H/W naming.
 */
struct sima_ev_strided_desc {
  uint32_t rank;                            ///< Number of valid entries in sizes/strides.
  uint8_t axis_semantics[SIMA_EV_MAX_RANK]; ///< Per-axis `sima_ev_axis_semantic` tags.
  uint8_t reserved0[4];                     ///< Padding to align the int64 arrays that follow.
  int64_t sizes[SIMA_EV_MAX_RANK];          ///< Per-axis size, in elements.
  int64_t strides_bytes[SIMA_EV_MAX_RANK];  ///< Per-axis stride, in bytes.
};

/*
 * Physical tiled view. This is separate from the logical view because a tiled
 * EV buffer is not just a dense tensor with different strides: it also has tile
 * shape, tile traversal order, inner blocking, and per-tile alignment rules.
 */
/**
 * @brief Physical tiled tensor view.
 *
 * Distinct from the logical view because a tiled EV buffer carries tile shape,
 * tile traversal order, inner blocking, and per-tile alignment rules in
 * addition to size/stride.
 */
struct sima_ev_tiled_desc {
  uint32_t rank;                             ///< Number of valid axes.
  uint8_t axis_semantics[SIMA_EV_MAX_RANK];  ///< Per-axis semantic tags.
  uint8_t tile_axis_order[SIMA_EV_MAX_RANK]; ///< Tile traversal order across axes.
  uint8_t inner_block[SIMA_EV_MAX_RANK];     ///< Per-axis inner blocking factor.
  uint8_t reserved0[4];                      ///< Padding to align the int64 arrays that follow.
  int64_t tile_sizes[SIMA_EV_MAX_RANK];      ///< Per-axis tile size, in elements.
  uint32_t tile_align_bytes;                 ///< Required per-tile alignment, in bytes.
  uint32_t flags;                            ///< Tiling-specific flags.
};

/*
 * Optional quantization metadata. Kernels that do not use quantization should
 * set scheme=SIMA_EV_QUANT_NONE and leave the remaining fields zeroed.
 */
/**
 * @brief Optional quantization metadata.
 *
 * Kernels that do not use quantization should set `scheme = SIMA_EV_QUANT_NONE`
 * and leave the remaining fields zeroed. Per-axis schemes use `scale_addr` /
 * `zero_point_addr` arrays of length `param_count`.
 */
struct sima_ev_quant_desc {
  uint32_t scheme;          ///< `sima_ev_quant_scheme` value.
  uint32_t quant_axis;      ///< Axis along which quantization is per-axis.
  float scale;              ///< Per-tensor scale (when scheme=PER_TENSOR_AFFINE).
  int32_t zero_point;       ///< Per-tensor zero-point.
  uint64_t scale_addr;      ///< Bus address of per-axis scale array (if used).
  uint64_t zero_point_addr; ///< Bus address of per-axis zero-point array (if used).
  uint32_t param_count;     ///< Length of per-axis arrays.
  uint32_t reserved0;       ///< Reserved for future use; zero on write.
};

/**
 * @brief Full per-tensor descriptor: dtype, layout, storage, optional quant.
 *
 * Carries both the logical (always strided) and physical (strided or tiled)
 * views of a tensor along with its backing storage and optional quantization
 * metadata. The `physical` union is selected by `physical_kind`.
 */
struct sima_ev_tensor_desc {
  uint32_t dtype;                      ///< `sima_ev_dtype` value.
  uint32_t logical_kind;               ///< Always `SIMA_EV_LAYOUT_STRIDED` today.
  uint32_t physical_kind;              ///< `sima_ev_layout_kind` selecting `physical`.
  uint32_t flags;                      ///< Tensor-specific flags.
  struct sima_ev_storage_desc storage; ///< Backing storage region.
  struct sima_ev_strided_desc logical; ///< Logical view (rank/sizes/strides).
  union {
    struct sima_ev_strided_desc strided; ///< Physical view, when STRIDED.
    struct sima_ev_tiled_desc tiled;     ///< Physical view, when TILED.
  } physical;                            ///< Physical view (selected by physical_kind).
  struct sima_ev_quant_desc quant;       ///< Optional quantization metadata.
};

/*
 * Single-input/single-output transforms.
 */
/// @brief Tess (tessellate) op config: 1 input, 1 output.
struct sima_ev_tess_config_v1 {
  struct sima_ev_abi_header hdr;     ///< ABI header (op type, version, IO arity).
  struct sima_ev_tensor_desc input;  ///< Input tensor descriptor.
  struct sima_ev_tensor_desc output; ///< Output tensor descriptor.
};

/// @brief Detess (detessellate) op config: 1 input, 1 output.
struct sima_ev_detess_config_v1 {
  struct sima_ev_abi_header hdr;     ///< ABI header (op type, version, IO arity).
  struct sima_ev_tensor_desc input;  ///< Input tensor descriptor.
  struct sima_ev_tensor_desc output; ///< Output tensor descriptor.
};

/// @brief Quantize op config: 1 input, 1 output, plus rounding/saturation modes.
struct sima_ev_quantize_config_v1 {
  struct sima_ev_abi_header hdr;     ///< ABI header (op type, version, IO arity).
  struct sima_ev_tensor_desc input;  ///< Input tensor descriptor.
  struct sima_ev_tensor_desc output; ///< Output tensor descriptor.
  uint32_t round_mode;               ///< `sima_ev_round_mode` value.
  uint32_t saturate_mode;            ///< Saturation mode (op-specific encoding).
  uint32_t reserved[2];              ///< Reserved for future use; zero on write.
};

/// @brief Dequantize op config: 1 input, 1 output.
struct sima_ev_dequantize_config_v1 {
  struct sima_ev_abi_header hdr;     ///< ABI header (op type, version, IO arity).
  struct sima_ev_tensor_desc input;  ///< Input tensor descriptor.
  struct sima_ev_tensor_desc output; ///< Output tensor descriptor.
  uint32_t reserved[4];              ///< Reserved for future use; zero on write.
};

/*
 * Shared multi-IO shell for kernels such as detessdequant that naturally operate
 * on many heads/tensors per dispatch.
 */
/**
 * @brief Multi-tensor IO shell used by kernels like detessdequant.
 *
 * Holds up to `SIMA_EV_MAX_IO` tensor descriptors with an explicit `count`.
 */
struct sima_ev_tensor_list_v1 {
  uint32_t count;                                  ///< Number of valid entries in `desc`.
  uint32_t reserved[3];                            ///< Reserved for future use; zero on write.
  struct sima_ev_tensor_desc desc[SIMA_EV_MAX_IO]; ///< Tensor descriptors.
};

/// @brief Per-output detessdequant parameters.
struct sima_ev_detessdequant_params_v1 {
  uint32_t fp16_output; ///< Non-zero to emit FP16 instead of FP32 (where supported).
  uint32_t round_mode;  ///< `sima_ev_round_mode` value.
  uint32_t reserved[2]; ///< Reserved for future use; zero on write.
};

/// @brief DetessDequant op config: many inputs, many outputs, per-output params.
struct sima_ev_detessdequant_config_v1 {
  struct sima_ev_abi_header hdr;         ///< ABI header (op type, version, IO arity).
  struct sima_ev_tensor_list_v1 inputs;  ///< Input tensor list.
  struct sima_ev_tensor_list_v1 outputs; ///< Output tensor list.
  struct sima_ev_detessdequant_params_v1 params[SIMA_EV_MAX_IO]; ///< Per-output parameters.
};

/*
 * Example preproc payload. Preproc is intentionally kept as a small per-op
 * wrapper over the shared tensor descriptors rather than forcing all kernels
 * into a single giant transform struct.
 */
/// @brief Region-of-interest rectangle, in pixels.
struct sima_ev_roi_v1 {
  int32_t x;      ///< Top-left X (pixels).
  int32_t y;      ///< Top-left Y (pixels).
  int32_t width;  ///< ROI width (pixels).
  int32_t height; ///< ROI height (pixels).
};

/**
 * @brief Preproc op config: 1 input image, 1 output tensor, plus crop/resize/normalize.
 *
 * Preproc is intentionally kept as a small per-op wrapper over the shared
 * tensor descriptors rather than forcing all kernels into a single giant
 * transform struct.
 */
struct sima_ev_preproc_config_v1 {
  struct sima_ev_abi_header hdr;     ///< ABI header (op type, version, IO arity).
  struct sima_ev_tensor_desc input;  ///< Input image tensor descriptor.
  struct sima_ev_tensor_desc output; ///< Output tensor descriptor.
  struct sima_ev_roi_v1 crop;        ///< Crop ROI in input pixels.
  int32_t resize_width;              ///< Output width (post-crop, post-resize), in pixels.
  int32_t resize_height;             ///< Output height, in pixels.
  uint32_t color_in;                 ///< Input color/format (op-specific encoding).
  uint32_t color_out;                ///< Output color/format (op-specific encoding).
  float normalize_scale[4];          ///< Per-channel scale for output normalization.
  float normalize_bias[4];           ///< Per-channel bias for output normalization.
};

#ifdef __cplusplus
} /* extern "C" */
#endif

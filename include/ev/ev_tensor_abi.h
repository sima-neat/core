#ifndef EV_TENSOR_ABI_H
#define EV_TENSOR_ABI_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#define SIMA_EV_ABI_VERSION_V1 1U
#define SIMA_EV_MAX_TENSORS 32U
#define SIMA_EV_MAX_RANK 6U

enum sima_ev_op_type {
  SIMA_EV_OP_INVALID = 0,
  SIMA_EV_OP_TESSELLATE = 1,
  SIMA_EV_OP_DETESSELLATE = 2,
  SIMA_EV_OP_CAST = 3,
  SIMA_EV_OP_CASTTESSELLATE = 4,
  SIMA_EV_OP_DETESSELLATECAST = 5,
  SIMA_EV_OP_QUANTIZE = 6,
  SIMA_EV_OP_QUANTIZETESSELLATE = 7,
  SIMA_EV_OP_DEQUANTIZE = 8,
  SIMA_EV_OP_QUANTTESS = 9,
  SIMA_EV_OP_DETESSDEQUANT = 10,
};

enum sima_ev_dtype {
  SIMA_EV_DTYPE_INT8 = 0,
  SIMA_EV_DTYPE_INT16 = 1,
  SIMA_EV_DTYPE_BF16 = SIMA_EV_DTYPE_INT16,
  SIMA_EV_DTYPE_INT32 = 2,
  SIMA_EV_DTYPE_FP32 = 3,
  SIMA_EV_DTYPE_FP16 = 4,
};

enum sima_ev_layout_kind {
  SIMA_EV_LAYOUT_STRIDED = 0,
  SIMA_EV_LAYOUT_TILED = 1,
  SIMA_EV_LAYOUT_OPAQUE = 2,
};

enum sima_ev_run_target {
  SIMA_EV_RUN_AUTO = 0,
  SIMA_EV_RUN_EV74 = 1,
  SIMA_EV_RUN_A65 = 2,
};

enum sima_ev_tiled_flags {
  SIMA_EV_TILED_FLAG_NONE = 0,
  SIMA_EV_TILED_FLAG_COMPACT_CHANNELS = 1U << 0,
};

enum sima_ev_axis_semantic {
  SIMA_EV_AXIS_UNKNOWN = 0,
  SIMA_EV_AXIS_N = 1,
  SIMA_EV_AXIS_D = 2,
  SIMA_EV_AXIS_H = 3,
  SIMA_EV_AXIS_W = 4,
  SIMA_EV_AXIS_C = 5,
};

enum sima_ev_dense_tensor_format {
  SIMA_EV_DENSE_FORMAT_UNKNOWN = 0,
  SIMA_EV_DENSE_FORMAT_NDHWC = 1,
  SIMA_EV_DENSE_FORMAT_NDCHW = 2,
};

struct __attribute__((packed)) sima_ev_abi_header {
  uint16_t abi_version;
  uint16_t op_type;
  uint32_t flags;
  uint32_t tensor_count;
  uint32_t run_target;
};

struct __attribute__((packed)) sima_ev_storage_desc {
  uint64_t addr;
  uint64_t nbytes;
};

struct __attribute__((packed)) sima_ev_shape_desc {
  uint32_t rank;
  uint8_t axis_semantics[SIMA_EV_MAX_RANK];
  int64_t sizes[SIMA_EV_MAX_RANK];
};

struct __attribute__((packed)) sima_ev_strided_layout_desc {
  int64_t strides_bytes[SIMA_EV_MAX_RANK];
};

struct __attribute__((packed)) sima_ev_tiled_layout_desc {
  int64_t tile_sizes[SIMA_EV_MAX_RANK];
  uint32_t tile_align_bytes;
  uint32_t flags;
};

struct __attribute__((packed)) sima_ev_tensor_desc {
  uint32_t dtype;
  uint32_t layout_kind;
  struct sima_ev_storage_desc storage;
  struct sima_ev_shape_desc shape;
  union {
    struct sima_ev_strided_layout_desc strided;
    struct sima_ev_tiled_layout_desc tiled;
  } layout;
};

static inline uint32_t sima_ev_tensor_count(const struct sima_ev_abi_header* hdr) {
  return hdr ? hdr->tensor_count : 0U;
}

static inline enum sima_ev_run_target
sima_ev_requested_run_target(const struct sima_ev_abi_header* hdr) {
  if (!hdr) {
    return SIMA_EV_RUN_AUTO;
  }
  switch ((enum sima_ev_run_target)hdr->run_target) {
  case SIMA_EV_RUN_AUTO:
  case SIMA_EV_RUN_EV74:
  case SIMA_EV_RUN_A65:
    return (enum sima_ev_run_target)hdr->run_target;
  default:
    return SIMA_EV_RUN_AUTO;
  }
}

static inline const char* sima_ev_run_target_name(enum sima_ev_run_target run_target) {
  switch (run_target) {
  case SIMA_EV_RUN_EV74:
    return "EV74";
  case SIMA_EV_RUN_A65:
    return "A65";
  case SIMA_EV_RUN_AUTO:
  default:
    return "AUTO";
  }
}

static inline int sima_ev_axis_index(const struct sima_ev_shape_desc* desc,
                                     enum sima_ev_axis_semantic axis) {
  if (!desc) {
    return -1;
  }
  const uint32_t rank = desc->rank < SIMA_EV_MAX_RANK ? desc->rank : SIMA_EV_MAX_RANK;
  for (uint32_t i = 0; i < rank; ++i) {
    if ((enum sima_ev_axis_semantic)desc->axis_semantics[i] == axis) {
      return (int)i;
    }
  }
  return -1;
}

static inline int64_t sima_ev_axis_size(const struct sima_ev_shape_desc* desc,
                                        enum sima_ev_axis_semantic axis, int64_t fallback) {
  const int axis_index = sima_ev_axis_index(desc, axis);
  return axis_index >= 0 ? desc->sizes[axis_index] : fallback;
}

static inline int64_t sima_ev_axis_stride_bytes(const struct sima_ev_shape_desc* shape,
                                                const struct sima_ev_strided_layout_desc* layout,
                                                enum sima_ev_axis_semantic axis, int64_t fallback) {
  if (!shape || !layout) {
    return fallback;
  }
  const int axis_index = sima_ev_axis_index(shape, axis);
  return axis_index >= 0 ? layout->strides_bytes[axis_index] : fallback;
}

static inline int64_t sima_ev_tensor_axis_stride_bytes(const struct sima_ev_tensor_desc* desc,
                                                       enum sima_ev_axis_semantic axis,
                                                       int64_t fallback) {
  if (!desc || desc->layout_kind != SIMA_EV_LAYOUT_STRIDED) {
    return fallback;
  }
  return sima_ev_axis_stride_bytes(&desc->shape, &desc->layout.strided, axis, fallback);
}

static inline int64_t sima_ev_tile_axis_size(const struct sima_ev_shape_desc* shape,
                                             const struct sima_ev_tiled_layout_desc* layout,
                                             enum sima_ev_axis_semantic axis, int64_t fallback) {
  if (!shape || !layout) {
    return fallback;
  }
  const int axis_index = sima_ev_axis_index(shape, axis);
  return axis_index >= 0 ? layout->tile_sizes[axis_index] : fallback;
}

static inline int64_t sima_ev_tensor_tile_axis_size(const struct sima_ev_tensor_desc* desc,
                                                    enum sima_ev_axis_semantic axis,
                                                    int64_t fallback) {
  if (!desc || desc->layout_kind != SIMA_EV_LAYOUT_TILED) {
    return fallback;
  }
  return sima_ev_tile_axis_size(&desc->shape, &desc->layout.tiled, axis, fallback);
}

static inline int sima_ev_elem_size_bytes(uint32_t dtype) {
  switch ((enum sima_ev_dtype)dtype) {
  case SIMA_EV_DTYPE_FP32:
  case SIMA_EV_DTYPE_INT32:
    return 4;
  case SIMA_EV_DTYPE_FP16:
  case SIMA_EV_DTYPE_INT16:
    return 2;
  case SIMA_EV_DTYPE_INT8:
  default:
    return 1;
  }
}

static inline enum sima_ev_dense_tensor_format
sima_ev_infer_dense_tensor_format(const struct sima_ev_tensor_desc* desc) {
  if (!desc || desc->layout_kind != SIMA_EV_LAYOUT_STRIDED) {
    return SIMA_EV_DENSE_FORMAT_UNKNOWN;
  }

  const int64_t height = sima_ev_axis_size(&desc->shape, SIMA_EV_AXIS_H, 0);
  const int64_t width = sima_ev_axis_size(&desc->shape, SIMA_EV_AXIS_W, 0);
  const int64_t channels = sima_ev_axis_size(&desc->shape, SIMA_EV_AXIS_C, 0);
  const int64_t elem_size = sima_ev_elem_size_bytes(desc->dtype);
  const int64_t stride_d = sima_ev_tensor_axis_stride_bytes(
      desc, SIMA_EV_AXIS_D,
      (height > 0 && width > 0 && channels > 0) ? height * width * channels * elem_size : -1);
  const int64_t stride_h = sima_ev_tensor_axis_stride_bytes(
      desc, SIMA_EV_AXIS_H, (width > 0 && channels > 0) ? width * channels * elem_size : -1);
  const int64_t stride_w = sima_ev_tensor_axis_stride_bytes(
      desc, SIMA_EV_AXIS_W, (channels > 0) ? channels * elem_size : -1);
  const int64_t stride_c = sima_ev_tensor_axis_stride_bytes(desc, SIMA_EV_AXIS_C, -1);

  const int64_t ndhwc_stride_c = elem_size;
  const int64_t ndhwc_stride_w = channels * elem_size;
  const int64_t ndhwc_stride_h = width * ndhwc_stride_w;
  const int64_t ndhwc_stride_d = height * ndhwc_stride_h;
  if (stride_c == ndhwc_stride_c && stride_w == ndhwc_stride_w && stride_h == ndhwc_stride_h &&
      stride_d == ndhwc_stride_d) {
    return SIMA_EV_DENSE_FORMAT_NDHWC;
  }

  const int64_t ndchw_stride_w = elem_size;
  const int64_t ndchw_stride_h = width * ndchw_stride_w;
  const int64_t ndchw_stride_c = height * ndchw_stride_h;
  const int64_t ndchw_stride_d = channels * ndchw_stride_c;
  if (stride_w == ndchw_stride_w && stride_h == ndchw_stride_h && stride_c == ndchw_stride_c &&
      stride_d == ndchw_stride_d) {
    return SIMA_EV_DENSE_FORMAT_NDCHW;
  }

  return SIMA_EV_DENSE_FORMAT_UNKNOWN;
}

static inline int sima_ev_tile_is_aligned(const struct sima_ev_tensor_desc* desc) {
  return desc && desc->layout_kind == SIMA_EV_LAYOUT_TILED &&
         desc->layout.tiled.tile_align_bytes != 0U;
}

static inline int sima_ev_tiled_uses_compact_channels(const struct sima_ev_tensor_desc* desc) {
  return desc && desc->layout_kind == SIMA_EV_LAYOUT_TILED &&
         ((desc->layout.tiled.flags & SIMA_EV_TILED_FLAG_COMPACT_CHANNELS) != 0U);
}

static inline int sima_ev_tensor_is_contiguous(const struct sima_ev_tensor_desc* desc) {
  if (!desc || desc->layout_kind != SIMA_EV_LAYOUT_STRIDED) {
    return 0;
  }

  int64_t expected_stride = sima_ev_elem_size_bytes(desc->dtype);
  for (uint32_t idx = desc->shape.rank; idx-- > 0U;) {
    if (desc->layout.strided.strides_bytes[idx] != expected_stride) {
      return 0;
    }
    if (desc->shape.sizes[idx] > 0 && expected_stride <= (INT64_MAX / desc->shape.sizes[idx])) {
      expected_stride *= desc->shape.sizes[idx];
    }
  }
  return 1;
}

#endif // EV_TENSOR_ABI_H

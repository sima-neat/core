/**
 * @file TensorAdapters.cpp
 * @brief Adapters between GStreamer (GstSample/GstCaps/GstVideoInfo) and Tensor.
 *
 * Supported inputs (via from_gst_sample):
 *  1) application/vnd.simaai.tensor
 *     - Wraps the GstSample in StorageKind::GstSample (read-only)
 *     - Attempts to infer dtype + shape from caps fields (width/height/depth/format)
 *     - Uses tight/contiguous strides (since caps do not provide per-dim stride metadata)
 *
 *  2) video/x-raw (and compatible raw video caps parsed by gst_video_info_from_caps)
 *     - Wraps the GstSample in StorageKind::GstSample (read-only)
 *     - Supports RGB/BGR/GRAY8/NV12/I420
 *     - For packed RGB/BGR/GRAY8: produces dense HWC (or HW for GRAY8) with row stride
 *     - For NV12/I420: produces a composite tensor with planes (Y/UV or Y/U/V)
 *
 *  3) encoded media caps recognized by caps_to_codec()
 *     - Wraps the GstSample in StorageKind::GstSample (read-only)
 *     - Represents the payload as a dense UInt8 byte vector view
 *
 * Semantics confirmed by other repo code:
 *  - Tensor::map_read() applies Tensor::byte_offset (already pointer-adjusted).
 *  - Composite tensors should keep Tensor::byte_offset == 0 and use plane byte_offsets.
 *  - Plane strides are byte-strides; if a second stride is present, it should equal elem_bytes
 *    (InputStream's tensor_plane_bytes_tight() enforces stride[1] == elem_bytes when present).
 *
 * TODO(repo): If your tensor caps can include explicit layout/strides (e.g., "stride0"/"stride1"),
 * parse them here instead of assuming tight contiguous strides for application/vnd.simaai.tensor.
 */

#include "pipeline/TensorAdapters.h"

#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/TensorUtil.h" // make_gst_sample_storage()
#include "pipeline/internal/TensorMath.h"

#include <gst/gst.h>
#include <gst/video/video.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace simaai::neat {

using simaai::neat::pipeline_internal::dtype_bytes;
using simaai::neat::pipeline_internal::upper_copy;

namespace {

//==============================================================================
// Small helpers
//==============================================================================

ImageSpec::PixelFormat pixel_format_from_gst(GstVideoFormat fmt) {
  switch (fmt) {
  case GST_VIDEO_FORMAT_RGB:
    return ImageSpec::PixelFormat::RGB;
  case GST_VIDEO_FORMAT_BGR:
    return ImageSpec::PixelFormat::BGR;
  case GST_VIDEO_FORMAT_GRAY8:
    return ImageSpec::PixelFormat::GRAY8;
  case GST_VIDEO_FORMAT_NV12:
    return ImageSpec::PixelFormat::NV12;
  case GST_VIDEO_FORMAT_I420:
    return ImageSpec::PixelFormat::I420;
  default:
    return ImageSpec::PixelFormat::UNKNOWN;
  }
}

/**
 * Create a 2D plane descriptor.
 *
 * stride_bytes is bytes/row.
 * stride[1] (bytes/element) is explicitly set to elem_bytes to match repo conventions.
 */
Plane make_plane(PlaneRole role, int64_t h, int64_t w, int64_t stride_bytes, int64_t offset_bytes,
                 int64_t elem_bytes) {
  Plane plane;
  plane.role = role;
  plane.shape = {h, w};
  plane.strides_bytes = {stride_bytes, elem_bytes};
  plane.byte_offset = offset_bytes;
  return plane;
}

/**
 * Map cap "format" strings (app/vnd.simaai.tensor) to simaai::neat::TensorDType.
 *
 * NOTE: This mapping is repo-specific. Keep it centralized here so format additions
 * don't get duplicated across files.
 *
 * TODO(repo): Confirm the canonical list of tensor "format" strings and extend the mapping.
 */
simaai::neat::TensorDType dtype_from_tensor_format(std::string_view fmt) {
  const std::string s = upper_copy(std::string(fmt));
  if (s == "DETESS")
    return simaai::neat::TensorDType::UInt16;
  if (s == "DETESSDEQUANT" || s == "FP32" || s == "FLOAT32" || s == "EVXX_FLOAT32")
    return simaai::neat::TensorDType::Float32;
  if (s == "EVXX_INT8" || s == "EV74_INT8" || s == "INT8")
    return simaai::neat::TensorDType::Int8;
  if (s == "MLA")
    return simaai::neat::TensorDType::Int8;
  if (s == "EVXX_BFLOAT16" || s == "BF16" || s == "BFLOAT16")
    return simaai::neat::TensorDType::BFloat16;
  if (s == "UINT8" || s == "EVXX_UINT8")
    return simaai::neat::TensorDType::UInt8;
  return simaai::neat::TensorDType::UInt8;
}

simaai::neat::TensorDType dtype_from_data_type(std::string_view dtype) {
  const std::string s = upper_copy(std::string(dtype));
  if (s == "INT8" || s == "EVXX_INT8")
    return simaai::neat::TensorDType::Int8;
  if (s == "UINT8" || s == "EVXX_UINT8")
    return simaai::neat::TensorDType::UInt8;
  if (s == "INT16" || s == "EVXX_INT16")
    return simaai::neat::TensorDType::Int16;
  if (s == "UINT16" || s == "EVXX_UINT16")
    return simaai::neat::TensorDType::UInt16;
  if (s == "INT32" || s == "EVXX_INT32")
    return simaai::neat::TensorDType::Int32;
  if (s == "BF16" || s == "BFLOAT16" || s == "EVXX_BFLOAT16")
    return simaai::neat::TensorDType::BFloat16;
  if (s == "FP32" || s == "FLOAT32" || s == "EVXX_FLOAT32")
    return simaai::neat::TensorDType::Float32;
  if (s == "FP64" || s == "FLOAT64")
    return simaai::neat::TensorDType::Float64;
  return simaai::neat::TensorDType::UInt8;
}

const char* find_data_type_field(const GstStructure* st) {
  if (!st)
    return nullptr;
  if (const char* dt = gst_structure_get_string(st, "data_type"))
    return dt;
  const std::string prefix = "data_type__";
  int best_idx = std::numeric_limits<int>::max();
  const char* best_val = nullptr;
  const int n = gst_structure_n_fields(st);
  for (int i = 0; i < n; ++i) {
    const char* name = gst_structure_nth_field_name(st, i);
    if (!name)
      continue;
    const std::string_view field(name);
    if (field.rfind(prefix, 0) == 0) {
      const std::string_view suffix = field.substr(prefix.size());
      bool numeric = !suffix.empty();
      for (char c : suffix) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
          numeric = false;
          break;
        }
      }
      if (!numeric)
        continue;
      const int idx = std::stoi(std::string(suffix));
      if (idx < best_idx) {
        if (const char* dt = gst_structure_get_string(st, name)) {
          best_idx = idx;
          best_val = dt;
        }
      }
      continue;
    }
    if (field.rfind("data_type", 0) == 0) {
      if (const char* dt = gst_structure_get_string(st, name)) {
        if (!best_val)
          best_val = dt;
      }
    }
  }
  return best_val;
}

bool find_indexed_int_field(const GstStructure* st, const char* base_name, int* out) {
  if (!st || !base_name || !out)
    return false;
  if (gst_structure_get_int(st, base_name, out))
    return true;

  const std::string prefix = std::string(base_name) + "__";
  int best_idx = std::numeric_limits<int>::max();
  int best_val = 0;
  bool found = false;

  const int n = gst_structure_n_fields(st);
  for (int i = 0; i < n; ++i) {
    const char* name = gst_structure_nth_field_name(st, i);
    if (!name)
      continue;
    const std::string_view field(name);
    if (field.rfind(prefix, 0) != 0)
      continue;

    const std::string_view suffix = field.substr(prefix.size());
    bool numeric = !suffix.empty();
    for (char c : suffix) {
      if (!std::isdigit(static_cast<unsigned char>(c))) {
        numeric = false;
        break;
      }
    }
    if (!numeric)
      continue;

    int value = 0;
    if (!gst_structure_get_int(st, name, &value))
      continue;
    const int idx = std::stoi(std::string(suffix));
    if (idx < best_idx) {
      best_idx = idx;
      best_val = value;
      found = true;
    }
  }

  if (found)
    *out = best_val;
  return found;
}

bool parse_shape_csv(const std::string& raw, std::vector<int64_t>* out) {
  if (!out)
    return false;
  out->clear();
  std::stringstream ss(raw);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token.erase(std::remove_if(token.begin(), token.end(),
                               [](unsigned char c) { return std::isspace(c) != 0; }),
                token.end());
    if (token.empty())
      return false;
    try {
      const long long value = std::stoll(token);
      if (value <= 0)
        return false;
      out->push_back(static_cast<int64_t>(value));
    } catch (...) {
      return false;
    }
  }
  return !out->empty();
}

bool find_tensor_shape(const GstStructure* st, std::vector<int64_t>* out) {
  if (!st || !out)
    return false;

  gint rank_i = 0;
  if (gst_structure_get_int(st, "rank", &rank_i) && rank_i > 0) {
    const guint rank = static_cast<guint>(rank_i);
    if (const char* shape_csv = gst_structure_get_string(st, "shape")) {
      std::vector<int64_t> parsed;
      if (parse_shape_csv(shape_csv, &parsed) && parsed.size() == static_cast<std::size_t>(rank)) {
        *out = std::move(parsed);
        return true;
      }
      return false;
    }

    out->clear();
    out->reserve(rank);
    for (guint i = 0; i < rank; ++i) {
      const std::string key = "dim" + std::to_string(i);
      gint dim_i = 0;
      if (!gst_structure_get_int(st, key.c_str(), &dim_i) || dim_i <= 0) {
        out->clear();
        return false;
      }
      out->push_back(static_cast<int64_t>(dim_i));
    }
    return true;
  }

  int w = 0;
  int h = 0;
  int d = 0;
  find_indexed_int_field(st, "width", &w);
  find_indexed_int_field(st, "height", &h);
  find_indexed_int_field(st, "depth", &d);
  if (w > 0 && h > 0 && d > 0) {
    *out = {h, w, d};
    return true;
  }
  if (w > 0 && h > 0) {
    *out = {h, w};
    return true;
  }
  if (w > 0) {
    *out = {w};
    return true;
  }
  return false;
}

TensorLayout layout_from_caps_token(const char* token) {
  const std::string layout = upper_copy(std::string(token ? token : ""));
  if (layout == "HW") {
    return TensorLayout::HW;
  }
  if (layout == "HWC" || layout == "NHWC" || layout == "NDHWC") {
    return TensorLayout::HWC;
  }
  if (layout == "CHW" || layout == "NCHW" || layout == "NDCHW") {
    return TensorLayout::CHW;
  }
  return TensorLayout::Unknown;
}

/**
 * Build a Tensor wrapping a GstSample that contains "application/vnd.simaai.tensor".
 *
 * Shape inference priority:
 *  - If caps provide width/height/depth => [H,W,D] (tight contiguous strides)
 *  - If caps provide width/height      => [H,W]
 *  - If caps provide width             => [W]
 *  - Else fallback to buffer size      => [N] where N = bytes/elem
 *
 * TODO(repo): If these tensors can be non-contiguous / padded, add stride fields to caps
 * or attach metadata (e.g., GstVideoMeta-like) and parse it here.
 */
Tensor from_gst_tensor_sample(GstSample* sample, const GstStructure* st, GstBuffer* buffer) {
  Tensor out;
  out.storage = simaai::neat::pipeline_internal::make_gst_sample_storage(sample);
  if (!out.storage) {
    throw std::runtime_error("from_gst_sample: missing tensor storage");
  }

  const char* fmt = gst_structure_get_string(st, "format");
  const std::string fmt_str = fmt ? fmt : "";
  out.dtype = dtype_from_tensor_format(fmt_str);
  if (const char* dt = find_data_type_field(st)) {
    out.dtype = dtype_from_data_type(dt);
  }
  out.device = out.storage->device;
  out.read_only = true;
  out.layout = layout_from_caps_token(gst_structure_get_string(st, "layout"));

  const std::size_t elem = dtype_bytes(out.dtype);
  if (elem == 0) {
    throw std::runtime_error("from_gst_sample: unknown tensor element size");
  }

  if (find_tensor_shape(st, &out.shape)) {
    out.strides_bytes = simaai::neat::pipeline_internal::contiguous_strides_bytes(out.shape, elem);
  } else {
    const std::size_t bytes = buffer ? gst_buffer_get_size(buffer) : 0;
    const std::size_t n = (bytes > 0) ? (bytes / elem) : 0;
    if (n > 0) {
      out.shape = {static_cast<int64_t>(n)};
      out.strides_bytes = {static_cast<int64_t>(elem)};
    }
  }

  if (!fmt_str.empty()) {
    TessSpec tess;
    tess.format = fmt_str;
    out.semantic.tess = tess;
  }
  if (const auto preprocess = read_simaai_preprocess_meta(buffer); preprocess.has_value()) {
    out.semantic.preprocess = *preprocess;
  }

  return out;
}

/**
 * Build a Tensor wrapping a GstSample that contains raw video.
 * Supports: RGB, BGR, GRAY8, NV12, I420.
 */
Tensor from_gst_video_sample(GstSample* sample, GstCaps* caps) {
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  GstVideoMeta* vmeta = buffer ? gst_buffer_get_video_meta(buffer) : nullptr;

  GstVideoInfo info;
  std::memset(&info, 0, sizeof(info));
  if (!vmeta) {
    if (!gst_video_info_from_caps(&info, caps)) {
      throw std::runtime_error("from_gst_sample: gst_video_info_from_caps failed");
    }
  }

  auto storage = simaai::neat::pipeline_internal::make_gst_sample_storage(sample);
  if (!storage) {
    throw std::runtime_error("from_gst_sample: missing video storage");
  }

  const GstVideoFormat fmt =
      vmeta ? static_cast<GstVideoFormat>(vmeta->format) : GST_VIDEO_INFO_FORMAT(&info);
  const ImageSpec::PixelFormat pixel = pixel_format_from_gst(fmt);
  if (pixel == ImageSpec::PixelFormat::UNKNOWN) {
    throw std::runtime_error("from_gst_sample: unsupported pixel format");
  }

  Tensor out;
  out.storage = std::move(storage);
  out.dtype = simaai::neat::TensorDType::UInt8;
  out.device = out.storage->device;
  out.read_only = true;

  const int w = vmeta ? static_cast<int>(vmeta->width) : GST_VIDEO_INFO_WIDTH(&info);
  const int h = vmeta ? static_cast<int>(vmeta->height) : GST_VIDEO_INFO_HEIGHT(&info);

  ImageSpec image;
  image.format = pixel;
  out.semantic.image = image;
  if (const auto preprocess = read_simaai_preprocess_meta(buffer); preprocess.has_value()) {
    out.semantic.preprocess = *preprocess;
  }

  // Planar YUV formats: represent as composite tensor with planes.
  if (fmt == GST_VIDEO_FORMAT_NV12) {
    // A convenient top-level (H,W) shape for dimensions; actual data accessed via planes.
    out.shape = {h, w};
    out.layout = simaai::neat::TensorLayout::HW;
    const int64_t y_stride = vmeta ? static_cast<int64_t>(vmeta->stride[0])
                                   : static_cast<int64_t>(GST_VIDEO_INFO_PLANE_STRIDE(&info, 0));
    out.strides_bytes = {y_stride, 1};

    const int64_t elem = 1;
    const int64_t uv_stride = vmeta ? static_cast<int64_t>(vmeta->stride[1])
                                    : static_cast<int64_t>(GST_VIDEO_INFO_PLANE_STRIDE(&info, 1));
    const int64_t y_offset = vmeta ? static_cast<int64_t>(vmeta->offset[0])
                                   : static_cast<int64_t>(GST_VIDEO_INFO_PLANE_OFFSET(&info, 0));
    const int64_t uv_offset = vmeta ? static_cast<int64_t>(vmeta->offset[1])
                                    : static_cast<int64_t>(GST_VIDEO_INFO_PLANE_OFFSET(&info, 1));

    out.planes.push_back(make_plane(PlaneRole::Y, h, w, y_stride, y_offset, elem));
    out.planes.push_back(make_plane(PlaneRole::UV, h / 2, w, uv_stride, uv_offset, elem));
    return out;
  }

  if (fmt == GST_VIDEO_FORMAT_I420) {
    out.shape = {h, w};
    out.layout = simaai::neat::TensorLayout::HW;
    const int64_t y_stride = vmeta ? static_cast<int64_t>(vmeta->stride[0])
                                   : static_cast<int64_t>(GST_VIDEO_INFO_PLANE_STRIDE(&info, 0));
    out.strides_bytes = {y_stride, 1};

    const int64_t elem = 1;
    const int64_t u_stride = vmeta ? static_cast<int64_t>(vmeta->stride[1])
                                   : static_cast<int64_t>(GST_VIDEO_INFO_PLANE_STRIDE(&info, 1));
    const int64_t v_stride = vmeta ? static_cast<int64_t>(vmeta->stride[2])
                                   : static_cast<int64_t>(GST_VIDEO_INFO_PLANE_STRIDE(&info, 2));
    const int64_t y_offset = vmeta ? static_cast<int64_t>(vmeta->offset[0])
                                   : static_cast<int64_t>(GST_VIDEO_INFO_PLANE_OFFSET(&info, 0));
    const int64_t u_offset = vmeta ? static_cast<int64_t>(vmeta->offset[1])
                                   : static_cast<int64_t>(GST_VIDEO_INFO_PLANE_OFFSET(&info, 1));
    const int64_t v_offset = vmeta ? static_cast<int64_t>(vmeta->offset[2])
                                   : static_cast<int64_t>(GST_VIDEO_INFO_PLANE_OFFSET(&info, 2));

    out.planes.push_back(make_plane(PlaneRole::Y, h, w, y_stride, y_offset, elem));
    out.planes.push_back(make_plane(PlaneRole::U, h / 2, w / 2, u_stride, u_offset, elem));
    out.planes.push_back(make_plane(PlaneRole::V, h / 2, w / 2, v_stride, v_offset, elem));
    return out;
  }

  // Packed formats: represent as dense HWC/HW tensor with row stride.
  if (fmt == GST_VIDEO_FORMAT_RGB || fmt == GST_VIDEO_FORMAT_BGR || fmt == GST_VIDEO_FORMAT_GRAY8) {
    const int channels = (fmt == GST_VIDEO_FORMAT_GRAY8) ? 1 : 3;
    const int64_t elem = 1;
    const int64_t row_stride = vmeta ? static_cast<int64_t>(vmeta->stride[0])
                                     : static_cast<int64_t>(GST_VIDEO_INFO_PLANE_STRIDE(&info, 0));

    if (channels == 1) {
      out.shape = {h, w};
      out.strides_bytes = {row_stride, elem};
      out.layout = simaai::neat::TensorLayout::HW;
    } else {
      out.shape = {h, w, channels};
      out.strides_bytes = {row_stride, static_cast<int64_t>(channels) * elem, elem};
      out.layout = simaai::neat::TensorLayout::HWC;
    }
    return out;
  }

  throw std::runtime_error("from_gst_sample: unsupported pixel format");
}

Tensor from_gst_encoded_sample(GstSample* sample, GstCaps* caps, GstBuffer* buffer) {
  auto storage = simaai::neat::pipeline_internal::make_gst_sample_storage(sample);
  if (!storage) {
    throw std::runtime_error("from_gst_sample: missing encoded storage");
  }

  const std::string caps_string = pipeline_internal::gst_caps_to_string_safe(caps);

  Tensor out;
  out.storage = std::move(storage);
  out.dtype = simaai::neat::TensorDType::UInt8;
  out.device = out.storage->device;
  out.read_only = true;
  out.layout = simaai::neat::TensorLayout::Unknown;
  out.shape = {static_cast<int64_t>(gst_buffer_get_size(buffer))};
  out.strides_bytes = {1};
  out.semantic.encoded = simaai::neat::EncodedSpec{};
  out.semantic.encoded->codec = caps_to_codec(caps_string);
  if (const auto preprocess = read_simaai_preprocess_meta(buffer); preprocess.has_value()) {
    out.semantic.preprocess = *preprocess;
  }

  return out;
}

} // namespace

//==============================================================================
// Public API
//==============================================================================

Tensor from_gst_sample(GstSample* sample) {
  if (!sample) {
    throw std::runtime_error("from_gst_sample: null sample");
  }

  GstCaps* caps = gst_sample_get_caps(sample);
  if (!caps) {
    throw std::runtime_error("from_gst_sample: missing caps");
  }

  const GstStructure* st = gst_caps_get_structure(caps, 0);
  const char* media = st ? gst_structure_get_name(st) : nullptr;
  const std::string caps_string = pipeline_internal::gst_caps_to_string_safe(caps);

  // Tensor sample path.
  if (media && std::string(media) == "application/vnd.simaai.tensor") {
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
      throw std::runtime_error("from_gst_sample: missing buffer");
    }
    return from_gst_tensor_sample(sample, st, buffer);
  }

  if (caps_to_codec(caps_string) != simaai::neat::EncodedSpec::Codec::UNKNOWN) {
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
      throw std::runtime_error("from_gst_sample: missing buffer");
    }
    return from_gst_encoded_sample(sample, caps, buffer);
  }

  // Raw video path.
  return from_gst_video_sample(sample, caps);
}

} // namespace simaai::neat

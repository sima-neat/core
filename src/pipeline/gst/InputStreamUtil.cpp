#include "InputStreamUtil.h"

#include "pipeline/GraphOptions.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/TensorTransfer.h"
#include "pipeline/internal/InputPolicy.h"
#include "pipeline/TensorOpenCV.h"
#include "pipeline/TessellatedTensor.h"
#include "nodes/io/Input.h"
#include "pipeline/internal/SimaaiGstCompat.h"

#include <gst/gst.h>

#include <opencv2/core/mat.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace simaai::neat {
using pipeline_internal::upper_copy;
namespace {

size_t tensor_dense_bytes_tight(const simaai::neat::Tensor& input);

const char* input_memory_policy_name(InputMemoryPolicy policy) {
  switch (policy) {
  case InputMemoryPolicy::Auto:
    return "auto";
  case InputMemoryPolicy::Ev74:
    return "ev74";
  case InputMemoryPolicy::Dms0:
    return "dms0";
  case InputMemoryPolicy::SystemMemory:
    return "system";
  }
  return "auto";
}

size_t dtype_bytes(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt8:
    return 1;
  case TensorDType::Int8:
    return 1;
  case TensorDType::UInt16:
    return 2;
  case TensorDType::Int16:
    return 2;
  case TensorDType::Int32:
    return 4;
  case TensorDType::BFloat16:
    return 2;
  case TensorDType::Float32:
    return 4;
  case TensorDType::Float64:
    return 8;
  }
  return 1;
}

int shape_dim(const std::vector<int64_t>& shape, size_t idx) {
  if (shape.size() <= idx)
    return -1;
  const int64_t v = shape[idx];
  return (v > 0) ? static_cast<int>(v) : -1;
}

std::string tensor_shape_csv_local(const std::vector<int64_t>& shape) {
  std::ostringstream oss;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i)
      oss << ",";
    oss << shape[i];
  }
  return oss.str();
}

std::vector<int64_t> tensor_shape_from_compat_dims_local(int width, int height, int depth,
                                                         TensorLayout layout) {
  if (width <= 0)
    return {};
  if (height <= 0)
    return {width};
  if (depth <= 0 || layout == TensorLayout::HW)
    return {height, width};
  if (layout == TensorLayout::CHW)
    return {depth, height, width};
  return {height, width, depth};
}

TensorCompatDims tensor_compat_dims_from_shape_local(const std::vector<int64_t>& shape,
                                                     TensorLayout layout) {
  TensorCompatDims out;
  if (shape.empty())
    return out;

  const auto dim_from_end = [&](size_t from_end) -> int {
    if (shape.size() < from_end)
      return -1;
    const int64_t v = shape[shape.size() - from_end];
    return v > 0 ? static_cast<int>(v) : -1;
  };

  if (layout == TensorLayout::HW) {
    if (shape.size() == 1U) {
      out.height = 1;
      out.width = dim_from_end(1);
      out.depth = (out.width > 0) ? 1 : -1;
      return out;
    }
    out.height = dim_from_end(2);
    out.width = dim_from_end(1);
    out.depth = (out.width > 0 && out.height > 0) ? 1 : -1;
    return out;
  }
  if (layout == TensorLayout::CHW) {
    out.depth = dim_from_end(3);
    out.height = dim_from_end(2);
    out.width = dim_from_end(1);
    return out;
  }
  if (layout == TensorLayout::HWC) {
    out.height = dim_from_end(3);
    out.width = dim_from_end(2);
    out.depth = dim_from_end(1);
    return out;
  }

  if (shape.size() >= 3U) {
    out.height = dim_from_end(3);
    out.width = dim_from_end(2);
    out.depth = dim_from_end(1);
  } else if (shape.size() == 2U) {
    out.height = dim_from_end(2);
    out.width = dim_from_end(1);
    out.depth = 1;
  } else if (shape.size() == 1U) {
    out.height = 1;
    out.width = dim_from_end(1);
    out.depth = 1;
  }
  return out;
}

bool gst_structure_set_int_vector_field_local(GstStructure* s, const char* field,
                                              const std::vector<int>& values) {
  if (!s || !field) {
    return false;
  }
  GValue list = G_VALUE_INIT;
  g_value_init(&list, GST_TYPE_LIST);
  for (const int value : values) {
    GValue item = G_VALUE_INIT;
    g_value_init(&item, G_TYPE_INT);
    g_value_set_int(&item, value);
    gst_value_list_append_value(&list, &item);
    g_value_unset(&item);
  }
  gst_structure_set_value(s, field, &list);
  g_value_unset(&list);
  return true;
}

bool gst_structure_get_int_vector_field_local(const GstStructure* s, const char* field,
                                              std::vector<int>* out) {
  if (!s || !field || !out) {
    return false;
  }
  const GValue* list = gst_structure_get_value(s, field);
  if (!list || !GST_VALUE_HOLDS_LIST(list)) {
    return false;
  }
  out->clear();
  const guint size = gst_value_list_get_size(list);
  out->reserve(size);
  for (guint i = 0; i < size; ++i) {
    const GValue* item = gst_value_list_get_value(list, i);
    if (!item || !G_VALUE_HOLDS_INT(item)) {
      return false;
    }
    out->push_back(g_value_get_int(item));
  }
  return true;
}

bool ensure_custom_meta_structure_mutable(GstBuffer* buffer, const char* meta_name,
                                          GstCustomMeta** meta_out, GstStructure** structure_out) {
  if (!buffer || !meta_name || !meta_out || !structure_out) {
    return false;
  }
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, meta_name);
  bool added_meta = false;
  if (!meta) {
    if (!gst_buffer_is_writable(buffer)) {
      return false;
    }
    meta = gst_buffer_add_custom_meta(buffer, meta_name);
    added_meta = true;
  }
  if (!meta) {
    return false;
  }
  GstStructure* s = gst_custom_meta_get_structure(meta);
  if (!s) {
    return false;
  }

  bool structure_mutable = added_meta;
#if defined(GST_STRUCTURE_IS_MUTABLE)
  structure_mutable = GST_STRUCTURE_IS_MUTABLE(s);
#elif defined(GST_STRUCTURE_IS_WRITABLE)
  structure_mutable = GST_STRUCTURE_IS_WRITABLE(s);
#endif
  if (!structure_mutable) {
    if (!gst_buffer_is_writable(buffer)) {
      return false;
    }
    GstStructure* snapshot = gst_structure_copy(s);
    gst_buffer_remove_meta(buffer, &meta->meta);
    meta = gst_buffer_add_custom_meta(buffer, meta_name);
    s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
    if (!s) {
      if (snapshot) {
        gst_structure_free(snapshot);
      }
      return false;
    }
    if (snapshot) {
      const gint n_fields = gst_structure_n_fields(snapshot);
      for (gint i = 0; i < n_fields; ++i) {
        const char* fname = gst_structure_nth_field_name(snapshot, i);
        if (!fname) {
          continue;
        }
        const GValue* val = gst_structure_get_value(snapshot, fname);
        if (val) {
          gst_structure_set_value(s, fname, val);
        }
      }
      gst_structure_free(snapshot);
    }
  }

  *meta_out = meta;
  *structure_out = s;
  return true;
}

std::optional<std::string> validate_axis_perm_vector_local(const std::vector<int>& perm,
                                                           const char* field) {
  std::vector<int> sorted = perm;
  for (const int axis : sorted) {
    if (axis < 0) {
      return std::string("invalid preprocess metadata field '") + field +
             "' (axis_perm must contain only non-negative indices)";
    }
  }
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
    return std::string("invalid preprocess metadata field '") + field +
           "' (axis_perm must not contain duplicate indices)";
  }
  return std::nullopt;
}

bool sample_uses_single_tensor_envelope_transport(const Sample& sample) {
  const Tensor* tensor = nullptr;
  if (sample.kind == SampleKind::TensorSet) {
    if (sample.tensors.size() != 1U || !sample.fields.empty()) {
      return false;
    }
    tensor = &sample.tensors.front();
  } else if (sample.kind == SampleKind::Tensor) {
    if (!sample.tensor.has_value() || !sample.fields.empty()) {
      return false;
    }
    tensor = &*sample.tensor;
  } else {
    return false;
  }

  const bool runtime_tensor_backing =
      tensor->storage &&
      (tensor->storage->kind == StorageKind::GstSample || !tensor->storage->sima_segments.empty());

  const auto runtime_view_requires_envelope_transport = [&]() {
    if (!runtime_tensor_backing) {
      return false;
    }
    if (tensor->byte_offset != 0 || !tensor->planes.empty()) {
      return true;
    }
    const std::size_t tensor_bytes = tensor_dense_bytes_tight(*tensor);
    if (tensor_bytes == 0U || !tensor->storage) {
      return false;
    }
    int memory_index = tensor->route.memory_index;
    if (memory_index < 0) {
      memory_index = tensor->route.physical_index;
    }
    if (memory_index >= 0 &&
        static_cast<std::size_t>(memory_index) < tensor->storage->sima_segments.size()) {
      const auto& segment = tensor->storage->sima_segments[static_cast<std::size_t>(memory_index)];
      if (segment.size_bytes > 0U && segment.size_bytes != tensor_bytes) {
        return true;
      }
    }
    return tensor->storage->size_bytes > 0U && tensor->storage->size_bytes != tensor_bytes;
  };

  const auto is_plain_dtype_format = [](const std::string& fmt) {
    return fmt == "UINT8" || fmt == "INT8" || fmt == "UINT16" || fmt == "INT16" || fmt == "BF16" ||
           fmt == "BFLOAT16" || fmt == "FP32" || fmt == "FLOAT32" || fmt == "FP64" ||
           fmt == "FLOAT64" || fmt == "INT32" || fmt == "UINT32";
  };
  const auto packed_transport_format = [](const std::string& fmt) {
    const std::string up = upper_copy(fmt);
    return up == "BYTESTREAM" || up == "BYTE_STREAM" || up == "BYTE-STREAM" || up == "RAW_BYTES" ||
           up == "RAW-BYTES" || up == "OPAQUE_BYTES" || up == "OPAQUE-BYTES" || up == "MLA" ||
           up.find("TESS") != std::string::npos;
  };
  const auto caps_transport_format = [&sample]() -> std::string {
    if (sample.caps_string.empty()) {
      return {};
    }
    GstCaps* caps = gst_caps_from_string(sample.caps_string.c_str());
    if (!caps) {
      return {};
    }
    const GstStructure* s = gst_caps_get_structure(caps, 0);
    const char* caps_fmt = s ? gst_structure_get_string(s, "format") : nullptr;
    const std::string out = caps_fmt ? upper_copy(std::string_view(caps_fmt)) : std::string{};
    gst_caps_unref(caps);
    return out;
  };

  std::string fmt = !sample.payload_tag.empty() ? sample.payload_tag : sample.format;
  fmt = upper_copy(fmt);
  if (fmt.empty() && tensor->semantic.tess.has_value()) {
    fmt = upper_copy(tensor->semantic.tess->format);
  }
  const std::string caps_fmt = caps_transport_format();
  const bool raw_video_sample = sample_payload_type(sample) == PayloadType::Image;
  if (raw_video_sample && !tensor->semantic.tess.has_value()) {
    return false;
  }
  if (tensor->semantic.byte_stream.has_value()) {
    return true;
  }
  if ((fmt.empty() || is_plain_dtype_format(fmt)) && packed_transport_format(caps_fmt)) {
    fmt = caps_fmt;
  }
  if (tensor->semantic.tess.has_value()) {
    return true;
  }
  if (runtime_view_requires_envelope_transport()) {
    return true;
  }
  if (fmt.empty()) {
    return false;
  }
  if (!packed_transport_format(fmt) && !packed_transport_format(caps_fmt)) {
    return false;
  }
  return runtime_tensor_backing || packed_transport_format(caps_fmt);
}

bool sample_uses_joined_tensor_envelope_transport(const Sample& sample) {
  if (!sample_has_tensor_list(sample) || sample.tensors.size() <= 1U || !sample.fields.empty()) {
    return false;
  }

  const auto& first = sample.tensors.front();
  if (!first.storage) {
    return false;
  }

  const std::string segment_name =
      !first.route.segment_name.empty() ? first.route.segment_name : first.route.name;
  if (segment_name.empty()) {
    return false;
  }

  std::size_t running_offset = 0U;
  for (const auto& tensor : sample.tensors) {
    if (!tensor.storage || tensor.storage != first.storage) {
      return false;
    }
    const std::string tensor_segment_name =
        !tensor.route.segment_name.empty() ? tensor.route.segment_name : tensor.route.name;
    if (tensor_segment_name != segment_name) {
      return false;
    }
    const std::size_t tensor_bytes = tensor.dense_bytes_tight();
    if (tensor_bytes == 0U || tensor.byte_offset < 0) {
      return false;
    }
    if (static_cast<std::size_t>(tensor.byte_offset) != running_offset) {
      return false;
    }
    running_offset += tensor_bytes;
  }
  return running_offset > 0U;
}

SampleSpec tensor_envelope_spec_from_sample_or_throw(const Sample& sample, const char* where) {
  const std::string tag = where ? where : "SampleSpec";
  const TensorList tensors = tensors_from_sample(sample, false);
  if (tensors.empty()) {
    throw std::invalid_argument(tag + ": tensor envelope transport missing tensor");
  }

  InputOptions first_tensor_opt;
  first_tensor_opt.payload_type = sample_payload_type(sample);
  if (first_tensor_opt.payload_type == PayloadType::Auto) {
    first_tensor_opt.payload_type = PayloadType::Tensor;
  }
  if (!sample.payload_tag.empty()) {
    first_tensor_opt.format = sample.payload_tag;
  } else if (!sample.format.empty()) {
    first_tensor_opt.format = sample.format;
  } else if (tensors.front().semantic.byte_stream.has_value()) {
    first_tensor_opt.format = FormatTag::ByteStream;
  }

  SampleSpec spec = derive_tensor_spec_or_throw(tensors.front(), first_tensor_opt, tag.c_str());
  spec.tensor_envelope_transport = true;

  pipeline_internal::TensorBufferView view;
  std::string view_err;
  if (pipeline_internal::tensor_buffer_view_from_sample(sample, &view, &view_err) && view.buffer) {
    spec.required_bytes_actual = static_cast<std::size_t>(gst_buffer_get_size(view.buffer));
  }
  if (spec.required_bytes_actual == 0U) {
    std::unordered_set<int> seen_memory_indices;
    std::size_t total_bytes = 0U;
    for (const auto& tensor : tensors) {
      std::size_t segment_bytes = 0U;
      if (tensor.storage) {
        const int memory_index = (tensor.route.memory_index >= 0) ? tensor.route.memory_index
                                                                  : tensor.route.physical_index;
        if (!seen_memory_indices.insert(memory_index).second) {
          continue;
        }
        segment_bytes = tensor.storage->size_bytes;
        if (memory_index >= 0 &&
            static_cast<std::size_t>(memory_index) < tensor.storage->sima_segments.size() &&
            tensor.storage->sima_segments[static_cast<std::size_t>(memory_index)].size_bytes > 0U) {
          segment_bytes =
              tensor.storage->sima_segments[static_cast<std::size_t>(memory_index)].size_bytes;
        }
      } else {
        // Collapsed packed tensors can be CPU-owned but still represent transport bytes.
        segment_bytes = tensor.dense_bytes_tight();
      }
      if (segment_bytes == 0U) {
        throw std::invalid_argument(tag + ": tensor envelope transport has zero logical bytes");
      }
      total_bytes += segment_bytes;
    }
    spec.required_bytes_actual = total_bytes;
  }
  if (spec.required_bytes_actual == 0U) {
    throw std::invalid_argument(tag + ": tensor envelope transport has zero runtime bytes");
  }
  if (spec.required_bytes_actual > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(tag + ": tensor envelope transport exceeds transport size limit");
  }
  spec.caps_key = capkey_from_spec(spec);
  spec.caps_string = caps_string_from_spec(spec);
  return spec;
}

int limit_for_axis(const InputStreamOptions::ResolvedShapeLimits& limits, char axis) {
  switch (axis) {
  case 'w':
    return limits.max_width;
  case 'h':
    return limits.max_height;
  case 'd':
    return limits.max_depth;
  default:
    return -1;
  }
}

void validate_dim_with_effective_max(const std::string& tag, const char* field, int value,
                                     const InputStreamOptions::ResolvedShapeLimits& limits,
                                     char axis, const char* fix_hint) {
  const int limit = limit_for_axis(limits, axis);
  if (value <= 0 || limit <= 0 || value <= limit)
    return;

  std::ostringstream oss;
  oss << tag << ": " << field << " exceeds effective max (" << value << " > " << limit << ")";
  if (fix_hint && *fix_hint) {
    oss << ". Fix: " << fix_hint;
  }
  throw std::invalid_argument(oss.str());
}

std::string fmt_from_tensor_image(const simaai::neat::Tensor& input) {
  if (!input.semantic.image.has_value())
    return {};
  switch (input.semantic.image->format) {
  case simaai::neat::ImageSpec::PixelFormat::RGB:
    return "RGB";
  case simaai::neat::ImageSpec::PixelFormat::BGR:
    return "BGR";
  case simaai::neat::ImageSpec::PixelFormat::GRAY8:
    return "GRAY8";
  case simaai::neat::ImageSpec::PixelFormat::NV12:
    return "NV12";
  case simaai::neat::ImageSpec::PixelFormat::I420:
    return "I420";
  case simaai::neat::ImageSpec::PixelFormat::UNKNOWN:
    return {};
  }
  return {};
}

std::string fmt_from_dtype(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt8:
    return "UINT8";
  case TensorDType::Int8:
    return "INT8";
  case TensorDType::UInt16:
    return "UINT16";
  case TensorDType::Int16:
    return "INT16";
  case TensorDType::Int32:
    return "INT32";
  case TensorDType::BFloat16:
    return "BF16";
  case TensorDType::Float32:
    return "FP32";
  case TensorDType::Float64:
    return "FP64";
  }
  return {};
}

std::string layout_caps_value(TensorLayout layout) {
  switch (layout) {
  case TensorLayout::HWC:
    return "HWC";
  case TensorLayout::CHW:
    return "CHW";
  case TensorLayout::HW:
    return "HW";
  case TensorLayout::Unknown:
    return {};
  }
  return {};
}

std::string caps_format_value_from_spec(const SampleSpec& spec) {
  return normalize_caps_format_for_media(spec.media_type, spec.format);
}

std::string dtype_caps_value_from_spec(const SampleSpec& spec) {
  const std::string format = caps_format_value_from_spec(spec);
  const std::string fmt_up = upper_copy(format);
  if (!fmt_up.empty()) {
    // Preserve EVXX and explicit dtype tokens when provided by the route/config.
    if (fmt_up.find("EVXX_") == 0 || fmt_up.find("INT") != std::string::npos ||
        fmt_up.find("UINT") != std::string::npos || fmt_up.find("FP") != std::string::npos ||
        fmt_up.find("FLOAT") != std::string::npos || fmt_up.find("BF16") != std::string::npos ||
        fmt_up.find("BFLOAT16") != std::string::npos) {
      return format;
    }
  }
  return fmt_from_dtype(spec.dtype);
}

uint16_t fp32_to_bf16_rne(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7FFFu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

bool tensor_plane_is_tight(const simaai::neat::Plane& plane, TensorDType dtype) {
  if (plane.shape.size() < 2)
    return false;
  if (plane.byte_offset < 0)
    return false;
  const int64_t h = plane.shape[0];
  const int64_t w = plane.shape[1];
  if (h <= 0 || w <= 0)
    return false;
  const size_t elem = dtype_bytes(dtype);
  const int64_t min_stride = static_cast<int64_t>(w * elem);
  const int64_t stride = !plane.strides_bytes.empty() ? plane.strides_bytes[0] : min_stride;
  if (stride < min_stride)
    return false;
  if (plane.strides_bytes.size() > 1 && plane.strides_bytes[1] != static_cast<int64_t>(elem)) {
    return false;
  }
  return true;
}

size_t tensor_plane_bytes_tight(const simaai::neat::Plane& plane, TensorDType dtype) {
  if (plane.shape.size() < 2)
    return 0;
  const int64_t h = plane.shape[0];
  const int64_t w = plane.shape[1];
  if (h <= 0 || w <= 0)
    return 0;
  const size_t elem = dtype_bytes(dtype);
  const int64_t min_stride = static_cast<int64_t>(w * elem);
  const int64_t stride = !plane.strides_bytes.empty() ? plane.strides_bytes[0] : min_stride;
  if (stride < min_stride)
    return 0;
  return static_cast<size_t>(stride) * static_cast<size_t>(h);
}

size_t tensor_dense_bytes_tight(const simaai::neat::Tensor& input) {
  if (!input.is_dense() || input.shape.empty())
    return 0;
  size_t total = dtype_bytes(input.dtype);
  for (const auto dim : input.shape) {
    if (dim <= 0)
      return 0;
    total *= static_cast<size_t>(dim);
  }
  return total;
}

int plane_dim(const simaai::neat::Plane& plane, size_t idx) {
  if (plane.shape.size() <= idx)
    return -1;
  const int64_t v = plane.shape[idx];
  return (v > 0) ? static_cast<int>(v) : -1;
}

PlaneInfo plane_info_from_tight(const simaai::neat::Plane& plane, TensorDType dtype) {
  PlaneInfo info;
  info.role = plane.role;
  info.height = plane_dim(plane, 0);
  info.width = plane_dim(plane, 1);
  const int64_t min_stride =
      static_cast<int64_t>(info.width) * static_cast<int64_t>(dtype_bytes(dtype));
  const int64_t stride = !plane.strides_bytes.empty() ? plane.strides_bytes[0] : min_stride;
  info.stride_bytes = stride;
  info.offset_bytes = plane.byte_offset;
  info.size_bytes = tensor_plane_bytes_tight(plane, dtype);
  return info;
}

void debug_pool_log(const char* msg) {
  if (pipeline_internal::env_bool("SIMA_DEBUG_INPUT_POOL", false)) {
    std::fprintf(stderr, "%s\n", msg);
  }
}

void debug_pool_log_alloc_policy(const InputOptions& opt, bool tensor_media,
                                 GstMemoryFlags target_flag, const char* source,
                                 bool use_simaai_pool_effective) {
  if (!pipeline_internal::env_bool("SIMA_DEBUG_INPUT_POOL", false) &&
      !pipeline_internal::env_bool("SIMA_INPUTSTREAM_ALLOC_DEBUG", false)) {
    return;
  }
  std::fprintf(stderr,
               "[DBG] input_alloc_policy policy=%s use_simaai_pool=%d effective_pool=%d media=%s "
               "format=%s target=0x%x source=%s buffer_name=%s\n",
               input_memory_policy_name(opt.memory_policy), opt.use_simaai_pool ? 1 : 0,
               use_simaai_pool_effective ? 1 : 0, tensor_media ? "tensor" : "other",
               opt.format.str().c_str(), static_cast<unsigned>(target_flag),
               source ? source : "unknown",
               opt.buffer_name.empty() ? "<default>" : opt.buffer_name.c_str());
}

bool pool_stats_enabled() {
  return pipeline_internal::env_bool("SIMA_INPUTSTREAM_POOL_DEBUG", false) ||
         pipeline_internal::env_bool("SIMA_DEBUG_INPUT_POOL", false) ||
         pipeline_internal::env_bool("SIMA_INPUTSTREAM_ALLOC_DEBUG", false);
}

int pool_wait_log_ms() {
  return pipeline_internal::env_int("SIMA_INPUTSTREAM_POOL_WAIT_LOG_MS", 5);
}

GQuark pool_qdata_key() {
  static GQuark key = g_quark_from_static_string("sima.input_pool");
  return key;
}

struct PoolStats {
  std::uint64_t acquired_ok = 0;
  std::uint64_t acquired_fail = 0;
  std::uint64_t released = 0;
  std::int64_t inflight = 0;
  std::size_t last_bytes = 0;
};

std::mutex& pool_stats_mu() {
  static std::mutex mu;
  return mu;
}

std::unordered_map<GstBufferPool*, PoolStats>& pool_stats_map() {
  static std::unordered_map<GstBufferPool*, PoolStats> stats;
  return stats;
}

void log_pool_stats(const char* stage, GstBufferPool* pool, const PoolStats& stats,
                    std::size_t bytes, double wait_ms, bool ok) {
  if (!pool_stats_enabled())
    return;
  const int wait_thresh = pool_wait_log_ms();
  if (wait_thresh > 0 && wait_ms < static_cast<double>(wait_thresh) && ok) {
    return;
  }
  std::fprintf(stderr,
               "[POOL] %s pool=%p ok=%d wait_ms=%.3f bytes=%zu inflight=%lld "
               "acq_ok=%llu acq_fail=%llu rel=%llu\n",
               stage ? stage : "pool_event", static_cast<void*>(pool), ok ? 1 : 0, wait_ms, bytes,
               static_cast<long long>(stats.inflight),
               static_cast<unsigned long long>(stats.acquired_ok),
               static_cast<unsigned long long>(stats.acquired_fail),
               static_cast<unsigned long long>(stats.released));
}

} // namespace

std::vector<int64_t> tensor_shape_from_compat_dims(int width, int height, int depth,
                                                   TensorLayout layout) {
  return tensor_shape_from_compat_dims_local(width, height, depth, layout);
}

TensorCompatDims tensor_compat_dims_from_shape(const std::vector<int64_t>& shape,
                                               TensorLayout layout) {
  return tensor_compat_dims_from_shape_local(shape, layout);
}

void track_input_pool_acquire(GstBufferPool* pool, GstBuffer* buffer, size_t bytes,
                              const char* where, double wait_ms, bool ok) {
  if (!pool || !pool_stats_enabled())
    return;
  PoolStats stats{};
  {
    std::lock_guard<std::mutex> lock(pool_stats_mu());
    auto& entry = pool_stats_map()[pool];
    if (ok) {
      entry.acquired_ok += 1;
      entry.inflight += 1;
    } else {
      entry.acquired_fail += 1;
    }
    entry.last_bytes = bytes;
    stats = entry;
  }
  if (buffer && ok) {
    gst_mini_object_set_qdata(GST_MINI_OBJECT(buffer), pool_qdata_key(), pool, nullptr);
  }
  log_pool_stats(where ? where : "pool_acquire", pool, stats, bytes, wait_ms, ok);
}

void track_input_pool_release(GstBuffer* buffer, const char* where) {
  if (!buffer || !pool_stats_enabled())
    return;
  auto* pool = static_cast<GstBufferPool*>(
      gst_mini_object_get_qdata(GST_MINI_OBJECT(buffer), pool_qdata_key()));
  if (!pool)
    return;
  PoolStats stats{};
  {
    std::lock_guard<std::mutex> lock(pool_stats_mu());
    auto& entry = pool_stats_map()[pool];
    entry.released += 1;
    if (entry.inflight > 0) {
      entry.inflight -= 1;
    }
    stats = entry;
  }
  log_pool_stats(where ? where : "pool_release", pool, stats,
                 static_cast<std::size_t>(gst_buffer_get_size(buffer)), 0.0, true);
}

namespace {

void debug_pool_state(const char* stage, GstBufferPool* pool, const InputOptions& opt,
                      size_t bytes) {
  if (!pipeline_internal::env_bool("SIMA_INPUTSTREAM_ALLOC_DEBUG", false))
    return;
  if (!pool) {
    std::fprintf(stderr, "[DBG] input_pool %s pool=null bytes=%zu min=%d max=%d\n",
                 stage ? stage : "unknown", bytes, opt.pool_min_buffers, opt.pool_max_buffers);
    return;
  }
  GstStructure* config = gst_buffer_pool_get_config(pool);
  guint min_buffers = 0;
  guint max_buffers = 0;
  guint pool_size = 0;
  GstCaps* caps = nullptr;
  if (config) {
    gst_buffer_pool_config_get_params(config, &caps, &pool_size, &min_buffers, &max_buffers);
  }
  std::fprintf(stderr,
               "[DBG] input_pool %s pool=%p active=%d bytes=%zu size=%zu min=%u max=%u opt_min=%d "
               "opt_max=%d\n",
               stage ? stage : "unknown", static_cast<void*>(pool),
               gst_buffer_pool_is_active(pool) ? 1 : 0, bytes, static_cast<size_t>(pool_size),
               min_buffers, max_buffers, opt.pool_min_buffers, opt.pool_max_buffers);
  if (config)
    gst_structure_free(config);
}

void debug_pool_timing(const char* stage, const InputOptions& opt, size_t bytes,
                       const std::chrono::steady_clock::time_point& start, bool ok,
                       bool used_pool) {
  if (!pipeline_internal::env_bool("SIMA_INPUTSTREAM_ALLOC_DEBUG", false))
    return;
  const auto end = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(end - start).count();
  std::fprintf(stderr, "[DBG] input_buffer %s bytes=%zu pool=%s ok=%d min=%d max=%d ms=%.3f\n",
               stage, bytes, used_pool ? "true" : "false", ok ? 1 : 0, opt.pool_min_buffers,
               opt.pool_max_buffers, ms);
}

void free_simaai_pool(GstBufferPool* pool) {
#if SIMA_HAS_SIMAAI_POOL
  if (!pool)
    return;
  gst_simaai_free_buffer_pool(pool);
#else
  (void)pool;
#endif
}

} // namespace

namespace {

std::size_t hash_combine(std::size_t seed, std::size_t value) {
  return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

std::size_t hash_string(const std::string& s) {
  return std::hash<std::string>{}(s);
}

} // namespace

CapKey capkey_from_spec(const SampleSpec& spec) {
  CapKey key;
  key.kind = spec.kind;
  if (spec.kind == SampleMediaKind::RawVideo) {
    key.media_type = spec.media_type;
    key.format = spec.format;
    key.width = spec.width;
    key.height = spec.height;
    key.fps_n = spec.fps_n;
    key.fps_d = spec.fps_d;
  } else if (spec.kind == SampleMediaKind::Tensor) {
    key.dtype = spec.dtype;
    key.layout = spec.layout;
    key.shape = spec.shape;
  } else if (spec.kind == SampleMediaKind::Encoded) {
    key.caps_hash = hash_string(spec.caps_string);
  }
  return key;
}

std::string caps_string_from_spec(const SampleSpec& spec) {
  if (spec.kind == SampleMediaKind::Encoded) {
    if (spec.caps_string.empty()) {
      throw std::runtime_error("SampleSpec: missing caps_string for encoded payload");
    }
    return spec.caps_string;
  }

  if (spec.media_type == "video/x-raw") {
    if (spec.format.empty() || spec.width <= 0 || spec.height <= 0) {
      throw std::runtime_error("SampleSpec: missing raw video caps fields");
    }
    std::ostringstream oss;
    oss << "video/x-raw,format=" << spec.format << ",width=" << spec.width
        << ",height=" << spec.height;
    if (spec.fps_n > 0) {
      if (spec.fps_d <= 0) {
        throw std::runtime_error("SampleSpec: invalid framerate denominator");
      }
      oss << ",framerate=" << spec.fps_n << "/" << spec.fps_d;
    }
    return oss.str();
  }

  if (spec.media_type == "application/vnd.simaai.tensor") {
    const std::vector<int64_t>& tensor_shape = spec.shape;
    if (spec.format.empty() || tensor_shape.empty()) {
      throw std::runtime_error("SampleSpec: missing tensor caps fields");
    }
    const std::string caps_format = caps_format_value_from_spec(spec);
    std::ostringstream oss;
    oss << "application/vnd.simaai.tensor,format=" << caps_format;
    oss << ",rank=" << tensor_shape.size();
    for (size_t i = 0; i < tensor_shape.size(); ++i) {
      oss << ",dim" << i << "=" << tensor_shape[i];
    }
    const std::string dtype = dtype_caps_value_from_spec(spec);
    if (!dtype.empty()) {
      oss << ",dtype=" << dtype;
    }
    return oss.str();
  }

  throw std::runtime_error("SampleSpec: unsupported media_type for caps");
}

std::string CapKey::to_string() const {
  std::ostringstream oss;
  switch (kind) {
  case SampleMediaKind::RawVideo:
    oss << "raw{media_type=" << media_type << ",format=" << format << ",width=" << width
        << ",height=" << height << ",fps=" << fps_n << "/" << fps_d << "}";
    break;
  case SampleMediaKind::Tensor:
    oss << "tensor{dtype=" << static_cast<int>(dtype) << ",layout=" << static_cast<int>(layout)
        << ",shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i)
        oss << ",";
      oss << shape[i];
    }
    oss << "]}";
    break;
  case SampleMediaKind::Encoded:
    oss << "encoded{caps_hash=" << caps_hash << "}";
    break;
  }
  return oss.str();
}

std::size_t CapKeyHash::operator()(const CapKey& key) const {
  std::size_t seed = 0;
  seed = hash_combine(seed, std::hash<int>{}(static_cast<int>(key.kind)));
  switch (key.kind) {
  case SampleMediaKind::RawVideo:
    seed = hash_combine(seed, hash_string(key.media_type));
    seed = hash_combine(seed, hash_string(key.format));
    seed = hash_combine(seed, std::hash<int>{}(key.width));
    seed = hash_combine(seed, std::hash<int>{}(key.height));
    seed = hash_combine(seed, std::hash<int>{}(key.fps_n));
    seed = hash_combine(seed, std::hash<int>{}(key.fps_d));
    break;
  case SampleMediaKind::Tensor:
    seed = hash_combine(seed, std::hash<int>{}(static_cast<int>(key.dtype)));
    for (const auto& dim : key.shape) {
      seed = hash_combine(seed, std::hash<std::int64_t>{}(dim));
    }
    break;
  case SampleMediaKind::Encoded:
    seed = hash_combine(seed, key.caps_hash);
    break;
  }
  return seed;
}

SampleSpec derive_tensor_spec_or_throw(const simaai::neat::Tensor& input, const InputOptions& opt,
                                       const char* where) {
  const std::string tag = where ? where : "derive_tensor_spec";
  if (!input.storage) {
    throw std::invalid_argument(tag + ": simaai::neat::Tensor missing storage");
  }
  if (input.device.type != simaai::neat::DeviceType::CPU &&
      input.storage->kind != simaai::neat::StorageKind::GstSample) {
    throw std::invalid_argument(tag + ": non-CPU tensors must be backed by GstSample storage");
  }
  if (input.semantic.encoded.has_value()) {
    throw std::invalid_argument(tag + ": encoded tensors require Sample caps_string");
  }

  SampleSpec spec;
  spec.dtype = input.dtype;
  spec.layout = input.layout;
  spec.shape = input.shape;

  std::string media = resolve_input_media_type(opt);
  if (media.empty()) {
    media = input.semantic.image.has_value() ? "video/x-raw" : "application/vnd.simaai.tensor";
  }
  spec.media_type = media;

  if (spec.media_type == "video/x-raw") {
    spec.kind = SampleMediaKind::RawVideo;

    std::string fmt = upper_copy(opt.format);
    std::string semantic_fmt = upper_copy(fmt_from_tensor_image(input));
    if (fmt.empty())
      fmt = semantic_fmt;
    if (fmt == "GRAY")
      fmt = "GRAY8";
    if (fmt.empty()) {
      throw std::invalid_argument(tag + ": simaai::neat::Tensor video input missing format");
    }
    if (fmt != "RGB" && fmt != "BGR" && fmt != "GRAY8" && fmt != "NV12" && fmt != "I420") {
      throw std::invalid_argument(tag + ": unsupported video format: " + fmt);
    }
    if (input.dtype != TensorDType::UInt8) {
      throw std::invalid_argument(tag + ": simaai::neat::Tensor video input must be UInt8");
    }
    if (!input.semantic.image.has_value()) {
      if (upper_copy(opt.format).empty()) {
        throw std::invalid_argument(tag +
                                    ": simaai::neat::Tensor video input missing ImageSpec and "
                                    "InputOptions.format");
      }
    } else if (!semantic_fmt.empty() && semantic_fmt != fmt) {
      throw std::invalid_argument(tag + ": simaai::neat::Tensor image format mismatch");
    }

    int h = shape_dim(input.shape, 0);
    int w = shape_dim(input.shape, 1);

    if (fmt == "NV12" || fmt == "I420") {
      if (input.byte_offset != 0) {
        throw std::invalid_argument(tag + ": composite tensor must have byte_offset == 0");
      }
      if ((fmt == "NV12" && input.planes.size() != 2) ||
          (fmt == "I420" && input.planes.size() != 3)) {
        throw std::invalid_argument(tag + ": invalid plane count for format");
      }
      const simaai::neat::Plane* y = input.try_plane(simaai::neat::PlaneRole::Y);
      if (!y) {
        throw std::invalid_argument(tag + ": missing Y plane");
      }
      if (!tensor_plane_is_tight(*y, input.dtype)) {
        throw std::invalid_argument(tag + ": Y plane is not tightly packed");
      }
      const int y_h = plane_dim(*y, 0);
      const int y_w = plane_dim(*y, 1);
      if (y_h <= 0 || y_w <= 0) {
        throw std::invalid_argument(tag + ": Y plane missing width/height");
      }
      if (h > 0 && h != y_h) {
        throw std::invalid_argument(tag + ": height does not match Y plane");
      }
      if (w > 0 && w != y_w) {
        throw std::invalid_argument(tag + ": width does not match Y plane");
      }
      h = y_h;
      w = y_w;
      if ((w % 2) != 0 || (h % 2) != 0) {
        throw std::invalid_argument(tag + ": NV12/I420 requires even width/height");
      }

      spec.planes.clear();
      size_t max_end = 0;
      auto add_plane = [&](const simaai::neat::Plane& plane, const char* label) {
        if (!tensor_plane_is_tight(plane, input.dtype)) {
          throw std::invalid_argument(tag + ": " + std::string(label) + " plane stride is invalid");
        }
        PlaneInfo info = plane_info_from_tight(plane, input.dtype);
        if (info.size_bytes == 0) {
          throw std::invalid_argument(tag + ": " + std::string(label) + " plane has invalid size");
        }
        spec.planes.push_back(info);
        const size_t end = static_cast<size_t>(info.offset_bytes) + info.size_bytes;
        if (end > max_end)
          max_end = end;
      };

      add_plane(*y, "Y");

      if (fmt == "NV12") {
        const simaai::neat::Plane* uv = input.try_plane(simaai::neat::PlaneRole::UV);
        if (!uv) {
          throw std::invalid_argument(tag + ": missing UV plane");
        }
        const int uv_h = plane_dim(*uv, 0);
        const int uv_w = plane_dim(*uv, 1);
        if (uv_h != h / 2 || uv_w != w) {
          throw std::invalid_argument(tag + ": UV plane shape mismatch");
        }
        add_plane(*uv, "UV");
      } else {
        const simaai::neat::Plane* u = input.try_plane(simaai::neat::PlaneRole::U);
        const simaai::neat::Plane* v = input.try_plane(simaai::neat::PlaneRole::V);
        if (!u || !v) {
          throw std::invalid_argument(tag + ": missing U/V planes");
        }
        const int u_h = plane_dim(*u, 0);
        const int u_w = plane_dim(*u, 1);
        const int v_h = plane_dim(*v, 0);
        const int v_w = plane_dim(*v, 1);
        if (u_h != h / 2 || v_h != h / 2 || u_w != w / 2 || v_w != w / 2) {
          throw std::invalid_argument(tag + ": U/V plane shape mismatch");
        }
        add_plane(*u, "U");
        add_plane(*v, "V");
      }

      spec.width = w;
      spec.height = h;
      spec.depth = -1;
      spec.required_bytes_actual = max_end;
    } else {
      if (input.is_composite()) {
        throw std::invalid_argument(tag + ": packed video must not use planes");
      }
      if (input.byte_offset != 0) {
        throw std::invalid_argument(tag + ": packed video must have byte_offset == 0");
      }
      if (h <= 0 || w <= 0) {
        throw std::invalid_argument(tag + ": video input missing width/height");
      }
      const int expected_depth = (fmt == "GRAY8") ? 1 : 3;
      const int shape_d = shape_dim(input.shape, 2);
      if ((fmt == "RGB" || fmt == "BGR") && shape_d <= 0) {
        throw std::invalid_argument(tag + ": video depth is required for RGB/BGR");
      }
      if (shape_d > 0 && shape_d != expected_depth) {
        throw std::invalid_argument(tag + ": video depth does not match format");
      }
      spec.width = w;
      spec.height = h;
      spec.depth = expected_depth;
      spec.planes.clear();
      PlaneInfo plane;
      plane.role = simaai::neat::PlaneRole::Unknown;
      plane.width = w;
      plane.height = h;
      const int64_t expected_stride =
          static_cast<int64_t>(w) * static_cast<int64_t>(expected_depth);
      const int64_t stride =
          !input.strides_bytes.empty() ? input.strides_bytes[0] : expected_stride;
      if (stride < expected_stride) {
        throw std::invalid_argument(tag + ": packed video stride too small");
      }
      if (input.strides_bytes.size() > 1 &&
          input.strides_bytes.back() != static_cast<int64_t>(dtype_bytes(input.dtype))) {
        throw std::invalid_argument(tag + ": packed video element stride mismatch");
      }
      plane.stride_bytes = stride;
      plane.offset_bytes = input.byte_offset;
      const size_t bytes = static_cast<size_t>(stride) * static_cast<size_t>(h);
      if (bytes == 0) {
        throw std::invalid_argument(tag + ": video input has invalid byte size");
      }
      plane.size_bytes = bytes;
      spec.planes.push_back(plane);
      spec.required_bytes_actual = bytes;
    }

    const auto limits = pipeline_internal::resolve_shape_limits(
        pipeline_internal::normalize_shape_bounds(opt), spec);
    validate_dim_with_effective_max(
        tag, "width", spec.width, limits, 'w',
        "resize input or increase max_width/width (Model::Options::input_max_width).");
    validate_dim_with_effective_max(
        tag, "height", spec.height, limits, 'h',
        "resize input or increase max_height/height (Model::Options::input_max_height).");
    if (spec.depth > 0) {
      validate_dim_with_effective_max(
          tag, "depth", spec.depth, limits, 'd',
          "reduce channels or increase max_depth/depth (Model::Options::input_max_depth).");
    }
    if (input.storage && input.storage->size_bytes > 0) {
      const size_t end = static_cast<size_t>(input.byte_offset) + spec.required_bytes_actual;
      if (end > input.storage->size_bytes) {
        throw std::invalid_argument(tag + ": storage too small for video input");
      }
    }
    spec.format = fmt;
  } else if (spec.media_type == "application/vnd.simaai.tensor") {
    spec.kind = SampleMediaKind::Tensor;
    if (!input.is_dense()) {
      throw std::invalid_argument(tag + ": tensor input must be dense");
    }
    if (!input.is_contiguous()) {
      throw std::invalid_argument(tag + ": tensor input must be contiguous");
    }

    std::vector<int64_t> normalized_shape = input.shape;
    const bool layout_is_explicit = input.layout != TensorLayout::Unknown;
    const size_t expected_rank = (input.layout == TensorLayout::HW) ? 2u : 3u;
    if (layout_is_explicit && normalized_shape.size() == expected_rank + 1) {
      const int64_t batch = normalized_shape.front();
      if (batch <= 0) {
        throw std::invalid_argument(tag + ": invalid leading batch dimension");
      }
      if (batch == 1) {
        normalized_shape.erase(normalized_shape.begin());
      }
    }

    const TensorCompatDims compat = tensor_compat_dims_from_shape(normalized_shape, input.layout);
    if (input.layout == TensorLayout::HWC) {
      if (normalized_shape.size() != 3 && normalized_shape.size() != 4) {
        throw std::invalid_argument(tag + ": HWC tensor must have shape [H,W,C] or [N,H,W,C]");
      }
    } else if (input.layout == TensorLayout::CHW) {
      if (normalized_shape.size() != 3 && normalized_shape.size() != 4) {
        throw std::invalid_argument(tag + ": CHW tensor must have shape [C,H,W] or [N,C,H,W]");
      }
    } else if (input.layout == TensorLayout::HW) {
      if (normalized_shape.size() != 1 && normalized_shape.size() != 2 &&
          normalized_shape.size() != 3) {
        throw std::invalid_argument(tag + ": HW tensor must have shape [N], [H,W], or [B,H,W]");
      }
    }

    if (compat.width <= 0 || compat.height <= 0 || compat.depth <= 0) {
      throw std::invalid_argument(tag + ": tensor input missing canonical shape");
    }
    SampleSpec shape_seed = spec;
    shape_seed.layout = input.layout;
    shape_seed.shape = normalized_shape;
    const auto limits = pipeline_internal::resolve_shape_limits(
        pipeline_internal::normalize_shape_bounds(opt), shape_seed);
    validate_dim_with_effective_max(
        tag, "tensor width", compat.width, limits, 'w',
        "resize input or increase max_width/width (Model::Options::input_max_width).");
    validate_dim_with_effective_max(
        tag, "tensor height", compat.height, limits, 'h',
        "resize input or increase max_height/height (Model::Options::input_max_height).");
    validate_dim_with_effective_max(
        tag, "tensor depth", compat.depth, limits, 'd',
        "reduce channels or increase max_depth/depth (Model::Options::input_max_depth).");

    std::string fmt = upper_copy(opt.format);
    const std::string dtype_fmt = upper_copy(fmt_from_dtype(input.dtype));
    if (fmt.empty() && input.semantic.byte_stream.has_value()) {
      fmt = "BYTESTREAM";
    }
    if (fmt.empty())
      fmt = dtype_fmt;
    if (fmt.empty()) {
      throw std::invalid_argument(tag + ": tensor input missing format");
    }
    if (!opt.format.empty() && !dtype_fmt.empty()) {
      const bool fmt_is_dtype = fmt == "UINT8" || fmt == "INT8" || fmt == "UINT16" ||
                                fmt == "INT16" || fmt == "INT32" || fmt == "BF16" ||
                                fmt == "FP32" || fmt == "FP64";
      const bool fmt_is_mla = fmt == "MLA";
      const bool fmt_is_tess_i8 = is_tessellated_int8_format(fmt);
      const bool fmt_is_tess_bf16 = is_tessellated_bf16_format(fmt);
      if (fmt_is_dtype && fmt != dtype_fmt) {
        throw std::invalid_argument(tag + ": tensor format does not match dtype");
      }
      if (fmt_is_mla && input.dtype != TensorDType::Int8 && input.dtype != TensorDType::UInt8 &&
          input.dtype != TensorDType::BFloat16) {
        throw std::invalid_argument(tag + ": tensor format does not match dtype");
      }
      if (fmt_is_tess_i8 && input.dtype != TensorDType::Int8 && input.dtype != TensorDType::UInt8) {
        throw std::invalid_argument(tag + ": tensor format does not match dtype");
      }
      if (fmt_is_tess_bf16 && input.dtype != TensorDType::BFloat16) {
        throw std::invalid_argument(tag + ": tensor format does not match dtype");
      }
    }
    if (input.semantic.byte_stream.has_value()) {
      if (input.dtype != TensorDType::UInt8 && input.dtype != TensorDType::Int8) {
        throw std::invalid_argument(tag + ": byte_stream tensor must be UInt8 or Int8");
      }
      if (input.layout != TensorLayout::Unknown) {
        throw std::invalid_argument(tag + ": byte_stream tensor layout must be Unknown");
      }
      if (input.shape.size() != 1U || input.shape[0] <= 0) {
        throw std::invalid_argument(tag + ": byte_stream tensor must have shape [num_bytes]");
      }
    }

    const size_t bytes = tensor_dense_bytes_tight(input);
    if (bytes == 0) {
      throw std::invalid_argument(tag + ": tensor input has invalid byte size");
    }
    if (input.storage && input.storage->size_bytes > 0) {
      const size_t end = static_cast<size_t>(input.byte_offset) + bytes;
      if (end > input.storage->size_bytes) {
        throw std::invalid_argument(tag + ": storage too small for tensor input");
      }
    }
    spec.shape = std::move(normalized_shape);
    spec.format = fmt;
    spec.required_bytes_actual = bytes;
  } else {
    throw std::invalid_argument(tag + ": unsupported media_type: " + spec.media_type);
  }

  spec.fps_n = opt.fps_n;
  spec.fps_d = opt.fps_d;
  spec.caps_key = capkey_from_spec(spec);
  spec.caps_string = caps_string_from_spec(spec);
  return spec;
}

simaai::neat::Tensor tensor_from_cv_mat(const cv::Mat& mat, const InputOptions& opt,
                                        const char* where) {
  const std::string tag = where ? where : "tensor_from_cv_mat";
  if (mat.empty() || mat.data == nullptr) {
    throw std::invalid_argument(tag + ": input frame is empty");
  }

  std::string media = resolve_input_media_type(opt);
  if (media.empty()) {
    media = (mat.depth() == CV_32F) ? "application/vnd.simaai.tensor" : "video/x-raw";
  }

  if (media == "video/x-raw") {
    if (mat.depth() != CV_8U) {
      throw std::invalid_argument(tag + ": video input must be CV_8U");
    }
    if (mat.channels() != 1 && mat.channels() != 3) {
      throw std::invalid_argument(tag + ": video input must be 1 or 3 channels");
    }
    std::string fmt = upper_copy(opt.format);
    if (fmt.empty()) {
      fmt = (mat.channels() == 1) ? "GRAY8" : "BGR";
    }
    if (fmt == "GRAY")
      fmt = "GRAY8";
    if (fmt != "RGB" && fmt != "BGR" && fmt != "GRAY8") {
      throw std::invalid_argument(tag + ": unsupported video format: " + fmt);
    }
    if ((fmt == "RGB" || fmt == "BGR") && mat.channels() != 3) {
      throw std::invalid_argument(tag + ": RGB/BGR requires 3 channels");
    }
    if (fmt == "GRAY8" && mat.channels() != 1) {
      throw std::invalid_argument(tag + ": GRAY8 requires 1 channel");
    }
    simaai::neat::ImageSpec::PixelFormat pf = simaai::neat::ImageSpec::PixelFormat::BGR;
    if (fmt == "RGB")
      pf = simaai::neat::ImageSpec::PixelFormat::RGB;
    if (fmt == "GRAY8")
      pf = simaai::neat::ImageSpec::PixelFormat::GRAY8;
    if (opt.memory_policy == InputMemoryPolicy::Ev74) {
      return simaai::neat::from_cv_mat(mat, pf, simaai::neat::TensorMemory::EV74);
    }
    if (opt.memory_policy == InputMemoryPolicy::Dms0) {
      return simaai::neat::from_cv_mat(mat, pf, simaai::neat::TensorMemory::MLA);
    }
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    return simaai::neat::from_cv_mat(mat, pf, true);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
  }

  if (media == "application/vnd.simaai.tensor") {
    if (mat.channels() <= 0) {
      throw std::invalid_argument(tag + ": tensor input must have a positive channel count");
    }
    std::string fmt = upper_copy(opt.format);
    const bool want_fp32 = (fmt.empty() || fmt == "FP32" || fmt == "EVXX_FLOAT32");
    if (!want_fp32) {
      throw std::invalid_argument(
          tag + ": cv::Mat tensor input supports FP32/EVXX_FLOAT32 only and must remain exact; "
                "pass simaai::neat::Tensor for BF16 or quantized tensor ingress");
    }
    if (mat.depth() != CV_32F) {
      throw std::invalid_argument(
          tag +
          ": cv::Mat tensor input must already be CV_32F; implicit numeric conversion is disabled");
    }

    auto holder = std::make_shared<cv::Mat>(mat);
    const std::size_t bytes = holder->step * static_cast<std::size_t>(holder->rows);
    simaai::neat::Tensor out;
    out.storage = simaai::neat::make_cpu_external_storage(holder->data, bytes, holder, true);
    out.dtype = TensorDType::Float32;
    out.device = {simaai::neat::DeviceType::CPU, 0};
    out.byte_offset = 0;
    out.read_only = true;

    const int rows = holder->rows;
    const int cols = holder->cols;
    const int channels = holder->channels();
    if (channels == 1) {
      out.layout = TensorLayout::HW;
      out.shape = {rows, cols};
      out.strides_bytes = {static_cast<int64_t>(holder->step),
                           static_cast<int64_t>(holder->elemSize1())};
    } else {
      out.layout = TensorLayout::HWC;
      out.shape = {rows, cols, channels};
      out.strides_bytes = {static_cast<int64_t>(holder->step),
                           static_cast<int64_t>(holder->elemSize()),
                           static_cast<int64_t>(holder->elemSize1())};
    }
    if (opt.memory_policy == InputMemoryPolicy::Ev74 ||
        opt.memory_policy == InputMemoryPolicy::Dms0) {
      const Device target = opt.memory_policy == InputMemoryPolicy::Ev74
                                ? Device{DeviceType::SIMA_CVU, 0}
                                : Device{DeviceType::SIMA_MLA, 0};
      const std::size_t device_bytes = out.dense_bytes_tight();
      if (device_bytes == 0U) {
        throw std::runtime_error(tag + ": unable to determine dense byte size for device tensor");
      }
      std::vector<Segment> segments{{"ifm0", device_bytes}};
      return pipeline_internal::transfer_to_device(out, target, &segments,
                                                   /*required_segment_names=*/nullptr);
    }
    return out;
  }

  throw std::invalid_argument(tag + ": unsupported media_type: " + media);
}

SampleSpec derive_sample_spec_or_throw(const Sample& sample) {
  if (sample.kind == SampleKind::Tensor && sample.tensor.has_value()) {
    if (sample_uses_single_tensor_envelope_transport(sample)) {
      return tensor_envelope_spec_from_sample_or_throw(sample, "SampleSpec");
    }

    const simaai::neat::Tensor& input = *sample.tensor;
    if (input.semantic.encoded.has_value()) {
      if (sample.caps_string.empty()) {
        throw std::invalid_argument("SampleSpec: encoded sample requires caps_string");
      }
      if (input.dtype != TensorDType::UInt8) {
        throw std::invalid_argument("SampleSpec: encoded simaai::neat::Tensor must be UInt8");
      }
      if (!input.is_dense() || !input.planes.empty()) {
        throw std::invalid_argument("SampleSpec: encoded simaai::neat::Tensor must be dense");
      }
      if (input.shape.size() != 1 || input.shape[0] <= 0) {
        throw std::invalid_argument(
            "SampleSpec: encoded simaai::neat::Tensor must have shape [num_bytes]");
      }
      SampleSpec spec;
      spec.kind = SampleMediaKind::Encoded;
      if (!sample.media_type.empty()) {
        spec.media_type = sample.media_type;
      } else {
        GstCaps* caps = gst_caps_from_string(sample.caps_string.c_str());
        if (!caps) {
          throw std::invalid_argument("SampleSpec: invalid caps_string");
        }
        const GstStructure* s = gst_caps_get_structure(caps, 0);
        const char* name = s ? gst_structure_get_name(s) : nullptr;
        if (name) {
          spec.media_type = name;
        }
        gst_caps_unref(caps);
        if (spec.media_type.empty()) {
          throw std::invalid_argument("SampleSpec: encoded caps missing media_type");
        }
      }
      spec.format = "ENCODED";
      spec.dtype = input.dtype;
      spec.layout = input.layout;
      spec.shape = input.shape;
      spec.required_bytes_actual = tensor_dense_bytes_tight(input);
      if (spec.required_bytes_actual == 0) {
        spec.required_bytes_actual = static_cast<size_t>(input.shape[0]);
      }
      spec.caps_string = sample.caps_string;
      spec.caps_key = capkey_from_spec(spec);
      return spec;
    }

    InputOptions opt;
    opt.payload_type = sample_payload_type(sample);
    if (!sample.payload_tag.empty()) {
      opt.format = sample.payload_tag;
    } else if (!sample.format.empty()) {
      opt.format = sample.format;
    }
    return derive_tensor_spec_or_throw(input, opt, "SampleSpec");
  }

  if (sample_has_tensor_list(sample)) {
    if (sample.tensors.size() > 1U || sample_uses_single_tensor_envelope_transport(sample) ||
        sample_uses_joined_tensor_envelope_transport(sample)) {
      return tensor_envelope_spec_from_sample_or_throw(sample, "SampleSpec");
    }
    if (sample.tensors.size() > 1U) {
      // TensorSet transport still pushes a single envelope buffer, but the
      // logical appsrc caps must match the representative tensor contract used
      // by downstream stages. The real envelope bytes are tracked separately.
      InputOptions first_tensor_opt;
      first_tensor_opt.payload_type = sample_payload_type(sample);
      if (first_tensor_opt.payload_type == PayloadType::Auto) {
        first_tensor_opt.payload_type = PayloadType::Tensor;
      }
      if (!sample.payload_tag.empty()) {
        first_tensor_opt.format = sample.payload_tag;
      } else if (!sample.format.empty()) {
        first_tensor_opt.format = sample.format;
      }
      SampleSpec spec =
          derive_tensor_spec_or_throw(sample.tensors.front(), first_tensor_opt, "SampleSpec");
      spec.tensor_envelope_transport = true;

      pipeline_internal::TensorBufferView view;
      std::string view_err;
      if (pipeline_internal::tensor_buffer_view_from_sample(sample, &view, &view_err) &&
          view.buffer) {
        spec.required_bytes_actual = static_cast<std::size_t>(gst_buffer_get_size(view.buffer));
      }
      if (spec.required_bytes_actual == 0U) {
        std::unordered_set<int> seen_memory_indices;
        std::size_t total_bytes = 0U;
        for (const auto& tensor : sample.tensors) {
          if (!tensor.storage) {
            throw std::invalid_argument(
                "SampleSpec: tensor-set transport missing storage for runtime segment sizing");
          }
          const int memory_index = (tensor.route.memory_index >= 0) ? tensor.route.memory_index
                                                                    : tensor.route.physical_index;
          if (!seen_memory_indices.insert(memory_index).second) {
            continue;
          }
          std::size_t segment_bytes = tensor.storage->size_bytes;
          if (memory_index >= 0 &&
              static_cast<std::size_t>(memory_index) < tensor.storage->sima_segments.size() &&
              tensor.storage->sima_segments[static_cast<std::size_t>(memory_index)].size_bytes >
                  0U) {
            segment_bytes =
                tensor.storage->sima_segments[static_cast<std::size_t>(memory_index)].size_bytes;
          }
          if (segment_bytes == 0U) {
            throw std::invalid_argument(
                "SampleSpec: tensor-set transport missing runtime segment size");
          }
          total_bytes += segment_bytes;
        }
        spec.required_bytes_actual = total_bytes;
      }
      if (spec.required_bytes_actual == 0U) {
        throw std::invalid_argument("SampleSpec: tensor-set envelope has zero runtime bytes");
      }
      if (spec.required_bytes_actual > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("SampleSpec: tensor-set envelope exceeds transport size limit");
      }
      spec.caps_key = capkey_from_spec(spec);
      spec.caps_string = caps_string_from_spec(spec);
      return spec;
    }
    const simaai::neat::Tensor& input = sample.tensors.front();
    if (input.semantic.encoded.has_value()) {
      if (sample.caps_string.empty()) {
        throw std::invalid_argument("SampleSpec: encoded sample requires caps_string");
      }
      if (input.dtype != TensorDType::UInt8) {
        throw std::invalid_argument("SampleSpec: encoded simaai::neat::Tensor must be UInt8");
      }
      if (!input.is_dense() || !input.planes.empty()) {
        throw std::invalid_argument("SampleSpec: encoded simaai::neat::Tensor must be dense");
      }
      if (input.shape.size() != 1 || input.shape[0] <= 0) {
        throw std::invalid_argument(
            "SampleSpec: encoded simaai::neat::Tensor must have shape [num_bytes]");
      }
      SampleSpec spec;
      spec.kind = SampleMediaKind::Encoded;
      if (!sample.media_type.empty()) {
        spec.media_type = sample.media_type;
      } else {
        GstCaps* caps = gst_caps_from_string(sample.caps_string.c_str());
        if (!caps) {
          throw std::invalid_argument("SampleSpec: invalid caps_string");
        }
        const GstStructure* s = gst_caps_get_structure(caps, 0);
        const char* name = s ? gst_structure_get_name(s) : nullptr;
        if (name) {
          spec.media_type = name;
        }
        gst_caps_unref(caps);
        if (spec.media_type.empty()) {
          throw std::invalid_argument("SampleSpec: encoded caps missing media_type");
        }
      }
      spec.format = "ENCODED";
      spec.dtype = input.dtype;
      spec.layout = input.layout;
      spec.shape = input.shape;
      spec.required_bytes_actual = tensor_dense_bytes_tight(input);
      if (spec.required_bytes_actual == 0) {
        spec.required_bytes_actual = static_cast<size_t>(input.shape[0]);
      }
      spec.caps_string = sample.caps_string;
      spec.caps_key = capkey_from_spec(spec);
      return spec;
    }

    InputOptions opt;
    opt.payload_type = sample_payload_type(sample);
    if (!sample.payload_tag.empty()) {
      opt.format = sample.payload_tag;
    } else if (!sample.format.empty()) {
      opt.format = sample.format;
    }
    return derive_tensor_spec_or_throw(input, opt, "SampleSpec");
  }
  if (sample.kind == SampleKind::Bundle) {
    if (sample.fields.empty()) {
      throw std::invalid_argument("SampleSpec: bundle sample is empty");
    }
    return tensor_envelope_spec_from_sample_or_throw(sample, "SampleSpec");
  }
  throw std::invalid_argument("SampleSpec: tensor payload must be TensorSet");
}

Expected<SampleSpec, Status> derive_sample_spec_or_error(const Sample& sample) {
  try {
    return Expected<SampleSpec, Status>(derive_sample_spec_or_throw(sample));
  } catch (const std::exception& e) {
    Status st;
    st.message = e.what();
    return Expected<SampleSpec, Status>(std::move(st));
  }
}

GstCaps* caps_from_spec(const SampleSpec& spec) {
  static std::unordered_map<CapKey, GstCaps*, CapKeyHash> cache;
  static std::mutex cache_mu;

  CapKey key = capkey_from_spec(spec);
  if (spec.caps_key != key) {
    throw std::runtime_error("caps_from_spec: caps_key mismatch");
  }

  {
    std::lock_guard<std::mutex> lock(cache_mu);
    auto it = cache.find(key);
    if (it != cache.end()) {
      return gst_caps_ref(it->second);
    }
  }

  GstCaps* caps = nullptr;
  if (spec.kind == SampleMediaKind::Encoded) {
    if (spec.caps_string.empty()) {
      throw std::runtime_error("caps_from_spec: encoded caps_string missing");
    }
    caps = gst_caps_from_string(spec.caps_string.c_str());
    if (!caps) {
      throw std::runtime_error("caps_from_spec: encoded caps parse failed");
    }
  } else if (spec.media_type == "video/x-raw") {
    if (spec.format.empty() || spec.width <= 0 || spec.height <= 0) {
      throw std::runtime_error("caps_from_spec: invalid raw video spec");
    }
    caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, spec.format.c_str(), "width",
                               G_TYPE_INT, spec.width, "height", G_TYPE_INT, spec.height, nullptr);
    if (!caps) {
      throw std::runtime_error("caps_from_spec: failed to create video caps");
    }
    if (spec.fps_n > 0) {
      if (spec.fps_d <= 0) {
        gst_caps_unref(caps);
        throw std::runtime_error("caps_from_spec: invalid framerate denominator");
      }
      gst_caps_set_simple(caps, "framerate", GST_TYPE_FRACTION, spec.fps_n, spec.fps_d, nullptr);
    }
  } else if (spec.media_type == "application/vnd.simaai.tensor") {
    const std::vector<int64_t>& tensor_shape = spec.shape;
    if (spec.format.empty() || tensor_shape.empty()) {
      throw std::runtime_error("caps_from_spec: invalid tensor spec");
    }
    const std::string caps_format = caps_format_value_from_spec(spec);
    caps = gst_caps_new_empty_simple("application/vnd.simaai.tensor");
    if (!caps) {
      throw std::runtime_error("caps_from_spec: failed to create tensor caps");
    }
    GstStructure* st = gst_caps_get_structure(caps, 0);
    gst_structure_set(st, "format", G_TYPE_STRING, caps_format.c_str(), nullptr);
    gst_structure_set(st, "rank", G_TYPE_INT, static_cast<int>(tensor_shape.size()), nullptr);
    for (size_t i = 0; i < tensor_shape.size(); ++i) {
      const std::string key = "dim" + std::to_string(i);
      gst_structure_set(st, key.c_str(), G_TYPE_INT, static_cast<int>(tensor_shape[i]), nullptr);
    }
    const std::string shape_csv = tensor_shape_csv_local(tensor_shape);
    if (!shape_csv.empty()) {
      gst_structure_set(st, "shape", G_TYPE_STRING, shape_csv.c_str(), nullptr);
    }
    const std::string dtype = dtype_caps_value_from_spec(spec);
    if (!dtype.empty()) {
      gst_caps_set_simple(caps, "dtype", G_TYPE_STRING, dtype.c_str(), nullptr);
    }
  } else {
    throw std::runtime_error("caps_from_spec: unsupported media_type");
  }

  {
    std::lock_guard<std::mutex> lock(cache_mu);
    auto it = cache.find(key);
    if (it == cache.end()) {
      cache.emplace(key, gst_caps_ref(caps));
    }
  }

  return caps;
}

GstBuffer* allocate_input_buffer(size_t bytes, const InputOptions& opt,
                                 InputBufferPoolGuard& guard) {
#if SIMA_HAS_SIMAAI_POOL
  const std::string media_type_up = upper_copy(resolve_input_media_type(opt));
  const std::string format_up = upper_copy(opt.format.str());
  const bool tensor_media = (media_type_up == "APPLICATION/VND.SIMAAI.TENSOR");
  const bool bf16_tensor = tensor_media && (format_up.find("BF16") != std::string::npos ||
                                            format_up.find("BFLOAT16") != std::string::npos);
  const bool ifm0_hint = upper_copy(opt.buffer_name) == "IFM0";
  GstMemoryFlags target_flag = static_cast<GstMemoryFlags>(GST_SIMAAI_MEMORY_TARGET_EV74);
  const char* target_source = "heuristic";
  bool force_system_memory = false;
  bool use_simaai_pool_effective = opt.use_simaai_pool;

  switch (opt.memory_policy) {
  case InputMemoryPolicy::Ev74:
    target_flag = static_cast<GstMemoryFlags>(GST_SIMAAI_MEMORY_TARGET_EV74);
    target_source = "policy";
    break;
  case InputMemoryPolicy::Dms0:
    target_flag = static_cast<GstMemoryFlags>(GST_SIMAAI_MEMORY_TARGET_DMS0);
    target_source = "policy";
    break;
  case InputMemoryPolicy::SystemMemory:
    force_system_memory = true;
    use_simaai_pool_effective = false;
    target_source = "policy";
    break;
  case InputMemoryPolicy::Auto:
    break;
  }

  if (!force_system_memory && opt.memory_policy == InputMemoryPolicy::Auto) {
    const std::string target_override =
        upper_copy(pipeline_internal::env_str("SIMA_INPUTSTREAM_TENSOR_TARGET", ""));
    if (target_override == "EV74") {
      target_flag = static_cast<GstMemoryFlags>(GST_SIMAAI_MEMORY_TARGET_EV74);
      target_source = "env";
    } else if (target_override == "DMS0") {
      target_flag = static_cast<GstMemoryFlags>(GST_SIMAAI_MEMORY_TARGET_DMS0);
      target_source = "env";
    } else {
      target_flag =
          static_cast<GstMemoryFlags>((bf16_tensor || ifm0_hint) ? GST_SIMAAI_MEMORY_TARGET_DMS0
                                                                 : GST_SIMAAI_MEMORY_TARGET_EV74);
      target_source = "heuristic";
    }
  }

  debug_pool_log_alloc_policy(opt, tensor_media, target_flag, target_source,
                              use_simaai_pool_effective && !force_system_memory);

  if (use_simaai_pool_effective && !force_system_memory) {
    GstBufferPool* pool = guard.pool.get();
    if (!pool) {
      const auto t_create_start = std::chrono::steady_clock::now();
      gst_simaai_segment_memory_init_once();
      GstMemoryFlags flags =
          static_cast<GstMemoryFlags>(target_flag | GST_SIMAAI_MEMORY_FLAG_CACHED);
      const bool tensor_input = tensor_media;
      const std::string segment_name =
          !opt.buffer_name.empty() ? opt.buffer_name
                                   : (tensor_input ? std::string("ifm0") : std::string("input"));
      const gsize segment_size = static_cast<gsize>(bytes);
      const char* segment_name_cstr = segment_name.c_str();
      GstBufferPool* new_pool = gst_simaai_allocate_buffer_pool2(
          /*allocator_user_data=*/nullptr, gst_simaai_memory_get_segment_allocator(),
          opt.pool_min_buffers, opt.pool_max_buffers, flags,
          /*num_segments=*/1, &segment_size, &segment_name_cstr);
      if (new_pool) {
        guard.pool =
            std::unique_ptr<GstBufferPool, void (*)(GstBufferPool*)>(new_pool, free_simaai_pool);
        pool = new_pool;
      }
      debug_pool_state("pool_create", pool, opt, bytes);
      debug_pool_timing("pool_create", opt, bytes, t_create_start, pool != nullptr, true);
    }

    if (pool) {
      debug_pool_state("pool_before_acquire", pool, opt, bytes);
      const auto t_acquire_start = std::chrono::steady_clock::now();
      GstBuffer* buf = nullptr;
      const GstFlowReturn ret = gst_buffer_pool_acquire_buffer(pool, &buf, nullptr);
      const auto t_acquire_end = std::chrono::steady_clock::now();
      const double wait_ms =
          std::chrono::duration<double, std::milli>(t_acquire_end - t_acquire_start).count();
      const bool ok = (ret == GST_FLOW_OK && buf);
      track_input_pool_acquire(pool, buf, bytes, "pool_acquire", wait_ms, ok);
      if (ok) {
        debug_pool_timing("pool_acquire", opt, bytes, t_acquire_start, true, true);
        debug_pool_state("pool_after_acquire_ok", pool, opt, bytes);
        return buf;
      }
      debug_pool_timing("pool_acquire", opt, bytes, t_acquire_start, false, true);
      debug_pool_state("pool_after_acquire_fail", pool, opt, bytes);
      debug_pool_log("Input: simaai pool acquired but buffer allocation failed; "
                     "falling back to system allocator.");
      pool = nullptr;
    }
    debug_pool_log("Input: simaai pool allocation failed; falling back to system allocator.");
  } else if (opt.use_simaai_pool && force_system_memory) {
    debug_pool_log("Input: memory_policy=SystemMemory; bypassing simaai pool.");
  }
#else
  (void)opt;
  (void)guard;
#endif

  const auto t_alloc_start = std::chrono::steady_clock::now();
  GstBuffer* buf = gst_buffer_new_allocate(nullptr, bytes, nullptr);
  debug_pool_timing("system_alloc", opt, bytes, t_alloc_start, buf != nullptr, false);
  return buf;
}

int64_t next_input_frame_id() {
  static std::atomic<int64_t> next_id{0};
  return next_id.fetch_add(1);
}

bool maybe_add_simaai_meta(GstBuffer* buffer, int64_t frame_id, const InputOptions& opt) {
#if SIMA_HAS_SIMAAI_POOL
  if (!buffer || !opt.use_simaai_pool)
    return false;
  dump_sima_meta(buffer, "maybe_add_simaai_meta(before)");
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  if (meta) {
    GstStructure* s = gst_custom_meta_get_structure(meta);
#if defined(GST_STRUCTURE_IS_WRITABLE)
    if (!s || !GST_STRUCTURE_IS_WRITABLE(s)) {
      gst_buffer_remove_meta(buffer, &meta->meta);
      meta = nullptr;
    }
#else
    if (!s)
      meta = nullptr;
#endif
  }
  if (!meta) {
    meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  }
  if (!meta) {
    if (pipeline_internal::env_bool("SIMA_INPUTSTREAM_META_DEBUG", false)) {
      std::fprintf(stderr, "[DBG] GstSimaMeta add failed (buffer=%p)\n",
                   static_cast<void*>(buffer));
    }
    return false;
  }
  GstStructure* s = gst_custom_meta_get_structure(meta);
  if (!s) {
    if (pipeline_internal::env_bool("SIMA_INPUTSTREAM_META_DEBUG", false)) {
      std::fprintf(stderr, "[DBG] GstSimaMeta missing structure (buffer=%p)\n",
                   static_cast<void*>(buffer));
    }
    return false;
  }
  const std::string name = opt.buffer_name;
  gint64 phys_addr = gst_simaai_segment_memory_get_phys_addr(gst_buffer_peek_memory(buffer, 0));
  gst_structure_set(s, "buffer-id", G_TYPE_INT64, phys_addr, "buffer-name", G_TYPE_STRING,
                    name.c_str(), "buffer-offset", G_TYPE_INT64, static_cast<gint64>(0), "frame-id",
                    G_TYPE_INT64, static_cast<gint64>(frame_id), "orig-input-seq", G_TYPE_INT64,
                    static_cast<gint64>(frame_id), "stream-id", G_TYPE_STRING, "0",
                    "origin_stage_id", G_TYPE_STRING, name.c_str(), "origin_output_slot",
                    G_TYPE_INT, 0, "timestamp", G_TYPE_UINT64, static_cast<guint64>(0), nullptr);
  if (pipeline_internal::env_bool("SIMA_INPUTSTREAM_META_DEBUG", false)) {
    std::fprintf(stderr, "[DBG] GstSimaMeta set name=%s phys=%lld frame=%lld\n", name.c_str(),
                 static_cast<long long>(phys_addr), static_cast<long long>(frame_id));
  }
  dump_sima_meta(buffer, "maybe_add_simaai_meta(after)");
  return true;
#else
  (void)buffer;
  (void)frame_id;
  (void)opt;
  return false;
#endif
}

void maybe_update_simaai_meta_name(GstBuffer* buffer, const std::string& name) {
#if SIMA_HAS_SIMAAI_POOL
  if (!buffer || name.empty())
    return;
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  if (!meta)
    return;
  GstStructure* s = gst_custom_meta_get_structure(meta);
  if (!s)
    return;
#if defined(GST_STRUCTURE_IS_WRITABLE)
  if (!GST_STRUCTURE_IS_WRITABLE(s))
    return;
#else
  return;
#endif
  gst_structure_set(s, "buffer-name", G_TYPE_STRING, name.c_str(), "origin_stage_id", G_TYPE_STRING,
                    name.c_str(), nullptr);
#else
  (void)buffer;
  (void)name;
#endif
}

void dump_buffer_memories(GstBuffer* buffer, const char* label) {
  if (!pipeline_internal::env_bool("SIMA_INPUTSTREAM_META_DEBUG", false))
    return;
  const char* tag = label ? label : "buffer";
  if (!buffer) {
    std::fprintf(stderr, "[DBG] %s mem_count=0 (null buffer)\n", tag);
    return;
  }
  const guint n_mems = gst_buffer_n_memory(buffer);
  std::fprintf(stderr, "[DBG] %s mem_count=%u\n", tag, n_mems);
  for (guint i = 0; i < n_mems; ++i) {
    GstMemory* mem = gst_buffer_peek_memory(buffer, i);
    if (!mem) {
      std::fprintf(stderr, "[DBG] %s mem[%u]=null\n", tag, i);
      continue;
    }
    gsize offset = 0;
    gsize maxsize = 0;
    const gsize size = gst_memory_get_sizes(mem, &offset, &maxsize);
    std::fprintf(stderr, "[DBG] %s mem[%u] size=%zu offset=%zu max=%zu\n", tag, i,
                 static_cast<size_t>(size), static_cast<size_t>(offset),
                 static_cast<size_t>(maxsize));
  }
}

void dump_sima_meta(GstBuffer* buffer, const char* label) {
  if (!pipeline_internal::env_bool("SIMA_INPUTSTREAM_META_DEBUG", false))
    return;
#if SIMA_HAS_SIMAAI_POOL
  const char* tag = label ? label : "buffer";
  if (!buffer) {
    std::fprintf(stderr, "[DBG] %s GstSimaMeta=missing (null buffer)\n", tag);
    return;
  }
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  if (!meta) {
    std::fprintf(stderr, "[DBG] %s GstSimaMeta=missing\n", tag);
    return;
  }
  GstStructure* s = gst_custom_meta_get_structure(meta);
  if (!s) {
    std::fprintf(stderr, "[DBG] %s GstSimaMeta=missing-structure\n", tag);
    return;
  }
  bool writable = true;
#if defined(GST_STRUCTURE_IS_WRITABLE)
  writable = GST_STRUCTURE_IS_WRITABLE(s);
#endif
  const char* name = gst_structure_get_string(s, "buffer-name");
  gint64 frame_id = 0;
  gint64 phys_addr = 0;
  gst_structure_get_int64(s, "frame-id", &frame_id);
  gst_structure_get_int64(s, "buffer-id", &phys_addr);
  std::fprintf(stderr, "[DBG] %s GstSimaMeta present=1 writable=%d name=%s frame=%lld phys=%lld\n",
               tag, writable ? 1 : 0, name ? name : "", static_cast<long long>(frame_id),
               static_cast<long long>(phys_addr));
#else
  (void)buffer;
  (void)label;
#endif
}

void dump_sima_meta_full(GstBuffer* buffer, const char* label) {
  if (!pipeline_internal::env_bool("SIMA_INPUTSTREAM_META_DEBUG", false))
    return;
#if SIMA_HAS_SIMAAI_POOL
  const char* tag = label ? label : "buffer";
  if (!buffer) {
    std::fprintf(stderr, "[DBG] %s GstSimaMeta=missing (null buffer)\n", tag);
    return;
  }
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  if (!meta) {
    std::fprintf(stderr, "[DBG] %s GstSimaMeta=missing\n", tag);
    return;
  }
  GstStructure* s = gst_custom_meta_get_structure(meta);
  if (!s) {
    std::fprintf(stderr, "[DBG] %s GstSimaMeta=missing-structure\n", tag);
    return;
  }
  bool writable = true;
#if defined(GST_STRUCTURE_IS_WRITABLE)
  writable = GST_STRUCTURE_IS_WRITABLE(s);
#endif
  const char* name = gst_structure_get_string(s, "buffer-name");
  const char* stream_id = gst_structure_get_string(s, "stream-id");
  gint64 frame_id = 0;
  gint64 buffer_id = 0;
  gint64 buffer_offset = 0;
  gint64 input_seq = 0;
  gint64 orig_input_seq = 0;
  guint64 timestamp = 0;
  gint64 pcie_buffer_id = 0;
  const bool has_frame = gst_structure_get_int64(s, "frame-id", &frame_id);
  const bool has_input_seq = gst_structure_get_int64(s, "input-seq", &input_seq);
  const bool has_orig_input_seq = gst_structure_get_int64(s, "orig-input-seq", &orig_input_seq);
  const bool has_buf = gst_structure_get_int64(s, "buffer-id", &buffer_id);
  const bool has_off = gst_structure_get_int64(s, "buffer-offset", &buffer_offset);
  const bool has_ts = gst_structure_get_uint64(s, "timestamp", &timestamp);
  const bool has_pcie = gst_structure_get_int64(s, "pcie-buffer-id", &pcie_buffer_id);

  gchar* raw = gst_structure_to_string(s);
  std::fprintf(
      stderr,
      "[DBG] %s GstSimaMeta present=1 writable=%d name=%s stream_id=%s frame=%s input_seq=%s "
      "orig_input_seq=%s buffer_id=%s buffer_offset=%s timestamp=%s pcie_buffer_id=%s raw=%s\n",
      tag, writable ? 1 : 0, name ? name : "", stream_id ? stream_id : "",
      has_frame ? std::to_string(static_cast<long long>(frame_id)).c_str() : "missing",
      has_input_seq ? std::to_string(static_cast<long long>(input_seq)).c_str() : "missing",
      has_orig_input_seq ? std::to_string(static_cast<long long>(orig_input_seq)).c_str()
                         : "missing",
      has_buf ? std::to_string(static_cast<long long>(buffer_id)).c_str() : "missing",
      has_off ? std::to_string(static_cast<long long>(buffer_offset)).c_str() : "missing",
      has_ts ? std::to_string(static_cast<unsigned long long>(timestamp)).c_str() : "missing",
      has_pcie ? std::to_string(static_cast<long long>(pcie_buffer_id)).c_str() : "missing",
      raw ? raw : "");
  if (raw)
    g_free(raw);
#else
  (void)buffer;
  (void)label;
#endif
}

void debug_input_buffer_release(GstBuffer* buffer, const char* where) {
  if (!buffer)
    return;
  if (!pipeline_internal::env_bool("SIMA_DEBUG_INPUT_POOL", false) &&
      !pipeline_internal::env_bool("SIMA_INPUTSTREAM_ALLOC_DEBUG", false) &&
      !pipeline_internal::env_bool("SIMA_INPUTSTREAM_META_DEBUG", false)) {
    return;
  }
  GstMemory* mem0 = gst_buffer_peek_memory(buffer, 0);
  const char* alloc_name = "unknown";
  if (mem0 && mem0->allocator) {
    alloc_name = GST_OBJECT_NAME(mem0->allocator);
  }
  std::fprintf(stderr, "[DBG] input_buffer release where=%s buf=%p size=%zu mem0=%p alloc=%s\n",
               where ? where : "unknown", static_cast<void*>(buffer),
               static_cast<size_t>(gst_buffer_get_size(buffer)), static_cast<void*>(mem0),
               alloc_name);
  dump_sima_meta(buffer, "input_buffer_release");
}

bool update_simaai_meta_fields(GstBuffer* buffer, const std::optional<int64_t>& frame_id_override,
                               const std::optional<int64_t>& input_seq_override,
                               const std::optional<int64_t>& orig_input_seq_override,
                               const std::optional<std::string>& stream_id_override,
                               const std::optional<std::string>& buffer_name_override,
                               const std::optional<uint64_t>& timestamp_override,
                               const std::optional<std::string>& origin_stage_id_override,
                               const std::optional<int>& origin_output_slot_override) {
#if SIMA_HAS_SIMAAI_POOL
  if (!buffer)
    return false;
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  if (!meta)
    return false;
  GstStructure* s = gst_custom_meta_get_structure(meta);
  if (!s)
    return false;
  bool structure_mutable = true;
#if defined(GST_STRUCTURE_IS_MUTABLE)
  structure_mutable = GST_STRUCTURE_IS_MUTABLE(s);
#elif defined(GST_STRUCTURE_IS_WRITABLE)
  structure_mutable = GST_STRUCTURE_IS_WRITABLE(s);
#else
  structure_mutable = false;
#endif
  GstStructure* snapshot = nullptr;
  if (!structure_mutable) {
    if (!gst_buffer_is_writable(buffer)) {
      return false;
    }
    snapshot = gst_structure_copy(s);
    gst_buffer_remove_meta(buffer, &meta->meta);
    meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
    s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
    if (!s) {
      if (snapshot)
        gst_structure_free(snapshot);
      return false;
    }
    if (snapshot) {
      const gint n_fields = gst_structure_n_fields(snapshot);
      for (gint i = 0; i < n_fields; ++i) {
        const char* fname = gst_structure_nth_field_name(snapshot, i);
        if (!fname)
          continue;
        const GValue* val = gst_structure_get_value(snapshot, fname);
        if (!val)
          continue;
        gst_structure_set_value(s, fname, val);
      }
      gst_structure_free(snapshot);
    }
  }
  if (frame_id_override.has_value()) {
    gst_structure_set(s, "frame-id", G_TYPE_INT64, static_cast<gint64>(*frame_id_override),
                      nullptr);
  }
  if (input_seq_override.has_value()) {
    gst_structure_set(s, "input-seq", G_TYPE_INT64, static_cast<gint64>(*input_seq_override),
                      nullptr);
  }
  if (orig_input_seq_override.has_value()) {
    gst_structure_set(s, "orig-input-seq", G_TYPE_INT64,
                      static_cast<gint64>(*orig_input_seq_override), nullptr);
  } else if (input_seq_override.has_value()) {
    gst_structure_set(s, "orig-input-seq", G_TYPE_INT64, static_cast<gint64>(*input_seq_override),
                      nullptr);
  }
  if (stream_id_override.has_value()) {
    gst_structure_set(s, "stream-id", G_TYPE_STRING, stream_id_override->c_str(), nullptr);
    gst_structure_set(s, "orig-stream-id", G_TYPE_STRING, stream_id_override->c_str(), nullptr);
  }
  if (buffer_name_override.has_value()) {
    gst_structure_set(s, "buffer-name", G_TYPE_STRING, buffer_name_override->c_str(), nullptr);
  }
  if (timestamp_override.has_value()) {
    gst_structure_set(s, "timestamp", G_TYPE_UINT64, static_cast<guint64>(*timestamp_override),
                      nullptr);
  }
  if (origin_stage_id_override.has_value()) {
    gst_structure_set(s, "origin_stage_id", G_TYPE_STRING, origin_stage_id_override->c_str(),
                      nullptr);
  } else if (buffer_name_override.has_value()) {
    gst_structure_set(s, "origin_stage_id", G_TYPE_STRING, buffer_name_override->c_str(), nullptr);
  }
  if (origin_output_slot_override.has_value()) {
    gst_structure_set(s, "origin_output_slot", G_TYPE_INT,
                      static_cast<gint>(*origin_output_slot_override), nullptr);
  }
  return true;
#else
  (void)buffer;
  (void)frame_id_override;
  (void)input_seq_override;
  (void)orig_input_seq_override;
  (void)stream_id_override;
  (void)buffer_name_override;
  (void)timestamp_override;
  (void)origin_stage_id_override;
  (void)origin_output_slot_override;
  return false;
#endif
}

SampleTimingOverrides sample_timing_overrides_from_sample(const Sample& sample) {
  SampleTimingOverrides out;
  if (sample.frame_id >= 0) {
    out.frame_id = sample.frame_id;
  }
  if (sample.pts_ns >= 0) {
    out.pts_ns = static_cast<uint64_t>(sample.pts_ns);
  }
  if (sample.dts_ns >= 0) {
    out.dts_ns = static_cast<uint64_t>(sample.dts_ns);
  }
  if (sample.duration_ns >= 0) {
    out.duration_ns = static_cast<uint64_t>(sample.duration_ns);
  }
  return out;
}

bool write_sample_timing_to_gst_buffer(GstBuffer* buffer, const SampleTimingOverrides& timing) {
  if (!buffer) {
    return false;
  }

  const bool debug_timing = pipeline_internal::env_bool("SIMA_SAMPLE_TIMING_DEBUG", false);
  const GstClockTime before_pts = GST_BUFFER_PTS(buffer);
  const GstClockTime before_dts = GST_BUFFER_DTS(buffer);
  const GstClockTime before_dur = GST_BUFFER_DURATION(buffer);

  if (timing.pts_ns.has_value()) {
    GST_BUFFER_PTS(buffer) = static_cast<GstClockTime>(*timing.pts_ns);
  }
  if (timing.dts_ns.has_value()) {
    GST_BUFFER_DTS(buffer) = static_cast<GstClockTime>(*timing.dts_ns);
  }
  if (timing.duration_ns.has_value()) {
    GST_BUFFER_DURATION(buffer) = static_cast<GstClockTime>(*timing.duration_ns);
  }

  if (debug_timing) {
    std::fprintf(stderr,
                 "[SAMPLE_TIMING] write buffer=%p pts_valid=%d pts=%llu dts_valid=%d dts=%llu "
                 "dur_valid=%d dur=%llu before_pts=%llu after_pts=%llu before_dts=%llu "
                 "after_dts=%llu before_dur=%llu after_dur=%llu\n",
                 static_cast<void*>(buffer), timing.pts_ns.has_value() ? 1 : 0,
                 static_cast<unsigned long long>(timing.pts_ns.value_or(0)),
                 timing.dts_ns.has_value() ? 1 : 0,
                 static_cast<unsigned long long>(timing.dts_ns.value_or(0)),
                 timing.duration_ns.has_value() ? 1 : 0,
                 static_cast<unsigned long long>(timing.duration_ns.value_or(0)),
                 static_cast<unsigned long long>(before_pts),
                 static_cast<unsigned long long>(GST_BUFFER_PTS(buffer)),
                 static_cast<unsigned long long>(before_dts),
                 static_cast<unsigned long long>(GST_BUFFER_DTS(buffer)),
                 static_cast<unsigned long long>(before_dur),
                 static_cast<unsigned long long>(GST_BUFFER_DURATION(buffer)));
  }

#if SIMA_HAS_SIMAAI_POOL
  GstCustomMeta* custom = nullptr;
  GstStructure* s = nullptr;
  if (!ensure_custom_meta_structure_mutable(buffer, "GstSimaMeta", &custom, &s)) {
    return timing.empty();
  }

  const gboolean pts_valid = timing.pts_ns.has_value() ? TRUE : FALSE;
  const gboolean dts_valid = timing.dts_ns.has_value() ? TRUE : FALSE;
  const gboolean duration_valid = timing.duration_ns.has_value() ? TRUE : FALSE;
  const gboolean frame_id_valid = timing.frame_id.has_value() ? TRUE : FALSE;
  gst_structure_set(s, "sample-frame-id-valid", G_TYPE_BOOLEAN, frame_id_valid, "sample-frame-id",
                    G_TYPE_INT64, static_cast<gint64>(timing.frame_id.value_or(0)),
                    "sample-pts-valid", G_TYPE_BOOLEAN, pts_valid, "sample-pts-ns", G_TYPE_UINT64,
                    static_cast<guint64>(timing.pts_ns.value_or(0)), "sample-dts-valid",
                    G_TYPE_BOOLEAN, dts_valid, "sample-dts-ns", G_TYPE_UINT64,
                    static_cast<guint64>(timing.dts_ns.value_or(0)), "sample-duration-valid",
                    G_TYPE_BOOLEAN, duration_valid, "sample-duration-ns", G_TYPE_UINT64,
                    static_cast<guint64>(timing.duration_ns.value_or(0)), nullptr);

  // Preserve the legacy SimaMeta timestamp field for existing plugins/tools, but make
  // Sample reconstruction key off the explicit sample-pts-valid/sample-pts-ns pair above.
  if (timing.pts_ns.has_value()) {
    gst_structure_set(s, "timestamp", G_TYPE_UINT64, static_cast<guint64>(*timing.pts_ns), nullptr);
  }
  (void)custom;
#else
  if (!timing.empty()) {
    return true;
  }
#endif
  return true;
}

void restore_sample_timing_from_gst_buffer(GstBuffer* buffer, Sample* out) {
  if (!buffer || !out) {
    return;
  }

  const bool debug_timing = pipeline_internal::env_bool("SIMA_SAMPLE_TIMING_DEBUG", false);
  const GstClockTime pts = GST_BUFFER_PTS(buffer);
  const GstClockTime dts = GST_BUFFER_DTS(buffer);
  const GstClockTime dur = GST_BUFFER_DURATION(buffer);
  if (pts != GST_CLOCK_TIME_NONE) {
    out->pts_ns = static_cast<int64_t>(pts);
  }
  if (dts != GST_CLOCK_TIME_NONE) {
    out->dts_ns = static_cast<int64_t>(dts);
  }
  if (dur != GST_CLOCK_TIME_NONE) {
    out->duration_ns = static_cast<int64_t>(dur);
  }

  int64_t restored_pts = out->pts_ns;
  int64_t restored_dts = out->dts_ns;
  int64_t restored_dur = out->duration_ns;

#if SIMA_HAS_SIMAAI_POOL
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s) {
    if (debug_timing) {
      std::fprintf(stderr,
                   "[SAMPLE_TIMING] restore buffer=%p meta=0 gst_pts=%llu sample_pts=%lld "
                   "gst_dts=%llu sample_dts=%lld gst_dur=%llu sample_dur=%lld\n",
                   static_cast<void*>(buffer), static_cast<unsigned long long>(pts),
                   static_cast<long long>(out->pts_ns), static_cast<unsigned long long>(dts),
                   static_cast<long long>(out->dts_ns), static_cast<unsigned long long>(dur),
                   static_cast<long long>(out->duration_ns));
    }
    return;
  }

  gboolean valid = FALSE;
  gint64 frame_value = -1;
  if (gst_structure_get_boolean(s, "sample-frame-id-valid", &valid) == TRUE) {
    if (valid == TRUE && gst_structure_get_int64(s, "sample-frame-id", &frame_value) == TRUE &&
        frame_value >= 0) {
      out->frame_id = static_cast<int64_t>(frame_value);
    } else if (valid == FALSE) {
      out->frame_id = -1;
    }
  }

  valid = FALSE;
  guint64 value = 0;
  if (gst_structure_get_boolean(s, "sample-pts-valid", &valid) == TRUE) {
    if (valid == TRUE && gst_structure_get_uint64(s, "sample-pts-ns", &value) == TRUE &&
        value <= static_cast<guint64>(std::numeric_limits<int64_t>::max())) {
      out->pts_ns = static_cast<int64_t>(value);
    } else if (valid == FALSE) {
      out->pts_ns = -1;
    }
  }
  valid = FALSE;
  value = 0;
  if (gst_structure_get_boolean(s, "sample-dts-valid", &valid) == TRUE) {
    if (valid == TRUE && gst_structure_get_uint64(s, "sample-dts-ns", &value) == TRUE &&
        value <= static_cast<guint64>(std::numeric_limits<int64_t>::max())) {
      out->dts_ns = static_cast<int64_t>(value);
    } else if (valid == FALSE) {
      out->dts_ns = -1;
    }
  }
  valid = FALSE;
  value = 0;
  if (gst_structure_get_boolean(s, "sample-duration-valid", &valid) == TRUE) {
    if (valid == TRUE && gst_structure_get_uint64(s, "sample-duration-ns", &value) == TRUE &&
        value <= static_cast<guint64>(std::numeric_limits<int64_t>::max())) {
      out->duration_ns = static_cast<int64_t>(value);
    } else if (valid == FALSE) {
      out->duration_ns = -1;
    }
  }
#endif
  if (debug_timing) {
    std::fprintf(stderr,
                 "[SAMPLE_TIMING] restore buffer=%p meta=1 gst_pts=%llu pre_meta_pts=%lld "
                 "sample_pts=%lld gst_dts=%llu pre_meta_dts=%lld sample_dts=%lld gst_dur=%llu "
                 "pre_meta_dur=%lld sample_dur=%lld\n",
                 static_cast<void*>(buffer), static_cast<unsigned long long>(pts),
                 static_cast<long long>(restored_pts), static_cast<long long>(out->pts_ns),
                 static_cast<unsigned long long>(dts), static_cast<long long>(restored_dts),
                 static_cast<long long>(out->dts_ns), static_cast<unsigned long long>(dur),
                 static_cast<long long>(restored_dur), static_cast<long long>(out->duration_ns));
  }
}

bool write_simaai_preprocess_meta(GstBuffer* buffer, const PreprocessRuntimeMeta& meta) {
#if SIMA_HAS_SIMAAI_POOL
  if (!buffer)
    return false;
  if (const auto perm_error = validate_axis_perm_vector_local(meta.axis_perm, "preproc_axis_perm");
      perm_error.has_value()) {
    return false;
  }
  GstCustomMeta* custom = nullptr;
  GstStructure* s = nullptr;
  if (!ensure_custom_meta_structure_mutable(buffer, "GstSimaMeta", &custom, &s)) {
    return false;
  }
  gst_structure_set(
      s, "preproc_original_width", G_TYPE_INT, meta.original_width, "preproc_original_height",
      G_TYPE_INT, meta.original_height, "preproc_resized_width", G_TYPE_INT, meta.resized_width,
      "preproc_resized_height", G_TYPE_INT, meta.resized_height, "preproc_scaled_width", G_TYPE_INT,
      meta.scaled_width, "preproc_scaled_height", G_TYPE_INT, meta.scaled_height,
      "preproc_pad_left", G_TYPE_INT, meta.pad_left, "preproc_pad_right", G_TYPE_INT,
      meta.pad_right, "preproc_pad_top", G_TYPE_INT, meta.pad_top, "preproc_pad_bottom", G_TYPE_INT,
      meta.pad_bottom, "preproc_resize_mode", G_TYPE_STRING, meta.resize_mode.c_str(),
      "preproc_color_in", G_TYPE_STRING, meta.color_in.c_str(), "preproc_color_out", G_TYPE_STRING,
      meta.color_out.c_str(), "preproc_normalize", G_TYPE_BOOLEAN, meta.normalize,
      "preproc_quantize", G_TYPE_BOOLEAN, meta.quantize, "preproc_tessellate", G_TYPE_BOOLEAN,
      meta.tessellate, "preproc_affine_m00", G_TYPE_DOUBLE, meta.affine_m00, "preproc_affine_m01",
      G_TYPE_DOUBLE, meta.affine_m01, "preproc_affine_m02", G_TYPE_DOUBLE, meta.affine_m02,
      "preproc_affine_m10", G_TYPE_DOUBLE, meta.affine_m10, "preproc_affine_m11", G_TYPE_DOUBLE,
      meta.affine_m11, "preproc_affine_m12", G_TYPE_DOUBLE, meta.affine_m12,
      "preproc_affine_scale_x", G_TYPE_DOUBLE, meta.affine_scale_x, "preproc_affine_scale_y",
      G_TYPE_DOUBLE, meta.affine_scale_y, "preproc_affine_offset_x", G_TYPE_DOUBLE,
      meta.affine_offset_x, "preproc_affine_offset_y", G_TYPE_DOUBLE, meta.affine_offset_y,
      nullptr);
  if (!gst_structure_set_int_vector_field_local(s, "preproc_axis_perm", meta.axis_perm)) {
    return false;
  }
  return true;
#else
  (void)buffer;
  (void)meta;
  return false;
#endif
}

bool merge_simaai_preprocess_axis_perm(GstBuffer* buffer, const std::vector<int>& axis_perm) {
#if SIMA_HAS_SIMAAI_POOL
  if (!buffer)
    return false;
  if (const auto perm_error = validate_axis_perm_vector_local(axis_perm, "preproc_axis_perm");
      perm_error.has_value()) {
    return false;
  }
  GstCustomMeta* custom = nullptr;
  GstStructure* s = nullptr;
  if (!ensure_custom_meta_structure_mutable(buffer, "GstSimaMeta", &custom, &s)) {
    return false;
  }
  return gst_structure_set_int_vector_field_local(s, "preproc_axis_perm", axis_perm);
#else
  (void)buffer;
  (void)axis_perm;
  return false;
#endif
}

std::optional<PreprocessRuntimeMeta> read_simaai_preprocess_meta(GstBuffer* buffer) {
#if SIMA_HAS_SIMAAI_POOL
  if (!buffer)
    return std::nullopt;
  GstCustomMeta* custom = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* s = custom ? gst_custom_meta_get_structure(custom) : nullptr;
  if (!s)
    return std::nullopt;

  PreprocessRuntimeMeta meta;
  if (!gst_structure_get_int(s, "preproc_original_width", &meta.original_width) ||
      !gst_structure_get_int(s, "preproc_original_height", &meta.original_height)) {
    return std::nullopt;
  }
  if (meta.original_width <= 0 || meta.original_height <= 0) {
    return std::nullopt;
  }

  gst_structure_get_int(s, "preproc_resized_width", &meta.resized_width);
  gst_structure_get_int(s, "preproc_resized_height", &meta.resized_height);
  gst_structure_get_int(s, "preproc_scaled_width", &meta.scaled_width);
  gst_structure_get_int(s, "preproc_scaled_height", &meta.scaled_height);
  gst_structure_get_int(s, "preproc_pad_left", &meta.pad_left);
  gst_structure_get_int(s, "preproc_pad_right", &meta.pad_right);
  gst_structure_get_int(s, "preproc_pad_top", &meta.pad_top);
  gst_structure_get_int(s, "preproc_pad_bottom", &meta.pad_bottom);

  const char* resize_mode = gst_structure_get_string(s, "preproc_resize_mode");
  const char* color_in = gst_structure_get_string(s, "preproc_color_in");
  const char* color_out = gst_structure_get_string(s, "preproc_color_out");
  if (resize_mode)
    meta.resize_mode = resize_mode;
  if (color_in)
    meta.color_in = color_in;
  if (color_out)
    meta.color_out = color_out;
  if (!gst_structure_has_field(s, "preproc_axis_perm")) {
    meta.axis_perm.clear();
  } else {
    if (!gst_structure_get_int_vector_field_local(s, "preproc_axis_perm", &meta.axis_perm)) {
      return std::nullopt;
    }
    if (const auto perm_error =
            validate_axis_perm_vector_local(meta.axis_perm, "preproc_axis_perm");
        perm_error.has_value()) {
      return std::nullopt;
    }
  }

  gboolean normalize = FALSE;
  gboolean quantize = FALSE;
  gboolean tessellate = FALSE;
  if (gst_structure_get_boolean(s, "preproc_normalize", &normalize)) {
    meta.normalize = normalize == TRUE;
  }
  if (gst_structure_get_boolean(s, "preproc_quantize", &quantize)) {
    meta.quantize = quantize == TRUE;
  }
  if (gst_structure_get_boolean(s, "preproc_tessellate", &tessellate)) {
    meta.tessellate = tessellate == TRUE;
  }

  gst_structure_get_double(s, "preproc_affine_m00", &meta.affine_m00);
  gst_structure_get_double(s, "preproc_affine_m01", &meta.affine_m01);
  gst_structure_get_double(s, "preproc_affine_m02", &meta.affine_m02);
  gst_structure_get_double(s, "preproc_affine_m10", &meta.affine_m10);
  gst_structure_get_double(s, "preproc_affine_m11", &meta.affine_m11);
  gst_structure_get_double(s, "preproc_affine_m12", &meta.affine_m12);
  gst_structure_get_double(s, "preproc_affine_scale_x", &meta.affine_scale_x);
  gst_structure_get_double(s, "preproc_affine_scale_y", &meta.affine_scale_y);
  gst_structure_get_double(s, "preproc_affine_offset_x", &meta.affine_offset_x);
  gst_structure_get_double(s, "preproc_affine_offset_y", &meta.affine_offset_y);
  return meta;
#else
  (void)buffer;
  return std::nullopt;
#endif
}

bool has_simaai_preprocess_meta(GstBuffer* buffer) {
  return read_simaai_preprocess_meta(buffer).has_value();
}

bool copy_simaai_preprocess_meta(GstBuffer* dst, GstBuffer* src, std::string* err) {
#if SIMA_HAS_SIMAAI_POOL
  if (!dst || !src) {
    if (err)
      *err = "copy preprocess metadata: missing source/destination buffer";
    return false;
  }
  const auto meta = read_simaai_preprocess_meta(src);
  if (!meta.has_value()) {
    if (err)
      *err = "copy preprocess metadata: source buffer has no valid preprocess metadata";
    return false;
  }
  if (!write_simaai_preprocess_meta(dst, *meta)) {
    if (err)
      *err = "copy preprocess metadata: failed to write GstSimaMeta fields";
    return false;
  }
  return true;
#else
  (void)dst;
  (void)src;
  if (err)
    *err = "copy preprocess metadata: GstSimaMeta is unavailable in this build";
  return false;
#endif
}

std::optional<std::string>
validate_simaai_preprocess_meta_required_fields(GstBuffer* buffer,
                                                const std::vector<std::string>& required_fields,
                                                PreprocessRuntimeMeta* out_meta) {
#if SIMA_HAS_SIMAAI_POOL
  if (!buffer) {
    return std::string("missing GstBuffer while reading preprocess metadata");
  }
  GstCustomMeta* custom = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* s = custom ? gst_custom_meta_get_structure(custom) : nullptr;
  if (!s) {
    return std::string("missing GstSimaMeta on input buffer");
  }

  auto require_int = [&](const char* field, bool positive) -> std::optional<std::string> {
    int value = 0;
    if (!gst_structure_get_int(s, field, &value)) {
      return std::string("missing or invalid int preprocess metadata field '") + field + "'";
    }
    if (positive && value <= 0) {
      return std::string("invalid preprocess metadata field '") + field + "' (must be > 0)";
    }
    return std::nullopt;
  };
  auto require_bool = [&](const char* field) -> std::optional<std::string> {
    gboolean value = FALSE;
    if (!gst_structure_get_boolean(s, field, &value)) {
      return std::string("missing or invalid bool preprocess metadata field '") + field + "'";
    }
    return std::nullopt;
  };
  auto require_double = [&](const char* field) -> std::optional<std::string> {
    double value = 0.0;
    if (!gst_structure_get_double(s, field, &value)) {
      return std::string("missing or invalid double preprocess metadata field '") + field + "'";
    }
    return std::nullopt;
  };
  auto require_string = [&](const char* field) -> std::optional<std::string> {
    const char* value = gst_structure_get_string(s, field);
    if (!value) {
      return std::string("missing or invalid string preprocess metadata field '") + field + "'";
    }
    return std::nullopt;
  };
  auto require_int_list = [&](const char* field) -> std::optional<std::string> {
    std::vector<int> values;
    if (!gst_structure_get_int_vector_field_local(s, field, &values)) {
      return std::string("missing or invalid int-list preprocess metadata field '") + field + "'";
    }
    return validate_axis_perm_vector_local(values, field);
  };

  for (const auto& field : required_fields) {
    if (field.empty())
      continue;
    if (!gst_structure_has_field(s, field.c_str())) {
      return std::string("missing required preprocess metadata field '") + field + "'";
    }

    std::optional<std::string> err;
    if (field == "preproc_original_width" || field == "preproc_original_height" ||
        field == "preproc_resized_width" || field == "preproc_resized_height" ||
        field == "preproc_scaled_width" || field == "preproc_scaled_height") {
      err = require_int(field.c_str(), /*positive=*/true);
    } else if (field == "preproc_pad_left" || field == "preproc_pad_right" ||
               field == "preproc_pad_top" || field == "preproc_pad_bottom") {
      err = require_int(field.c_str(), /*positive=*/false);
    } else if (field == "preproc_resize_mode" || field == "preproc_color_in" ||
               field == "preproc_color_out") {
      err = require_string(field.c_str());
    } else if (field == "preproc_axis_perm") {
      err = require_int_list(field.c_str());
    } else if (field == "preproc_normalize" || field == "preproc_quantize" ||
               field == "preproc_tessellate") {
      err = require_bool(field.c_str());
    } else if (field == "preproc_affine_m00" || field == "preproc_affine_m01" ||
               field == "preproc_affine_m02" || field == "preproc_affine_m10" ||
               field == "preproc_affine_m11" || field == "preproc_affine_m12" ||
               field == "preproc_affine_scale_x" || field == "preproc_affine_scale_y" ||
               field == "preproc_affine_offset_x" || field == "preproc_affine_offset_y") {
      err = require_double(field.c_str());
    }

    if (err.has_value()) {
      return err;
    }
  }

  const auto parsed = read_simaai_preprocess_meta(buffer);
  if (!parsed.has_value()) {
    return std::string("preprocess metadata exists but failed semantic validation");
  }
  if (out_meta) {
    *out_meta = *parsed;
  }
  return std::nullopt;
#else
  (void)buffer;
  (void)required_fields;
  (void)out_meta;
  return std::string("GstSimaMeta is unavailable in this build");
#endif
}

bool apply_simaai_preprocess_meta_template(GstBuffer* buffer, const InputOptions& opt,
                                           int input_width, int input_height) {
  if (!buffer || !opt.preprocess_meta.has_value() || !opt.preprocess_meta->enabled)
    return false;
  if (input_width <= 0 || input_height <= 0)
    return false;

  const PreprocessMetaTemplate& tmpl = *opt.preprocess_meta;
  PreprocessRuntimeMeta meta;
  meta.original_width = input_width;
  meta.original_height = input_height;
  meta.color_in = tmpl.color_in;
  meta.color_out = tmpl.color_out;
  meta.axis_perm = tmpl.axis_perm;
  meta.normalize = tmpl.normalize;
  meta.quantize = tmpl.quantize;
  meta.tessellate = tmpl.tessellate;

  const int target_w = (tmpl.target_width > 0) ? tmpl.target_width : input_width;
  const int target_h = (tmpl.target_height > 0) ? tmpl.target_height : input_height;
  meta.resized_width = target_w;
  meta.resized_height = target_h;
  meta.scaled_width = (tmpl.scaled_width > 0) ? tmpl.scaled_width : target_w;
  meta.scaled_height = (tmpl.scaled_height > 0) ? tmpl.scaled_height : target_h;
  meta.resize_mode = tmpl.resize_mode.empty() ? "none" : tmpl.resize_mode;

  const std::string mode = upper_copy(meta.resize_mode);
  if (mode == "LETTERBOX") {
    const double sx = static_cast<double>(target_w) / static_cast<double>(input_width);
    const double sy = static_cast<double>(target_h) / static_cast<double>(input_height);
    const double scale = std::min(sx, sy);
    meta.scaled_width = std::max(1, static_cast<int>(std::llround(input_width * scale)));
    meta.scaled_height = std::max(1, static_cast<int>(std::llround(input_height * scale)));
    meta.pad_left = std::max(0, (target_w - meta.scaled_width) / 2);
    meta.pad_top = std::max(0, (target_h - meta.scaled_height) / 2);
    meta.pad_right = std::max(0, target_w - meta.scaled_width - meta.pad_left);
    meta.pad_bottom = std::max(0, target_h - meta.scaled_height - meta.pad_top);
    const double inv = (scale > 0.0) ? (1.0 / scale) : 1.0;
    meta.affine_m00 = inv;
    meta.affine_m01 = 0.0;
    meta.affine_m02 = -static_cast<double>(meta.pad_left) * inv;
    meta.affine_m10 = 0.0;
    meta.affine_m11 = inv;
    meta.affine_m12 = -static_cast<double>(meta.pad_top) * inv;
    meta.affine_scale_x = inv;
    meta.affine_scale_y = inv;
    meta.affine_offset_x = meta.affine_m02;
    meta.affine_offset_y = meta.affine_m12;
  } else if (mode == "STRETCH") {
    const double sx = static_cast<double>(target_w) / static_cast<double>(input_width);
    const double sy = static_cast<double>(target_h) / static_cast<double>(input_height);
    const double inv_x = (sx > 0.0) ? (1.0 / sx) : 1.0;
    const double inv_y = (sy > 0.0) ? (1.0 / sy) : 1.0;
    meta.pad_left = meta.pad_right = meta.pad_top = meta.pad_bottom = 0;
    meta.affine_m00 = inv_x;
    meta.affine_m01 = 0.0;
    meta.affine_m02 = 0.0;
    meta.affine_m10 = 0.0;
    meta.affine_m11 = inv_y;
    meta.affine_m12 = 0.0;
    meta.affine_scale_x = inv_x;
    meta.affine_scale_y = inv_y;
    meta.affine_offset_x = 0.0;
    meta.affine_offset_y = 0.0;
  } else if (mode == "CROP") {
    const double sx = static_cast<double>(target_w) / static_cast<double>(input_width);
    const double sy = static_cast<double>(target_h) / static_cast<double>(input_height);
    const double scale = std::max(sx, sy);
    const int scaled_w = std::max(1, static_cast<int>(std::llround(input_width * scale)));
    const int scaled_h = std::max(1, static_cast<int>(std::llround(input_height * scale)));
    const int crop_left = std::max(0, (scaled_w - target_w) / 2);
    const int crop_top = std::max(0, (scaled_h - target_h) / 2);
    const double inv = (scale > 0.0) ? (1.0 / scale) : 1.0;
    meta.scaled_width = scaled_w;
    meta.scaled_height = scaled_h;
    meta.pad_left = meta.pad_right = meta.pad_top = meta.pad_bottom = 0;
    meta.affine_m00 = inv;
    meta.affine_m01 = 0.0;
    meta.affine_m02 = static_cast<double>(crop_left) * inv;
    meta.affine_m10 = 0.0;
    meta.affine_m11 = inv;
    meta.affine_m12 = static_cast<double>(crop_top) * inv;
    meta.affine_scale_x = inv;
    meta.affine_scale_y = inv;
    meta.affine_offset_x = meta.affine_m02;
    meta.affine_offset_y = meta.affine_m12;
  } else {
    meta.pad_left = meta.pad_right = meta.pad_top = meta.pad_bottom = 0;
    meta.affine_m00 = 1.0;
    meta.affine_m01 = 0.0;
    meta.affine_m02 = 0.0;
    meta.affine_m10 = 0.0;
    meta.affine_m11 = 1.0;
    meta.affine_m12 = 0.0;
    meta.affine_scale_x = 1.0;
    meta.affine_scale_y = 1.0;
    meta.affine_offset_x = 0.0;
    meta.affine_offset_y = 0.0;
  }

  return write_simaai_preprocess_meta(buffer, meta);
}

GstBuffer* attach_simaai_meta_inplace(GstBuffer* buffer, const InputOptions& opt,
                                      InputBufferPoolGuard& guard, const char* label,
                                      const std::optional<int64_t>& frame_id_override,
                                      const StreamIdOverride& stream_id_override,
                                      const BufferNameOverride& buffer_name_override) {
#if SIMA_HAS_SIMAAI_POOL
  if (!buffer)
    return nullptr;
  buffer = gst_buffer_make_writable(buffer);
  if (!buffer)
    return nullptr;
  const std::string name = buffer_name_override.value.value_or(opt.buffer_name);
  dump_sima_meta(buffer, label);

  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  GstStructure* old_s = s ? gst_structure_copy(s) : nullptr;
  if (meta) {
    // Recycled/pool buffers can carry a previous custom meta whose structure is
    // immutable.  Replacing the meta avoids gst_structure_set mutability
    // assertions while preserving any existing preprocess fields below.
    gst_buffer_remove_meta(buffer, &meta->meta);
    meta = nullptr;
    s = nullptr;
  }
  meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (old_s && s) {
    gst_structure_foreach(
        old_s,
        +[](GQuark field_id, const GValue* value, gpointer user_data) -> gboolean {
          auto* dst = static_cast<GstStructure*>(user_data);
          gst_structure_set_value(dst, g_quark_to_string(field_id), value);
          return TRUE;
        },
        s);
  }
  if (old_s)
    gst_structure_free(old_s);
  if (!s) {
    std::fprintf(stderr, "[DBG] %s GstSimaMeta add failed\n", label ? label : "buffer");
    return buffer;
  }
  gint64 phys_addr = 0;
  if (gst_buffer_n_memory(buffer) > 0) {
    phys_addr = gst_simaai_segment_memory_get_phys_addr(gst_buffer_peek_memory(buffer, 0));
  }
  const gint64 frame_id = frame_id_override.has_value()
                              ? static_cast<gint64>(*frame_id_override)
                              : static_cast<gint64>(next_input_frame_id());
  const std::string stream_id = stream_id_override.value.value_or("0");
  gst_structure_set(s, "buffer-id", G_TYPE_INT64, phys_addr, "buffer-name", G_TYPE_STRING,
                    name.c_str(), "buffer-offset", G_TYPE_INT64, static_cast<gint64>(0), "frame-id",
                    G_TYPE_INT64, frame_id, "orig-input-seq", G_TYPE_INT64, frame_id, "stream-id",
                    G_TYPE_STRING, stream_id.c_str(), "timestamp", G_TYPE_UINT64,
                    static_cast<guint64>(0), nullptr);
  dump_sima_meta(buffer, label);
  return buffer;
#else
  (void)buffer;
  (void)opt;
  (void)guard;
  (void)label;
  return buffer;
#endif
}

} // namespace simaai::neat

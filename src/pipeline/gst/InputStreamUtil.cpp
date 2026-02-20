#include "InputStreamUtil.h"

#include "pipeline/SessionOptions.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/InputPolicy.h"
#include "pipeline/TensorOpenCV.h"
#include "pipeline/TessellatedTensor.h"
#include "nodes/io/Input.h"
#include "pipeline/internal/SimaaiGstCompat.h"

#include <gst/gst.h>

#include <opencv2/core/mat.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace simaai::neat {
using pipeline_internal::upper_copy;
namespace {

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

void validate_dim_with_effective_max(
    const std::string& tag, const char* field, int value,
    const InputStreamOptions::ResolvedShapeLimits& limits, char axis, const char* fix_hint) {
  const int limit = limit_for_axis(limits, axis);
  if (value <= 0 || limit <= 0 || value <= limit)
    return;

  std::ostringstream oss;
  oss << tag << ": " << field << " exceeds effective max (" << value << " > " << limit
      << ")";
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
    if (spec.format.empty() || spec.width <= 0 || spec.height <= 0 || spec.depth <= 0) {
      throw std::runtime_error("SampleSpec: missing tensor caps fields");
    }
    std::ostringstream oss;
    oss << "application/vnd.simaai.tensor,format=" << spec.format << ",width=" << spec.width
        << ",height=" << spec.height << ",depth=" << spec.depth;
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
    seed = hash_combine(seed, std::hash<int>{}(static_cast<int>(key.layout)));
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
  if (input.device.type != simaai::neat::DeviceType::CPU) {
    throw std::invalid_argument(tag + ": simaai::neat::Tensor must be on CPU");
  }
  if (input.semantic.encoded.has_value()) {
    throw std::invalid_argument(tag + ": encoded tensors require Sample caps_string");
  }

  SampleSpec spec;
  spec.dtype = input.dtype;
  spec.layout = input.layout;
  spec.shape = input.shape;

  std::string media = opt.media_type;
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
    if (input.layout == TensorLayout::Unknown || input.layout == TensorLayout::Planar) {
      throw std::invalid_argument(tag + ": tensor layout must be explicit (HWC/CHW/HW)");
    }

    std::vector<int64_t> normalized_shape = input.shape;
    const size_t expected_rank = (input.layout == TensorLayout::HW) ? 2u : 3u;
    if (normalized_shape.size() == expected_rank + 1) {
      const int64_t batch = normalized_shape.front();
      if (batch <= 0) {
        throw std::invalid_argument(tag + ": invalid leading batch dimension");
      }
      if (batch == 1) {
        normalized_shape.erase(normalized_shape.begin());
      } else {
        throw std::invalid_argument(
            tag + ": batched tensor input is not supported in this pipeline (leading batch=" +
            std::to_string(batch) + "). Fix: use batch=1 or remove the batch axis.");
      }
    }

    int h = -1;
    int w = -1;
    int d = -1;
    if (input.layout == TensorLayout::HWC) {
      if (normalized_shape.size() != 3) {
        throw std::invalid_argument(tag + ": HWC tensor must have shape [H,W,C]");
      }
      h = shape_dim(normalized_shape, 0);
      w = shape_dim(normalized_shape, 1);
      d = shape_dim(normalized_shape, 2);
    } else if (input.layout == TensorLayout::CHW) {
      if (normalized_shape.size() != 3) {
        throw std::invalid_argument(tag + ": CHW tensor must have shape [C,H,W]");
      }
      d = shape_dim(normalized_shape, 0);
      h = shape_dim(normalized_shape, 1);
      w = shape_dim(normalized_shape, 2);
    } else if (input.layout == TensorLayout::HW) {
      if (normalized_shape.size() != 2) {
        throw std::invalid_argument(tag + ": HW tensor must have shape [H,W]");
      }
      h = shape_dim(normalized_shape, 0);
      w = shape_dim(normalized_shape, 1);
      d = 1;
    }

    if (w <= 0 || h <= 0 || d <= 0) {
      throw std::invalid_argument(tag + ": tensor input missing width/height/depth");
    }
    SampleSpec shape_seed = spec;
    shape_seed.width = w;
    shape_seed.height = h;
    shape_seed.depth = d;
    const auto limits = pipeline_internal::resolve_shape_limits(
        pipeline_internal::normalize_shape_bounds(opt), shape_seed);
    validate_dim_with_effective_max(
        tag, "tensor width", w, limits, 'w',
        "resize input or increase max_width/width (Model::Options::input_max_width).");
    validate_dim_with_effective_max(
        tag, "tensor height", h, limits, 'h',
        "resize input or increase max_height/height (Model::Options::input_max_height).");
    validate_dim_with_effective_max(
        tag, "tensor depth", d, limits, 'd',
        "reduce channels or increase max_depth/depth (Model::Options::input_max_depth).");

    std::string fmt = upper_copy(opt.format);
    const std::string dtype_fmt = upper_copy(fmt_from_dtype(input.dtype));
    if (fmt.empty())
      fmt = dtype_fmt;
    if (fmt.empty()) {
      throw std::invalid_argument(tag + ": tensor input missing format");
    }
    if (!opt.format.empty() && !dtype_fmt.empty()) {
      const bool fmt_is_dtype = fmt == "UINT8" || fmt == "INT8" || fmt == "UINT16" ||
                                fmt == "INT16" || fmt == "INT32" || fmt == "BF16" ||
                                fmt == "FP32" || fmt == "FP64";
      const bool fmt_is_tess_i8 = is_tessellated_int8_format(fmt) || fmt == "MLA";
      const bool fmt_is_tess_bf16 = is_tessellated_bf16_format(fmt);
      if (fmt_is_dtype && fmt != dtype_fmt) {
        throw std::invalid_argument(tag + ": tensor format does not match dtype");
      }
      if (fmt_is_tess_i8 && input.dtype != TensorDType::Int8 && input.dtype != TensorDType::UInt8) {
        throw std::invalid_argument(tag + ": tensor format does not match dtype");
      }
      if (fmt_is_tess_bf16 && input.dtype != TensorDType::BFloat16) {
        throw std::invalid_argument(tag + ": tensor format does not match dtype");
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
    spec.width = w;
    spec.height = h;
    spec.depth = d;
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

  std::string media = opt.media_type;
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
    return simaai::neat::from_cv_mat(mat, pf, true);
  }

  if (media == "application/vnd.simaai.tensor") {
    if (mat.depth() != CV_32F) {
      throw std::invalid_argument(tag + ": tensor input must be CV_32F");
    }
    if (mat.channels() != 1 && mat.channels() != 3) {
      throw std::invalid_argument(tag + ": tensor input must be 1 or 3 channels");
    }
    std::string fmt = upper_copy(opt.format);
    if (!fmt.empty() && fmt != "FP32") {
      throw std::invalid_argument(tag + ": tensor format must be FP32");
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
    return out;
  }

  throw std::invalid_argument(tag + ": unsupported media_type: " + media);
}

SampleSpec derive_sample_spec_or_throw(const Sample& sample) {
  if (sample.kind == SampleKind::Bundle) {
    if (sample.fields.empty()) {
      throw std::invalid_argument("SampleSpec: bundle samples are empty");
    }
    return derive_sample_spec_or_throw(sample.fields.front());
  }
  if (!sample.tensor.has_value()) {
    throw std::invalid_argument("SampleSpec: missing simaai::neat::Tensor");
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
  opt.media_type = sample.media_type;
  if (!sample.payload_tag.empty()) {
    opt.format = sample.payload_tag;
  } else if (!sample.format.empty()) {
    opt.format = sample.format;
  }
  SampleSpec spec = derive_tensor_spec_or_throw(input, opt, "SampleSpec");
  return spec;
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
    if (spec.format.empty() || spec.width <= 0 || spec.height <= 0 || spec.depth <= 0) {
      throw std::runtime_error("caps_from_spec: invalid tensor spec");
    }
    caps = gst_caps_new_simple("application/vnd.simaai.tensor", "format", G_TYPE_STRING,
                               spec.format.c_str(), "width", G_TYPE_INT, spec.width, "height",
                               G_TYPE_INT, spec.height, "depth", G_TYPE_INT, spec.depth, nullptr);
    if (!caps) {
      throw std::runtime_error("caps_from_spec: failed to create tensor caps");
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
  if (opt.use_simaai_pool) {
    GstBufferPool* pool = guard.pool.get();
    if (!pool) {
      const auto t_create_start = std::chrono::steady_clock::now();
      gst_simaai_segment_memory_init_once();
      GstMemoryFlags flags = static_cast<GstMemoryFlags>(GST_SIMAAI_MEMORY_TARGET_EV74 |
                                                         GST_SIMAAI_MEMORY_FLAG_CACHED);
      GstBufferPool* new_pool = gst_simaai_allocate_buffer_pool(
          /*allocator_user_data=*/nullptr, gst_simaai_memory_get_segment_allocator(), bytes,
          opt.pool_min_buffers, opt.pool_max_buffers, flags);
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
                    "origin_stage_id", G_TYPE_STRING, name.c_str(), "origin_output_slot", G_TYPE_INT,
                    0, "timestamp", G_TYPE_UINT64, static_cast<guint64>(0), nullptr);
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
  gst_structure_set(s, "buffer-name", G_TYPE_STRING, name.c_str(), "origin_stage_id",
                    G_TYPE_STRING, name.c_str(), nullptr);
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
  bool writable = true;
#if defined(GST_STRUCTURE_IS_WRITABLE)
  writable = GST_STRUCTURE_IS_WRITABLE(s);
#else
  writable = false;
#endif
  GstStructure* snapshot = nullptr;
  if (!writable) {
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

GstBuffer* attach_simaai_meta_inplace(GstBuffer* buffer, const InputOptions& opt,
                                      InputBufferPoolGuard& guard, const char* label,
                                      const std::optional<int64_t>& frame_id_override,
                                      const StreamIdOverride& stream_id_override,
                                      const BufferNameOverride& buffer_name_override) {
#if SIMA_HAS_SIMAAI_POOL
  if (!buffer)
    return nullptr;
  const std::string name = buffer_name_override.value.value_or(opt.buffer_name);
  dump_sima_meta(buffer, label);

  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  bool writable = true;
#if defined(GST_STRUCTURE_IS_WRITABLE)
  writable = s && GST_STRUCTURE_IS_WRITABLE(s);
#else
  writable = s != nullptr;
#endif
  if (meta && s && writable) {
    gst_structure_set(s, "buffer-name", G_TYPE_STRING, name.c_str(), nullptr);
    if (frame_id_override.has_value()) {
      gst_structure_set(s, "frame-id", G_TYPE_INT64, static_cast<gint64>(*frame_id_override),
                        nullptr);
      gst_structure_set(s, "orig-input-seq", G_TYPE_INT64, static_cast<gint64>(*frame_id_override),
                        nullptr);
    }
    if (stream_id_override.value.has_value()) {
      gst_structure_set(s, "stream-id", G_TYPE_STRING, stream_id_override.value->c_str(), nullptr);
      gst_structure_set(s, "orig-stream-id", G_TYPE_STRING, stream_id_override.value->c_str(),
                        nullptr);
    }
    dump_sima_meta(buffer, label);
    return buffer;
  }
  if (meta && s && !writable) {
    gst_buffer_remove_meta(buffer, &meta->meta);
    meta = nullptr;
    s = nullptr;
  }
  if (!meta) {
    meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
    s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  }
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

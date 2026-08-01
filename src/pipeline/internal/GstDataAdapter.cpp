// src/pipeline/internal/GstDataAdapter.cpp
#include "pipeline/internal/GstDataAdapter.h"

#include "InputStreamUtil.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/TensorUtil.h"
#include "nodes/io/Input.h"

#include <gst/gst.h>
#include <gst/video/video.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal {

bool attach_video_meta(GstBuffer** buffer, const SampleSpec& spec, std::string* err);
bool apply_tensor_size(GstBuffer** buffer, const SampleSpec& spec, std::string* err);
const char* storage_kind_name(simaai::neat::StorageKind kind);

namespace {

bool data_adapter_debug_enabled() {
  return env_bool("SIMA_GST_DATA_ADAPTER_DEBUG", false);
}

bool zero_copy_writable_view_enabled() {
  return env_bool("SIMA_GST_ZERO_COPY_WRITABLE_VIEW", true);
}

std::size_t dtype_bytes_for_copy(TensorDType dtype) {
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

const char* media_kind_name(SampleMediaKind kind) {
  switch (kind) {
  case SampleMediaKind::RawVideo:
    return "RawVideo";
  case SampleMediaKind::Tensor:
    return "Tensor";
  case SampleMediaKind::Encoded:
    return "Encoded";
  }
  return "Unknown";
}

void log_zero_copy_skip(const char* tag, const char* detail, const simaai::neat::Tensor& tensor,
                        const SampleSpec& spec) {
  if (!data_adapter_debug_enabled())
    return;
  const char* r = tag ? tag : "unspecified";
  const char* d = detail ? detail : "";
  const char* storage = "none";
  if (tensor.storage)
    storage = storage_kind_name(tensor.storage->kind);
  std::fprintf(stderr, "[GstDataAdapter] zero-copy skip reason=%s detail=%s storage=%s kind=%s\n",
               r, d, storage, media_kind_name(spec.kind));
}

bool finalize_buffer(GstBuffer** buffer, const SampleSpec& spec, const GstBufferBuildPolicy& policy,
                     std::string* err) {
  if (!buffer || !*buffer) {
    if (err)
      *err = "finalize: missing GstBuffer";
    return false;
  }
  if (spec.kind == SampleMediaKind::RawVideo) {
    if (!policy.require_video_meta)
      return true;
    if (!attach_video_meta(buffer, spec, err)) {
      return false;
    }
    return true;
  }
  if (!apply_tensor_size(buffer, spec, err)) {
    return false;
  }
  return true;
}

} // namespace

bool wrap_cpu_dense_zero_copy(const simaai::neat::Tensor& tensor, GstBuffer** out,
                              std::string* err) {
  if (!out)
    return false;
  *out = nullptr;
  if (!tensor.storage || !tensor.storage->holder) {
    if (err)
      *err = "zero-copy: missing tensor holder";
    return false;
  }
  if (tensor.storage->kind != simaai::neat::StorageKind::CpuOwned &&
      tensor.storage->kind != simaai::neat::StorageKind::CpuExternal) {
    if (err)
      *err = "zero-copy: storage is not CPU-backed";
    return false;
  }
  if (!tensor.is_dense() || !tensor.is_contiguous()) {
    if (err)
      *err = "zero-copy: tensor is not dense/contiguous";
    return false;
  }
  if (!tensor.storage->data || tensor.storage->size_bytes == 0) {
    if (err)
      *err = "zero-copy: missing tensor data";
    return false;
  }
  if (tensor.byte_offset < 0) {
    if (err)
      *err = "zero-copy: negative tensor byte_offset";
    return false;
  }

  const std::size_t bytes = tensor.dense_bytes_tight();
  if (bytes == 0) {
    if (err)
      *err = "zero-copy: tensor dense bytes unknown";
    return false;
  }
  const std::size_t end = static_cast<std::size_t>(tensor.byte_offset) + bytes;
  if (end > tensor.storage->size_bytes) {
    if (err)
      *err = "zero-copy: tensor exceeds storage size";
    return false;
  }

  GstMemoryFlags flags =
      tensor.read_only ? GST_MEMORY_FLAG_READONLY : static_cast<GstMemoryFlags>(0);
  auto* keepalive = new std::shared_ptr<void>(tensor.storage->holder);
  GstBuffer* buf = gst_buffer_new_wrapped_full(
      flags, tensor.storage->data, static_cast<gsize>(tensor.storage->size_bytes),
      static_cast<gsize>(tensor.byte_offset), static_cast<gsize>(bytes), keepalive,
      [](gpointer p) { delete static_cast<std::shared_ptr<void>*>(p); });
  if (!buf) {
    delete keepalive;
    if (err)
      *err = "zero-copy: gst_buffer_new_wrapped_full failed";
    return false;
  }
  *out = buf;
  return true;
}

bool wrap_cpu_video_zero_copy(const simaai::neat::Tensor& tensor, const SampleSpec& spec,
                              GstBuffer** out, std::string* err) {
  if (!out)
    return false;
  *out = nullptr;
  if (spec.kind != SampleMediaKind::RawVideo) {
    if (err)
      *err = "zero-copy: not raw video";
    return false;
  }
  if (!tensor.storage || !tensor.storage->holder) {
    if (err)
      *err = "zero-copy: missing tensor holder";
    return false;
  }
  if (tensor.storage->kind != simaai::neat::StorageKind::CpuOwned &&
      tensor.storage->kind != simaai::neat::StorageKind::CpuExternal) {
    if (err)
      *err = "zero-copy: storage is not CPU-backed";
    return false;
  }
  if (!tensor.storage->data || tensor.storage->size_bytes == 0) {
    if (err)
      *err = "zero-copy: missing tensor data";
    return false;
  }
  if (tensor.byte_offset != 0) {
    if (err)
      *err = "zero-copy: raw video requires byte_offset == 0";
    return false;
  }
  if (spec.required_bytes_actual == 0) {
    if (err)
      *err = "zero-copy: required bytes unknown";
    return false;
  }
  if (spec.required_bytes_actual > tensor.storage->size_bytes) {
    if (err)
      *err = "zero-copy: required bytes exceed storage size";
    return false;
  }

  GstMemoryFlags flags =
      tensor.read_only ? GST_MEMORY_FLAG_READONLY : static_cast<GstMemoryFlags>(0);
  auto* keepalive = new std::shared_ptr<void>(tensor.storage->holder);
  GstBuffer* buf = gst_buffer_new_wrapped_full(
      flags, tensor.storage->data, static_cast<gsize>(tensor.storage->size_bytes), 0,
      static_cast<gsize>(spec.required_bytes_actual), keepalive,
      [](gpointer p) { delete static_cast<std::shared_ptr<void>*>(p); });
  if (!buf) {
    delete keepalive;
    if (err)
      *err = "zero-copy: gst_buffer_new_wrapped_full failed";
    return false;
  }
  *out = buf;
  return true;
}

std::size_t tensor_plane_bytes_tight(const simaai::neat::Plane& plane, TensorDType dtype) {
  if (plane.shape.size() < 2)
    return 0;
  const int64_t h = plane.shape[0];
  const int64_t w = plane.shape[1];
  if (h <= 0 || w <= 0)
    return 0;
  const std::size_t elem = dtype_bytes_for_copy(dtype);
  const int64_t min_stride = static_cast<int64_t>(w * elem);
  const int64_t stride = !plane.strides_bytes.empty() ? plane.strides_bytes[0] : min_stride;
  if (stride < min_stride)
    return 0;
  if (plane.strides_bytes.size() > 1 && plane.strides_bytes[1] != static_cast<int64_t>(elem)) {
    return 0;
  }
  return static_cast<std::size_t>(stride) * static_cast<std::size_t>(h);
}

std::size_t tensor_bytes_tight(const simaai::neat::Tensor& input) {
  if (input.is_composite()) {
    std::size_t total = 0;
    for (const auto& plane : input.planes) {
      total += tensor_plane_bytes_tight(plane, input.dtype);
    }
    return total;
  }
  if (!input.is_dense() || input.shape.empty())
    return 0;
  std::size_t total = dtype_bytes_for_copy(input.dtype);
  for (const auto dim : input.shape) {
    if (dim <= 0)
      return 0;
    total *= static_cast<std::size_t>(dim);
  }
  return total;
}

bool copy_tensor_payload_to(const simaai::neat::Tensor& tensor, uint8_t* dst, std::size_t dst_bytes,
                            std::string* err) {
  if (!dst && dst_bytes > 0) {
    if (err)
      *err = "tensor copy: missing destination";
    return false;
  }
  if (dst_bytes == 0)
    return true;
  if (!tensor.storage) {
    if (err)
      *err = "tensor copy: missing storage";
    return false;
  }

  simaai::neat::Mapping mapping = tensor.map(simaai::neat::MapMode::Read);
  if (!mapping.data) {
    if (err)
      *err = "tensor copy: map failed";
    return false;
  }
  const uint8_t* base = static_cast<const uint8_t*>(mapping.data);

  if (tensor.is_composite()) {
    std::memset(dst, 0, dst_bytes);
    for (const auto& plane : tensor.planes) {
      const std::size_t plane_bytes = tensor_plane_bytes_tight(plane, tensor.dtype);
      if (plane_bytes == 0)
        continue;
      if (plane.byte_offset < 0) {
        if (err)
          *err = "tensor copy: plane offset invalid";
        return false;
      }
      const std::size_t plane_offset = static_cast<std::size_t>(plane.byte_offset);
      if (plane_offset + plane_bytes > dst_bytes) {
        if (err)
          *err = "tensor copy: plane exceeds buffer";
        return false;
      }
      if (plane_offset + plane_bytes > mapping.size_bytes) {
        if (err)
          *err = "tensor copy: plane out of range";
        return false;
      }
      std::memcpy(dst + plane_offset, base + plane_offset, plane_bytes);
    }
    return true;
  }

  if (dst_bytes > mapping.size_bytes) {
    if (err)
      *err = "tensor copy: buffer out of range";
    return false;
  }
  std::memcpy(dst, base, dst_bytes);
  return true;
}

bool canonicalize_sample_spec(SampleSpec* spec, std::string* err) {
  if (!spec) {
    if (err)
      *err = "spec: missing SampleSpec";
    return false;
  }
  if (!spec->format.empty()) {
    std::string fmt = upper_copy(spec->format);
    if (fmt == "GRAY")
      fmt = "GRAY8";
    spec->format = std::move(fmt);
  }
  return true;
}

bool validate_buffer_video_meta(GstBuffer* buffer, const SampleSpec& spec, std::string* err) {
  if (spec.kind != SampleMediaKind::RawVideo)
    return true;
  if (!buffer) {
    if (err)
      *err = "video meta: missing GstBuffer";
    return false;
  }
  GstVideoMeta* meta = gst_buffer_get_video_meta(buffer);
  if (!meta) {
    if (err)
      *err = "video meta: missing GstVideoMeta";
    return false;
  }
  if (spec.width > 0 && meta->width != static_cast<guint>(spec.width)) {
    if (err)
      *err = "video meta: width mismatch";
    return false;
  }
  if (spec.height > 0 && meta->height != static_cast<guint>(spec.height)) {
    if (err)
      *err = "video meta: height mismatch";
    return false;
  }
  if (!spec.format.empty()) {
    GstVideoFormat expected = gst_video_format_from_string(spec.format.c_str());
    if (expected != GST_VIDEO_FORMAT_UNKNOWN && meta->format != expected) {
      if (err)
        *err = "video meta: format mismatch";
      return false;
    }
  }
  if (!spec.planes.empty()) {
    if (meta->n_planes != spec.planes.size()) {
      if (err)
        *err = "video meta: plane count mismatch";
      return false;
    }
    for (guint i = 0; i < meta->n_planes; ++i) {
      const PlaneInfo& p = spec.planes[i];
      if (meta->offset[i] != static_cast<gsize>(p.offset_bytes) ||
          meta->stride[i] != static_cast<gint>(p.stride_bytes)) {
        if (err)
          *err = "video meta: plane stride/offset mismatch";
        return false;
      }
    }
  }
  if (spec.required_bytes_actual > 0) {
    const size_t buf_size = gst_buffer_get_size(buffer);
    if (spec.required_bytes_actual > buf_size) {
      if (err)
        *err = "video meta: buffer smaller than required bytes";
      return false;
    }
  }
  return true;
}

bool append_memory_shares(GstBuffer* dst, GstBuffer* src, std::size_t size_needed,
                          std::string* err) {
  if (!dst || !src) {
    if (err)
      *err = "zero-copy view: missing buffer";
    return false;
  }
  const guint n_mems = gst_buffer_n_memory(src);
  if (n_mems == 0) {
    if (err)
      *err = "zero-copy view: source has no memories";
    return false;
  }

  std::size_t remaining = size_needed;
  for (guint i = 0; i < n_mems; ++i) {
    GstMemory* mem = gst_buffer_peek_memory(src, i);
    if (!mem) {
      if (err)
        *err = "zero-copy view: missing source memory";
      return false;
    }
    if (!mem->allocator || !mem->allocator->mem_share) {
      if (err) {
        const char* mem_type = mem->allocator ? mem->allocator->mem_type : "unknown";
        std::string msg = "zero-copy view: memory not shareable";
        if (mem_type) {
          msg += " (";
          msg += mem_type;
          msg += ")";
        }
        *err = std::move(msg);
      }
      return false;
    }
    gsize maxsize = 0;
    const gsize mem_size = gst_memory_get_sizes(mem, nullptr, &maxsize);
    if (mem_size == 0) {
      continue;
    }
    gsize take = mem_size;
    if (size_needed > 0) {
      if (remaining == 0)
        break;
      take = static_cast<gsize>(std::min<std::size_t>(mem_size, remaining));
    }
    GstMemory* share = gst_memory_share(mem, 0, take);
    if (!share) {
      if (err)
        *err = "zero-copy view: gst_memory_share failed";
      return false;
    }
    gst_buffer_append_memory(dst, share);
    if (size_needed > 0) {
      remaining -= static_cast<std::size_t>(take);
    }
  }

  if (size_needed > 0 && remaining > 0) {
    if (err)
      *err = "zero-copy view: source buffer smaller than required bytes";
    return false;
  }
  return true;
}

gboolean copy_structure_field(GQuark field_id, const GValue* value, gpointer user_data) {
  GstStructure* dst = static_cast<GstStructure*>(user_data);
  if (!dst || !value)
    return TRUE;
  const gchar* name = g_quark_to_string(field_id);
  if (!name)
    return TRUE;
  gst_structure_set_value(dst, name, value);
  return TRUE;
}

void copy_custom_meta(GstBuffer* dst, GstBuffer* src, const char* meta_name) {
  if (!dst || !src || !meta_name)
    return;
  GstCustomMeta* src_meta = gst_buffer_get_custom_meta(src, meta_name);
  if (!src_meta)
    return;
  GstCustomMeta* dst_meta = gst_buffer_add_custom_meta(dst, meta_name);
  if (!dst_meta)
    return;
  GstStructure* src_struct = gst_custom_meta_get_structure(src_meta);
  GstStructure* dst_struct = gst_custom_meta_get_structure(dst_meta);
  if (!src_struct || !dst_struct)
    return;
  gst_structure_remove_all_fields(dst_struct);
  gst_structure_foreach(src_struct, copy_structure_field, dst_struct);
}

GstBuffer* make_zero_copy_view(GstBuffer* src, const SampleSpec& spec, std::string* err) {
  if (!src) {
    if (err)
      *err = "zero-copy view: missing source buffer";
    return nullptr;
  }
  GstBuffer* view = gst_buffer_new();
  if (!view) {
    if (err)
      *err = "zero-copy view: gst_buffer_new failed";
    return nullptr;
  }

  gst_buffer_copy_into(view, src,
                       static_cast<GstBufferCopyFlags>(GST_BUFFER_COPY_FLAGS |
                                                       GST_BUFFER_COPY_TIMESTAMPS |
                                                       GST_BUFFER_COPY_MEMORY),
                       0, -1);

  gst_buffer_add_parent_buffer_meta(view, src);
  copy_custom_meta(view, src, "GstSimaMeta");
  copy_custom_meta(view, src, "GstSimaSampleMeta");
  if (has_simaai_preprocess_meta(src)) {
    std::string copy_err;
    if (!copy_simaai_preprocess_meta(view, src, &copy_err)) {
      if (err) {
        *err = copy_err.empty()
                   ? "zero-copy view: failed to preserve preprocess metadata"
                   : "zero-copy view: failed to preserve preprocess metadata: " + copy_err;
      }
      gst_buffer_unref(view);
      return nullptr;
    }
  }
  return view;
}

bool ensure_writable_for_meta(GstBuffer** buffer, const SampleSpec& spec, const char* tag,
                              std::string* err) {
  if (!buffer || !*buffer) {
    if (err)
      *err = "zero-copy view: missing GstBuffer";
    return false;
  }
  if (gst_buffer_is_writable(*buffer)) {
    return true;
  }
  if (!zero_copy_writable_view_enabled()) {
    if (err) {
      std::string msg = "buffer not writable for ";
      msg += (tag ? tag : "meta");
      msg += " (enable SIMA_GST_ZERO_COPY_WRITABLE_VIEW=1 to test zero-copy view)";
      *err = msg;
    }
    return false;
  }
  std::string view_err;
  GstBuffer* view = make_zero_copy_view(*buffer, spec, &view_err);
  if (!view) {
    if (err) {
      std::string msg = "zero-copy view failed";
      if (!view_err.empty()) {
        msg += ": ";
        msg += view_err;
      }
      *err = msg;
    }
    return false;
  }
  if (!gst_buffer_is_writable(view)) {
    gst_buffer_unref(view);
    if (err) {
      std::string msg = "zero-copy view still not writable for ";
      msg += (tag ? tag : "meta");
      *err = msg;
    }
    return false;
  }
  gst_buffer_unref(*buffer);
  *buffer = view;
  return true;
}

bool resolve_encoded_payload_bytes(const simaai::neat::Tensor& tensor, const SampleSpec& spec,
                                   std::size_t* out_bytes, std::string* err) {
  if (!out_bytes) {
    if (err)
      *err = "encoded bytes: missing output";
    return false;
  }
  if (spec.kind != SampleMediaKind::Encoded) {
    if (err)
      *err = "encoded bytes: SampleSpec is not encoded";
    return false;
  }
  simaai::neat::Mapping mapping = tensor.map(simaai::neat::MapMode::Read);
  if (!mapping.data) {
    if (err)
      *err = "encoded bytes: tensor map failed";
    return false;
  }
  std::size_t nbytes = spec.required_bytes_actual;
  if (nbytes == 0) {
    nbytes = static_cast<std::size_t>(mapping.size_bytes);
  }
  if (nbytes == 0) {
    if (err)
      *err = "encoded bytes: empty payload";
    return false;
  }
  if (nbytes > mapping.size_bytes) {
    if (err)
      *err = "encoded bytes: payload exceeds mapped size";
    return false;
  }
  *out_bytes = nbytes;
  return true;
}

GstBuffer* buffer_from_holder_if_gstsample(const simaai::neat::Tensor& tensor, std::string* err) {
  if (!tensor.storage) {
    if (err)
      *err = "holder: missing storage";
    return nullptr;
  }
  if (tensor.storage->kind != simaai::neat::StorageKind::GstSample) {
    if (err)
      *err = "holder: storage is not GstSample";
    return nullptr;
  }
  if (!tensor.storage->holder) {
    if (err)
      *err = "holder: missing holder";
    return nullptr;
  }
  GstBuffer* buf = buffer_from_tensor_holder(tensor.storage->holder);
  if (!buf) {
    if (err)
      *err = "holder: missing GstBuffer";
    return nullptr;
  }
  return buf;
}

GstBuffer* build_copy_buffer_from_tensor(const simaai::neat::Tensor& tensor, const SampleSpec& spec,
                                         std::string* err) {
  std::size_t bytes = spec.required_bytes_actual;
  if (bytes == 0) {
    try {
      const std::vector<uint8_t> payload = tensor.copy_payload_bytes();
      bytes = payload.size();
      GstBuffer* buf = gst_buffer_new_allocate(nullptr, bytes, nullptr);
      if (!buf) {
        if (err)
          *err = "copy path: GstBuffer allocation failed";
        return nullptr;
      }
      if (!payload.empty()) {
        GstMapInfo map{};
        if (!gst_buffer_map(buf, &map, GST_MAP_WRITE)) {
          gst_buffer_unref(buf);
          if (err)
            *err = "copy path: GstBuffer map failed";
          return nullptr;
        }
        std::memcpy(map.data, payload.data(), payload.size());
        gst_buffer_unmap(buf, &map);
      }
      if (tensor.semantic.preprocess.has_value() &&
          !write_simaai_preprocess_meta(buf, *tensor.semantic.preprocess)) {
        gst_buffer_unref(buf);
        if (err)
          *err = "copy path: failed to apply tensor preprocess metadata";
        return nullptr;
      }
      return buf;
    } catch (const std::exception& e) {
      if (err)
        *err = std::string("copy path: ") + e.what();
      return nullptr;
    }
  }

  GstBuffer* buf = gst_buffer_new_allocate(nullptr, bytes, nullptr);
  if (!buf) {
    if (err)
      *err = "copy path: GstBuffer allocation failed";
    return nullptr;
  }
  if (bytes > 0) {
    GstMapInfo map{};
    if (!gst_buffer_map(buf, &map, GST_MAP_WRITE)) {
      gst_buffer_unref(buf);
      if (err)
        *err = "copy path: GstBuffer map failed";
      return nullptr;
    }
    std::string copy_err;
    if (!copy_tensor_payload_to(tensor, static_cast<uint8_t*>(map.data), bytes, &copy_err)) {
      gst_buffer_unmap(buf, &map);
      gst_buffer_unref(buf);
      if (err)
        *err = copy_err.empty() ? "copy path: tensor copy failed" : copy_err;
      return nullptr;
    }
    gst_buffer_unmap(buf, &map);
  }
  if (tensor.semantic.preprocess.has_value() &&
      !write_simaai_preprocess_meta(buf, *tensor.semantic.preprocess)) {
    gst_buffer_unref(buf);
    if (err)
      *err = "copy path: failed to apply tensor preprocess metadata";
    return nullptr;
  }
  return buf;
}

bool derive_field_spec(const Sample& field, SampleSpec* out, std::string* err) {
  if (!out) {
    if (err)
      *err = "field spec: missing output spec";
    return false;
  }
  const Sample normalized = (field.kind == SampleKind::Tensor && field.tensor.has_value())
                                ? canonicalize_tensor_transport_sample(field)
                                : field;
  if (!sample_has_tensor_list(normalized) || normalized.tensors.empty()) {
    if (err)
      *err = "field spec: missing tensor";
    return false;
  }

  const simaai::neat::Tensor& t = normalized.tensors.front();
  InputOptions opt;
  opt.payload_type =
      !normalized.media_type.empty()
          ? input_type_from_media_type(normalized.media_type)
          : (t.semantic.image.has_value() ? PayloadType::Image : PayloadType::Tensor);
  if (!normalized.payload_tag.empty()) {
    opt.format = normalized.payload_tag;
  } else if (!normalized.format.empty()) {
    opt.format = normalized.format;
  }

  try {
    SampleSpec spec = derive_tensor_spec_or_throw(t, opt, "GstDataAdapter::derive_field_spec");
    if (spec.caps_string.empty()) {
      if (err)
        *err = "field spec: empty caps_string";
      return false;
    }
    if (!normalized.caps_string.empty()) {
      spec.caps_string = normalized.caps_string;
    }
    *out = std::move(spec);
    return true;
  } catch (const std::exception& e) {
    if (data_adapter_debug_enabled()) {
      const auto& tensor = normalized.tensors.front();
      std::fprintf(stderr,
                   "[GstDataAdapter] derive_field_spec failure name=%s segment=%s media=%s "
                   "format=%s tensor=%s error=%s\n",
                   normalized.stream_label.empty() ? "<empty>" : normalized.stream_label.c_str(),
                   normalized.segment_name.empty() ? "<empty>" : normalized.segment_name.c_str(),
                   normalized.media_type.empty() ? "<empty>" : normalized.media_type.c_str(),
                   normalized.format.empty() ? "<empty>" : normalized.format.c_str(),
                   tensor.debug_string().c_str(), e.what());
    }
    if (err)
      *err = std::string("field spec: ") + e.what();
    return false;
  }
}

bool attach_video_meta(GstBuffer** buffer, const SampleSpec& spec, std::string* err) {
  if (spec.kind != SampleMediaKind::RawVideo)
    return true;
  if (!buffer || !*buffer) {
    if (err)
      *err = "video meta: missing GstBuffer";
    return false;
  }
  if (spec.width <= 0 || spec.height <= 0) {
    if (err)
      *err = "video meta: invalid video dimensions";
    return false;
  }
  if (spec.required_bytes_actual == 0) {
    if (err)
      *err = "video meta: bytes unknown";
    return false;
  }
  GstBuffer* buf = *buffer;
  const size_t buf_size = gst_buffer_get_size(buf);
  if (spec.required_bytes_actual > buf_size) {
    if (err)
      *err = "video meta: payload exceeds buffer size";
    return false;
  }

  GstVideoFormat fmt = gst_video_format_from_string(spec.format.c_str());
  if (fmt == GST_VIDEO_FORMAT_UNKNOWN) {
    if (err)
      *err = "video meta: unknown format";
    return false;
  }

  GstVideoMeta* existing = gst_buffer_get_video_meta(buf);
  if (existing) {
    bool mismatch = false;
    if (existing->width != static_cast<guint>(spec.width) ||
        existing->height != static_cast<guint>(spec.height) || existing->format != fmt) {
      mismatch = true;
    }
    if (!spec.planes.empty()) {
      if (existing->n_planes != spec.planes.size()) {
        mismatch = true;
      } else {
        for (guint i = 0; i < existing->n_planes; ++i) {
          const PlaneInfo& p = spec.planes[i];
          if (existing->offset[i] != static_cast<gsize>(p.offset_bytes) ||
              existing->stride[i] != static_cast<gint>(p.stride_bytes)) {
            mismatch = true;
            break;
          }
        }
      }
    }
    if (!mismatch) {
      if (buf_size == spec.required_bytes_actual) {
        return true;
      }
      if (!ensure_writable_for_meta(&buf, spec, "video meta resize", err)) {
        return false;
      }
      if (buf != *buffer) {
        *buffer = buf;
      }
      gst_buffer_resize(buf, 0, spec.required_bytes_actual);
      return true;
    }
    if (!ensure_writable_for_meta(&buf, spec, "video meta replace", err)) {
      return false;
    }
    if (buf != *buffer) {
      *buffer = buf;
    }
    gst_buffer_remove_meta(buf, &existing->meta);
  }

  if (spec.planes.empty()) {
    if (err)
      *err = "video meta: missing plane layout";
    return false;
  }
  if (spec.planes.size() > GST_VIDEO_MAX_PLANES) {
    if (err)
      *err = "video meta: too many planes";
    return false;
  }

  const std::string fmt_name = upper_copy(spec.format);
  if (fmt_name == "NV12") {
    if (spec.planes.size() != 2) {
      if (err)
        *err = "video meta: NV12 requires 2 planes";
      return false;
    }
    const PlaneInfo& y = spec.planes[0];
    const PlaneInfo& uv = spec.planes[1];
    if (y.width != spec.width || y.height != spec.height || uv.width != spec.width ||
        uv.height != spec.height / 2) {
      if (err)
        *err = "video meta: NV12 plane dims mismatch";
      return false;
    }
  } else if (fmt_name == "I420") {
    if (spec.planes.size() != 3) {
      if (err)
        *err = "video meta: I420 requires 3 planes";
      return false;
    }
    const PlaneInfo& y = spec.planes[0];
    const PlaneInfo& u = spec.planes[1];
    const PlaneInfo& v = spec.planes[2];
    if (y.width != spec.width || y.height != spec.height || u.width != spec.width / 2 ||
        u.height != spec.height / 2 || v.width != spec.width / 2 || v.height != spec.height / 2) {
      if (err)
        *err = "video meta: I420 plane dims mismatch";
      return false;
    }
  } else {
    if (spec.planes.size() != 1) {
      if (err)
        *err = "video meta: packed video requires 1 plane";
      return false;
    }
    const PlaneInfo& p = spec.planes[0];
    if (p.width != spec.width || p.height != spec.height) {
      if (err)
        *err = "video meta: packed plane dims mismatch";
      return false;
    }
  }

  gsize offsets[GST_VIDEO_MAX_PLANES] = {0};
  gint strides[GST_VIDEO_MAX_PLANES] = {0};
  size_t max_end = 0;
  for (size_t i = 0; i < spec.planes.size(); ++i) {
    const PlaneInfo& p = spec.planes[i];
    if (p.stride_bytes <= 0 || p.offset_bytes < 0) {
      if (err)
        *err = "video meta: invalid plane stride/offset";
      return false;
    }
    offsets[i] = static_cast<gsize>(p.offset_bytes);
    strides[i] = static_cast<gint>(p.stride_bytes);
    const size_t end = offsets[i] + p.size_bytes;
    if (end > max_end)
      max_end = end;
  }
  if (max_end > spec.required_bytes_actual) {
    if (err)
      *err = "video meta: plane bytes exceed expected size";
    return false;
  }

  if (!ensure_writable_for_meta(&buf, spec, "video meta attach", err)) {
    return false;
  }
  if (buf != *buffer) {
    *buffer = buf;
  }

  gst_buffer_resize(buf, 0, spec.required_bytes_actual);
  GstVideoMeta* meta = gst_buffer_add_video_meta_full(
      buf, GST_VIDEO_FRAME_FLAG_NONE, fmt, static_cast<guint>(spec.width),
      static_cast<guint>(spec.height), static_cast<guint>(spec.planes.size()), offsets, strides);
  if (!meta) {
    if (err)
      *err = "video meta: attach failed";
    return false;
  }
  return true;
}

bool apply_tensor_size(GstBuffer** buffer, const SampleSpec& spec, std::string* err) {
  if (spec.kind != SampleMediaKind::Tensor)
    return true;
  if (!buffer || !*buffer) {
    if (err)
      *err = "tensor size: missing GstBuffer";
    return false;
  }
  if (spec.required_bytes_actual == 0) {
    if (err)
      *err = "tensor size: bytes unknown";
    return false;
  }
  GstBuffer* buf = *buffer;
  const size_t buf_size = gst_buffer_get_size(buf);
  if (spec.required_bytes_actual > buf_size) {
    if (err)
      *err = "tensor size: payload exceeds buffer size";
    return false;
  }
  if (buf_size == spec.required_bytes_actual)
    return true;
  if (!ensure_writable_for_meta(&buf, spec, "tensor size", err)) {
    return false;
  }
  if (buf != *buffer) {
    *buffer = buf;
  }
  gst_buffer_resize(buf, 0, spec.required_bytes_actual);
  return true;
}

GstBuffer* build_gst_buffer_from_tensor(const simaai::neat::Tensor& tensor, const SampleSpec& spec,
                                        const GstBufferBuildPolicy& policy, std::string* err) {
  auto fail = [&](const std::string& msg) -> GstBuffer* {
    if (err)
      *err = msg;
    return nullptr;
  };

  if (policy.allow_zero_copy) {
    std::string holder_err;
    GstBuffer* buf = buffer_from_holder_if_gstsample(tensor, &holder_err);
    if (buf) {
      std::string finalize_err;
      if (!finalize_buffer(&buf, spec, policy, &finalize_err)) {
        gst_buffer_unref(buf);
        if (!finalize_err.empty()) {
          log_zero_copy_skip("holder_finalize", finalize_err.c_str(), tensor, spec);
        }
        if (!zero_copy_writable_view_enabled()) {
          return fail(finalize_err);
        }
      } else {
        return buf;
      }
    }
    if (!holder_err.empty()) {
      log_zero_copy_skip("holder_path", holder_err.c_str(), tensor, spec);
    }

    if (tensor.storage && (tensor.storage->kind == simaai::neat::StorageKind::CpuOwned ||
                           tensor.storage->kind == simaai::neat::StorageKind::CpuExternal)) {
      GstBuffer* buf = nullptr;
      std::string zc_err;
      bool ok = false;
      if (spec.kind == SampleMediaKind::RawVideo) {
        ok = wrap_cpu_video_zero_copy(tensor, spec, &buf, &zc_err);
      } else if (spec.kind == SampleMediaKind::Tensor) {
        if (policy.require_contiguous && (!tensor.is_dense() || !tensor.is_contiguous())) {
          zc_err = "zero-copy: policy requires contiguous tensor";
          ok = false;
        } else {
          ok = wrap_cpu_dense_zero_copy(tensor, &buf, &zc_err);
        }
      }

      if (ok && buf) {
        std::string finalize_err;
        if (!finalize_buffer(&buf, spec, policy, &finalize_err)) {
          gst_buffer_unref(buf);
          if (!finalize_err.empty()) {
            log_zero_copy_skip("cpu_finalize", finalize_err.c_str(), tensor, spec);
          }
          if (!zero_copy_writable_view_enabled()) {
            return fail(finalize_err);
          }
        } else {
          return buf;
        }
      }
      if (!zc_err.empty()) {
        log_zero_copy_skip("cpu_zero_copy", zc_err.c_str(), tensor, spec);
      }
    } else if (!policy.allow_device_memory) {
      log_zero_copy_skip("device_memory", "not_allowed", tensor, spec);
    } else {
      log_zero_copy_skip("device_memory", "not_implemented", tensor, spec);
    }
  }

  GstBuffer* buf = build_copy_buffer_from_tensor(tensor, spec, err);
  if (!buf) {
    return nullptr;
  }

  std::string finalize_err;
  if (!finalize_buffer(&buf, spec, policy, &finalize_err)) {
    gst_buffer_unref(buf);
    return fail(finalize_err);
  }
  return buf;
}

GstSample* build_gst_sample_from_tensor(const simaai::neat::Tensor& tensor, const SampleSpec& spec,
                                        const GstBufferBuildPolicy& policy, GstCaps* override_caps,
                                        std::string* err) {
  GstBuffer* buf = build_gst_buffer_from_tensor(tensor, spec, policy, err);
  if (!buf)
    return nullptr;

  GstCaps* caps = nullptr;
  if (override_caps) {
    caps = gst_caps_ref(override_caps);
  } else {
    caps = caps_from_spec(spec);
  }

  GstSample* sample = gst_sample_new(buf, caps, nullptr, nullptr);
  if (!sample) {
    if (caps)
      gst_caps_unref(caps);
    gst_buffer_unref(buf);
    if (err)
      *err = "sample build: gst_sample_new failed";
    return nullptr;
  }
  return sample;
}

} // namespace simaai::neat::pipeline_internal

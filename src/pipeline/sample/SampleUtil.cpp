// src/pipeline/internal/SampleUtil.cpp
#include "pipeline/internal/SampleUtil.h"

#include "InputStreamUtil.h"
#include "pipeline/internal/GstDataAdapter.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"

#include <gst/gst.h>

#include <cstdio>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal {
namespace {

constexpr const char* kSampleMetaName = "GstSimaSampleMeta";

const char* storage_kind_name(simaai::neat::StorageKind kind) {
  switch (kind) {
  case simaai::neat::StorageKind::CpuOwned:
    return "CpuOwned";
  case simaai::neat::StorageKind::CpuExternal:
    return "CpuExternal";
  case simaai::neat::StorageKind::GstSample:
    return "GstSample";
  case simaai::neat::StorageKind::DeviceHandle:
    return "DeviceHandle";
  case simaai::neat::StorageKind::Unknown:
  default:
    return "Unknown";
  }
}

const char* plane_role_name(simaai::neat::PlaneRole role) {
  switch (role) {
  case simaai::neat::PlaneRole::Y:
    return "Y";
  case simaai::neat::PlaneRole::U:
    return "U";
  case simaai::neat::PlaneRole::V:
    return "V";
  case simaai::neat::PlaneRole::UV:
    return "UV";
  case simaai::neat::PlaneRole::Unknown:
  default:
    return "Unknown";
  }
}

std::string join_dims(const std::vector<int64_t>& dims, char sep) {
  std::ostringstream ss;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i)
      ss << sep;
    ss << dims[i];
  }
  return ss.str();
}

bool sample_debug_enabled() {
  return env_bool("SIMA_SAMPLE_DEBUG", false);
}

bool sample_bytes_enabled() {
  return env_bool("SIMA_SAMPLE_BYTES", false);
}

void log_bundle_field(const Sample& field) {
  std::ostringstream ss;
  const std::string name = field.port_name.empty() ? "field" : field.port_name;
  ss << "[SAMPLE] field name=" << name;
  if (!field.caps_string.empty()) {
    ss << " caps=" << field.caps_string;
  }
  if (!field.tensor.has_value()) {
    ss << " neat=<missing>";
    std::fprintf(stderr, "%s\n", ss.str().c_str());
    return;
  }
  const simaai::neat::Tensor& t = *field.tensor;
  ss << " " << t.debug_string();
  if (t.storage) {
    ss << " storage=" << storage_kind_name(t.storage->kind) << " size=" << t.storage->size_bytes;
  } else {
    ss << " storage=<none>";
  }
  for (size_t i = 0; i < t.planes.size(); ++i) {
    const simaai::neat::Plane& plane = t.planes[i];
    ss << " plane[" << i << "]=" << plane_role_name(plane.role)
       << " shape=" << join_dims(plane.shape, 'x')
       << " strides=" << join_dims(plane.strides_bytes, ',') << " offset=" << plane.byte_offset;
  }
  std::fprintf(stderr, "%s\n", ss.str().c_str());
}

void log_bundle(const Sample& bundle) {
  std::ostringstream ss;
  ss << "[SAMPLE] bundle fields=" << bundle.fields.size() << " frame_id=" << bundle.frame_id;
  if (!bundle.stream_id.empty()) {
    ss << " stream_id=" << bundle.stream_id;
  }
  std::fprintf(stderr, "%s\n", ss.str().c_str());
  for (const auto& field : bundle.fields) {
    log_bundle_field(field);
  }
}

GstBuffer* buffer_from_tensor_or_copy(const Sample& field, const SampleSpec& spec,
                                      std::string* err) {
  if (!field.tensor.has_value()) {
    if (err)
      *err = "Sample field missing tensor";
    return nullptr;
  }
  const simaai::neat::Tensor& t = *field.tensor;
  const char* name = field.port_name.empty() ? "field" : field.port_name.c_str();
  GstBufferBuildPolicy policy;
  policy.allow_zero_copy = true;
  policy.require_video_meta = true;
  policy.allow_appsrc_pool = false;
  policy.require_contiguous = true;
  policy.allow_device_memory = false;

  GstBuffer* buf = build_gst_buffer_from_tensor(t, spec, policy, err);
  if (!buf)
    return nullptr;
  if (sample_bytes_enabled()) {
    const size_t buf_bytes = static_cast<size_t>(gst_buffer_get_size(buf));
    std::fprintf(stderr, "[SAMPLE] field name=%s source=adapter bytes=%zu\n", name, buf_bytes);
  }
  return buf;
}

bool add_field_to_list(GValue* list, const Sample& field, GstBuffer* buf,
                       const std::string& buffer_name) {
  if (!list || !buf)
    return false;
  const char* field_name = field.port_name.empty() ? "field" : field.port_name.c_str();
  const char* caps = field.caps_string.empty() ? nullptr : field.caps_string.c_str();

  GstStructure* entry = gst_structure_new("simaai-sample-field", "name", G_TYPE_STRING, field_name,
                                          "buffer", GST_TYPE_BUFFER, buf, "buffer-name",
                                          G_TYPE_STRING, buffer_name.c_str(), nullptr);
  if (caps) {
    gst_structure_set(entry, "caps", G_TYPE_STRING, caps, nullptr);
  }

  GValue entry_val = G_VALUE_INIT;
  g_value_init(&entry_val, GST_TYPE_STRUCTURE);
  g_value_take_boxed(&entry_val, entry);

  gst_value_list_append_value(list, &entry_val);
  g_value_unset(&entry_val);
  return true;
}

bool ensure_sima_meta_fields(GstBuffer* buffer, const std::optional<int64_t>& frame_id,
                             const std::optional<int64_t>& input_seq,
                             const std::optional<int64_t>& orig_input_seq,
                             const std::optional<std::string>& stream_id,
                             const std::optional<std::string>& buffer_name) {
  if (!buffer)
    return false;
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  if (!meta) {
    meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  }
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s)
    return false;

  if (frame_id.has_value()) {
    gst_structure_set(s, "frame-id", G_TYPE_INT64, static_cast<gint64>(*frame_id), nullptr);
  }
  if (input_seq.has_value()) {
    gst_structure_set(s, "input-seq", G_TYPE_INT64, static_cast<gint64>(*input_seq), nullptr);
  }
  if (orig_input_seq.has_value()) {
    gst_structure_set(s, "orig-input-seq", G_TYPE_INT64, static_cast<gint64>(*orig_input_seq),
                      nullptr);
  } else if (input_seq.has_value()) {
    gst_structure_set(s, "orig-input-seq", G_TYPE_INT64, static_cast<gint64>(*input_seq), nullptr);
  } else if (frame_id.has_value()) {
    gst_structure_set(s, "orig-input-seq", G_TYPE_INT64, static_cast<gint64>(*frame_id), nullptr);
  }
  if (stream_id.has_value()) {
    gst_structure_set(s, "stream-id", G_TYPE_STRING, stream_id->c_str(), nullptr);
    gst_structure_set(s, "orig-stream-id", G_TYPE_STRING, stream_id->c_str(), nullptr);
  }
  if (buffer_name.has_value()) {
    gst_structure_set(s, "buffer-name", G_TYPE_STRING, buffer_name->c_str(), nullptr);
  }
  gst_structure_set(s, "buffer-offset", G_TYPE_INT64, static_cast<gint64>(0), nullptr);
  return true;
}

} // namespace

std::shared_ptr<void> make_sample_holder_from_bundle(const Sample& bundle, std::string* err) {
  if (bundle.kind != SampleKind::Bundle) {
    if (err)
      *err = "Sample bundle expected";
    return {};
  }
  if (bundle.fields.empty()) {
    if (err)
      *err = "Sample bundle has no fields";
    return {};
  }
  if (sample_debug_enabled()) {
    log_bundle(bundle);
  }

  GstBuffer* sample_buf = gst_buffer_new();
  if (!sample_buf) {
    if (err)
      *err = "Sample buffer allocation failed";
    return {};
  }

  GstCustomMeta* meta = gst_buffer_add_custom_meta(sample_buf, kSampleMetaName);
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s) {
    gst_buffer_unref(sample_buf);
    if (err)
      *err = "Sample meta attach failed";
    return {};
  }
  const std::optional<int64_t> bundle_input_seq =
      bundle.input_seq >= 0 ? std::optional<int64_t>(bundle.input_seq) : std::nullopt;
  const std::optional<int64_t> bundle_orig_input_seq =
      bundle.orig_input_seq >= 0 ? std::optional<int64_t>(bundle.orig_input_seq) : bundle_input_seq;
  ensure_sima_meta_fields(
      sample_buf, bundle.frame_id >= 0 ? std::optional<int64_t>(bundle.frame_id) : std::nullopt,
      bundle_input_seq, bundle_orig_input_seq,
      bundle.stream_id.empty() ? std::nullopt : std::optional<std::string>(bundle.stream_id),
      std::optional<std::string>("bundle"));

  GValue list = G_VALUE_INIT;
  g_value_init(&list, GST_TYPE_LIST);

  for (const auto& field : bundle.fields) {
    SampleSpec spec;
    std::string caps_err;
    if (!derive_field_spec(field, &spec, &caps_err)) {
      gst_buffer_unref(sample_buf);
      if (err)
        *err = caps_err.empty() ? "Sample field caps missing" : caps_err;
      return {};
    }

    std::string field_err;
    GstBuffer* buf = buffer_from_tensor_or_copy(field, spec, &field_err);
    if (!buf) {
      gst_buffer_unref(sample_buf);
      if (err)
        *err = field_err.empty() ? "Sample field buffer failed" : field_err;
      return {};
    }
    if (sample_debug_enabled()) {
      std::fprintf(stderr, "[SAMPLE] field name=%s caps_string=%s\n",
                   field.port_name.empty() ? "field" : field.port_name.c_str(),
                   spec.caps_string.empty() ? "<empty>" : spec.caps_string.c_str());
    }
    if (sample_bytes_enabled()) {
      const size_t buf_bytes = static_cast<size_t>(gst_buffer_get_size(buf));
      std::fprintf(stderr, "[SAMPLE] bundle field=%s buffer-bytes=%zu\n",
                   field.port_name.empty() ? "field" : field.port_name.c_str(), buf_bytes);
    }
    buf = gst_buffer_make_writable(buf);
    if (!buf) {
      gst_buffer_unref(sample_buf);
      if (err)
        *err = "Sample field buffer not writable";
      return {};
    }

    const std::string buffer_name =
        field.port_name.empty() ? std::string("field") : field.port_name;
    const std::optional<int64_t> field_input_seq =
        bundle.input_seq >= 0
            ? std::optional<int64_t>(bundle.input_seq)
            : (field.input_seq >= 0 ? std::optional<int64_t>(field.input_seq) : std::nullopt);
    const std::optional<int64_t> field_orig_input_seq =
        bundle.orig_input_seq >= 0
            ? std::optional<int64_t>(bundle.orig_input_seq)
            : (field.orig_input_seq >= 0 ? std::optional<int64_t>(field.orig_input_seq)
                                         : field_input_seq);
    update_simaai_meta_fields(
        buf, bundle.frame_id >= 0 ? std::optional<int64_t>(bundle.frame_id) : std::nullopt,
        field_input_seq, field_orig_input_seq,
        bundle.stream_id.empty() ? std::nullopt : std::optional<std::string>(bundle.stream_id),
        buffer_name);
    ensure_sima_meta_fields(
        buf, bundle.frame_id >= 0 ? std::optional<int64_t>(bundle.frame_id) : std::nullopt,
        field_input_seq, field_orig_input_seq,
        bundle.stream_id.empty() ? std::nullopt : std::optional<std::string>(bundle.stream_id),
        std::optional<std::string>(buffer_name));

    Sample field_with_caps = field;
    field_with_caps.caps_string = spec.caps_string;
    if (!add_field_to_list(&list, field_with_caps, buf, buffer_name)) {
      gst_buffer_unref(buf);
      gst_buffer_unref(sample_buf);
      if (err)
        *err = "Sample meta field insert failed";
      return {};
    }
    gst_buffer_unref(buf);
  }
  gst_structure_set_value(s, "fields", &list);
  g_value_unset(&list);

  GstSample* sample = gst_sample_new(sample_buf, nullptr, nullptr, nullptr);
  gst_buffer_unref(sample_buf);
  if (!sample) {
    if (err)
      *err = "Sample wrap failed";
    return {};
  }
  auto holder = std::shared_ptr<void>(
      gst_sample_ref(sample), [](void* p) { gst_sample_unref(static_cast<GstSample*>(p)); });
  gst_sample_unref(sample);
  return holder;
}

} // namespace simaai::neat::pipeline_internal

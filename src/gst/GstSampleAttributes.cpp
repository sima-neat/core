#include "gst/GstSampleAttributes.h"

#include <string>

namespace simaai::neat::gst_internal {
namespace {

constexpr const char* kSimaMetaName = "GstSimaMeta";

/// Nested attribute structures are always built with this name so serialized dumps and
/// debug output identify them unambiguously.
constexpr const char* kAttributesStructureName = "sima.sample.attributes";

GstStructure* meta_structure(GstBuffer* buffer, bool create) {
  if (!buffer) {
    return nullptr;
  }
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, kSimaMetaName);
  if (!meta && create) {
    if (!gst_buffer_is_writable(buffer)) {
      return nullptr;
    }
    meta = gst_buffer_add_custom_meta(buffer, kSimaMetaName);
  }
  return meta ? gst_custom_meta_get_structure(meta) : nullptr;
}

gboolean collect_field(GQuark field_id, const GValue* value, gpointer user_data) {
  auto* out = static_cast<SampleAttributes*>(user_data);
  if (G_VALUE_HOLDS_STRING(value)) {
    const gchar* text = g_value_get_string(value);
    (*out)[g_quark_to_string(field_id)] = text ? text : "";
  }
  return TRUE;
}

} // namespace

void read_attributes_from_structure(const GstStructure* meta_structure_in, SampleAttributes* out) {
  if (!out) {
    return;
  }
  out->clear();
  if (!meta_structure_in) {
    return;
  }
  const GValue* value = gst_structure_get_value(meta_structure_in, kSimaMetaAttributesField);
  if (!value || !GST_VALUE_HOLDS_STRUCTURE(value)) {
    return;
  }
  const GstStructure* nested = gst_value_get_structure(value);
  if (!nested) {
    return;
  }
  gst_structure_foreach(nested, collect_field, out);
}

void read_attributes(GstBuffer* buffer, SampleAttributes* out) {
  if (!out) {
    return;
  }
  read_attributes_from_structure(meta_structure(buffer, false), out);
}

bool write_attributes_to_structure(GstStructure* target, const SampleAttributes& attributes) {
  if (!target) {
    return false;
  }
  for (const auto& [key, value] : attributes) {
    if (key.find('\0') != std::string::npos || value.find('\0') != std::string::npos) {
      return false;
    }
  }
  // Always drop the previous value first: a reused destination must never retain stale
  // attributes when the incoming set is smaller or empty.
  gst_structure_remove_field(target, kSimaMetaAttributesField);
  if (attributes.empty()) {
    return true;
  }
  GstStructure* nested = gst_structure_new_empty(kAttributesStructureName);
  for (const auto& [key, value] : attributes) {
    gst_structure_set(nested, key.c_str(), G_TYPE_STRING, value.c_str(), nullptr);
  }
  // Hand the structure over instead of letting gst_structure_set() deep-copy it: this runs
  // once per decoded frame, and at line-scan rates the redundant copy is measurable.
  GValue value = G_VALUE_INIT;
  g_value_init(&value, GST_TYPE_STRUCTURE);
  g_value_take_boxed(&value, nested);
  gst_structure_take_value(target, kSimaMetaAttributesField, &value);
  return true;
}

bool write_attributes(GstBuffer* buffer, const SampleAttributes& attributes) {
  if (!buffer) {
    return false;
  }
  if (attributes.empty()) {
    // Nothing to write; only clear when a meta already exists.
    GstStructure* existing = meta_structure(buffer, false);
    if (!existing) {
      return true;
    }
    if (!gst_buffer_is_writable(buffer)) {
      return false;
    }
    gst_structure_remove_field(existing, kSimaMetaAttributesField);
    return true;
  }
  GstStructure* target = meta_structure(buffer, true);
  if (!target) {
    return false;
  }
  return write_attributes_to_structure(target, attributes);
}

bool has_attributes(GstBuffer* buffer) {
  GstStructure* existing = meta_structure(buffer, false);
  if (!existing) {
    return false;
  }
  const GValue* value = gst_structure_get_value(existing, kSimaMetaAttributesField);
  return value != nullptr && GST_VALUE_HOLDS_STRUCTURE(value);
}

bool clear_attributes(GstBuffer* buffer) {
  GstStructure* existing = meta_structure(buffer, false);
  if (!existing) {
    return true;
  }
  if (!gst_buffer_is_writable(buffer)) {
    return false;
  }
  gst_structure_remove_field(existing, kSimaMetaAttributesField);
  return true;
}

bool copy_attributes(GstBuffer* src, GstBuffer* dst) {
  if (!dst) {
    return false;
  }
  SampleAttributes attributes;
  read_attributes(src, &attributes);
  return write_attributes(dst, attributes);
}

} // namespace simaai::neat::gst_internal

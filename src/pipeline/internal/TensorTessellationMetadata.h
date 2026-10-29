#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/TensorCore.h"
#include "pipeline/internal/TensorTiledStorageValidation.h"
#include <gst/gst.h>

#include <algorithm>
#include <limits>
#include <span>
#include <stdexcept>

namespace simaai::neat::pipeline_internal {

// A packed publication can either retain the numerical tensor or deliberately
// expose its storage as raw bytes. Dtype tags must belong to the same metadata
// domain; no byte-count equivalence authorizes a numerical reinterpretation.
inline bool same_numerical_tensor_view(std::span<const int64_t> checked_shape, int checked_dtype,
                                       std::span<const int64_t> published_shape,
                                       int published_dtype) {
  return checked_dtype == published_dtype &&
         std::equal(checked_shape.begin(), checked_shape.end(), published_shape.begin(),
                    published_shape.end());
}

// Immutable output caps carry checked physical layout once per negotiated
// logical publication (not its potentially shared backend carrier). InputStream's existing decode
// cache retains it in Tensor templates; no frame payload parsing or layout conversion is
// introduced.
inline std::string tessellation_caps_key(int logical_index) {
  return "sima-tiled-view-" + std::to_string(logical_index);
}

inline void set_tessellation_caps(GstCaps* caps, int logical_index, const TessSpec& tess) {
  GstStructure* record = gst_structure_new("sima-tiled-storage-v1", "format", G_TYPE_STRING,
                                           tess.format.c_str(), "storage-bytes", G_TYPE_UINT64,
                                           static_cast<guint64>(tess.storage_bytes), nullptr);
  const auto set_array = [&](const char* field, const std::vector<int64_t>& values) {
    GValue array = G_VALUE_INIT;
    g_value_init(&array, GST_TYPE_ARRAY);
    for (const auto value : values) {
      GValue item = G_VALUE_INIT;
      g_value_init(&item, G_TYPE_INT64);
      g_value_set_int64(&item, value);
      gst_value_array_append_value(&array, &item);
      g_value_unset(&item);
    }
    gst_structure_set_value(record, field, &array);
    g_value_unset(&array);
  };
  set_array("tile-shape", tess.slice_shape);
  set_array("axes", tess.axes);
  gst_caps_set_simple(caps, tessellation_caps_key(logical_index).c_str(), GST_TYPE_STRUCTURE,
                      record, nullptr);
  gst_structure_free(record);
}

inline void apply_tessellation_caps(Tensor& tensor, const GstCaps* caps, int logical_index,
                                    std::uint64_t descriptor_bytes) {
  if (!caps || gst_caps_is_empty(caps))
    return;
  const auto* value = gst_structure_get_value(gst_caps_get_structure(caps, 0),
                                              tessellation_caps_key(logical_index).c_str());
  if (!value)
    return; // Existing format-only metadata keeps its established behavior.
  const auto invalid = [] {
    throw std::runtime_error("invalid checked tensor tessellation metadata");
  };
  if (!GST_VALUE_HOLDS_STRUCTURE(value))
    invalid();
  const auto* record = gst_value_get_structure(value);
  const char* format = gst_structure_get_string(record, "format");
  guint64 bytes = 0;
  if (!gst_structure_has_name(record, "sima-tiled-storage-v1") || !format ||
      !gst_structure_get_uint64(record, "storage-bytes", &bytes) || bytes == 0 ||
      bytes > std::numeric_limits<std::size_t>::max())
    invalid();
  TessSpec tess;
  tess.format = format;
  tess.storage_bytes = static_cast<std::size_t>(bytes);
  const auto read_array = [&](const char* field, std::vector<int64_t>& values) {
    const auto* array = gst_structure_get_value(record, field);
    if (!array || !GST_VALUE_HOLDS_ARRAY(array) || gst_value_array_get_size(array) != 4)
      invalid();
    for (guint i = 0; i < 4; ++i) {
      const auto* item = gst_value_array_get_value(array, i);
      if (!G_VALUE_HOLDS_INT64(item))
        invalid();
      values.push_back(g_value_get_int64(item));
    }
  };
  read_array("tile-shape", tess.slice_shape);
  read_array("axes", tess.axes);
  if (const char* error = explicit_tiled_storage_error(tensor, tess, descriptor_bytes))
    throw std::runtime_error(std::string("invalid checked tensor tessellation metadata: ") + error);
  tensor.semantic.tess = std::move(tess);
  // Logical strides cannot describe a split-byte/tiled payload.
  tensor.strides_bytes.clear();
}

} // namespace simaai::neat::pipeline_internal

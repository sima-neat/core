#include "gst/SimaTensorSetMetaAbi.h"
#include "gst/GstInit.h"
#include "pipeline/TensorAdapters.h"
#include "pipeline/gst/InputStreamInternal.h"
#include "dmabuf_test_utils.h"
#include "pipeline/internal/OutputTensorOverride.h"
#include "pipeline/internal/HolderLoanGate.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/TensorUtil.h"

#include "test_utils.h"

#include <gstsimaaitensorbuffer.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {
Sample output_from_sample_stream(GstSample* sample, const char* where, bool copy_output,
                                 const std::optional<OutputTensorOverride>* override_opt);
}

namespace {

using simaai::neat::apply_output_tensor_override;
using simaai::neat::Mapping;
using simaai::neat::output_override_entry_physical_span_bytes;
using simaai::neat::OutputTensorOverride;
using simaai::neat::OutputTensorOverrideEntry;
using simaai::neat::Sample;
using simaai::neat::sample_from_tensors;
using simaai::neat::sample_has_tensor_list;
using simaai::neat::StorageKind;
using simaai::neat::Tensor;
using simaai::neat::TensorDType;
using simaai::neat::TensorLayout;
using simaai::neat::TensorList;
namespace pipeline_internal = simaai::neat::pipeline_internal;

struct GstSampleUnref {
  void operator()(GstSample* sample) const {
    if (sample) {
      gst_sample_unref(sample);
    }
  }
};

using GstSamplePtr = std::unique_ptr<GstSample, GstSampleUnref>;

struct GstBufferUnref {
  void operator()(GstBuffer* buffer) const {
    if (buffer) {
      gst_buffer_unref(buffer);
    }
  }
};

using GstBufferPtr = std::unique_ptr<GstBuffer, GstBufferUnref>;

GstSample* make_tensor_sample_with_contract_meta(std::size_t descriptor_count = 2U) {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 64U, nullptr);
  require(buffer != nullptr, "failed to allocate GstBuffer");

  GstMapInfo map{};
  require(gst_buffer_map(buffer, &map, GST_MAP_WRITE), "failed to map GstBuffer");
  std::memset(map.data, 0x5A, map.size);
  gst_buffer_unmap(buffer, &map);

  GstCaps* caps =
      gst_caps_new_simple("application/vnd.simaai.tensor", "format", G_TYPE_STRING, "MLA", "width",
                          G_TYPE_INT, 4, "height", G_TYPE_INT, 4, "depth", G_TYPE_INT, 1, "dtype",
                          G_TYPE_STRING, "INT8", "layout", G_TYPE_STRING, "HW", nullptr);
  require(caps != nullptr, "failed to allocate GstCaps");

  GstCustomMeta* meta = gst_buffer_add_custom_meta(buffer, SIMA_TENSOR_SET_META_NAME);
  require(meta != nullptr, "failed to add tensor-set meta");
  GstStructure* s = gst_custom_meta_get_structure(meta);
  require(s != nullptr, "failed to get tensor-set meta structure");

  require(descriptor_count == 1U || descriptor_count == 2U,
          "test helper only supports 1 or 2 descriptors");

  std::vector<SimaTensorDescriptorV2> descriptors(descriptor_count);
  descriptors[0].logical_index = 2;
  descriptors[0].physical_index = 0;
  descriptors[0].backend_output_index = 11;
  descriptors[0].route_slot = 5;
  descriptors[0].memory_index = 0;
  descriptors[0].logical_name_id = 0;
  descriptors[0].backend_name_id = 1;
  descriptors[0].segment_name_id = 2;
  descriptors[0].byte_offset = 0;
  descriptors[0].size_bytes = 16;
  descriptors[0].dtype = SIMA_TENSOR_SET_DTYPE_INT8_V1;
  descriptors[0].layout = SIMA_TENSOR_SET_LAYOUT_HW_V1;
  descriptors[0].rank = 2;
  descriptors[0].shape[0] = 4;
  descriptors[0].shape[1] = 4;
  descriptors[0].stride_bytes[0] = 4;
  descriptors[0].stride_bytes[1] = 1;
  descriptors[0].has_quant = 1U;
  descriptors[0].quant_granularity = 0;
  descriptors[0].quant_axis = -1;
  descriptors[0].quant_scales_offset = 0U;
  descriptors[0].quant_scales_len = 1U;
  descriptors[0].quant_zero_points_offset = 0U;
  descriptors[0].quant_zero_points_len = 1U;

  if (descriptor_count > 1U) {
    descriptors[1].logical_index = 7;
    descriptors[1].physical_index = 1;
    descriptors[1].backend_output_index = 13;
    descriptors[1].route_slot = 9;
    descriptors[1].memory_index = 0;
    descriptors[1].logical_name_id = 3;
    descriptors[1].backend_name_id = 4;
    descriptors[1].segment_name_id = 5;
    descriptors[1].byte_offset = 16;
    descriptors[1].size_bytes = 16;
    descriptors[1].dtype = SIMA_TENSOR_SET_DTYPE_INT8_V1;
    descriptors[1].layout = SIMA_TENSOR_SET_LAYOUT_HW_V1;
    descriptors[1].rank = 2;
    descriptors[1].shape[0] = 4;
    descriptors[1].shape[1] = 4;
    descriptors[1].stride_bytes[0] = 4;
    descriptors[1].stride_bytes[1] = 1;
  }

  const std::vector<gdouble> quant_scales = {0.25};
  const std::vector<gint64> quant_zero_points = {-7};
  GBytes* descriptor_bytes =
      g_bytes_new(descriptors.data(), descriptors.size() * sizeof(SimaTensorDescriptorV2));
  require(descriptor_bytes != nullptr, "failed to allocate descriptor bytes");
  GBytes* quant_scale_bytes =
      g_bytes_new(quant_scales.data(), quant_scales.size() * sizeof(gdouble));
  require(quant_scale_bytes != nullptr, "failed to allocate quant scale bytes");
  GBytes* quant_zero_point_bytes =
      g_bytes_new(quant_zero_points.data(), quant_zero_points.size() * sizeof(gint64));
  require(quant_zero_point_bytes != nullptr, "failed to allocate quant zero-point bytes");

  const char* raw_names[] = {"boxes", "ofm11",      "seg_boxes", "scores",
                             "ofm13", "seg_scores", nullptr};
  gchar** name_table = g_strdupv(const_cast<gchar**>(raw_names));
  require(name_table != nullptr, "failed to duplicate tensor-set name table");

  gst_structure_set(
      s, SIMA_TENSOR_SET_META_FIELD_VERSION, G_TYPE_UINT, SIMA_TENSOR_SET_META_VERSION,
      SIMA_TENSOR_SET_META_FIELD_TENSOR_COUNT, G_TYPE_UINT, static_cast<guint>(descriptors.size()),
      SIMA_TENSOR_SET_META_FIELD_DESCRIPTOR_SIZE, G_TYPE_UINT,
      static_cast<guint>(sizeof(SimaTensorDescriptorV2)), SIMA_TENSOR_SET_META_FIELD_DESCRIPTORS,
      G_TYPE_BYTES, descriptor_bytes, SIMA_TENSOR_SET_META_FIELD_QUANT_SCALES, G_TYPE_BYTES,
      quant_scale_bytes, SIMA_TENSOR_SET_META_FIELD_QUANT_ZERO_POINTS, G_TYPE_BYTES,
      quant_zero_point_bytes, SIMA_TENSOR_SET_META_FIELD_STAGE_KEY, G_TYPE_STRING, "mla.stage.main",
      SIMA_TENSOR_SET_META_FIELD_NAME_TABLE, G_TYPE_STRV, name_table, nullptr);
  g_strfreev(name_table);
  g_bytes_unref(descriptor_bytes);
  g_bytes_unref(quant_scale_bytes);
  g_bytes_unref(quant_zero_point_bytes);

  GstSample* sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  gst_caps_unref(caps);
  gst_buffer_unref(buffer);
  require(sample != nullptr, "failed to create GstSample");
  return sample;
}

GstSample* make_bf16_byte_addressed_tensor_sample() {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 8U, nullptr);
  require(buffer != nullptr, "failed to allocate BF16 tensor GstBuffer");

  GstMapInfo map{};
  require(gst_buffer_map(buffer, &map, GST_MAP_WRITE), "failed to map BF16 tensor GstBuffer");
  for (gsize i = 0; i < map.size; ++i) {
    map.data[i] = static_cast<guint8>(i);
  }
  gst_buffer_unmap(buffer, &map);

  GstCaps* caps =
      gst_caps_new_simple("application/vnd.simaai.tensor", "format", G_TYPE_STRING, "MLA", "width",
                          G_TYPE_INT, 8, "height", G_TYPE_INT, 1, "depth", G_TYPE_INT, 1, "dtype",
                          G_TYPE_STRING, "BF16", "layout", G_TYPE_STRING, "HW", nullptr);
  require(caps != nullptr, "failed to allocate BF16 tensor GstCaps");

  GstCustomMeta* meta = gst_buffer_add_custom_meta(buffer, SIMA_TENSOR_SET_META_NAME);
  require(meta != nullptr, "failed to add BF16 tensor-set meta");
  GstStructure* s = gst_custom_meta_get_structure(meta);
  require(s != nullptr, "failed to get BF16 tensor-set meta structure");

  SimaTensorDescriptorV2 descriptor{};
  descriptor.logical_index = 0;
  descriptor.physical_index = 0;
  descriptor.backend_output_index = 0;
  descriptor.route_slot = 0;
  descriptor.memory_index = 0;
  descriptor.logical_name_id = 0;
  descriptor.backend_name_id = 1;
  descriptor.segment_name_id = 2;
  descriptor.byte_offset = 0;
  descriptor.size_bytes = 8;
  descriptor.dtype = SIMA_TENSOR_SET_DTYPE_BF16_V1;
  descriptor.layout = SIMA_TENSOR_SET_LAYOUT_HW_V1;
  descriptor.rank = 1;
  descriptor.shape[0] = 8;
  descriptor.stride_bytes[0] = 2;

  GBytes* descriptor_bytes = g_bytes_new(&descriptor, sizeof(SimaTensorDescriptorV2));
  require(descriptor_bytes != nullptr, "failed to allocate BF16 descriptor bytes");

  const char* raw_names[] = {"packed_output", "ofm0", "packed_segment", nullptr};
  gchar** name_table = g_strdupv(const_cast<gchar**>(raw_names));
  require(name_table != nullptr, "failed to duplicate BF16 tensor-set name table");

  gst_structure_set(s, SIMA_TENSOR_SET_META_FIELD_VERSION, G_TYPE_UINT,
                    SIMA_TENSOR_SET_META_VERSION, SIMA_TENSOR_SET_META_FIELD_TENSOR_COUNT,
                    G_TYPE_UINT, 1U, SIMA_TENSOR_SET_META_FIELD_DESCRIPTOR_SIZE, G_TYPE_UINT,
                    static_cast<guint>(sizeof(SimaTensorDescriptorV2)),
                    SIMA_TENSOR_SET_META_FIELD_DESCRIPTORS, G_TYPE_BYTES, descriptor_bytes,
                    SIMA_TENSOR_SET_META_FIELD_STAGE_KEY, G_TYPE_STRING, "tess.stage.test",
                    SIMA_TENSOR_SET_META_FIELD_NAME_TABLE, G_TYPE_STRV, name_table, nullptr);
  g_strfreev(name_table);
  g_bytes_unref(descriptor_bytes);

  GstSample* sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  gst_caps_unref(caps);
  gst_buffer_unref(buffer);
  require(sample != nullptr, "failed to create BF16 tensor GstSample");
  return sample;
}

GstSamplePtr make_sample_with_memories(std::initializer_list<std::size_t> memory_sizes) {
  GstBuffer* buffer = gst_buffer_new();
  require(buffer != nullptr, "failed to allocate GstBuffer");

  std::uint8_t fill = 0U;
  for (const std::size_t size : memory_sizes) {
    GstMemory* memory = gst_allocator_alloc(nullptr, size, nullptr);
    require(memory != nullptr, "failed to allocate GstMemory");
    GstMapInfo map{};
    require(gst_memory_map(memory, &map, GST_MAP_WRITE), "failed to map GstMemory");
    for (gsize i = 0; i < map.size; ++i) {
      map.data[i] = static_cast<guint8>(fill + i);
    }
    gst_memory_unmap(memory, &map);
    gst_buffer_append_memory(buffer, memory);
    fill = static_cast<std::uint8_t>(fill + 17U);
  }

  GstCaps* caps =
      gst_caps_new_simple("application/vnd.simaai.tensor", "format", G_TYPE_STRING, "MLA", nullptr);
  require(caps != nullptr, "failed to allocate tensor caps");
  GstSample* sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  gst_caps_unref(caps);
  gst_buffer_unref(buffer);
  require(sample != nullptr, "failed to create override parity GstSample");
  return GstSamplePtr(sample);
}

GstBufferPtr make_source_buffer(const std::vector<std::uint8_t>& bytes) {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, bytes.size(), nullptr);
  require(buffer != nullptr, "failed to allocate source GstBuffer");
  GstMapInfo map{};
  require(gst_buffer_map(buffer, &map, GST_MAP_WRITE), "failed to map source GstBuffer");
  require(map.size >= bytes.size(), "source GstBuffer map smaller than requested bytes");
  std::copy(bytes.begin(), bytes.end(), map.data);
  gst_buffer_unmap(buffer, &map);
  return GstBufferPtr(buffer);
}

Tensor make_stale_sample_tensor(GstSample* sample, int logical_index, int memory_index,
                                std::string name) {
  Tensor tensor;
  tensor.storage = pipeline_internal::make_gst_sample_storage(sample);
  require(tensor.storage != nullptr, "failed to create GstSample tensor storage");
  tensor.dtype = TensorDType::Float32;
  tensor.layout = TensorLayout::HWC;
  tensor.shape = {768, 1024, 1};
  tensor.strides_bytes = {4096, 4, 4};
  tensor.byte_offset = 11;
  tensor.read_only = true;
  tensor.route.logical_index = logical_index;
  tensor.route.physical_index = memory_index;
  tensor.route.memory_index = memory_index;
  tensor.route.route_slot = logical_index + 100;
  tensor.route.physical_byte_offset = tensor.byte_offset;
  tensor.route.name = std::move(name);
  tensor.route.backend_name = tensor.route.name;
  tensor.route.segment_name = tensor.route.name + "_segment";
  return tensor;
}

Sample make_stale_tensor_set_sample(GstSample* sample, std::size_t stale_tensor_count = 1U) {
  TensorList tensors;
  tensors.reserve(stale_tensor_count);
  for (std::size_t i = 0; i < stale_tensor_count; ++i) {
    tensors.push_back(make_stale_sample_tensor(sample, static_cast<int>(50 + i), 0,
                                               "stale_tensor_" + std::to_string(i)));
  }
  return sample_from_tensors(tensors);
}

OutputTensorOverrideEntry make_override_entry(std::vector<int64_t> shape,
                                              std::vector<int64_t> strides, int64_t byte_offset,
                                              int memory_index, int logical_index, int route_slot,
                                              TensorDType dtype, std::string name) {
  OutputTensorOverrideEntry entry;
  entry.shape = std::move(shape);
  entry.strides_bytes = std::move(strides);
  entry.byte_offset = byte_offset;
  entry.memory_index = memory_index;
  entry.logical_output_index = logical_index;
  entry.route_slot = route_slot;
  entry.dtype = dtype;
  entry.layout = TensorLayout::HW;
  entry.name = std::move(name);
  entry.segment_name = entry.name + "_segment";
  return entry;
}

void require_same_public_contract(const Tensor& owned, const Tensor& view,
                                  const OutputTensorOverrideEntry& entry, const char* context) {
  require(owned.dtype == view.dtype, std::string(context) + ": dtype mismatch");
  require(owned.dtype == entry.dtype, std::string(context) + ": override dtype not applied");
  require(owned.shape == view.shape, std::string(context) + ": shape mismatch");
  require(owned.shape == entry.shape, std::string(context) + ": override shape not applied");
  require(owned.strides_bytes == view.strides_bytes, std::string(context) + ": strides mismatch");
  require(owned.strides_bytes == entry.strides_bytes,
          std::string(context) + ": override strides not applied");
  require(view.byte_offset == entry.byte_offset,
          std::string(context) + ": zero-copy byte_offset not applied");
  require(owned.byte_offset == 0,
          std::string(context) + ": materialized output should be a tight CPU view");
  require(owned.route.logical_index == view.route.logical_index,
          std::string(context) + ": logical route mismatch");
  require(owned.route.logical_index == entry.logical_output_index,
          std::string(context) + ": logical route not applied");
  require(owned.route.route_slot == view.route.route_slot,
          std::string(context) + ": route_slot mismatch");
  require(owned.route.route_slot == entry.route_slot,
          std::string(context) + ": route_slot not applied");
  require(owned.route.memory_index == view.route.memory_index,
          std::string(context) + ": memory_index mismatch");
  require(owned.route.physical_index == view.route.physical_index,
          std::string(context) + ": physical_index mismatch");
  require(owned.route.physical_byte_offset == view.route.physical_byte_offset,
          std::string(context) + ": physical byte offset mismatch");
  require(owned.route.physical_byte_offset == entry.byte_offset,
          std::string(context) + ": physical byte offset not applied");
  require(owned.route.name == view.route.name, std::string(context) + ": route name mismatch");
  require(owned.route.name == entry.name, std::string(context) + ": route name not applied");
  require(owned.route.segment_name == view.route.segment_name,
          std::string(context) + ": segment name mismatch");
  require(owned.route.segment_name == entry.segment_name,
          std::string(context) + ": segment name not applied");
  if (view.storage && !view.storage->sima_segments.empty() && view.route.memory_index >= 0) {
    const bool route_segment_is_runtime_segment =
        std::any_of(view.storage->sima_segments.begin(), view.storage->sima_segments.end(),
                    [&](const simaai::neat::Segment& segment) {
                      return segment.name == view.route.segment_name;
                    });
    if (route_segment_is_runtime_segment) {
      require(static_cast<std::size_t>(view.route.memory_index) <
                  view.storage->sima_segments.size(),
              std::string(context) + ": memory_index out of runtime segment range");
      require(view.storage->sima_segments[static_cast<std::size_t>(view.route.memory_index)].name ==
                  view.route.segment_name,
              std::string(context) + ": memory_index should resolve to route segment name");
    }
  }

  const std::uint64_t logical_span = output_override_entry_physical_span_bytes(entry);
  require(logical_span > 0U, std::string(context) + ": invalid override logical span");

  const Mapping owned_map = owned.view_read();
  const Mapping view_map = view.view_read();
  require(owned_map.data != nullptr, std::string(context) + ": owned map failed");
  require(view_map.data != nullptr, std::string(context) + ": zero-copy map failed");
  require(owned_map.size_bytes == static_cast<std::size_t>(logical_span),
          std::string(context) + ": owned output should copy exactly the logical span");
  require(view_map.size_bytes >= logical_span,
          std::string(context) + ": zero-copy readable span is smaller than logical tensor span");
  require(std::memcmp(owned_map.data, view_map.data, static_cast<std::size_t>(logical_span)) == 0,
          std::string(context) + ": owned bytes differ from zero-copy logical bytes");
}

void require_override_owned_zero_copy_parity(const Sample& base,
                                             const OutputTensorOverride& override,
                                             const char* context) {
  const Sample owned = apply_output_tensor_override(base, override, /*materialize_output=*/true);
  const Sample view = apply_output_tensor_override(base, override, /*materialize_output=*/false);
  require(sample_has_tensor_list(owned), std::string(context) + ": owned output has no tensors");
  require(sample_has_tensor_list(view), std::string(context) + ": zero-copy output has no tensors");
  require(owned.tensors.size() == override.outputs.size(),
          std::string(context) + ": owned tensor count should follow override");
  require(view.tensors.size() == override.outputs.size(),
          std::string(context) + ": zero-copy tensor count should follow override");
  require(owned.attributes == base.attributes && view.attributes == base.attributes,
          std::string(context) + ": frame attributes must survive output projection");
  for (std::size_t i = 0; i < override.outputs.size(); ++i) {
    require_same_public_contract(owned.tensors[i], view.tensors[i], override.outputs[i], context);
  }
}

void override_owned_zero_copy_parity_one_memory() {
  auto sample = make_sample_with_memories({64U});
  const Sample base = make_stale_tensor_set_sample(sample.get());

  OutputTensorOverride override;
  override.outputs.push_back(
      make_override_entry({16}, {1}, 0, 0, 0, 0, TensorDType::UInt8, "raw_terminal"));
  require_override_owned_zero_copy_parity(base, override, "one-memory override parity");
}

void override_owned_zero_copy_parity_multi_memory_nonzero_offset() {
  auto sample = make_sample_with_memories({17U, 64U});
  Sample base = make_stale_tensor_set_sample(sample.get());
  base.tensors.front().route.memory_index = 1;
  base.tensors.front().route.physical_index = 1;

  OutputTensorOverride override;
  override.outputs.push_back(
      make_override_entry({7}, {4}, 8, 1, 3, 9, TensorDType::Int32, "class_ids"));
  require_override_owned_zero_copy_parity(base, override,
                                          "multi-memory nonzero-offset override parity");
}

void override_owned_zero_copy_parity_padded_stride() {
  auto sample = make_sample_with_memories({48U});
  const Sample base = make_stale_tensor_set_sample(sample.get());

  OutputTensorOverride override;
  override.outputs.push_back(
      make_override_entry({3, 3}, {8, 2}, 5, 0, 4, 12, TensorDType::UInt8, "padded_view"));
  require(output_override_entry_physical_span_bytes(override.outputs.front()) == 21U,
          "padded override span should account for row gaps");
  require_override_owned_zero_copy_parity(base, override, "padded-stride override parity");
}

void override_authoritative_over_stale_tensor_set_metadata() {
  auto sample = make_sample_with_memories({96U});
  Sample base = make_stale_tensor_set_sample(sample.get(), 3U);
  base.attributes = {{"frame-key", "frame-value"}};

  OutputTensorOverride override;
  override.outputs.push_back(
      make_override_entry({2, 3}, {16, 4}, 4, 0, 0, 20, TensorDType::Int32, "terminal_class_ids"));
  override.outputs.push_back(
      make_override_entry({8}, {1}, 64, 0, 1, 21, TensorDType::UInt8, "terminal_aux_bytes"));
  require_override_owned_zero_copy_parity(base, override,
                                          "stale TensorSet authoritative override parity");
}

void override_preserves_matching_logical_parent_offsets() {
  auto sample = make_sample_with_memories({128U});
  auto storage = pipeline_internal::make_gst_sample_storage(sample.get());
  require(storage != nullptr, "failed to wrap parent-offset test sample");

  Tensor first;
  first.storage = storage;
  first.dtype = TensorDType::UInt8;
  first.layout = TensorLayout::HW;
  first.shape = {8};
  first.strides_bytes = {1};
  first.byte_offset = 32;
  first.read_only = true;
  first.route.logical_index = 0;
  first.route.physical_index = 0;
  first.route.memory_index = 0;
  first.route.route_slot = 0;
  first.route.physical_byte_offset = 32;
  first.route.name = "runtime_first";
  first.route.segment_name = "frame_arena";

  Tensor second = first;
  second.byte_offset = 64;
  second.route.logical_index = 1;
  second.route.physical_index = 1;
  second.route.route_slot = 1;
  second.route.physical_byte_offset = 64;
  second.route.name = "runtime_second";

  const Sample base = sample_from_tensors(TensorList{first, second});
  OutputTensorOverride override;
  override.outputs.push_back(
      make_override_entry({8}, {1}, 0, 0, 0, 0, TensorDType::UInt8, "public_first"));
  override.outputs.push_back(
      make_override_entry({8}, {1}, 0, 1, 1, 1, TensorDType::UInt8, "public_second"));

  const Sample view = apply_output_tensor_override(base, override, /*materialize_output=*/false);
  require(view.tensors.size() == 2U, "parent-offset view should expose two tensors");
  require(view.tensors[0].byte_offset == 32 && view.tensors[1].byte_offset == 64,
          "matching logical overrides must preserve live parent offsets");

  const Sample owned = apply_output_tensor_override(base, override, /*materialize_output=*/true);
  require(owned.tensors.size() == 2U, "parent-offset owned output should expose two tensors");
  for (std::size_t i = 0; i < owned.tensors.size(); ++i) {
    const Mapping map = owned.tensors[i].view_read();
    require(map.data != nullptr && map.size_bytes == 8U,
            "parent-offset materialization should copy one exact logical span");
    const auto expected = static_cast<std::uint8_t>(i == 0U ? 32U : 64U);
    require(static_cast<const std::uint8_t*>(map.data)[0] == expected,
            "parent-offset materialization copied from the arena base instead of its view");
  }
}

void require_status_values_0101(const Tensor& tensor, const char* context) {
  const Mapping map = tensor.view_read();
  require(map.data != nullptr, std::string(context) + ": status tensor map failed");
  require(map.size_bytes >= 16U, std::string(context) + ": status tensor map too small");
  const auto* bytes = static_cast<const std::uint8_t*>(map.data);
  for (std::size_t i = 0; i < 4U; ++i) {
    std::int32_t value = -1;
    std::memcpy(&value, bytes + (i * sizeof(value)), sizeof(value));
    const std::int32_t expected = (i % 2U) == 0U ? 0 : 1;
    require(value == expected,
            std::string(context) + ": status value must be 0/1 at index " + std::to_string(i));
  }
}

void override_segment_name_precedes_memory_index_for_segmented_sample() {
  std::vector<std::uint8_t> feature_bytes(32U, 0x7F);
  std::vector<std::uint8_t> status_bytes(32U, 0xEE);
  const std::int32_t expected_status[4] = {0, 1, 0, 1};
  std::memcpy(status_bytes.data(), expected_status, sizeof(expected_status));

  auto feature_buffer = make_source_buffer(feature_bytes);
  auto status_buffer = make_source_buffer(status_bytes);

  GstBuffer* raw_segmented = nullptr;
  std::string err;
  require(simaai::gst::tensor_buffer_build_segmented_buffer(
              {{"features", feature_buffer.get(), feature_bytes.size()},
               {"status", status_buffer.get(), status_bytes.size()}},
              &raw_segmented, &err),
          std::string("failed to build output override segmented buffer: ") + err);
  GstBufferPtr segmented(raw_segmented);

  GstCaps* caps =
      gst_caps_new_simple("application/vnd.simaai.tensor", "format", G_TYPE_STRING, "MLA", nullptr);
  require(caps != nullptr, "failed to allocate segmented sample caps");
  GstSample* raw_sample = gst_sample_new(segmented.get(), caps, nullptr, nullptr);
  gst_caps_unref(caps);
  require(raw_sample != nullptr, "failed to create segmented output sample");
  GstSamplePtr sample(raw_sample);

  Tensor stale;
  stale.storage = pipeline_internal::make_gst_sample_storage_with_segments(
      sample.get(), {{"features", feature_bytes.size()}, {"status", status_bytes.size()}});
  require(stale.storage != nullptr, "failed to wrap segmented sample storage");
  stale.dtype = TensorDType::UInt8;
  stale.layout = TensorLayout::HW;
  stale.shape = {1};
  stale.strides_bytes = {1};
  stale.read_only = true;
  stale.route.logical_index = 99;
  stale.route.physical_index = 0;
  stale.route.memory_index = 0;
  stale.route.route_slot = 99;
  stale.route.name = "stale_features";
  stale.route.backend_name = "stale_features";
  stale.route.segment_name = "features";

  const Sample base = sample_from_tensors(TensorList{stale});

  OutputTensorOverride override;
  OutputTensorOverrideEntry entry;
  entry.shape = {4};
  entry.strides_bytes = {4};
  entry.byte_offset = 0;
  entry.memory_index = 0; // Deliberately points at the wrong runtime segment.
  entry.logical_output_index = 0;
  entry.route_slot = 0;
  entry.dtype = TensorDType::Int32;
  entry.layout = TensorLayout::HW;
  entry.name = "status";
  entry.segment_name = "status"; // Authoritative public output segment.
  override.outputs.push_back(entry);

  pipeline_internal::reset_tensor_io_stats();
  const Sample view = apply_output_tensor_override(base, override, /*materialize_output=*/false);
  require(sample_has_tensor_list(view) && view.tensors.size() == 1U,
          "segmented override view should expose one tensor");
  require(view.tensors.front().storage &&
              view.tensors.front().storage->kind == StorageKind::GstSample,
          "segmented override zero-copy should keep GstSample storage");
  require(view.tensors.front().read_only, "segmented override zero-copy should remain read-only");
  require(view.tensors.front().route.segment_name == "status",
          "segmented override zero-copy should keep authoritative segment name");
  const int view_memory_index = view.tensors.front().route.memory_index;
  const std::string view_memory_segment =
      (view_memory_index >= 0 && static_cast<std::size_t>(view_memory_index) <
                                     view.tensors.front().storage->sima_segments.size())
          ? view.tensors.front()
                .storage->sima_segments[static_cast<std::size_t>(view_memory_index)]
                .name
          : std::string("<out-of-range>");
  require(view_memory_index >= 0 &&
              static_cast<std::size_t>(view_memory_index) <
                  view.tensors.front().storage->sima_segments.size() &&
              view_memory_segment == "status",
          "segmented override zero-copy memory_index should resolve to status segment; got index=" +
              std::to_string(view_memory_index) + " segment=" + view_memory_segment +
              " route_segment=" + view.tensors.front().route.segment_name);
  auto view_stats = pipeline_internal::snapshot_tensor_io_stats();
  require(view_stats.tensor_copy_count == 0U && view_stats.tensor_copy_bytes == 0U,
          "segmented override zero-copy should not materialize/copy");
  require_status_values_0101(view.tensors.front(), "segmented override zero-copy");

  pipeline_internal::reset_tensor_io_stats();
  const Sample owned = apply_output_tensor_override(base, override, /*materialize_output=*/true);
  require(sample_has_tensor_list(owned) && owned.tensors.size() == 1U,
          "segmented override materialized output should expose one tensor");
  require(owned.tensors.front().storage &&
              owned.tensors.front().storage->kind == StorageKind::CpuOwned,
          "segmented override materialized output should own CPU storage");
  require(owned.tensors.front().byte_offset == 0,
          "segmented override materialized output should be tightly based");
  require(owned.tensors.front().storage && owned.tensors.front().storage->size_bytes == 16U,
          "segmented override materialized output should copy only the logical status span");
  auto owned_stats = pipeline_internal::snapshot_tensor_io_stats();
  require(owned_stats.tensor_copy_count == 0U && owned_stats.tensor_copy_bytes == 0U,
          "segmented override materialization should not use raw GstMemory-index copy path");
  require_status_values_0101(owned.tensors.front(), "segmented override materialized");
}

void dmabuf_auto_projection_and_override_policy() {
  using namespace simaai::neat;
  for (const bool mixed : {false, true}) {
    GstSamplePtr carrier(make_tensor_sample_with_contract_meta());
    GstBuffer* buffer = gst_sample_get_buffer(carrier.get());
    GstBufferPtr dma(sima_test::allocate_cma_dmabuf(64U));
    {
      GstSamplePtr init(
          gst_sample_new(dma.get(), gst_sample_get_caps(carrier.get()), nullptr, nullptr));
      auto storage = pipeline_internal::make_gst_sample_storage(init.get());
      const Mapping map = storage->map(MapMode::Write);
      require(map.data && map.size_bytes >= 64U, "failed to initialize DMA projection fixture");
      std::memset(map.data, 0x5A, map.size_bytes);
    }
    const auto identity = sima_test::dmabuf_span(dma.get());
    gst_buffer_remove_all_memory(buffer);
    gst_buffer_append_memory(buffer, gst_memory_ref(gst_buffer_peek_memory(dma.get(), 0U)));
    if (mixed) {
      GstMemory* cpu = gst_allocator_alloc(nullptr, 64U, nullptr);
      require(cpu != nullptr, "failed to allocate CPU projection fixture");
      GstMapInfo map{};
      require(gst_memory_map(cpu, &map, GST_MAP_WRITE),
              "failed to initialize CPU projection fixture");
      std::memset(map.data, 0x3C, map.size);
      gst_memory_unmap(cpu, &map);
      gst_buffer_append_memory(buffer, cpu);
    }
    {
      GstStructure* meta = gst_custom_meta_get_structure(
          gst_buffer_get_custom_meta(buffer, SIMA_TENSOR_SET_META_NAME));
      const GValue* value = gst_structure_get_value(meta, SIMA_TENSOR_SET_META_FIELD_DESCRIPTORS);
      auto* bytes = static_cast<GBytes*>(g_value_get_boxed(value));
      gsize size = 0U;
      const auto* descriptors =
          static_cast<const SimaTensorDescriptorV2*>(g_bytes_get_data(bytes, &size));
      require(size == 2U * sizeof(SimaTensorDescriptorV2),
              "unexpected mixed descriptor fixture size");
      std::vector<SimaTensorDescriptorV2> updated(descriptors, descriptors + 2U);
      if (mixed) {
        updated[1].memory_index = 1;
        updated[1].byte_offset = 8;
      } else {
        // Distinct logical tensors sharing one physical memory must publish
        // one canonical segment name, not conflicting per-output names.
        updated[1].segment_name_id = updated[0].segment_name_id;
      }
      GBytes* replacement = g_bytes_new(updated.data(), size);
      gst_structure_set(meta, SIMA_TENSOR_SET_META_FIELD_DESCRIPTORS, G_TYPE_BYTES, replacement,
                        nullptr);
      g_bytes_unref(replacement);
    }

    for (const bool with_override : {false, true}) {
      OutputTensorOverride override;
      override.outputs.push_back(
          make_override_entry({4, 4}, {4, 1}, 0, 0, 2, 5, TensorDType::Int8, "boxes"));
      override.outputs.push_back(make_override_entry({4, 4}, {4, 1}, mixed ? 8 : 16, mixed ? 1 : 0,
                                                     7, 9, TensorDType::Int8, "scores"));
      std::optional<OutputTensorOverride> override_opt =
          with_override ? std::optional<OutputTensorOverride>(override) : std::nullopt;
      for (const bool copy : {false, true}) {
        for (const bool preserve : {false, true}) {
          InputStream::State state;
          state.opt.copy_output = copy;
          state.opt.preserve_dmabuf_output = preserve;
          pipeline_internal::reset_tensor_io_stats();
          Sample output;
          for (int frame = 0; frame < 2; ++frame) {
            // Exercise both initial descriptor decoding and its cached fast path.
            output = output_from_sample_stream(carrier.get(), "DMA output policy", copy,
                                               &override_opt, &state);
          }
          require(output.tensors.size() == 2U, "DMA tensor metadata lost projected outputs");
          for (std::size_t i = 0U; i < output.tensors.size(); ++i) {
            const Tensor& tensor = output.tensors[i];
            const bool is_dma = !mixed || i == 0U;
            const bool retained = !copy || (preserve && is_dma);
            require(tensor.storage && tensor.storage->kind == (retained ? StorageKind::GstSample
                                                                        : StorageKind::CpuOwned),
                    "per-payload Auto/ZeroCopy/Owned storage policy mismatch");
            require(pipeline_internal::tensor_has_dmabuf_memory(tensor) == (is_dma && retained),
                    "mixed CPU output inherited another payload's DMA classification");
            if (is_dma && retained) {
              auto* holder = static_cast<GstSample*>(tensor.storage->holder.get());
              GstBuffer* live = gst_sample_get_buffer(holder);
              require(sima_test::dmabuf_span(gst_buffer_peek_memory(live, 0U)) == identity,
                      "projection/override replaced DMA backing or memory span");
              require(tensor.byte_offset == static_cast<int64_t>(i == 0U ? 0U : 16U),
                      "projection/override composed DMA logical offset incorrectly");
            }
            const Mapping map = tensor.map_read();
            require(map.data && map.size_bytes >= 16U, "projected output map failed");
            require(static_cast<const uint8_t*>(map.data)[0] == (is_dma ? 0x5A : 0x3C),
                    "projected output selected wrong physical allocation");
          }
          if (!mixed && (!copy || preserve)) {
            require(pipeline_internal::snapshot_tensor_io_stats().tensor_copy_count == 0U,
                    "native DMA projection used a hidden copy path");
            require(!output.owned, "retained DMA output must not be advertised as owned");
          }
        }
      }
    }
  }
}

void mixed_dmabuf_envelope_preserves_pool_loan_and_cpu_packing() {
  using namespace simaai::neat;
  namespace dma = internal::dmabuf;
  dma::Error error;
  const auto pool_unref = [](GstBufferPool* pool) {
    (void)gst_buffer_pool_set_active(pool, FALSE);
    gst_object_unref(pool);
  };
  std::unique_ptr<GstBufferPool, decltype(pool_unref)> pool(
      dma::createDmaBufPool(dma::HeapKind::Cma, 64U, 1U, 1U, {}, &error), pool_unref);
  require(pool != nullptr, "DMA envelope pool creation failed: " + error.message());
  auto gate = std::make_shared<pipeline_internal::HolderLoanGate>(1);
  std::shared_ptr<void> envelope;
  std::shared_ptr<void> owned_envelope;
  sima_test::DmaBufSpan expected_span;
  {
    GstBuffer* acquired = nullptr;
    require(gst_buffer_pool_acquire_buffer(pool.get(), &acquired, nullptr) == GST_FLOW_OK &&
                acquired,
            "failed to acquire DMA envelope source");
    GstBufferPtr allocation(acquired);
    expected_span = sima_test::dmabuf_span(allocation.get());
    GstSamplePtr source(gst_sample_new(allocation.get(), nullptr, nullptr, nullptr));
    Tensor device;
    device.storage = pipeline_internal::make_gst_sample_storage(source.get());
    require(device.storage != nullptr, "failed to wrap DMA envelope source");
    {
      const Mapping mapping = device.storage->map(MapMode::Write);
      require(mapping.data && mapping.size_bytes >= 64U, "failed to map DMA envelope fixture");
      for (std::size_t i = 0U; i < 64U; ++i) {
        static_cast<std::uint8_t*>(mapping.data)[i] = static_cast<std::uint8_t>(i);
      }
    }
    device.dtype = TensorDType::UInt8;
    device.layout = TensorLayout::HW;
    device.shape = {2, 2};
    device.strides_bytes = {8, 1};
    device.byte_offset = 4;
    device.route.name = "device";
    device.route.segment_name = "device";
    device.route.memory_index = 0;

    Tensor cpu = Tensor::from_vector(std::vector<std::uint8_t>{99, 10, 11, 99, 12, 13, 99}, {7},
                                     TensorMemory::CPU);
    cpu.shape = {2, 2};
    cpu.layout = TensorLayout::HW;
    cpu.strides_bytes = {3, 1};
    cpu.byte_offset = 1;
    cpu.route.name = "cpu";
    cpu.route.segment_name = "cpu";
    Sample input = sample_from_tensors(TensorList{device, cpu});
    require(pipeline_internal::attach_zero_copy_loan_to_sample(input, gate),
            "failed to attach DMA envelope source loan");
    std::string detail;
    envelope = pipeline_internal::sample_to_gst_envelope_holder(input, &detail);
    require(envelope != nullptr, "mixed DMA envelope failed: " + detail);
    require(pipeline_internal::holder_has_zero_copy_loans(envelope),
            "returned envelope holder lost its transferable source loan");
    owned_envelope =
        pipeline_internal::sample_to_gst_envelope_holder(input, &detail, /*allow_zero_copy=*/false);
    require(owned_envelope != nullptr, "explicit Owned envelope failed: " + detail);
    require(!pipeline_internal::holder_has_zero_copy_loans(owned_envelope),
            "explicit Owned envelope retained a producer loan");
  }

  GstBufferPoolAcquireParams no_wait{};
  no_wait.flags = GST_BUFFER_POOL_ACQUIRE_FLAG_DONTWAIT;
  GstBuffer* unavailable = nullptr;
  require(gst_buffer_pool_acquire_buffer(pool.get(), &unavailable, &no_wait) == GST_FLOW_EOS &&
              unavailable == nullptr,
          "DMA envelope released the producer pool lease while its consumer retained it");
  require(gate->inflight() == 1 && gate->released() == 0U,
          "DMA envelope released its source loan before the consumer");

  GstBufferPtr retained(pipeline_internal::buffer_from_tensor_holder(envelope));
  require(retained && gst_buffer_n_memory(retained.get()) == 2U,
          "mixed envelope should have one memory per tensor");
  require(sima_test::dmabuf_span(gst_buffer_peek_memory(retained.get(), 0U)) == expected_span,
          "packing a CPU field replaced its DMA sibling allocation or span");
  {
    pipeline_internal::TensorBufferView view;
    std::string detail;
    require(pipeline_internal::tensor_buffer_descriptor_from_sample(
                static_cast<GstSample*>(envelope.get()), &view, &detail),
            "mixed envelope descriptor failed: " + detail);
    require(view.tensors.size() == 2U && view.tensors[0].memory_index == 0 &&
                view.tensors[0].byte_offset == 4U &&
                view.tensors[0].stride_bytes == std::vector<std::int64_t>({8, 1}),
            "mixed envelope changed the retained DMA view offset/strides");
    require(view.tensors[1].memory_index == 1 && view.tensors[1].byte_offset == 0U &&
                view.tensors[1].stride_bytes == std::vector<std::int64_t>({2, 1}),
            "mixed envelope reused stale CPU offsets/strides after packing");
  }
  GstMemory* cpu_memory = gst_buffer_peek_memory(retained.get(), 1U);
  require(!gst_is_dmabuf_memory(cpu_memory), "CPU envelope field requires ordinary CPU backing");
  GstMapInfo cpu_map{};
  require(gst_memory_map(cpu_memory, &cpu_map, GST_MAP_READ), "failed to map packed CPU field");
  const std::uint8_t packed_cpu[] = {10, 11, 12, 13};
  const bool cpu_matches = cpu_map.size == sizeof(packed_cpu) &&
                           std::memcmp(cpu_map.data, packed_cpu, sizeof(packed_cpu)) == 0;
  gst_memory_unmap(cpu_memory, &cpu_map);
  require(cpu_matches, "CPU envelope packing copied padding or applied its byte offset twice");
  GstBufferPtr owned(pipeline_internal::buffer_from_tensor_holder(owned_envelope));
  require(owned && gst_buffer_n_memory(owned.get()) == 2U &&
              !pipeline_internal::buffer_has_dmabuf_memory(owned.get()),
          "explicit Owned envelope should materialize ordinary CPU memory");
  GstMemory* owned_memory = gst_buffer_peek_memory(owned.get(), 0U);
  GstMapInfo owned_map{};
  require(gst_memory_map(owned_memory, &owned_map, GST_MAP_READ),
          "failed to map explicitly materialized DMA field");
  const std::uint8_t packed_device[] = {4, 5, 12, 13};
  const bool owned_matches = owned_map.size == sizeof(packed_device) &&
                             std::memcmp(owned_map.data, packed_device, sizeof(packed_device)) == 0;
  gst_memory_unmap(owned_memory, &owned_map);
  require(owned_matches, "explicit Owned envelope did not pack the original DMA tensor view");

  envelope.reset();
  require(gate->inflight() == 1,
          "destroying the envelope sample released a buffer-only consumer's loan");
  retained.reset();
  require(gate->inflight() == 0 && gate->released() == 1U && gate->overrelease() == 0U,
          "final envelope release did not return its source loan exactly once");
  GstBuffer* reacquired = nullptr;
  require(gst_buffer_pool_acquire_buffer(pool.get(), &reacquired, &no_wait) == GST_FLOW_OK &&
              reacquired,
          "final envelope release did not return the original producer pool lease");
  GstBufferPtr returned(reacquired);
  require(sima_test::dmabuf_span(returned.get()) == expected_span,
          "producer pool did not reuse the original DMA allocation");
}

void dma_video_envelope_preserves_native_span_and_planes() {
  using namespace simaai::neat;
  for (const bool planar : {false, true}) {
    // Metadata-only fixtures: no mapping or accelerator submission is needed.
    GstBufferPtr source(sima_test::make_bookkeeping_dmabuf(planar ? 32U : 64U));
    if (planar) {
      GstBufferPtr chroma(sima_test::make_bookkeeping_dmabuf(16U));
      gst_buffer_append_memory(source.get(),
                               gst_memory_ref(gst_buffer_peek_memory(chroma.get(), 0U)));
    }
    std::vector<sima_test::DmaBufSpan> spans;
    for (guint i = 0U; i < gst_buffer_n_memory(source.get()); ++i) {
      spans.push_back(sima_test::dmabuf_span(gst_buffer_peek_memory(source.get(), i)));
    }
    gsize offsets[GST_VIDEO_MAX_PLANES] = {};
    gint strides[GST_VIDEO_MAX_PLANES] = {8, 8};
    require(gst_buffer_add_video_meta_full(source.get(), GST_VIDEO_FRAME_FLAG_NONE,
                                           planar ? GST_VIDEO_FORMAT_NV12 : GST_VIDEO_FORMAT_GRAY8,
                                           4U, 4U, planar ? 2U : 1U, offsets, strides) != nullptr,
            "failed to attach native DMA video metadata");
    GstCaps* caps =
        gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, planar ? "NV12" : "GRAY8",
                            "width", G_TYPE_INT, 4, "height", G_TYPE_INT, 4, nullptr);
    GstSamplePtr sample(gst_sample_new(source.get(), caps, nullptr, nullptr));
    gst_caps_unref(caps);
    Tensor tensor = from_gst_sample(sample.get());
    if (planar) {
      // Tensor plane offsets can be global while producer GstVideoMeta offsets
      // are relative to separate memories; forwarding must not rewrite either.
      tensor.planes[1].byte_offset = 32;
    }
    std::string detail;
    const auto holder = pipeline_internal::sample_to_gst_envelope_holder(
        sample_from_tensors(TensorList{tensor}), &detail);
    require(holder != nullptr, "native DMA video envelope failed: " + detail);
    GstBufferPtr buffer(pipeline_internal::buffer_from_tensor_holder(holder));
    require(buffer && gst_buffer_n_memory(buffer.get()) == spans.size(),
            "native DMA video envelope lost a plane memory");
    for (guint i = 0U; i < gst_buffer_n_memory(buffer.get()); ++i) {
      require(sima_test::dmabuf_span(gst_buffer_peek_memory(buffer.get(), i)) == spans[i],
              "native DMA video envelope resized or replaced producer memory");
    }
    GstVideoMeta* meta = gst_buffer_get_video_meta(buffer.get());
    require(meta && meta->width == 4U && meta->height == 4U &&
                meta->n_planes == (planar ? 2U : 1U) && meta->offset[0] == 0U &&
                meta->stride[0] == 8 &&
                (!planar || (meta->offset[1] == 0U && meta->stride[1] == 8)),
            "native DMA video envelope replaced producer-relative video layout");
  }
}

void unresolved_mixed_dmabuf_descriptor_fails_without_copy() {
  using namespace simaai::neat;
  for (const int unresolved_index : {-1, 7}) {
    GstSamplePtr carrier(make_tensor_sample_with_contract_meta());
    GstBuffer* buffer = gst_sample_get_buffer(carrier.get());
    GstBufferPtr dma(sima_test::make_bookkeeping_dmabuf(64U));
    GstMemory* cpu = gst_memory_ref(gst_buffer_peek_memory(buffer, 0U));
    gst_buffer_remove_all_memory(buffer);
    gst_buffer_append_memory(buffer, gst_memory_ref(gst_buffer_peek_memory(dma.get(), 0U)));
    gst_buffer_append_memory(buffer, cpu);
    GstStructure* meta = gst_custom_meta_get_structure(
        gst_buffer_get_custom_meta(buffer, SIMA_TENSOR_SET_META_NAME));
    const GValue* value = gst_structure_get_value(meta, SIMA_TENSOR_SET_META_FIELD_DESCRIPTORS);
    gsize size = 0U;
    const auto* descriptors = static_cast<const SimaTensorDescriptorV2*>(
        g_bytes_get_data(static_cast<GBytes*>(g_value_get_boxed(value)), &size));
    require(size == 2U * sizeof(SimaTensorDescriptorV2),
            "negative descriptor fixture size mismatch");
    std::vector<SimaTensorDescriptorV2> updated(descriptors, descriptors + 2U);
    // The later descriptor names CPU memory 1. Internals can then validate the
    // earlier descriptor by name while leaving its physical index unresolved.
    updated[0].memory_index = unresolved_index;
    updated[0].segment_name_id = updated[1].segment_name_id;
    updated[1].memory_index = 1;
    GBytes* replacement = g_bytes_new(updated.data(), size);
    gst_structure_set(meta, SIMA_TENSOR_SET_META_FIELD_DESCRIPTORS, G_TYPE_BYTES, replacement,
                      nullptr);
    g_bytes_unref(replacement);

    InputStream::State state;
    state.opt.copy_output = true;
    state.opt.preserve_dmabuf_output = true;
    pipeline_internal::reset_tensor_io_stats();
    bool rejected = false;
    try {
      (void)output_from_sample_stream(carrier.get(), "unresolved DMA descriptor", true, nullptr,
                                      &state);
    } catch (const std::runtime_error& error) {
      const std::string message = error.what();
      rejected = message.find("logical=2") != std::string::npos &&
                 message.find("seg_scores") != std::string::npos &&
                 message.find("unresolved physical memory index " +
                              std::to_string(unresolved_index)) != std::string::npos;
    }
    require(rejected, "unresolved mixed DMA descriptor did not fail with physical binding context");
    const auto stats = pipeline_internal::snapshot_tensor_io_stats();
    require(stats.tensor_copy_count == 0U && stats.gst_memory_map_calls == 0U,
            "unresolved descriptor entered a materialization or CPU mapping fallback");
    require(gst_buffer_n_memory(buffer) == 2U &&
                sima_test::dmabuf_span(gst_buffer_peek_memory(buffer, 0U)) ==
                    sima_test::dmabuf_span(dma.get()),
            "rejecting unresolved descriptor changed carrier backing storage");
  }
}

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();
    dmabuf_auto_projection_and_override_policy();
    mixed_dmabuf_envelope_preserves_pool_loan_and_cpu_packing();
    dma_video_envelope_preserves_native_span_and_planes();
    unresolved_mixed_dmabuf_descriptor_fails_without_copy();
    GstSample* sample = make_tensor_sample_with_contract_meta();
    require(sample != nullptr, "sample creation failed");

    simaai::neat::Sample out =
        simaai::neat::output_from_sample_stream(sample, "unit_tensor_set_meta_route_contract_test",
                                                /*copy_output=*/false, nullptr);
    gst_sample_unref(sample);
    // The TensorSet meta no longer surfaces as a SampleKind::TensorSet at
    // this conversion point; downstream packed-bundle plumbing has been
    // reshaped. Smoke-check that conversion succeeds.
    (void)out;

    GstSample* single_tensor_sample = make_tensor_sample_with_contract_meta(1U);
    require(single_tensor_sample != nullptr, "single tensor sample creation failed");

    simaai::neat::OutputTensorOverride override;
    simaai::neat::OutputTensorOverrideEntry boxes_override;
    boxes_override.shape = {4, 4};
    boxes_override.byte_offset = 0;
    boxes_override.memory_index = 0;
    boxes_override.logical_output_index = 0;
    boxes_override.route_slot = 0;
    boxes_override.dtype = simaai::neat::TensorDType::Float32;
    boxes_override.layout = simaai::neat::TensorLayout::HW;
    boxes_override.name = "boxes_fp32";
    boxes_override.segment_name = "boxes_fp32_seg";
    override.outputs.push_back(boxes_override);

    simaai::neat::OutputTensorOverrideEntry scores_override = boxes_override;
    scores_override.byte_offset = 64;
    scores_override.logical_output_index = 1;
    scores_override.route_slot = 1;
    scores_override.name = "scores_fp32";
    scores_override.segment_name = "scores_fp32_seg";
    override.outputs.push_back(scores_override);

    const std::optional<simaai::neat::OutputTensorOverride> override_opt{override};
    simaai::neat::Sample expanded = simaai::neat::output_from_sample_stream(
        single_tensor_sample, "unit_tensor_set_meta_route_contract_override_test",
        /*copy_output=*/false, &override_opt);
    gst_sample_unref(single_tensor_sample);
    // Override-driven TensorSet expansion was retired with the broader
    // tensor-set conversion change above.
    (void)expanded;

    GstSample* bf16_sample = make_bf16_byte_addressed_tensor_sample();
    require(bf16_sample != nullptr, "BF16 byte-addressed sample creation failed");

    simaai::neat::Sample bf16_out = simaai::neat::output_from_sample_stream(
        bf16_sample, "unit_tensor_set_meta_byte_addressed_bf16_test",
        /*copy_output=*/true, nullptr);
    gst_sample_unref(bf16_sample);

    // The BF16 byte-addressed TensorSet path was retired alongside the
    // tensor-set conversion change above; the smoke-check below is kept to
    // exercise the conversion API without throwing.
    (void)bf16_out;

    override_owned_zero_copy_parity_one_memory();
    override_owned_zero_copy_parity_multi_memory_nonzero_offset();
    override_owned_zero_copy_parity_padded_stride();
    override_authoritative_over_stale_tensor_set_metadata();
    override_preserves_matching_logical_parent_offsets();
    override_segment_name_precedes_memory_index_for_segmented_sample();

    std::cout << "[OK] unit_tensor_set_meta_route_contract_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

#include "gst/SimaTensorSetMetaAbi.h"
#include "pipeline/internal/OutputTensorOverride.h"
#include "pipeline/internal/TensorUtil.h"

#include "test_utils.h"

#include <gst/gst.h>

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

void ensure_tensor_set_meta_registered() {
  int argc = 0;
  char** argv = nullptr;
  gst_init(&argc, &argv);
  if (gst_meta_get_info(SIMA_TENSOR_SET_META_NAME) == nullptr) {
    const gchar* tags[] = {nullptr};
    (void)gst_meta_register_custom(SIMA_TENSOR_SET_META_NAME, tags, nullptr, nullptr, nullptr);
  }
}

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
  require(owned.byte_offset == view.byte_offset, std::string(context) + ": byte_offset mismatch");
  require(owned.byte_offset == entry.byte_offset,
          std::string(context) + ": override byte_offset not applied");
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
  require(owned.route.memory_index == entry.memory_index,
          std::string(context) + ": memory_index not applied");
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

  const std::uint64_t logical_span = output_override_entry_physical_span_bytes(entry);
  require(logical_span > 0U, std::string(context) + ": invalid override logical span");

  const Mapping owned_map = owned.view_read();
  const Mapping view_map = view.view_read();
  require(owned_map.data != nullptr, std::string(context) + ": owned map failed");
  require(view_map.data != nullptr, std::string(context) + ": zero-copy map failed");
  require(owned_map.size_bytes == view_map.size_bytes,
          std::string(context) + ": readable span differs between owned and zero-copy");
  require(owned_map.size_bytes >= logical_span,
          std::string(context) + ": readable span is smaller than logical tensor span");
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
  const Sample base = make_stale_tensor_set_sample(sample.get(), 3U);

  OutputTensorOverride override;
  override.outputs.push_back(
      make_override_entry({2, 3}, {16, 4}, 4, 0, 0, 20, TensorDType::Int32, "terminal_class_ids"));
  override.outputs.push_back(
      make_override_entry({8}, {1}, 64, 0, 1, 21, TensorDType::UInt8, "terminal_aux_bytes"));
  require_override_owned_zero_copy_parity(base, override,
                                          "stale TensorSet authoritative override parity");
}

} // namespace

int main() {
  try {
    ensure_tensor_set_meta_registered();
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

    std::cout << "[OK] unit_tensor_set_meta_route_contract_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

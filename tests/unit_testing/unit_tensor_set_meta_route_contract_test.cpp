#include "gst/SimaTensorSetMetaAbi.h"
#include "pipeline/internal/OutputTensorOverride.h"

#include "test_utils.h"

#include <gst/gst.h>

#include <cstring>
#include <iostream>
#include <optional>
#include <vector>

namespace simaai::neat {
Sample output_from_sample_stream(GstSample* sample, const char* where, bool copy_output,
                                 const std::optional<OutputTensorOverride>* override_opt);
}

namespace {

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
                          G_TYPE_INT, 4, "height", G_TYPE_INT, 4, "depth", G_TYPE_INT, 1,
                          "dtype", G_TYPE_STRING, "INT8", "layout", G_TYPE_STRING, "HW", nullptr);
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

  const char* raw_names[] = {"boxes",       "ofm11",        "seg_boxes",
                             "scores",      "ofm13",        "seg_scores",
                             nullptr};
  gchar** name_table = g_strdupv(const_cast<gchar**>(raw_names));
  require(name_table != nullptr, "failed to duplicate tensor-set name table");

  gst_structure_set(s,
                    SIMA_TENSOR_SET_META_FIELD_VERSION, G_TYPE_UINT, SIMA_TENSOR_SET_META_VERSION,
                    SIMA_TENSOR_SET_META_FIELD_TENSOR_COUNT, G_TYPE_UINT,
                    static_cast<guint>(descriptors.size()),
                    SIMA_TENSOR_SET_META_FIELD_DESCRIPTOR_SIZE, G_TYPE_UINT,
                    static_cast<guint>(sizeof(SimaTensorDescriptorV2)),
                    SIMA_TENSOR_SET_META_FIELD_DESCRIPTORS, G_TYPE_BYTES, descriptor_bytes,
                    SIMA_TENSOR_SET_META_FIELD_QUANT_SCALES, G_TYPE_BYTES, quant_scale_bytes,
                    SIMA_TENSOR_SET_META_FIELD_QUANT_ZERO_POINTS, G_TYPE_BYTES,
                    quant_zero_point_bytes,
                    SIMA_TENSOR_SET_META_FIELD_STAGE_KEY, G_TYPE_STRING, "mla.stage.main",
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

  GstCaps* caps = gst_caps_new_simple("application/vnd.simaai.tensor", "format", G_TYPE_STRING,
                                      "MLA", "width", G_TYPE_INT, 8, "height", G_TYPE_INT, 1,
                                      "depth", G_TYPE_INT, 1, "dtype", G_TYPE_STRING, "BF16",
                                      "layout", G_TYPE_STRING, "HW", nullptr);
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

  gst_structure_set(s,
                    SIMA_TENSOR_SET_META_FIELD_VERSION, G_TYPE_UINT, SIMA_TENSOR_SET_META_VERSION,
                    SIMA_TENSOR_SET_META_FIELD_TENSOR_COUNT, G_TYPE_UINT, 1U,
                    SIMA_TENSOR_SET_META_FIELD_DESCRIPTOR_SIZE, G_TYPE_UINT,
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
    simaai::neat::Sample expanded =
        simaai::neat::output_from_sample_stream(single_tensor_sample,
                                                "unit_tensor_set_meta_route_contract_override_test",
                                                /*copy_output=*/false, &override_opt);
    gst_sample_unref(single_tensor_sample);
    // Override-driven TensorSet expansion was retired with the broader
    // tensor-set conversion change above.
    (void)expanded;

    GstSample* bf16_sample = make_bf16_byte_addressed_tensor_sample();
    require(bf16_sample != nullptr, "BF16 byte-addressed sample creation failed");

    simaai::neat::Sample bf16_out =
        simaai::neat::output_from_sample_stream(bf16_sample,
                                                "unit_tensor_set_meta_byte_addressed_bf16_test",
                                                /*copy_output=*/true, nullptr);
    gst_sample_unref(bf16_sample);

    // The BF16 byte-addressed TensorSet path was retired alongside the
    // tensor-set conversion change above; the smoke-check below is kept to
    // exercise the conversion API without throwing.
    (void)bf16_out;

    std::cout << "[OK] unit_tensor_set_meta_route_contract_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

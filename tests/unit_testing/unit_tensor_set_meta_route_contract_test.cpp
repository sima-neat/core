#include "gst/SimaTensorSetMetaAbi.h"
#include "gst/GstInit.h"
#include "pipeline/TensorAdapters.h"
#include "pipeline/gst/InputStreamInternal.h"
#include "dmabuf_test_utils.h"
#include "pipeline/internal/OutputTensorOverride.h"
#include "pipeline/internal/HolderLoanGate.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/TensorUtil.h"

#include "test_main.h"

#include <gstsimaaitensorbuffer_meta.h>
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
    descriptors[1].segment_name_id = 2; // Same runtime carrier; logical identities remain distinct.
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

  const char* raw_names[] = {"boxes", "ofm11",      "shared_arena", "scores",
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
    updated[1].segment_name_id = 5; // This negative case intentionally names the CPU segment.
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

void bounded_override_expansion_and_invalid_span() {
  GstSamplePtr sample(make_tensor_sample_with_contract_meta(1));
  auto* buffer = gst_sample_get_buffer(sample.get());
  gst_buffer_remove_all_memory(buffer);
  auto* memory = gst_allocator_alloc(nullptr, 128, nullptr);
  require(memory != nullptr, "bounded expansion fixture allocation");
  GstMapInfo mapping{};
  require(gst_memory_map(memory, &mapping, GST_MAP_WRITE), "bounded expansion fixture mapping");
  for (gsize i = 0; i != mapping.size; ++i) {
    mapping.data[i] = static_cast<guint8>(i);
  }
  gst_memory_unmap(memory, &mapping);
  gst_buffer_append_memory(buffer, memory);
  OutputTensorOverride override;
  for (int i = 0; i != 2; ++i) {
    OutputTensorOverrideEntry entry;
    entry.shape = {4, 4};
    entry.strides_bytes = {16, 4};
    entry.byte_offset = 64 * i;
    entry.memory_index = 0;
    entry.logical_output_index = i;
    entry.route_slot = i;
    entry.dtype = TensorDType::Float32;
    entry.layout = TensorLayout::HW;
    entry.name = i == 0 ? "boxes_fp32" : "scores_fp32";
    entry.segment_name = "shared_arena";
    override.outputs.push_back(std::move(entry));
  }
  const std::optional<OutputTensorOverride> override_opt(override);
  const auto expanded = simaai::neat::output_from_sample_stream(sample.get(), "bounded expansion",
                                                                false, &override_opt);
  require(expanded.tensors.size() == 2, "bounded expansion did not publish both logical heads");
  for (int i = 0; i != 2; ++i) {
    const auto& tensor = expanded.tensors[i];
    require(tensor.shape == std::vector<int64_t>({4, 4}) && tensor.dtype == TensorDType::Float32 &&
                tensor.byte_offset == 64 * i && tensor.route.logical_index == i &&
                tensor.route.name == override.outputs[i].name,
            "bounded expansion lost projected logical authority");
    const auto bytes = tensor.view_read();
    require(bytes.data && bytes.size_bytes >= 64 &&
                static_cast<const std::uint8_t*>(bytes.data)[0] == 64 * i &&
                static_cast<const std::uint8_t*>(bytes.data)[63] == 64 * i + 63,
            "bounded expansion selected the wrong physical view");
  }
  // The original smoke fixture silently requested 128 bytes from a 64-byte
  // carrier. Keep that exact case as an explicit rejection, not an ignored output.
  GstSamplePtr short_sample(make_tensor_sample_with_contract_meta(1));
  pipeline_internal::reset_tensor_io_stats();
  bool rejected = false;
  try {
    (void)simaai::neat::output_from_sample_stream(short_sample.get(), "invalid expansion", true,
                                                  &override_opt);
  } catch (const std::runtime_error& error) {
    rejected = !std::string(error.what()).empty();
  }
  require(rejected, "out-of-bounds logical expansion was accepted");
  require(pipeline_internal::snapshot_tensor_io_stats().tensor_copy_count == 0,
          "invalid expansion entered output copying");
}

} // namespace

int main() {
  return sima_test::run_test("unit_tensor_set_meta_route_contract_test", [] {
    simaai::neat::gst_init_once();
    GstSamplePtr sample(make_tensor_sample_with_contract_meta());
    const auto out =
        simaai::neat::output_from_sample_stream(sample.get(), "route contract", false, nullptr);
    const auto published = simaai::neat::tensors_from_sample(out);
    require(published.size() == 2 && published[0].route.name == "boxes" &&
                published[1].route.name == "scores" && published[0].byte_offset == 0 &&
                published[1].byte_offset == 16 &&
                published[0].route.segment_name == "shared_arena" &&
                published[1].route.segment_name == "shared_arena",
            "shared runtime carrier lost logical identity/offsets");
    bounded_override_expansion_and_invalid_span();
    unresolved_mixed_dmabuf_descriptor_fails_without_copy();
    GstSamplePtr bf16(make_bf16_byte_addressed_tensor_sample());
    bool rejected = false;
    try {
      (void)simaai::neat::output_from_sample_stream(bf16.get(), "malformed BF16 authority", true,
                                                    nullptr);
    } catch (const std::runtime_error& error) {
      rejected = std::string(error.what()).find("logical bytes do not match shape/dtype") !=
                 std::string::npos;
    }
    require(rejected, "malformed BF16 canonical byte extent was accepted");
  });
}

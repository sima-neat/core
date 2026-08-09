#include "pipeline/TensorAdapters.h"
#include "pipeline/internal/TensorTransfer.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/SimaaiMemory.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/internal/SimaaiGstCompat.h"
#include "gst/GstInit.h"

#include "test_utils.h"

#include <gst/gst.h>
#include <gst/allocators/gstdmabuf.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#if SIMA_HAS_SIMAAI_POOL
static void free_pool(GstBufferPool* pool) {
  if (pool) {
    gst_simaai_free_buffer_pool(pool);
  }
}

static std::unique_ptr<GstBufferPool, void (*)(GstBufferPool*)>
make_pool(const std::vector<simaai::neat::Segment>& segments, GstMemoryFlags flags) {
  gst_simaai_segment_memory_init_once();
  std::vector<gsize> sizes;
  std::vector<const char*> names;
  sizes.reserve(segments.size());
  names.reserve(segments.size());
  for (const auto& seg : segments) {
    sizes.push_back(static_cast<gsize>(seg.size_bytes));
    names.push_back(seg.name.c_str());
  }
  GstBufferPool* pool = gst_simaai_allocate_buffer_pool2(
      /*object=*/nullptr, gst_simaai_memory_get_segment_allocator(),
      /*min_buffers=*/1,
      /*max_buffers=*/1, flags, static_cast<gsize>(segments.size()), sizes.data(), names.data());
  return std::unique_ptr<GstBufferPool, void (*)(GstBufferPool*)>(pool, free_pool);
}

static GstBuffer* acquire_buffer(GstBufferPool* pool) {
  if (!pool)
    return nullptr;
  GstBuffer* buf = nullptr;
  if (gst_buffer_pool_acquire_buffer(pool, &buf, nullptr) != GST_FLOW_OK)
    return nullptr;
  return buf;
}

static void add_sima_meta(GstBuffer* buffer) {
  if (!gst_meta_get_info("GstSimaMeta")) {
    static const gchar* tags[] = {nullptr};
    gst_meta_register_custom("GstSimaMeta", tags, nullptr, nullptr, nullptr);
  }
  GstCustomMeta* meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  require(meta != nullptr, "failed to add GstSimaMeta");
  GstStructure* s = gst_custom_meta_get_structure(meta);
  require(s != nullptr, "missing GstSimaMeta structure");
  gst_structure_set(s, "buffer-id", G_TYPE_INT64, static_cast<gint64>(42), "buffer-name",
                    G_TYPE_STRING, "unit-test", "frame-id", G_TYPE_INT64, static_cast<gint64>(7),
                    "stream-id", G_TYPE_STRING, "0", "timestamp", G_TYPE_UINT64,
                    static_cast<guint64>(123), "pcie-buffer-id", G_TYPE_INT64,
                    static_cast<gint64>(99), nullptr);
}
#endif

int main() {
  try {
#if !SIMA_HAS_SIMAAI_POOL
    require(false, "tensor_device_placement_test requires simaai buffer pool");
#else
    simaai::neat::gst_init_once();

    std::vector<simaai::neat::Segment> segments = {
        {"seg0", 64},
        {"seg1", 32},
    };

    GstCaps* caps = gst_caps_new_simple("application/vnd.simaai.tensor", "format", G_TYPE_STRING,
                                        "UINT8", "width", G_TYPE_INT, 8, "height", G_TYPE_INT, 12,
                                        "depth", G_TYPE_INT, 1, nullptr);
    require(caps != nullptr, "failed to create caps");

    auto pool_ev74 =
        make_pool(segments, static_cast<GstMemoryFlags>(GST_SIMAAI_MEMORY_TARGET_EV74 |
                                                        GST_SIMAAI_MEMORY_FLAG_CACHED));
    GstBuffer* ev74_buf = acquire_buffer(pool_ev74.get());
    require(ev74_buf != nullptr, "EV74 buffer allocation failed");
    add_sima_meta(ev74_buf);
    GstSample* sample_ev74 = gst_sample_new(ev74_buf, gst_caps_ref(caps), nullptr, nullptr);
    require(sample_ev74 != nullptr, "failed to create EV74 sample");
    simaai::neat::Tensor t_ev74 = simaai::neat::from_gst_sample(sample_ev74);
    gst_sample_unref(sample_ev74);
    require(t_ev74.device.type == simaai::neat::DeviceType::SIMA_CVU, "EV74 device mismatch");
    require(t_ev74.device.id == 0, "EV74 device id mismatch");

    auto pool_dms0 =
        make_pool(segments, static_cast<GstMemoryFlags>(GST_SIMAAI_MEMORY_TARGET_DMS0 |
                                                        GST_SIMAAI_MEMORY_FLAG_CACHED));
    GstBuffer* dms_buf = acquire_buffer(pool_dms0.get());
    require(dms_buf != nullptr, "DMS0 buffer allocation failed");
    GstSample* sample_dms = gst_sample_new(dms_buf, gst_caps_ref(caps), nullptr, nullptr);
    require(sample_dms != nullptr, "failed to create DMS0 sample");
    simaai::neat::Tensor t_dms = simaai::neat::from_gst_sample(sample_dms);
    gst_sample_unref(sample_dms);
    require(t_dms.device.type == simaai::neat::DeviceType::SIMA_MLA, "DMS0 device mismatch");
    require(t_dms.device.id == 0, "DMS0 device id mismatch");

    gst_caps_unref(caps);

    auto cpu_storage = simaai::neat::make_cpu_owned_storage(96);
    simaai::neat::Tensor cpu;
    cpu.storage = cpu_storage;
    cpu.dtype = simaai::neat::TensorDType::UInt8;
    cpu.shape = {96};
    cpu.strides_bytes = {1};
    cpu.layout = simaai::neat::TensorLayout::Unknown;
    cpu.device = {simaai::neat::DeviceType::CPU, 0};
    cpu.read_only = false;

    simaai::neat::Tensor mla_pref = cpu.mla(false);
    require(mla_pref.device.type == simaai::neat::DeviceType::SIMA_CVU,
            "mla(false) should prefer CVU");

    simaai::neat::Tensor mla_force = cpu.mla(true);
    require(mla_force.device.type == simaai::neat::DeviceType::SIMA_MLA && mla_force.device.id == 0,
            "mla(true) should yield DMS0");

    require(static_cast<bool>(t_ev74.storage), "missing EV74 storage");
    t_ev74.storage->sima_segments = segments;

    simaai::neat::Tensor dms_copy = t_ev74.mla(true);
    auto stats_mid = simaai::neat::pipeline_internal::tensor_transfer_pool_stats();

    GstBuffer* out_buf =
        simaai::neat::pipeline_internal::buffer_from_tensor_holder(dms_copy.storage->holder);
    require(out_buf != nullptr, "missing output GstBuffer");
    GstCustomMeta* out_meta = gst_buffer_get_custom_meta(out_buf, "GstSimaMeta");
    require(out_meta != nullptr, "missing GstSimaMeta on transfer");
    GstStructure* meta_s = gst_custom_meta_get_structure(out_meta);
    require(meta_s != nullptr, "missing meta structure");
    gint64 out_id = 0;
    gst_structure_get_int64(meta_s, "buffer-id", &out_id);
    require(out_id == 42, "meta buffer-id mismatch");
    const gchar* out_name = gst_structure_get_string(meta_s, "buffer-name");
    require(out_name && std::string(out_name) == "unit-test", "meta buffer-name mismatch");

    GstMemory* mem = gst_buffer_peek_memory(out_buf, 0);
    require(mem != nullptr, "missing output memory");
    for (const auto& seg : segments) {
      void* ptr = gst_simaai_memory_get_segment(mem, seg.name.c_str());
      require(ptr != nullptr, "missing output segment");
    }
    gst_buffer_unref(out_buf);

    simaai::neat::Tensor dms_copy2 = t_ev74.mla(true);
    require(dms_copy2.device.type == simaai::neat::DeviceType::SIMA_MLA,
            "second transfer should stay on MLA");
    auto stats_after = simaai::neat::pipeline_internal::tensor_transfer_pool_stats();
    require(stats_after.hits > stats_mid.hits, "expected pool cache hit");

    // The strict driver-mode placement uses the same public Tensor API but
    // must produce ordinary GstDmaBufMemory from the CMA heap. It must not
    // enter the legacy segmented allocator or disguise that allocator as a
    // DMA-BUF.
    std::vector<std::uint8_t> direct_data(4096U);
    for (std::size_t i = 0; i < direct_data.size(); ++i) {
      direct_data[i] = static_cast<std::uint8_t>(i & 0xffU);
    }
    auto direct_cpu = simaai::neat::Tensor::from_vector(
        direct_data, {static_cast<std::int64_t>(direct_data.size())},
        simaai::neat::TensorMemory::CPU);
    std::vector<simaai::neat::Segment> direct_segments{{"ifm0", direct_data.size()}};
    auto direct = simaai::neat::pipeline_internal::transfer_to_device(
        direct_cpu, {simaai::neat::DeviceType::SIMA_CVU, 0}, &direct_segments, nullptr,
        simaai::neat::pipeline_internal::MemoryBackendPolicy::DmaBufPlan);
    require(direct.device.type == simaai::neat::DeviceType::SIMA_CVU,
            "direct EV74 device mismatch");
    require(direct.storage && direct.storage->sima_segments.size() == 1U,
            "direct EV74 storage contract mismatch");
    GstBuffer* direct_buffer =
        simaai::neat::pipeline_internal::buffer_from_tensor_holder(direct.storage->holder);
    require(direct_buffer != nullptr, "missing direct EV74 GstBuffer");
    require(gst_buffer_n_memory(direct_buffer) == 1U, "direct EV74 buffer must contain one memory");
    require(gst_is_dmabuf_memory(gst_buffer_peek_memory(direct_buffer, 0U)),
            "direct EV74 memory is not standard GstDmaBufMemory");
    gst_buffer_unref(direct_buffer);

    auto direct_map = direct.map_read();
    require(direct_map.data != nullptr && direct_map.size_bytes >= direct_data.size(),
            "direct EV74 CPU map failed");
    require(std::memcmp(direct_map.data, direct_data.data(), direct_data.size()) == 0,
            "direct EV74 payload mismatch");

    std::vector<std::uint8_t> direct_data_1(2048U, 0x5aU);
    auto direct_cpu_1 = simaai::neat::Tensor::from_vector(
        direct_data_1, {static_cast<std::int64_t>(direct_data_1.size())},
        simaai::neat::TensorMemory::CPU);
    std::vector<simaai::neat::Segment> direct_segments_1{{"ifm0", direct_data_1.size()}};
    auto direct_1 = simaai::neat::pipeline_internal::transfer_to_device(
        direct_cpu_1, {simaai::neat::DeviceType::SIMA_CVU, 0}, &direct_segments_1, nullptr,
        simaai::neat::pipeline_internal::MemoryBackendPolicy::DmaBufPlan);
    direct.route.name = "image_l";
    direct.route.backend_name = "input_tensor";
    direct.route.segment_name = "input_tensor";
    direct.route.logical_index = 0;
    direct.route.physical_index = 0;
    direct.route.route_slot = 0;
    direct.route.memory_index = 0;
    direct_1.route.name = "image_uv";
    direct_1.route.backend_name = "input_tensor";
    direct_1.route.segment_name = "input_tensor";
    direct_1.route.logical_index = 1;
    direct_1.route.physical_index = 0;
    direct_1.route.route_slot = 1;
    direct_1.route.memory_index = 1;

    auto ingress = simaai::neat::sample_from_tensors({direct, direct_1});
    ingress.payload_type = simaai::neat::PayloadType::Tensor;
    ingress.media_type = "application/vnd.simaai.tensor";
    ingress.segment_name = "input_tensor";
    std::string ingress_error;
    auto ingress_holder = simaai::neat::pipeline_internal::sample_to_gst_envelope_holder(
        ingress, &ingress_error, /*allow_zero_copy=*/true,
        simaai::neat::pipeline_internal::MemoryBackendPolicy::DmaBufPlan);
    require(ingress_holder != nullptr,
            ingress_error.empty() ? "direct tensor-set envelope failed" : ingress_error);
    GstBuffer* ingress_buffer =
        simaai::neat::pipeline_internal::buffer_from_tensor_holder(ingress_holder);
    require(ingress_buffer != nullptr, "missing direct tensor-set GstBuffer");
    require(gst_buffer_n_memory(ingress_buffer) == 2U,
            "direct tensor-set must preserve one memory per public input");
    for (guint i = 0U; i < gst_buffer_n_memory(ingress_buffer); ++i) {
      require(gst_is_dmabuf_memory(gst_buffer_peek_memory(ingress_buffer, i)),
              "direct tensor-set contains non-DMA-BUF memory");
    }
    GstCustomMeta* tensor_set_meta =
        gst_buffer_get_custom_meta(ingress_buffer, SIMA_TENSOR_SET_META_NAME);
    require(tensor_set_meta != nullptr,
            "direct tensor-set is missing canonical route metadata");
    GstStructure* tensor_set_structure =
        gst_custom_meta_get_structure(tensor_set_meta);
    require(tensor_set_structure != nullptr,
            "direct tensor-set route metadata has no structure");
    guint physical_binding_count = 0U;
    require(gst_structure_get_uint(
                tensor_set_structure,
                "physical-binding-count",
                &physical_binding_count) &&
                physical_binding_count == 2U,
            "direct tensor-set must publish two dense physical carriers");
    const GValue* descriptor_value = gst_structure_get_value(
        tensor_set_structure, SIMA_TENSOR_SET_META_FIELD_DESCRIPTORS);
    auto* descriptor_blob =
        descriptor_value && G_VALUE_HOLDS(descriptor_value, G_TYPE_BYTES)
            ? static_cast<GBytes*>(g_value_get_boxed(descriptor_value))
            : nullptr;
    gsize descriptor_bytes = 0U;
    const auto* ingress_descriptors = descriptor_blob
                                          ? static_cast<const SimaTensorDescriptorV2*>(
                                                g_bytes_get_data(descriptor_blob,
                                                                 &descriptor_bytes))
                                          : nullptr;
    require(ingress_descriptors != nullptr &&
                descriptor_bytes == 2U * sizeof(SimaTensorDescriptorV2),
            "direct tensor-set descriptor table is malformed");
    require(ingress_descriptors[0].physical_index == 0 &&
                ingress_descriptors[0].memory_index == 0 &&
                ingress_descriptors[1].physical_index == 1 &&
                ingress_descriptors[1].memory_index == 1,
            "direct tensor-set descriptor identities must mirror its carriers");
    gst_buffer_unref(ingress_buffer);

    std::cout << "[OK] tensor_device_placement_test passed\n";
    return 0;
#endif
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

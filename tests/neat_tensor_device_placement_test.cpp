#include "pipeline/TensorAdapters.h"
#include "pipeline/internal/TensorTransfer.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "nodes/io/Input.h"
#include <simaai/scheduler/client/dma_memory.h>
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/internal/SimaaiGstCompat.h"
#include "gst/GstInit.h"

#include "test_utils.h"

#include <gst/gst.h>
#include <gst/allocators/gstdmabuf.h>
#include <gstsimaaitensorbuffer_meta.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

static void check_transfer_metadata_failure(const simaai::neat::Tensor& source) {
  using namespace simaai::gst;
  auto* buffer = gst_sample_get_buffer(static_cast<GstSample*>(source.storage->holder.get()));
  const auto bytes = source.dense_bytes_tight();
  TensorBufferPublishContract contract;
  contract.stage_key = "transfer-metadata-test";
  contract.physical_outputs = {{0, bytes, "tensor"}};
  TensorBufferPublishLogicalOutput output;
  output.logical_index = output.physical_index = output.memory_index = 0;
  output.backend_output_index = output.route_slot = 0;
  output.logical_name = output.segment_name = "tensor";
  output.size_bytes = bytes;
  output.dtype = SIMA_TENSOR_SET_DTYPE_UINT8_V1;
  output.shape = source.shape;
  output.stride_bytes = source.strides_bytes;
  contract.logical_outputs = {output};
  TensorBufferPreparedMetaTemplate prepared;
  std::string error;
  require(tensor_buffer_prepare_meta_template_from_contract(contract, {{0, 0, bytes}}, &prepared,
                                                            &error),
          error);
  const auto publication = tensor_buffer_freeze_publication(prepared, &error);
  require(publication && tensor_buffer_attach_publication(buffer, publication, &error), error);
  require(tensor_buffer_attach_frame_metadata(buffer, TensorBufferFrameMetadata{}, &error), error);
  add_sima_meta(buffer); // Competing authorities must fail without leaking the retained source.
  const auto references = GST_MINI_OBJECT_REFCOUNT_VALUE(buffer);
  bool rejected = false;
  try {
    (void)source.mla(true);
  } catch (const std::runtime_error& exception) {
    rejected = std::string(exception.what()).find("transfer: compact frame metadata:") == 0;
  }
  require(GST_MINI_OBJECT_REFCOUNT_VALUE(buffer) == references,
          "failed metadata transfer leaked its source buffer reference");
  require(rejected, "transfer did not report the compact metadata error");
}

static void check_selected_carrier_transfer(simaai::neat::Device target) {
  using namespace simaai::neat;
  using namespace simaai::neat::pipeline_internal;
  bool released = false;
  auto buffer = std::shared_ptr<GstBuffer>(gst_buffer_new(), gst_buffer_unref);
  gst_buffer_append_memory(buffer.get(), gst_allocator_alloc(nullptr, 32, nullptr));
  gst_buffer_append_memory(buffer.get(), gst_allocator_alloc(nullptr, 128, nullptr));
  gst_mini_object_weak_ref(
      GST_MINI_OBJECT(buffer.get()),
      [](gpointer flag, GstMiniObject*) { *static_cast<bool*>(flag) = true; }, &released);
  auto parent = std::shared_ptr<GstBuffer>(gst_buffer_new(), gst_buffer_unref);
  require(gst_buffer_add_parent_buffer_meta(buffer.get(), parent.get()) != nullptr,
          "producer loan fixture");
  add_sima_meta(buffer.get());
  InputOptions options;
  options.preprocess_meta.emplace();
  options.preprocess_meta->enabled = true;
  options.preprocess_meta->target_width = options.preprocess_meta->target_height = 640;
  options.preprocess_meta->resize_mode = "letterbox";
  require(apply_simaai_preprocess_meta_template(buffer.get(), options, 1280, 720),
          "transfer preprocess fixture");
  GST_BUFFER_PTS(buffer.get()) = 1234;
  auto sample = std::shared_ptr<GstSample>(gst_sample_new(buffer.get(), nullptr, nullptr, nullptr),
                                           gst_sample_unref);
  Tensor source;
  source.storage = make_gst_sample_storage(sample.get());
  source = tensor_view_from_sample_memory(source, 1);
  source.storage->sima_segments = {{"small", 32}, {"selected", 128}};
  source.dtype = TensorDType::UInt8;
  source.layout = TensorLayout::HW;
  source.shape = {4, 16};
  source.strides_bytes = {24, 1};
  source.byte_offset = 8;
  source.route.physical_byte_offset = 8;
  source.route.stage_key = "encoder";
  source.route.name = "features";
  source.route.logical_index = 7;
  source.route.physical_index = 17;
  source.route.memory_index = 1;
  source.route.segment_name = "selected";
  {
    auto mapping = source.storage->map(MapMode::Write);
    require(mapping.data && mapping.size_bytes >= 128, "selected carrier map");
    for (unsigned i = 0; i < 128; ++i)
      static_cast<std::uint8_t*>(mapping.data)[i] = static_cast<std::uint8_t>(i);
  }
  const auto transfer = [&](const Tensor& tensor) {
    return target.type == DeviceType::CPU ? transfer_to_cpu(tensor) : tensor.to(target);
  };
  auto copy = transfer(source);
  require(copy.byte_offset == 0 && copy.strides_bytes == std::vector<int64_t>({16, 1}),
          "transfer must pack the selected strided view");
  require(copy.route.memory_index == 0 && copy.route.physical_byte_offset == 0 &&
              copy.route.physical_index == 17 && copy.route.logical_index == 7 &&
              copy.route.stage_key == "encoder" && copy.route.name == "features",
          "transfer must rebase storage selection and preserve logical provenance");
  if (target.type != DeviceType::CPU) {
    require(copy.storage->sima_segments.size() == 1 &&
                copy.storage->sima_segments[0].size_bytes == 64,
            "transfer retained unrelated producer segments");
    auto* destination = gst_sample_get_buffer(static_cast<GstSample*>(copy.storage->holder.get()));
    require(gst_buffer_get_parent_buffer_meta(destination) == nullptr,
            "transfer retained the producer loan");
    require(GST_BUFFER_PTS(destination) == 1234, "transfer lost presentation time");
    const auto preprocess = read_simaai_preprocess_meta(destination);
    require(preprocess && preprocess->original_width == 1280 && preprocess->original_height == 720,
            "transfer lost preprocessing geometry");
    auto* metadata = gst_buffer_get_custom_meta(destination, "GstSimaMeta");
    require(metadata &&
                !gst_structure_has_field(gst_custom_meta_get_structure(metadata), "buffer-id"),
            "transfer copied the producer buffer-id with preprocessing metadata");
    std::string error;
    auto ingress = sample_to_gst_envelope_holder(sample_from_tensors({copy}), &error, true);
    require(ingress != nullptr, "transferred tensor cannot reenter a graph: " + error);
  }
  {
    Tensor tiled = source;
    tiled.dtype = TensorDType::BFloat16;
    tiled.shape = {1, 1, 1, 9};
    tiled.strides_bytes.clear();
    TessSpec tess;
    tess.format = "MLA_TILED_BF16_PAD16";
    tess.slice_shape = {1, 1, 1, 16};
    tess.axes = {0, 1, 2, 3};
    tess.storage_bytes = 32;
    tiled.semantic.tess = tess;
    const auto transferred = transfer(tiled);
    require(transferred.has_tiled_storage() && transferred.strides_bytes.empty() &&
                transferred.semantic.tess->storage_bytes == 32,
            "transfer lost the padded tiled layout");
    const auto bytes = transferred.copy_payload_bytes();
    require(bytes.size() == 32, "transfer truncated tile padding");
    for (unsigned i = 0; i < bytes.size(); ++i)
      require(bytes[i] == 8 + i, "transfer changed tiled payload or padding");
  }
  source = {};
  sample.reset();
  buffer.reset();
  require(released, "transfer retained its producer");
  const auto bytes = copy.copy_dense_bytes_tight();
  require(bytes.size() == 64, "transferred payload length");
  for (unsigned row = 0; row < 4; ++row)
    for (unsigned col = 0; col < 16; ++col)
      require(bytes[row * 16 + col] == 8 + row * 24 + col, "transferred payload mismatch");
}

int main() {
  try {
    simaai::neat::gst_init_once();

    check_selected_carrier_transfer({simaai::neat::DeviceType::SIMA_CVU, 0});
    check_selected_carrier_transfer({simaai::neat::DeviceType::SIMA_MLA, 0});
    check_selected_carrier_transfer({simaai::neat::DeviceType::CPU, 0});

    std::vector<simaai::neat::Segment> segments = {
        {"seg0", 64},
        {"seg1", 32},
    };

    auto seed_storage = simaai::neat::make_cpu_owned_storage(96);
    simaai::neat::Tensor seed;
    seed.storage = seed_storage;
    seed.dtype = simaai::neat::TensorDType::UInt8;
    seed.shape = {96};
    seed.strides_bytes = {1};
    seed.layout = simaai::neat::TensorLayout::Unknown;
    seed.device = {simaai::neat::DeviceType::CPU, 0};
    seed.read_only = false;

    simaai::neat::Tensor t_ev74 = simaai::neat::pipeline_internal::transfer_to_device(
        seed, {simaai::neat::DeviceType::SIMA_CVU, 0}, &segments);
    require(t_ev74.storage->sima_segments.size() == 2U &&
                t_ev74.storage->sima_segments[0].name == "seg0" &&
                t_ev74.storage->sima_segments[0].size_bytes == 64U &&
                t_ev74.storage->sima_segments[1].name == "seg1" &&
                t_ev74.storage->sima_segments[1].size_bytes == 32U,
            "transfer lost the explicit destination segment layout");
    auto* ev74_sample = static_cast<GstSample*>(t_ev74.storage->holder.get());
    GstBuffer* ev74_buf = ev74_sample ? gst_sample_get_buffer(ev74_sample) : nullptr;
    require(ev74_buf != nullptr, "missing EV74 DMA-BUF");
    require(gst_buffer_n_memory(ev74_buf) == 1U &&
                gst_is_dmabuf_memory(gst_buffer_peek_memory(ev74_buf, 0U)),
            "EV74 placement must use standard DMA-BUF memory");
    add_sima_meta(ev74_buf);
    require(t_ev74.device.type == simaai::neat::DeviceType::SIMA_CVU, "EV74 device mismatch");
    require(t_ev74.device.id == 0, "EV74 device id mismatch");

    simaai::neat::Tensor t_dms = simaai::neat::pipeline_internal::transfer_to_device(
        seed, {simaai::neat::DeviceType::SIMA_MLA, 0}, &segments);
    require(t_dms.device.type == simaai::neat::DeviceType::SIMA_MLA, "DMS0 device mismatch");
    require(t_dms.device.id == 0, "DMS0 device id mismatch");

    GstBuffer* dms_buf =
        simaai::neat::pipeline_internal::buffer_from_tensor_holder(t_dms.storage->holder);
    require(dms_buf != nullptr, "missing DMS DMA-BUF");
    require(gst_buffer_n_memory(dms_buf) == 1U &&
                gst_is_dmabuf_memory(gst_buffer_peek_memory(dms_buf, 0U)),
            "MLA placement must use the DMS DMA heap");
    gst_buffer_unref(dms_buf);

    {
      // Auto selects DMS for MLA tensor ingress. Allocation must follow the
      // resolved placement rather than interpreting Auto as a CMA request.
      simaai::neat::InputOptions input_options;
      input_options.payload_type = simaai::neat::PayloadType::Tensor;
      input_options.format = "FLOAT32";
      input_options.buffer_name = "ifm0";
      input_options.memory_policy = simaai::neat::InputMemoryPolicy::Auto;
      input_options.pool_min_buffers = 1;
      input_options.pool_max_buffers = 1;
      simaai::neat::InputBufferPoolGuard pool;
      std::unique_ptr<GstBuffer, decltype(&gst_buffer_unref)> buffer(
          simaai::neat::allocate_input_buffer(4096U, input_options, pool), gst_buffer_unref);
      require(buffer != nullptr && gst_buffer_n_memory(buffer.get()) == 1U,
              "Auto tensor ingress did not allocate one DMA-BUF memory");
      auto memory = simaai::scheduler::client::dma::DmaBufMemory::retain(
          gst_buffer_peek_memory(buffer.get(), 0U));
      require(memory.has_value() &&
                  memory->knownHeapKind() == simaai::scheduler::client::dma::HeapKind::MlaDms,
              "Auto tensor ingress must allocate from the resolved DMS heap");
    }

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

    simaai::neat::Tensor dms_copy = t_ev74.mla(true);

    GstBuffer* out_buf =
        simaai::neat::pipeline_internal::buffer_from_tensor_holder(dms_copy.storage->holder);
    require(out_buf != nullptr, "missing output GstBuffer");
    require(gst_buffer_get_custom_meta(out_buf, "GstSimaMeta") == nullptr,
            "transfer must not retain the producer's allocation identity");
    require(gst_buffer_n_memory(out_buf) == 1U &&
                gst_is_dmabuf_memory(gst_buffer_peek_memory(out_buf, 0U)),
            "MLA transfer must preserve standard DMA-BUF storage");
    require(dms_copy.storage->sima_segments.size() == 1U &&
                dms_copy.storage->sima_segments[0].size_bytes == 96U,
            "implicit transfer must describe only the copied payload");
    gst_buffer_unref(out_buf);

    simaai::neat::Tensor dms_copy2 = t_ev74.mla(true);
    require(dms_copy2.device.type == simaai::neat::DeviceType::SIMA_MLA,
            "second transfer should stay on MLA");

    // Explicit CVU placement produces standard DMA-BUF memory from the CMA heap.
    std::vector<std::uint8_t> direct_data(4096U);
    for (std::size_t i = 0; i < direct_data.size(); ++i) {
      direct_data[i] = static_cast<std::uint8_t>(i & 0xffU);
    }
    auto direct_cpu = simaai::neat::Tensor::from_vector(
        direct_data, {static_cast<std::int64_t>(direct_data.size())},
        simaai::neat::TensorMemory::CPU);
    std::vector<simaai::neat::Segment> direct_segments{{"ifm0", direct_data.size()}};
    auto direct = simaai::neat::pipeline_internal::transfer_to_device(
        direct_cpu, {simaai::neat::DeviceType::SIMA_CVU, 0}, &direct_segments);
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
        direct_cpu_1, {simaai::neat::DeviceType::SIMA_CVU, 0}, &direct_segments_1);
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
    for (guint i = 2U; i <= gst_buffer_get_max_memory(); ++i) {
      ingress.tensors.push_back(i % 2U == 0U ? direct : direct_1);
    }
    ingress.payload_type = simaai::neat::PayloadType::Tensor;
    ingress.media_type = "application/vnd.simaai.tensor";
    ingress.segment_name = "input_tensor";
    std::string ingress_error;
    simaai::neat::pipeline_internal::reset_tensor_io_stats();
    auto ingress_holder = simaai::neat::pipeline_internal::sample_to_gst_envelope_holder(
        ingress, &ingress_error, /*allow_zero_copy=*/true);
    const auto ingress_stats = simaai::neat::pipeline_internal::snapshot_tensor_io_stats();
    require(ingress_holder != nullptr,
            ingress_error.empty() ? "direct tensor-set envelope failed" : ingress_error);
    GstBuffer* ingress_buffer =
        simaai::neat::pipeline_internal::buffer_from_tensor_holder(ingress_holder);
    require(ingress_buffer != nullptr, "missing direct tensor-set GstBuffer");
    require(gst_buffer_n_memory(ingress_buffer) == 2U,
            "repeated logical tensors must reuse two memory slots");
    for (guint i = 0U; i < gst_buffer_n_memory(ingress_buffer); ++i) {
      require(gst_is_dmabuf_memory(gst_buffer_peek_memory(ingress_buffer, i)),
              "direct tensor-set contains non-DMA-BUF memory");
      auto* source = static_cast<GstSample*>(ingress.tensors[i].storage->holder.get());
      require(gst_buffer_peek_memory(ingress_buffer, i) ==
                  gst_buffer_peek_memory(gst_sample_get_buffer(source), 0U),
              "repeated tensors must retain the original DMA memory");
    }
    require(ingress_stats.tensor_copy_count == 0U && ingress_stats.tensor_copy_bytes == 0U &&
                ingress_stats.gst_memory_map_calls == 0U,
            "repeated-buffer admission must not copy or map DMA payloads");
    GstCustomMeta* tensor_set_meta =
        gst_buffer_get_custom_meta(ingress_buffer, SIMA_TENSOR_SET_META_NAME);
    require(tensor_set_meta != nullptr, "direct tensor-set is missing canonical route metadata");
    GstStructure* tensor_set_structure = gst_custom_meta_get_structure(tensor_set_meta);
    require(tensor_set_structure != nullptr, "direct tensor-set route metadata has no structure");
    guint physical_binding_count = 0U;
    require(gst_structure_get_uint(tensor_set_structure, "physical-binding-count",
                                   &physical_binding_count) &&
                physical_binding_count == ingress.tensors.size(),
            "direct tensor-set must publish a physical binding for each input view");
    const GValue* descriptor_value =
        gst_structure_get_value(tensor_set_structure, SIMA_TENSOR_SET_META_FIELD_DESCRIPTORS);
    auto* descriptor_blob = descriptor_value && G_VALUE_HOLDS(descriptor_value, G_TYPE_BYTES)
                                ? static_cast<GBytes*>(g_value_get_boxed(descriptor_value))
                                : nullptr;
    gsize descriptor_bytes = 0U;
    const auto* ingress_descriptors =
        descriptor_blob ? static_cast<const SimaTensorDescriptorV2*>(
                              g_bytes_get_data(descriptor_blob, &descriptor_bytes))
                        : nullptr;
    require(ingress_descriptors != nullptr &&
                descriptor_bytes == ingress.tensors.size() * sizeof(SimaTensorDescriptorV2),
            "direct tensor-set descriptor table is malformed");
    for (std::size_t i = 0U; i < ingress.tensors.size(); ++i) {
      require(ingress_descriptors[i].physical_index == static_cast<int>(i) &&
                  ingress_descriptors[i].memory_index == static_cast<int>(i % 2U),
              "direct tensor-set descriptors must preserve every view and its backing memory");
    }
    gst_buffer_unref(ingress_buffer);

    {
      simaai::neat::TensorList distinct;
      for (guint i = 0U; i <= gst_buffer_get_max_memory(); ++i) {
        distinct.push_back(simaai::neat::pipeline_internal::transfer_to_device(
            direct_cpu, {simaai::neat::DeviceType::SIMA_CVU, 0}, &direct_segments));
      }
      auto rejected_input = simaai::neat::sample_from_tensors(distinct);
      rejected_input.payload_type = simaai::neat::PayloadType::Tensor;
      rejected_input.media_type = "application/vnd.simaai.tensor";
      std::string detail;
      simaai::neat::pipeline_internal::reset_tensor_io_stats();
      require(!simaai::neat::pipeline_internal::sample_to_gst_envelope_holder(
                  rejected_input, &detail, /*allow_zero_copy=*/true) &&
                  detail.find("memory-slot limit") != std::string::npos,
              "excess distinct DMA buffers must reject without a copying fallback");
      const auto stats = simaai::neat::pipeline_internal::snapshot_tensor_io_stats();
      require(stats.tensor_copy_count == 0U && stats.tensor_copy_bytes == 0U &&
                  stats.gst_memory_map_calls == 0U,
              "distinct-buffer rejection must not copy or map DMA payloads");
    }

    ingress = {};
    ingress_holder.reset();
    {
      // These views retain the exact same GstSample, not merely two wrappers
      // around one fd. Rebinding must describe the requested views rather than
      // reuse the parent's tensor order, extent or offsets.
      auto first = direct;
      first.shape = {32};
      first.byte_offset = 64;
      first.route.name = "first_view";
      auto second = direct;
      second.shape = {16};
      second.byte_offset = 256;
      second.route.name = "second_view";
      require(first.storage->holder == second.storage->holder,
              "shared-view fixture must retain one GstSample holder");
      auto* parent_sample = static_cast<GstSample*>(direct.storage->holder.get());
      GstBuffer* parent_buffer = gst_sample_get_buffer(parent_sample);
      GstMemory* parent_memory = gst_buffer_peek_memory(parent_buffer, 0U);
      const int parent_fd = gst_dmabuf_memory_get_fd(parent_memory);
      std::weak_ptr<void> parent_lifetime = direct.storage->holder;
      std::shared_ptr<void> retained_envelope;
      for (int frame = 0; frame < 2; ++frame) {
        second.byte_offset = 256 + frame * 64;
        auto views =
            simaai::neat::sample_from_tensors(frame == 0 ? simaai::neat::TensorList{second, first}
                                                         : simaai::neat::TensorList{second});
        views.payload_type = simaai::neat::PayloadType::Tensor;
        views.media_type = "application/vnd.simaai.tensor";
        std::string view_error;
        retained_envelope = simaai::neat::pipeline_internal::sample_to_gst_envelope_holder(
            views, &view_error, /*allow_zero_copy=*/true);
        require(retained_envelope != nullptr, view_error);
        auto* view_sample = static_cast<GstSample*>(retained_envelope.get());
        GstBuffer* view_buffer = gst_sample_get_buffer(view_sample);
        require(gst_buffer_n_memory(view_buffer) == 1U &&
                    gst_buffer_peek_memory(view_buffer, 0U) == parent_memory &&
                    gst_dmabuf_memory_get_fd(gst_buffer_peek_memory(view_buffer, 0U)) == parent_fd,
                "rebinding shared views must preserve the original DMA memory");
        require(gst_buffer_get_parent_buffer_meta(view_buffer) != nullptr,
                "rebinding must retain the producer buffer lease");
        auto* meta = gst_buffer_get_custom_meta(view_buffer, SIMA_TENSOR_SET_META_NAME);
        require(meta != nullptr, "shared views lost tensor metadata");
        GstStructure* structure = gst_custom_meta_get_structure(meta);
        guint count = 0U;
        require(gst_structure_get_uint(structure, "physical-binding-count", &count) &&
                    count == views.tensors.size(),
                "shared views must publish one physical binding per requested view");
        const GValue* value =
            gst_structure_get_value(structure, SIMA_TENSOR_SET_META_FIELD_DESCRIPTORS);
        require(value && G_VALUE_HOLDS(value, G_TYPE_BYTES), "missing shared-view descriptors");
        gsize bytes = 0U;
        const auto* descriptors = static_cast<const SimaTensorDescriptorV2*>(
            g_bytes_get_data(static_cast<GBytes*>(g_value_get_boxed(value)), &bytes));
        require(bytes == count * sizeof(SimaTensorDescriptorV2),
                "shared-view descriptor count retained stale parent entries");
        for (guint i = 0U; i < count; ++i) {
          require(descriptors[i].logical_index == static_cast<int>(i) &&
                      descriptors[i].physical_index == static_cast<int>(i) &&
                      descriptors[i].backend_output_index == static_cast<int>(i) &&
                      descriptors[i].route_slot == static_cast<int>(i) &&
                      descriptors[i].memory_index == 0 &&
                      descriptors[i].byte_offset == views.tensors[i].byte_offset &&
                      descriptors[i].size_bytes ==
                          static_cast<guint64>(views.tensors[i].shape.front()),
                  "shared-view binding does not describe this frame's requested view");
        }
      }
      // The wrapper retains the producer GstBuffer, even once all Tensor views
      // and their shared GstSample holder have gone away.
      direct_map = {};
      direct = {};
      first = {};
      second = {};
      require(parent_lifetime.expired(), "fixture still retains its Tensor holder");
      auto* retained_sample = static_cast<GstSample*>(retained_envelope.get());
      GstBuffer* retained_buffer = gst_sample_get_buffer(retained_sample);
      require(gst_buffer_peek_memory(retained_buffer, 0U) == parent_memory &&
                  gst_buffer_get_parent_buffer_meta(retained_buffer) != nullptr,
              "retained envelope lost its DMA allocation or producer lease");
    }

    check_transfer_metadata_failure(mla_pref);

    std::cout << "[OK] tensor_device_placement_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

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

void cpu_tensor_set_fallback_preserves_dma_policy() {
  using namespace simaai::neat;
  const guint count = gst_buffer_get_max_memory() + 1U;
  TensorList tensors;
  tensors.reserve(count);
  for (guint i = 0U; i < count; ++i) {
    Tensor tensor = Tensor::from_vector(std::vector<std::uint8_t>{static_cast<std::uint8_t>(i)},
                                        {1}, TensorMemory::CPU);
    tensor.route.name = "field" + std::to_string(i);
    tensor.route.logical_index = static_cast<int>(i);
    tensor.route.physical_index = 0;
    tensor.route.segment_name = "packed_cpu";
    tensors.push_back(std::move(tensor));
  }
  Sample input = sample_from_tensors(tensors);
  std::string detail;
  const auto holder = pipeline_internal::sample_to_gst_envelope_holder(input, &detail);
  require(holder != nullptr, "copyable CPU packed-parent fallback failed: " + detail);
  GstBufferPtr buffer(pipeline_internal::buffer_from_tensor_holder(holder));
  require(buffer && gst_buffer_n_memory(buffer.get()) == 1U &&
              !pipeline_internal::buffer_has_dmabuf_memory(buffer.get()),
          "CPU packed-parent fallback must produce one ordinary memory");
  pipeline_internal::TensorBufferView descriptor;
  require(pipeline_internal::tensor_buffer_descriptor_from_sample(
              static_cast<GstSample*>(holder.get()), &descriptor, &detail) &&
              descriptor.tensors.size() == count,
          "CPU fallback lost its logical tensor descriptors: " + detail);
  GstMapInfo map{};
  require(gst_buffer_map(buffer.get(), &map, GST_MAP_READ), "CPU fallback payload map failed");
  bool correct = map.size == count;
  for (guint i = 0U; i < count; ++i) {
    const auto& tensor = descriptor.tensors[i];
    correct = correct && map.data[i] == static_cast<std::uint8_t>(i) && tensor.memory_index == 0 &&
              tensor.byte_offset == i && tensor.size_bytes == 1U &&
              tensor.logical_index == static_cast<int>(i);
  }
  gst_buffer_unmap(buffer.get(), &map);
  require(correct, "CPU fallback changed payload order or descriptor offsets");

  // One DMA sibling makes implicit packing forbidden. Reject before mapping
  // rather than relaxing the slot limit and letting GStreamer merge memories.
  GstBufferPtr dma(sima_test::make_bookkeeping_dmabuf(64U));
  GstSamplePtr dma_sample(gst_sample_new(dma.get(), nullptr, nullptr, nullptr));
  input.tensors.front().storage = pipeline_internal::make_gst_sample_storage(dma_sample.get());
  input.tensors.front().route.memory_index = 0;
  pipeline_internal::reset_tensor_io_stats();
  const auto mixed = pipeline_internal::sample_to_gst_envelope_holder(input, &detail);
  const auto stats = pipeline_internal::snapshot_tensor_io_stats();
  require(!mixed && detail.find("memory-slot limit") != std::string::npos,
          "oversized mixed tensor set must not silently pack its DMA sibling");
  require(stats.tensor_copy_count == 0U && stats.gst_memory_map_calls == 0U,
          "rejected mixed tensor set entered a copy or mapping fallback");
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

} // namespace

int main() {
  return sima_test::run_test("unit_output_transport_ownership_contract_test", [] {
    simaai::neat::gst_init_once();
    mixed_dmabuf_envelope_preserves_pool_loan_and_cpu_packing();
    cpu_tensor_set_fallback_preserves_dma_policy();
    dma_video_envelope_preserves_native_span_and_planes();
  });
}

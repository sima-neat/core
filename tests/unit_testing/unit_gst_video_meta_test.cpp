#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "pipeline/Graph.h"
#include "pipeline/TensorAdapters.h"
#include "pipeline/gst/InputStreamInternal.h"
#include "dmabuf_test_utils.h"
#include "pipeline/TensorCore.h"
#include "pipeline/internal/TensorUtil.h"
#include "nodes/io/Input.h"
#include "nodes/common/Output.h"

#include "test_utils.h"

#include <gst/gst.h>
#include <gst/video/video.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <thread>

namespace {

simaai::neat::Tensor make_i420_tensor(int w, int h) {
  const std::size_t y_size = static_cast<std::size_t>(w * h);
  const std::size_t u_size = static_cast<std::size_t>(w * h / 4);
  const std::size_t v_size = u_size;
  auto storage = simaai::neat::make_cpu_owned_storage(y_size + u_size + v_size);
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes > 0) {
    std::memset(map.data, 0, map.size_bytes);
  }

  simaai::neat::Tensor t;
  t.storage = storage;
  t.dtype = simaai::neat::TensorDType::UInt8;
  t.layout = simaai::neat::TensorLayout::HW;
  t.shape = {h, w};
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  t.semantic.image = simaai::neat::ImageSpec{simaai::neat::ImageSpec::PixelFormat::I420, ""};

  simaai::neat::Plane y;
  y.role = simaai::neat::PlaneRole::Y;
  y.shape = {h, w};
  y.strides_bytes = {w, 1};
  y.byte_offset = 0;

  simaai::neat::Plane u;
  u.role = simaai::neat::PlaneRole::U;
  u.shape = {h / 2, w / 2};
  u.strides_bytes = {w / 2, 1};
  u.byte_offset = static_cast<int64_t>(y_size);

  simaai::neat::Plane v;
  v.role = simaai::neat::PlaneRole::V;
  v.shape = {h / 2, w / 2};
  v.strides_bytes = {w / 2, 1};
  v.byte_offset = static_cast<int64_t>(y_size + u_size);

  t.planes = {y, u, v};
  return t;
}

simaai::neat::Tensor make_shared_nv12_holder_without_meta(int w, int h) {
  simaai::neat::Tensor tensor = make_nv12_tensor(w, h);
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, tensor.storage->size_bytes, nullptr);
  require(buffer != nullptr, "failed to allocate shared NV12 holder buffer");
  GstCaps* caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12", "width",
                                      G_TYPE_INT, w, "height", G_TYPE_INT, h, nullptr);
  require(caps != nullptr, "failed to allocate shared NV12 holder caps");
  GstSample* sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  gst_caps_unref(caps);
  gst_buffer_unref(buffer);
  require(sample != nullptr, "failed to allocate shared NV12 holder sample");
  tensor.storage = simaai::neat::pipeline_internal::make_gst_sample_storage(sample);
  gst_sample_unref(sample);
  require(tensor.storage != nullptr, "failed to wrap shared NV12 holder storage");
  return tensor;
}

simaai::neat::Tensor make_multimemory_nv12_holder_with_relative_offsets(int w, int h) {
  simaai::neat::Tensor tensor = make_nv12_tensor(w, h);
  const gsize y_size = static_cast<gsize>(w * h);
  const gsize uv_size = static_cast<gsize>(w * h / 2);
  GstBuffer* buffer = gst_buffer_new();
  require(buffer != nullptr, "failed to allocate multi-memory NV12 buffer");
  GstMemory* y_memory = gst_allocator_alloc(nullptr, y_size, nullptr);
  GstMemory* uv_memory = gst_allocator_alloc(nullptr, uv_size, nullptr);
  require(y_memory != nullptr && uv_memory != nullptr,
          "failed to allocate multi-memory NV12 planes");
  gst_buffer_append_memory(buffer, y_memory);
  gst_buffer_append_memory(buffer, uv_memory);

  gsize offsets[GST_VIDEO_MAX_PLANES] = {};
  gint strides[GST_VIDEO_MAX_PLANES] = {};
  strides[0] = w;
  strides[1] = w;
  GstVideoMeta* meta = gst_buffer_add_video_meta_full(buffer, GST_VIDEO_FRAME_FLAG_NONE,
                                                      GST_VIDEO_FORMAT_NV12, static_cast<guint>(w),
                                                      static_cast<guint>(h), 2U, offsets, strides);
  require(meta != nullptr, "failed to attach relative-offset multi-memory GstVideoMeta");

  GstCaps* caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12", "width",
                                      G_TYPE_INT, w, "height", G_TYPE_INT, h, nullptr);
  require(caps != nullptr, "failed to allocate multi-memory NV12 caps");
  GstSample* sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  gst_caps_unref(caps);
  gst_buffer_unref(buffer);
  require(sample != nullptr, "failed to allocate multi-memory NV12 sample");
  tensor.storage = simaai::neat::pipeline_internal::make_gst_sample_storage(sample);
  gst_sample_unref(sample);
  require(tensor.storage != nullptr, "failed to wrap multi-memory NV12 holder storage");
  return tensor;
}

void test_dmabuf_metadata_envelope_lifetime() {
  using namespace simaai::neat;
  namespace internal = pipeline_internal;
  GstBuffer* allocation = sima_test::allocate_cma_dmabuf(128U);
  GstBuffer* source = gst_buffer_new();
  GstMemory* view = gst_memory_share(gst_buffer_peek_memory(allocation, 0U), 16, 96);
  require(view != nullptr, "failed to create DMA metadata subview");
  gst_buffer_append_memory(source, view);
  // Preserve the allocator's original parent, not just the FD or shared memory.
  require(gst_buffer_add_parent_buffer_meta(source, allocation) != nullptr,
          "failed to retain DMA allocation parent");
  auto parent_released = std::make_shared<std::atomic<bool>>(false);
  gst_mini_object_weak_ref(
      GST_MINI_OBJECT_CAST(allocation),
      [](gpointer data, GstMiniObject*) {
        std::unique_ptr<std::shared_ptr<std::atomic<bool>>> released(
            static_cast<std::shared_ptr<std::atomic<bool>>*>(data));
        (*released)->store(true);
      },
      new std::shared_ptr<std::atomic<bool>>(parent_released));
  gst_buffer_unref(allocation);
  const auto expected_span = sima_test::dmabuf_span(source);
  gsize offsets[GST_VIDEO_MAX_PLANES] = {0U, 64U};
  gint strides[GST_VIDEO_MAX_PLANES] = {8, 8};
  require(gst_buffer_add_video_meta_full(source, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_NV12,
                                         8U, 8U, 2U, offsets, strides) != nullptr,
          "failed to attach authored DMA video layout");
  GstCaps* caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12", "width",
                                      G_TYPE_INT, 8, "height", G_TYPE_INT, 8, nullptr);
  GstSample* original = gst_sample_new(source, caps, nullptr, nullptr);
  gst_caps_unref(caps);
  gst_buffer_unref(source);
  Sample sample = sample_from_tensors(TensorList{from_gst_sample(original)});
  sample.input_seq = 23;
  auto gate = std::make_shared<internal::HolderLoanGate>(1);
  require(internal::attach_zero_copy_loan_to_sample(sample, gate), "metadata fixture loan failed");
  InputStream::State state;
  PreprocessRuntimeMeta meta;
  meta.original_width = 8;
  meta.original_height = 8;
  meta.resized_width = 8;
  meta.resized_height = 8;
  state.preprocess_meta_by_input_seq.emplace(23, meta);
  // The original is deliberately still shared and lacks preprocess GstMeta.
  maybe_restore_cached_preprocess_meta_on_sample(state, &sample);
  require(sample.tensors.size() == 1U && sample.tensors.front().semantic.preprocess.has_value(),
          "preprocess semantic restoration failed");
  GstBuffer* envelope = internal::buffer_from_tensor_holder(sample.tensors.front().storage->holder);
  require(envelope != nullptr && envelope != gst_sample_get_buffer(original),
          "non-writable DMA parent needs a distinct metadata envelope");
  require(sima_test::dmabuf_span(envelope) == expected_span,
          "metadata restoration changed DMA backing, root offset or span");
  GstVideoMeta* video = gst_buffer_get_video_meta(envelope);
  require(video && video->n_planes == 2U && video->offset[0] == 0U && video->offset[1] == 64U &&
              video->stride[0] == 8 && video->stride[1] == 8,
          "metadata envelope changed authored video offsets/strides");
  require(has_simaai_preprocess_meta(envelope), "DMA envelope lacks restored preprocess GstMeta");
  require(gst_buffer_get_parent_buffer_meta(envelope) != nullptr,
          "metadata envelope lost decoder pool parent ownership");
  gst_sample_unref(original);
  sample = {};
  require(!parent_released->load() && gate->inflight() == 1,
          "metadata envelope released original pool parent/loan before its consumer");
  gst_buffer_unref(envelope);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!parent_released->load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  require(parent_released->load(),
          "metadata envelope retained its pool parent after final release");
  require(gate->inflight() == 0 && gate->released() == 1U,
          "metadata envelope did not release its output loan exactly once");
}

void test_mixed_dmabuf_metadata_selected_memory() {
  using namespace simaai::neat;
  namespace internal = pipeline_internal;
  GstBuffer* dma = sima_test::allocate_cma_dmabuf(64U);
  const auto expected_span = sima_test::dmabuf_span(dma);
  GstCaps* caps = gst_caps_new_simple(
      "application/vnd.simaai.tensor", "width", G_TYPE_INT, 8, "height", G_TYPE_INT, 8, "depth",
      G_TYPE_INT, 1, "dtype", G_TYPE_STRING, "UINT8", "layout", G_TYPE_STRING, "HW", nullptr);
  {
    GstSample* init = gst_sample_new(dma, caps, nullptr, nullptr);
    auto storage = internal::make_gst_sample_storage(init);
    gst_sample_unref(init);
    const Mapping write = storage->map(MapMode::Write);
    require(write.data && write.size_bytes >= 64U, "mixed metadata DMA initialization failed");
    std::memset(write.data, 0x6D, write.size_bytes);
  }
  GstBuffer* mixed = gst_buffer_new_allocate(nullptr, 64U, nullptr);
  gst_buffer_memset(mixed, 0U, 0xA1, 64U);
  gst_buffer_append_memory(mixed, gst_memory_ref(gst_buffer_peek_memory(dma, 0U)));
  require(gst_buffer_add_parent_buffer_meta(mixed, dma) != nullptr,
          "mixed metadata fixture did not retain allocation parent");
  gst_buffer_unref(dma);
  GstSample* original = gst_sample_new(mixed, caps, nullptr, nullptr);
  gst_caps_unref(caps);
  gst_buffer_unref(mixed);
  Tensor tensor = internal::tensor_view_from_sample_memory(from_gst_sample(original), 1, true);
  tensor.byte_offset = 8;
  tensor.shape = {4, 8};
  tensor.strides_bytes = {8, 1};
  Sample sample = sample_from_tensors(TensorList{tensor});
  sample.input_seq = 25;
  InputStream::State state;
  PreprocessRuntimeMeta meta;
  meta.original_width = 8;
  meta.original_height = 8;
  state.preprocess_meta_by_input_seq.emplace(25, meta);
  maybe_restore_cached_preprocess_meta_on_sample(state, &sample);
  const Tensor& restored = sample.tensors.front();
  require(internal::tensor_has_dmabuf_memory(restored) && restored.route.memory_index == 1 &&
              restored.byte_offset == 8,
          "metadata restoration lost selected DMA memory or logical offset");
  const Mapping mapped = restored.map_read();
  require(mapped.data && mapped.size_bytes == 56U &&
              static_cast<const uint8_t*>(mapped.data)[0] == 0x6D,
          "metadata restoration mapped or merged the CPU carrier instead of selected DMA");
  GstBuffer* envelope = internal::buffer_from_tensor_holder(restored.storage->holder);
  require(envelope && gst_buffer_n_memory(envelope) == 2U &&
              sima_test::dmabuf_span(gst_buffer_peek_memory(envelope, 1U)) == expected_span,
          "mapping restored metadata changed the mixed carrier's DMA allocation");
  gst_buffer_unref(envelope);
  gst_sample_unref(original);
}

} // namespace

int main() {
  try {
    using namespace simaai::neat;
    setenv("SIMA_INPUTSTREAM_RESTORE_PREPROC_BUFFER_META", "1", 1);

    const int w = 4;
    const int h = 4;
    {
      simaai::neat::Tensor input = make_nv12_tensor(w, h);

      Graph p;
      InputOptions src_opt;
      src_opt.payload_type = simaai::neat::PayloadType::Image;
      src_opt.format = simaai::neat::FormatTag::NV12;
      src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
      p.add(nodes::Input(src_opt));
      p.add(nodes::Output(OutputOptions::Latest()));

      RunOptions opt;
      Run run = p.build(TensorList{input}, opt);

      TensorList outs = run.run(TensorList{input}, 1000);
      require(outs.size() == 1, "output missing simaai::neat::Tensor");
      require(outs.front().storage != nullptr, "output storage missing");
      require(outs.front().storage->holder != nullptr, "output holder missing");

      auto* sample = static_cast<GstSample*>(outs.front().storage->holder.get());
      require(sample != nullptr, "output holder is not a GstSample");
      GstBuffer* buf = gst_sample_get_buffer(sample);
      require(buf != nullptr, "output buffer missing");

      GstVideoMeta* meta = gst_buffer_get_video_meta(buf);
      require(meta != nullptr, "missing GstVideoMeta");
      require(meta->format == GST_VIDEO_FORMAT_NV12, "video format mismatch");
      require(meta->width == static_cast<guint>(w), "video width mismatch");
      require(meta->height == static_cast<guint>(h), "video height mismatch");
      require(meta->n_planes == 2, "NV12 plane count mismatch");

      const std::size_t y_size = static_cast<std::size_t>(w * h);
      const std::size_t total_bytes = y_size + static_cast<std::size_t>(w * h / 2);
      require(meta->offset[0] == 0, "Y plane offset mismatch");
      require(meta->stride[0] == w, "Y plane stride mismatch");
      require(meta->offset[1] == y_size, "UV plane offset mismatch");
      require(meta->stride[1] == w, "UV plane stride mismatch");
      require(gst_buffer_get_size(buf) == total_bytes, "buffer size mismatch");
    }

    {
      simaai::neat::Tensor input = make_i420_tensor(w, h);

      Graph p;
      InputOptions src_opt;
      src_opt.payload_type = simaai::neat::PayloadType::Image;
      src_opt.format = simaai::neat::FormatTag::I420;
      src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
      p.add(nodes::Input(src_opt));
      p.add(nodes::Output(OutputOptions::Latest()));

      RunOptions opt;
      Run run = p.build(TensorList{input}, opt);

      TensorList outs = run.run(TensorList{input}, 1000);
      require(outs.size() == 1, "output missing simaai::neat::Tensor");
      require(outs.front().storage != nullptr, "output storage missing");
      require(outs.front().storage->holder != nullptr, "output holder missing");

      auto* sample = static_cast<GstSample*>(outs.front().storage->holder.get());
      require(sample != nullptr, "output holder is not a GstSample");
      GstBuffer* buf = gst_sample_get_buffer(sample);
      require(buf != nullptr, "output buffer missing");

      GstVideoMeta* meta = gst_buffer_get_video_meta(buf);
      require(meta != nullptr, "missing GstVideoMeta");
      require(meta->format == GST_VIDEO_FORMAT_I420, "video format mismatch");
      require(meta->width == static_cast<guint>(w), "video width mismatch");
      require(meta->height == static_cast<guint>(h), "video height mismatch");
      require(meta->n_planes == 3, "I420 plane count mismatch");

      const std::size_t y_size = static_cast<std::size_t>(w * h);
      const std::size_t u_size = static_cast<std::size_t>(w * h / 4);
      const std::size_t v_size = u_size;
      const std::size_t total_bytes = y_size + u_size + v_size;
      require(meta->offset[0] == 0, "Y plane offset mismatch");
      require(meta->stride[0] == w, "Y plane stride mismatch");
      require(meta->offset[1] == y_size, "U plane offset mismatch");
      require(meta->stride[1] == w / 2, "U plane stride mismatch");
      require(meta->offset[2] == y_size + u_size, "V plane offset mismatch");
      require(meta->stride[2] == w / 2, "V plane stride mismatch");
      require(gst_buffer_get_size(buf) == total_bytes, "buffer size mismatch");
    }

    {
      simaai::neat::Tensor input = make_shared_nv12_holder_without_meta(w, h);
      GstBuffer* input_buffer =
          simaai::neat::pipeline_internal::buffer_from_tensor_holder(input.storage->holder);
      require(input_buffer != nullptr, "shared input holder missing GstBuffer");
      require(gst_buffer_get_video_meta(input_buffer) == nullptr,
              "shared input holder should start without GstVideoMeta");
      GstMemory* input_memory = gst_buffer_peek_memory(input_buffer, 0);
      require(input_memory != nullptr, "shared input holder missing GstMemory");

      Graph graph;
      InputOptions src_opt;
      src_opt.payload_type = PayloadType::Image;
      src_opt.format = FormatTag::NV12;
      src_opt.memory_policy = InputMemoryPolicy::SystemMemory;
      src_opt.max_width = w;
      src_opt.max_height = h;
      src_opt.max_depth = 3;
      graph.add(nodes::Input(src_opt));
      graph.add(nodes::Output(OutputOptions::EveryFrame(4)));

      RunOptions run_opt;
      run_opt.output_memory = OutputMemory::ZeroCopy;
      run_opt.advanced.copy_input = false;
      Run run = graph.build(TensorList{input}, run_opt);
      Sample sample = sample_from_tensors(TensorList{input});
      sample.stream_id = "shared-nv12";
      sample.frame_id = 17;
      require(run.push(sample), "shared NV12 holder push failed");
      const std::optional<Sample> output = run.pull(1000);
      require(output.has_value(), "shared NV12 holder output missing");
      const TensorList output_tensors = tensors_from_sample(*output, true);
      require(output_tensors.size() == 1U && output_tensors.front().storage != nullptr &&
                  output_tensors.front().storage->holder != nullptr,
              "shared NV12 holder output storage missing");
      GstBuffer* output_buffer = simaai::neat::pipeline_internal::buffer_from_tensor_holder(
          output_tensors.front().storage->holder);
      require(output_buffer != nullptr, "shared NV12 output holder missing GstBuffer");
      require(gst_buffer_peek_memory(output_buffer, 0) == input_memory,
              "metadata envelope should preserve shared holder payload memory");
      GstVideoMeta* meta = gst_buffer_get_video_meta(output_buffer);
      require(meta != nullptr, "writable shared-holder envelope lost GstVideoMeta");
      require(meta->format == GST_VIDEO_FORMAT_NV12, "shared-holder video format mismatch");
      require(meta->width == static_cast<guint>(w), "shared-holder video width mismatch");
      require(meta->height == static_cast<guint>(h), "shared-holder video height mismatch");
      require(meta->n_planes == 2, "shared-holder NV12 plane count mismatch");
      gst_buffer_unref(output_buffer);
      gst_buffer_unref(input_buffer);
      run.close();
    }

    {
      simaai::neat::Tensor input = make_multimemory_nv12_holder_with_relative_offsets(w, h);
      GstBuffer* input_buffer =
          simaai::neat::pipeline_internal::buffer_from_tensor_holder(input.storage->holder);
      require(input_buffer != nullptr && gst_buffer_n_memory(input_buffer) == 2U,
              "multi-memory input holder layout mismatch");
      GstMemory* y_memory = gst_buffer_peek_memory(input_buffer, 0);
      GstMemory* uv_memory = gst_buffer_peek_memory(input_buffer, 1);

      Graph graph;
      InputOptions src_opt;
      src_opt.payload_type = PayloadType::Image;
      src_opt.format = FormatTag::NV12;
      src_opt.memory_policy = InputMemoryPolicy::SystemMemory;
      src_opt.max_width = w;
      src_opt.max_height = h;
      src_opt.max_depth = 3;
      graph.add(nodes::Input(src_opt));
      graph.add(nodes::Output(OutputOptions::EveryFrame(4)));

      RunOptions run_opt;
      run_opt.output_memory = OutputMemory::ZeroCopy;
      run_opt.advanced.copy_input = false;
      Run run = graph.build(TensorList{input}, run_opt);
      require(run.try_push_holder(input.storage->holder),
              "multi-memory relative-offset holder push failed");
      const std::optional<Sample> output = run.pull(1000);
      require(output.has_value(), "multi-memory relative-offset holder output missing");
      const TensorList output_tensors = tensors_from_sample(*output, true);
      require(output_tensors.size() == 1U && output_tensors.front().storage != nullptr &&
                  output_tensors.front().storage->holder != nullptr,
              "multi-memory relative-offset output holder missing");
      GstBuffer* output_buffer = simaai::neat::pipeline_internal::buffer_from_tensor_holder(
          output_tensors.front().storage->holder);
      require(output_buffer != nullptr && gst_buffer_n_memory(output_buffer) == 2U,
              "multi-memory metadata envelope lost plane memories");
      require(gst_buffer_peek_memory(output_buffer, 0) == y_memory &&
                  gst_buffer_peek_memory(output_buffer, 1) == uv_memory,
              "multi-memory metadata envelope copied payload memories");
      GstVideoMeta* meta = gst_buffer_get_video_meta(output_buffer);
      require(meta != nullptr, "multi-memory metadata envelope lost GstVideoMeta");
      require(meta->n_planes == 2U && meta->offset[0] == 0U && meta->offset[1] == 0U,
              "multi-memory relative plane offsets were not preserved");
      require(meta->stride[0] == w && meta->stride[1] == w,
              "multi-memory plane strides were not preserved");
      gst_buffer_unref(output_buffer);
      gst_buffer_unref(input_buffer);
      run.close();
    }

    test_dmabuf_metadata_envelope_lifetime();
    test_mixed_dmabuf_metadata_selected_memory();

    std::cout << "[OK] unit_gst_video_meta_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "pipeline/Graph.h"
#include "pipeline/TensorCore.h"
#include "pipeline/internal/TensorUtil.h"
#include "nodes/io/Input.h"
#include "nodes/common/Output.h"

#include "test_utils.h"

#include <gst/gst.h>
#include <gst/video/video.h>

#include <chrono>
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

} // namespace

int main() {
  try {
    using namespace simaai::neat;

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

    std::cout << "[OK] unit_gst_video_meta_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

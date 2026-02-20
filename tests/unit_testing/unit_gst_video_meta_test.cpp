#include "pipeline/Session.h"
#include "pipeline/TensorCore.h"
#include "nodes/io/Input.h"
#include "nodes/common/Output.h"

#include "test_utils.h"

#include <gst/gst.h>
#include <gst/video/video.h>

#include <chrono>
#include <iostream>
#include <cstring>
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

} // namespace

int main() {
  try {
    using namespace simaai::neat;

    const int w = 4;
    const int h = 4;
    {
      simaai::neat::Tensor input = make_nv12_tensor(w, h);

      Session p;
      InputOptions src_opt;
      src_opt.media_type = "video/x-raw";
      src_opt.format = "NV12";
      src_opt.use_simaai_pool = false;
      p.add(nodes::Input(src_opt));
      p.add(nodes::Output(OutputOptions::Latest()));

      RunOptions opt;
      Run run = p.build(input, RunMode::Async, opt);

      Sample out = run.push_and_pull(input, 1000);
      require(out.tensor.has_value(), "output missing simaai::neat::Tensor");
      require(out.tensor->storage != nullptr, "output storage missing");
      require(out.tensor->storage->holder != nullptr, "output holder missing");

      auto* sample = static_cast<GstSample*>(out.tensor->storage->holder.get());
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

      Session p;
      InputOptions src_opt;
      src_opt.media_type = "video/x-raw";
      src_opt.format = "I420";
      src_opt.use_simaai_pool = false;
      p.add(nodes::Input(src_opt));
      p.add(nodes::Output(OutputOptions::Latest()));

      RunOptions opt;
      Run run = p.build(input, RunMode::Async, opt);

      Sample out = run.push_and_pull(input, 1000);
      require(out.tensor.has_value(), "output missing simaai::neat::Tensor");
      require(out.tensor->storage != nullptr, "output storage missing");
      require(out.tensor->storage->holder != nullptr, "output holder missing");

      auto* sample = static_cast<GstSample*>(out.tensor->storage->holder.get());
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

    std::cout << "[OK] unit_gst_video_meta_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

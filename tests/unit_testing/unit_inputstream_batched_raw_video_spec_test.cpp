#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL
#endif
#include "pipeline/internal/InputStreamUtil.h"
#include "nodes/io/Input.h"
#include "pipeline/TensorCore.h"
#include "test_main.h"
#include "test_utils.h"

#include <exception>
#include <string>
#include <vector>

namespace {

simaai::neat::Tensor make_batched_bgr_tensor(int n, int h, int w) {
  const int c = 3;
  const std::size_t row_bytes = static_cast<std::size_t>(w * c);
  const std::size_t frame_bytes = row_bytes * static_cast<std::size_t>(h);
  auto storage = simaai::neat::make_cpu_owned_storage(frame_bytes * static_cast<std::size_t>(n));

  simaai::neat::Tensor tensor;
  tensor.storage = storage;
  tensor.dtype = simaai::neat::TensorDType::UInt8;
  tensor.layout = simaai::neat::TensorLayout::HWC;
  tensor.shape = {n, h, w, c};
  tensor.strides_bytes = {static_cast<int64_t>(frame_bytes), static_cast<int64_t>(row_bytes), c, 1};
  tensor.axis_semantics = {simaai::neat::TensorAxisSemantic::N, simaai::neat::TensorAxisSemantic::H,
                           simaai::neat::TensorAxisSemantic::W,
                           simaai::neat::TensorAxisSemantic::C};
  tensor.device = {simaai::neat::DeviceType::CPU, 0};
  tensor.semantic.image = simaai::neat::ImageSpec{simaai::neat::ImageSpec::PixelFormat::BGR, ""};
  tensor.read_only = true;
  return tensor;
}

} // namespace

RUN_TEST("unit_inputstream_batched_raw_video_spec_test", ([] {
           simaai::neat::InputOptions opt;
           opt.payload_type = simaai::neat::PayloadType::Image;
           opt.format = "BGR";

           const int n = 2;
           const int h = 3;
           const int w = 4;
           simaai::neat::Tensor tensor = make_batched_bgr_tensor(n, h, w);

           simaai::neat::SampleSpec spec = simaai::neat::derive_tensor_spec_or_throw(
               tensor, opt, "unit_inputstream_batched_raw_video_spec_test");
           require(spec.kind == simaai::neat::SampleMediaKind::RawVideo,
                   "batched spec kind mismatch");
           require(spec.format == "BGR", "batched spec format mismatch");
           require(spec.width == w && spec.height == h && spec.depth == 3,
                   "batched spec dims mismatch");
           require(spec.required_bytes_actual == static_cast<std::size_t>(n * h * w * 3),
                   "batched spec byte size mismatch");
           require(spec.planes.size() == 1U, "batched spec plane count mismatch");
           require(spec.planes.front().stride_bytes == w * 3, "batched spec stride mismatch");
           require(spec.planes.front().size_bytes == static_cast<std::size_t>(h * w * 3),
                   "batched spec video-meta frame size mismatch");

           bool threw = false;
           tensor.strides_bytes[0] += 1;
           try {
             (void)simaai::neat::derive_tensor_spec_or_throw(
                 tensor, opt, "unit_inputstream_batched_raw_video_spec_test_bad_stride");
           } catch (const std::exception& e) {
             threw = true;
             require(std::string(e.what()).find("tightly contiguous") != std::string::npos,
                     std::string("unexpected batched stride rejection: ") + e.what());
           }
           require(threw, "expected non-tight batched video tensor to be rejected");
         }));

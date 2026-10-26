// groups::MjpegDecode on hardware: the same solid-colour frame encoded as 4:2:2 and as 4:2:0
// MJPEG must both come out as NV12 with matching pixels. 4:2:2 decodes zero-copy to NV16 and
// is converted on the EV74; 4:2:0 decodes straight to NV12.
#include <neat.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace simaai::neat;
constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr int kFrames = 3;

void require(bool ok, const std::string& message) {
  if (!ok) {
    throw std::runtime_error(message);
  }
}

struct Means {
  double y = 0, u = 0, v = 0;
  int frames = 0;
};

Means decode(const std::string& raw_format, FormatTag source_format) {
  Graph graph;
  // ARGB 0xff3070c0: a blue-ish colour, so both chroma planes are far from neutral.
  graph.custom("videotestsrc num-buffers=" + std::to_string(kFrames) +
                   " pattern=solid-color foreground-color=0xff3070c0 ! video/x-raw,format=" +
                   raw_format + ",width=" + std::to_string(kWidth) +
                   ",height=" + std::to_string(kHeight) + ",framerate=30/1 ! jpegenc quality=95",
               InputRole::Source);
  graph.add(nodes::JpegParse());
  SimaDecodeOptions decode;
  decode.type = SimaDecodeType::MJPEG;
  decode.raw_output = true;
  decode.dec_width = kWidth;
  decode.dec_height = kHeight;
  decode.dec_fps = 30;
  graph.add(nodes::groups::MjpegDecode(decode, source_format));
  graph.add(nodes::Output(OutputOptions::EveryFrame(kFrames)));

  Means means;
  graph.set_tensor_callback([&](const Tensor& tensor) {
    require(tensor.storage && tensor.storage->kind == StorageKind::GstSample &&
                tensor.storage->holder,
            raw_format + ": output lost its zero-copy GstSample");
    auto* buffer = gst_sample_get_buffer(static_cast<GstSample*>(tensor.storage->holder.get()));
    const auto* meta = gst_buffer_get_video_meta(buffer);
    require(meta && meta->format == GST_VIDEO_FORMAT_NV12 && meta->width == unsigned(kWidth) &&
                meta->height == unsigned(kHeight),
            raw_format + ": MjpegDecode output must be NV12 640x480");
    const auto map = tensor.storage->map(MapMode::Read);
    require(map.data != nullptr, raw_format + ": NV12 output map failed");
    const auto* bytes = static_cast<const std::uint8_t*>(map.data);
    double y = 0, u = 0, v = 0;
    for (int row = 0; row < kHeight; ++row) {
      for (int col = 0; col < kWidth; ++col) {
        y += bytes[meta->offset[0] + std::size_t(row) * meta->stride[0] + col];
      }
    }
    for (int row = 0; row < kHeight / 2; ++row) {
      for (int col = 0; col < kWidth; col += 2) {
        const auto* uv = bytes + meta->offset[1] + std::size_t(row) * meta->stride[1] + col;
        u += uv[0];
        v += uv[1];
      }
    }
    const double chroma = double(kWidth / 2) * (kHeight / 2);
    means.y += y / (double(kWidth) * kHeight);
    means.u += u / chroma;
    means.v += v / chroma;
    ++means.frames;
    return true;
  });
  graph.run();
  require(means.frames == kFrames, raw_format + ": expected " + std::to_string(kFrames) +
                                       " decoded frames, got " + std::to_string(means.frames));
  means.y /= kFrames;
  means.u /= kFrames;
  means.v /= kFrames;
  return means;
}

} // namespace

int main() {
  try {
    const Means yuv422 = decode("Y42B", FormatTag::NV16);
    const Means yuv420 = decode("I420", FormatTag::NV12);
    std::cout << "4:2:2 Y/U/V=" << yuv422.y << "/" << yuv422.u << "/" << yuv422.v
              << "  4:2:0 Y/U/V=" << yuv420.y << "/" << yuv420.u << "/" << yuv420.v << "\n";
    require(std::abs(yuv420.u - 128) > 10 && std::abs(yuv420.v - 128) > 10,
            "test colour must have non-neutral chroma");
    // A solid colour has constant chroma, so halving the chroma rows keeps it exactly; allow
    // JPEG rounding only.
    require(std::abs(yuv422.y - yuv420.y) <= 2 && std::abs(yuv422.u - yuv420.u) <= 2 &&
                std::abs(yuv422.v - yuv420.v) <= 2,
            "4:2:2 NV12 output differs from the 4:2:0 reference");
    std::cout << "[OK] mjpeg_decode_group_e2e_test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << "\n";
    return 1;
  }
}

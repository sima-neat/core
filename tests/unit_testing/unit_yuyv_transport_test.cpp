#include "pipeline/Tensor.h"
#include "pipeline/TensorAdapters.h"
#include "pipeline/FormatSpec.h"
#include "nodes/io/UsbCameraInput.h"
#include "nodes/sima/SimaEncode.h"
#include <gst/video/video.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>

using namespace simaai::neat;
namespace {
void require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
void packed_view() {
  constexpr int width = 6, height = 3, pitch = 20, offset = 7;
  constexpr int extent = offset + (height - 1) * pitch + width * 2;
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, extent, nullptr);
  gsize offsets[GST_VIDEO_MAX_PLANES] = {offset};
  gint strides[GST_VIDEO_MAX_PLANES] = {pitch};
  require(gst_buffer_add_video_meta_full(buffer, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_YUY2,
                                         width, height, 1, offsets, strides),
          "video meta");
  GstCaps* caps = gst_caps_from_string("video/x-raw,format=YUY2,width=6,height=3,framerate=30000/"
                                       "1001,colorimetry=bt709,chroma-site=jpeg");
  GstSample* sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  Tensor tensor = from_gst_sample(sample);
  gst_sample_unref(sample);
  gst_caps_unref(caps);
  gst_buffer_unref(buffer);
  require(tensor.semantic.image && tensor.semantic.image->format == ImageType::YUYV,
          "packed semantic format");
  require(tensor.shape == std::vector<int64_t>{height, width, 2}, "storage shape");
  require(tensor.strides_bytes == std::vector<int64_t>{pitch, 2, 1}, "negotiated stride");
  require(tensor.byte_offset == offset && tensor.storage->size_bytes == extent,
          "preserve offset and short final padding");
  require(tensor.storage->holder && tensor.storage->data == nullptr,
          "retained owner without eager CPU mapping");
  require(tensor.semantic.image->color_space == "bt709", "preserve color metadata");
  std::string error;
  require(tensor.validate(&error), error.c_str());
  tensor.shape[1] = 5;
  require(!tensor.validate(&error), "odd width must fail");
  tensor.shape[1] = width;
  tensor.strides_bytes[0] = 4;
  require(!tensor.validate(&error), "overlapping rows must fail");
}
} // namespace
int main(int argc, char** argv) {
  gst_init(&argc, &argv);
  try {
    // Old function-pointer forms remain unambiguous with explicit signatures.
    [[maybe_unused]] auto old_source =
        static_cast<std::shared_ptr<Node> (*)(UsbCameraInputOptions)>(&nodes::UsbCameraInput);
    [[maybe_unused]] auto old_info =
        static_cast<UsbCameraInfo (*)(const UsbCameraInputOptions&)>(&usb_camera_info);
    [[maybe_unused]] auto new_source =
        static_cast<std::shared_ptr<Node> (*)(UsbCameraInputOptions, UsbCameraFormat)>(
            &nodes::UsbCameraInput);
    require(format_tag_from_string("YUY2") == FormatTag::YUYV, "GStreamer alias");
    SimaEncodeOptions encode;
    const auto strict = nodes::SimaEncode(encode, EncoderInputPolicy::RequireDirect);
    require(strict->backend_fragment(0).find("require-direct=true") != std::string::npos,
            "strict encoder property");
    require(nodes::SimaEncode(encode)->backend_fragment(0).find("require-direct=true") ==
                std::string::npos,
            "legacy encoder policy");
    require(strict->backend_fragment(0).find("signal-input-color=true") != std::string::npos,
            "strict encoder must signal the negotiated color");
    encode.type = SimaEncodeType::H265;
    bool rejected = false;
    try {
      (void)nodes::SimaEncode(encode, EncoderInputPolicy::RequireDirect);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "unqualified strict codec must fail");
    packed_view();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "PASS YUYV packed transport and additive API contracts\n";
}

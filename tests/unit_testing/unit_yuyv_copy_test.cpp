#include "pipeline/FormatSpec.h"
#include "pipeline/internal/GstDataAdapter.h"
#include <gst/video/video.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace simaai::neat;
namespace {
void require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
void copies() {
  constexpr std::size_t span = 52; // Three 12-byte rows with 20-byte pitch.
  for (const std::size_t offset : {0U, 7U}) {
    for (const std::size_t spare : {0U, 21U}) {
      std::vector<uint8_t> source(offset + span + spare);
      std::iota(source.begin(), source.end(), uint8_t{0});
      Tensor tensor;
      tensor.shape = {3, 6, 2};
      tensor.strides_bytes = {20, 2, 1};
      tensor.byte_offset = offset;
      tensor.dtype = TensorDType::UInt8;
      tensor.layout = TensorLayout::HWC;
      // Synthetic host mapping obeying Tensor::map's view-relative contract.
      Mapping mapping;
      mapping.data = source.data() + offset;
      mapping.size_bytes = source.size() - offset;
      for (const auto bytes : {offset + span, source.size()}) {
        std::vector<uint8_t> destination(bytes + 1, 0xff);
        std::string error;
        const bool copied = pipeline_internal::copy_video_payload_to(
            tensor, mapping, destination.data(), bytes, &error);
        require(copied, error.c_str());
        require(std::all_of(destination.begin(), destination.begin() + offset,
                            [](auto byte) { return byte == 0; }),
                "carrier prefix must be initialized");
        require(std::equal(source.begin() + offset, source.begin() + bytes,
                           destination.begin() + offset),
                "copied rows/padding must retain carrier-relative positions");
        require(destination.back() == 0xff, "copy must respect destination bounds");
      }
      std::vector<uint8_t> packed(span);
      std::string error;
      require(
          pipeline_internal::copy_tensor_payload_to(tensor, mapping, packed.data(), span, &error) &&
              std::equal(packed.begin(), packed.end(), source.begin() + offset),
          "generic tensor copy must remain view-relative");
      std::vector<uint8_t> rejected(source.size() + 1, 0xff);
      require(
          !pipeline_internal::copy_video_payload_to(tensor, mapping, rejected.data(),
                                                    rejected.size(), &error) &&
              std::all_of(rejected.begin(), rejected.end(), [](auto byte) { return byte == 0xff; }),
          "oversized copy must fail before writing");
      tensor.byte_offset = -1;
      require(
          !pipeline_internal::copy_video_payload_to(tensor, mapping, packed.data(), span, &error),
          "negative offsets must fail");
      tensor.byte_offset = span + 1;
      require(
          !pipeline_internal::copy_video_payload_to(tensor, mapping, packed.data(), span, &error),
          "offsets beyond destination must fail");
    }
  }
}
void aliases() {
  for (const auto* alias : {"YUYV", "YUY2", "yuyv", "yuy2"}) {
    const auto normalized = normalize_caps_format_for_media("video/x-raw", alias);
    require(normalized == "YUY2", "raw caps must canonicalize packed aliases");
    const auto text = "video/x-raw,format=" + normalized + ",width=6,height=4";
    auto* caps = gst_caps_from_string(text.c_str());
    GstVideoInfo info{};
    require(gst_video_info_from_caps(&info, caps) &&
                GST_VIDEO_INFO_FORMAT(&info) == GST_VIDEO_FORMAT_YUY2,
            "canonical packed caps must be understood by GStreamer");
    gst_caps_unref(caps);
  }
  require(FormatSpec("YUY2").str() == "YUYV", "public enum spelling must remain compatible");
  require(normalize_caps_format_for_media("image/jpeg", "YUYV") == "YUYV" &&
              normalize_caps_format_for_media("video/x-raw", "NV12") == "NV12",
          "normalization must preserve unrelated media and formats");
}
} // namespace
int main() {
  gst_init(nullptr, nullptr);
  try {
    copies();
    aliases();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "PASS packed copy placement, bounds, generic tensor semantics and raw aliases\n";
}

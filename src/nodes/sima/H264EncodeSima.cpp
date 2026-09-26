#include "nodes/sima/H264EncodeSima.h"

#include "gst/GstHelpers.h"
#include "nodes/sima/internal/SimaEncode.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class H264EncodeSWNode final : public simaai::neat::Node {
public:
  explicit H264EncodeSWNode(int bitrate_kbps) : bitrate_kbps_(bitrate_kbps) {}
  std::string kind() const override {
    return "H264EncodeSW";
  }
  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Static;
  }

  std::string backend_fragment(int node_index) const override {
    const std::string element_name = "n" + std::to_string(node_index) + "_swenc";
    std::string factory;
    std::string input_adapter;
    std::string props;

    if (simaai::neat::element_exists("x264enc")) {
      factory = "x264enc";
      int bitrate = bitrate_kbps_;
      if (const char* env_bitrate = std::getenv("SIMA_H264ENC_BITRATE_KBPS")) {
        const int v = std::atoi(env_bitrate);
        if (v >= 0)
          bitrate = v;
      }
      props = "tune=zerolatency speed-preset=ultrafast "
              "key-int-max=1 bframes=0 "
              "bitrate=" +
              std::to_string(bitrate) +
              " "
              "byte-stream=true";
      if (const char* lossless = std::getenv("SIMA_H264ENC_LOSSLESS")) {
        if (std::string(lossless) != "0") {
          props += " qp=0";
        }
      }
      if (const char* qp = std::getenv("SIMA_H264ENC_QP")) {
        if (*qp) {
          props += " qp=" + std::string(qp);
        }
      }
    } else if (simaai::neat::element_exists("openh264enc")) {
      factory = "openh264enc";
      input_adapter = "videoconvert name=" + element_name +
                      "_convert ! capsfilter name=" + element_name +
                      "_i420 caps=\"video/x-raw,format=I420\" ! ";
      props = "";
    } else if (simaai::neat::element_exists("avenc_h264")) {
      factory = "avenc_h264";
      props = "";
    } else {
      throw std::runtime_error(
          "H264EncodeSW: no software H264 encoder found. Install one of: "
          "x264enc (gst-plugins-ugly), openh264enc (gst-plugins-bad), avenc_h264 (gst-libav).");
    }

    std::ostringstream ss;
    ss << input_adapter << factory << " name=" << element_name;
    if (!props.empty())
      ss << " " << props;
    return ss.str();
  }

  std::vector<std::string> element_names(int node_index) const override {
    const std::string element_name = "n" + std::to_string(node_index) + "_swenc";
    if (!simaai::neat::element_exists("x264enc") && simaai::neat::element_exists("openh264enc")) {
      return {element_name + "_convert", element_name + "_i420", element_name};
    }
    return {element_name};
  }

private:
  int bitrate_kbps_ = 4000;
};

} // namespace

namespace simaai::neat {

H264EncodeSima::H264EncodeSima(int w, int h, int fps, int bitrate_kbps, std::string profile,
                               std::string level)
    : options_{.width = w,
               .height = h,
               .fps = fps,
               .bitrate_kbps = bitrate_kbps,
               .profile = std::move(profile),
               .level = std::move(level)} {}

std::string H264EncodeSima::backend_fragment(int node_index) const {
  return internal::encoder_fragment(options_, node_index);
}

std::vector<std::string> H264EncodeSima::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_encoder"};
}

OutputSpec H264EncodeSima::output_spec(const OutputSpec& /*input*/) const {
  OutputSpec out;
  out.media_type = "video/x-h264";
  out.format = "H264";
  out.certainty = SpecCertainty::Hint;
  out.note = "H264 encoded stream";
  return out;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> H264EncodeSima(int w, int h, int fps, int bitrate_kbps,
                                                   std::string profile, std::string level) {
  return std::make_shared<simaai::neat::H264EncodeSima>(w, h, fps, bitrate_kbps, std::move(profile),
                                                        std::move(level));
}

std::shared_ptr<simaai::neat::Node> H264EncodeSW(int bitrate_kbps) {
  return std::make_shared<H264EncodeSWNode>(bitrate_kbps);
}

} // namespace simaai::neat::nodes

#include "nodes/sima/H264EncodeSima.h"

#include "gst/GstHelpers.h"

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
    std::string factory;
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
    ss << factory << " name=n" << node_index << "_swenc";
    if (!props.empty())
      ss << " " << props;
    return ss.str();
  }

  std::vector<std::string> element_names(int node_index) const override {
    return {"n" + std::to_string(node_index) + "_swenc"};
  }

private:
  int bitrate_kbps_ = 4000;
};

} // namespace

namespace simaai::neat {

H264EncodeSima::H264EncodeSima(int w, int h, int fps, int bitrate_kbps, std::string profile,
                               std::string level)
    : w_(w), h_(h), fps_(fps), bitrate_kbps_(bitrate_kbps), profile_(std::move(profile)),
      level_(std::move(level)) {}

std::string H264EncodeSima::backend_fragment(int node_index) const {
  std::ostringstream ss;
  ss << "neatencoder name=n" << node_index << "_encoder " << "enc-type=h264 "
     << "enc-profile=" << profile_ << " " << "enc-level=" << level_ << " " << "enc-fmt=NV12 "
     << "enc-width=" << w_ << " " << "enc-height=" << h_ << " " << "enc-frame-rate=" << fps_ << " "
     << "enc-bitrate=" << bitrate_kbps_ << " " << "enc-ip-mode=async " << "ip-rate-ctrl=false";
  if (const char* dump_cnt = std::getenv("SIMA_NEATENCODER_DUMP_CNT")) {
    if (*dump_cnt) {
      ss << " dump-cnt=" << dump_cnt;
    }
  }
  if (const char* dump_path = std::getenv("SIMA_NEATENCODER_DUMP_PATH")) {
    if (*dump_path) {
      ss << " dump-path=" << dump_path;
    }
  }
  return ss.str();
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

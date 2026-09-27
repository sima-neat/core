#include "nodes/sima/SimaEncode.h"

#include "nodes/groups/internal/VideoSenderRawIngress.h"
#include "nodes/sima/internal/SimaEncode.h"

#include <cstdlib>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace simaai::neat {
namespace internal {
namespace {

void require(bool valid, const char* message) {
  if (!valid) {
    throw std::invalid_argument(std::string("SimaEncode: ") + message);
  }
}

bool one_of(const std::string& value, std::initializer_list<const char*> choices) {
  for (const auto* choice : choices) {
    if (value == choice) {
      return true;
    }
  }
  return false;
}

} // namespace

void validate_encode_options(const SimaEncodeOptions& o) {
  require(o.type == SimaEncodeType::H264 || o.type == SimaEncodeType::H265 ||
              o.type == SimaEncodeType::MJPEG,
          "unsupported codec");
  require(o.width > 0 && o.width <= 3840 && o.width % 2 == 0,
          "width must be even and within 2..3840");
  require(o.height > 0 && o.height <= 2160 && o.height % 2 == 0,
          "height must be even and within 2..2160");
  require(o.fps > 0, "fps must be positive");
  require(o.num_buffers == -1 || (o.num_buffers >= 1 && o.num_buffers <= 20),
          "num_buffers must be -1 or within 1..20");
  if (o.type == SimaEncodeType::MJPEG) {
    require(!o.bitrate_kbps && !o.rate_control && !o.profile && !o.level && !o.gop_length &&
                !o.idr_interval,
            "MJPEG does not accept bitrate, rate control, profile, level, GOP or IDR settings");
    require(!o.quality || (*o.quality >= 1 && *o.quality <= 100), "quality must be within 1..100");
    return;
  }
  require(!o.quality, "quality applies only to MJPEG");
  require(!o.bitrate_kbps || *o.bitrate_kbps > 0, "bitrate_kbps must be positive");
  require(!o.rate_control || one_of(*o.rate_control, {"vbr", "cbr"}),
          "rate_control must be vbr or cbr");
  require(!o.profile ||
              (o.type == SimaEncodeType::H264 ? one_of(*o.profile, {"baseline", "main", "high"})
                                              : *o.profile == "main"),
          "profile is unsupported for the selected codec");
  require(!o.level || one_of(*o.level, {"4.0", "4.1", "4.2", "5.0", "5.1", "5.2"}),
          "level must be 4.0, 4.1, 4.2, 5.0, 5.1 or 5.2");
  require(!o.gop_length || (*o.gop_length >= 0 && *o.gop_length <= 1000),
          "gop_length must be within 0..1000");
  require(!o.idr_interval || *o.idr_interval >= 0, "idr_interval must be nonnegative");
}

std::string encoder_fragment(const SimaEncodeOptions& o, int node_index) {
  const char* codec = o.type == SimaEncodeType::MJPEG  ? "mjpeg"
                      : o.type == SimaEncodeType::H265 ? "h265"
                                                       : "h264";
  std::ostringstream ss;
  ss << "neatencoder name=n" << node_index << "_encoder enc-type=" << codec;
  if (o.type == SimaEncodeType::MJPEG) {
    ss << " enc-quality=" << o.quality.value_or(80);
  } else {
    ss << " enc-profile="
       << o.profile.value_or(o.type == SimaEncodeType::H265 ? "main" : "baseline")
       << " enc-level=" << o.level.value_or("4.0")
       << " enc-bitrate=" << o.bitrate_kbps.value_or(4000);
    if (o.rate_control)
      ss << " enc-bitrate-mode=" << *o.rate_control;
    if (o.gop_length)
      ss << " enc-gop-length=" << *o.gop_length;
    if (o.idr_interval)
      ss << " enc-idr-interval=" << *o.idr_interval;
  }
  ss << " enc-fmt=NV12 enc-width=" << o.width << " enc-height=" << o.height
     << " enc-frame-rate=" << o.fps << " enc-ip-mode=async ip-rate-ctrl=false";
  if (o.num_buffers != -1)
    ss << " num-output-buffers=" << o.num_buffers;
  if (const char* value = std::getenv("SIMA_NEATENCODER_DUMP_CNT")) {
    if (*value)
      ss << " dump-cnt=" << value;
  }
  if (const char* value = std::getenv("SIMA_NEATENCODER_DUMP_PATH")) {
    if (*value)
      ss << " dump-path=" << value;
  }
  return ss.str();
}

} // namespace internal

SimaEncode::SimaEncode(SimaEncodeOptions options) : SimaEncode(std::move(options), true) {}

SimaEncode::SimaEncode(SimaEncodeOptions options, bool prepare_input)
    : options_(std::move(options)), prepare_input_(prepare_input) {
  internal::validate_encode_options(options_);
}

std::string SimaEncode::backend_fragment(int node_index) const {
  auto fragment = internal::encoder_fragment(options_, node_index);
  if (prepare_input_) {
    fragment = nodes::groups::internal::VideoSenderRawIngress(options_.width, options_.height,
                                                              options_.fps)
                   ->backend_fragment(node_index) +
               " ! " + fragment;
  }
  return fragment;
}

std::vector<std::string> SimaEncode::element_names(int node_index) const {
  std::vector<std::string> names;
  if (prepare_input_) {
    names = nodes::groups::internal::VideoSenderRawIngress(options_.width, options_.height,
                                                           options_.fps)
                ->element_names(node_index);
  }
  names.push_back("n" + std::to_string(node_index) + "_encoder");
  return names;
}

OutputSpec SimaEncode::output_spec(const OutputSpec&) const {
  OutputSpec out;
  out.payload_type = PayloadType::Encoded;
  out.media_type = options_.type == SimaEncodeType::MJPEG  ? "image/jpeg"
                   : options_.type == SimaEncodeType::H265 ? "video/x-h265"
                                                           : "video/x-h264";
  out.format = options_.type == SimaEncodeType::MJPEG  ? "JPEG"
               : options_.type == SimaEncodeType::H265 ? "H265"
                                                       : "H264";
  out.width = options_.width;
  out.height = options_.height;
  out.fps_num = options_.fps;
  out.fps_den = 1;
  out.certainty = SpecCertainty::Derived;
  out.note = "Encoded frames";
  return out;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> SimaEncode(SimaEncodeOptions options) {
  return std::make_shared<simaai::neat::SimaEncode>(std::move(options));
}
} // namespace simaai::neat::nodes

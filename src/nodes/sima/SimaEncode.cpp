#include "nodes/sima/SimaEncode.h"

#include "builder/internal/InputSpecSpecialization.h"
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
  // simaaiencoder enc-type takes h264 or h265 only: gstsimaaiencoder.cpp:2215-2228.
  require(o.type != SimaEncodeType::MJPEG, "MJPEG encode is not supported by simaaiencoder");
  require(o.type == SimaEncodeType::H264 || o.type == SimaEncodeType::H265, "unsupported codec");
  require(o.fps > 0, "fps must be positive");
  // op-pool-cnt range: gstsimaaiencoder.cpp:101-103, :1645-1651.
  require(o.num_buffers == -1 || (o.num_buffers >= 2 && o.num_buffers <= 64),
          "num_buffers must be -1 or within 2..64");
  require(!o.quality, "quality applies only to MJPEG");
  require(!o.bitrate_kbps || *o.bitrate_kbps > 0, "bitrate_kbps must be positive");
  require(!o.rate_control || one_of(*o.rate_control, {"vbr", "cbr"}),
          "rate_control must be vbr or cbr");
  require(!o.profile ||
              (o.type == SimaEncodeType::H264 ? one_of(*o.profile, {"baseline", "main", "high"})
                                              : *o.profile == "main"),
          "profile is unsupported for the selected codec");
  require(!o.level || (o.type == SimaEncodeType::H264
                           ? one_of(*o.level, {"4.0", "4.1", "4.2", "5.0", "5.1", "5.2"})
                           : one_of(*o.level, {"4.0", "4.1", "5.0", "5.1", "5.2"})),
          o.type == SimaEncodeType::H264 ? "H.264 level must be 4.0, 4.1, 4.2, 5.0, 5.1 or 5.2"
                                         : "H.265 level must be 4.0, 4.1, 5.0, 5.1 or 5.2");
  require(!o.gop_length || (*o.gop_length >= 0 && *o.gop_length <= 1000),
          "gop_length must be within 0..1000");
  require(!o.idr_interval || *o.idr_interval >= 0, "idr_interval must be nonnegative");
}

std::string encoder_fragment(const SimaEncodeOptions& o, int node_index) {
  const char* codec = o.type == SimaEncodeType::H265 ? "h265" : "h264";
  std::ostringstream ss;
  ss << "simaaiencoder name=n" << node_index << "_encoder enc-type=" << codec;
  ss << " enc-profile=" << o.profile.value_or(o.type == SimaEncodeType::H265 ? "main" : "baseline")
     << " enc-level=" << o.level.value_or("4.0")
     << " enc-bitrate=" << o.bitrate_kbps.value_or(4000);
  if (o.rate_control)
    ss << " enc-rc-mode=" << *o.rate_control;
  if (o.gop_length)
    ss << " enc-gop-length=" << *o.gop_length;
  if (o.idr_interval)
    ss << " enc-idr-interval=" << *o.idr_interval;
  ss << " enc-fmt=NV12 enc-frame-rate=" << o.fps << " enc-ip-mode=async ip-rate-ctrl=false";
  if (o.num_buffers != -1)
    ss << " op-pool-cnt=" << o.num_buffers;
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
    : options_(std::move(options)) {
  internal::validate_encode_options(options_);
  if (prepare_input) {
    input_adapter_ = nodes::groups::internal::VideoSenderRawIngress(0, 0, options_.fps);
  }
}

MemoryContract SimaEncode::memory_contract() const {
  return input_adapter_ ? input_adapter_->memory_contract() : MemoryContract::AllowEitherButReport;
}

std::string SimaEncode::backend_fragment(int node_index) const {
  auto fragment = internal::encoder_fragment(options_, node_index);
  if (input_adapter_) {
    fragment = input_adapter_->backend_fragment(node_index) + " ! " + fragment;
  }
  return fragment;
}

std::vector<std::string> SimaEncode::element_names(int node_index) const {
  std::vector<std::string> names;
  if (input_adapter_) {
    names = input_adapter_->element_names(node_index);
  }
  names.push_back("n" + std::to_string(node_index) + "_encoder");
  return names;
}

std::shared_ptr<Node>
internal::SimaEncodeAccess::specialize_for_input(const SimaEncode& node, const OutputSpec& input,
                                                 const InputSpecSpecializationContext& context) {
  auto selected = std::make_shared<SimaEncode>(node);
  if (node.input_adapter_) {
    auto adapter = specialize_nodes_for_input(std::span(&node.input_adapter_, 1), input, context);
    selected->input_adapter_ = std::move(adapter.nodes.front());
  }
  return selected;
}

OutputSpec SimaEncode::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.payload_type = PayloadType::Encoded;
  out.media_type = options_.type == SimaEncodeType::MJPEG  ? "image/jpeg"
                   : options_.type == SimaEncodeType::H265 ? "video/x-h265"
                                                           : "video/x-h264";
  out.format = options_.type == SimaEncodeType::MJPEG  ? "JPEG"
               : options_.type == SimaEncodeType::H265 ? "H265"
                                                       : "H264";
  out.width = input.width;
  out.height = input.height;
  out.fps_num = options_.fps;
  out.fps_den = 1;
  out.certainty = input.certainty;
  out.note = "Encoded frames";
  return out;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> SimaEncode(SimaEncodeOptions options) {
  return std::make_shared<simaai::neat::SimaEncode>(std::move(options));
}
} // namespace simaai::neat::nodes

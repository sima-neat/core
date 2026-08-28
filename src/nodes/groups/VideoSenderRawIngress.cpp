#include "nodes/groups/internal/VideoSenderRawIngress.h"

#include "builder/internal/InputSpecSpecialization.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::nodes::groups::internal {
namespace {

enum class IngressVariant {
  ConvertToNv12,
  MaterializeNv12,
  DirectNv12,
};

bool certainty_is_stable(SpecCertainty certainty) {
  return certainty == SpecCertainty::Derived || certainty == SpecCertainty::Authoritative;
}

class VideoSenderRawIngressNode final : public Node,
                                        public OutputSpecProvider,
                                        public simaai::neat::internal::InputSpecSpecializer {
public:
  explicit VideoSenderRawIngressNode(VideoSenderRawIngressConfig config,
                                     IngressVariant variant = IngressVariant::ConvertToNv12)
      : config_(std::move(config)), variant_(variant) {
    if (!config_.fallback_element_names.empty() && config_.fallback_element_names.size() != 3U) {
      throw std::invalid_argument(
          "VideoSenderRawIngress requires exactly three persisted fallback element names");
    }
  }

  std::string kind() const override {
    // Connected describe_backend() reports Node kinds, so include the selected
    // materialization while retaining one semantic Node in either variant.
    switch (variant_) {
    case IngressVariant::DirectNv12:
      return std::string(kVideoSenderRawIngressDirectKind);
    case IngressVariant::MaterializeNv12:
      return std::string(kVideoSenderRawIngressMaterializeKind);
    case IngressVariant::ConvertToNv12:
      return std::string(kVideoSenderRawIngressConvertKind);
    }
    throw std::logic_error("invalid VideoSender raw-ingress variant");
  }

  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override {
    const auto names = fallback_element_names(node_index);
    std::ostringstream caps;
    caps << "video/x-raw,format=NV12,width=" << config_.width << ",height=" << config_.height
         << ",framerate=" << config_.fps << "/1";

    if (variant_ != IngressVariant::ConvertToNv12) {
      return "capsfilter name=" + names[2] + " caps=\"" + caps.str() + "\"";
    }

    std::ostringstream input_caps;
    input_caps << "video/x-raw,width=" << config_.width << ",height=" << config_.height
               << ",framerate=" << config_.fps << "/1";
    return "capsfilter name=" + names[0] + " caps=\"" + input_caps.str() +
           "\" ! videoconvert name=" + names[1] + " ! capsfilter name=" + names[2] + " caps=\"" +
           caps.str() + "\"";
  }

  std::vector<std::string> element_names(int node_index) const override {
    const auto names = fallback_element_names(node_index);
    if (variant_ != IngressVariant::ConvertToNv12) {
      // The direct variant is the fallback's final NV12 caps element, so a
      // loaded graph keeps exactly the serialized, collision-safe name.
      return {names[2]};
    }
    return names;
  }

  OutputSpec output_spec(const OutputSpec& input) const override {
    OutputSpec out = input;
    out.payload_type = PayloadType::Image;
    out.media_type = "video/x-raw";
    out.format = "NV12";
    out.width = config_.width;
    out.height = config_.height;
    out.fps_num = config_.fps;
    out.fps_den = 1;
    out.layout = "Planar";
    out.dtype = "UInt8";
    out.depth = -1;
    out.byte_size = 0;
    out.certainty = SpecCertainty::Derived;
    switch (variant_) {
    case IngressVariant::DirectNv12:
      out.note = "VideoSender raw ingress (direct NV12)";
      break;
    case IngressVariant::MaterializeNv12:
      out.note = "VideoSender raw ingress (explicit SystemMemory to CMA NV12 materialization)";
      break;
    case IngressVariant::ConvertToNv12:
      out.note = "VideoSender raw ingress (converted to NV12)";
      break;
    }
    out.byte_size = expected_byte_size(out);
    return out;
  }

  std::shared_ptr<Node> specialize_for_input(
      const OutputSpec& input,
      const simaai::neat::internal::InputSpecSpecializationContext& context) const override {
    const bool layout_aware = context.capability(kNeatEncoderInputLayoutAwareCapability);
    const bool direct = can_encode_nv12_direct(input, layout_aware);
    const bool materialize = !direct && layout_aware && certainty_is_stable(input.certainty) &&
                             input.media_type == "video/x-raw" && input.format == "NV12" &&
                             input.memory == "SystemMemory";
    return std::make_shared<VideoSenderRawIngressNode>(
        config_, direct ? IngressVariant::DirectNv12
                        : materialize ? IngressVariant::MaterializeNv12
                                      : IngressVariant::ConvertToNv12);
  }

  const VideoSenderRawIngressConfig& config() const {
    return config_;
  }

private:
  std::vector<std::string> fallback_element_names(int node_index) const {
    if (!config_.fallback_element_names.empty()) {
      return config_.fallback_element_names;
    }
    const std::string prefix = "n" + std::to_string(node_index);
    return {prefix + "_raw_input_caps", prefix + "_videoconvert", prefix + "_nv12_caps"};
  }

  VideoSenderRawIngressConfig config_;
  IngressVariant variant_;
};

} // namespace

bool can_encode_nv12_direct(const OutputSpec& input, bool simaai_layout_aware) {
  if (!certainty_is_stable(input.certainty) || input.media_type != "video/x-raw" ||
      input.format != "NV12") {
    return false;
  }

  /* Direct means that the producer already owns a device-visible allocation.
   * SystemMemory still uses this existing semantic node's explicit
   * videoconvert path; allocation negotiation lets that operation render its
   * NV12 result directly into the encoder-proposed standard CMA DMA-BUF.
   * There is no implicit encoder upload and no second conversion node. */
  return input.memory == "SimaAI" && simaai_layout_aware;
}

std::shared_ptr<Node> VideoSenderRawIngress(int width, int height, int fps) {
  return VideoSenderRawIngress(VideoSenderRawIngressConfig{
      .width = width,
      .height = height,
      .fps = fps,
      .fallback_element_names = {},
  });
}

std::shared_ptr<Node> VideoSenderRawIngress(VideoSenderRawIngressConfig config) {
  return std::make_shared<VideoSenderRawIngressNode>(std::move(config));
}

std::optional<VideoSenderRawIngressConfig> video_sender_raw_ingress_config(const Node& node) {
  const auto* ingress = dynamic_cast<const VideoSenderRawIngressNode*>(&node);
  if (!ingress) {
    return std::nullopt;
  }
  return ingress->config();
}

} // namespace simaai::neat::nodes::groups::internal

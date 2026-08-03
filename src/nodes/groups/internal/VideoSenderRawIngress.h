/**
 * @file
 * @brief Private adaptive raw ingress for the H.264 VideoSender.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace simaai::neat::nodes::groups::internal {

inline constexpr std::string_view kNeatEncoderInputLayoutAwareCapability =
    "neatencoder.input-layout-aware";

/**
 * Decide whether an upstream NV12 contract is strong and layout-safe enough to
 * connect directly to neatencoder.
 *
 * `simaai_layout_aware` is supplied by the plugin capability adapter so this
 * pure policy remains deterministic and independently testable.
 */
bool can_encode_nv12_direct(const OutputSpec& input, bool simaai_layout_aware);

struct VideoSenderRawIngressConfig {
  int width;
  int height;
  int fps;
  // Empty for newly-authored graphs. Loaded graphs retain the three names
  // already transformed into their serialized fallback fragment.
  std::vector<std::string> fallback_element_names;
};

std::shared_ptr<Node> VideoSenderRawIngress(int width, int height, int fps);
std::shared_ptr<Node> VideoSenderRawIngress(VideoSenderRawIngressConfig config);

/// Return the semantic configuration for typed Graph save/load, if applicable.
std::optional<VideoSenderRawIngressConfig> video_sender_raw_ingress_config(const Node& node);

} // namespace simaai::neat::nodes::groups::internal

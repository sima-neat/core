#include "nodes/groups/RtspDecodedInput.h"

#include "nodes/common/Caps.h"
#include "nodes/common/Queue.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264CapsFixup.h"
#include "nodes/rtp/H264Depacketize.h"
#include "nodes/sima/H264DecodeSima.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/common/VideoScale.h"
#include "pipeline/internal/SyncBuild.h"

#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::nodes::groups {
namespace {

bool caps_enabled(const RtspDecodedInputOptions::OutputCaps& c) {
  return c.enable || c.width > 0 || c.height > 0 || c.fps > 0;
}

} // namespace

simaai::neat::Graph RtspDecodedInput(const RtspDecodedInputOptions& opt) {
  std::vector<std::shared_ptr<simaai::neat::Node>> nodes;

  const bool force_sync = simaai::neat::pipeline_internal::sync_build_mode();
  if (force_sync && opt.insert_queue && !opt.sync_mode) {
    simaai::neat::pipeline_internal::warn_sync_override("RtspDecodedInput");
  }
  nodes.push_back(nodes::RTSPInput(opt.url, opt.latency_ms, opt.tcp));
  const bool insert_queue = opt.insert_queue && !(opt.sync_mode || force_sync);
  if (insert_queue)
    nodes.push_back(nodes::Queue());
  const bool use_auto_caps = opt.auto_caps_from_stream &&
                             (opt.h264_fps <= 0 || opt.h264_width <= 0 || opt.h264_height <= 0);
  nodes.push_back(nodes::H264Depacketize(opt.payload_type, opt.h264_parse_config_interval,
                                         opt.h264_fps, opt.h264_width, opt.h264_height,
                                         /*enforce_h264_caps=*/!use_auto_caps));

  if (insert_queue)
    nodes.push_back(nodes::Queue());

  if (use_auto_caps) {
    nodes.push_back(nodes::H264CapsFixup(opt.fallback_h264_fps, opt.fallback_h264_width,
                                         opt.fallback_h264_height));
  }

  const int dec_w = (opt.h264_width > 0) ? opt.h264_width : opt.fallback_h264_width;
  const int dec_h = (opt.h264_height > 0) ? opt.h264_height : opt.fallback_h264_height;
  const int dec_fps = (opt.h264_fps > 0) ? opt.h264_fps : opt.fallback_h264_fps;
  if (opt.decoder_raw_output && !use_auto_caps && (dec_w <= 0 || dec_h <= 0 || dec_fps <= 0)) {
    throw std::runtime_error("RtspDecodedInput: decoder_raw_output requires h264 width/height/fps");
  }
  nodes.push_back(nodes::H264Decode(opt.sima_allocator_type, opt.out_format, opt.decoder_name,
                                    opt.decoder_raw_output, opt.decoder_next_element, dec_w, dec_h,
                                    dec_fps));

  if (opt.use_videoconvert)
    nodes.push_back(nodes::VideoConvert());
  if (opt.use_videoscale)
    nodes.push_back(nodes::VideoScale());

  if (caps_enabled(opt.output_caps)) {
    const auto& c = opt.output_caps;
    nodes.push_back(nodes::CapsRaw(c.format, c.width, c.height, c.fps, c.memory));
  }

  if (!opt.extra_fragment.empty()) {
    nodes.push_back(nodes::Custom(opt.extra_fragment));
  }

  simaai::neat::Graph graph;
  for (auto& node : nodes) {
    graph.add(std::move(node));
  }
  return graph;
}

} // namespace simaai::neat::nodes::groups

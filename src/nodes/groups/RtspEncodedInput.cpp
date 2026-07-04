#include "nodes/groups/RtspEncodedInput.h"

#include "nodes/common/EncodedCapsFixup.h"
#include "nodes/common/JpegParse.h"
#include "nodes/common/Queue.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264CapsFixup.h"
#include "nodes/rtp/H264Depacketize.h"
#include "nodes/rtp/RTPJpegDepacketize.h"
#include "pipeline/internal/SyncBuild.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::nodes::groups {
namespace {

void require_same_fps_if_set(const char* group, int source_fps, const char* option_name,
                             int option_fps) {
  if (source_fps > 0 && option_fps > 0 && option_fps != source_fps) {
    throw std::invalid_argument(std::string(group) + ": source_fps conflicts with " + option_name);
  }
}

int h264_source_fps(const RtspEncodedInputOptions& opt) {
  require_same_fps_if_set("RtspEncodedInput", opt.source_fps, "h264_fps", opt.h264_fps);
  return (opt.source_fps > 0) ? opt.source_fps : opt.h264_fps;
}

int h264_fixup_fps(const RtspEncodedInputOptions& opt) {
  const int source = h264_source_fps(opt);
  return (source > 0) ? source : opt.fallback_h264_fps;
}

bool use_h264_auto_caps(const RtspEncodedInputOptions& opt) {
  return opt.auto_caps_from_stream &&
         (h264_source_fps(opt) <= 0 || opt.h264_width <= 0 || opt.h264_height <= 0);
}

void add_source_and_optional_queue(std::vector<std::shared_ptr<simaai::neat::Node>>& nodes,
                                   const RtspEncodedInputOptions& opt, bool insert_queue) {
  nodes.push_back(
      nodes::RTSPInput(opt.url, opt.latency_ms, opt.tcp, opt.drop_on_latency, opt.buffer_mode));
  if (insert_queue) {
    nodes.push_back(nodes::Queue());
  }
}

void add_h264_path(std::vector<std::shared_ptr<simaai::neat::Node>>& nodes,
                   const RtspEncodedInputOptions& opt, bool insert_queue) {
  const bool auto_caps = use_h264_auto_caps(opt);
  const int source_fps = h264_source_fps(opt);
  nodes.push_back(nodes::H264Depacketize(opt.h264_payload_type, opt.h264_parse_config_interval,
                                         source_fps, opt.h264_width, opt.h264_height,
                                         /*enforce_h264_caps=*/!auto_caps));
  if (insert_queue) {
    nodes.push_back(nodes::Queue());
  }
  if (auto_caps) {
    nodes.push_back(nodes::H264CapsFixup(h264_fixup_fps(opt), opt.fallback_h264_width,
                                         opt.fallback_h264_height));
  }
}

void add_mjpeg_path(std::vector<std::shared_ptr<simaai::neat::Node>>& nodes,
                    const RtspEncodedInputOptions& opt, bool insert_queue) {
  nodes.push_back(nodes::RTPJpegDepacketize(opt.mjpeg_payload_type));
  nodes.push_back(nodes::JpegParse());
  if (insert_queue) {
    nodes.push_back(nodes::Queue());
  }
  if (opt.source_fps > 0 || opt.auto_caps_from_stream) {
    EncodedCapsFixupOptions fixup{"image/jpeg", opt.source_fps};
    fixup.use_rtsp_sdp_fps = opt.auto_caps_from_stream && opt.source_fps <= 0;
    nodes.push_back(nodes::EncodedCapsFixup(fixup));
  }
}

} // namespace

simaai::neat::Graph RtspEncodedInput(const RtspEncodedInputOptions& opt) {
  std::vector<std::shared_ptr<simaai::neat::Node>> nodes;

  const bool force_sync = simaai::neat::pipeline_internal::sync_build_mode();
  if (force_sync && opt.insert_queue && !opt.sync_mode) {
    simaai::neat::pipeline_internal::warn_sync_override("RtspEncodedInput");
  }
  const bool insert_queue = opt.insert_queue && !(opt.sync_mode || force_sync);

  add_source_and_optional_queue(nodes, opt, insert_queue);
  switch (opt.codec) {
  case RtspCodec::H264:
    add_h264_path(nodes, opt, insert_queue);
    break;
  case RtspCodec::MJPEG:
    add_mjpeg_path(nodes, opt, insert_queue);
    break;
  default:
    throw std::invalid_argument("RtspEncodedInput: unsupported codec");
  }

  simaai::neat::Graph graph;
  for (auto& node : nodes) {
    graph.add(std::move(node));
  }
  return graph;
}

} // namespace simaai::neat::nodes::groups

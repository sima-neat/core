#include "nodes/groups/RtspEncodedInput.h"

#include "nodes/common/JpegParse.h"
#include "nodes/common/Queue.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264CapsFixup.h"
#include "nodes/rtp/H264Depacketize.h"
#include "nodes/rtp/RTPJpegDepacketize.h"
#include "pipeline/internal/SyncBuild.h"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace simaai::neat::nodes::groups {
namespace {

struct ResolvedH264Caps {
  int fps = -1;
  int width = -1;
  int height = -1;
  bool auto_caps = false;
};

bool use_h264_auto_caps(const RtspEncodedInputOptions& opt) {
  return opt.auto_caps_from_stream &&
         (opt.h264_fps <= 0 || opt.h264_width <= 0 || opt.h264_height <= 0);
}

ResolvedH264Caps resolve_h264_caps(const RtspEncodedInputOptions& opt) {
  ResolvedH264Caps out;
  out.fps = opt.h264_fps;
  out.width = opt.h264_width;
  out.height = opt.h264_height;
  out.auto_caps = use_h264_auto_caps(opt);

  if (!out.auto_caps) {
    if (out.fps <= 0) {
      out.fps = opt.fallback_h264_fps;
    }
    if (out.width <= 0) {
      out.width = opt.fallback_h264_width;
    }
    if (out.height <= 0) {
      out.height = opt.fallback_h264_height;
    }
    if (out.fps <= 0 || out.width <= 0 || out.height <= 0) {
      throw std::runtime_error(
          "RtspEncodedInput: H.264 explicit caps require width, height, and fps when "
          "auto_caps_from_stream is false");
    }
  }

  return out;
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
  const ResolvedH264Caps caps = resolve_h264_caps(opt);
  nodes.push_back(nodes::H264Depacketize(opt.h264_payload_type, opt.h264_parse_config_interval,
                                         caps.fps, caps.width, caps.height,
                                         /*enforce_h264_caps=*/!caps.auto_caps));
  if (insert_queue) {
    nodes.push_back(nodes::Queue());
  }
  if (caps.auto_caps) {
    nodes.push_back(nodes::H264CapsFixup(opt.fallback_h264_fps, opt.fallback_h264_width,
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

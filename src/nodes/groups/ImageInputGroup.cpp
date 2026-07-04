#include "nodes/groups/ImageInputGroup.h"

#include "nodes/common/Caps.h"
#include "nodes/common/FileInput.h"
#include "nodes/common/ImageDecode.h"
#include "nodes/common/ImageFreeze.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/common/VideoRate.h"
#include "nodes/common/VideoScale.h"
#include "nodes/common/JpegDecode.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/SimaDecode.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::nodes::groups {
namespace {

bool caps_enabled(const ImageInputGroupOptions::OutputCaps& c) {
  return c.enable || c.width > 0 || c.height > 0 || c.fps > 0;
}

} // namespace

simaai::neat::Graph ImageInputGroup(const ImageInputGroupOptions& opt) {
  std::vector<std::shared_ptr<simaai::neat::Node>> nodes;

  nodes.push_back(nodes::FileInput(opt.path));

  switch (opt.decoder) {
  case ImageInputGroupOptions::Decoder::Auto:
    nodes.push_back(nodes::ImageDecode());
    break;
  case ImageInputGroupOptions::Decoder::ForceJpeg:
    nodes.push_back(nodes::JpegDecode());
    break;
  case ImageInputGroupOptions::Decoder::ForcePng:
    nodes.push_back(nodes::Custom("pngdec"));
    break;
  case ImageInputGroupOptions::Decoder::Custom:
    nodes.push_back(nodes::Custom(opt.custom_decoder_fragment));
    break;
  }

  if (!opt.extra_fragment.empty()) {
    nodes.push_back(nodes::Custom(opt.extra_fragment));
  }

  // Honor explicit finite source counts exactly.  A still-image fixture that asks for one frame
  // must not be silently inflated into a large burst before the hardware encode/decode path; burst
  // stress belongs in dedicated decoder tests, not in every finite image source.
  nodes.push_back(nodes::ImageFreeze(opt.imagefreeze_num_buffers));

  auto caps = opt.output_caps;
  bool want_videorate = opt.use_videorate;
  bool want_videoscale = opt.use_videoscale;

  if (caps.fps > 0) {
    want_videorate = true;
  }
  if (caps.fps <= 0 && (want_videorate || opt.sima_decoder.enable)) {
    caps.fps = opt.fps;
    want_videorate = true;
  }
  if (caps.width > 0 || caps.height > 0) {
    want_videoscale = true;
  }

  if (want_videorate) {
    nodes.push_back(nodes::VideoRate());
  }

  if (opt.use_videoconvert) {
    nodes.push_back(nodes::VideoConvert());
  }

  if (want_videoscale) {
    nodes.push_back(nodes::VideoScale());
  }

  // caps configured above (may be updated to match requested rate/size)

  if (opt.sima_decoder.enable) {
    if (caps.width <= 0 || caps.height <= 0) {
      throw std::runtime_error("ImageInputGroup: sima_decoder requires output_caps.width/height");
    }
    if (caps.fps <= 0) {
      throw std::runtime_error("ImageInputGroup: sima_decoder requires fps or output_caps.fps");
    }

    nodes.push_back(nodes::CapsRaw("NV12", caps.width, caps.height, caps.fps,
                                   simaai::neat::CapsMemory::SystemMemory));
    if (opt.sima_decoder.use_sw_encoder) {
      nodes.push_back(nodes::H264EncodeSW(opt.sima_decoder.sw_bitrate_kbps));
    } else {
      nodes.push_back(nodes::H264EncodeSima(caps.width, caps.height, caps.fps));
    }
    nodes.push_back(nodes::H264Parse(/*config_interval=*/1));

    const std::string out_fmt =
        caps.format.empty() ? (opt.sima_decoder.raw_output ? "NV12" : "RGB") : caps.format;
    simaai::neat::SimaDecodeOptions dec;
    dec.type = simaai::neat::SimaDecodeType::H264;
    dec.sima_allocator_type = opt.sima_decoder.sima_allocator_type;
    dec.out_format = out_fmt;
    dec.decoder_name = opt.sima_decoder.decoder_name;
    dec.raw_output = opt.sima_decoder.raw_output;
    dec.next_element = opt.sima_decoder.next_element;
    dec.dec_width = caps.width;
    dec.dec_height = caps.height;
    dec.dec_fps = caps.fps;
    nodes.push_back(nodes::SimaDecode(dec));
  }

  if (caps_enabled(caps)) {
    nodes.push_back(nodes::CapsRaw(caps.format, caps.width, caps.height, caps.fps, caps.memory));
  }

  simaai::neat::Graph graph;
  for (auto& node : nodes) {
    graph.add(std::move(node));
  }
  return graph;
}

} // namespace simaai::neat::nodes::groups

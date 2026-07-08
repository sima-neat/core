#include "nodes/groups/HttpMjpegDecodedInput.h"

#include "nodes/common/Caps.h"
#include "nodes/common/EncodedCapsFixup.h"
#include "nodes/common/JpegParse.h"
#include "nodes/common/MultipartJpegDemux.h"
#include "nodes/common/Queue.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/common/VideoRate.h"
#include "nodes/common/VideoScale.h"
#include "nodes/io/HttpSource.h"
#include "nodes/sima/SimaDecode.h"
#include "pipeline/internal/SyncBuild.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::nodes::groups {
namespace {

bool caps_enabled(const HttpMjpegDecodedInputOptions::OutputCaps& c) {
  return c.enable;
}

int resolve_source_fps(const HttpMjpegDecodedInputOptions& opt) {
  if (opt.source_fps > 0 && opt.dec_fps > 0 && opt.dec_fps != opt.source_fps) {
    throw std::invalid_argument("HttpMjpegDecodedInput: source_fps conflicts with dec_fps");
  }
  return (opt.source_fps > 0) ? opt.source_fps : opt.dec_fps;
}

int resolve_video_rate_fps(const HttpMjpegDecodedInputOptions& opt, int source_fps) {
  if (opt.video_rate_fps > 0 && !opt.use_videorate) {
    throw std::invalid_argument(
        "HttpMjpegDecodedInput: video_rate_fps requires use_videorate=true");
  }
  if (!opt.use_videorate) {
    if (opt.source_fps > 0 && caps_enabled(opt.output_caps) && opt.output_caps.fps > 0 &&
        opt.output_caps.fps != source_fps) {
      throw std::invalid_argument(
          "HttpMjpegDecodedInput: output_caps.fps conflicts with source_fps; use video_rate_fps "
          "for rate conversion");
    }
    return -1;
  }

  const int fps = (opt.video_rate_fps > 0) ? opt.video_rate_fps : source_fps;
  if (fps <= 0) {
    throw std::invalid_argument(
        "HttpMjpegDecodedInput: use_videorate requires video_rate_fps or source_fps");
  }
  if (caps_enabled(opt.output_caps) && opt.output_caps.fps > 0 && opt.output_caps.fps != fps) {
    throw std::invalid_argument(
        "HttpMjpegDecodedInput: output_caps.fps conflicts with video_rate_fps");
  }
  return fps;
}

} // namespace

simaai::neat::Graph HttpMjpegDecodedInput(const HttpMjpegDecodedInputOptions& opt) {
  std::vector<std::shared_ptr<simaai::neat::Node>> nodes;
  const int source_fps = resolve_source_fps(opt);
  const int video_rate_fps = resolve_video_rate_fps(opt, source_fps);

  const bool force_sync = simaai::neat::pipeline_internal::sync_build_mode();
  if (force_sync && opt.insert_queue && !opt.sync_mode) {
    simaai::neat::pipeline_internal::warn_sync_override("HttpMjpegDecodedInput");
  }
  const bool insert_queue = opt.insert_queue && !(opt.sync_mode || force_sync);

  simaai::neat::HttpSourceOptions source;
  source.location = opt.url;
  source.timeout_seconds = opt.timeout_seconds;
  source.retries = opt.retries;
  source.is_live = opt.is_live;
  source.do_timestamp = opt.do_timestamp;
  source.user_agent = opt.user_agent;
  source.ssl_strict = opt.ssl_strict;
  nodes.push_back(nodes::HttpSource(std::move(source)));

  if (insert_queue)
    nodes.push_back(nodes::Queue());

  simaai::neat::MultipartJpegDemuxOptions demux;
  demux.boundary = opt.multipart_boundary;
  demux.single_stream = opt.multipart_single_stream;
  nodes.push_back(nodes::MultipartJpegDemux(std::move(demux)));

  nodes.push_back(nodes::JpegParse());

  if (insert_queue)
    nodes.push_back(nodes::Queue());

  if (source_fps > 0) {
    nodes.push_back(nodes::EncodedCapsFixup({"image/jpeg", source_fps}));
  }

  simaai::neat::SimaDecodeOptions dec;
  dec.type = simaai::neat::SimaDecodeType::MJPEG;
  dec.sima_allocator_type = opt.sima_allocator_type;
  dec.out_format = opt.out_format;
  dec.decoder_name = opt.decoder_name;
  dec.raw_output = opt.decoder_raw_output;
  dec.next_element = opt.decoder_next_element;
  dec.dec_width = opt.dec_width;
  dec.dec_height = opt.dec_height;
  dec.dec_fps = source_fps;
  dec.num_buffers = opt.num_buffers;
  nodes.push_back(nodes::SimaDecode(dec));

  if (opt.use_videoconvert)
    nodes.push_back(nodes::VideoConvert());
  if (opt.use_videorate)
    nodes.push_back(nodes::VideoRate());
  if (opt.use_videoscale)
    nodes.push_back(nodes::VideoScale());

  if (caps_enabled(opt.output_caps) || opt.use_videorate) {
    const auto& c = opt.output_caps;
    const bool tail_caps = caps_enabled(c);
    const auto memory = tail_caps ? c.memory : simaai::neat::CapsMemory::Any;
    const int fps = (video_rate_fps > 0) ? video_rate_fps : c.fps;
    nodes.push_back(nodes::CapsRaw(tail_caps ? c.format : simaai::neat::FormatSpec{},
                                   tail_caps ? c.width : -1, tail_caps ? c.height : -1, fps,
                                   memory));
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

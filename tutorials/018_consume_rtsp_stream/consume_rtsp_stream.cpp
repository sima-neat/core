// Consume a live H.264 or H.265 RTSP stream via RtspDecodedInput.
//
// The fragment handles RTSP connect, codec-specific depacketize/parse, and
// hardware decode. This chapter is about the input fragment only.
//
// Usage:
//   tutorial_018_consume_rtsp_stream --url rtsp://host/path
//     [--codec h264|avc|h265|hevc] [--source-fps 30] [--frames 5]

#include "neat.h"
#include "nodes/groups/RtspDecodedInput.h"

#include <opencv2/videoio.hpp>

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string v;
  if (!get_arg(argc, argv, key, v))
    return def;
  return std::stoi(v);
}

simaai::neat::nodes::groups::RtspCodec parse_codec(const std::string& value) {
  using simaai::neat::nodes::groups::RtspCodec;
  if (value == "h264" || value == "avc")
    return RtspCodec::H264;
  if (value == "h265" || value == "hevc")
    return RtspCodec::H265;
  throw std::invalid_argument("--codec must be h264, avc, h265, or hevc");
}

int probe_source_fps(const std::string& url) {
  cv::VideoCapture capture(url);
  if (!capture.isOpened()) {
    throw std::runtime_error("failed to open the RTSP source for FPS probing");
  }
  const int fps = static_cast<int>(std::lround(capture.get(cv::CAP_PROP_FPS)));
  capture.release();
  if (fps <= 0) {
    throw std::runtime_error("failed to probe a positive RTSP source FPS");
  }
  return fps;
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::string url;
    if (!get_arg(argc, argv, "--url", url)) {
      std::cerr << "Usage: tutorial_018_consume_rtsp_stream --url <rtsp://...> "
                   "[--codec h264|avc|h265|hevc] [--source-fps <n>] [--frames <n>]\n";
      return 1;
    }
    std::string codec_name = "h264";
    get_arg(argc, argv, "--codec", codec_name);
    const int frames = parse_int_arg(argc, argv, "--frames", 5);
    int source_fps = parse_int_arg(argc, argv, "--source-fps", -1);
    if (source_fps != -1 && source_fps <= 0) {
      throw std::invalid_argument("--source-fps must be positive");
    }
    if (source_fps == -1) {
      source_fps = probe_source_fps(url);
    }

    // CORE LOGIC
    // STEP configure-rtsp
    // Configure the URL, codec, source cadence, and RTSP transport.
    simaai::neat::nodes::groups::RtspDecodedInputOptions rtsp_opt;
    rtsp_opt.url = url;
    rtsp_opt.codec = parse_codec(codec_name);
    rtsp_opt.source_fps = source_fps;
    rtsp_opt.tcp = true;
    // END STEP

    // STEP compose-graph
    // Build a Graph whose only stages are the RTSP group and an Output node.
    simaai::neat::Graph graph;
    graph.add(simaai::neat::nodes::groups::RtspDecodedInput(rtsp_opt));
    graph.add(simaai::neat::nodes::Output());
    auto run = graph.build(simaai::neat::RunOptions{});
    // END STEP

    // STEP pull-frames
    for (int i = 0; i < frames; ++i) {
      auto sample = run.pull(/*timeout_ms=*/5000);
      if (!sample.has_value() || simaai::neat::tensors_from_sample(*sample, true).empty()) {
        std::cout << "frame=" << i << " rtsp_timeout\n";
        break;
      }
      const auto tensors = simaai::neat::tensors_from_sample(*sample, true);
      const auto& shape = tensors.front().shape;
      std::cout << "frame=" << i << " shape=[";
      for (std::size_t d = 0; d < shape.size(); ++d) {
        std::cout << shape[d] << (d + 1 < shape.size() ? ", " : "");
      }
      std::cout << "]\n";
    }
    // END STEP
    // END CORE LOGIC

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

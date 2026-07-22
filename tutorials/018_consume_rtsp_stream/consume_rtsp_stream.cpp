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

#include <array>
#include <cmath>
#include <cstdio>
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

std::string shell_quote(const std::string& value) {
  std::string out = "'";
  for (const char c : value) {
    out += c == '\'' ? "'\\''" : std::string(1, c);
  }
  out += "'";
  return out;
}

int fps_from_rate(const std::string& value) {
  try {
    const auto slash = value.find('/');
    const double numerator = std::stod(value.substr(0, slash));
    const double denominator =
        slash == std::string::npos ? 1.0 : std::stod(value.substr(slash + 1));
    const double fps = denominator > 0.0 ? numerator / denominator : 0.0;
    return fps > 0.0 ? static_cast<int>(std::lround(fps)) : 0;
  } catch (...) {
    return 0;
  }
}

int probe_source_fps(const std::string& url) {
  const std::string command = "ffprobe -v error -rw_timeout 5000000 -select_streams v:0 "
                              "-show_entries stream=avg_frame_rate,r_frame_rate -of default=nw=1 " +
                              shell_quote(url) + " 2>/dev/null";
  FILE* pipe = ::popen(command.c_str(), "r");
  if (!pipe) {
    throw std::runtime_error("failed to run ffprobe for RTSP source FPS");
  }

  int average_fps = 0;
  int reported_fps = 0;
  std::array<char, 256> buffer{};
  while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) {
    const std::string line(buffer.data());
    const auto separator = line.find('=');
    if (separator == std::string::npos) {
      continue;
    }
    const std::string key = line.substr(0, separator);
    const int fps = fps_from_rate(line.substr(separator + 1));
    if (key == "avg_frame_rate") {
      average_fps = fps;
    } else if (key == "r_frame_rate") {
      reported_fps = fps;
    }
  }
  const int status = ::pclose(pipe);
  if (status != 0) {
    throw std::runtime_error("ffprobe failed to probe RTSP source FPS");
  }

  const int fps = average_fps > 0 ? average_fps : reported_fps;
  if (fps <= 0) {
    throw std::runtime_error("ffprobe did not report a positive RTSP source FPS");
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

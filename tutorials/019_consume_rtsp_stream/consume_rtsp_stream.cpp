// Consume a live H.264 RTSP stream via the RtspDecodedInput node group.
//
// The group handles RTSP connect, depacketize, and H.264 decode — you hand it
// a URL and pull decoded frames. This chapter is about the input group only.
//
// Usage:
//   tutorial_v2_019_consume_rtsp_stream --url rtsp://host/path [--frames 5]

#include "neat.h"
#include "nodes/groups/RtspDecodedInput.h"

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

} // namespace

int main(int argc, char** argv) {
  try {
    std::string url;
    if (!get_arg(argc, argv, "--url", url)) {
      std::cerr << "Usage: tutorial_v2_019_consume_rtsp_stream --url <rtsp://...> [--frames <n>]\n";
      return 1;
    }
    const int frames = parse_int_arg(argc, argv, "--frames", 5);

    // CORE LOGIC
    // Configure RtspDecodedInputOptions, build a Session whose only stages
    // are the RTSP group and an Output node, and pull decoded frames.
    simaai::neat::nodes::groups::RtspDecodedInputOptions rtsp_opt;
    rtsp_opt.url = url;
    rtsp_opt.tcp = true;

    simaai::neat::Session s;
    s.add(simaai::neat::nodes::groups::RtspDecodedInput(rtsp_opt));
    s.add(simaai::neat::nodes::Output());
    auto run = s.build(simaai::neat::RunOptions{});

    for (int i = 0; i < frames; ++i) {
      auto sample = run.pull(/*timeout_ms=*/5000);
      if (!sample.has_value() || !sample->tensor.has_value()) {
        std::cout << "frame=" << i << " rtsp_timeout\n";
        break;
      }
      const auto& shape = sample->tensor->shape;
      std::cout << "frame=" << i << " shape=[";
      for (std::size_t d = 0; d < shape.size(); ++d) {
        std::cout << shape[d] << (d + 1 < shape.size() ? ", " : "");
      }
      std::cout << "]\n";
    }
    // END CORE LOGIC

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

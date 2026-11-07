#include "graphs/Fragments.h"
#include "nodes/common/Output.h"
#include "nodes/groups/RtspDecodedInput.h"
#include "nodes/io/Input.h"
#include "nodes/io/StillImageInput.h"
#include "nodes/sima/H264Packetize.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/SimaEncode.h"
#include "pipeline/Graph.h"
#include "rtsp_port_utils.h"

#include "cli_utils.h"
#include "test_utils.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

namespace {

namespace neat = simaai::neat;

constexpr int kFps = 45;
constexpr int kKeyframeInterval = 30;
constexpr int kKeyframes = 12;
constexpr int kWarmupFrames = 2 * kKeyframeInterval;
constexpr int kPullTimeoutMs = 2000;
constexpr double kFrameNs = 1e9 / kFps;
constexpr std::chrono::milliseconds kConsumerTime(12);

neat::RtspServerHandle start_keyframe_stream(const std::string& image_path) {
  neat::SimaEncodeOptions encode;
  encode.fps = kFps;
  encode.gop_length = kKeyframeInterval;
  encode.idr_interval = kKeyframeInterval;

  neat::Graph server;
  server.add(neat::nodes::StillImageInput(image_path, 1280, 720, 1280, 720, kFps));
  server.add(neat::nodes::SimaEncode(encode));
  server.add(neat::nodes::H264Parse(/*config_interval=*/1));
  server.add(neat::nodes::H264Packetize(/*pt=*/96, /*config_interval=*/1));

  constexpr int kRtpPortOffset = 10000;
  constexpr int kRtpPorts = 8;
  const int port =
      rtsp_find_free_port_range_with_rtp(8554, 1, 1, 32, kRtpPortOffset, kRtpPorts, kRtpPorts);
  require(port > 0, "no free RTSP port");
  return server.run_rtsp({
      .mount = "keyframes",
      .port = port,
      .rtp_port_base = port + kRtpPortOffset,
      .rtp_port_count = kRtpPorts,
  });
}

neat::Graph sink_branch(const std::string& name) {
  neat::Graph sink(name);
  sink.add(neat::nodes::Input(name));
  sink.add(neat::nodes::Output(name + "_out", neat::OutputOptions::EveryFrame(4)));
  return sink;
}

neat::Sample pull_frame(neat::Run& run, const std::string& output) {
  neat::Sample frame;
  neat::PullError error;
  const neat::PullStatus status = run.pull(output, kPullTimeoutMs, frame, &error);
  require(status == neat::PullStatus::Ok, "pull from " + output + " failed: " + error.message);
  require(frame.pts_ns >= 0, "decoded frame has no timestamp");
  return frame;
}

// Each output has its own queue, so either one can lose frames without the other.
struct OutputTimeline {
  std::string name;
  std::int64_t previous_pts_ns = -1;
  int lost = 0;
};

int pull_and_count_frames_advanced(neat::Run& run, OutputTimeline& output) {
  const std::int64_t pts_ns = pull_frame(run, output.name).pts_ns;
  const int advanced = static_cast<int>(std::lround((pts_ns - output.previous_pts_ns) / kFrameNs));
  require(advanced > 0, output.name + " left the stream out of order");
  if (advanced > 1) {
    std::cerr << "[rtsp_keyframe_frame_loss] " << output.name << " lost " << advanced - 1
              << " frames between pts " << output.previous_pts_ns << " and " << pts_ns << " ns\n";
  }
  output.lost += advanced - 1;
  output.previous_pts_ns = pts_ns;
  return advanced;
}

// The consumer is slower than a burst of frames and faster than the stream, so
// it keeps up unless the decoder releases frames faster than the stream's rate.
int frames_lost(neat::Run& run) {
  OutputTimeline frames{"frames_out"};
  OutputTimeline copy{"copy_out"};
  for (int n = 0; n < kWarmupFrames; ++n) {
    frames.previous_pts_ns = pull_frame(run, frames.name).pts_ns;
    copy.previous_pts_ns = pull_frame(run, copy.name).pts_ns;
  }

  for (int n = 0; n < kKeyframes * kKeyframeInterval;) {
    std::this_thread::sleep_for(kConsumerTime);
    n += pull_and_count_frames_advanced(run, frames);
    pull_and_count_frames_advanced(run, copy);
  }
  std::cout << "[rtsp_keyframe_frame_loss] frames_out lost " << frames.lost << ", copy_out lost "
            << copy.lost << "\n";
  return frames.lost + copy.lost;
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::string image_path = "test.jpg";
    sima_test::get_arg(argc, argv, "--image", image_path);

    neat::RtspServerHandle server = start_keyframe_stream(image_path);

    neat::nodes::groups::RtspDecodedInputOptions input;
    input.url = server.url();
    // Two branches give the source an internal boundary, which drops frames when
    // full. A single Output would hold the decoder back instead of losing frames.
    neat::GraphLinkOptions live;
    live.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
    neat::Graph branch = neat::graphs::Branch("source", {"frames", "copy"});
    neat::Graph graph;
    graph.connect(neat::nodes::groups::RtspDecodedInput(input), branch);
    graph.connect(branch, sink_branch("frames"), live);
    graph.connect(branch, sink_branch("copy"), live);

    neat::RunOptions options;
    options.preset = neat::RunPreset::Balanced;
    options.queue_depth = 3;
    options.overflow_policy = neat::OverflowPolicy::Block;
    options.output_memory = neat::OutputMemory::Owned;
    neat::Run run = graph.build(options);

    const int lost = frames_lost(run);
    run.close();
    server.stop();

    std::cout << "[rtsp_keyframe_frame_loss] lost " << lost << " frames across " << kKeyframes
              << " keyframes\n";
    require(lost == 0, "frames were lost around keyframes");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << "\n";
    return 1;
  }
}

// An encoded source split to a decoder branch and VideoSender passthroughs must keep flowing:
// every sender transmits every frame, and a pooled source keeps recycling its buffers. Mirrors
// the USB camera passthrough graph with system-memory stand-ins for the capture buffers.
#include "pipeline/Graph.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/TensorAdapters.h"
#include "gst/GstInit.h"
#include "nodes/common/Caps.h"
#include "nodes/common/JpegDecode.h"
#include "nodes/common/Output.h"
#include "nodes/groups/VideoSender.h"
#include "nodes/io/Input.h"
#include "test_utils.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace simaai::neat;
namespace groups = simaai::neat::nodes::groups;

constexpr int kPoolSize = 4;
constexpr int kFrames = 40;
constexpr int kPortA = 9311;
constexpr int kPortB = 9312;
// The camera's caps, with JPEGs of that size, so jpegparse and rtpjpegpay see real input.
const char* const kCaps =
    "image/jpeg, width=(int)1920, height=(int)1080, framerate=(fraction)30/1, "
    "parsed=(boolean)true, pixel-aspect-ratio=(fraction)1/1, "
    "interlace-mode=(string)progressive, colorimetry=(string)2:4:5:1";

std::vector<std::uint8_t> encode_test_jpeg(const char* pattern) {
  const std::string launch = std::string("videotestsrc num-buffers=1 pattern=") + pattern +
                             " ! video/x-raw,format=I420,width=1920,height=1080 ! jpegenc ! "
                             "appsink name=sink";
  GstElement* pipeline = gst_parse_launch(launch.c_str(), nullptr);
  require(pipeline != nullptr, "failed to build JPEG fixture pipeline");
  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 5 * GST_SECOND);
  require(sample != nullptr, "JPEG fixture pull timed out");
  GstMapInfo map{};
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  require(gst_buffer_map(buffer, &map, GST_MAP_READ), "failed to map JPEG fixture");
  std::vector<std::uint8_t> bytes(map.data, map.data + map.size);
  gst_buffer_unmap(buffer, &map);
  gst_sample_unref(sample);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(sink);
  gst_object_unref(pipeline);
  return bytes;
}

GstBuffer* acquire_within(GstBufferPool* pool, std::chrono::milliseconds limit) {
  GstBufferPoolAcquireParams params{};
  params.flags = GST_BUFFER_POOL_ACQUIRE_FLAG_DONTWAIT;
  const auto deadline = std::chrono::steady_clock::now() + limit;
  while (std::chrono::steady_clock::now() < deadline) {
    GstBuffer* buffer = nullptr;
    if (gst_buffer_pool_acquire_buffer(pool, &buffer, &params) == GST_FLOW_OK) {
      return buffer;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return nullptr;
}

class UdpCounter {
public:
  explicit UdpCounter(int port) : fd_(::socket(AF_INET, SOCK_DGRAM, 0)) {
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<std::uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    require(fd_ >= 0 && ::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0,
            "failed to bind UDP listener on port " + std::to_string(port));
  }
  ~UdpCounter() {
    ::close(fd_);
  }
  UdpCounter(const UdpCounter&) = delete;
  UdpCounter& operator=(const UdpCounter&) = delete;
  int drain() {
    char scratch[65536];
    while (::recv(fd_, scratch, sizeof(scratch), MSG_DONTWAIT) > 0) {
      ++packets_;
    }
    return packets_;
  }

private:
  int fd_;
  int packets_ = 0;
};

Graph sender(int port) {
  auto opt = groups::VideoSenderOptions::Passthrough(groups::RtspCodec::MJPEG);
  opt.host = "127.0.0.1";
  opt.video_port_base = port;
  return groups::VideoSender(opt);
}

Graph decoder() {
  Graph graph("drain_decode");
  graph.add(nodes::JpegDecode());
  graph.add(nodes::Output());
  return graph;
}

// A pushed source whose frames come from a small pool, so a branch that keeps frames stalls it.
void pooled_source_keeps_recycling(GraphLinkPolicy policy, const char* label) {
  const std::vector<std::vector<std::uint8_t>> jpegs{encode_test_jpeg("smpte"),
                                                     encode_test_jpeg("snow")};
  GstCaps* caps = gst_caps_from_string(kCaps);
  GstBufferPool* pool = gst_buffer_pool_new();
  GstStructure* config = gst_buffer_pool_get_config(pool);
  const auto max_size = std::max(jpegs[0].size(), jpegs[1].size());
  gst_buffer_pool_config_set_params(config, caps, static_cast<guint>(max_size), kPoolSize,
                                    kPoolSize);
  require(gst_buffer_pool_set_config(pool, config), "failed to configure source pool");
  require(gst_buffer_pool_set_active(pool, TRUE), "failed to activate source pool");
  UdpCounter udp(kPortA);

  InputOptions src_opt;
  src_opt.payload_type = PayloadType::Encoded;
  src_opt.format = FormatTag::ENCODED;
  src_opt.caps_override = kCaps;
  src_opt.memory_policy = InputMemoryPolicy::SystemMemory;
  Graph source("drain_source");
  source.add(nodes::Input(src_opt));
  GraphLinkOptions link;
  link.policy = policy;
  Graph split("drain_split");
  split.connect(source, decoder());
  split.connect(source, sender(kPortA), link);
  Run run = split.build();

  for (int frame = 0; frame < kFrames; ++frame) {
    GstBuffer* buffer = acquire_within(pool, std::chrono::milliseconds(2000));
    require(buffer != nullptr,
            std::string(label) + ": source pool exhausted at frame " + std::to_string(frame));
    // Real cameras emit a different JPEG size every frame.
    const auto& bytes = jpegs[static_cast<std::size_t>(frame) % jpegs.size()];
    GstMapInfo map{};
    require(gst_buffer_map(buffer, &map, GST_MAP_WRITE), "failed to map pooled buffer");
    std::memcpy(map.data, bytes.data(), bytes.size());
    gst_buffer_unmap(buffer, &map);
    gst_buffer_set_size(buffer, static_cast<gssize>(bytes.size()));
    GST_BUFFER_PTS(buffer) = static_cast<GstClockTime>(frame) * GST_SECOND / 30;
    GstSample* gst_sample = gst_sample_new(buffer, caps, nullptr, nullptr);
    gst_buffer_unref(buffer);
    Sample sample;
    sample.kind = SampleKind::TensorSet;
    sample.payload_type = PayloadType::Encoded;
    sample.media_type = "image/jpeg";
    sample.caps_string = kCaps;
    sample.frame_id = frame;
    sample.tensors = TensorList{from_gst_sample(gst_sample)};
    gst_sample_unref(gst_sample);
    require(run.push(sample),
            std::string(label) + ": push failed at frame " + std::to_string(frame));
    sample = Sample{};
    require(run.pull(3000).has_value(),
            std::string(label) + ": no decoded frame " + std::to_string(frame));
    udp.drain();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  const int packets = udp.drain();
  std::cout << "[INFO] " << label << ": " << packets << " RTP packets\n";
  // A 1080p JPEG needs well over one RTP packet; every frame must have been sent.
  require(packets >= 2 * kFrames, std::string(label) + ": VideoSender did not send every frame");
  run.stop();
  gst_buffer_pool_set_active(pool, FALSE);
  gst_object_unref(pool);
  gst_caps_unref(caps);
}

// A live source fanned out to a decoder and two senders: both senders copy the same encoded
// Sample at once.
void live_source_feeds_two_senders() {
  UdpCounter udp_a(kPortA);
  UdpCounter udp_b(kPortB);
  Graph source("drain_live_source");
  source.add(nodes::Custom("videotestsrc is-live=true ! video/x-raw,format=I420,width=1920,"
                           "height=1080,framerate=30/1 ! jpegenc ! image/jpeg",
                           InputRole::Source));
  Graph split("drain_live_split");
  split.connect(source, decoder());
  split.connect(source, sender(kPortA));
  split.connect(source, sender(kPortB));
  Run run = split.build();
  for (int frame = 0; frame < kFrames; ++frame) {
    require(run.pull(3000).has_value(), "live: no decoded frame " + std::to_string(frame));
    udp_a.drain();
    udp_b.drain();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  const int packets_a = udp_a.drain();
  const int packets_b = udp_b.drain();
  std::cout << "[INFO] live: " << packets_a << " / " << packets_b << " RTP packets\n";
  require(packets_a >= 2 * kFrames && packets_b >= 2 * kFrames,
          "live: a VideoSender did not send every frame");
  run.stop();
}

} // namespace

int main() {
  try {
    gst_init_once();
    pooled_source_keeps_recycling(GraphLinkPolicy::Default, "pooled default link");
    pooled_source_keeps_recycling(GraphLinkPolicy::RealtimeLatestByStream, "pooled realtime link");
    live_source_feeds_two_senders();
  } catch (const std::exception& e) {
    return fail_test(e.what());
  }
  std::cout << "[OK] unit_video_sender_passthrough_drain_test\n";
  return 0;
}

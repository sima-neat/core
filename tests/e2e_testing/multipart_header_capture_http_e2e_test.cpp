/**
 * End-to-end per-frame attribute association over a real HTTP multipart stream.
 *
 * A finite localhost server on an ephemeral port serves distinguishable JPEG frames whose
 * selected headers alternate between present, absent, empty, and repeated. The pipeline
 * fans out through a tee into two queues with asymmetric delay, so a frame is still in
 * flight on the slow branch while later frames are already delivered on the fast one.
 *
 * This proves association through parsing, HTTP chunk boundaries, queues, branches, and
 * downstream consumption. The decode hop is covered separately by the decoder's own
 * correlation tests: this test deliberately terminates before the decoder so it can run
 * without hardware decode.
 */
#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "gst/GstSampleAttributes.h"

#include "test_main.h"
#include "test_utils.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace {

using simaai::neat::SampleAttributes;

constexpr int kFrameCount = 12;
constexpr const char* kBoundary = "frame";

/// Minimal JPEG with a real SOF0 plus a marker byte identifying the frame.
std::string make_marked_jpeg(uint8_t marker) {
  std::string jpeg;
  auto push = [&jpeg](std::initializer_list<uint8_t> bytes) {
    for (const uint8_t b : bytes) {
      jpeg.push_back(static_cast<char>(b));
    }
  };
  push({0xFF, 0xD8});
  push({0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x30, 0x00, 0x40, 0x03});
  push({0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01});
  push({0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x00});
  jpeg.push_back(static_cast<char>(marker));
  // Pad so frames are large enough to span several socket reads.
  jpeg.append(512, '\x5A');
  push({0xFF, 0xD9});
  return jpeg;
}

/// What frame `index` is expected to carry. Index is also the marker byte.
SampleAttributes expected_attributes(int index) {
  SampleAttributes expected;
  expected["image-index"] = std::to_string(100 + index);
  if (index % 3 == 0) {
    expected["image-time"] = "t" + std::to_string(index);
  } else if (index % 3 == 1) {
    expected["image-time"] = ""; // present but empty
  }
  // index % 3 == 2 -> header absent entirely
  return expected;
}

std::string build_part(int index) {
  std::string headers = "Content-Type: image/jpeg\r\n";
  headers += "Image-Index: " + std::to_string(100 + index) + "\r\n";
  if (index % 3 == 0) {
    headers += "Image-Time: t" + std::to_string(index) + "\r\n";
  } else if (index % 3 == 1) {
    headers += "Image-Time:\r\n";
  }
  if (index % 4 == 3) {
    // A repeated selected header: the last value must win, and it is the expected one.
    headers += "Image-Index: " + std::to_string(999) + "\r\n";
    headers += "Image-Index: " + std::to_string(100 + index) + "\r\n";
  }
  headers += "X-Not-Selected: noise\r\n";

  std::string out = "--";
  out += kBoundary;
  out += "\r\n" + headers + "\r\n";
  out += make_marked_jpeg(static_cast<uint8_t>(index));
  out += "\r\n";
  return out;
}

/// Finite single-connection HTTP server. Serves one multipart response and exits.
class MjpegServer {
public:
  bool start() {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
      return false;
    }
    int reuse = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
    addr.sin_port = 0; // ephemeral
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      return false;
    }
    socklen_t len = sizeof(addr);
    if (::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
      return false;
    }
    port_ = ::ntohs(addr.sin_port);
    if (::listen(listen_fd_, 1) != 0) {
      return false;
    }
    worker_ = std::thread([this] { serve(); });
    return true;
  }

  /// Joining on destruction matters: a failed assertion unwinds through this object, and a
  /// still-joinable worker would call std::terminate and hide the real failure message.
  ~MjpegServer() {
    stop();
  }

  void stop() {
    stopping_.store(true);
    if (listen_fd_ >= 0) {
      ::shutdown(listen_fd_, SHUT_RDWR);
      ::close(listen_fd_);
      listen_fd_ = -1;
    }
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  int port() const {
    return port_;
  }

private:
  void serve() {
    const int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) {
      return;
    }
    char scratch[2048];
    (void)::recv(fd, scratch, sizeof(scratch), 0); // consume the request line/headers

    std::string header = "HTTP/1.0 200 OK\r\n";
    header += "Content-Type: multipart/x-mixed-replace; boundary=";
    header += kBoundary;
    header += "\r\nConnection: close\r\n\r\n";
    (void)::send(fd, header.data(), header.size(), 0);

    std::string body;
    for (int i = 0; i < kFrameCount; ++i) {
      body += build_part(i);
    }
    body += "--";
    body += kBoundary;
    body += "--\r\n";

    // Deliberately awkward write sizes so part boundaries land mid-chunk.
    constexpr std::size_t kWrite = 277U;
    for (std::size_t off = 0; off < body.size() && !stopping_.load(); off += kWrite) {
      const std::size_t n = std::min(kWrite, body.size() - off);
      if (::send(fd, body.data() + off, n, 0) < 0) {
        break;
      }
    }
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
  }

  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> stopping_{false};
  std::thread worker_;
};

struct Frame {
  uint8_t marker = 0;
  SampleAttributes attributes;
};

std::vector<Frame> drain(GstElement* sink) {
  std::vector<Frame> frames;
  for (;;) {
    GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 5 * GST_SECOND);
    if (!sample) {
      break;
    }
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    Frame frame;
    if (buffer) {
      GstMapInfo map;
      if (gst_buffer_map(buffer, &map, GST_MAP_READ) == TRUE) {
        // SOI(2) + SOF0(10) + components(9) + SOS(9) = 30 bytes before the marker.
        constexpr std::size_t kMarkerOffset = 30U;
        if (map.size > kMarkerOffset) {
          frame.marker = map.data[kMarkerOffset];
        }
        gst_buffer_unmap(buffer, &map);
      }
      simaai::neat::gst_internal::read_attributes(buffer, &frame.attributes);
    }
    frames.push_back(frame);
    gst_sample_unref(sample);
  }
  return frames;
}

void verify_branch(const std::vector<Frame>& frames, const std::string& label) {
  require(frames.size() == static_cast<std::size_t>(kFrameCount),
          label + ": expected " + std::to_string(kFrameCount) + " frames, got " +
              std::to_string(frames.size()));
  for (int i = 0; i < kFrameCount; ++i) {
    const Frame& frame = frames[static_cast<std::size_t>(i)];
    const std::string ctx = label + " frame " + std::to_string(i);
    require(frame.marker == static_cast<uint8_t>(i),
            ctx + ": pixel marker mismatch (got " + std::to_string(frame.marker) + ")");

    const SampleAttributes expected = expected_attributes(i);
    require(frame.attributes.size() == expected.size(),
            ctx + ": attribute count mismatch (got " + std::to_string(frame.attributes.size()) +
                ", want " + std::to_string(expected.size()) + ")");
    for (const auto& [key, value] : expected) {
      const auto it = frame.attributes.find(key);
      require(it != frame.attributes.end(), ctx + ": missing attribute '" + key + "'");
      require(it->second == value,
              ctx + ": attribute '" + key + "' is '" + it->second + "', want '" + value + "'");
    }
    require(frame.attributes.count("x-not-selected") == 0U,
            ctx + ": unselected header must not be captured");
    // An absent header must not have been inherited from an earlier frame.
    if (i % 3 == 2) {
      require(frame.attributes.count("image-time") == 0U,
              ctx + ": absent header leaked from an earlier frame");
    }
  }
}

void test_http_multipart_capture_through_branches() {
  simaai::neat::gst_init_once();
  require(simaai::neat::element_exists("souphttpsrc"), "souphttpsrc must be available");
  require(simaai::neat::element_exists("neatmultipartjpegdemux"),
          "neatmultipartjpegdemux must be registered");

  MjpegServer server;
  require(server.start(), "localhost MJPEG server must start on an ephemeral port");

  // Asymmetric branch delay: the slow branch holds each buffer long enough that the fast
  // branch is several frames ahead, which is where cross-branch aliasing would show up.
  const std::string launch =
      "souphttpsrc location=http://127.0.0.1:" + std::to_string(server.port()) +
      "/stream is-live=false ! neatmultipartjpegdemux boundary=" + std::string(kBoundary) +
      " capture-headers=\"image-index,image-time\" ! tee name=t "
      "t. ! queue max-size-buffers=64 ! appsink name=fast sync=false max-buffers=64 "
      "t. ! queue max-size-buffers=64 ! identity sleep-time=15000 ! appsink name=slow "
      "sync=false max-buffers=64";

  GError* error = nullptr;
  GstElement* pipeline = gst_parse_launch(launch.c_str(), &error);
  if (error) {
    const std::string message = error->message ? error->message : "unknown";
    g_error_free(error);
    server.stop();
    throw std::runtime_error("failed to build capture e2e pipeline: " + message);
  }
  require(pipeline != nullptr, "capture e2e pipeline must be created");

  GstElement* fast = gst_bin_get_by_name(GST_BIN(pipeline), "fast");
  GstElement* slow = gst_bin_get_by_name(GST_BIN(pipeline), "slow");
  require(fast && slow, "both appsinks must be present");

  require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
          "capture e2e pipeline must reach PLAYING");

  const std::vector<Frame> fast_frames = drain(fast);
  const std::vector<Frame> slow_frames = drain(slow);

  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(fast);
  gst_object_unref(slow);
  gst_object_unref(pipeline);
  server.stop();

  verify_branch(fast_frames, "fast branch");
  verify_branch(slow_frames, "slow branch");

  // Both branches must observe identical association; a shared or aliased attribute map
  // would show up here as a divergence.
  for (std::size_t i = 0; i < fast_frames.size(); ++i) {
    require(fast_frames[i].attributes == slow_frames[i].attributes,
            "branches disagree on attributes for frame " + std::to_string(i));
  }
}

} // namespace

RUN_TEST("multipart_header_capture_http_e2e",
         [] { test_http_multipart_capture_through_branches(); })

/**
 * Through-decode per-frame attribute association on real hardware.
 *
 * This is the correctness gate for the whole feature: it proves that a header captured from
 * an HTTP multipart part is still attached to the *same picture* after the picture has gone
 * through the SiMa decoder, an output buffer pool, and asymmetric branch delay.
 *
 * Each frame is a solid-grey JPEG with a distinct luma level, so the decoded NV12 pixels
 * identify the frame independently of arrival order. The test asserts pixel identity and
 * attributes together — a shifted, stale, or cross-branch attribute shows up as a mismatch
 * between what the pixels say and what the attributes say.
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
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using simaai::neat::SampleAttributes;

constexpr int kFrameCount = 12;
constexpr const char* kBoundary = "frame";
constexpr int kWidth = 320;
constexpr int kHeight = 240;

/// Grey level encoded into frame `index`. Widely spaced so JPEG loss cannot confuse two.
int grey_for(int index) {
  return 24 + index * 18;
}

/// Encode one solid-grey JPEG through GStreamer so the bytes are genuinely decodable.
std::string encode_grey_jpeg(int grey) {
  const guint32 argb = 0xFF000000u | (static_cast<guint32>(grey) << 16) |
                       (static_cast<guint32>(grey) << 8) | static_cast<guint32>(grey);
  std::string launch = "videotestsrc pattern=solid-color foreground-color=" + std::to_string(argb) +
                       " num-buffers=1 ! " + "video/x-raw,width=" + std::to_string(kWidth) +
                       ",height=" + std::to_string(kHeight) +
                       ",format=I420 ! jpegenc quality=95 ! appsink name=out sync=false";

  GError* error = nullptr;
  GstElement* pipeline = gst_parse_launch(launch.c_str(), &error);
  if (error) {
    const std::string message = error->message ? error->message : "unknown";
    g_error_free(error);
    throw std::runtime_error("jpeg encode pipeline failed: " + message);
  }
  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "out");
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  std::string jpeg;
  GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 10 * GST_SECOND);
  if (sample) {
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstMapInfo map;
    if (buffer && gst_buffer_map(buffer, &map, GST_MAP_READ) == TRUE) {
      jpeg.assign(reinterpret_cast<const char*>(map.data), map.size);
      gst_buffer_unmap(buffer, &map);
    }
    gst_sample_unref(sample);
  }
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(sink);
  gst_object_unref(pipeline);
  return jpeg;
}

SampleAttributes expected_attributes(int index) {
  SampleAttributes expected;
  expected["image-index"] = std::to_string(100 + index);
  if (index % 3 == 0) {
    expected["image-time"] = "t" + std::to_string(index);
  } else if (index % 3 == 1) {
    expected["image-time"] = ""; // present but empty
  }
  // index % 3 == 2 -> absent entirely
  return expected;
}

std::string build_part(int index, const std::string& jpeg) {
  std::string headers = "Content-Type: image/jpeg\r\n";
  headers += "Image-Index: " + std::to_string(100 + index) + "\r\n";
  if (index % 3 == 0) {
    headers += "Image-Time: t" + std::to_string(index) + "\r\n";
  } else if (index % 3 == 1) {
    headers += "Image-Time:\r\n";
  }
  headers += "X-Not-Selected: noise\r\n";

  std::string out = "--";
  out += kBoundary;
  out += "\r\n" + headers + "\r\n";
  out += jpeg;
  out += "\r\n";
  return out;
}

class MjpegServer {
public:
  bool start(std::string body) {
    body_ = std::move(body);
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
      return false;
    }
    int reuse = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
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
    char scratch[4096];
    (void)::recv(fd, scratch, sizeof(scratch), 0);

    std::string header = "HTTP/1.0 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=";
    header += kBoundary;
    header += "\r\nConnection: close\r\n\r\n";
    (void)::send(fd, header.data(), header.size(), 0);

    constexpr std::size_t kWrite = 1409U; // awkward size: boundaries land mid-chunk
    for (std::size_t off = 0; off < body_.size() && !stopping_.load(); off += kWrite) {
      const std::size_t n = std::min(kWrite, body_.size() - off);
      if (::send(fd, body_.data() + off, n, 0) < 0) {
        break;
      }
    }
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
  }

  int listen_fd_ = -1;
  int port_ = 0;
  std::string body_;
  std::atomic<bool> stopping_{false};
  std::thread worker_;
};

struct Decoded {
  int luma = -1;
  SampleAttributes attributes;
};

/// Average luma over a central patch, away from any edge artifacts.
int center_luma(GstBuffer* buffer) {
  GstMapInfo map;
  if (!buffer || gst_buffer_map(buffer, &map, GST_MAP_READ) != TRUE) {
    return -1;
  }
  long total = 0;
  int count = 0;
  // NV12: luma plane is width*height at the start. Sample a small central block.
  const int stride = kWidth;
  for (int y = kHeight / 2 - 4; y < kHeight / 2 + 4; ++y) {
    for (int x = kWidth / 2 - 4; x < kWidth / 2 + 4; ++x) {
      const std::size_t offset = static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x);
      if (offset < map.size) {
        total += map.data[offset];
        ++count;
      }
    }
  }
  gst_buffer_unmap(buffer, &map);
  return count > 0 ? static_cast<int>(total / count) : -1;
}

/// Decode one input JPEG through the software decoder. This creates the pixel oracle without
/// consulting the frame attributes whose association the hardware path is meant to prove.
int decode_reference_luma(const std::string& jpeg) {
  const std::string launch =
      "appsrc name=source format=bytes caps=image/jpeg ! jpegdec ! videoconvert ! "
      "video/x-raw,format=NV12,width=" +
      std::to_string(kWidth) + ",height=" + std::to_string(kHeight) +
      " ! appsink name=sink sync=false max-buffers=1";

  GError* error = nullptr;
  GstElement* pipeline = gst_parse_launch(launch.c_str(), &error);
  if (error || !pipeline) {
    const std::string message = error && error->message ? error->message : "unknown";
    if (error) {
      g_error_free(error);
    }
    if (pipeline) {
      gst_object_unref(pipeline);
    }
    throw std::runtime_error("software reference decode pipeline failed: " + message);
  }

  GstElement* source = gst_bin_get_by_name(GST_BIN(pipeline), "source");
  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
  if (!source || !sink ||
      gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    if (source) {
      gst_object_unref(source);
    }
    if (sink) {
      gst_object_unref(sink);
    }
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    throw std::runtime_error("software reference decode pipeline could not start");
  }

  GstBuffer* input = gst_buffer_new_allocate(nullptr, jpeg.size(), nullptr);
  if (!input || gst_buffer_fill(input, 0U, jpeg.data(), jpeg.size()) != jpeg.size()) {
    if (input) {
      gst_buffer_unref(input);
    }
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(source);
    gst_object_unref(sink);
    gst_object_unref(pipeline);
    throw std::runtime_error("software reference decode input allocation failed");
  }

  const GstFlowReturn push_flow = gst_app_src_push_buffer(GST_APP_SRC(source), input);
  const GstFlowReturn eos_flow = gst_app_src_end_of_stream(GST_APP_SRC(source));
  GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 10 * GST_SECOND);
  const int luma = sample ? center_luma(gst_sample_get_buffer(sample)) : -1;
  if (sample) {
    gst_sample_unref(sample);
  }

  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(source);
  gst_object_unref(sink);
  gst_object_unref(pipeline);

  if (push_flow != GST_FLOW_OK || eos_flow != GST_FLOW_OK || luma < 0) {
    throw std::runtime_error("software reference decode produced no usable frame");
  }
  return luma;
}

std::vector<Decoded> drain(GstElement* sink) {
  std::vector<Decoded> frames;
  for (;;) {
    GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 8 * GST_SECOND);
    if (!sample) {
      break;
    }
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    Decoded decoded;
    decoded.luma = center_luma(buffer);
    simaai::neat::gst_internal::read_attributes(buffer, &decoded.attributes);
    frames.push_back(decoded);
    gst_sample_unref(sample);
  }
  return frames;
}

/// Map an observed luma back to the frame index that produced it.
int index_from_luma(int luma, const std::vector<int>& reference) {
  int best = -1;
  int best_delta = 1 << 30;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const int delta = std::abs(luma - reference[i]);
    if (delta < best_delta) {
      best_delta = delta;
      best = static_cast<int>(i);
    }
  }
  // Grey levels are 18 apart; anything beyond half that is not a confident match.
  return best_delta <= 8 ? best : -1;
}

void verify_branch(const std::vector<Decoded>& frames, const std::vector<int>& reference,
                   const std::string& label) {
  require(frames.size() == static_cast<std::size_t>(kFrameCount),
          label + ": expected " + std::to_string(kFrameCount) + " decoded frames, got " +
              std::to_string(frames.size()));

  std::vector<bool> seen(kFrameCount, false);
  for (std::size_t position = 0; position < frames.size(); ++position) {
    const Decoded& frame = frames[position];
    const int index = index_from_luma(frame.luma, reference);
    const std::string ctx = label + " position " + std::to_string(position) + " (luma " +
                            std::to_string(frame.luma) + ")";
    require(index >= 0, ctx + ": decoded pixels match no known frame");
    require(!seen[static_cast<std::size_t>(index)],
            ctx + ": frame " + std::to_string(index) + " decoded more than once");
    seen[static_cast<std::size_t>(index)] = true;

    const SampleAttributes expected = expected_attributes(index);
    const std::string what = ctx + " -> frame " + std::to_string(index);
    require(frame.attributes.size() == expected.size(),
            what + ": attribute count mismatch (got " + std::to_string(frame.attributes.size()) +
                ", want " + std::to_string(expected.size()) + ")");
    for (const auto& [key, value] : expected) {
      const auto it = frame.attributes.find(key);
      require(it != frame.attributes.end(), what + ": missing attribute '" + key + "'");
      require(it->second == value, what + ": attribute '" + key + "' is '" + it->second +
                                       "', want '" + value +
                                       "' — attributes followed the wrong "
                                       "picture through decode");
    }
    if (index % 3 == 2) {
      require(frame.attributes.count("image-time") == 0U,
              what + ": an absent header was inherited across decode");
    }
  }
  for (int i = 0; i < kFrameCount; ++i) {
    require(seen[static_cast<std::size_t>(i)],
            label + ": frame " + std::to_string(i) + " never arrived");
  }
}

void test_attributes_survive_decode() {
  simaai::neat::gst_init_once();
  require(simaai::neat::element_exists("neatdecoder"),
          "neatdecoder must be available for the through-decode gate");
  require(simaai::neat::element_exists("neatmultipartjpegdemux"),
          "neatmultipartjpegdemux must be registered");

  std::string body;
  std::vector<int> reference;
  reference.reserve(kFrameCount);
  for (int i = 0; i < kFrameCount; ++i) {
    const std::string jpeg = encode_grey_jpeg(grey_for(i));
    require(jpeg.size() > 128U, "encoded JPEG for frame " + std::to_string(i) + " is too small");
    reference.push_back(decode_reference_luma(jpeg));
    body += build_part(i, jpeg);
  }
  body += "--";
  body += kBoundary;
  body += "--\r\n";

  MjpegServer server;
  require(server.start(body), "localhost MJPEG server must start");

  const std::string launch =
      "souphttpsrc location=http://127.0.0.1:" + std::to_string(server.port()) +
      "/stream is-live=false ! neatmultipartjpegdemux boundary=" + std::string(kBoundary) +
      " capture-headers=\"image-index,image-time\" ! "
      "neatdecoder name=dec dec-type=mjpeg sima-allocator-type=2 dec-fmt=NV12 dec-width=" +
      std::to_string(kWidth) + " dec-height=" + std::to_string(kHeight) +
      " dec-fps=30 ! video/x-raw,format=NV12 ! tee name=t "
      "t. ! queue max-size-buffers=32 ! appsink name=fast sync=false max-buffers=32 "
      "t. ! queue max-size-buffers=32 ! identity sleep-time=20000 ! appsink name=slow "
      "sync=false max-buffers=32";

  GError* error = nullptr;
  GstElement* pipeline = gst_parse_launch(launch.c_str(), &error);
  if (error) {
    const std::string message = error->message ? error->message : "unknown";
    g_error_free(error);
    server.stop();
    throw std::runtime_error("failed to build decode e2e pipeline: " + message);
  }

  GstElement* fast = gst_bin_get_by_name(GST_BIN(pipeline), "fast");
  GstElement* slow = gst_bin_get_by_name(GST_BIN(pipeline), "slow");
  require(fast && slow, "both appsinks must be present");

  const GstStateChangeReturn state_ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
  if (state_ret == GST_STATE_CHANGE_FAILURE) {
    // Surface why, otherwise a decoder/caps problem is indistinguishable from a test bug.
    std::string detail;
    GstBus* bus = gst_element_get_bus(pipeline);
    if (bus) {
      while (GstMessage* msg = gst_bus_pop_filtered(
                 bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING))) {
        GError* err = nullptr;
        gchar* dbg = nullptr;
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
          gst_message_parse_error(msg, &err, &dbg);
        } else {
          gst_message_parse_warning(msg, &err, &dbg);
        }
        detail += std::string("\n  [") + GST_OBJECT_NAME(GST_MESSAGE_SRC(msg)) + "] " +
                  (err && err->message ? err->message : "?") + " | " + (dbg ? dbg : "");
        if (err) {
          g_error_free(err);
        }
        g_free(dbg);
        gst_message_unref(msg);
      }
      gst_object_unref(bus);
    }
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(fast);
    gst_object_unref(slow);
    gst_object_unref(pipeline);
    throw std::runtime_error("decode e2e pipeline must reach PLAYING;" +
                             (detail.empty() ? std::string(" no bus detail") : detail));
  }

  const std::vector<Decoded> fast_frames = drain(fast);
  const std::vector<Decoded> slow_frames = drain(slow);

  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(fast);
  gst_object_unref(slow);
  gst_object_unref(pipeline);
  server.stop();

  // Every grey level must be distinct enough to identify a frame on its own.
  for (int i = 0; i < kFrameCount; ++i) {
    require(reference[static_cast<std::size_t>(i)] >= 0,
            "no luma reference learned for frame " + std::to_string(i));
    for (int j = i + 1; j < kFrameCount; ++j) {
      require(std::abs(reference[static_cast<std::size_t>(i)] -
                       reference[static_cast<std::size_t>(j)]) > 8,
              "grey levels for frames " + std::to_string(i) + " and " + std::to_string(j) +
                  " are not distinguishable after decode");
    }
  }

  verify_branch(fast_frames, reference, "fast branch");
  verify_branch(slow_frames, reference, "slow branch");
}

} // namespace

RUN_TEST("multipart_header_capture_decode_e2e", [] { test_attributes_survive_decode(); })

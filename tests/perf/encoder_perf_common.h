#pragma once
#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "perf_metrics_common.h"
#include "test_utils.h"
#include "udp_test_utils.h"
#include <gst/gst.h>
#include <atomic>
#include <mutex>

namespace sima_encoder_perf {
using sima_perf::Clock;
constexpr std::uint64_t kWarmup = 200, kMinimumFrames = 1000, kMaximumFrames = 1000000;
constexpr int kInputFps = 30, kBuffers = 4, kCaptureFrames = 64;
constexpr double kMinimumSeconds = 10, kPacedSeconds = 60;
inline GstClockTime pts_for(std::uint64_t id) {
  return gst_util_uint64_scale(id, GST_SECOND, kInputFps);
}
inline std::uint64_t id_from_pts(GstClockTime pts) {
  require(GST_CLOCK_TIME_IS_VALID(pts), "output has no PTS");
  auto id = gst_util_uint64_scale_round(pts, kInputFps, GST_SECOND);
  require(pts == pts_for(id), "output PTS differs from submitted PTS");
  return id;
}
inline std::int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
      .count();
}
inline double seconds(std::int64_t first, std::int64_t last) {
  return (last - first) / 1e9;
}

struct RtpHeader {
  std::uint16_t sequence;
  std::uint32_t timestamp, ssrc;
  bool marker;
};
inline RtpHeader header(const unsigned char* p, std::size_t size, int payload_type) {
  require(size >= 12 && p[0] >> 6U == 2 && (p[1] & 0x7fU) == payload_type, "invalid RTP header");
  const auto read32 = [&](int i) {
    return (std::uint32_t(p[i]) << 24U) | (std::uint32_t(p[i + 1]) << 16U) |
           (std::uint32_t(p[i + 2]) << 8U) | p[i + 3];
  };
  return {static_cast<std::uint16_t>((p[2] << 8U) | p[3]), read32(4), read32(8),
          (p[1] & 0x80U) != 0};
}

// Fixed submission storage avoids reallocations racing with callbacks. A release
// of attempted publishes each immutable timestamp before Run::push can complete.
// Each latency vector has exactly one writer; inspect only after Run::stop/join.
struct Accounting {
  explicit Accounting(int pt) : payload_type(pt), submitted(kMaximumFrames) {
    output_latency.reserve(65536);
    encoder_latency.reserve(65536);
  }
  std::atomic<std::uint64_t> attempted{0}, accepted{0}, completed{0}, output{0};
  std::atomic<std::uint64_t> sent_frames{0}, sent_packets{0}, received_packets{0};
  std::atomic<std::int64_t> start{0}, last_output{0}, last_encoder{0}, producer_end{0};
  std::atomic<bool> failed{false}, producer_done{false};
  std::vector<double> output_latency, encoder_latency;
  std::vector<std::vector<std::uint8_t>> reference;
  std::string error;
  int payload_type;
  std::vector<std::int64_t> submitted;
  RtpHeader first{}; // Written before the first release of sent_packets; immutable afterward.
  std::mutex error_mutex, callback_mutex;

  void fail(const std::string& message) {
    std::lock_guard lock(error_mutex);
    if (!failed)
      error = message;
    failed = true;
  }
  std::uint64_t submit() {
    const auto id = attempted.load();
    require(id < submitted.size(), "encoder perf frame limit exceeded");
    submitted[id] = now_ns();
    attempted.store(id + 1);
    return id;
  }
  void complete(GstBuffer* buffer) {
    std::lock_guard lock(callback_mutex);
    const auto now = now_ns(), id = static_cast<std::int64_t>(id_from_pts(GST_BUFFER_PTS(buffer)));
    require(id == static_cast<std::int64_t>(completed.load()) &&
                id < static_cast<std::int64_t>(attempted.load()),
            "encoded/forwarded AU missing, duplicated or reordered");
    require(gst_buffer_get_size(buffer) != 0, "empty encoded access unit");
    if (id >= kWarmup)
      encoder_latency.push_back(seconds(submitted[id], now) * 1000);
    if (id < kCaptureFrames) {
      std::vector<std::uint8_t> bytes(gst_buffer_get_size(buffer));
      require(gst_buffer_extract(buffer, 0, bytes.data(), bytes.size()) == bytes.size(),
              "capture copy failed");
      reference.push_back(std::move(bytes));
    }
    last_encoder = now;
    ++completed;
  }
  void delivered(std::uint64_t id) {
    require(id == output.load() && id < attempted.load(),
            "output missing, duplicated or reordered");
    const auto now = now_ns();
    if (id >= kWarmup)
      output_latency.push_back(seconds(submitted[id], now) * 1000);
    last_output = now;
    ++output;
  }
  std::uint64_t rtp_id(const RtpHeader& h) const {
    const std::uint64_t delta = std::uint32_t(h.timestamp - first.timestamp);
    const auto id = (delta + 1500) / 3000; // Fixed 30 FPS caps; a 90 kHz RTP clock.
    const auto error = static_cast<std::int64_t>(delta) - static_cast<std::int64_t>(id * 3000);
    require(id < kMaximumFrames && error >= -1 && error <= 1, "unexpected RTP timestamp cadence");
    return id; // +/- one tick accommodates nanosecond-to-RTP rounding, never a frame gap.
  }
  void sent(GstBuffer* buffer) {
    std::lock_guard lock(callback_mutex);
    unsigned char bytes[12];
    require(gst_buffer_extract(buffer, 0, bytes, sizeof(bytes)) == sizeof(bytes),
            "short RTP packet");
    const auto h = header(bytes, sizeof(bytes), payload_type);
    const auto id = id_from_pts(GST_BUFFER_PTS(buffer));
    if (sent_packets == 0) {
      require(id == 0, "first RTP input was lost");
      first = h;
    }
    require(h.ssrc == first.ssrc &&
                h.sequence == std::uint16_t(first.sequence + sent_packets.load()),
            "outgoing RTP sequence/SSRC changed");
    require(id == sent_frames.load() && id == rtp_id(h), "outgoing RTP frame mismatch");
    if (h.marker)
      ++sent_frames;
    ++sent_packets;
  }
  void received(const std::string& packet) {
    require(sent_packets.load() != 0, "received RTP before this sender emitted a packet");
    const auto h =
        header(reinterpret_cast<const unsigned char*>(packet.data()), packet.size(), payload_type);
    require(h.ssrc == first.ssrc &&
                h.sequence == std::uint16_t(first.sequence + received_packets.load()),
            "RTP receiver packet loss, duplication or reordering");
    require(received_packets < sent_packets && rtp_id(h) == output.load(),
            "RTP receiver frame mismatch");
    ++received_packets;
    if (h.marker)
      delivered(rtp_id(h));
  }
};

inline GstElement* find_factory(const simaai::neat::Run& run, const std::string& factory) {
  auto core = simaai::neat::run_internal::core(run);
  auto* pipeline = core ? core->pipeline.stream.pipeline_handle() : nullptr;
  require(pipeline != nullptr, "Core pipeline is unavailable for accounting");
  auto* it = gst_bin_iterate_recurse(GST_BIN(pipeline));
  require(it != nullptr, "Core pipeline cannot be inspected");
  GstElement* found = nullptr;
  unsigned count = 0;
  GValue value = G_VALUE_INIT;
  while (gst_iterator_next(it, &value) == GST_ITERATOR_OK) {
    auto* element = GST_ELEMENT(g_value_get_object(&value));
    auto* f = gst_element_get_factory(element);
    if (f && factory == gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(f))) {
      ++count;
      if (!found)
        found = GST_ELEMENT(gst_object_ref(element));
    }
    g_value_reset(&value);
  }
  g_value_unset(&value);
  gst_iterator_free(it);
  if (count != 1) {
    if (found)
      gst_object_unref(found);
    throw std::runtime_error("expected exactly one Core " + factory);
  }
  return found;
}

class PadCounter {
public:
  PadCounter(GstElement* element, const char* pad, Accounting& state, bool rtp)
      : state_(state), rtp_(rtp), pad_(gst_element_get_static_pad(element, pad)) {
    require(pad_ != nullptr, "missing accounting pad");
    probe_ = gst_pad_add_probe(
        pad_,
        static_cast<GstPadProbeType>(GST_PAD_PROBE_TYPE_BUFFER | GST_PAD_PROBE_TYPE_BUFFER_LIST),
        callback, this, nullptr);
    if (!probe_) {
      gst_object_unref(pad_);
      throw std::runtime_error("accounting probe failed");
    }
  }
  PadCounter(const PadCounter&) = delete;
  PadCounter& operator=(const PadCounter&) = delete;
  ~PadCounter() {
    gst_pad_remove_probe(pad_, probe_);
    gst_object_unref(pad_);
  }

private:
  static GstPadProbeReturn callback(GstPad*, GstPadProbeInfo* info, gpointer data) noexcept {
    auto& self = *static_cast<PadCounter*>(data);
    try {
      const auto record = [&](GstBuffer* b) {
        if (self.rtp_)
          self.state_.sent(b);
        else
          self.state_.complete(b);
      };
      if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER)
        record(GST_PAD_PROBE_INFO_BUFFER(info));
      else {
        auto* list = GST_PAD_PROBE_INFO_BUFFER_LIST(info);
        for (guint i = 0; i < gst_buffer_list_length(list); ++i)
          record(gst_buffer_list_get(list, i));
      }
    } catch (const std::exception& error) {
      self.state_.fail(error.what());
    }
    return GST_PAD_PROBE_OK;
  }
  Accounting& state_;
  bool rtp_;
  GstPad* pad_;
  gulong probe_;
};
} // namespace sima_encoder_perf

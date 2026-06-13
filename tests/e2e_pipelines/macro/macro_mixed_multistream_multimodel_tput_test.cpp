#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "graph/Graph.h"
#include "graph/GraphHelpers.h"
#include "graph/GraphMetadata.h"
#include "graph/nodes/FanOut.h"
#include "graph/nodes/Map.h"
#include "graph/nodes/StreamMetadata.h"
#include "graph/nodes/StreamScheduler.h"
#include "gst/GstHelpers.h"
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "nodes/common/Output.h"
#include "nodes/io/MetadataSender.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/io/StillImageInput.h"
#include "nodes/io/UdpOutput.h"
#include "nodes/rtp/H264Depacketize.h"
#include "nodes/sima/H264DecodeSima.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/H264Packetize.h"
#include "pipeline/Graph.h"
#include "pipeline/runtime/RunInternal.h"
#include "pipeline/internal/TensorUtil.h"

#include "asset_utils.h"
#include "cli_utils.h"
#include "e2e_pipelines/e2e_utils.h"
#include "rtsp_port_utils.h"
#include "test_utils.h"
#include "udp_test_utils.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using NodeId = simaai::neat::graph::NodeId;
using GraphRun = simaai::neat::graph::GraphRun;
using GraphRunStats = simaai::neat::graph::GraphRunStats;
using StreamMetadataDefaults = simaai::neat::graph::StreamMetadataDefaults;

constexpr int kDefaultRtspPort = 8557;
constexpr int kRtspRtpPortOffset = 10000;
constexpr int kPayloadType = 96;
constexpr int kEncWidth = 256;
constexpr int kEncHeight = 256;
constexpr int kDefaultFps = 18;
constexpr int kGoldfishId = 1; // ILSVRC2012 0-based index for "goldfish".
constexpr const char* kGoldfishUrl =
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
    "n01443537_goldfish.JPEG";

struct Args {
  fs::path root = fs::current_path();
  std::string image;
  std::string goldfish_url = kGoldfishUrl;
  std::string resnet_model;
  std::string yolo_model;
  std::string json_out;

  int streams = 16;
  int rtsp_servers = 8;
  int fps = kDefaultFps;
  int stream_width = kEncWidth;
  int stream_height = kEncHeight;
  int iters = 300;
  int resnet_lanes = 2;
  int yolo_lanes = 2;
  int port = kDefaultRtspPort;
  int rtsp_tries = 3;
  int rtsp_timeout_ms = 15000;
  int max_runtime_ms = 900000;
  int duration_ms = 0;
  int min_measured_ms = 0;
  int stall_timeout_ms = 60000;
  int warmup_per_stream = 1;
  int warmup_timeout_ms = 0;
  int scheduler_queue = 2;
  int scheduler_max_batch = 1;
  int edge_queue = 512;
  int push_timeout_ms = 20000;
  int pull_timeout_ms = 50;
  int decoder_buffers = -1;
  int yolo_top_k = 100;
  int metadata_port_base = 9900;
  int video_port_base = 9000;
  int h264_bitrate_kbps = 500;
  int processcvu_profile_every = 50;
  int processcvu_async_in_flight = -1;
  int processcvu_pool_buffers = -1;

  float yolo_score_threshold = 0.25f;
  float yolo_nms_iou = 0.50f;
  simaai::neat::BoxDecodeType yolo_decode_type = simaai::neat::BoxDecodeType::YoloV8;

  std::string branch_mode = "fanout-both"; // fanout-both|split|resnet-only|yolo-only|encoded-only|decode-only
  std::string scheduler_drop = "oldest";   // oldest|newest
  std::string output_memory = "owned";     // owned|zero-copy
  std::string source = "rtsp";             // v1 only supports RTSP
  std::string processcvu_target = "AUTO";  // AUTO|EV74|A65
  std::string processcvu_profile_jsonl;
  std::string video_host = "127.0.0.1";

  bool metadata_udp = true;
  bool insight_video = false;
  bool latency_profile = false;
  bool yolo_discard_boxes = false;
  bool print_pipeline = false;
  bool rtsp_debug = false;
  bool lossless_rtsp = false;
  bool validate_resnet_goldfish = false;
  bool require_assets = false;
  bool allow_heavy = false;
  bool serial_pipeline_build = false;
  bool gst_element_timings = false;
  bool gst_flow_debug = false;
  bool gst_boundary_probes = false;
  bool processcvu_profile_repro = false;
  bool processcvu_timeline = false;

  bool edge_queue_cli = false;
  bool push_timeout_cli = false;
  bool output_memory_cli = false;
  bool h264_bitrate_cli = false;
  bool serial_pipeline_build_cli = false;
  bool gst_element_timings_cli = false;
  bool gst_flow_debug_cli = false;
  bool gst_boundary_probes_cli = false;
  bool processcvu_profile_repro_cli = false;
  bool processcvu_timeline_cli = false;
};

enum class Branch { ResNet, Yolo, Raw, Video };

struct OutputDef {
  GraphRun::Output handle;
  NodeId node = simaai::neat::graph::kInvalidNode;
  Branch branch = Branch::ResNet;
  int lane = -1;
  std::string label;
};

struct StreamDef {
  std::string stream_id;
  int index = 0;
  std::string branch; // both|resnet|yolo
  int resnet_lane = -1;
  int yolo_lane = -1;
};

struct Counter {
  int64_t count = 0;
  Clock::time_point first{};
  Clock::time_point last{};
  bool initialized = false;
  std::vector<double> inter_output_gap_ms;
};

struct Metrics {
  int64_t total_outputs = 0;
  int64_t resnet_outputs = 0;
  int64_t yolo_outputs = 0;
  int64_t raw_outputs = 0;
  int64_t video_packets = 0;
  int64_t video_bytes = 0;
  int64_t video_send_failed = 0;
  int64_t timeouts = 0;
  int64_t stalls = 0;
  int64_t metadata_sent = 0;
  int64_t metadata_send_failed = 0;
  int64_t metadata_received = 0;
  bool max_runtime_hit = false;
  bool stall_hit = false;
  bool saw_empty_stream_id = false;
  bool first_output_seen = false;
  Clock::time_point first_output{};

  std::map<std::string, Counter> by_stream_total;
  std::map<std::string, Counter> by_stream_resnet;
  std::map<std::string, Counter> by_stream_yolo;
  std::map<std::string, Counter> by_lane;
};

struct Timings {
  double build_ms = 0.0;
  double warmup_ms = 0.0;
  double measured_ms = 0.0;
  double first_output_ms = 0.0;
};

struct Top1Result {
  int index = -1;
  float prob = 0.0f;
};

struct RtspServerContext {
  simaai::neat::Graph graph;
  simaai::neat::RtspServerHandle handle;
};

struct RtspHandleGroup {
  RtspHandleGroup() = default;
  RtspHandleGroup(const RtspHandleGroup&) = delete;
  RtspHandleGroup& operator=(const RtspHandleGroup&) = delete;
  ~RtspHandleGroup() {
    stop();
  }

  void add(simaai::neat::RtspServerHandle* handle) {
    if (handle)
      handles_.push_back(handle);
  }

  void stop() {
    for (auto* handle : handles_) {
      if (handle)
        handle->stop();
    }
    handles_.clear();
  }

private:
  std::vector<simaai::neat::RtspServerHandle*> handles_;
};

struct UdpPacketSender {
  UdpPacketSender() = default;
  UdpPacketSender(const UdpPacketSender&) = delete;
  UdpPacketSender& operator=(const UdpPacketSender&) = delete;
  UdpPacketSender(UdpPacketSender&& other) noexcept {
    move_from(std::move(other));
  }
  UdpPacketSender& operator=(UdpPacketSender&& other) noexcept {
    if (this != &other) {
      close_fd();
      move_from(std::move(other));
    }
    return *this;
  }
  ~UdpPacketSender() {
    close_fd();
  }

  bool init(const std::string& host, int port, std::string* err = nullptr) {
    close_fd();
    struct addrinfo hints {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_protocol = IPPROTO_UDP;
    struct addrinfo* res = nullptr;
    const std::string port_s = std::to_string(port);
    const int rc = getaddrinfo(host.c_str(), port_s.c_str(), &hints, &res);
    if (rc != 0 || !res) {
      if (err)
        *err = std::string("getaddrinfo failed: ") + gai_strerror(rc);
      return false;
    }
    for (struct addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
      const int candidate = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
      if (candidate < 0)
        continue;
      fd = candidate;
      addr_len = ai->ai_addrlen;
      std::memcpy(&addr, ai->ai_addr, ai->ai_addrlen);
      break;
    }
    freeaddrinfo(res);
    if (fd < 0) {
      if (err)
        *err = "socket() failed for UDP video sender";
      return false;
    }
    return true;
  }

  bool send_bytes(const void* data, std::size_t size, std::string* err = nullptr) const {
    if (fd < 0) {
      if (err)
        *err = "UDP video sender not initialized";
      return false;
    }
    if (!data || size == 0) {
      if (err)
        *err = "empty UDP video payload";
      return false;
    }
    const ssize_t n = ::sendto(fd, data, size, MSG_NOSIGNAL,
                               reinterpret_cast<const struct sockaddr*>(&addr), addr_len);
    if (n < 0 || static_cast<std::size_t>(n) != size) {
      if (err)
        *err = "sendto() failed for UDP video payload";
      return false;
    }
    return true;
  }

  int fd = -1;
  struct sockaddr_storage addr {};
  socklen_t addr_len = 0;

private:
  void close_fd() {
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
    addr_len = 0;
  }
  void move_from(UdpPacketSender&& other) {
    fd = other.fd;
    addr = other.addr;
    addr_len = other.addr_len;
    other.fd = -1;
    other.addr_len = 0;
  }
};

static std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

static std::optional<simaai::neat::BoxDecodeType> parse_boxdecode_type_token(std::string raw) {
  if (raw.empty())
    return std::nullopt;
  std::transform(raw.begin(), raw.end(), raw.begin(), [](unsigned char c) {
    if (c == '_' || c == '.')
      return '-';
    return static_cast<char>(std::tolower(c));
  });
  auto is = [&](const char* value) { return raw == value; };
  if (is("yolo") || is("yolo-generic"))
    return simaai::neat::BoxDecodeType::Yolo;
  if (is("yolov5") || is("yolo5") || is("yolo-v5"))
    return simaai::neat::BoxDecodeType::YoloV5;
  if (is("yolov5-seg") || is("yolo5-seg") || is("yolo-v5-seg"))
    return simaai::neat::BoxDecodeType::YoloV5Seg;
  if (is("yolov7") || is("yolo7") || is("yolo-v7"))
    return simaai::neat::BoxDecodeType::YoloV7;
  if (is("yolov7-seg") || is("yolo7-seg") || is("yolo-v7-seg"))
    return simaai::neat::BoxDecodeType::YoloV7Seg;
  if (is("yolov8") || is("yolo8") || is("yolo-v8"))
    return simaai::neat::BoxDecodeType::YoloV8;
  if (is("yolov8-seg") || is("yolo8-seg") || is("yolo-v8-seg"))
    return simaai::neat::BoxDecodeType::YoloV8Seg;
  if (is("yolov8-pose") || is("yolo8-pose") || is("yolo-v8-pose"))
    return simaai::neat::BoxDecodeType::YoloV8Pose;
  if (is("yolov9") || is("yolo9") || is("yolo-v9"))
    return simaai::neat::BoxDecodeType::YoloV9;
  if (is("yolov9-seg") || is("yolo9-seg") || is("yolo-v9-seg"))
    return simaai::neat::BoxDecodeType::YoloV9Seg;
  if (is("yolov10") || is("yolo10") || is("yolo-v10"))
    return simaai::neat::BoxDecodeType::YoloV10;
  if (is("yolov10-seg") || is("yolo10-seg") || is("yolo-v10-seg"))
    return simaai::neat::BoxDecodeType::YoloV10Seg;
  if (is("yolo26") || is("yolov26") || is("yolo-v26"))
    return simaai::neat::BoxDecodeType::YoloV26;
  if (is("yolo26-pose") || is("yolov26-pose") || is("yolo-v26-pose"))
    return simaai::neat::BoxDecodeType::YoloV26Pose;
  if (is("yolo26-seg") || is("yolov26-seg") || is("yolo-v26-seg"))
    return simaai::neat::BoxDecodeType::YoloV26Seg;
  if (is("yolov6") || is("yolo6") || is("yolo-v6"))
    return simaai::neat::BoxDecodeType::YoloV6;
  if (is("yolox") || is("yolo-x"))
    return simaai::neat::BoxDecodeType::YoloX;
  return std::nullopt;
}

static std::string to_upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return s;
}

static bool edge_log_enabled() {
  return std::getenv("SIMA_GRAPH_EDGE_LOG") != nullptr;
}

static int env_int(const char* key, int fallback) {
  const char* v = std::getenv(key);
  if (!v || !*v)
    return fallback;
  return std::atoi(v);
}

static std::string env_string(const char* key) {
  const char* v = std::getenv(key);
  return (v && *v) ? std::string(v) : std::string{};
}

static bool parse_bool_token(const std::string& raw, bool& out) {
  const std::string v = to_lower(raw);
  if (v == "1" || v == "true" || v == "yes" || v == "on") {
    out = true;
    return true;
  }
  if (v == "0" || v == "false" || v == "no" || v == "off") {
    out = false;
    return true;
  }
  return false;
}

static bool parse_bool_arg(int argc, char** argv, const std::string& key, bool& out) {
  std::string raw;
  if (!sima_test::get_arg(argc, argv, key, raw))
    return false;
  if (!parse_bool_token(raw, out)) {
    throw std::runtime_error(key + " expects boolean value 0/1/true/false");
  }
  return true;
}

static double elapsed_ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(b - a).count();
}

static double elapsed_seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(b - a).count();
}

static double percentile(std::vector<double> values, double p) {
  if (values.empty())
    return 0.0;
  std::sort(values.begin(), values.end());
  if (p <= 0.0)
    return values.front();
  if (p >= 100.0)
    return values.back();
  const double pos = (p / 100.0) * static_cast<double>(values.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
  const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
  if (lo == hi)
    return values[lo];
  const double frac = pos - static_cast<double>(lo);
  return values[lo] * (1.0 - frac) + values[hi] * frac;
}

static double mean(const std::vector<double>& values) {
  if (values.empty())
    return 0.0;
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  return sum / static_cast<double>(values.size());
}

static double stddev(const std::vector<double>& values) {
  if (values.size() < 2)
    return 0.0;
  const double m = mean(values);
  double accum = 0.0;
  for (double v : values) {
    const double d = v - m;
    accum += d * d;
  }
  return std::sqrt(accum / static_cast<double>(values.size()));
}

static int64_t sample_latency_sequence(const simaai::neat::Sample& sample) {
  if (sample.orig_input_seq >= 0)
    return sample.orig_input_seq;
  if (sample.input_seq >= 0)
    return sample.input_seq;
  if (sample.frame_id >= 0)
    return sample.frame_id;
  return -1;
}

class LatencyTracker {
public:
  explicit LatencyTracker(bool enabled, std::size_t max_entries = 200000)
      : enabled_(enabled), accepting_(enabled), max_entries_(max_entries) {}

  bool enabled() const {
    return enabled_;
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mu_);
    accepting_ = enabled_;
    entries_.clear();
    encoded_to_decoded_ms_.clear();
    decoded_to_boxes_ms_.clear();
    encoded_to_boxes_ms_.clear();
    boxes_to_metadata_send_ms_.clear();
    encoded_to_metadata_send_ms_.clear();
    encoded_to_video_send_ms_.clear();
    skipped_no_key_ = 0;
    evicted_entries_ = 0;
    out_of_order_durations_ = 0;
    encoded_marks_ = 0;
    decoded_marks_ = 0;
    boxes_marks_ = 0;
    metadata_sent_marks_ = 0;
    video_sent_marks_ = 0;
  }

  void freeze() {
    std::lock_guard<std::mutex> lock(mu_);
    accepting_ = false;
  }

  void mark_encoded(const simaai::neat::Sample& sample) {
    mark(sample, Stage::Encoded);
  }

  void mark_decoded(const simaai::neat::Sample& sample) {
    mark(sample, Stage::Decoded);
  }

  void mark_boxes(const simaai::neat::Sample& sample) {
    mark(sample, Stage::Boxes);
  }

  void mark_metadata_sent(const simaai::neat::Sample& sample) {
    mark(sample, Stage::MetadataSent);
  }

  void mark_video_sent(const simaai::neat::Sample& sample) {
    mark(sample, Stage::VideoSent);
  }

  json to_json() const {
    std::lock_guard<std::mutex> lock(mu_);
    json stages = json::object();
    stages["encoded_to_decoded"] = stage_json(encoded_to_decoded_ms_);
    stages["decoded_to_boxes"] = stage_json(decoded_to_boxes_ms_);
    stages["encoded_to_boxes"] = stage_json(encoded_to_boxes_ms_);
    stages["boxes_to_metadata_send"] = stage_json(boxes_to_metadata_send_ms_);
    stages["encoded_to_metadata_send"] = stage_json(encoded_to_metadata_send_ms_);
    stages["encoded_to_video_send"] = stage_json(encoded_to_video_send_ms_);
    return {
        {"enabled", enabled_},
        {"note",
         "App-level timestamps keyed by stream_id and orig_input_seq/input_seq. Stages are "
         "queue-inclusive wall-clock latencies observed at graph taps and send completion points."},
        {"counts",
         {{"tracked_entries", entries_.size()},
          {"max_entries", max_entries_},
          {"accepting_marks", accepting_},
          {"skipped_no_key", skipped_no_key_},
          {"evicted_entries", evicted_entries_},
          {"out_of_order_durations", out_of_order_durations_},
          {"encoded_marks", encoded_marks_},
          {"decoded_marks", decoded_marks_},
          {"boxes_marks", boxes_marks_},
          {"metadata_sent_marks", metadata_sent_marks_},
          {"video_sent_marks", video_sent_marks_}}},
        {"quality", quality_json_locked()},
        {"stages_ms", stages},
    };
  }

  void print_summary() const {
    const json j = to_json();
    const json& counts = j.at("counts");
    const json& quality = j.at("quality");
    std::cout << "[latency_profile] enabled=" << (enabled_ ? "true" : "false")
              << " tracked_entries=" << counts.value("tracked_entries", 0U)
              << " skipped_no_key=" << counts.value("skipped_no_key", 0)
              << " evicted_entries=" << counts.value("evicted_entries", 0)
              << " quality=" << quality.value("status", std::string("unknown"))
              << " correlated_samples=" << quality.value("best_correlated_stage_count", 0)
              << " min_correlated_samples="
              << quality.value("minimum_correlated_samples", 0) << "\n";
    for (const auto& warning : quality.value("warnings", json::array())) {
      if (warning.is_string())
        std::cout << "[latency_profile_warning] " << warning.get<std::string>() << "\n";
    }
    const json& stages = j.at("stages_ms");
    for (const char* name : {"encoded_to_decoded", "decoded_to_boxes", "encoded_to_boxes",
                             "boxes_to_metadata_send", "encoded_to_metadata_send",
                             "encoded_to_video_send"}) {
      const json& s = stages.at(name);
      const int64_t count = s.value("count", 0);
      if (count <= 0)
        continue;
      std::cout << "[latency_stage] stage=" << name << " count=" << count
                << " avg_ms=" << s.value("avg_ms", 0.0)
                << " p50_ms=" << s.value("p50_ms", 0.0)
                << " p95_ms=" << s.value("p95_ms", 0.0)
                << " p99_ms=" << s.value("p99_ms", 0.0)
                << " max_ms=" << s.value("max_ms", 0.0) << "\n";
    }
  }

private:
  static constexpr int64_t kMinCorrelatedLatencySamples = 5;

  enum class Stage { Encoded, Decoded, Boxes, MetadataSent, VideoSent };

  struct Key {
    std::string stream_id;
    int64_t seq = -1;

    bool operator==(const Key& other) const {
      return seq == other.seq && stream_id == other.stream_id;
    }
  };

  struct KeyHash {
    std::size_t operator()(const Key& k) const {
      const std::size_t h1 = std::hash<std::string>{}(k.stream_id);
      const std::size_t h2 = std::hash<int64_t>{}(k.seq);
      return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6U) + (h1 >> 2U));
    }
  };

  struct Entry {
    Clock::time_point encoded{};
    Clock::time_point decoded{};
    Clock::time_point boxes{};
    Clock::time_point metadata_sent{};
    Clock::time_point video_sent{};
    bool encoded_set = false;
    bool decoded_set = false;
    bool boxes_set = false;
    bool metadata_sent_set = false;
    bool video_sent_set = false;
    bool encoded_to_decoded_recorded = false;
    bool decoded_to_boxes_recorded = false;
    bool encoded_to_boxes_recorded = false;
    bool boxes_to_metadata_recorded = false;
    bool encoded_to_metadata_recorded = false;
    bool encoded_to_video_recorded = false;
  };

  static std::optional<Key> make_key(const simaai::neat::Sample& sample) {
    const int64_t seq = sample_latency_sequence(sample);
    if (sample.stream_id.empty() || seq < 0)
      return std::nullopt;
    return Key{sample.stream_id, seq};
  }

  static json stage_json(const std::vector<double>& values) {
    if (values.empty()) {
      return {{"count", 0},
              {"avg_ms", 0.0},
              {"min_ms", 0.0},
              {"p50_ms", 0.0},
              {"p95_ms", 0.0},
              {"p99_ms", 0.0},
              {"max_ms", 0.0}};
    }
    return {{"count", static_cast<int64_t>(values.size())},
            {"avg_ms", mean(values)},
            {"min_ms", *std::min_element(values.begin(), values.end())},
            {"p50_ms", percentile(values, 50.0)},
            {"p95_ms", percentile(values, 95.0)},
            {"p99_ms", percentile(values, 99.0)},
            {"max_ms", *std::max_element(values.begin(), values.end())}};
  }

  static double coverage_ratio(int64_t correlated, int64_t denominator) {
    return denominator > 0 ? static_cast<double>(correlated) / static_cast<double>(denominator)
                           : 0.0;
  }

  json quality_json_locked() const {
    const int64_t encoded_to_decoded =
        static_cast<int64_t>(encoded_to_decoded_ms_.size());
    const int64_t decoded_to_boxes = static_cast<int64_t>(decoded_to_boxes_ms_.size());
    const int64_t encoded_to_boxes = static_cast<int64_t>(encoded_to_boxes_ms_.size());
    const int64_t boxes_to_metadata =
        static_cast<int64_t>(boxes_to_metadata_send_ms_.size());
    const int64_t encoded_to_metadata =
        static_cast<int64_t>(encoded_to_metadata_send_ms_.size());
    const int64_t encoded_to_video =
        static_cast<int64_t>(encoded_to_video_send_ms_.size());
    const int64_t best_correlated =
        std::max({encoded_to_decoded, decoded_to_boxes, encoded_to_boxes, boxes_to_metadata,
                  encoded_to_metadata, encoded_to_video});
    const int64_t total_marks = encoded_marks_ + decoded_marks_ + boxes_marks_ +
                                metadata_sent_marks_ + video_sent_marks_;

    std::string status;
    if (!enabled_) {
      status = "disabled";
    } else if (total_marks == 0) {
      status = "no_marks";
    } else if (best_correlated >= kMinCorrelatedLatencySamples) {
      status = "correlated";
    } else {
      status = "insufficient_correlated_samples";
    }

    json warnings = json::array();
    if (enabled_ && total_marks > 0 && best_correlated == 0) {
      warnings.push_back("Latency marks were observed, but no complete stage correlation was "
                         "formed; check stream_id and orig_input_seq/input_seq preservation.");
    } else if (enabled_ && best_correlated > 0 &&
               best_correlated < kMinCorrelatedLatencySamples) {
      warnings.push_back("Correlated latency sample count is below the minimum for a trustworthy "
                         "profile; use a longer measured window.");
    }
    if (boxes_marks_ > 0 && encoded_to_boxes == 0) {
      warnings.push_back("YOLO boxes were observed without encoded-to-boxes correlation; this "
                         "usually means identity propagation is incomplete on that path.");
    }
    if (skipped_no_key_ > 0) {
      warnings.push_back("Some latency marks were skipped because stream_id or sequence id was "
                         "missing.");
    }
    if (evicted_entries_ > 0) {
      warnings.push_back("Some latency tracker entries were evicted before all stages arrived.");
    }
    if (out_of_order_durations_ > 0) {
      warnings.push_back("Some latency durations were dropped because stage timestamps arrived "
                         "out of order.");
    }

    const int64_t encoded_decoded_possible = std::min(encoded_marks_, decoded_marks_);
    const int64_t decoded_boxes_possible = std::min(decoded_marks_, boxes_marks_);
    const int64_t encoded_boxes_possible = std::min(encoded_marks_, boxes_marks_);
    const int64_t boxes_metadata_possible = std::min(boxes_marks_, metadata_sent_marks_);
    const int64_t encoded_metadata_possible = std::min(encoded_marks_, metadata_sent_marks_);
    const int64_t encoded_video_possible = std::min(encoded_marks_, video_sent_marks_);

    return {
        {"status", status},
        {"minimum_correlated_samples", kMinCorrelatedLatencySamples},
        {"minimum_correlated_samples_met", best_correlated >= kMinCorrelatedLatencySamples},
        {"best_correlated_stage_count", best_correlated},
        {"correlated_stage_counts",
         {{"encoded_to_decoded", encoded_to_decoded},
          {"decoded_to_boxes", decoded_to_boxes},
          {"encoded_to_boxes", encoded_to_boxes},
          {"boxes_to_metadata_send", boxes_to_metadata},
          {"encoded_to_metadata_send", encoded_to_metadata},
          {"encoded_to_video_send", encoded_to_video}}},
        {"coverage",
         {{"encoded_to_decoded",
           coverage_ratio(encoded_to_decoded, encoded_decoded_possible)},
          {"decoded_to_boxes", coverage_ratio(decoded_to_boxes, decoded_boxes_possible)},
          {"encoded_to_boxes", coverage_ratio(encoded_to_boxes, encoded_boxes_possible)},
          {"boxes_to_metadata_send",
           coverage_ratio(boxes_to_metadata, boxes_metadata_possible)},
          {"encoded_to_metadata_send",
           coverage_ratio(encoded_to_metadata, encoded_metadata_possible)},
          {"encoded_to_video_send", coverage_ratio(encoded_to_video, encoded_video_possible)}}},
        {"warnings", warnings},
    };
  }

  void mark(const simaai::neat::Sample& sample, Stage stage) {
    if (!enabled_)
      return;
    const auto key = make_key(sample);
    const auto now = Clock::now();
    std::lock_guard<std::mutex> lock(mu_);
    if (!accepting_)
      return;
    if (!key.has_value()) {
      ++skipped_no_key_;
      return;
    }

    Entry& entry = entries_[*key];
    switch (stage) {
    case Stage::Encoded:
      ++encoded_marks_;
      if (!entry.encoded_set) {
        entry.encoded = now;
        entry.encoded_set = true;
      }
      break;
    case Stage::Decoded:
      ++decoded_marks_;
      if (!entry.decoded_set) {
        entry.decoded = now;
        entry.decoded_set = true;
      }
      break;
    case Stage::Boxes:
      ++boxes_marks_;
      if (!entry.boxes_set) {
        entry.boxes = now;
        entry.boxes_set = true;
      }
      break;
    case Stage::MetadataSent:
      ++metadata_sent_marks_;
      if (!entry.metadata_sent_set) {
        entry.metadata_sent = now;
        entry.metadata_sent_set = true;
      }
      break;
    case Stage::VideoSent:
      ++video_sent_marks_;
      if (!entry.video_sent_set) {
        entry.video_sent = now;
        entry.video_sent_set = true;
      }
      break;
    }

    record_ready_durations(entry);
    evict_if_needed();
  }

  void append_duration(std::vector<double>& out, Clock::time_point start, Clock::time_point end) {
    const double ms = elapsed_ms(start, end);
    if (ms < 0.0) {
      ++out_of_order_durations_;
      return;
    }
    out.push_back(ms);
  }

  void record_ready_durations(Entry& entry) {
    if (entry.encoded_set && entry.decoded_set && !entry.encoded_to_decoded_recorded) {
      append_duration(encoded_to_decoded_ms_, entry.encoded, entry.decoded);
      entry.encoded_to_decoded_recorded = true;
    }
    if (entry.decoded_set && entry.boxes_set && !entry.decoded_to_boxes_recorded) {
      append_duration(decoded_to_boxes_ms_, entry.decoded, entry.boxes);
      entry.decoded_to_boxes_recorded = true;
    }
    if (entry.encoded_set && entry.boxes_set && !entry.encoded_to_boxes_recorded) {
      append_duration(encoded_to_boxes_ms_, entry.encoded, entry.boxes);
      entry.encoded_to_boxes_recorded = true;
    }
    if (entry.boxes_set && entry.metadata_sent_set && !entry.boxes_to_metadata_recorded) {
      append_duration(boxes_to_metadata_send_ms_, entry.boxes, entry.metadata_sent);
      entry.boxes_to_metadata_recorded = true;
    }
    if (entry.encoded_set && entry.metadata_sent_set && !entry.encoded_to_metadata_recorded) {
      append_duration(encoded_to_metadata_send_ms_, entry.encoded, entry.metadata_sent);
      entry.encoded_to_metadata_recorded = true;
    }
    if (entry.encoded_set && entry.video_sent_set && !entry.encoded_to_video_recorded) {
      append_duration(encoded_to_video_send_ms_, entry.encoded, entry.video_sent);
      entry.encoded_to_video_recorded = true;
    }
  }

  void evict_if_needed() {
    while (entries_.size() > max_entries_ && !entries_.empty()) {
      entries_.erase(entries_.begin());
      ++evicted_entries_;
    }
  }

  bool enabled_ = false;
  bool accepting_ = false;
  std::size_t max_entries_ = 0;
  mutable std::mutex mu_;
  std::unordered_map<Key, Entry, KeyHash> entries_;
  std::vector<double> encoded_to_decoded_ms_;
  std::vector<double> decoded_to_boxes_ms_;
  std::vector<double> encoded_to_boxes_ms_;
  std::vector<double> boxes_to_metadata_send_ms_;
  std::vector<double> encoded_to_metadata_send_ms_;
  std::vector<double> encoded_to_video_send_ms_;
  int64_t skipped_no_key_ = 0;
  int64_t evicted_entries_ = 0;
  int64_t out_of_order_durations_ = 0;
  int64_t encoded_marks_ = 0;
  int64_t decoded_marks_ = 0;
  int64_t boxes_marks_ = 0;
  int64_t metadata_sent_marks_ = 0;
  int64_t video_sent_marks_ = 0;
};

static void write_json_atomic(const std::string& path, const json& j) {
  if (path.empty())
    return;
  const fs::path out(path);
  if (out.has_parent_path()) {
    std::error_code mk_ec;
    fs::create_directories(out.parent_path(), mk_ec);
    if (mk_ec) {
      throw std::runtime_error("failed to create JSON output directory: " + mk_ec.message());
    }
  }
  const fs::path tmp = out.string() + ".tmp";
  {
    std::ofstream os(tmp);
    if (!os) {
      throw std::runtime_error("failed to open JSON tmp output: " + tmp.string());
    }
    os << j.dump(2) << "\n";
    os.flush();
    if (!os) {
      throw std::runtime_error("failed to write JSON tmp output: " + tmp.string());
    }
  }
  std::error_code ec;
  fs::rename(tmp, out, ec);
  if (ec) {
    std::error_code rm_ec;
    fs::remove(out, rm_ec);
    ec.clear();
    fs::rename(tmp, out, ec);
  }
  if (ec) {
    throw std::runtime_error("failed to rename JSON output into place: " + ec.message());
  }
}

static json load_processcvu_profile_jsonl(const std::string& path) {
  json out = {{"enabled", !path.empty()},
              {"path", path},
              {"found", false},
              {"entries", json::array()}};
  if (path.empty())
    return out;
  std::ifstream is(path);
  if (!is) {
    out["note"] = "profile JSONL file was not found; ensure SIMA_PROCESSCVU_PROFILE_JSONL is "
                  "supported by the loaded plugin";
    return out;
  }
  out["found"] = true;
  json latest_by_node = json::object();
  std::string line;
  int64_t parse_errors = 0;
  while (std::getline(is, line)) {
    if (line.empty())
      continue;
    try {
      json entry = json::parse(line);
      const std::string key = entry.value("stage", std::string{}) + "|" +
                              entry.value("node", std::string{});
      latest_by_node[key.empty() ? "<unknown>" : key] = entry;
      out["entries"].push_back(std::move(entry));
    } catch (const std::exception&) {
      ++parse_errors;
    }
  }
  out["parse_errors"] = parse_errors;
  out["latest_by_node"] = latest_by_node;
  out["entry_count"] = static_cast<int64_t>(out["entries"].size());
  return out;
}

static double json_number_at(const json& j, const std::string& object_key,
                             const std::string& value_key) {
  if (!j.contains(object_key) || !j.at(object_key).is_object())
    return 0.0;
  return j.at(object_key).value(value_key, 0.0);
}

static void print_processcvu_profile_summary(const json& profile) {
  if (!profile.value("found", false))
    return;
  const json latest = profile.value("latest_by_node", json::object());
  for (const auto& [_, entry] : latest.items()) {
    std::cout << "[processcvu_profile] stage=" << entry.value("stage", "")
              << " node=" << entry.value("node", "")
              << " graph=" << entry.value("graph_id", -1)
              << " samples=" << entry.value("samples", 0)
              << " avg_ms{total=" << json_number_at(entry, "avg_ms", "total")
              << ",dispatcher_call=" << json_number_at(entry, "avg_ms", "dispatcher_call")
              << ",dispatcher_queue=" << json_number_at(entry, "avg_ms", "dispatcher_queue")
              << ",dispatcher_exec=" << json_number_at(entry, "avg_ms", "dispatcher_exec")
              << ",dispatcher_result_wait="
              << json_number_at(entry, "avg_ms", "dispatcher_result_wait")
              << ",peek_inputs=" << json_number_at(entry, "avg_ms", "peek_inputs")
              << ",acquire_outbuf=" << json_number_at(entry, "avg_ms", "acquire_outbuf")
              << ",configure_job=" << json_number_at(entry, "avg_ms", "configure_job")
              << ",update_metadata=" << json_number_at(entry, "avg_ms", "update_metadata")
              << ",pre_handoff=" << json_number_at(entry, "avg_ms", "pre_handoff")
              << ",finish_buffer=" << json_number_at(entry, "avg_ms", "finish_buffer")
              << "} max_ms{total=" << json_number_at(entry, "max_ms", "total")
              << ",dispatcher_call=" << json_number_at(entry, "max_ms", "dispatcher_call")
              << ",peek_inputs=" << json_number_at(entry, "max_ms", "peek_inputs")
              << ",acquire_outbuf=" << json_number_at(entry, "max_ms", "acquire_outbuf")
              << "}\n";
  }
}

static const char* sample_kind_name(simaai::neat::SampleKind kind) {
  switch (kind) {
  case simaai::neat::SampleKind::Tensor:
    return "tensor";
  case simaai::neat::SampleKind::TensorSet:
    return "tensor_set";
  case simaai::neat::SampleKind::Bundle:
    return "bundle";
  default:
    return "other";
  }
}

static const char* device_type_name(const simaai::neat::Tensor& t) {
  switch (t.device.type) {
  case simaai::neat::DeviceType::CPU:
    return "cpu";
  default:
    return "other";
  }
}

static void log_edge(const char* tag, const simaai::neat::Sample& sample) {
  if (!edge_log_enabled())
    return;
  const char* dev = "-";
  if (sample.kind == simaai::neat::SampleKind::Tensor && sample.tensor.has_value()) {
    dev = device_type_name(sample.tensor.value());
  } else if (simaai::neat::sample_has_tensor_list(sample) && !sample.tensors.empty()) {
    dev = device_type_name(sample.tensors.front());
  }
  std::fprintf(stderr,
               "[EDGE] %s stream=%s frame=%lld input_seq=%lld orig_seq=%lld port=%s kind=%s dev=%s "
               "caps=%s\n",
               tag ? tag : "edge", sample.stream_id.empty() ? "<empty>" : sample.stream_id.c_str(),
               static_cast<long long>(sample.frame_id), static_cast<long long>(sample.input_seq),
               static_cast<long long>(sample.orig_input_seq),
               sample.port_name.empty() ? "<empty>" : sample.port_name.c_str(),
               sample_kind_name(sample.kind), dev,
               sample.caps_string.empty() ? "<empty>" : sample.caps_string.c_str());
}

static std::string branch_name(Branch b) {
  if (b == Branch::ResNet)
    return "resnet";
  if (b == Branch::Yolo)
    return "yolo";
  if (b == Branch::Raw)
    return "raw";
  return "video";
}

static bool expects_resnet(const StreamDef& def) {
  return def.branch == "both" || def.branch == "resnet";
}

static bool expects_yolo(const StreamDef& def) {
  return def.branch == "both" || def.branch == "yolo";
}

static bool expects_raw(const StreamDef& def) {
  return def.branch == "encoded" || def.branch == "decode";
}

static std::string branch_for_stream(int stream_index, const Args& args) {
  if (args.branch_mode == "fanout-both")
    return "both";
  if (args.branch_mode == "resnet-only")
    return "resnet";
  if (args.branch_mode == "yolo-only")
    return "yolo";
  if (args.branch_mode == "encoded-only")
    return "encoded";
  if (args.branch_mode == "decode-only")
    return "decode";
  if (args.branch_mode == "split")
    return (stream_index % 2 == 0) ? "resnet" : "yolo";
  throw std::runtime_error("invalid --branch-mode: " + args.branch_mode);
}

static simaai::neat::graph::nodes::StreamDropPolicy parse_drop_policy(const std::string& s) {
  if (s == "oldest")
    return simaai::neat::graph::nodes::StreamDropPolicy::DropOldest;
  if (s == "newest")
    return simaai::neat::graph::nodes::StreamDropPolicy::DropNewest;
  throw std::runtime_error("--scheduler-drop must be oldest or newest");
}

static simaai::neat::OutputMemory parse_output_memory(const std::string& s) {
  if (s == "owned")
    return simaai::neat::OutputMemory::Owned;
  if (s == "zero-copy")
    return simaai::neat::OutputMemory::ZeroCopy;
  throw std::runtime_error("--output-memory must be owned or zero-copy");
}

static bool is_yolo26_family(simaai::neat::BoxDecodeType type) {
  return type == simaai::neat::BoxDecodeType::YoloV26 ||
         type == simaai::neat::BoxDecodeType::YoloV26Pose ||
         type == simaai::neat::BoxDecodeType::YoloV26Seg;
}

static bool nodes_contain_kind(const std::vector<std::shared_ptr<simaai::neat::Node>>& nodes,
                               const std::string& kind) {
  return std::any_of(nodes.begin(), nodes.end(),
                     [&](const auto& n) { return n && n->kind() == kind; });
}

static std::string frame_id_string(const simaai::neat::Sample& sample) {
  if (sample.frame_id >= 0)
    return std::to_string(sample.frame_id);
  if (sample.orig_input_seq >= 0)
    return std::to_string(sample.orig_input_seq);
  if (sample.input_seq >= 0)
    return std::to_string(sample.input_seq);
  return "-1";
}

static bool has_sw_h264_encoder() {
  return simaai::neat::element_exists("x264enc") || simaai::neat::element_exists("openh264enc") ||
         simaai::neat::element_exists("avenc_h264");
}

static void require_element_or_skip(const char* factory) {
  if (!simaai::neat::element_exists(factory)) {
    skip_long_test_exception(
        std::string("missing NEAT/GStreamer element for macro mixed stress: ") + factory);
  }
}

static void require_assets_or_skip(bool require_assets, const std::string& msg) {
  if (require_assets) {
    throw std::runtime_error(msg);
  }
  skip_long_test_exception(msg);
}

static bool wait_for_rtsp_running(simaai::neat::RtspServerHandle& handle, int timeout_ms) {
  const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  while (Clock::now() < deadline) {
    if (handle.running())
      return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return handle.running();
}

static RtspServerContext start_rtsp_server(const std::string& image_path, int content_w,
                                           int content_h, int enc_w, int enc_h, int fps, int port,
                                           int rtp_port_base, int rtp_port_count) {
  RtspServerContext ctx;
  ctx.graph.add(
      simaai::neat::nodes::StillImageInput(image_path, content_w, content_h, enc_w, enc_h, fps));
  ctx.graph.add(simaai::neat::nodes::H264EncodeSW());
  ctx.graph.add(simaai::neat::nodes::H264Parse(/*config_interval=*/1));
  ctx.graph.add(simaai::neat::nodes::H264Packetize(/*pt=*/kPayloadType, /*config_interval=*/1));

  ctx.handle = ctx.graph.run_rtsp({
      .mount = "image",
      .port = port,
      .rtp_port_base = rtp_port_base,
      .rtp_port_count = rtp_port_count,
  });
  std::cout << "[rtsp] url=" << ctx.handle.url() << "\n";
  return ctx;
}

static RtspServerContext start_rtsp_server_with_retry(const std::string& image_path, int content_w,
                                                      int content_h, int enc_w, int enc_h, int fps,
                                                      int base_port, int rtp_port_base,
                                                      int rtp_port_count, int max_tries,
                                                      bool debug) {
  for (int i = 0; i < max_tries; ++i) {
    const int port = base_port + i;
    const int rtp_base = rtp_port_base + i;
    RtspServerContext ctx = start_rtsp_server(image_path, content_w, content_h, enc_w, enc_h, fps,
                                              port, rtp_base, rtp_port_count);
    if (wait_for_rtsp_running(ctx.handle, 2000)) {
      if (debug) {
        std::cerr << "[rtsp] server running url=" << ctx.handle.url() << "\n";
      }
      return ctx;
    }
    if (debug) {
      std::cerr << "[rtsp] server not running (port " << port << "), stopping\n";
    }
    ctx.handle.stop();
  }
  throw std::runtime_error("Failed to start RTSP server (port unavailable?)");
}

static std::vector<RtspServerContext>
start_rtsp_servers_with_retry(const std::string& image_path, int content_w, int content_h,
                              int enc_w, int enc_h, int fps, int base_port, int server_count,
                              int port_stride, int rtp_port_offset, int rtp_ports_per_server,
                              int rtp_port_stride, int max_tries, bool debug) {
  std::vector<RtspServerContext> servers;
  servers.reserve(static_cast<std::size_t>(server_count));
  for (int i = 0; i < server_count; ++i) {
    const int port = base_port + i * port_stride;
    const int rtp_port_base = base_port + rtp_port_offset + i * rtp_port_stride;
    servers.push_back(start_rtsp_server_with_retry(image_path, content_w, content_h, enc_w, enc_h,
                                                   fps, port, rtp_port_base, rtp_ports_per_server,
                                                   max_tries, debug));
  }
  return servers;
}

static bool probe_rtsp_encoded(const std::string& url, int fps, int w, int h, int tries,
                               int timeout_ms, bool print_pipeline, bool debug) {
  if (debug) {
    std::cerr << "[rtsp] probe start url=" << url << " tries=" << tries
              << " timeout_ms=" << timeout_ms << "\n";
  }
  simaai::neat::Graph p;
  p.add(simaai::neat::nodes::RTSPInput(url, /*latency_ms=*/200, /*tcp=*/true));
  p.add(simaai::neat::nodes::H264Depacketize(kPayloadType,
                                             /*config_interval=*/1, fps, w, h,
                                             /*enforce_caps=*/true));
  p.add(simaai::neat::nodes::Output());

  if (print_pipeline) {
    std::cout << "[rtsp-probe] pipeline:\n" << p.describe_backend() << "\n";
  }

  simaai::neat::RunOptions run_opt;
  run_opt.output_memory = simaai::neat::OutputMemory::Owned;
  simaai::neat::Run runner = p.build(run_opt);
  bool ok = false;
  for (int i = 0; i < tries; ++i) {
    auto out = runner.pull(timeout_ms);
    if (out.has_value()) {
      ok = true;
      break;
    }
  }
  runner.stop();
  if (debug) {
    std::cerr << "[rtsp] probe done ok=" << (ok ? "true" : "false") << "\n";
  }
  return ok;
}

static std::string resolve_image_path(const std::string& image_path,
                                      const std::string& goldfish_url) {
  if (!image_path.empty()) {
    if (!file_exists(image_path)) {
      throw std::runtime_error("Image not found: " + image_path);
    }
    return image_path;
  }

  const auto dst = sima_test::default_goldfish_path();
  if (!file_exists(dst.string())) {
    if (!sima_test::download_file(goldfish_url, dst)) {
      throw std::runtime_error("Failed to download goldfish image");
    }
  }
  return dst.string();
}

static bool looks_like_top1_tensor(const simaai::neat::Tensor& t) {
  if (t.dtype != simaai::neat::TensorDType::Float32)
    return false;
  if (!t.is_dense())
    return false;
  if (t.shape.size() != 1 || t.shape[0] != 2)
    return false;
  if (t.device.type != simaai::neat::DeviceType::CPU)
    return false;
  return true;
}

static Top1Result top1_from_tensor_payload(const simaai::neat::Tensor& t) {
  auto map = t.map(simaai::neat::MapMode::Read);
  if (!map.data || map.size_bytes < sizeof(float) * 2) {
    throw std::runtime_error("Top1 tensor payload is empty");
  }
  const float* data = static_cast<const float*>(map.data);
  const int idx = static_cast<int>(std::llround(static_cast<double>(data[0])));
  const float prob = data[1];
  return Top1Result{idx, prob};
}

static simaai::neat::Tensor make_top1_tensor(const Top1Result& res) {
  auto storage = simaai::neat::make_cpu_owned_storage(sizeof(float) * 2);
  simaai::neat::Tensor out;
  out.storage = std::move(storage);
  out.dtype = simaai::neat::TensorDType::Float32;
  out.shape = {2};
  out.strides_bytes = {static_cast<int64_t>(sizeof(float))};
  out.device = {simaai::neat::DeviceType::CPU, 0};
  out.read_only = false;
  auto map = out.map(simaai::neat::MapMode::Write);
  if (!map.data || map.size_bytes < sizeof(float) * 2) {
    throw std::runtime_error("Failed to map top1 tensor payload");
  }
  float* data = static_cast<float*>(map.data);
  data[0] = static_cast<float>(res.index);
  data[1] = res.prob;
  if (map.unmap)
    map.unmap();
  return out;
}

static Top1Result argmax_from_tensor(const simaai::neat::Tensor& t,
                                     const char* stream_id = nullptr) {
  if (t.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error("Expected Float32 tensor output");
  }
  if (!t.is_dense()) {
    throw std::runtime_error("Expected dense tensor output");
  }
  const std::size_t dense_bytes = t.dense_bytes_tight();
  auto compute_argmax = [dense_bytes](const simaai::neat::Mapping& map) -> Top1Result {
    if (!map.data || map.size_bytes < sizeof(float)) {
      throw std::runtime_error("Empty tensor output");
    }
    std::size_t bytes = map.size_bytes;
    if (dense_bytes > 0 && dense_bytes < bytes) {
      bytes = dense_bytes;
    }
    const std::size_t count = bytes / sizeof(float);
    if (count == 0) {
      throw std::runtime_error("Empty tensor output");
    }
    const float* data = static_cast<const float*>(map.data);
    int best = 0;
    float best_val = data[0];
    for (std::size_t i = 1; i < count; ++i) {
      if (data[i] > best_val) {
        best_val = data[i];
        best = static_cast<int>(i);
      }
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
      sum += std::exp(static_cast<double>(data[i] - best_val));
    }
    const float prob = (sum > 0.0) ? static_cast<float>(1.0 / sum) : 0.0f;
    return Top1Result{best, prob};
  };

  const bool force_cpu = env_flag("SIMA_ARGMAX_FORCE_CPU");
  const bool compare_cpu = env_flag("SIMA_ARGMAX_COMPARE_CPU");
  const bool debug = env_flag("SIMA_ARGMAX_DEBUG");

  Top1Result result{};
  bool have_cpu = false;
  Top1Result cpu_result{};

  if (!force_cpu) {
    auto map = t.view(simaai::neat::MapMode::Read);
    if (map.data && map.size_bytes >= sizeof(float)) {
      result = compute_argmax(map);
    } else {
      simaai::neat::Tensor cpu = t.to_cpu_if_needed();
      result = compute_argmax(cpu.map(simaai::neat::MapMode::Read));
      have_cpu = true;
      cpu_result = result;
    }
  }

  auto compute_cpu = [&]() -> Top1Result {
    simaai::neat::Tensor cpu = t.to_cpu_if_needed();
    auto map = cpu.map(simaai::neat::MapMode::Read);
    return compute_argmax(map);
  };

  if (force_cpu) {
    result = compute_cpu();
    have_cpu = true;
    cpu_result = result;
  }

  if (compare_cpu && !have_cpu) {
    cpu_result = compute_cpu();
    have_cpu = true;
  }

  if (compare_cpu && have_cpu) {
    if (result.index != cpu_result.index) {
      std::fprintf(stderr, "[ARGMAX] mismatch stream=%s cvu=%d prob=%.6f cpu=%d prob=%.6f\n",
                   stream_id ? stream_id : "<unknown>", result.index, result.prob, cpu_result.index,
                   cpu_result.prob);
    } else if (debug) {
      std::fprintf(stderr, "[ARGMAX] match stream=%s idx=%d prob=%.6f\n",
                   stream_id ? stream_id : "<unknown>", result.index, result.prob);
    }
  } else if (debug) {
    std::fprintf(stderr, "[ARGMAX] stream=%s idx=%d prob=%.6f\n",
                 stream_id ? stream_id : "<unknown>", result.index, result.prob);
  }

  simaai::neat::pipeline_internal::drop_tensor_holder(t);
  return result;
}

static Top1Result argmax_from_sample(const simaai::neat::Sample& sample) {
  if (simaai::neat::sample_has_tensor_list(sample) && sample.tensors.size() == 1U) {
    const simaai::neat::Tensor& t = sample.tensors.front();
    if (looks_like_top1_tensor(t)) {
      return top1_from_tensor_payload(t);
    }
    return argmax_from_tensor(t, sample.stream_id.c_str());
  }
  if (sample.kind == simaai::neat::SampleKind::Tensor && sample.tensor.has_value()) {
    const simaai::neat::Tensor& t = sample.tensor.value();
    if (looks_like_top1_tensor(t)) {
      return top1_from_tensor_payload(t);
    }
    return argmax_from_tensor(t, sample.stream_id.c_str());
  }
  throw std::runtime_error("Expected tensor sample output");
}

static Args parse_args(int argc, char** argv) {
  Args args;
  std::string raw;
  if (sima_test::get_arg(argc, argv, "--root", raw))
    args.root = fs::path(raw);
  sima_test::get_arg(argc, argv, "--image", args.image);
  sima_test::get_arg(argc, argv, "--goldfish-url", args.goldfish_url);
  sima_test::get_arg(argc, argv, "--resnet-model", args.resnet_model);
  sima_test::get_arg(argc, argv, "--yolo-model", args.yolo_model);
  sima_test::get_arg(argc, argv, "--json-out", args.json_out);

  sima_test::parse_int_arg(argc, argv, "--streams", args.streams);
  sima_test::parse_int_arg(argc, argv, "--rtsp-servers", args.rtsp_servers);
  sima_test::parse_int_arg(argc, argv, "--fps", args.fps);
  if (!sima_test::parse_int_arg(argc, argv, "--stream-width", args.stream_width))
    sima_test::parse_int_arg(argc, argv, "--width", args.stream_width);
  if (!sima_test::parse_int_arg(argc, argv, "--stream-height", args.stream_height))
    sima_test::parse_int_arg(argc, argv, "--height", args.stream_height);
  sima_test::parse_int_arg(argc, argv, "--iters", args.iters);
  sima_test::parse_int_arg(argc, argv, "--resnet-lanes", args.resnet_lanes);
  sima_test::parse_int_arg(argc, argv, "--yolo-lanes", args.yolo_lanes);
  sima_test::parse_int_arg(argc, argv, "--port", args.port);
  sima_test::parse_int_arg(argc, argv, "--rtsp-tries", args.rtsp_tries);
  sima_test::parse_int_arg(argc, argv, "--rtsp-timeout-ms", args.rtsp_timeout_ms);
  sima_test::parse_int_arg(argc, argv, "--max-runtime-ms", args.max_runtime_ms);
  sima_test::parse_int_arg(argc, argv, "--duration-ms", args.duration_ms);
  sima_test::parse_int_arg(argc, argv, "--min-measured-ms", args.min_measured_ms);
  sima_test::parse_int_arg(argc, argv, "--stall-timeout-ms", args.stall_timeout_ms);
  sima_test::parse_int_arg(argc, argv, "--warmup-per-stream", args.warmup_per_stream);
  sima_test::parse_int_arg(argc, argv, "--warmup-timeout-ms", args.warmup_timeout_ms);
  sima_test::parse_int_arg(argc, argv, "--scheduler-queue", args.scheduler_queue);
  sima_test::parse_int_arg(argc, argv, "--scheduler-max-batch", args.scheduler_max_batch);
  args.edge_queue_cli = sima_test::parse_int_arg(argc, argv, "--edge-queue", args.edge_queue);
  args.push_timeout_cli =
      sima_test::parse_int_arg(argc, argv, "--push-timeout-ms", args.push_timeout_ms);
  sima_test::parse_int_arg(argc, argv, "--pull-timeout-ms", args.pull_timeout_ms);
  sima_test::parse_int_arg(argc, argv, "--decoder-buffers", args.decoder_buffers);
  sima_test::parse_int_arg(argc, argv, "--yolo-top-k", args.yolo_top_k);
  sima_test::parse_int_arg(argc, argv, "--metadata-port-base", args.metadata_port_base);
  sima_test::parse_int_arg(argc, argv, "--video-port-base", args.video_port_base);
  args.h264_bitrate_cli =
      sima_test::parse_int_arg(argc, argv, "--h264-bitrate-kbps", args.h264_bitrate_kbps);
  sima_test::parse_int_arg(argc, argv, "--processcvu-profile-every",
                           args.processcvu_profile_every);
  sima_test::parse_int_arg(argc, argv, "--processcvu-async-in-flight",
                           args.processcvu_async_in_flight);
  sima_test::parse_int_arg(argc, argv, "--processcvu-pool-buffers",
                           args.processcvu_pool_buffers);

  sima_test::parse_float_arg(argc, argv, "--yolo-score-threshold", args.yolo_score_threshold);
  sima_test::parse_float_arg(argc, argv, "--yolo-nms-iou", args.yolo_nms_iou);

  std::string yolo_decode_type_raw;
  if (sima_test::get_arg(argc, argv, "--yolo-decode-type", yolo_decode_type_raw)) {
    auto parsed = parse_boxdecode_type_token(yolo_decode_type_raw);
    if (!parsed.has_value()) {
      throw std::runtime_error("unsupported --yolo-decode-type: " + yolo_decode_type_raw);
    }
    args.yolo_decode_type = *parsed;
  }

  if (sima_test::get_arg(argc, argv, "--branch-mode", args.branch_mode))
    args.branch_mode = to_lower(args.branch_mode);
  if (sima_test::get_arg(argc, argv, "--scheduler-drop", args.scheduler_drop))
    args.scheduler_drop = to_lower(args.scheduler_drop);
  if (sima_test::get_arg(argc, argv, "--output-memory", args.output_memory)) {
    args.output_memory = to_lower(args.output_memory);
    args.output_memory_cli = true;
  }
  if (sima_test::get_arg(argc, argv, "--source", args.source))
    args.source = to_lower(args.source);
  if (sima_test::get_arg(argc, argv, "--processcvu-target", args.processcvu_target))
    args.processcvu_target = to_upper(args.processcvu_target);
  sima_test::get_arg(argc, argv, "--processcvu-profile-jsonl",
                     args.processcvu_profile_jsonl);
  sima_test::get_arg(argc, argv, "--video-host", args.video_host);

  parse_bool_arg(argc, argv, "--metadata-udp", args.metadata_udp);
  parse_bool_arg(argc, argv, "--insight-video", args.insight_video);
  parse_bool_arg(argc, argv, "--latency-profile", args.latency_profile);
  parse_bool_arg(argc, argv, "--yolo-discard-boxes", args.yolo_discard_boxes);
  parse_bool_arg(argc, argv, "--validate-resnet-goldfish", args.validate_resnet_goldfish);
  parse_bool_arg(argc, argv, "--require-assets", args.require_assets);
  args.serial_pipeline_build_cli =
      parse_bool_arg(argc, argv, "--serial-pipeline-build", args.serial_pipeline_build);
  args.gst_element_timings_cli =
      parse_bool_arg(argc, argv, "--gst-element-timings", args.gst_element_timings);
  args.gst_flow_debug_cli = parse_bool_arg(argc, argv, "--gst-flow-debug", args.gst_flow_debug);
  args.gst_boundary_probes_cli =
      parse_bool_arg(argc, argv, "--gst-boundary-probes", args.gst_boundary_probes);
  args.processcvu_profile_repro_cli =
      parse_bool_arg(argc, argv, "--processcvu-profile-repro", args.processcvu_profile_repro);
  args.processcvu_timeline_cli =
      parse_bool_arg(argc, argv, "--processcvu-timeline", args.processcvu_timeline);
  args.print_pipeline = sima_test::has_flag(argc, argv, "--print-pipeline");
  args.rtsp_debug = sima_test::has_flag(argc, argv, "--rtsp-debug");
  args.lossless_rtsp = sima_test::has_flag(argc, argv, "--lossless-rtsp");
  args.allow_heavy = sima_test::has_flag(argc, argv, "--allow-heavy") ||
                     env_flag("SIMA_ALLOW_HEAVY_MACRO_STRESS", false);

  if (!args.edge_queue_cli && std::getenv("SIMA_GRAPH_EDGE_QUEUE"))
    args.edge_queue = env_int("SIMA_GRAPH_EDGE_QUEUE", args.edge_queue);
  if (!args.push_timeout_cli && std::getenv("SIMA_GRAPH_PUSH_TIMEOUT_MS"))
    args.push_timeout_ms = env_int("SIMA_GRAPH_PUSH_TIMEOUT_MS", args.push_timeout_ms);
  if (!args.output_memory_cli && std::getenv("SIMA_GRAPH_COPY_OUTPUT"))
    args.output_memory = env_flag("SIMA_GRAPH_COPY_OUTPUT", true) ? "owned" : "zero-copy";
  if (!args.h264_bitrate_cli && std::getenv("SIMA_H264ENC_BITRATE_KBPS"))
    args.h264_bitrate_kbps = env_int("SIMA_H264ENC_BITRATE_KBPS", args.h264_bitrate_kbps);
  if (!args.gst_element_timings_cli && std::getenv("SIMA_GST_ELEMENT_TIMINGS"))
    args.gst_element_timings = env_flag("SIMA_GST_ELEMENT_TIMINGS", args.gst_element_timings);
  if (!args.gst_flow_debug_cli && std::getenv("SIMA_GST_FLOW_DEBUG"))
    args.gst_flow_debug = env_flag("SIMA_GST_FLOW_DEBUG", args.gst_flow_debug);
  if (!args.gst_boundary_probes_cli && std::getenv("SIMA_GST_BOUNDARY_PROBES"))
    args.gst_boundary_probes = env_flag("SIMA_GST_BOUNDARY_PROBES", args.gst_boundary_probes);

  if (args.source != "rtsp") {
    throw std::runtime_error(
        "--source " + args.source +
        " is not implemented in macro_mixed_multistream_multimodel_tput_test v1");
  }
  if (args.streams < 1 || args.streams > 32)
    throw std::runtime_error("--streams must be in range 1..32");
  if (args.iters <= 0)
    throw std::runtime_error("--iters must be > 0");
  if (args.fps <= 0)
    throw std::runtime_error("--fps must be > 0");
  if (args.stream_width <= 0 || args.stream_height <= 0)
    throw std::runtime_error("--stream-width/--stream-height must be > 0");
  if ((args.stream_width % 2) != 0 || (args.stream_height % 2) != 0)
    throw std::runtime_error("--stream-width/--stream-height must be even for NV12/H264");
  if (args.duration_ms < 0)
    throw std::runtime_error("--duration-ms must be >= 0");
  if (args.min_measured_ms < 0)
    throw std::runtime_error("--min-measured-ms must be >= 0");
  if (args.rtsp_servers <= 0)
    args.rtsp_servers = std::min(args.streams, 8);
  if (args.rtsp_servers > args.streams)
    args.rtsp_servers = args.streams;
  if (args.resnet_lanes < 0 || args.resnet_lanes > 8)
    throw std::runtime_error("--resnet-lanes must be in range 0..8");
  if (args.yolo_lanes < 0 || args.yolo_lanes > 8)
    throw std::runtime_error("--yolo-lanes must be in range 0..8");
  if (args.branch_mode != "fanout-both" && args.branch_mode != "split" &&
      args.branch_mode != "resnet-only" && args.branch_mode != "yolo-only" &&
      args.branch_mode != "encoded-only" && args.branch_mode != "decode-only") {
    throw std::runtime_error("--branch-mode must be fanout-both, split, resnet-only, yolo-only, "
                             "encoded-only, or decode-only");
  }
  if (args.branch_mode == "resnet-only") {
    args.yolo_lanes = 0;
  } else if (args.branch_mode == "yolo-only") {
    args.resnet_lanes = 0;
  } else if (args.branch_mode == "encoded-only" || args.branch_mode == "decode-only") {
    args.resnet_lanes = 0;
    args.yolo_lanes = 0;
  }
  if (args.branch_mode == "fanout-both" && (args.resnet_lanes <= 0 || args.yolo_lanes <= 0))
    throw std::runtime_error("fanout-both requires --resnet-lanes > 0 and --yolo-lanes > 0");
  if (args.branch_mode == "split") {
    if (args.streams < 2)
      throw std::runtime_error("split mode requires --streams >= 2");
    if (args.resnet_lanes <= 0 || args.yolo_lanes <= 0)
      throw std::runtime_error("split requires --resnet-lanes > 0 and --yolo-lanes > 0");
  }
  if (args.branch_mode == "resnet-only" && args.resnet_lanes <= 0)
    throw std::runtime_error("resnet-only requires --resnet-lanes > 0");
  if (args.branch_mode == "yolo-only" && args.yolo_lanes <= 0)
    throw std::runtime_error("yolo-only requires --yolo-lanes > 0");
  if (args.insight_video &&
      (args.branch_mode == "encoded-only" || args.branch_mode == "decode-only")) {
    throw std::runtime_error("--insight-video is only supported with model branch modes");
  }
  if (args.resnet_lanes + args.yolo_lanes > 4 && !args.allow_heavy) {
    throw std::runtime_error(
        "total lanes > 4 requires --allow-heavy or SIMA_ALLOW_HEAVY_MACRO_STRESS=1");
  }
  if (args.scheduler_queue <= 0)
    throw std::runtime_error("--scheduler-queue must be > 0");
  if (args.scheduler_max_batch <= 0)
    throw std::runtime_error("--scheduler-max-batch must be > 0");
  if (args.edge_queue < 0)
    throw std::runtime_error("--edge-queue must be >= 0");
  if (args.push_timeout_ms < 0 || args.pull_timeout_ms < 0)
    throw std::runtime_error("push/pull timeouts must be >= 0");
  if (args.decoder_buffers == 0 || args.decoder_buffers < -1)
    throw std::runtime_error("--decoder-buffers must be -1 or > 0");
  if (args.h264_bitrate_kbps < 0)
    throw std::runtime_error("--h264-bitrate-kbps must be >= 0");
  if (args.processcvu_profile_every <= 0)
    throw std::runtime_error("--processcvu-profile-every must be > 0");
  if (args.processcvu_async_in_flight != -1 && args.processcvu_async_in_flight <= 0)
    throw std::runtime_error("--processcvu-async-in-flight must be -1 or > 0");
  if (args.processcvu_pool_buffers != -1 && args.processcvu_pool_buffers <= 0)
    throw std::runtime_error("--processcvu-pool-buffers must be -1 or > 0");
  if (args.video_port_base <= 0)
    throw std::runtime_error("--video-port-base must be > 0");
  if (args.video_host.empty())
    throw std::runtime_error("--video-host must not be empty");
  if (args.stall_timeout_ms <= 0)
    throw std::runtime_error("--stall-timeout-ms must be > 0");
  if (args.max_runtime_ms <= 0)
    throw std::runtime_error("--max-runtime-ms must be > 0");
  if (args.processcvu_target != "AUTO" && args.processcvu_target != "EV74" &&
      args.processcvu_target != "A65") {
    throw std::runtime_error("--processcvu-target must be AUTO, EV74, or A65");
  }
  parse_drop_policy(args.scheduler_drop);
  parse_output_memory(args.output_memory);
  return args;
}

static void record_counter(Counter& counter, Clock::time_point now) {
  if (!counter.initialized) {
    counter.initialized = true;
    counter.first = now;
  } else {
    counter.inter_output_gap_ms.push_back(elapsed_ms(counter.last, now));
  }
  counter.last = now;
  ++counter.count;
}

static int64_t counter_count(const std::map<std::string, Counter>& counters,
                             const std::string& key) {
  const auto it = counters.find(key);
  return it == counters.end() ? 0 : it->second.count;
}

static bool warmup_done(const std::map<std::string, int64_t>& resnet_counts,
                        const std::map<std::string, int64_t>& yolo_counts,
                        const std::map<std::string, int64_t>& raw_counts,
                        const std::vector<StreamDef>& streams, int target) {
  if (target <= 0)
    return true;
  for (const auto& stream : streams) {
    if (expects_resnet(stream)) {
      const auto it = resnet_counts.find(stream.stream_id);
      if (it == resnet_counts.end() || it->second < target)
        return false;
    }
    if (expects_yolo(stream)) {
      const auto it = yolo_counts.find(stream.stream_id);
      if (it == yolo_counts.end() || it->second < target)
        return false;
    }
    if (expects_raw(stream)) {
      const auto it = raw_counts.find(stream.stream_id);
      if (it == raw_counts.end() || it->second < target)
        return false;
    }
  }
  return true;
}

static bool measured_done(const Metrics& metrics, const std::vector<StreamDef>& streams,
                          int target) {
  for (const auto& stream : streams) {
    if (expects_resnet(stream) &&
        counter_count(metrics.by_stream_resnet, stream.stream_id) < target) {
      return false;
    }
    if (expects_yolo(stream) && counter_count(metrics.by_stream_yolo, stream.stream_id) < target) {
      return false;
    }
    if (expects_raw(stream) && counter_count(metrics.by_stream_total, stream.stream_id) < target) {
      return false;
    }
  }
  return true;
}

static std::vector<std::string> active_branches_json(const Args& args) {
  std::vector<std::string> out;
  if (args.branch_mode != "yolo-only" && args.resnet_lanes > 0)
    out.push_back("resnet");
  if (args.branch_mode != "resnet-only" && args.yolo_lanes > 0)
    out.push_back("yolo");
  if (args.branch_mode == "encoded-only")
    out.push_back("encoded");
  if (args.branch_mode == "decode-only")
    out.push_back("decode");
  if (args.insight_video)
    out.push_back("insight-video");
  return out;
}

static int64_t expected_outputs_for(const std::vector<StreamDef>& streams, int iters) {
  int64_t expected = 0;
  for (const auto& stream : streams) {
    if (expects_resnet(stream))
      expected += iters;
    if (expects_yolo(stream))
      expected += iters;
    if (expects_raw(stream))
      expected += iters;
  }
  return expected;
}

static std::vector<double> stream_counts_as_double(const std::vector<StreamDef>& streams,
                                                   const std::map<std::string, Counter>& counts) {
  std::vector<double> values;
  values.reserve(streams.size());
  for (const auto& stream : streams) {
    values.push_back(static_cast<double>(counter_count(counts, stream.stream_id)));
  }
  return values;
}

static std::vector<double> branch_counts_as_double(const std::vector<StreamDef>& streams,
                                                   const std::map<std::string, Counter>& counts,
                                                   Branch branch) {
  std::vector<double> values;
  values.reserve(streams.size());
  for (const auto& stream : streams) {
    if ((branch == Branch::ResNet && !expects_resnet(stream)) ||
        (branch == Branch::Yolo && !expects_yolo(stream))) {
      continue;
    }
    values.push_back(static_cast<double>(counter_count(counts, stream.stream_id)));
  }
  return values;
}

static std::vector<std::string> expected_lane_labels(const std::vector<StreamDef>& streams) {
  std::vector<std::string> labels;
  std::unordered_set<std::string> seen;
  for (const auto& stream : streams) {
    if (expects_resnet(stream) && stream.resnet_lane >= 0) {
      const std::string label = "resnet_" + std::to_string(stream.resnet_lane);
      if (seen.insert(label).second)
        labels.push_back(label);
    }
    if (expects_yolo(stream) && stream.yolo_lane >= 0) {
      const std::string label = "yolo_" + std::to_string(stream.yolo_lane);
      if (seen.insert(label).second)
        labels.push_back(label);
    }
    if (expects_raw(stream)) {
      const std::string label = stream.branch + "_" + stream.stream_id;
      if (seen.insert(label).second)
        labels.push_back(label);
    }
  }
  std::sort(labels.begin(), labels.end());
  return labels;
}

static std::vector<double> all_gaps(const Metrics& metrics) {
  std::vector<double> gaps;
  for (const auto& [_, counter] : metrics.by_stream_total) {
    gaps.insert(gaps.end(), counter.inter_output_gap_ms.begin(), counter.inter_output_gap_ms.end());
  }
  return gaps;
}

static json graph_run_stats_json(const GraphRunStats& stats) {
  json nodes = json::array();
  int64_t total = 0;
  for (const auto& snap : stats.snapshot()) {
    json streams = json::object();
    for (const auto& [stream, count] : snap.counts) {
      streams[stream] = count;
    }
    nodes.push_back({{"node_id", snap.node_id}, {"total", snap.total}, {"streams", streams}});
    total += snap.total;
  }
  return {{"enabled", true}, {"total_samples", total}, {"nodes", nodes}};
}

static json make_report(const Args& args, const Metrics& metrics, const Timings& timings,
                        const std::vector<StreamDef>& streams, int64_t expected_outputs,
                        int graph_nodes, int graph_edges, const std::string& status,
                        const std::string& failure_reason) {
  const double measured_s = timings.measured_ms / 1000.0;
  const double aggregate_fps =
      measured_s > 0.0 ? static_cast<double>(metrics.total_outputs) / measured_s : 0.0;
  const double resnet_fps =
      measured_s > 0.0 ? static_cast<double>(metrics.resnet_outputs) / measured_s : 0.0;
  const double yolo_fps =
      measured_s > 0.0 ? static_cast<double>(metrics.yolo_outputs) / measured_s : 0.0;
  const double raw_fps =
      measured_s > 0.0 ? static_cast<double>(metrics.raw_outputs) / measured_s : 0.0;
  const int64_t target_counted_outputs = std::min<int64_t>(metrics.total_outputs, expected_outputs);
  const int64_t excess_outputs =
      metrics.total_outputs > expected_outputs ? metrics.total_outputs - expected_outputs : 0;
  const double target_normalized_fps =
      measured_s > 0.0 ? static_cast<double>(target_counted_outputs) / measured_s : 0.0;
  json throughput_warnings = json::array();
  if (excess_outputs > 0) {
    throughput_warnings.push_back(
        "Aggregate output FPS includes excess async/multi-output samples beyond the target count; "
        "use headline_fps/target_normalized_output_fps for benchmark comparisons.");
  }

  json j;
  j["schema_version"] = 1;
  j["workload_type"] = "macro_mixed_multistream_multimodel";
  j["test_name"] = "macro_mixed_multistream_multimodel_tput_test";
  j["status"] = status;
  j["failure_reason"] = failure_reason;

  j["config"] = {
      {"source", args.source},
      {"streams", args.streams},
      {"rtsp_servers", args.rtsp_servers},
      {"fps", args.fps},
      {"stream_width", args.stream_width},
      {"stream_height", args.stream_height},
      {"iters_per_stream_per_branch", args.iters},
      {"duration_ms", args.duration_ms},
      {"min_measured_ms", args.min_measured_ms},
      {"branch_mode", args.branch_mode},
      {"active_branches", active_branches_json(args)},
      {"resnet_lanes", args.resnet_lanes},
      {"yolo_lanes", args.yolo_lanes},
      {"scheduler_queue", args.scheduler_queue},
      {"scheduler_drop", args.scheduler_drop},
      {"scheduler_max_batch", args.scheduler_max_batch},
      {"edge_queue", args.edge_queue},
      {"push_timeout_ms", args.push_timeout_ms},
      {"pull_timeout_ms", args.pull_timeout_ms},
      {"decoder_buffers", args.decoder_buffers},
      {"output_memory", args.output_memory},
      {"metadata_udp", args.metadata_udp},
      {"insight_video", args.insight_video},
      {"latency_profile", args.latency_profile},
      {"yolo_discard_boxes", args.yolo_discard_boxes},
      {"video_host", args.video_host},
      {"video_port_base", args.video_port_base},
      {"h264_bitrate_kbps", args.h264_bitrate_kbps},
      {"yolo_decode_type", simaai::neat::box_decode_type_token(args.yolo_decode_type)},
      {"gst_element_timings", args.gst_element_timings},
      {"gst_flow_debug", args.gst_flow_debug},
      {"gst_boundary_probes", args.gst_boundary_probes},
      {"processcvu_profile_repro", args.processcvu_profile_repro},
      {"processcvu_profile_jsonl", args.processcvu_profile_jsonl},
      {"processcvu_profile_every", args.processcvu_profile_every},
      {"processcvu_async_in_flight", args.processcvu_async_in_flight},
      {"processcvu_pool_buffers", args.processcvu_pool_buffers},
      {"processcvu_timeline", args.processcvu_timeline},
      {"serial_pipeline_build", args.serial_pipeline_build_cli
                                    ? args.serial_pipeline_build
                                    : env_flag("SIMA_GRAPH_SERIAL_PIPELINE_BUILD", false)},
      {"processcvu_target", args.processcvu_target},
      {"models", {{"resnet", args.resnet_model}, {"yolo", args.yolo_model}}},
      {"graph", {{"nodes", graph_nodes}, {"edges", graph_edges}}},
  };
  j["env"] = {
      {"SIMA_GRAPH_EDGE_QUEUE", env_string("SIMA_GRAPH_EDGE_QUEUE")},
      {"SIMA_GRAPH_COPY_OUTPUT", env_string("SIMA_GRAPH_COPY_OUTPUT")},
      {"SIMA_GRAPH_PUSH_TIMEOUT_MS", env_string("SIMA_GRAPH_PUSH_TIMEOUT_MS")},
      {"SIMA_GRAPH_SERIAL_PIPELINE_BUILD", env_string("SIMA_GRAPH_SERIAL_PIPELINE_BUILD")},
      {"SIMA_H264ENC_BITRATE_KBPS", env_string("SIMA_H264ENC_BITRATE_KBPS")},
      {"SIMA_PROCESSCVU_RUN_TARGET", env_string("SIMA_PROCESSCVU_RUN_TARGET")},
      {"SIMA_PROCESSCVU_ASYNC_IN_FLIGHT", env_string("SIMA_PROCESSCVU_ASYNC_IN_FLIGHT")},
      {"SIMA_PROCESSCVU_PREPARED_POOL_BUFFERS",
       env_string("SIMA_PROCESSCVU_PREPARED_POOL_BUFFERS")},
      {"SIMA_PROCESSCVU_PROFILE", env_string("SIMA_PROCESSCVU_PROFILE")},
      {"SIMA_PROCESSCVU_PROFILE_VERBOSE", env_string("SIMA_PROCESSCVU_PROFILE_VERBOSE")},
      {"SIMA_PROCESSCVU_PROFILE_EVERY", env_string("SIMA_PROCESSCVU_PROFILE_EVERY")},
      {"SIMA_PROCESSCVU_PROFILE_JSONL", env_string("SIMA_PROCESSCVU_PROFILE_JSONL")},
      {"SIMA_PROCESSCVU_TIMELINE", env_string("SIMA_PROCESSCVU_TIMELINE")},
      {"SIMA_PROCESSCVU_TIMELINE_LIMIT", env_string("SIMA_PROCESSCVU_TIMELINE_LIMIT")},
      {"SIMA_PROCESSMLA_SAFE_ASYNC", env_string("SIMA_PROCESSMLA_SAFE_ASYNC")},
      {"SIMA_PROCESSMLA_SAFE_ASYNC_DEPTH", env_string("SIMA_PROCESSMLA_SAFE_ASYNC_DEPTH")},
      {"SIMA_GST_ELEMENT_TIMINGS", env_string("SIMA_GST_ELEMENT_TIMINGS")},
      {"SIMA_GST_FLOW_DEBUG", env_string("SIMA_GST_FLOW_DEBUG")},
      {"SIMA_GST_BOUNDARY_PROBES", env_string("SIMA_GST_BOUNDARY_PROBES")},
      {"SIMA_DISPATCHER_PROFILE", env_string("SIMA_DISPATCHER_PROFILE")},
      {"SIMA_RUNTIME_PROFILE", env_string("SIMA_RUNTIME_PROFILE")},
      {"SIMA_RUNTIME_PROFILE_JSONL", env_string("SIMA_RUNTIME_PROFILE_JSONL")},
  };
  j["timing"] = {{"build_ms", timings.build_ms},
                 {"warmup_ms", timings.warmup_ms},
                 {"measured_ms", timings.measured_ms},
                 {"first_output_ms", timings.first_output_ms}};
  j["throughput"] = {
      {"offered_input_fps", static_cast<double>(args.streams * args.fps)},
      {"headline_fps", target_normalized_fps},
      {"headline_fps_semantics", "target_counted_outputs_per_measured_second"},
      {"aggregate_output_fps", aggregate_fps},
      {"aggregate_output_fps_semantics",
       "all_observed_outputs_per_measured_second_including_excess"},
      {"target_normalized_output_fps", target_normalized_fps},
      {"resnet_output_fps", resnet_fps},
      {"yolo_output_fps", yolo_fps},
      {"raw_output_fps", raw_fps},
      {"effective_output_per_input_ratio",
       args.streams > 0 && args.fps > 0
           ? aggregate_fps / static_cast<double>(args.streams * args.fps)
           : 0.0},
      {"warnings", throughput_warnings},
  };
  j["counts"] = {{"expected_outputs", expected_outputs},
                 {"target_counted_outputs", target_counted_outputs},
                 {"excess_outputs", excess_outputs},
                 {"total_outputs", metrics.total_outputs},
                 {"resnet_outputs", metrics.resnet_outputs},
                 {"yolo_outputs", metrics.yolo_outputs},
                 {"raw_outputs", metrics.raw_outputs},
                 {"video_packets", metrics.video_packets},
                 {"video_bytes", metrics.video_bytes},
                 {"video_send_failed", metrics.video_send_failed},
                 {"timeouts", metrics.timeouts},
                 {"stalls", metrics.stalls},
                 {"max_runtime_hit", metrics.max_runtime_hit},
                 {"metadata_sent", metrics.metadata_sent},
                 {"metadata_send_failed", metrics.metadata_send_failed},
                 {"metadata_received", metrics.metadata_received}};

  std::vector<double> total_counts = stream_counts_as_double(streams, metrics.by_stream_total);
  std::vector<double> resnet_counts =
      branch_counts_as_double(streams, metrics.by_stream_resnet, Branch::ResNet);
  std::vector<double> yolo_counts =
      branch_counts_as_double(streams, metrics.by_stream_yolo, Branch::Yolo);
  json starved = json::array();
  for (const auto& stream : streams) {
    if (expects_resnet(stream) && counter_count(metrics.by_stream_resnet, stream.stream_id) == 0)
      starved.push_back(stream.stream_id + ":resnet");
    if (expects_yolo(stream) && counter_count(metrics.by_stream_yolo, stream.stream_id) == 0)
      starved.push_back(stream.stream_id + ":yolo");
    if (expects_raw(stream) && counter_count(metrics.by_stream_total, stream.stream_id) == 0)
      starved.push_back(stream.stream_id + ":" + stream.branch);
  }
  json starved_lanes = json::array();
  for (const auto& label : expected_lane_labels(streams)) {
    if (counter_count(metrics.by_lane, label) == 0)
      starved_lanes.push_back(label);
  }
  j["fairness"] = {
      {"stream_total_min",
       total_counts.empty() ? 0.0 : *std::min_element(total_counts.begin(), total_counts.end())},
      {"stream_total_max",
       total_counts.empty() ? 0.0 : *std::max_element(total_counts.begin(), total_counts.end())},
      {"stream_total_avg", mean(total_counts)},
      {"stream_total_stddev", stddev(total_counts)},
      {"resnet_min",
       resnet_counts.empty() ? 0.0 : *std::min_element(resnet_counts.begin(), resnet_counts.end())},
      {"resnet_max",
       resnet_counts.empty() ? 0.0 : *std::max_element(resnet_counts.begin(), resnet_counts.end())},
      {"yolo_min",
       yolo_counts.empty() ? 0.0 : *std::min_element(yolo_counts.begin(), yolo_counts.end())},
      {"yolo_max",
       yolo_counts.empty() ? 0.0 : *std::max_element(yolo_counts.begin(), yolo_counts.end())},
      {"starved_streams", starved},
      {"starved_lanes", starved_lanes},
  };

  const std::vector<double> gaps = all_gaps(metrics);
  j["latency_proxy_ms"] = {
      {"note", "RTSP v1 records output inter-arrival gaps and time-to-first-output, not true "
               "source-to-output latency."},
      {"global_gap_p50", percentile(gaps, 50.0)},
      {"global_gap_p95", percentile(gaps, 95.0)},
      {"global_gap_p99", percentile(gaps, 99.0)},
  };

  json per_stream = json::object();
  for (const auto& stream : streams) {
    const auto total = counter_count(metrics.by_stream_total, stream.stream_id);
    const auto resnet = counter_count(metrics.by_stream_resnet, stream.stream_id);
    const auto yolo = counter_count(metrics.by_stream_yolo, stream.stream_id);
    std::vector<double> gaps_for_stream;
    if (const auto it = metrics.by_stream_total.find(stream.stream_id);
        it != metrics.by_stream_total.end())
      gaps_for_stream = it->second.inter_output_gap_ms;
    per_stream[stream.stream_id] = {
        {"total", total},
        {"resnet", resnet},
        {"yolo", yolo},
        {"raw", expects_raw(stream) ? total : 0},
        {"fps_total", measured_s > 0.0 ? static_cast<double>(total) / measured_s : 0.0},
        {"gap_p95_ms", percentile(gaps_for_stream, 95.0)}};
  }
  j["per_stream"] = per_stream;

  json per_lane = json::object();
  for (const auto& [lane, counter] : metrics.by_lane) {
    per_lane[lane] = {
        {"outputs", counter.count},
        {"fps", measured_s > 0.0 ? static_cast<double>(counter.count) / measured_s : 0.0}};
  }
  j["per_lane"] = per_lane;
  return j;
}

static void record_sample_metrics(Metrics& metrics, Timings& timings, const OutputDef& def,
                                  const simaai::neat::Sample& sample, Clock::time_point measured_t0,
                                  Clock::time_point now) {
  std::string sid = sample.stream_id;
  if (sid.empty()) {
    sid = "<empty>";
    metrics.saw_empty_stream_id = true;
  }
  if (!metrics.first_output_seen) {
    metrics.first_output_seen = true;
    metrics.first_output = now;
    timings.first_output_ms = elapsed_ms(measured_t0, now);
  }

  ++metrics.total_outputs;
  record_counter(metrics.by_stream_total[sid], now);
  record_counter(metrics.by_lane[def.label], now);

  if (def.branch == Branch::ResNet) {
    ++metrics.resnet_outputs;
    record_counter(metrics.by_stream_resnet[sid], now);
  } else if (def.branch == Branch::Yolo) {
    ++metrics.yolo_outputs;
    record_counter(metrics.by_stream_yolo[sid], now);
  } else if (def.branch == Branch::Raw) {
    ++metrics.raw_outputs;
  }
}

static void
send_yolo_metadata_if_enabled(const Args& args, Metrics& metrics, const OutputDef& def,
                              const simaai::neat::Sample& sample,
                              const std::vector<simaai::neat::MetadataSender>& senders,
                              const std::shared_ptr<LatencyTracker>& latency_tracker) {
  if (!args.metadata_udp || def.branch != Branch::Yolo)
    return;
  if (def.lane < 0 || static_cast<std::size_t>(def.lane) >= senders.size()) {
    ++metrics.metadata_send_failed;
    return;
  }

  json data;
  data["stream_id"] = sample.stream_id;
  data["branch"] = "yolo";
  data["lane"] = def.lane;
  data["frame_id"] = frame_id_string(sample);
  data["input_seq"] = sample.input_seq;
  data["orig_input_seq"] = sample.orig_input_seq;
  data["payload_kind"] = sample_kind_name(sample.kind);
  data["logical_payload_count"] = sample.size();

  const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
  std::string err;
  if (senders[static_cast<std::size_t>(def.lane)].send_metadata("object-detection", data.dump(),
                                                                static_cast<int64_t>(now_ms),
                                                                frame_id_string(sample), &err)) {
    ++metrics.metadata_sent;
    if (latency_tracker)
      latency_tracker->mark_metadata_sent(sample);
  } else {
    ++metrics.metadata_send_failed;
    if (env_flag("SIMA_METADATA_DEBUG", false)) {
      std::cerr << "[metadata] send failed lane=" << def.lane << " err=" << err << "\n";
    }
  }
}

static void send_video_packet_if_enabled(Metrics& metrics, const OutputDef& def,
                                         const simaai::neat::Sample& sample,
                                         const std::vector<UdpPacketSender>& senders,
                                         const std::shared_ptr<LatencyTracker>& latency_tracker) {
  if (def.branch != Branch::Video)
    return;
  if (def.lane < 0 || static_cast<std::size_t>(def.lane) >= senders.size()) {
    ++metrics.video_send_failed;
    return;
  }
  if (!simaai::neat::sample_has_tensor_list(sample) || sample.tensors.empty()) {
    ++metrics.video_send_failed;
    return;
  }
  const simaai::neat::Tensor& t = sample.tensors.front();
  auto map = t.map(simaai::neat::MapMode::Read);
  std::size_t bytes = map.size_bytes;
  const std::size_t tight = t.dense_bytes_tight();
  if (tight > 0 && tight < bytes)
    bytes = tight;
  std::string err;
  if (map.data && bytes > 0 &&
      senders[static_cast<std::size_t>(def.lane)].send_bytes(map.data, bytes, &err)) {
    ++metrics.video_packets;
    metrics.video_bytes += static_cast<int64_t>(bytes);
    if (latency_tracker)
      latency_tracker->mark_video_sent(sample);
  } else {
    ++metrics.video_send_failed;
    if (env_flag("SIMA_VIDEO_DEBUG", false)) {
      std::cerr << "[video] send failed stream=" << sample.stream_id << " lane=" << def.lane
                << " bytes=" << bytes << " err=" << err << "\n";
    }
  }
  if (map.unmap)
    map.unmap();
}

static void print_summary(const Args& args, const Metrics& metrics, const Timings& timings,
                          const std::vector<StreamDef>& streams) {
  const double measured_s = timings.measured_ms / 1000.0;
  const int64_t expected_outputs = expected_outputs_for(streams, args.iters);
  const int64_t target_counted_outputs = std::min<int64_t>(metrics.total_outputs, expected_outputs);
  const int64_t excess_outputs =
      metrics.total_outputs > expected_outputs ? metrics.total_outputs - expected_outputs : 0;
  const double aggregate_fps =
      measured_s > 0.0 ? static_cast<double>(metrics.total_outputs) / measured_s : 0.0;
  const double target_normalized_fps =
      measured_s > 0.0 ? static_cast<double>(target_counted_outputs) / measured_s : 0.0;
  std::cout << "[result] outputs_total=" << metrics.total_outputs
            << " resnet_outputs=" << metrics.resnet_outputs
            << " yolo_outputs=" << metrics.yolo_outputs << " raw_outputs=" << metrics.raw_outputs
            << " target_counted_outputs=" << target_counted_outputs
            << " excess_outputs=" << excess_outputs
            << " target_normalized_fps=" << target_normalized_fps
            << " aggregate_output_fps=" << aggregate_fps << "\n";
  if (excess_outputs > 0) {
    std::cout << "[tput_note] target_normalized_fps excludes " << excess_outputs
              << " excess async/multi-output samples; aggregate_output_fps includes them.\n";
  }
  std::cout << "[branch_tput] branch=resnet outputs=" << metrics.resnet_outputs << " fps="
            << (measured_s > 0.0 ? static_cast<double>(metrics.resnet_outputs) / measured_s : 0.0)
            << "\n";
  std::cout << "[branch_tput] branch=yolo outputs=" << metrics.yolo_outputs << " fps="
            << (measured_s > 0.0 ? static_cast<double>(metrics.yolo_outputs) / measured_s : 0.0)
            << "\n";
  std::cout << "[branch_tput] branch=raw outputs=" << metrics.raw_outputs << " fps="
            << (measured_s > 0.0 ? static_cast<double>(metrics.raw_outputs) / measured_s : 0.0)
            << "\n";
  for (const auto& [lane, counter] : metrics.by_lane) {
    std::cout << "[lane_tput] lane=" << lane << " outputs=" << counter.count << " fps="
              << (measured_s > 0.0 ? static_cast<double>(counter.count) / measured_s : 0.0) << "\n";
  }

  int64_t min_total = std::numeric_limits<int64_t>::max();
  int64_t max_total = 0;
  double sum_fps = 0.0;
  int starved = 0;
  for (const auto& stream : streams) {
    const int64_t total = counter_count(metrics.by_stream_total, stream.stream_id);
    const int64_t resnet = counter_count(metrics.by_stream_resnet, stream.stream_id);
    const int64_t yolo = counter_count(metrics.by_stream_yolo, stream.stream_id);
    min_total = std::min(min_total, total);
    max_total = std::max(max_total, total);
    const double fps = measured_s > 0.0 ? static_cast<double>(total) / measured_s : 0.0;
    sum_fps += fps;
    if ((expects_resnet(stream) && resnet == 0) || (expects_yolo(stream) && yolo == 0))
      ++starved;
    std::cout << "[stream_tput] stream=" << stream.stream_id << " total=" << total
              << " resnet=" << resnet << " yolo=" << yolo
              << " raw=" << (expects_raw(stream) ? total : 0) << " fps=" << fps << "\n";
  }
  if (streams.empty())
    min_total = 0;
  std::cout << "[stream_tput_summary] streams=" << streams.size() << " min_total=" << min_total
            << " max_total=" << max_total << " avg_fps="
            << (streams.empty() ? 0.0 : sum_fps / static_cast<double>(streams.size()))
            << " starved=" << starved << "\n";
  std::cout << "[metadata] sent=" << metrics.metadata_sent
            << " send_failed=" << metrics.metadata_send_failed
            << " received=" << metrics.metadata_received << "\n";
  std::cout << "[video] packets=" << metrics.video_packets << " bytes=" << metrics.video_bytes
            << " send_failed=" << metrics.video_send_failed << "\n";
  (void)args;
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);

  Args args;
  sima_test::get_arg(argc, argv, "--json-out", args.json_out);
  Metrics metrics;
  Timings timings;
  std::vector<StreamDef> stream_defs;
  int64_t expected_outputs = 0;
  int graph_nodes = 0;
  int graph_edges = 0;
  json profile_json = json::object();
  std::shared_ptr<LatencyTracker> latency_tracker;

  auto emit_json = [&](const std::string& status, const std::string& reason) {
    if (args.json_out.empty())
      return;
    if (latency_tracker)
      latency_tracker->freeze();
    if (!args.processcvu_profile_jsonl.empty()) {
      profile_json["processcvu_internal"] =
          load_processcvu_profile_jsonl(args.processcvu_profile_jsonl);
    }
    json report = make_report(args, metrics, timings, stream_defs, expected_outputs, graph_nodes,
                              graph_edges, status, reason);
    report["profile"] = profile_json;
    report["latency_profile"] =
        latency_tracker ? latency_tracker->to_json() : json{{"enabled", false}};
    write_json_atomic(args.json_out, report);
    std::cout << "[json] wrote " << args.json_out << "\n";
  };

  try {
    args = parse_args(argc, argv);

    const bool processcvu_profile_requested =
        args.processcvu_profile_repro || !args.processcvu_profile_jsonl.empty();
    if (processcvu_profile_requested) {
      args.processcvu_profile_repro = true;
      if (args.processcvu_profile_jsonl.empty()) {
        args.processcvu_profile_jsonl =
            args.json_out.empty()
                ? (std::string("/tmp/macro_mixed_processcvu_profile_") +
                   std::to_string(static_cast<long long>(::getpid())) + ".jsonl")
                : (args.json_out + ".processcvu.jsonl");
      }
      const fs::path profile_path(args.processcvu_profile_jsonl);
      if (profile_path.has_parent_path()) {
        std::error_code mk_ec;
        fs::create_directories(profile_path.parent_path(), mk_ec);
        if (mk_ec) {
          throw std::runtime_error("failed to create ProcessCVU profile directory: " +
                                   mk_ec.message());
        }
      }
      std::error_code rm_ec;
      fs::remove(profile_path, rm_ec);
      setenv("SIMA_PROCESSCVU_PROFILE", "1", 1);
      setenv("SIMA_RUNTIME_PROFILE", "1", 1);
      setenv("SIMA_PROCESSCVU_PROFILE_EVERY",
             std::to_string(args.processcvu_profile_every).c_str(), 1);
      setenv("SIMA_PROCESSCVU_PROFILE_JSONL", args.processcvu_profile_jsonl.c_str(), 1);
    } else if (args.processcvu_profile_jsonl.empty()) {
      args.processcvu_profile_jsonl = env_string("SIMA_PROCESSCVU_PROFILE_JSONL");
      if (args.processcvu_profile_jsonl.empty())
        args.processcvu_profile_jsonl = env_string("SIMA_RUNTIME_PROFILE_JSONL");
    }

    if (args.processcvu_timeline) {
      setenv("SIMA_PROCESSCVU_TIMELINE", "1", 1);
      if (!std::getenv("SIMA_PROCESSCVU_TIMELINE_LIMIT")) {
        setenv("SIMA_PROCESSCVU_TIMELINE_LIMIT", "256", 1);
      }
    }
    if (args.processcvu_async_in_flight > 0) {
      setenv("SIMA_PROCESSCVU_ASYNC_IN_FLIGHT",
             std::to_string(args.processcvu_async_in_flight).c_str(), 1);
    }
    if (args.processcvu_pool_buffers > 0) {
      setenv("SIMA_PROCESSCVU_PREPARED_POOL_BUFFERS",
             std::to_string(args.processcvu_pool_buffers).c_str(), 1);
    }

    latency_tracker = std::make_shared<LatencyTracker>(args.latency_profile);

    require_element_or_skip("appsrc");
    require_element_or_skip("appsink");
    require_element_or_skip("identity");
    require_element_or_skip("rtspsrc");
    require_element_or_skip("rtph264depay");
    require_element_or_skip("h264parse");
    require_element_or_skip("rtph264pay");
    if (args.insight_video)
      require_element_or_skip("udpsink");
    require_element_or_skip("neatdecoder");
    require_element_or_skip("neatprocesscvu");
    require_element_or_skip("neatprocessmla");
    if (args.yolo_lanes > 0)
      require_element_or_skip("neatobjectdecode");
    if (!has_sw_h264_encoder()) {
      skip_long_test_exception("missing software H264 encoder for RTSP source (need x264enc, "
                               "openh264enc, or avenc_h264)");
    }

    if (args.rtsp_debug) {
      setenv("SIMA_GST_FLOW_DEBUG", "1", 1);
      setenv("SIMA_APPSINK_CAPS_DEBUG", "1", 1);
    }
    if (args.gst_element_timings_cli ||
        (args.gst_element_timings && !std::getenv("SIMA_GST_ELEMENT_TIMINGS"))) {
      setenv("SIMA_GST_ELEMENT_TIMINGS", args.gst_element_timings ? "1" : "0", 1);
    }
    if (args.gst_flow_debug_cli || (args.gst_flow_debug && !std::getenv("SIMA_GST_FLOW_DEBUG"))) {
      setenv("SIMA_GST_FLOW_DEBUG", args.gst_flow_debug ? "1" : "0", 1);
    }
    if (args.gst_boundary_probes_cli ||
        (args.gst_boundary_probes && !std::getenv("SIMA_GST_BOUNDARY_PROBES"))) {
      setenv("SIMA_GST_BOUNDARY_PROBES", args.gst_boundary_probes ? "1" : "0", 1);
      if (args.gst_boundary_probes && !std::getenv("SIMA_GST_RUN_INSERT_BOUNDARIES")) {
        setenv("SIMA_GST_RUN_INSERT_BOUNDARIES", "1", 1);
      }
    }
    if (args.lossless_rtsp) {
      setenv("SIMA_H264ENC_LOSSLESS", "1", 1);
    }
    if (args.h264_bitrate_cli || !std::getenv("SIMA_H264ENC_BITRATE_KBPS")) {
      setenv("SIMA_H264ENC_BITRATE_KBPS", std::to_string(args.h264_bitrate_kbps).c_str(), 1);
    }
    if (args.serial_pipeline_build_cli) {
      setenv("SIMA_GRAPH_SERIAL_PIPELINE_BUILD", args.serial_pipeline_build ? "1" : "0", 1);
    }
    if (!std::getenv("SIMA_DISPATCHER_AUTO_RECOVER")) {
      setenv("SIMA_DISPATCHER_AUTO_RECOVER", "1", 1);
    }

    std::string image_path;
    try {
      image_path = resolve_image_path(args.image, args.goldfish_url);
    } catch (const std::exception& e) {
      require_assets_or_skip(args.require_assets, e.what());
    }

    if (args.resnet_lanes > 0 && args.resnet_model.empty())
      args.resnet_model = sima_test::resolve_resnet50_tar(args.root);
    if (args.resnet_lanes > 0 && args.resnet_model.empty()) {
      require_assets_or_skip(args.require_assets,
                             "ResNet50 model pack not found; set --resnet-model, SIMA_MODEL_TAR, "
                             "SIMA_RESNET50_TAR, or run modelzoo get resnet_50");
    }
    if (args.yolo_lanes > 0 && args.yolo_model.empty() &&
        is_yolo26_family(args.yolo_decode_type)) {
      args.yolo_model = sima_test::env_existing_model_tar_path("SIMA_YOLO26_TAR");
      if (args.yolo_model.empty())
        args.yolo_model = sima_test::env_existing_model_tar_path("SIMA_YOLO26N_TAR");
      if (args.yolo_model.empty())
        args.yolo_model = sima_test::env_existing_model_tar_path("SIMA_YOLO_TAR");
    }
    if (args.yolo_lanes > 0 && args.yolo_model.empty() &&
        !is_yolo26_family(args.yolo_decode_type)) {
      args.yolo_model = sima_test::resolve_yolov8s_tar(args.root);
    }
    if (args.yolo_lanes > 0 && args.yolo_model.empty()) {
      if (is_yolo26_family(args.yolo_decode_type)) {
        require_assets_or_skip(args.require_assets,
                               "YOLO26 model pack not found; set --yolo-model, "
                               "SIMA_YOLO26_TAR, SIMA_YOLO26N_TAR, or SIMA_YOLO_TAR");
      }
      require_assets_or_skip(args.require_assets,
                             "YOLOv8s model pack not found; set --yolo-model, SIMA_YOLO_TAR, "
                             "SIMA_MODEL_TAR, or run modelzoo get yolo_v8s");
    }

    for (int i = 0; i < args.streams; ++i) {
      StreamDef stream;
      stream.index = i;
      stream.stream_id = "stream" + std::to_string(i);
      stream.branch = branch_for_stream(i, args);
      if (expects_resnet(stream))
        stream.resnet_lane = i % args.resnet_lanes;
      if (expects_yolo(stream))
        stream.yolo_lane = i % args.yolo_lanes;
      stream_defs.push_back(stream);
    }
    expected_outputs = expected_outputs_for(stream_defs, args.iters);

    const int port_range = std::max(1, env_int("SIMA_RTSP_PORT_RANGE", 128));
    const int rtsp_port_stride = 10;
    const int rtp_port_offset =
        std::max(0, env_int("SIMA_RTSP_RTP_PORT_OFFSET", kRtspRtpPortOffset));
    const int rtp_ports_per_server =
        std::max(2, env_int("SIMA_RTSP_RTP_PORT_COUNT", rtsp_port_stride - 1));
    const int rtp_port_stride = std::max(1, env_int("SIMA_RTSP_RTP_PORT_STRIDE", rtsp_port_stride));
    const int chosen_port = rtsp_find_free_port_range_with_rtp(
        args.port, args.rtsp_servers, rtsp_port_stride, port_range, rtp_port_offset,
        rtp_ports_per_server, rtp_port_stride);
    if (chosen_port < 0) {
      throw std::runtime_error("Failed to find free RTSP port range");
    }
    if (chosen_port != args.port) {
      std::cerr << "[rtsp] base port " << args.port << " busy for " << args.rtsp_servers
                << " servers; using " << chosen_port << "\n";
      args.port = chosen_port;
    }

    auto rtsp_servers = start_rtsp_servers_with_retry(
        image_path, /*content_w=*/args.stream_width, /*content_h=*/args.stream_height,
        /*enc_w=*/args.stream_width, /*enc_h=*/args.stream_height, args.fps, args.port,
        args.rtsp_servers, rtsp_port_stride,
        rtp_port_offset, rtp_ports_per_server, rtp_port_stride, /*max_tries=*/5, args.rtsp_debug);
    RtspHandleGroup rtsp_guard;
    for (auto& rtsp : rtsp_servers)
      rtsp_guard.add(&rtsp.handle);

    for (auto& rtsp : rtsp_servers) {
      if (!probe_rtsp_encoded(rtsp.handle.url(), args.fps, args.stream_width, args.stream_height,
                              args.rtsp_tries, args.rtsp_timeout_ms, args.print_pipeline,
                              args.rtsp_debug)) {
        skip_long_test_exception("RTSP probe failed before macro mixed stress graph");
      }
    }

    std::cout << "[setup] streams=" << args.streams << " rtsp_servers=" << args.rtsp_servers
              << " fps=" << args.fps << " branch_mode=" << args.branch_mode
              << " stream=" << args.stream_width << "x" << args.stream_height
              << " resnet_lanes=" << args.resnet_lanes << " yolo_lanes=" << args.yolo_lanes
              << " insight_video=" << (args.insight_video ? "on" : "off") << "\n";
    std::cout << "[model] resnet=" << (args.resnet_model.empty() ? "<inactive>" : args.resnet_model)
              << " yolo=" << (args.yolo_model.empty() ? "<inactive>" : args.yolo_model) << "\n";

    std::optional<simaai::neat::Model> resnet_model;
    std::optional<simaai::neat::Model> yolo_model;

    if (args.resnet_lanes > 0) {
      simaai::neat::Model::Options resnet_cfg;
      resnet_cfg.preprocess.kind = simaai::neat::InputKind::Image;
      resnet_cfg.preprocess.enable = simaai::neat::AutoFlag::On;
      resnet_cfg.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
      resnet_cfg.preprocess.preset = simaai::neat::NormalizePreset::ImageNet;
      resnet_cfg.processcvu.pre_run_target = args.processcvu_target;
      resnet_cfg.processcvu.post_run_target = args.processcvu_target;
      resnet_model.emplace(args.resnet_model, resnet_cfg);
    }
    if (args.yolo_lanes > 0) {
      simaai::neat::Model::Options yolo_cfg;
      yolo_cfg.preprocess.kind = simaai::neat::InputKind::Image;
      yolo_cfg.preprocess.enable = simaai::neat::AutoFlag::On;
      yolo_cfg.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
      yolo_cfg.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
      yolo_cfg.decode_type = args.yolo_decode_type;
      yolo_cfg.score_threshold = args.yolo_score_threshold;
      yolo_cfg.nms_iou_threshold = args.yolo_nms_iou;
      yolo_cfg.top_k = args.yolo_top_k;
      yolo_cfg.processcvu.pre_run_target = args.processcvu_target;
      yolo_cfg.processcvu.post_run_target = args.processcvu_target;
      yolo_model.emplace(args.yolo_model, yolo_cfg);
    }

    simaai::neat::Model::RouteOptions base_route_opt;
    base_route_opt.include_input = false;
    base_route_opt.include_output = false;
    base_route_opt.expose_all_outputs = false;
    base_route_opt.processcvu_requested_run_target = args.processcvu_target;
    base_route_opt.processcvu.pre_run_target = args.processcvu_target;
    base_route_opt.processcvu.post_run_target = args.processcvu_target;

    std::vector<simaai::neat::Model> resnet_lane_models;
    std::vector<simaai::neat::Model> yolo_lane_models;
    resnet_lane_models.reserve(static_cast<std::size_t>(args.resnet_lanes));
    yolo_lane_models.reserve(static_cast<std::size_t>(args.yolo_lanes));

    auto build_resnet_lane_nodes = [&](int lane) {
      if (!resnet_model)
        throw std::runtime_error("ResNet lane requested but ResNet model is inactive");
      const std::string suffix = "_resnet_" + std::to_string(lane);
      auto model_options = simaai::neat::internal::ModelAccess::options(*resnet_model);
      model_options.name_suffix = suffix;
      resnet_lane_models.emplace_back(
          simaai::neat::internal::ModelAccess::clone_with_options(*resnet_model, model_options));
      const simaai::neat::Model& lane_model = resnet_lane_models.back();
      auto opt = base_route_opt;
      opt.name_suffix = suffix;
      return simaai::neat::internal::ModelAccess::build_public_route_nodes(lane_model, opt);
    };
    auto build_yolo_lane_nodes = [&](int lane) {
      if (!yolo_model)
        throw std::runtime_error("YOLO lane requested but YOLO model is inactive");
      const std::string suffix = "_yolo_" + std::to_string(lane);
      auto model_options = simaai::neat::internal::ModelAccess::options(*yolo_model);
      model_options.name_suffix = suffix;
      yolo_lane_models.emplace_back(
          simaai::neat::internal::ModelAccess::clone_with_options(*yolo_model, model_options));
      const simaai::neat::Model& lane_model = yolo_lane_models.back();
      auto opt = base_route_opt;
      opt.name_suffix = suffix;
      auto nodes = simaai::neat::internal::ModelAccess::build_public_route_nodes(lane_model, opt);
      if (!nodes_contain_kind(nodes, "SimaBoxDecode")) {
        throw std::runtime_error(
            std::string("YOLO route did not contain SimaBoxDecode even though decode_type=") +
            simaai::neat::box_decode_type_token(args.yolo_decode_type));
      }
      return nodes;
    };

    simaai::neat::graph::Graph g;
    simaai::neat::graph::nodes::StreamSchedulerOptions sched_opt;
    sched_opt.per_stream_queue = static_cast<std::size_t>(args.scheduler_queue);
    sched_opt.drop_policy = parse_drop_policy(args.scheduler_drop);
    sched_opt.max_batch = args.scheduler_max_batch;

    std::vector<NodeId> resnet_sched_nodes;
    std::vector<NodeId> resnet_argmax_nodes;
    for (int lane = 0; lane < args.resnet_lanes; ++lane) {
      auto sched = g.add(simaai::neat::graph::nodes::StreamSchedulerNode(
          sched_opt, "resnet_sched_" + std::to_string(lane)));
      auto route = build_resnet_lane_nodes(lane);
      auto model_node = simaai::neat::graph::helpers::add_pipeline(
          g, std::move(route), "resnet_" + std::to_string(lane));
      auto argmax_node = g.add(simaai::neat::graph::nodes::TensorMap(
          [](simaai::neat::Sample& sample, simaai::neat::Tensor& tensor) {
            const Top1Result top1 = argmax_from_tensor(tensor, sample.stream_id.c_str());
            sample.tensors = simaai::neat::TensorList{make_top1_tensor(top1)};
            sample.tensor.reset();
            sample.kind = simaai::neat::SampleKind::TensorSet;
          },
          "argmax_resnet_" + std::to_string(lane)));
      simaai::neat::graph::helpers::chain(g, {sched, model_node, argmax_node});
      resnet_sched_nodes.push_back(sched);
      resnet_argmax_nodes.push_back(argmax_node);
    }

    std::vector<NodeId> yolo_sched_nodes;
    std::vector<NodeId> yolo_output_nodes;
    std::vector<NodeId> raw_output_nodes;
    std::vector<std::string> raw_output_labels;
    std::vector<NodeId> video_output_nodes;
    std::vector<int> video_output_stream_indexes;
    for (int lane = 0; lane < args.yolo_lanes; ++lane) {
      auto sched = g.add(simaai::neat::graph::nodes::StreamSchedulerNode(
          sched_opt, "yolo_sched_" + std::to_string(lane)));
      auto route = build_yolo_lane_nodes(lane);
      auto model_node = simaai::neat::graph::helpers::add_pipeline(g, std::move(route),
                                                                   "yolo_" + std::to_string(lane));
      simaai::neat::graph::helpers::chain(g, {sched, model_node});
      NodeId output_node = model_node;
      if (latency_tracker && latency_tracker->enabled()) {
        auto boxes_tap = g.add(simaai::neat::graph::nodes::Map(
            [latency_tracker](simaai::neat::Sample& sample) {
              latency_tracker->mark_boxes(sample);
            },
            "latency_boxes_yolo_" + std::to_string(lane)));
        g.connect(model_node, boxes_tap);
        output_node = boxes_tap;
      }
      if (args.yolo_discard_boxes) {
        auto discard = g.add(simaai::neat::graph::nodes::Map(
            [](simaai::neat::Sample& sample) {
              if (sample.tensor.has_value())
                simaai::neat::pipeline_internal::drop_tensor_holder(sample.tensor.value());
              for (const auto& tensor : sample.tensors)
                simaai::neat::pipeline_internal::drop_tensor_holder(tensor);
              sample.tensors = simaai::neat::TensorList{make_top1_tensor(Top1Result{0, 0.0f})};
              sample.tensor.reset();
              sample.kind = simaai::neat::SampleKind::TensorSet;
            },
            "discard_yolo_boxes_" + std::to_string(lane)));
        g.connect(output_node, discard);
        output_node = discard;
      }
      yolo_sched_nodes.push_back(sched);
      yolo_output_nodes.push_back(output_node);
    }

    auto add_insight_video_sender = [&](const StreamDef& stream) {
      std::vector<std::shared_ptr<simaai::neat::Node>> video_group{
          simaai::neat::nodes::H264Parse(/*config_interval=*/1),
          simaai::neat::nodes::H264Packetize(/*pt=*/kPayloadType, /*config_interval=*/1),
      };
      auto node = simaai::neat::graph::helpers::add_pipeline(g, std::move(video_group),
                                                             "insight_video_" + stream.stream_id);
      video_output_nodes.push_back(node);
      video_output_stream_indexes.push_back(stream.index);
      return node;
    };

    for (const auto& stream : stream_defs) {
      const auto& rtsp_ctx =
          rtsp_servers[static_cast<std::size_t>(stream.index % args.rtsp_servers)];
      std::vector<std::shared_ptr<simaai::neat::Node>> cap_group{
          simaai::neat::nodes::RTSPInput(rtsp_ctx.handle.url(), /*latency_ms=*/200, /*tcp=*/true),
          simaai::neat::nodes::H264Depacketize(kPayloadType, /*config_interval=*/1, args.fps,
                                               args.stream_width, args.stream_height,
                                               /*enforce_caps=*/true),
      };

      auto cap = simaai::neat::graph::helpers::add_pipeline(g, std::move(cap_group),
                                                            "cap_" + stream.stream_id);
      auto cap_log = g.add(simaai::neat::graph::nodes::Map(
          [](simaai::neat::Sample& sample) { log_edge("cap_out", sample); },
          "log_cap_" + stream.stream_id));
      StreamMetadataDefaults defaults;
      defaults.stream_id = stream.stream_id;
      defaults.stream_label = stream.stream_id;
      auto meta = g.add(
          simaai::neat::graph::nodes::StreamMetadataNode(defaults, "meta_" + stream.stream_id));
      auto meta_log = g.add(simaai::neat::graph::nodes::Map(
          [latency_tracker](simaai::neat::Sample& sample) {
            log_edge("meta_out", sample);
            if (latency_tracker)
              latency_tracker->mark_encoded(sample);
          },
          "log_meta_" + stream.stream_id));
      simaai::neat::graph::helpers::chain(g, {cap, cap_log, meta, meta_log});

      if (stream.branch == "encoded") {
        if (args.insight_video) {
          auto fan = g.add(simaai::neat::graph::nodes::FanOutNode(
              {"raw", "video"}, "fanout_encoded_" + stream.stream_id, "in"));
          auto video = add_insight_video_sender(stream);
          g.connect(meta_log, fan);
          g.connect(fan, video, "video", "in");
          raw_output_nodes.push_back(fan);
        } else {
          raw_output_nodes.push_back(meta_log);
        }
        raw_output_labels.push_back("encoded_" + stream.stream_id);
        continue;
      }

      auto dec = simaai::neat::graph::helpers::add_pipeline(
          g,
          simaai::neat::nodes::H264Decode(/*sima_allocator_type=*/2, /*out_format=*/"NV12",
                                          /*decoder_name=*/"decoder_" + stream.stream_id,
                                          /*raw_output=*/true, /*next_element=*/"CVU",
                                          /*dec_width=*/args.stream_width,
                                          /*dec_height=*/args.stream_height,
                                          /*dec_fps=*/args.fps,
                                          /*num_buffers=*/args.decoder_buffers),
          "dec_" + stream.stream_id);
      auto dec_log = g.add(simaai::neat::graph::nodes::Map(
          [latency_tracker](simaai::neat::Sample& sample) {
            log_edge("dec_out", sample);
            if (latency_tracker)
              latency_tracker->mark_decoded(sample);
          },
          "log_dec_" + stream.stream_id));
      if (args.insight_video) {
        auto fan = g.add(simaai::neat::graph::nodes::FanOutNode(
            {"decode", "video"}, "fanout_video_" + stream.stream_id, "in"));
        auto video = add_insight_video_sender(stream);
        g.connect(meta_log, fan);
        g.connect(fan, dec, "decode", "in");
        g.connect(fan, video, "video", "in");
        g.connect(dec, dec_log);
      } else {
        simaai::neat::graph::helpers::chain(g, {meta_log, dec, dec_log});
      }

      if (stream.branch == "decode") {
        raw_output_nodes.push_back(dec_log);
        raw_output_labels.push_back("decode_" + stream.stream_id);
        continue;
      }

      if (stream.branch == "both") {
        auto fan = g.add(simaai::neat::graph::nodes::FanOutNode(
            {"resnet", "yolo"}, "fanout_" + stream.stream_id, "in"));
        g.connect(dec_log, fan);
        g.connect(fan, resnet_sched_nodes[static_cast<std::size_t>(stream.resnet_lane)], "resnet",
                  "in");
        g.connect(fan, yolo_sched_nodes[static_cast<std::size_t>(stream.yolo_lane)], "yolo", "in");
      } else if (stream.branch == "resnet") {
        g.connect(dec_log, resnet_sched_nodes[static_cast<std::size_t>(stream.resnet_lane)]);
      } else if (stream.branch == "yolo") {
        g.connect(dec_log, yolo_sched_nodes[static_cast<std::size_t>(stream.yolo_lane)]);
      } else {
        throw std::runtime_error("unexpected stream branch: " + stream.branch);
      }
    }

    graph_nodes = static_cast<int>(g.node_count());
    graph_edges = static_cast<int>(g.edges().size());

    simaai::neat::graph::GraphRunOptions run_opt;
    run_opt.edge_queue = static_cast<std::size_t>(args.edge_queue);
    run_opt.push_timeout_ms = args.push_timeout_ms;
    run_opt.pull_timeout_ms = args.pull_timeout_ms;
    run_opt.pipeline.output_memory = parse_output_memory(args.output_memory);

    std::cout << "[graph] nodes=" << graph_nodes << " edges=" << graph_edges
              << " edge_queue=" << args.edge_queue << " output_memory=" << args.output_memory
              << "\n";
    const auto build_t0 = Clock::now();
    GraphRun run = simaai::neat::graph::helpers::build(std::move(g), run_opt);
    const auto build_t1 = Clock::now();
    timings.build_ms = elapsed_ms(build_t0, build_t1);

    std::vector<OutputDef> output_defs;
    output_defs.reserve(static_cast<std::size_t>(args.resnet_lanes + args.yolo_lanes +
                                                raw_output_nodes.size() +
                                                video_output_nodes.size()));
    for (int lane = 0; lane < args.resnet_lanes; ++lane) {
      output_defs.push_back({run.output(resnet_argmax_nodes[static_cast<std::size_t>(lane)]),
                             resnet_argmax_nodes[static_cast<std::size_t>(lane)], Branch::ResNet,
                             lane, "resnet_" + std::to_string(lane)});
    }
    for (int lane = 0; lane < args.yolo_lanes; ++lane) {
      output_defs.push_back({run.output(yolo_output_nodes[static_cast<std::size_t>(lane)]),
                             yolo_output_nodes[static_cast<std::size_t>(lane)], Branch::Yolo, lane,
                             "yolo_" + std::to_string(lane)});
    }
    for (std::size_t i = 0; i < raw_output_nodes.size(); ++i) {
      output_defs.push_back({run.output(raw_output_nodes[i]), raw_output_nodes[i], Branch::Raw,
                             static_cast<int>(i), raw_output_labels[i]});
    }
    for (std::size_t i = 0; i < video_output_nodes.size(); ++i) {
      const int stream_index = video_output_stream_indexes.at(i);
      output_defs.push_back({run.output(video_output_nodes[i]), video_output_nodes[i],
                             Branch::Video, stream_index,
                             "video_" + std::to_string(stream_index)});
    }
    std::vector<GraphRun::Output> outputs;
    std::unordered_map<NodeId, std::size_t> output_index_by_node;
    outputs.reserve(output_defs.size());
    for (std::size_t i = 0; i < output_defs.size(); ++i) {
      outputs.push_back(output_defs[i].handle);
      output_index_by_node.emplace(output_defs[i].node, i);
    }

    std::vector<sima_test::UdpReceiver> metadata_rx;
    std::vector<simaai::neat::MetadataSender> metadata_senders;
    std::vector<UdpPacketSender> video_senders;
    if (args.metadata_udp && args.yolo_lanes > 0) {
      metadata_rx.reserve(static_cast<std::size_t>(args.yolo_lanes));
      metadata_senders.reserve(static_cast<std::size_t>(args.yolo_lanes));
      for (int lane = 0; lane < args.yolo_lanes; ++lane) {
        metadata_rx.emplace_back(args.metadata_port_base + lane);
        simaai::neat::MetadataSenderOptions opt;
        opt.host = "127.0.0.1";
        opt.channel = lane;
        opt.metadata_port_base = args.metadata_port_base;
        std::string init_err;
        metadata_senders.emplace_back(opt, &init_err);
        require(metadata_senders.back().ok(), "MetadataSender init failed: " + init_err);
      }
    }
    if (args.insight_video) {
      video_senders.resize(static_cast<std::size_t>(args.streams));
      for (int stream_index = 0; stream_index < args.streams; ++stream_index) {
        std::string init_err;
        if (!video_senders[static_cast<std::size_t>(stream_index)].init(
                args.video_host, args.video_port_base + stream_index, &init_err)) {
          throw std::runtime_error("UDP video sender init failed for stream " +
                                   std::to_string(stream_index) + ": " + init_err);
        }
      }
    }

    std::map<std::string, int64_t> warm_resnet;
    std::map<std::string, int64_t> warm_yolo;
    std::map<std::string, int64_t> warm_raw;
    GraphRunStats warm_stats;
    const int64_t active_branch_instances = expected_outputs_for(stream_defs, 1);
    const int warm_timeout_ms = args.warmup_timeout_ms > 0
                                    ? args.warmup_timeout_ms
                                    : std::max<int64_t>(60000, active_branch_instances * 5000);
    const auto warm_t0 = Clock::now();
    auto warm_last_progress = warm_t0;
    while (!warmup_done(warm_resnet, warm_yolo, warm_raw, stream_defs, args.warmup_per_stream)) {
      if (elapsed_ms(warm_t0, Clock::now()) > static_cast<double>(warm_timeout_ms)) {
        throw std::runtime_error("Warmup timed out waiting for active stream/branch outputs");
      }
      if (elapsed_ms(warm_last_progress, Clock::now()) >
          static_cast<double>(args.stall_timeout_ms)) {
        throw std::runtime_error("Warmup stalled waiting for active stream/branch outputs");
      }
      NodeId out_node = simaai::neat::graph::kInvalidNode;
      auto sample = run.pull_any(outputs, args.pull_timeout_ms, &warm_stats, &out_node);
      if (!sample.has_value()) {
        const std::string err = run.last_error();
        if (!err.empty())
          throw std::runtime_error(err);
        continue;
      }
      warm_last_progress = Clock::now();
      const auto it = output_index_by_node.find(out_node);
      if (it == output_index_by_node.end())
        throw std::runtime_error("Warmup received output from unknown node");
      const OutputDef& def = output_defs.at(it->second);
      if (def.branch == Branch::ResNet)
        ++warm_resnet[sample->stream_id];
      else if (def.branch == Branch::Yolo)
        ++warm_yolo[sample->stream_id];
      else if (def.branch == Branch::Raw)
        ++warm_raw[sample->stream_id];
    }
    const auto warm_t1 = Clock::now();
    timings.warmup_ms = elapsed_ms(warm_t0, warm_t1);
    int64_t warm_outputs = 0;
    for (const auto& [_, c] : warm_resnet)
      warm_outputs += c;
    for (const auto& [_, c] : warm_yolo)
      warm_outputs += c;
    for (const auto& [_, c] : warm_raw)
      warm_outputs += c;
    std::cout << "[warmup] target_per_stream_branch=" << args.warmup_per_stream
              << " outputs=" << warm_outputs << " ms=" << timings.warmup_ms << "\n";
    if (latency_tracker && latency_tracker->enabled())
      latency_tracker->reset();
    GraphRunStats measured_stats;
    std::unordered_map<std::string, Top1Result> first_top1;
    std::unordered_map<std::string, Top1Result> last_top1;
    std::unordered_map<std::string, int> per_stream_goldfish;

    const auto measured_t0 = Clock::now();
    auto last_progress = measured_t0;
    std::string stop_reason;
    while (true) {
      const auto now_check = Clock::now();
      const double measured_elapsed_ms = elapsed_ms(measured_t0, now_check);
      const bool counts_done = measured_done(metrics, stream_defs, args.iters);
      if (counts_done && measured_elapsed_ms >= static_cast<double>(args.min_measured_ms)) {
        break;
      }
      if (args.duration_ms > 0 && measured_elapsed_ms >= static_cast<double>(args.duration_ms)) {
        stop_reason = "duration reached";
        break;
      }
      if (measured_elapsed_ms > static_cast<double>(args.max_runtime_ms)) {
        metrics.max_runtime_hit = true;
        stop_reason = "max runtime reached";
        break;
      }
      if (elapsed_ms(last_progress, now_check) > static_cast<double>(args.stall_timeout_ms)) {
        metrics.stall_hit = true;
        ++metrics.stalls;
        stop_reason = "no progress before stall timeout";
        break;
      }

      NodeId out_node = simaai::neat::graph::kInvalidNode;
      auto sample = run.pull_any(outputs, args.pull_timeout_ms, &measured_stats, &out_node);
      if (!sample.has_value()) {
        const std::string err = run.last_error();
        if (!err.empty())
          throw std::runtime_error(err);
        ++metrics.timeouts;
        continue;
      }

      const auto now = Clock::now();
      last_progress = now;
      const auto it = output_index_by_node.find(out_node);
      if (it == output_index_by_node.end())
        throw std::runtime_error("Received output from unknown node");
      const OutputDef& def = output_defs.at(it->second);
      log_edge(def.branch == Branch::ResNet
                   ? "resnet_out"
                   : (def.branch == Branch::Yolo
                          ? "yolo_out"
                          : (def.branch == Branch::Raw ? "raw_out" : "video_out")),
               *sample);
      if (def.branch == Branch::Video) {
        send_video_packet_if_enabled(metrics, def, *sample, video_senders, latency_tracker);
        continue;
      }
      record_sample_metrics(metrics, timings, def, *sample, measured_t0, now);

      if (def.branch == Branch::ResNet) {
        const Top1Result top1 = argmax_from_sample(*sample);
        if (first_top1.find(sample->stream_id) == first_top1.end())
          first_top1.emplace(sample->stream_id, top1);
        last_top1[sample->stream_id] = top1;
        if (args.validate_resnet_goldfish && top1.index == kGoldfishId)
          per_stream_goldfish[sample->stream_id]++;
      } else if (def.branch == Branch::Yolo) {
        send_yolo_metadata_if_enabled(args, metrics, def, *sample, metadata_senders,
                                      latency_tracker);
      }
    }
    const auto measured_t1 = Clock::now();
    timings.measured_ms = elapsed_ms(measured_t0, measured_t1);
    if (latency_tracker && latency_tracker->enabled())
      latency_tracker->freeze();

    profile_json = json::object();
    profile_json["graph_runtime"] = graph_run_stats_json(measured_stats);
    if (!args.processcvu_profile_jsonl.empty()) {
      profile_json["processcvu_internal"] =
          load_processcvu_profile_jsonl(args.processcvu_profile_jsonl);
    }

    if (args.metadata_udp) {
      for (const auto& rx : metadata_rx) {
        std::vector<std::string> packets;
        metrics.metadata_received +=
            rx.drain(&packets, /*max_packets=*/10000, /*timeout_ms_each=*/2);
      }
    }

    print_summary(args, metrics, timings, stream_defs);
    if (profile_json.contains("processcvu_internal"))
      print_processcvu_profile_summary(profile_json.at("processcvu_internal"));
    if (latency_tracker && latency_tracker->enabled())
      latency_tracker->print_summary();

    if (!stop_reason.empty()) {
      if (metrics.stall_hit)
        std::cout << "[stall] no progress for ms=" << args.stall_timeout_ms
                  << " outputs=" << metrics.total_outputs << " expected=" << expected_outputs
                  << "\n";
      if (metrics.max_runtime_hit)
        std::cout << "[max_runtime] reached ms=" << args.max_runtime_ms
                  << " outputs=" << metrics.total_outputs << " expected=" << expected_outputs
                  << "\n";
      throw std::runtime_error(stop_reason);
    }
    if (metrics.saw_empty_stream_id)
      throw std::runtime_error("Observed output with empty stream_id");
    if (metrics.total_outputs < expected_outputs)
      throw std::runtime_error("Measured loop ended before expected output count");
    if (args.resnet_lanes > 0 && metrics.resnet_outputs == 0)
      throw std::runtime_error("ResNet branch had lanes but produced zero outputs");
    if (args.yolo_lanes > 0 && metrics.yolo_outputs == 0)
      throw std::runtime_error("YOLO branch had lanes but produced zero outputs");
    if ((args.branch_mode == "encoded-only" || args.branch_mode == "decode-only") &&
        metrics.raw_outputs == 0)
      throw std::runtime_error("Raw branch was active but produced zero outputs");
    if (args.metadata_udp && args.yolo_lanes > 0 && metrics.metadata_sent == 0)
      throw std::runtime_error("MetadataSender was enabled but no YOLO metadata packets were sent");
    if (args.insight_video && metrics.video_packets == 0)
      throw std::runtime_error("Insight video sender was enabled but no RTP/H264 packets were sent");

    for (const auto& label : expected_lane_labels(stream_defs)) {
      if (counter_count(metrics.by_lane, label) == 0) {
        throw std::runtime_error("Assigned lane produced zero outputs: " + label);
      }
    }

    for (const auto& stream : stream_defs) {
      const int64_t resnet = counter_count(metrics.by_stream_resnet, stream.stream_id);
      const int64_t yolo = counter_count(metrics.by_stream_yolo, stream.stream_id);
      if (expects_resnet(stream) && resnet < args.iters) {
        std::cout << "[missing] stream=" << stream.stream_id << " branch=resnet count=" << resnet
                  << " target=" << args.iters << "\n";
        throw std::runtime_error("Missing ResNet outputs for stream " + stream.stream_id);
      }
      if (expects_yolo(stream) && yolo < args.iters) {
        std::cout << "[missing] stream=" << stream.stream_id << " branch=yolo count=" << yolo
                  << " target=" << args.iters << "\n";
        throw std::runtime_error("Missing YOLO outputs for stream " + stream.stream_id);
      }
      if (expects_raw(stream)) {
        const int64_t raw = counter_count(metrics.by_stream_total, stream.stream_id);
        if (raw < args.iters) {
          std::cout << "[missing] stream=" << stream.stream_id << " branch=" << stream.branch
                    << " count=" << raw << " target=" << args.iters << "\n";
          throw std::runtime_error("Missing raw outputs for stream " + stream.stream_id);
        }
      }
    }

    if (args.validate_resnet_goldfish && args.resnet_lanes > 0) {
      std::string bad_streams;
      for (const auto& stream : stream_defs) {
        if (!expects_resnet(stream))
          continue;
        const auto it_last = last_top1.find(stream.stream_id);
        if (it_last == last_top1.end() || it_last->second.index != kGoldfishId) {
          if (!bad_streams.empty())
            bad_streams += ", ";
          bad_streams += stream.stream_id;
        }
      }
      if (!bad_streams.empty())
        throw std::runtime_error("Goldfish top1 mismatch for streams: " + bad_streams);
    }

    run.stop();
    rtsp_guard.stop();

    emit_json("PASS", "");
    std::cout << "[OK] macro_mixed_multistream_multimodel_tput_test passed\n";
    return 0;
  } catch (const SkipTest& e) {
    try {
      emit_json("SKIP", e.what());
    } catch (const std::exception& write_err) {
      std::cerr << "[json] failed to write skip JSON: " << write_err.what() << "\n";
    }
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    if (is_dispatcher_unavailable(e.what())) {
      try {
        emit_json("SKIP", "dispatcher unavailable");
      } catch (const std::exception& write_err) {
        std::cerr << "[json] failed to write dispatcher-skip JSON: " << write_err.what() << "\n";
      }
      return skip_long_test("dispatcher unavailable");
    }
    try {
      emit_json("FAIL", e.what());
    } catch (const std::exception& write_err) {
      std::cerr << "[json] failed to write failure JSON: " << write_err.what() << "\n";
    }
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

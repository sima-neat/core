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
#include "nodes/rtp/H264Depacketize.h"
#include "nodes/sima/H264DecodeSima.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/H264Packetize.h"
#include "pipeline/Graph.h"
#include "pipeline/LatencyProfiler.h"
#include "pipeline/RuntimeMetrics.h"
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
  std::string profile_trace_out;

  int streams = 16;
  int rtsp_servers = 8;
  int fps = kDefaultFps;
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
  int yolo_top_k = 100;
  int metadata_port_base = 9900;
  int h264_bitrate_kbps = 500;
  int profiler_ring_capacity = 1048576;

  float yolo_score_threshold = 0.25f;
  float yolo_nms_iou = 0.50f;

  std::string branch_mode = "fanout-both"; // fanout-both|split|resnet-only|yolo-only
  std::string scheduler_drop = "oldest";   // oldest|newest
  std::string output_memory = "owned";     // owned|zero-copy
  std::string source = "rtsp";             // v1 only supports RTSP
  std::string processcvu_target = "AUTO";  // AUTO|EV74|A65

  bool metadata_udp = true;
  bool print_pipeline = false;
  bool rtsp_debug = false;
  bool lossless_rtsp = false;
  bool validate_resnet_goldfish = false;
  bool require_assets = false;
  bool allow_heavy = false;
  bool serial_pipeline_build = false;
  bool latency_profiler = true;
  bool gst_element_timings = false;
  bool gst_flow_debug = false;
  bool gst_boundary_probes = false;

  bool edge_queue_cli = false;
  bool push_timeout_cli = false;
  bool output_memory_cli = false;
  bool h264_bitrate_cli = false;
  bool serial_pipeline_build_cli = false;
  bool gst_element_timings_cli = false;
  bool gst_flow_debug_cli = false;
  bool gst_boundary_probes_cli = false;
};

enum class Branch { ResNet, Yolo };

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

static std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
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
  return b == Branch::ResNet ? "resnet" : "yolo";
}

static bool expects_resnet(const StreamDef& def) {
  return def.branch == "both" || def.branch == "resnet";
}

static bool expects_yolo(const StreamDef& def) {
  return def.branch == "both" || def.branch == "yolo";
}

static std::string branch_for_stream(int stream_index, const Args& args) {
  if (args.branch_mode == "fanout-both")
    return "both";
  if (args.branch_mode == "resnet-only")
    return "resnet";
  if (args.branch_mode == "yolo-only")
    return "yolo";
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
  sima_test::get_arg(argc, argv, "--profile-trace-out", args.profile_trace_out);

  sima_test::parse_int_arg(argc, argv, "--streams", args.streams);
  sima_test::parse_int_arg(argc, argv, "--rtsp-servers", args.rtsp_servers);
  sima_test::parse_int_arg(argc, argv, "--fps", args.fps);
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
  sima_test::parse_int_arg(argc, argv, "--yolo-top-k", args.yolo_top_k);
  sima_test::parse_int_arg(argc, argv, "--metadata-port-base", args.metadata_port_base);
  args.h264_bitrate_cli =
      sima_test::parse_int_arg(argc, argv, "--h264-bitrate-kbps", args.h264_bitrate_kbps);
  sima_test::parse_int_arg(argc, argv, "--profiler-ring-capacity", args.profiler_ring_capacity);

  sima_test::parse_float_arg(argc, argv, "--yolo-score-threshold", args.yolo_score_threshold);
  sima_test::parse_float_arg(argc, argv, "--yolo-nms-iou", args.yolo_nms_iou);

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

  parse_bool_arg(argc, argv, "--metadata-udp", args.metadata_udp);
  parse_bool_arg(argc, argv, "--validate-resnet-goldfish", args.validate_resnet_goldfish);
  parse_bool_arg(argc, argv, "--require-assets", args.require_assets);
  args.serial_pipeline_build_cli =
      parse_bool_arg(argc, argv, "--serial-pipeline-build", args.serial_pipeline_build);
  parse_bool_arg(argc, argv, "--latency-profiler", args.latency_profiler);
  args.gst_element_timings_cli =
      parse_bool_arg(argc, argv, "--gst-element-timings", args.gst_element_timings);
  args.gst_flow_debug_cli = parse_bool_arg(argc, argv, "--gst-flow-debug", args.gst_flow_debug);
  args.gst_boundary_probes_cli =
      parse_bool_arg(argc, argv, "--gst-boundary-probes", args.gst_boundary_probes);
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
  if (args.duration_ms < 0)
    throw std::runtime_error("--duration-ms must be >= 0");
  if (args.min_measured_ms < 0)
    throw std::runtime_error("--min-measured-ms must be >= 0");
  if (args.profiler_ring_capacity < 0)
    throw std::runtime_error("--profiler-ring-capacity must be >= 0");
  if (args.rtsp_servers <= 0)
    args.rtsp_servers = std::min(args.streams, 8);
  if (args.rtsp_servers > args.streams)
    args.rtsp_servers = args.streams;
  if (args.resnet_lanes < 0 || args.resnet_lanes > 8)
    throw std::runtime_error("--resnet-lanes must be in range 0..8");
  if (args.yolo_lanes < 0 || args.yolo_lanes > 8)
    throw std::runtime_error("--yolo-lanes must be in range 0..8");
  if (args.branch_mode != "fanout-both" && args.branch_mode != "split" &&
      args.branch_mode != "resnet-only" && args.branch_mode != "yolo-only") {
    throw std::runtime_error("--branch-mode must be fanout-both, split, resnet-only, or yolo-only");
  }
  if (args.branch_mode == "resnet-only") {
    args.yolo_lanes = 0;
  } else if (args.branch_mode == "yolo-only") {
    args.resnet_lanes = 0;
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
  if (args.h264_bitrate_kbps < 0)
    throw std::runtime_error("--h264-bitrate-kbps must be >= 0");
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
  }
  return true;
}

static std::vector<std::string> active_branches_json(const Args& args) {
  std::vector<std::string> out;
  if (args.branch_mode != "yolo-only" && args.resnet_lanes > 0)
    out.push_back("resnet");
  if (args.branch_mode != "resnet-only" && args.yolo_lanes > 0)
    out.push_back("yolo");
  return out;
}

static int64_t expected_outputs_for(const std::vector<StreamDef>& streams, int iters) {
  int64_t expected = 0;
  for (const auto& stream : streams) {
    if (expects_resnet(stream))
      expected += iters;
    if (expects_yolo(stream))
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

static json latency_profiler_json(const simaai::neat::ProfilerReport& report) {
  json kernels = json::array();
  for (const auto& agg : report.kernel_aggregates) {
    kernels.push_back({{"backend", agg.backend},
                       {"phase", "aggregate"},
                       {"kernel_name", agg.kernel_name},
                       {"stage_name", agg.stage_name},
                       {"physical_input_index", agg.physical_input_index},
                       {"output_slot", agg.output_slot},
                       {"count", agg.count},
                       {"total_ms", agg.total_ms},
                       {"avg_ms", agg.avg_ms()},
                       {"min_ms", agg.min_ms},
                       {"max_ms", agg.max_ms}});
  }

  json memcpy = json::array();
  for (const auto& site : report.memcpy_sites) {
    memcpy.push_back({{"site_name", site.site_name},
                      {"calls", site.calls},
                      {"total_bytes", site.total_bytes},
                      {"total_ms", site.total_ms()},
                      {"avg_ms", site.avg_ms()},
                      {"max_ms", static_cast<double>(site.max_ns) / 1.0e6}});
  }

  std::map<std::string, std::vector<double>> by_backend_phase;
  for (const auto& inv : report.kernel_invocations) {
    by_backend_phase[inv.backend + ":" + inv.phase].push_back(inv.duration_ms());
  }
  json percentiles = json::array();
  for (auto& [key, values] : by_backend_phase) {
    percentiles.push_back(
        {{"bucket", key},
         {"count", values.size()},
         {"p50_ms", percentile(values, 50.0)},
         {"p95_ms", percentile(values, 95.0)},
         {"p99_ms", percentile(values, 99.0)},
         {"max_ms", values.empty() ? 0.0 : *std::max_element(values.begin(), values.end())}});
  }

  return {{"enabled", true},
          {"profiler_emits", report.profiler_emits},
          {"profiler_dropped", report.profiler_dropped},
          {"kernel_invocation_count", report.kernel_invocations.size()},
          {"kernel_aggregates", kernels},
          {"kernel_latency_percentiles", percentiles},
          {"memcpy_sites", memcpy}};
}

static json runtime_metrics_json(const simaai::neat::RuntimeMetrics& metrics) {
  const std::string text = simaai::neat::runtime_metrics_to_json(metrics, 0);
  try {
    return json::parse(text);
  } catch (const std::exception& e) {
    return {{"parse_error", e.what()}, {"raw", text}};
  }
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
  const int64_t target_counted_outputs = std::min<int64_t>(metrics.total_outputs, expected_outputs);
  const int64_t excess_outputs =
      metrics.total_outputs > expected_outputs ? metrics.total_outputs - expected_outputs : 0;

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
      {"output_memory", args.output_memory},
      {"metadata_udp", args.metadata_udp},
      {"h264_bitrate_kbps", args.h264_bitrate_kbps},
      {"latency_profiler", args.latency_profiler},
      {"profiler_ring_capacity", args.profiler_ring_capacity},
      {"profile_trace_out", args.profile_trace_out},
      {"gst_element_timings", args.gst_element_timings},
      {"gst_flow_debug", args.gst_flow_debug},
      {"gst_boundary_probes", args.gst_boundary_probes},
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
      {"SIMA_PROCESSMLA_SAFE_ASYNC", env_string("SIMA_PROCESSMLA_SAFE_ASYNC")},
      {"SIMA_PROCESSMLA_SAFE_ASYNC_DEPTH", env_string("SIMA_PROCESSMLA_SAFE_ASYNC_DEPTH")},
      {"SIMA_GST_ELEMENT_TIMINGS", env_string("SIMA_GST_ELEMENT_TIMINGS")},
      {"SIMA_GST_FLOW_DEBUG", env_string("SIMA_GST_FLOW_DEBUG")},
      {"SIMA_GST_BOUNDARY_PROBES", env_string("SIMA_GST_BOUNDARY_PROBES")},
      {"SIMA_DISPATCHER_PROFILE", env_string("SIMA_DISPATCHER_PROFILE")},
      {"SIMA_RUNTIME_PROFILE", env_string("SIMA_RUNTIME_PROFILE")},
  };
  j["timing"] = {{"build_ms", timings.build_ms},
                 {"warmup_ms", timings.warmup_ms},
                 {"measured_ms", timings.measured_ms},
                 {"first_output_ms", timings.first_output_ms}};
  j["throughput"] = {
      {"offered_input_fps", static_cast<double>(args.streams * args.fps)},
      {"aggregate_output_fps", aggregate_fps},
      {"target_normalized_output_fps",
       measured_s > 0.0 ? static_cast<double>(target_counted_outputs) / measured_s : 0.0},
      {"resnet_output_fps", resnet_fps},
      {"yolo_output_fps", yolo_fps},
      {"effective_output_per_input_ratio",
       args.streams > 0 && args.fps > 0
           ? aggregate_fps / static_cast<double>(args.streams * args.fps)
           : 0.0},
  };
  j["counts"] = {{"expected_outputs", expected_outputs},
                 {"target_counted_outputs", target_counted_outputs},
                 {"excess_outputs", excess_outputs},
                 {"total_outputs", metrics.total_outputs},
                 {"resnet_outputs", metrics.resnet_outputs},
                 {"yolo_outputs", metrics.yolo_outputs},
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
  } else {
    ++metrics.yolo_outputs;
    record_counter(metrics.by_stream_yolo[sid], now);
  }
}

static void
send_yolo_metadata_if_enabled(const Args& args, Metrics& metrics, const OutputDef& def,
                              const simaai::neat::Sample& sample,
                              const std::vector<simaai::neat::MetadataSender>& senders) {
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
  } else {
    ++metrics.metadata_send_failed;
    if (env_flag("SIMA_METADATA_DEBUG", false)) {
      std::cerr << "[metadata] send failed lane=" << def.lane << " err=" << err << "\n";
    }
  }
}

static void print_summary(const Args& args, const Metrics& metrics, const Timings& timings,
                          const std::vector<StreamDef>& streams) {
  const double measured_s = timings.measured_ms / 1000.0;
  const double aggregate_fps =
      measured_s > 0.0 ? static_cast<double>(metrics.total_outputs) / measured_s : 0.0;
  std::cout << "[result] outputs_total=" << metrics.total_outputs
            << " resnet_outputs=" << metrics.resnet_outputs
            << " yolo_outputs=" << metrics.yolo_outputs << " aggregate_fps=" << aggregate_fps
            << "\n";
  std::cout << "[branch_tput] branch=resnet outputs=" << metrics.resnet_outputs << " fps="
            << (measured_s > 0.0 ? static_cast<double>(metrics.resnet_outputs) / measured_s : 0.0)
            << "\n";
  std::cout << "[branch_tput] branch=yolo outputs=" << metrics.yolo_outputs << " fps="
            << (measured_s > 0.0 ? static_cast<double>(metrics.yolo_outputs) / measured_s : 0.0)
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
              << " resnet=" << resnet << " yolo=" << yolo << " fps=" << fps << "\n";
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

  auto emit_json = [&](const std::string& status, const std::string& reason) {
    if (args.json_out.empty())
      return;
    json report = make_report(args, metrics, timings, stream_defs, expected_outputs, graph_nodes,
                              graph_edges, status, reason);
    report["profile"] = profile_json;
    write_json_atomic(args.json_out, report);
    std::cout << "[json] wrote " << args.json_out << "\n";
  };

  try {
    args = parse_args(argc, argv);

    require_element_or_skip("appsrc");
    require_element_or_skip("appsink");
    require_element_or_skip("identity");
    require_element_or_skip("rtspsrc");
    require_element_or_skip("rtph264depay");
    require_element_or_skip("h264parse");
    require_element_or_skip("rtph264pay");
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

    std::unique_ptr<simaai::neat::LatencyProfiler> latency_profiler;
    if (args.latency_profiler) {
      simaai::neat::LatencyProfilerOptions profiler_opt;
      profiler_opt.ring_capacity = static_cast<std::size_t>(args.profiler_ring_capacity);
      latency_profiler = std::make_unique<simaai::neat::LatencyProfiler>(profiler_opt);
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
    if (args.yolo_lanes > 0 && args.yolo_model.empty())
      args.yolo_model = sima_test::resolve_yolov8s_tar(args.root);
    if (args.yolo_lanes > 0 && args.yolo_model.empty()) {
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
        image_path, /*content_w=*/kEncWidth, /*content_h=*/kEncHeight, /*enc_w=*/kEncWidth,
        /*enc_h=*/kEncHeight, args.fps, args.port, args.rtsp_servers, rtsp_port_stride,
        rtp_port_offset, rtp_ports_per_server, rtp_port_stride, /*max_tries=*/5, args.rtsp_debug);
    RtspHandleGroup rtsp_guard;
    for (auto& rtsp : rtsp_servers)
      rtsp_guard.add(&rtsp.handle);

    for (auto& rtsp : rtsp_servers) {
      if (!probe_rtsp_encoded(rtsp.handle.url(), args.fps, kEncWidth, kEncHeight, args.rtsp_tries,
                              args.rtsp_timeout_ms, args.print_pipeline, args.rtsp_debug)) {
        skip_long_test_exception("RTSP probe failed before macro mixed stress graph");
      }
    }

    std::cout << "[setup] streams=" << args.streams << " rtsp_servers=" << args.rtsp_servers
              << " fps=" << args.fps << " branch_mode=" << args.branch_mode
              << " resnet_lanes=" << args.resnet_lanes << " yolo_lanes=" << args.yolo_lanes << "\n";
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
      yolo_cfg.decode_type = simaai::neat::BoxDecodeType::YoloV8;
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
            "YOLO route did not contain SimaBoxDecode even though decode_type=YoloV8");
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
    for (int lane = 0; lane < args.yolo_lanes; ++lane) {
      auto sched = g.add(simaai::neat::graph::nodes::StreamSchedulerNode(
          sched_opt, "yolo_sched_" + std::to_string(lane)));
      auto route = build_yolo_lane_nodes(lane);
      auto model_node = simaai::neat::graph::helpers::add_pipeline(g, std::move(route),
                                                                   "yolo_" + std::to_string(lane));
      simaai::neat::graph::helpers::chain(g, {sched, model_node});
      yolo_sched_nodes.push_back(sched);
      yolo_output_nodes.push_back(model_node);
    }

    for (const auto& stream : stream_defs) {
      const auto& rtsp_ctx =
          rtsp_servers[static_cast<std::size_t>(stream.index % args.rtsp_servers)];
      std::vector<std::shared_ptr<simaai::neat::Node>> cap_group{
          simaai::neat::nodes::RTSPInput(rtsp_ctx.handle.url(), /*latency_ms=*/200, /*tcp=*/true),
          simaai::neat::nodes::H264Depacketize(kPayloadType, /*config_interval=*/1, args.fps,
                                               kEncWidth, kEncHeight, /*enforce_caps=*/true),
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
          [](simaai::neat::Sample& sample) { log_edge("meta_out", sample); },
          "log_meta_" + stream.stream_id));
      auto dec = simaai::neat::graph::helpers::add_pipeline(
          g,
          simaai::neat::nodes::H264Decode(/*sima_allocator_type=*/2, /*out_format=*/"NV12",
                                          /*decoder_name=*/"decoder",
                                          /*raw_output=*/true),
          "dec_" + stream.stream_id);
      auto dec_log = g.add(simaai::neat::graph::nodes::Map(
          [](simaai::neat::Sample& sample) { log_edge("dec_out", sample); },
          "log_dec_" + stream.stream_id));
      simaai::neat::graph::helpers::chain(g, {cap, cap_log, meta, meta_log, dec, dec_log});

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
    output_defs.reserve(static_cast<std::size_t>(args.resnet_lanes + args.yolo_lanes));
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
    std::vector<GraphRun::Output> outputs;
    std::unordered_map<NodeId, std::size_t> output_index_by_node;
    outputs.reserve(output_defs.size());
    for (std::size_t i = 0; i < output_defs.size(); ++i) {
      outputs.push_back(output_defs[i].handle);
      output_index_by_node.emplace(output_defs[i].node, i);
    }

    std::vector<sima_test::UdpReceiver> metadata_rx;
    std::vector<simaai::neat::MetadataSender> metadata_senders;
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

    std::map<std::string, int64_t> warm_resnet;
    std::map<std::string, int64_t> warm_yolo;
    GraphRunStats warm_stats;
    const int64_t active_branch_instances = expected_outputs_for(stream_defs, 1);
    const int warm_timeout_ms = args.warmup_timeout_ms > 0
                                    ? args.warmup_timeout_ms
                                    : std::max<int64_t>(60000, active_branch_instances * 5000);
    const auto warm_t0 = Clock::now();
    auto warm_last_progress = warm_t0;
    while (!warmup_done(warm_resnet, warm_yolo, stream_defs, args.warmup_per_stream)) {
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
      else
        ++warm_yolo[sample->stream_id];
    }
    const auto warm_t1 = Clock::now();
    timings.warmup_ms = elapsed_ms(warm_t0, warm_t1);
    int64_t warm_outputs = 0;
    for (const auto& [_, c] : warm_resnet)
      warm_outputs += c;
    for (const auto& [_, c] : warm_yolo)
      warm_outputs += c;
    std::cout << "[warmup] target_per_stream_branch=" << args.warmup_per_stream
              << " outputs=" << warm_outputs << " ms=" << timings.warmup_ms << "\n";
    if (latency_profiler) {
      latency_profiler->mark_warmup_done();
    }

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
      log_edge(def.branch == Branch::ResNet ? "resnet_out" : "yolo_out", *sample);
      record_sample_metrics(metrics, timings, def, *sample, measured_t0, now);

      if (def.branch == Branch::ResNet) {
        const Top1Result top1 = argmax_from_sample(*sample);
        if (first_top1.find(sample->stream_id) == first_top1.end())
          first_top1.emplace(sample->stream_id, top1);
        last_top1[sample->stream_id] = top1;
        if (args.validate_resnet_goldfish && top1.index == kGoldfishId)
          per_stream_goldfish[sample->stream_id]++;
      } else {
        send_yolo_metadata_if_enabled(args, metrics, def, *sample, metadata_senders);
      }
    }
    const auto measured_t1 = Clock::now();
    timings.measured_ms = elapsed_ms(measured_t0, measured_t1);

    profile_json = json::object();
    if (latency_profiler) {
      simaai::neat::ProfilerReport profiler_report = latency_profiler->finalize();
      profile_json["latency_profiler"] = latency_profiler_json(profiler_report);
      if (!args.profile_trace_out.empty()) {
        const std::string trace = simaai::neat::LatencyProfiler::to_chrome_trace(profiler_report);
        try {
          write_json_atomic(args.profile_trace_out, json::parse(trace));
        } catch (const std::exception&) {
          std::ofstream out(args.profile_trace_out, std::ios::binary | std::ios::trunc);
          out << trace;
        }
        profile_json["trace_out"] = args.profile_trace_out;
      }
    } else {
      profile_json["latency_profiler"] = {{"enabled", false}};
    }
    profile_json["graph_runtime"] =
        runtime_metrics_json(run.metrics(simaai::neat::RuntimeMetricsOptions{
            .include_power = false,
            .include_diagnostics = true,
            .include_pipeline = false,
            .include_percentiles = false,
        }));

    if (args.metadata_udp) {
      for (const auto& rx : metadata_rx) {
        std::vector<std::string> packets;
        metrics.metadata_received +=
            rx.drain(&packets, /*max_packets=*/10000, /*timeout_ms_each=*/2);
      }
    }

    print_summary(args, metrics, timings, stream_defs);

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
    if (args.metadata_udp && args.yolo_lanes > 0 && metrics.metadata_sent == 0)
      throw std::runtime_error("MetadataSender was enabled but no YOLO metadata packets were sent");

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

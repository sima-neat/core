#include "graph/Graph.h"
#include "graph/GraphHelpers.h"
#include "graph/GraphMetadata.h"
#include "graph/nodes/Map.h"
#include "graph/nodes/StreamMetadata.h"
#include "graph/nodes/StreamScheduler.h"
#include "nodes/common/Output.h"
#include "nodes/io/StillImageInput.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264Depacketize.h"
#include "nodes/sima/H264DecodeSima.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/H264Packetize.h"
#include "pipeline/Session.h"
#include "gst/GstHelpers.h"
#include "model/Model.h"
#include "pipeline/internal/TensorUtil.h"

#include "asset_utils.h"
#include "cli_utils.h"
#include "rtsp_port_utils.h"
#include "test_utils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int kRtspPort = 8557;
constexpr int kRtspRtpPortOffset = 10000;
constexpr int kRtspFps = 30;
constexpr int kEncWidth = 256;
constexpr int kEncHeight = 256;
constexpr int kPayloadType = 96;
constexpr const char* kGoldfishUrl =
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
    "n01443537_goldfish.JPEG";
constexpr int kGoldfishId = 1; // ILSVRC2012 0-based index for "goldfish"

static bool edge_log_enabled() {
  return std::getenv("SIMA_GRAPH_EDGE_LOG") != nullptr;
}

static int env_int(const char* key, int fallback) {
  const char* v = std::getenv(key);
  if (!v || !*v)
    return fallback;
  return std::atoi(v);
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

struct RtspServerContext {
  simaai::neat::Session session;
  simaai::neat::RtspServerHandle handle;
};

struct RtspHandleGuard {
  explicit RtspHandleGuard(simaai::neat::RtspServerHandle* handle = nullptr) : handle_(handle) {}
  RtspHandleGuard(const RtspHandleGuard&) = delete;
  RtspHandleGuard& operator=(const RtspHandleGuard&) = delete;
  ~RtspHandleGuard() {
    if (handle_)
      handle_->stop();
  }
  void reset(simaai::neat::RtspServerHandle* handle) {
    handle_ = handle;
  }
  void release() {
    handle_ = nullptr;
  }

private:
  simaai::neat::RtspServerHandle* handle_ = nullptr;
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

static bool wait_for_rtsp_running(simaai::neat::RtspServerHandle& handle, int timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
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
  ctx.session.add(
      simaai::neat::nodes::StillImageInput(image_path, content_w, content_h, enc_w, enc_h, fps));
  // Use software encoder to force frequent IDR (x264enc uses key-int-max=1).
  ctx.session.add(simaai::neat::nodes::H264EncodeSW());
  ctx.session.add(simaai::neat::nodes::H264Parse(/*config_interval=*/1));
  ctx.session.add(simaai::neat::nodes::H264Packetize(/*pt=*/kPayloadType, /*config_interval=*/1));

  ctx.handle = ctx.session.run_rtsp({
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
  simaai::neat::Session p;
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
                                      const std::string& goldfish_url, bool use_goldfish) {
  if (!image_path.empty()) {
    if (!file_exists(image_path)) {
      throw std::runtime_error("Image not found: " + image_path);
    }
    return image_path;
  }

  if (!use_goldfish) {
    throw std::runtime_error("No image provided (use --image or --goldfish)");
  }

  const auto dst = sima_test::default_goldfish_path();
  if (!file_exists(dst.string())) {
    if (!sima_test::download_file(goldfish_url, dst)) {
      throw std::runtime_error("Failed to download goldfish image");
    }
  }
  return dst.string();
}

struct Top1Result {
  int index = -1;
  float prob = 0.0f;
};

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

  // Drop holder to return CVU buffers to pool ASAP.
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

} // namespace

int main(int argc, char** argv) {
  try {
    std::cout.setf(std::ios::unitbuf);
    if (!simaai::neat::element_exists("appsrc") || !simaai::neat::element_exists("appsink") ||
        !simaai::neat::element_exists("rtspsrc") || !simaai::neat::element_exists("rtph264depay") ||
        !simaai::neat::element_exists("h264parse") || !simaai::neat::element_exists("rtph264pay") ||
        !simaai::neat::element_exists("neatencoder") ||
        !simaai::neat::element_exists("neatdecoder")) {
      skip_long_test_exception("Missing required GStreamer elements for RTSP H264");
    }

    int num_streams = 2;
    int iterations = 200;
    int fps = kRtspFps;
    int port = kRtspPort;
    int rtsp_servers = 2;
    int rtsp_probe_tries = 8;
    int rtsp_probe_timeout_ms = 500;
    int max_runtime_ms = 0;
    int stall_timeout_ms = 30000;
    int model_instances = 1;
    bool use_goldfish = sima_test::has_flag(argc, argv, "--goldfish");
    const bool print_pipeline = sima_test::has_flag(argc, argv, "--print-pipeline");
    const bool rtsp_debug = sima_test::has_flag(argc, argv, "--rtsp-debug");
    const bool lossless_rtsp = sima_test::has_flag(argc, argv, "--lossless-rtsp");

    sima_test::parse_int_arg(argc, argv, "--streams", num_streams);
    sima_test::parse_int_arg(argc, argv, "--iters", iterations);
    sima_test::parse_int_arg(argc, argv, "--fps", fps);
    sima_test::parse_int_arg(argc, argv, "--port", port);
    sima_test::parse_int_arg(argc, argv, "--rtsp-servers", rtsp_servers);
    sima_test::parse_int_arg(argc, argv, "--rtsp-tries", rtsp_probe_tries);
    sima_test::parse_int_arg(argc, argv, "--rtsp-timeout-ms", rtsp_probe_timeout_ms);
    sima_test::parse_int_arg(argc, argv, "--max-runtime-ms", max_runtime_ms);
    sima_test::parse_int_arg(argc, argv, "--stall-timeout-ms", stall_timeout_ms);
    sima_test::parse_int_arg(argc, argv, "--models", model_instances);

    std::string image_path;
    std::string model_tar;
    std::string goldfish_url = kGoldfishUrl;

    sima_test::get_arg(argc, argv, "--image", image_path);
    sima_test::get_arg(argc, argv, "--model", model_tar);
    sima_test::get_arg(argc, argv, "--goldfish-url", goldfish_url);

    if (image_path.empty() && !use_goldfish) {
      use_goldfish = true;
    }

    if (num_streams <= 0) {
      throw std::runtime_error("--streams must be > 0");
    }
    if (iterations <= 0) {
      throw std::runtime_error("--iters must be > 0");
    }
    if (model_instances <= 0 || model_instances > 4) {
      throw std::runtime_error("--models must be between 1 and 4");
    }
    if (stall_timeout_ms < 30000) {
      throw std::runtime_error("--stall-timeout-ms must be >= 30000");
    }
    if (rtsp_servers <= 0) {
      rtsp_servers = std::min(num_streams, 8);
    }
    if (rtsp_servers > num_streams) {
      rtsp_servers = num_streams;
    }

    const int port_range = std::max(1, env_int("SIMA_RTSP_PORT_RANGE", 128));
    const int rtsp_port_stride = 10;
    const int rtp_port_offset =
        std::max(0, env_int("SIMA_RTSP_RTP_PORT_OFFSET", kRtspRtpPortOffset));
    const int rtp_ports_per_server =
        std::max(2, env_int("SIMA_RTSP_RTP_PORT_COUNT", rtsp_port_stride - 1));
    const int rtp_port_stride = std::max(1, env_int("SIMA_RTSP_RTP_PORT_STRIDE", rtsp_port_stride));
    const int chosen_port =
        rtsp_find_free_port_range_with_rtp(port, rtsp_servers, rtsp_port_stride, port_range,
                                           rtp_port_offset, rtp_ports_per_server, rtp_port_stride);
    if (chosen_port < 0) {
      throw std::runtime_error("Failed to find free RTSP port range");
    }
    if (chosen_port != port) {
      std::cerr << "[rtsp] base port " << port << " busy for " << rtsp_servers << " servers; using "
                << chosen_port << "\n";
      port = chosen_port;
    }

    if (rtsp_debug) {
      setenv("SIMA_GST_FLOW_DEBUG", "1", 1);
      setenv("SIMA_APPSINK_CAPS_DEBUG", "1", 1);
    }
    if (lossless_rtsp) {
      setenv("SIMA_H264ENC_LOSSLESS", "1", 1);
    }
    if (!std::getenv("SIMA_DISPATCHER_AUTO_RECOVER")) {
      setenv("SIMA_DISPATCHER_AUTO_RECOVER", "1", 1);
    }

    const std::string img = resolve_image_path(image_path, goldfish_url, use_goldfish);
    if (model_tar.empty()) {
      model_tar = sima_test::resolve_resnet50_tar();
    }
    if (model_tar.empty()) {
      skip_long_test_exception(
          "ResNet50 model pack not found (set SIMA_MODEL_TAR or SIMA_RESNET50_TAR or run "
          "'sima-cli modelzoo get resnet_50')");
    }

    auto rtsp_servers_vec = start_rtsp_servers_with_retry(
        img,
        /*content_w=*/kEncWidth,
        /*content_h=*/kEncHeight,
        /*enc_w=*/kEncWidth,
        /*enc_h=*/kEncHeight, fps, port, rtsp_servers, rtsp_port_stride, rtp_port_offset,
        rtp_ports_per_server, rtp_port_stride,
        /*max_tries=*/5, rtsp_debug);
    RtspHandleGroup rtsp_guard;
    for (auto& rtsp : rtsp_servers_vec) {
      rtsp_guard.add(&rtsp.handle);
    }

    for (auto& rtsp : rtsp_servers_vec) {
      if (!probe_rtsp_encoded(rtsp.handle.url(), fps, kEncWidth, kEncHeight, rtsp_probe_tries,
                              rtsp_probe_timeout_ms, print_pipeline, rtsp_debug)) {
        if (rtsp_debug)
          std::cerr << "[rtsp] probe failed; server will stop\n";
        skip_long_test_exception("RTSP probe failed (no output). Check encoder/RTSP server.");
      }
    }

    std::cout << "[setup] streams=" << num_streams << " iters=" << iterations << " fps=" << fps
              << " model=" << model_tar << " rtsp_servers=" << rtsp_servers << "\n";

    if (rtsp_debug)
      std::cerr << "[model] loading simaai::neat::Model\n";
    simaai::neat::Model::Options model_cfg;
    model_cfg.preprocess.kind = simaai::neat::InputKind::Image;
    model_cfg.preprocess.enable = simaai::neat::AutoFlag::On;
    model_cfg.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
    model_cfg.preprocess.preset = simaai::neat::NormalizePreset::ImageNet;
    simaai::neat::Model model(model_tar, model_cfg);
    if (rtsp_debug)
      std::cerr << "[model] simaai::neat::Model loaded\n";

    simaai::neat::Model::SessionOptions model_opt;
    model_opt.include_appsrc = false;
    model_opt.include_appsink = false;

    // ---------------------------------------------------------------------
    // Build pipeline definitions first.
    // ---------------------------------------------------------------------
    struct StreamPipelineDef {
      std::string stream_id;
      simaai::neat::NodeGroup cap;
      std::shared_ptr<simaai::neat::Node> dec;
      simaai::neat::graph::StreamMetadataDefaults meta_defaults;
      int sched_idx = 0;
    };

    std::vector<simaai::neat::NodeGroup> model_pipelines;
    for (int i = 0; i < model_instances; ++i) {
      model_pipelines.push_back(model.session(model_opt));
    }

    std::vector<std::string> stream_ids;
    std::vector<StreamPipelineDef> stream_defs;

    for (int i = 0; i < num_streams; ++i) {
      const std::string sid = "stream" + std::to_string(i);
      stream_ids.push_back(sid);

      const auto& rtsp_ctx =
          rtsp_servers_vec[static_cast<std::size_t>(i % std::max(1, rtsp_servers))];
      simaai::neat::NodeGroup cap_group({
          simaai::neat::nodes::RTSPInput(rtsp_ctx.handle.url(),
                                         /*latency_ms=*/200,
                                         /*tcp=*/true),
          simaai::neat::nodes::H264Depacketize(kPayloadType,
                                               /*config_interval=*/1,
                                               /*fps=*/fps,
                                               /*w=*/kEncWidth,
                                               /*h=*/kEncHeight,
                                               /*enforce_caps=*/true),
      });

      StreamPipelineDef def;
      def.stream_id = sid;
      def.meta_defaults.stream_id = sid;
      def.cap = std::move(cap_group);
      def.dec = simaai::neat::nodes::H264Decode(/*sima_allocator_type=*/2,
                                                /*out_format=*/"NV12",
                                                /*decoder_name=*/"decoder",
                                                /*raw_output=*/true);
      if (model_instances == 2) {
        def.sched_idx = (i < num_streams / 2) ? 0 : 1;
      } else if (model_instances > 2) {
        def.sched_idx = i % model_instances;
      }
      stream_defs.emplace_back(std::move(def));
    }

    // ---------------------------------------------------------------------
    // Build graph and connect pipelines.
    // ---------------------------------------------------------------------
    simaai::neat::graph::Graph g;

    simaai::neat::graph::nodes::StreamSchedulerOptions sched_opt;
    sched_opt.per_stream_queue = 2;
    sched_opt.drop_policy = simaai::neat::graph::nodes::StreamDropPolicy::DropOldest;
    sched_opt.max_batch = 1;

    const auto label_for_index = [](int idx) -> std::string {
      if (idx >= 0 && idx < 26) {
        return std::string(1, static_cast<char>('a' + idx));
      }
      return std::to_string(idx);
    };

    std::vector<simaai::neat::graph::NodeId> sched_nodes;
    std::vector<simaai::neat::graph::NodeId> argmax_nodes;

    for (int i = 0; i < model_instances; ++i) {
      const std::string suffix = label_for_index(i);
      auto sched =
          g.add(simaai::neat::graph::nodes::StreamSchedulerNode(sched_opt, "sched_" + suffix));
      auto model_node = simaai::neat::graph::helpers::add_pipeline(
          g, std::move(model_pipelines[static_cast<std::size_t>(i)]), "resnet_" + suffix);
      auto argmax_node = g.add(simaai::neat::graph::nodes::TensorMap(
          [](simaai::neat::Sample& sample, simaai::neat::Tensor& tensor) {
            const Top1Result top1 = argmax_from_tensor(tensor, sample.stream_id.c_str());
            sample.tensors = simaai::neat::TensorList{make_top1_tensor(top1)};
            sample.tensor.reset();
            sample.kind = simaai::neat::SampleKind::TensorSet;
          },
          "argmax_" + suffix));

      simaai::neat::graph::helpers::chain(g, {sched, model_node, argmax_node});
      sched_nodes.push_back(sched);
      argmax_nodes.push_back(argmax_node);
    }

    for (auto& def : stream_defs) {
      auto cap =
          simaai::neat::graph::helpers::add_pipeline(g, std::move(def.cap), "cap_" + def.stream_id);
      auto cap_log = g.add(simaai::neat::graph::nodes::Map(
          [](simaai::neat::Sample& sample) { log_edge("cap_out", sample); },
          "log_cap_" + def.stream_id));
      auto meta = g.add(simaai::neat::graph::nodes::StreamMetadataNode(def.meta_defaults,
                                                                       "meta_" + def.stream_id));
      auto meta_log = g.add(simaai::neat::graph::nodes::Map(
          [](simaai::neat::Sample& sample) { log_edge("meta_out", sample); },
          "log_meta_" + def.stream_id));
      auto dec =
          simaai::neat::graph::helpers::add_pipeline(g, std::move(def.dec), "dec_" + def.stream_id);
      auto dec_log = g.add(simaai::neat::graph::nodes::Map(
          [](simaai::neat::Sample& sample) { log_edge("dec_out", sample); },
          "log_dec_" + def.stream_id));
      simaai::neat::graph::helpers::chain(g,
                                          {cap, cap_log, meta, meta_log, dec, dec_log,
                                           sched_nodes[static_cast<std::size_t>(def.sched_idx)]});
    }

    simaai::neat::graph::GraphRunOptions run_opt;
    // This graph fan-in/fan-out path can transiently backlog when RTSP sources
    // and MLA stages are not phase-aligned. Keep a wider queue and push budget
    // so test stability does not depend on host scheduling jitter.
    run_opt.edge_queue = std::max(128, env_int("SIMA_GRAPH_EDGE_QUEUE", 512));
    run_opt.push_timeout_ms = std::max(5000, env_int("SIMA_GRAPH_PUSH_TIMEOUT_MS", 20000));
    run_opt.pull_timeout_ms = 50;
    run_opt.pipeline.output_memory = env_flag("SIMA_GRAPH_COPY_OUTPUT", true)
                                         ? simaai::neat::OutputMemory::Owned
                                         : simaai::neat::OutputMemory::ZeroCopy;

    if (rtsp_debug)
      std::cerr << "[graph] build start\n";
    simaai::neat::graph::GraphRun run = simaai::neat::graph::helpers::build(std::move(g), run_opt);
    if (rtsp_debug)
      std::cerr << "[graph] build done\n";

    std::vector<simaai::neat::graph::GraphRun::Output> outputs;
    std::unordered_map<simaai::neat::graph::NodeId, std::size_t> output_index;
    for (std::size_t i = 0; i < argmax_nodes.size(); ++i) {
      outputs.emplace_back(run.output(argmax_nodes[i]));
      output_index.emplace(argmax_nodes[i], i);
    }

    const int warmup = std::max(1, num_streams);
    int warmup_timeout_ms = 30000;
    sima_test::parse_int_arg(argc, argv, "--warmup-timeout-ms", warmup_timeout_ms);
    if (rtsp_debug)
      std::cerr << "[graph] warmup start\n";
    if (!run.warmup(outputs, warmup, warmup_timeout_ms)) {
      const std::string err = run.last_error();
      if (!err.empty()) {
        throw std::runtime_error(err);
      }
      throw std::runtime_error("Warmup timed out waiting for model output");
    }
    if (rtsp_debug)
      std::cerr << "[graph] warmup done\n";

    auto session = run.collect(outputs);
    auto& output_stats = session.stats();

    int seen = 0;
    std::vector<int> seen_by_model(outputs.size(), 0);

    std::unordered_map<std::string, Top1Result> first_top1;
    std::unordered_map<std::string, Top1Result> last_top1;
    std::unordered_map<std::string, int> per_stream_goldfish;

    const auto t0 = std::chrono::steady_clock::now();

    session.per_stream_target(iterations)
        .stall_after_ms(stall_timeout_ms)
        .timeout_ms(100)
        .max_runtime_ms(max_runtime_ms)
        .expect_streams(stream_ids)
        .on_sample([&](const simaai::neat::Sample& sample, simaai::neat::graph::NodeId out_node) {
          seen++;
          auto it_idx = output_index.find(out_node);
          if (it_idx != output_index.end()) {
            seen_by_model[it_idx->second]++;
          }
          log_edge("model_out", sample);
          const Top1Result top1 = argmax_from_sample(sample);
          if (first_top1.find(sample.stream_id) == first_top1.end()) {
            first_top1.emplace(sample.stream_id, top1);
          }
          last_top1[sample.stream_id] = top1;
          if (use_goldfish && top1.index == kGoldfishId) {
            per_stream_goldfish[sample.stream_id]++;
          }
        });

    try {
      session.run();
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      if (msg.find("max runtime") != std::string::npos) {
        std::cout << "[partial] max-runtime reached ms=" << max_runtime_ms << " seen=" << seen
                  << "\n";
      } else if (msg.find("stalled") != std::string::npos) {
        std::cout << "[stall] per-stream target stalled ms=" << stall_timeout_ms << " seen=" << seen
                  << "\n";
      } else {
        std::cout << "[timeout] no outputs\n";
      }
      run.emit_summary();
      throw;
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    const double tput = (secs > 0.0) ? (static_cast<double>(seen) / secs) : 0.0;

    std::cout << "[result] outputs_total=" << seen;
    for (std::size_t i = 0; i < seen_by_model.size(); ++i) {
      const std::string suffix = label_for_index(static_cast<int>(i));
      std::cout << " outputs_" << suffix << "=" << seen_by_model[i];
    }
    std::cout << " tput_fps=" << tput << "\n";
    const auto stream_counts = output_stats.stream_counts();
    if (!stream_ids.empty() && secs > 0.0) {
      int min_seen = std::numeric_limits<int>::max();
      int max_seen = std::numeric_limits<int>::min();
      double sum_fps = 0.0;
      for (const auto& sid : stream_ids) {
        const auto it = stream_counts.find(sid);
        const int count = (it == stream_counts.end()) ? 0 : static_cast<int>(it->second);
        const double fps = static_cast<double>(count) / secs;
        min_seen = std::min(min_seen, count);
        max_seen = std::max(max_seen, count);
        sum_fps += fps;
        std::cout << "[stream_tput] stream=" << sid << " seen=" << count << " fps=" << fps << "\n";
      }
      const double avg_fps = sum_fps / static_cast<double>(stream_ids.size());
      std::cout << "[stream_tput_summary] streams=" << stream_ids.size() << " min_seen=" << min_seen
                << " max_seen=" << max_seen << " avg_fps=" << avg_fps << "\n";
    }
    run.emit_summary();

    std::string missing_list;
    for (const auto& sid : stream_ids) {
      if (first_top1.find(sid) == first_top1.end()) {
        if (!missing_list.empty())
          missing_list += ", ";
        missing_list += sid;
      }
    }
    if (!missing_list.empty()) {
      throw std::runtime_error("Missing outputs for streams: " + missing_list);
    }

    int printed = 0;
    for (const auto& [stream_id, top1] : first_top1) {
      std::cout << "[top1] stream=" << stream_id << " class=" << top1.index << " prob=" << top1.prob
                << "\n";
      printed++;
      if (printed >= num_streams)
        break;
    }

    if (use_goldfish) {
      std::string bad_streams;
      for (const auto& sid : stream_ids) {
        const auto it_last = last_top1.find(sid);
        if (it_last != last_top1.end() && it_last->second.index != kGoldfishId) {
          if (!bad_streams.empty())
            bad_streams += ", ";
          bad_streams += sid;
        }
      }
      if (!bad_streams.empty()) {
        throw std::runtime_error("Goldfish top1 mismatch for streams: " + bad_streams);
      }
      bool any_stream_miss = false;
      for (const auto& sid : stream_ids) {
        const auto it = stream_counts.find(sid);
        const int total = (it == stream_counts.end()) ? 0 : static_cast<int>(it->second);
        const int ok = per_stream_goldfish[sid];
        if (total > 0 && ok < total) {
          any_stream_miss = true;
          const auto it_last = last_top1.find(sid);
          const int last_class = (it_last != last_top1.end()) ? it_last->second.index : -1;
          std::cout << "[goldfish] stream=" << sid << " ok=" << ok << " total=" << total
                    << " last_class=" << last_class << "\n";
        }
      }
      int ok = 0;
      for (const auto& [stream_id, top1] : first_top1) {
        if (top1.index == kGoldfishId)
          ok++;
      }
      if (!first_top1.empty() && ok == 0) {
        throw std::runtime_error("Goldfish top1 mismatch (no stream predicted class 1)");
      }
      if (any_stream_miss) {
        std::cout << "[goldfish] note: some streams produced non-goldfish outputs\n";
      }
    }

    run.stop();

    std::cout << "[OK] hybrid_graph_rtsp_resnet50_tput_test passed\n";
    return 0;
  } catch (const SkipTest& e) {
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    if (is_dispatcher_unavailable(e.what())) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

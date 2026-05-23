#include "asset_utils.h"
#include "cli_utils.h"
#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "nodes/common/Output.h"
#include "nodes/common/Caps.h"
#include "nodes/common/Queue.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/io/StillImageInput.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264Depacketize.h"
#include "nodes/sima/H264DecodeSima.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/H264Packetize.h"
#include "pipeline/Graph.h"
#include "rtsp_port_utils.h"
#include "test_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kDefaultFps = 30;
constexpr int kDefaultPort = 8557;
constexpr int kDefaultRtpPortOffset = 10000;
constexpr int kDefaultEncWidth = 256;
constexpr int kDefaultEncHeight = 256;
constexpr int kPayloadType = 96;
constexpr int kDefaultMaxFrames = 60;
constexpr int kDefaultTimeoutMs = 15000;
constexpr const char* kGoldfishUrl =
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
    "n01443537_goldfish.JPEG";

struct RtspServerContext {
  simaai::neat::Graph graph;
  simaai::neat::RtspServerHandle handle;
};

struct RtspHandleGuard {
  explicit RtspHandleGuard(simaai::neat::RtspServerHandle* handle) : handle_(handle) {}
  ~RtspHandleGuard() {
    if (handle_)
      handle_->stop();
  }
  RtspHandleGuard(const RtspHandleGuard&) = delete;
  RtspHandleGuard& operator=(const RtspHandleGuard&) = delete;

private:
  simaai::neat::RtspServerHandle* handle_ = nullptr;
};

struct Nv12Frame {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> nv12;
};

struct Metrics {
  double mae = 0.0;
  double mse = 0.0;
  double psnr = 0.0;
  double max_abs = 0.0;
};

static int env_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v)
    return fallback;
  try {
    return std::stoi(v);
  } catch (...) {
    return fallback;
  }
}

static double env_double(const char* name, double fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v)
    return fallback;
  try {
    return std::stod(v);
  } catch (...) {
    return fallback;
  }
}

static std::string env_or(const char* name, const char* def_value) {
  const char* v = std::getenv(name);
  if (v && *v)
    return std::string(v);
  return std::string(def_value ? def_value : "");
}

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
  ctx.graph.add(
      simaai::neat::nodes::StillImageInput(image_path, content_w, content_h, enc_w, enc_h, fps));
  ctx.graph.add(simaai::neat::nodes::H264EncodeSW(/*bitrate_kbps=*/400));
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
                                                      int rtp_port_count, int max_tries) {
  for (int i = 0; i < max_tries; ++i) {
    const int port = base_port + i;
    const int rtp_base = rtp_port_base + i;
    RtspServerContext ctx = start_rtsp_server(image_path, content_w, content_h, enc_w, enc_h, fps,
                                              port, rtp_base, rtp_port_count);
    if (wait_for_rtsp_running(ctx.handle, 2000)) {
      return ctx;
    }
    ctx.handle.stop();
  }
  throw std::runtime_error("Failed to start RTSP server (port unavailable?)");
}

static Metrics compare_bytes(const uint8_t* a, const uint8_t* b, std::size_t n) {
  if (!a || !b || n == 0)
    return {};
  double sum_abs = 0.0;
  double sum_sq = 0.0;
  double max_abs = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const int diff = static_cast<int>(a[i]) - static_cast<int>(b[i]);
    const double ad = std::abs(diff);
    sum_abs += ad;
    sum_sq += static_cast<double>(diff * diff);
    if (ad > max_abs)
      max_abs = ad;
  }
  const double mae = sum_abs / static_cast<double>(n);
  const double mse = sum_sq / static_cast<double>(n);
  const double psnr = (mse <= 1e-12) ? 1e9 : (10.0 * std::log10((255.0 * 255.0) / mse));
  return {mae, mse, psnr, max_abs};
}

static Metrics compare_nv12_luma(const Nv12Frame& a, const Nv12Frame& b) {
  if (a.width != b.width || a.height != b.height) {
    throw std::runtime_error("compare_nv12_luma: size mismatch");
  }
  const std::size_t y_bytes =
      static_cast<std::size_t>(a.width) * static_cast<std::size_t>(a.height);
  if (a.nv12.size() < y_bytes || b.nv12.size() < y_bytes) {
    throw std::runtime_error("compare_nv12_luma: buffer too small");
  }
  return compare_bytes(a.nv12.data(), b.nv12.data(), y_bytes);
}

static void dump_nv12(const Nv12Frame& f, const std::string& path) {
  std::error_code ec;
  const std::filesystem::path p(path);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path(), ec);
  }
  std::ofstream ofs(path, std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(f.nv12.data()),
            static_cast<std::streamsize>(f.nv12.size()));
  std::cerr << "[DUMP] " << path << " (" << f.nv12.size() << " bytes)\n";
}

static Nv12Frame tensor_to_nv12(const simaai::neat::Tensor& t) {
  if (!t.is_nv12()) {
    throw std::runtime_error("Expected NV12 tensor output");
  }
  const int w = t.width();
  const int h = t.height();
  if (w <= 0 || h <= 0) {
    throw std::runtime_error("NV12 tensor has invalid dimensions");
  }
  Nv12Frame out;
  out.width = w;
  out.height = h;
  out.nv12 = t.copy_nv12_contiguous();
  return out;
}

static std::vector<Nv12Frame> pull_frames(simaai::neat::Run& runner, int max_frames, int timeout_ms,
                                          const std::string& label) {
  if (!runner.can_pull()) {
    throw std::runtime_error(label + ": pipeline cannot pull (missing Output)");
  }
  if (runner.can_push()) {
    throw std::runtime_error(label + ": pipeline expects input (AppSrc present)");
  }

  std::vector<Nv12Frame> frames;
  frames.reserve(static_cast<std::size_t>(max_frames));

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (static_cast<int>(frames.size()) < max_frames) {
    const int remaining_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                  deadline - std::chrono::steady_clock::now())
                                                  .count());
    if (remaining_ms <= 0)
      break;

    simaai::neat::Sample out;
    simaai::neat::PullError err;
    const int per_try = std::min(200, remaining_ms);
    const simaai::neat::PullStatus status = runner.pull(per_try, out, &err);
    if (status == simaai::neat::PullStatus::Timeout) {
      continue;
    }
    if (status == simaai::neat::PullStatus::Closed) {
      break;
    }
    if (status == simaai::neat::PullStatus::Error) {
      std::string msg = err.message.empty() ? (label + ": pull failed") : err.message;
      if (!err.code.empty())
        msg += " (code=" + err.code + ")";
      throw std::runtime_error(msg);
    }

    const auto tensors = simaai::neat::tensors_from_sample(out, true);
    if (tensors.size() != 1U) {
      throw std::runtime_error(label + ": expected one tensor output");
    }
    frames.push_back(tensor_to_nv12(tensors.front()));
  }

  if (static_cast<int>(frames.size()) < max_frames) {
    throw std::runtime_error(label + ": insufficient frames (got " + std::to_string(frames.size()) +
                             " expected " + std::to_string(max_frames) + ")");
  }
  return frames;
}

static std::string pick_os_decoder() {
  std::string env = env_or("SIMA_OS_DECODER_ELEMENT", "");
  if (!env.empty())
    return env;
  if (simaai::neat::element_exists("avdec_h264"))
    return "avdec_h264";
  if (simaai::neat::element_exists("openh264dec"))
    return "openh264dec";
  return "";
}

static std::string find_image_path(int argc, char** argv) {
  std::string image_path;
  if (sima_test::get_arg(argc, argv, "--image", image_path)) {
    return image_path;
  }
  image_path = env_or("SIMA_DECODER_COMPARE_IMAGE", "");
  if (!image_path.empty())
    return image_path;

  std::error_code ec;
  std::filesystem::path cur = std::filesystem::current_path(ec);
  for (int i = 0; i < 6 && !cur.empty(); ++i) {
    const std::filesystem::path candidate = cur / "test.jpg";
    if (std::filesystem::exists(candidate, ec))
      return candidate.string();
    const auto parent = cur.parent_path();
    if (parent == cur)
      break;
    cur = parent;
  }

  const std::filesystem::path goldfish = sima_test::default_goldfish_path();
  if (sima_test::download_file(kGoldfishUrl, goldfish)) {
    return goldfish.string();
  }
  return "";
}

static std::vector<Nv12Frame> decode_rtsp_frames(const std::string& url, const std::string& decoder,
                                                 bool use_neatdecoder, int max_frames,
                                                 int timeout_ms) {
  simaai::neat::Graph p;
  p.add(simaai::neat::nodes::RTSPInput(url, /*latency_ms=*/200, /*tcp=*/true));
  p.add(simaai::neat::nodes::H264Depacketize(/*payload_type=*/kPayloadType,
                                             /*h264_parse_config_interval=*/1));
  p.add(simaai::neat::nodes::Queue());
  if (use_neatdecoder) {
    p.add(simaai::neat::nodes::H264Decode(/*sima_allocator_type=*/2, /*out_format=*/"NV12"));
  } else {
    p.custom(decoder);
    p.add(simaai::neat::nodes::VideoConvert());
    p.add(simaai::neat::nodes::CapsNV12SysMem(-1, -1, -1));
  }
  p.add(simaai::neat::nodes::Queue());
  p.add(simaai::neat::nodes::Output(simaai::neat::OutputOptions::EveryFrame(max_frames)));

  simaai::neat::RunOptions run_opt;
  run_opt.output_memory = simaai::neat::OutputMemory::Owned;
  simaai::neat::Run runner = p.build(run_opt);

  auto frames =
      pull_frames(runner, max_frames, timeout_ms, use_neatdecoder ? "neatdecoder" : decoder);
  runner.stop();
  return frames;
}

} // namespace

int main(int argc, char** argv) {
  try {
    simaai::neat::gst_init_once();

    if (!simaai::neat::element_exists("rtspsrc") || !simaai::neat::element_exists("rtph264depay") ||
        !simaai::neat::element_exists("h264parse") || !simaai::neat::element_exists("rtph264pay") ||
        !simaai::neat::element_exists("appsink") || !simaai::neat::element_exists("videoconvert")) {
      throw std::runtime_error("Missing required GStreamer elements for RTSP decode pipeline");
    }

    if (!simaai::neat::element_exists("neatdecoder")) {
      throw std::runtime_error("Missing required GStreamer element: neatdecoder");
    }

    const std::string os_decoder = pick_os_decoder();
    if (os_decoder.empty() || !simaai::neat::element_exists(os_decoder.c_str())) {
      throw std::runtime_error("Missing open-source H264 decoder (avdec_h264 or openh264dec)");
    }

    if (!(simaai::neat::element_exists("x264enc") || simaai::neat::element_exists("openh264enc") ||
          simaai::neat::element_exists("avenc_h264"))) {
      throw std::runtime_error(
          "Missing software H264 encoder for RTSP server (x264enc/openh264enc/avenc_h264)");
    }

    const std::string image_path = find_image_path(argc, argv);
    if (image_path.empty()) {
      throw std::runtime_error("Failed to resolve image path for RTSP server");
    }
    if (!file_exists(image_path)) {
      throw std::runtime_error("Image path missing: " + image_path);
    }

    int enc_w = kDefaultEncWidth;
    int enc_h = kDefaultEncHeight;
    int fps = kDefaultFps;
    int max_frames = kDefaultMaxFrames;
    int timeout_ms = kDefaultTimeoutMs;
    sima_test::parse_int_arg(argc, argv, "--w", enc_w);
    sima_test::parse_int_arg(argc, argv, "--h", enc_h);
    sima_test::parse_int_arg(argc, argv, "--fps", fps);
    sima_test::parse_int_arg(argc, argv, "--max-frames", max_frames);
    sima_test::parse_int_arg(argc, argv, "--timeout-ms", timeout_ms);
    enc_w &= ~1;
    enc_h &= ~1;
    if (enc_w <= 0 || enc_h <= 0) {
      throw std::runtime_error("Invalid dimensions");
    }
    if (fps <= 0) {
      throw std::runtime_error("Invalid fps");
    }
    if (max_frames <= 0) {
      throw std::runtime_error("Invalid max-frames");
    }

    const double mae_max = env_double("SIMA_DECODER_COMPARE_MAE_MAX", 2.0);
    const double psnr_min = env_double("SIMA_DECODER_COMPARE_PSNR_MIN", 38.0);
    const double max_abs_max = env_double("SIMA_DECODER_COMPARE_MAX_ABS", 8.0);
    const bool dump_on_fail = env_int("SIMA_DECODER_COMPARE_DUMP", 0) != 0;

    const int base_port_env = env_int("SIMA_RTSP_PORT", kDefaultPort);
    const int max_tries = std::max(1, env_int("SIMA_RTSP_PORT_RANGE", 32));
    const int rtp_port_offset =
        std::max(0, env_int("SIMA_RTSP_RTP_PORT_OFFSET", kDefaultRtpPortOffset));
    const int rtp_ports_per_server = std::max(2, env_int("SIMA_RTSP_RTP_PORT_COUNT", 8));
    const int rtp_port_stride =
        std::max(1, env_int("SIMA_RTSP_RTP_PORT_STRIDE", rtp_ports_per_server));
    const int chosen_port = rtsp_find_free_port_range_with_rtp(
        base_port_env, 1, 1, max_tries, rtp_port_offset, rtp_ports_per_server, rtp_port_stride);
    if (chosen_port < 0) {
      throw std::runtime_error("Failed to find free RTSP port");
    }
    if (chosen_port != base_port_env) {
      std::cerr << "[rtsp] base port " << base_port_env << " busy; using " << chosen_port << "\n";
    }
    const int port_offset = chosen_port - base_port_env;
    const int tries_left = std::max(1, max_tries - port_offset);
    const int rtp_port_base = chosen_port + rtp_port_offset;
    RtspServerContext server =
        start_rtsp_server_with_retry(image_path, enc_w, enc_h, enc_w, enc_h, fps, chosen_port,
                                     rtp_port_base, rtp_ports_per_server, tries_left);
    RtspHandleGuard guard(&server.handle);

    const std::string url = server.handle.url();
    auto frames_neat =
        decode_rtsp_frames(url, os_decoder, /*use_neatdecoder=*/true, max_frames, timeout_ms);
    auto frames_os =
        decode_rtsp_frames(url, os_decoder, /*use_neatdecoder=*/false, max_frames, timeout_ms);

    double worst_mae = 0.0;
    double worst_psnr = 1e9;
    double worst_max_abs = 0.0;

    for (int i = 0; i < max_frames; ++i) {
      const Nv12Frame& a = frames_neat[i];
      const Nv12Frame& b = frames_os[i];
      if (a.width != b.width || a.height != b.height) {
        throw std::runtime_error("Frame size mismatch at index " + std::to_string(i));
      }

      Metrics y = compare_nv12_luma(a, b);
      const std::size_t total_bytes = a.nv12.size();
      Metrics full = compare_bytes(a.nv12.data(), b.nv12.data(), total_bytes);

      worst_mae = std::max(worst_mae, y.mae);
      worst_psnr = std::min(worst_psnr, y.psnr);
      worst_max_abs = std::max(worst_max_abs, y.max_abs);

      if (y.mae > mae_max || y.psnr < psnr_min || y.max_abs > max_abs_max) {
        if (dump_on_fail) {
          dump_nv12(a, "tmp/neat_frame_" + std::to_string(i) + ".nv12");
          dump_nv12(b, "tmp/os_frame_" + std::to_string(i) + ".nv12");
        }
        std::ostringstream oss;
        oss << "Quality mismatch at frame " << i << " (Y: MAE=" << y.mae << " PSNR=" << y.psnr
            << " MaxAbs=" << y.max_abs << ") (Full: MAE=" << full.mae << " PSNR=" << full.psnr
            << " MaxAbs=" << full.max_abs << ")";
        throw std::runtime_error(oss.str());
      }
    }

    std::cout << "[OK] decoder_quality_compare_test passed (" << max_frames << " frames)"
              << " worst_y_mae=" << worst_mae << " worst_y_psnr=" << worst_psnr
              << " worst_y_max_abs=" << worst_max_abs << " os_decoder=" << os_decoder << "\n";
    return 0;

  } catch (const std::runtime_error& e) {
    return fail_test(e.what());
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

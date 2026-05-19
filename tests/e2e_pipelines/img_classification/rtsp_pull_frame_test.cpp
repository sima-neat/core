#include "nodes/common/Output.h"
#include "nodes/groups/RtspDecodedInput.h"
#include "nodes/io/StillImageInput.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/H264Packetize.h"
#include "pipeline/Session.h"
#include "rtsp_port_utils.h"

#include "cli_utils.h"
#include "gst/GstHelpers.h"
#include "test_utils.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static int env_int(const char* key, int fallback) {
  const char* v = std::getenv(key);
  if (!v || !*v)
    return fallback;
  return std::atoi(v);
}

static std::string upper_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

static std::string tensor_format(const simaai::neat::Tensor& t) {
  if (!t.semantic.image.has_value())
    return {};
  switch (t.semantic.image->format) {
  case simaai::neat::ImageSpec::PixelFormat::RGB:
    return "RGB";
  case simaai::neat::ImageSpec::PixelFormat::BGR:
    return "BGR";
  case simaai::neat::ImageSpec::PixelFormat::GRAY8:
    return "GRAY8";
  case simaai::neat::ImageSpec::PixelFormat::NV12:
    return "NV12";
  case simaai::neat::ImageSpec::PixelFormat::I420:
    return "I420";
  case simaai::neat::ImageSpec::PixelFormat::UNKNOWN:
    return {};
  }
  return {};
}

static size_t plane_bytes(const simaai::neat::Plane& p, simaai::neat::TensorDType dtype) {
  size_t elem = 1;
  switch (dtype) {
  case simaai::neat::TensorDType::UInt8:
    elem = 1;
    break;
  case simaai::neat::TensorDType::Int8:
    elem = 1;
    break;
  case simaai::neat::TensorDType::UInt16:
    elem = 2;
    break;
  case simaai::neat::TensorDType::Int16:
    elem = 2;
    break;
  case simaai::neat::TensorDType::Int32:
    elem = 4;
    break;
  case simaai::neat::TensorDType::BFloat16:
    elem = 2;
    break;
  case simaai::neat::TensorDType::Float32:
    elem = 4;
    break;
  case simaai::neat::TensorDType::Float64:
    elem = 8;
    break;
  }
  if (p.shape.size() >= 2 && !p.strides_bytes.empty()) {
    return static_cast<size_t>(p.strides_bytes[0]) * static_cast<size_t>(p.shape[0]);
  }
  if (p.shape.size() >= 2) {
    return static_cast<size_t>(p.shape[0]) * static_cast<size_t>(p.shape[1]) * elem;
  }
  if (p.shape.size() == 1) {
    return static_cast<size_t>(p.shape[0]) * elem;
  }
  return 0;
}

static bool format_matches(const std::string& expect, const std::string& actual) {
  if (expect == "NV12")
    return actual == "NV12";
  if (expect == "I420")
    return actual == "I420" || actual == "YUV420P" || actual == "YUV420";
  return false;
}

static std::atomic<bool> g_keep_running{true};
static void on_sigint(int) {
  g_keep_running.store(false);
}

static void wait_forever_until_ctrl_c() {
  std::signal(SIGINT, on_sigint);
  std::cerr << "[SERVER] Press Ctrl+C to stop.\n";
  while (g_keep_running.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}

int main(int argc, char** argv) {
  simaai::neat::RtspServerHandle server;
  bool server_started = false;
  try {
    require(simaai::neat::element_exists("neatencoder"),
            "Missing SIMA encoder plugin (neatencoder).");

    std::string url;
    sima_test::get_arg(argc, argv, "--rtsp", url);
    std::string image_path;
    sima_test::get_arg(argc, argv, "--image", image_path);
    const bool local = sima_test::has_flag(argc, argv, "--local");
    const bool server_only = sima_test::has_flag(argc, argv, "--server-only");
    int timeout_ms = 1000;
    sima_test::parse_int_arg(argc, argv, "--timeout-ms", timeout_ms);
    int tries = 20;
    sima_test::parse_int_arg(argc, argv, "--tries", tries);
    int h264_fps = -1;
    sima_test::parse_int_arg(argc, argv, "--h264-fps", h264_fps);
    int h264_width = -1;
    sima_test::parse_int_arg(argc, argv, "--h264-width", h264_width);
    int h264_height = -1;
    sima_test::parse_int_arg(argc, argv, "--h264-height", h264_height);
    const bool print_pipeline = sima_test::has_flag(argc, argv, "--print-pipeline");
    const bool diagnose = sima_test::has_flag(argc, argv, "--diagnose");
    std::string expect_format;
    sima_test::get_arg(argc, argv, "--expect-format", expect_format);
    expect_format = upper_copy(expect_format);

    if (local && !url.empty()) {
      throw std::runtime_error("Use either --local or --rtsp, not both.");
    }
    if (server_only && !url.empty()) {
      throw std::runtime_error("--server-only cannot be used with --rtsp.");
    }
    if (tries <= 0) {
      throw std::runtime_error("Invalid --tries: must be > 0");
    }
    if (!expect_format.empty() && expect_format != "NV12" && expect_format != "I420") {
      throw std::runtime_error("Invalid --expect-format: " + expect_format +
                               " (expected NV12|I420)");
    }

    if (url.empty() || local) {
      if (image_path.empty()) {
        image_path = "test.jpg";
      }
      require(std::filesystem::exists(image_path),
              "Missing image for local RTSP server. Use --image <path>.");

      const int content_w = 256;
      const int content_h = 256;
      const int enc_w = 256;
      const int enc_h = 256;
      const int fps = 30;

      simaai::neat::SessionOptions sess_opt;
      simaai::neat::Session s(sess_opt);
      s.add(simaai::neat::nodes::StillImageInput(image_path, content_w, content_h, enc_w, enc_h,
                                                 fps));
      // Use software encoder to force frequent IDR (x264enc uses key-int-max=1).
      s.add(simaai::neat::nodes::H264EncodeSW(/*bitrate_kbps=*/400));
      s.add(simaai::neat::nodes::H264Parse(/*config_interval=*/1));
      s.add(simaai::neat::nodes::H264Packetize(/*pt=*/96, /*config_interval=*/1));

      const int base_port_env = env_int("SIMA_RTSP_PORT", 8554);
      const int max_tries = std::max(1, env_int("SIMA_RTSP_PORT_RANGE", 32));
      const int rtp_port_offset = std::max(0, env_int("SIMA_RTSP_RTP_PORT_OFFSET", 10000));
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
      const int rtp_port_base = chosen_port + rtp_port_offset;
      server = s.run_rtsp({
          .mount = "image",
          .port = chosen_port,
          .rtp_port_base = rtp_port_base,
          .rtp_port_count = rtp_ports_per_server,
      });
      server_started = true;
      url = server.url();
      std::cerr << "[INFO] RTSP URL: " << url << "\n";
      if (print_pipeline) {
        std::cerr << "[INFO] Server pipeline: " << s.last_pipeline() << "\n";
      }

      if (server_only) {
        wait_forever_until_ctrl_c();
        server.stop();
        return 0;
      }
    } else if (server_only) {
      throw std::runtime_error("--server-only requires --local or no --rtsp.");
    }

    simaai::neat::SessionOptions sess_opt;
    simaai::neat::Session p(sess_opt);
    simaai::neat::nodes::groups::RtspDecodedInputOptions ropt;
    ropt.url = url;
    ropt.latency_ms = 200;
    ropt.tcp = true;
    ropt.h264_parse_config_interval = 1;
    if (h264_fps > 0)
      ropt.h264_fps = h264_fps;
    if (h264_width > 0)
      ropt.h264_width = h264_width;
    if (h264_height > 0)
      ropt.h264_height = h264_height;
    p.add(simaai::neat::nodes::groups::RtspDecodedInput(ropt));
    p.add(simaai::neat::nodes::Output());

    if (print_pipeline) {
      std::cout << "[rtsp] pipeline:\n" << p.describe_backend() << "\n";
    }

    simaai::neat::RunOptions run_opt;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;
    simaai::neat::Run runner = p.build(run_opt);

    if (!runner.can_pull()) {
      throw std::runtime_error("rtsp: pipeline cannot pull (missing Output).");
    }
    if (runner.can_push()) {
      throw std::runtime_error("rtsp: pipeline expects input (AppSrc present). This test only "
                               "supports RTSP source pipelines.");
    }

    simaai::neat::Sample out;
    simaai::neat::PullError err;
    simaai::neat::PullStatus status = simaai::neat::PullStatus::Timeout;
    for (int i = 0; i < tries; ++i) {
      status = runner.pull(timeout_ms, out, &err);
      if (status == simaai::neat::PullStatus::Timeout) {
        continue;
      }
      break;
    }

    if (status == simaai::neat::PullStatus::Timeout) {
      throw std::runtime_error(
          "rtsp: no output received (timeout). Check RTSP server and try "
          "higher --timeout-ms/--tries. Use --print-pipeline/--diagnose for details.");
    }
    if (status == simaai::neat::PullStatus::Closed) {
      throw std::runtime_error("rtsp: pipeline closed before any output (EOS/teardown). Check RTSP "
                               "server or credentials.");
    }
    if (status == simaai::neat::PullStatus::Error) {
      std::string msg = err.message.empty() ? "rtsp: pull failed" : err.message;
      if (!err.code.empty()) {
        msg += " (code=" + err.code + ")";
      }
      if (diagnose && err.report.has_value()) {
        msg += "\nreport_json=" + err.report->to_json();
      } else if (diagnose) {
        msg += "\nreport=" + runner.report();
      }
      throw std::runtime_error(msg);
    }

    const auto tensors = simaai::neat::tensors_from_sample(out, true);
    if (tensors.size() != 1U) {
      throw std::runtime_error("rtsp: expected exactly one tensor output, got " +
                               std::to_string(tensors.size()));
    }
    const simaai::neat::Tensor& neat = tensors.front();

    std::string actual_format = tensor_format(neat);
    if (actual_format.empty()) {
      actual_format = upper_copy(out.payload_tag);
    }

    if (!expect_format.empty() && !format_matches(expect_format, actual_format)) {
      std::string extra;
      if (!out.caps_string.empty()) {
        extra += " caps=\"" + out.caps_string + "\"";
      }
      if (!out.media_type.empty()) {
        extra += " media=\"" + out.media_type + "\"";
      }
      throw std::runtime_error("rtsp: unexpected format: got " + actual_format + " expected " +
                               expect_format + extra);
    }

    const int64_t h = neat.shape.size() > 0 ? neat.shape[0] : 0;
    const int64_t w = neat.shape.size() > 1 ? neat.shape[1] : 0;
    size_t total = 0;
    for (const auto& p : neat.planes)
      total += plane_bytes(p, neat.dtype);
    const size_t bytes = neat.storage ? neat.storage->size_bytes : total;
    std::cout << "[rtsp] neat format=" << actual_format << " w=" << w << " h=" << h
              << " planes=" << neat.planes.size() << " bytes=" << bytes;
    if (!out.payload_tag.empty()) {
      std::cout << " tag=" << out.payload_tag;
    }
    if (!out.caps_string.empty()) {
      std::cout << " caps=\"" << out.caps_string << "\"";
    }
    std::cout << "\n";

    runner.close();
    if (server_started)
      server.stop();
    return 0;
  } catch (const std::exception& e) {
    if (server_started)
      server.stop();
    std::cerr << "Error: " << e.what() << "\n";
    return 5;
  }
}

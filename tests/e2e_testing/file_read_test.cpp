#include "pipeline/Graph.h"
#include "gst/GstInit.h"
#include "gst/GstHelpers.h"
#include "nodes/sima/SimaDecode.h"

#include "cli_utils.h"
#include "test_utils.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

using namespace simaai::neat::nodes;

static bool frt_env_flag(const char* name) {
  const char* v = std::getenv(name);
  if (!v || !*v)
    return false;
  return std::string(v) != "0";
}

static int frt_env_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v)
    return fallback;
  try {
    return std::stoi(v);
  } catch (...) {
    return fallback;
  }
}

static int64_t frt_now_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

static void frt_dbg_log(const std::string& msg) {
  if (!frt_env_flag("SIMA_FILE_READ_TEST_DEBUG"))
    return;
  std::cerr << "[DBG " << frt_now_ms() << "] " << msg << "\n";
}

static void frt_start_watchdog(std::atomic<bool>& done, const std::string& label) {
  if (!frt_env_flag("SIMA_FILE_READ_TEST_DEBUG"))
    return;
  std::thread([&done, label]() {
    const int interval_s = frt_env_int("SIMA_FILE_READ_TEST_WATCHDOG_S", 5);
    int tick = 0;
    while (!done.load()) {
      std::this_thread::sleep_for(std::chrono::seconds(interval_s));
      if (done.load())
        break;
      std::cerr << "[DBG " << frt_now_ms() << "] watchdog(" << label << ") tick=" << tick++ << "\n";
    }
  }).detach();
}

static std::vector<uint8_t> copy_nv12_from_tensor(const simaai::neat::Tensor& t, int& out_w,
                                                  int& out_h) {
  if (!t.is_nv12()) {
    throw std::runtime_error("Expected NV12 simaai::neat::Tensor output");
  }
  out_h = t.height();
  out_w = t.width();
  if (out_w <= 0 || out_h <= 0) {
    throw std::runtime_error("NV12 tensor invalid dimensions");
  }
  return t.copy_nv12_contiguous();
}

static cv::Mat nv12_to_bgr(const std::vector<uint8_t>& nv12, int w, int h) {
  const size_t expected = static_cast<size_t>(w) * static_cast<size_t>(h) * 3 / 2;
  if (nv12.size() < expected) {
    throw std::runtime_error("NV12 buffer too small: got " + std::to_string(nv12.size()) +
                             " expected >= " + std::to_string(expected));
  }

  cv::Mat yuv(h + h / 2, w, CV_8UC1, const_cast<uint8_t*>(nv12.data()));
  cv::Mat bgr;
  cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);
  return bgr;
}

static void dump_nv12_raw(const std::vector<uint8_t>& nv12, const std::string& path_nv12) {
  std::ofstream ofs(path_nv12, std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(nv12.data()), static_cast<std::streamsize>(nv12.size()));
  std::cerr << "[DUMP] " << path_nv12 << " (" << nv12.size() << " bytes)\n";
}

static void save_bgr(const cv::Mat& bgr, const std::string& path) {
  if (!cv::imwrite(path, bgr)) {
    throw std::runtime_error("OpenCV imwrite failed: " + path);
  }
  std::cerr << "[SAVE] " << path << "\n";
}

struct Metrics {
  double mae = 0.0;
  double mse = 0.0;
  double psnr = 0.0;
  double max_abs = 0.0;
};

static Metrics compare_bgr(const cv::Mat& a, const cv::Mat& b) {
  if (a.empty() || b.empty())
    throw std::runtime_error("compare_bgr: empty image");
  if (a.size() != b.size() || a.type() != b.type())
    throw std::runtime_error("compare_bgr: size/type mismatch");

  cv::Mat diff;
  cv::absdiff(a, b, diff);

  cv::Scalar mean_abs = cv::mean(diff);
  const double mae = (mean_abs[0] + mean_abs[1] + mean_abs[2]) / 3.0;

  cv::Mat diff_f;
  diff.convertTo(diff_f, CV_32F);
  diff_f = diff_f.mul(diff_f);
  cv::Scalar mean_sq = cv::mean(diff_f);
  const double mse = (mean_sq[0] + mean_sq[1] + mean_sq[2]) / 3.0;

  const double psnr = (mse <= 1e-12) ? 1e9 : (10.0 * std::log10((255.0 * 255.0) / mse));

  cv::Mat diff_gray;
  cv::cvtColor(diff, diff_gray, cv::COLOR_BGR2GRAY);
  double minv = 0.0, maxv = 0.0;
  cv::minMaxLoc(diff_gray, &minv, &maxv);

  return {mae, mse, psnr, maxv};
}

static int run_image_once(const std::string& image_path, int out_w_in, int out_h_in, double mae_thr,
                          double psnr_thr) {
  frt_dbg_log("run_image_once: enter");
  std::atomic<bool> done{false};
  frt_start_watchdog(done, "run_image_once");
  // Deterministic defaults for CI
  int out_w = (out_w_in > 0) ? out_w_in : 224;
  int out_h = (out_h_in > 0) ? out_h_in : 224;

  // NV12 requires even dims
  out_w &= ~1;
  out_h &= ~1;

  const int fps = 30;
  const int bitrate_kbps = 400;

  simaai::neat::Graph p;
  frt_dbg_log("pipeline created");

  // JPEG -> raw video frames
  p.add(FileInput(image_path));
  p.add(JpegDecode());
  frt_dbg_log("added FileInput + JpegDecode");

  // CRITICAL: turn single image into a timestamped stream
  // Keep it short for ctest, but long enough for encode/decode to start.
  p.custom("imagefreeze num-buffers=45"); // ~1.5s @ 30fps
  p.custom("videorate");
  frt_dbg_log("added imagefreeze + videorate");

  // Normalize format/size/rate for SIMA encoder contract
  p.add(VideoConvert());
  p.add(VideoScale());
  p.add(CapsNV12SysMem(out_w, out_h, fps));
  p.add(Queue());
  frt_dbg_log("added VideoConvert + VideoScale + CapsNV12SysMem + Queue");

  // Encode -> Parse -> Decode (SIMA)
  p.add(H264EncodeSima(out_w, out_h, fps, bitrate_kbps, "baseline", "4.0"));
  frt_dbg_log("added H264EncodeSima");
  p.add(H264Parse(/*config_interval=*/1));
  frt_dbg_log("added H264Parse");
  simaai::neat::SimaDecodeOptions image_dec;
  image_dec.type = simaai::neat::SimaDecodeType::H264;
  image_dec.sima_allocator_type = 2;
  image_dec.out_format = simaai::neat::FormatTag::NV12;
  image_dec.raw_output = false;
  p.add(SimaDecode(image_dec));
  frt_dbg_log("added SimaDecode");

  // Make sure appsink pull isn't blocked by upstream scheduling
  p.add(Queue());
  p.add(Output());
  frt_dbg_log("added Output");

  bool got = false;
  simaai::neat::Tensor captured{};
  p.set_tensor_callback([&](const simaai::neat::Tensor& t) {
    captured = t;
    got = true;
    return false; // first frame is fine for this test
  });
  frt_dbg_log("callback set; starting pipeline run");
  const int alarm_s = frt_env_int("SIMA_FILE_READ_TEST_ALARM_S", 0);
  if (alarm_s > 0) {
    frt_dbg_log("arming alarm: " + std::to_string(alarm_s) + "s");
    alarm(static_cast<unsigned int>(alarm_s));
  }
  p.run();
  frt_dbg_log("pipeline run complete");
  done.store(true);

  if (!got) {
    std::cerr << "[ERR] No frame received from image pipeline.\n";
    std::cerr << "[DBG] Pipeline: " << p.last_pipeline() << "\n";
    return 1;
  }

  out_w = 0;
  out_h = 0;
  std::vector<uint8_t> nv12 = copy_nv12_from_tensor(captured, out_w, out_h);

  // Decode output -> BGR
  cv::Mat dec_bgr = nv12_to_bgr(nv12, out_w, out_h);
  dump_nv12_raw(nv12, "decoded_image.nv12");
  save_bgr(dec_bgr, "decoded_image_full.jpg");

  // Reference via OpenCV decode + resize to decoded size
  cv::Mat ref = cv::imread(image_path, cv::IMREAD_COLOR);
  if (ref.empty())
    throw std::runtime_error("OpenCV imread failed: " + image_path);

  const int tgt_w = out_w;
  const int tgt_h = out_h;
  int interp = (tgt_w < ref.cols || tgt_h < ref.rows) ? cv::INTER_AREA : cv::INTER_LINEAR;

  cv::Mat ref_rs;
  cv::resize(ref, ref_rs, cv::Size(tgt_w, tgt_h), 0, 0, interp);

  Metrics m = compare_bgr(dec_bgr, ref_rs);

  std::cerr << "[METRICS] MAE=" << m.mae << "  PSNR=" << m.psnr << " dB  MaxAbs=" << m.max_abs
            << "\n";

  if (m.mae > mae_thr || m.psnr < psnr_thr) {
    save_bgr(ref_rs, "reference_resized.jpg");
    std::cerr << "[FAIL] Image mismatch: requires MAE <= " << mae_thr << " and PSNR >= " << psnr_thr
              << " dB\n";
    return 3;
  }

  std::cout << "[OK] file_read_test passed.\n";
  return 0;
}

static int run_video_frames(const std::string& video_path, int nframes) {
  frt_dbg_log("run_video_frames: enter");
  std::atomic<bool> done{false};
  frt_start_watchdog(done, "run_video_frames");
  simaai::neat::Graph p;

  p.add(FileInput(video_path));
  p.add(VideoTrackSelect(0));
  p.add(Queue());
  p.add(H264ParseAu());
  simaai::neat::SimaDecodeOptions video_dec;
  video_dec.type = simaai::neat::SimaDecodeType::H264;
  video_dec.sima_allocator_type = 2;
  video_dec.out_format = simaai::neat::FormatTag::NV12;
  video_dec.raw_output = false;
  p.add(SimaDecode(video_dec));
  p.add(Queue());
  p.add(Output());
  frt_dbg_log("video pipeline built");

  int got = 0;
  std::vector<simaai::neat::Tensor> frames;
  p.set_tensor_callback([&](const simaai::neat::Tensor& t) {
    frames.push_back(t);
    ++got;
    return got < nframes;
  });
  frt_dbg_log("callback set; starting pipeline run");
  const int alarm_s = frt_env_int("SIMA_FILE_READ_TEST_ALARM_S", 0);
  if (alarm_s > 0) {
    frt_dbg_log("arming alarm: " + std::to_string(alarm_s) + "s");
    alarm(static_cast<unsigned int>(alarm_s));
  }
  p.run();
  frt_dbg_log("pipeline run complete");
  done.store(true);
  for (int i = 0; i < static_cast<int>(frames.size()); ++i) {
    int out_w = 0;
    int out_h = 0;
    std::vector<uint8_t> nv12 = copy_nv12_from_tensor(frames[i], out_w, out_h);
    cv::Mat bgr = nv12_to_bgr(nv12, out_w, out_h);
    char path[256];
    std::snprintf(path, sizeof(path), "frame_%03d.jpg", i);
    save_bgr(bgr, path);
  }
  std::cerr << "[DONE] Saved " << got << " frames.\n";
  return (got > 0) ? 0 : 2;
}

int main(int argc, char** argv) {
  try {
    frt_dbg_log("file_read_test: enter");
    if (frt_env_flag("SIMA_FILE_READ_TEST_DEBUG")) {
      std::cerr << "[DBG " << frt_now_ms() << "] argv:";
      for (int i = 0; i < argc; ++i) {
        std::cerr << " " << argv[i];
      }
      std::cerr << "\n";
    }
    if (!std::getenv("SIMA_DEC_IPC_PROTOCOL")) {
      // Force legacy IPC for older decoder daemon; allow override via env.
      setenv("SIMA_DEC_IPC_PROTOCOL", "legacy", 0);
    }
    setenv("SIMA_ALLOW_GST_INIT", "1", 1);
    simaai::neat::gst_init_once();

    require(simaai::neat::element_exists("neatencoder") &&
                simaai::neat::element_exists("neatdecoder"),
            "Missing SIMA encoder/decoder plugins (neatencoder/neatdecoder).");

    if (argc < 3) {
      std::cerr << "Usage:\n"
                << "  " << argv[0] << " --image /path/to.jpg [--w N --h N] [--mae X] [--psnr Y]\n"
                << "  " << argv[0] << " --video /path/to.mp4 [--nframes N]\n";
      return 2;
    }

    std::string mode = argv[1];
    std::string path = argv[2];

    int w = -1;
    int h = -1;
    int nframes = 30;
    sima_test::parse_int_arg(argc, argv, "--w", w);
    sima_test::parse_int_arg(argc, argv, "--h", h);
    sima_test::parse_int_arg(argc, argv, "--nframes", nframes);

    double mae_thr = 18.0;
    double psnr_thr = 25.0;
    sima_test::parse_double_arg(argc, argv, "--mae", mae_thr);
    sima_test::parse_double_arg(argc, argv, "--psnr", psnr_thr);

    require(file_exists(path), "Input path missing: " + path);

    if (mode == "--image") {
      return run_image_once(path, w, h, mae_thr, psnr_thr);
    } else if (mode == "--video") {
      return run_video_frames(path, nframes);
    } else {
      std::cerr << "[ERR] Unknown mode: " << mode << "\n";
      return 2;
    }

  } catch (const std::exception& e) {
    std::cerr << "[FATAL] " << e.what() << "\n";
    return 1;
  }
}

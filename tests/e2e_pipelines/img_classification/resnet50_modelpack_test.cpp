#include "pipeline/Session.h"
#include "model/Model.h"
#include "nodes/groups/ImageInputGroup.h"
#include "nodes/groups/RtspDecodedInput.h"
#include "nodes/io/StillImageInput.h"
#include "nodes/io/Input.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/H264Packetize.h"

#include "asset_utils.h"
#include "cli_utils.h"
#include "rtsp_port_utils.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static simaai::neat::Model model_from_format(const std::string& tar_gz, int width, int height,
                                             const std::string& format,
                                             const std::string& upstream_name = "decoder") {
  int depth = 0;
  if (format == "GRAY")
    depth = 1;
  if (format == "RGB" || format == "BGR")
    depth = 3;
  simaai::neat::Model::Options opt;
  opt.media_type = "video/x-raw";
  opt.format = format;
  opt.input_max_width = width;
  opt.input_max_height = height;
  opt.input_max_depth = depth;
  opt.preproc.normalize = true;
  opt.upstream_name = upstream_name;
  return simaai::neat::Model(tar_gz, opt);
}

static void usage(const char* prog) {
  std::cerr
      << "Usage:\n"
      << "  " << prog << " <image> [model.tar.gz]\n"
      << "  " << prog << " --image <path> [--model <model.tar.gz>]\n"
      << "  " << prog << " --goldfish [--model <model.tar.gz>]\n"
      << "\n"
      << "Options:\n"
      << "  --goldfish             Download ImageNet goldfish sample and run accuracy check.\n"
      << "  --goldfish-url <url>   Override goldfish image URL.\n"
      << "  --expect-id <int>      Expected top-1 class id (0-based).\n"
      << "  --min-prob <float>     Minimum softmax probability for top-1 when checking accuracy.\n"
      << "  --image <path>         Input image path.\n"
      << "  --model <path>         Model pack tar.gz path.\n"
      << "  --only-direct          Skip JPEG/RTSP paths; run direct model only.\n"
      << "  --only-rtsp            Skip direct and JPEG paths; run RTSP only.\n"
      << "  --only-jpeg            Skip direct and RTSP paths; run JPEG only.\n"
      << "  --skip-direct          Skip the direct Input path.\n"
      << "  --skip-jpeg            Skip the JPEG ImageInputGroup path.\n";
}

static std::vector<float> tensor_to_floats(const simaai::neat::Tensor& t) {
  if (t.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error("Expected Float32 tensor output");
  }
  if (!t.is_dense()) {
    throw std::runtime_error("Expected dense tensor output");
  }

  simaai::neat::Tensor cpu = t.clone();
  simaai::neat::Mapping map = cpu.map(simaai::neat::MapMode::Read);
  if (!map.data || map.size_bytes == 0) {
    throw std::runtime_error("Tensor output is empty");
  }
  if (map.size_bytes % sizeof(float) != 0) {
    throw std::runtime_error("Tensor size is not a multiple of float");
  }

  const size_t elems = map.size_bytes / sizeof(float);
  std::vector<float> out(elems);
  std::memcpy(out.data(), map.data, elems * sizeof(float));
  return out;
}

struct ScoredIndex {
  int index = -1;
  float value = 0.0f;
  float prob = 0.0f;
};

static bool env_bool(const char* name, bool def) {
  const char* val = std::getenv(name);
  if (!val || !*val)
    return def;
  std::string v(val);
  for (char& c : v) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (v == "0" || v == "false" || v == "no" || v == "off")
    return false;
  if (v == "1" || v == "true" || v == "yes" || v == "on")
    return true;
  return def;
}

static int env_int(const char* name, int def) {
  const char* val = std::getenv(name);
  if (!val || !*val)
    return def;
  char* end = nullptr;
  long out = std::strtol(val, &end, 10);
  if (end == val)
    return def;
  if (out > std::numeric_limits<int>::max())
    return std::numeric_limits<int>::max();
  if (out < std::numeric_limits<int>::min())
    return std::numeric_limits<int>::min();
  return static_cast<int>(out);
}

static std::vector<ScoredIndex> topk_with_softmax(const std::vector<float>& v, int k) {
  if (v.empty() || k <= 0)
    return {};
  const int n = static_cast<int>(v.size());
  k = std::min(k, n);

  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&v](int a, int b) { return v[a] > v[b]; });

  const float maxv = *std::max_element(v.begin(), v.end());
  double sum = 0.0;
  for (float x : v) {
    sum += std::exp(static_cast<double>(x - maxv));
  }

  std::vector<ScoredIndex> out;
  out.reserve(k);
  for (int i = 0; i < k; ++i) {
    const int id = idx[i];
    const double prob = std::exp(static_cast<double>(v[id] - maxv)) / sum;
    out.push_back(ScoredIndex{id, v[id], static_cast<float>(prob)});
  }
  return out;
}

constexpr int kInferWidth = 224;
constexpr int kInferHeight = 224;
constexpr int kInferFps = 30;
constexpr int kJpegInputWidth = 256;
constexpr int kJpegInputHeight = 256;
constexpr int kRtspInputWidth = 256;
constexpr int kRtspInputHeight = 256;
constexpr int kRtspPort = 8557;
constexpr int kRtspRtpPortOffset = 10000;
constexpr const char* kJpegDecoderName = "decoder_jpeg";
constexpr const char* kRtspDecoderName = "decoder_rtsp";

static cv::Mat load_rgb_resized(const std::string& image_path, int w, int h) {
  cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("Failed to read image: " + image_path);
  }

  if (w > 0 && h > 0 && (bgr.cols != w || bgr.rows != h)) {
    cv::resize(bgr, bgr, cv::Size(w, h), 0, 0, cv::INTER_AREA);
  }

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

static simaai::neat::Tensor require_tensor(const simaai::neat::Sample& out,
                                           const std::string& label) {
  if (out.kind != simaai::neat::SampleKind::Tensor || !out.tensor.has_value()) {
    throw std::runtime_error(label + ": expected tensor output");
  }
  return *out.tensor;
}

const char* device_type_name(simaai::neat::DeviceType type) {
  switch (type) {
  case simaai::neat::DeviceType::CPU:
    return "CPU";
  case simaai::neat::DeviceType::SIMA_APU:
    return "APU";
  case simaai::neat::DeviceType::SIMA_CVU:
    return "CVU";
  case simaai::neat::DeviceType::SIMA_MLA:
    return "MLA";
  case simaai::neat::DeviceType::UNKNOWN:
  default:
    return "UNKNOWN";
  }
}

const char* storage_kind_name(simaai::neat::StorageKind kind) {
  switch (kind) {
  case simaai::neat::StorageKind::CpuOwned:
    return "CpuOwned";
  case simaai::neat::StorageKind::CpuExternal:
    return "CpuExternal";
  case simaai::neat::StorageKind::GstSample:
    return "GstSample";
  case simaai::neat::StorageKind::DeviceHandle:
    return "DeviceHandle";
  case simaai::neat::StorageKind::Unknown:
  default:
    return "Unknown";
  }
}

std::string segments_debug(const std::vector<simaai::neat::Segment>& segs) {
  if (segs.empty())
    return {};
  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < segs.size(); ++i) {
    if (i)
      ss << ",";
    ss << segs[i].name << ":" << segs[i].size_bytes;
  }
  ss << "]";
  return ss.str();
}

void log_tensor_sample(const simaai::neat::Sample& s, const std::string& label) {
  std::ostringstream ss;
  ss << "[DBG] " << label;
  if (!s.port_name.empty())
    ss << " port=" << s.port_name;
  if (s.output_index >= 0)
    ss << " output=" << s.output_index;
  if (!s.media_type.empty())
    ss << " media_type=" << s.media_type;
  if (!s.caps_string.empty())
    ss << " caps=" << s.caps_string;
  if (!s.payload_tag.empty())
    ss << " payload_tag=" << s.payload_tag;
  if (!s.format.empty())
    ss << " format=" << s.format;
  if (!s.tensor.has_value()) {
    ss << " neat=<missing>";
    std::cerr << ss.str() << "\n";
    return;
  }
  const auto& t = s.tensor.value();
  ss << " " << t.debug_string();
  if (t.storage) {
    ss << " storage_kind=" << storage_kind_name(t.storage->kind)
       << " storage_size=" << t.storage->size_bytes
       << " storage_device=" << device_type_name(t.storage->device.type) << ":"
       << t.storage->device.id;
    if (t.storage->sima_mem_target_flags || t.storage->sima_mem_flags) {
      ss << " sima_mem_target_flags=0x" << std::hex << t.storage->sima_mem_target_flags
         << " sima_mem_flags=0x" << t.storage->sima_mem_flags << std::dec;
    }
    const std::string segs = segments_debug(t.storage->sima_segments);
    if (!segs.empty())
      ss << " sima_segments=" << segs;
  } else {
    ss << " storage=<none>";
  }
  std::cerr << ss.str() << "\n";
}

void log_sample_caps(const simaai::neat::Sample& s, const std::string& label) {
  if (s.kind != simaai::neat::SampleKind::Bundle) {
    log_tensor_sample(s, label);
    return;
  }
  std::ostringstream ss;
  ss << "[DBG] " << label << " bundle fields=" << s.fields.size();
  if (!s.media_type.empty())
    ss << " media_type=" << s.media_type;
  if (!s.caps_string.empty())
    ss << " caps=" << s.caps_string;
  std::cerr << ss.str() << "\n";
  for (size_t i = 0; i < s.fields.size(); ++i) {
    log_tensor_sample(s.fields[i], label + ".field" + std::to_string(i));
  }
}

static std::vector<float> scores_from_tensor(const simaai::neat::Tensor& t,
                                             const std::string& label) {
  auto scores_full = tensor_to_floats(t);
  if (scores_full.empty()) {
    throw std::runtime_error(label + ": empty tensor output");
  }
  if (scores_full.size() < 1000) {
    throw std::runtime_error(label + ": expected at least 1000 scores, got " +
                             std::to_string(scores_full.size()));
  }
  if (scores_full.size() > 1000) {
    scores_full.resize(1000);
  }
  return scores_full;
}

static void check_top1(const std::vector<float>& scores, int expected_id, float min_prob,
                       const std::string& label) {
  const auto top = topk_with_softmax(scores, 5);
  std::cout << "[" << label << "] top1 index=" << top[0].index << " score=" << top[0].value
            << " prob=" << top[0].prob << "\n";
  std::cout << "[" << label << "] top5:";
  for (const auto& t : top) {
    std::cout << " " << t.index << ":" << t.prob;
  }
  std::cout << "\n";

  if (expected_id < 0) {
    return;
  }

  if (top[0].index != expected_id) {
    throw std::runtime_error(label + ": top-1 mismatch: expected " + std::to_string(expected_id) +
                             " got " + std::to_string(top[0].index));
  }
  if (min_prob > 0.0f && top[0].prob < min_prob) {
    throw std::runtime_error(label + ": top-1 probability too low: " + std::to_string(top[0].prob) +
                             " < " + std::to_string(min_prob));
  }
  std::cout << "[" << label << "] top-1 matches expected class " << expected_id << "\n";
}

static simaai::neat::Sample pull_sample_with_retry(simaai::neat::Run& runner,
                                                   const std::string& label, int per_try_ms,
                                                   int tries, bool allow_push = false) {
  if (!runner.can_pull()) {
    throw std::runtime_error(label + ": pipeline cannot pull (missing Output)");
  }
  if (runner.can_push() && !allow_push) {
    throw std::runtime_error(label + ": pipeline expects input (AppSrc present)");
  }

  auto dump_on_fail = [&](const char* reason) {
    if (!std::getenv("SIMA_DUMP_PIPELINE_ON_FAIL"))
      return;
    std::cerr << "[DBG] " << label << " " << reason << "\n";
    const std::string last_err = runner.last_error();
    if (!last_err.empty()) {
      std::cerr << "[DBG] " << label << " last_error: " << last_err << "\n";
    }
    std::cerr << runner.diagnostics_summary() << "\n";
    simaai::neat::RunReportOptions opt;
    opt.include_node_reports = true;
    opt.include_next_cpu = true;
    std::cerr << runner.report(opt) << "\n";
  };

  simaai::neat::Sample out;
  simaai::neat::PullError err;
  simaai::neat::PullStatus status = simaai::neat::PullStatus::Timeout;
  for (int i = 0; i < tries; ++i) {
    status = runner.pull(per_try_ms, out, &err);
    if (status == simaai::neat::PullStatus::Timeout) {
      continue;
    }
    break;
  }

  if (status == simaai::neat::PullStatus::Timeout) {
    dump_on_fail("timeout");
    throw std::runtime_error(label + ": no output received (timeout/EOS)");
  }
  if (status == simaai::neat::PullStatus::Closed) {
    dump_on_fail("closed");
    throw std::runtime_error(label + ": pipeline closed before any output");
  }
  if (status == simaai::neat::PullStatus::Error) {
    dump_on_fail("error");
    std::string msg = err.message.empty() ? (label + ": pull failed") : err.message;
    if (!err.code.empty()) {
      msg += " (code=" + err.code + ")";
    }
    throw std::runtime_error(msg);
  }
  return out;
}

static simaai::neat::Tensor run_direct_infer(const simaai::neat::Model& model, const cv::Mat& rgb) {
  simaai::neat::Session p;
  p.add(model.session());

  simaai::neat::Sample out;
  bool logged_sample = false;
  for (int i = 0; i < 20; ++i) {
    out = p.run(rgb);
    if (!logged_sample) {
      log_sample_caps(out, "direct");
      logged_sample = true;
    }
  }
  return require_tensor(out, "direct");
}

static void compare_scores(const std::vector<float>& a, const std::vector<float>& b,
                           const std::string& a_label, const std::string& b_label) {
  if (a.size() != b.size()) {
    std::cout << "[compare] size mismatch " << a.size() << " vs " << b.size() << "\n";
    return;
  }
  float max_abs = 0.0f;
  std::size_t max_idx = 0;
  double mse = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const float diff = std::fabs(a[i] - b[i]);
    if (diff > max_abs) {
      max_abs = diff;
      max_idx = i;
    }
    mse += static_cast<double>(diff) * static_cast<double>(diff);
  }
  mse /= static_cast<double>(a.size());
  std::cout << "[compare] " << a_label << " vs " << b_label << " max_abs=" << max_abs
            << " at=" << max_idx << " mse=" << mse << "\n";
}

static simaai::neat::Sample run_image_decode_to_appsink(const std::string& image_path, int w, int h,
                                                        int fps, bool print_pipeline) {
  simaai::neat::nodes::groups::ImageInputGroupOptions opt;
  opt.path = image_path;
  opt.imagefreeze_num_buffers = env_int("SIMA_JPEG_IMAGEFREEZE_BUFFERS", 120);
  if (opt.imagefreeze_num_buffers > 0) {
    std::cout << "[jpeg_decode] imagefreeze buffers=" << opt.imagefreeze_num_buffers << "\n";
  }
  opt.fps = fps;
  opt.use_videorate = true;
  opt.use_videoscale = true;
  opt.output_caps.enable = true;
  opt.output_caps.format = "NV12";
  opt.output_caps.width = w;
  opt.output_caps.height = h;
  opt.output_caps.fps = fps;
  opt.output_caps.memory = simaai::neat::CapsMemory::Any;
  opt.sima_decoder.enable = true;
  opt.sima_decoder.decoder_name = kJpegDecoderName;
  opt.sima_decoder.raw_output = true;
  opt.sima_decoder.next_element = "CVU";
  opt.sima_decoder.use_sw_encoder = env_bool("SIMA_TEST_SW_ENCODER", false);
  if (opt.sima_decoder.use_sw_encoder) {
    std::cout << "[jpeg_decode] using SW encoder\n";
  }

  simaai::neat::Session p;
  p.add(simaai::neat::nodes::groups::ImageInputGroup(opt));
  p.add(simaai::neat::nodes::Output());

  if (print_pipeline) {
    std::cout << "[jpeg_decode] pipeline:\n" << p.describe_backend() << "\n";
  }

  simaai::neat::RunOptions run_opt;
  run_opt.output_memory = simaai::neat::OutputMemory::Owned;
  simaai::neat::Run runner = p.build(run_opt);
  const int per_try_ms = env_int("SIMA_JPEG_PULL_TIMEOUT_MS", 1000);
  const int tries = env_int("SIMA_JPEG_PULL_TRIES", 30);
  std::cout << "[jpeg_decode] pull timeout=" << (per_try_ms * tries) << " ms\n";
  auto out = pull_sample_with_retry(runner, "jpeg_decode", per_try_ms, tries);
  log_sample_caps(out, "jpeg_decode");
  return out;
}

static simaai::neat::Tensor run_image_group_infer(const simaai::neat::Model& model,
                                                  const std::string& image_path, int w, int h,
                                                  int fps, bool print_pipeline) {
  auto decoded = run_image_decode_to_appsink(image_path, w, h, fps, print_pipeline);
  auto decoded_tensor = require_tensor(decoded, "jpeg_decode");
  std::cout << "[jpeg_decode] got tensor: " << decoded_tensor.debug_string() << "\n";

  simaai::neat::Session p;
  simaai::neat::Model::SessionOptions session_opt;
  session_opt.include_appsrc = true;
  session_opt.include_appsink = true;
  p.add(model.session(session_opt));

  if (print_pipeline) {
    std::cout << "[jpeg_model] pipeline:\n" << p.describe_backend() << "\n";
  }

  simaai::neat::RunOptions run_opt;
  run_opt.output_memory = simaai::neat::OutputMemory::Owned;
  simaai::neat::Run runner = p.build(decoded_tensor, simaai::neat::RunMode::Async, run_opt);
  const bool pushed = runner.push(decoded_tensor);
  require(pushed, "jpeg_model: push decoded tensor failed");
  auto out = pull_sample_with_retry(runner, "jpeg_model", 1000, 20, true);
  log_sample_caps(out, "jpeg_model");
  return require_tensor(out, "jpeg_model");
}

struct RtspServerContext {
  simaai::neat::Session session;
  simaai::neat::RtspServerHandle handle;
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
  ctx.session.add(simaai::neat::nodes::H264EncodeSima(enc_w, enc_h, fps));
  ctx.session.add(simaai::neat::nodes::H264Parse(/*config_interval=*/1));
  ctx.session.add(simaai::neat::nodes::H264Packetize(/*pt=*/96, /*config_interval=*/1));

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

static std::vector<float> run_rtsp_scores(const simaai::neat::Model& model, const std::string& url,
                                          int w, int h, int fps, bool print_pipeline,
                                          int warmup_count, int expected_id, float min_prob,
                                          int max_attempts) {
  simaai::neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = url;
  opt.latency_ms = 200;
  opt.tcp = true;
  opt.payload_type = 96;
  opt.h264_parse_config_interval = 1;
  opt.h264_fps = fps;
  opt.h264_width = w;
  opt.h264_height = h;
  opt.auto_caps_from_stream = false;
  opt.use_videoconvert = false;
  opt.use_videoscale = false;
  opt.output_caps.enable = true;
  opt.output_caps.format = "NV12";
  opt.output_caps.width = w;
  opt.output_caps.height = h;
  opt.output_caps.fps = fps;
  opt.output_caps.memory = simaai::neat::CapsMemory::Any;
  opt.decoder_name = kRtspDecoderName;
  opt.decoder_raw_output = true;
  opt.decoder_next_element = "CVU";

  simaai::neat::Session p;
  p.add(simaai::neat::nodes::groups::RtspDecodedInput(opt));
  simaai::neat::Model::SessionOptions session_opt;
  session_opt.include_appsrc = false;
  session_opt.include_appsink = false;
  p.add(model.session(session_opt));
  p.add(simaai::neat::nodes::Output());

  if (print_pipeline) {
    std::cout << "[rtsp] pipeline:\n" << p.describe_backend() << "\n";
  }

  simaai::neat::RunOptions run_opt;
  run_opt.output_memory = simaai::neat::OutputMemory::Owned;
  simaai::neat::Run runner = p.build(run_opt);
  for (int i = 0; i < warmup_count; ++i) {
    (void)pull_sample_with_retry(runner, "rtsp_warmup", 500, 6);
  }
  std::vector<float> last_scores;
  const int attempts = std::max(1, max_attempts);
  for (int i = 0; i < attempts; ++i) {
    auto out = pull_sample_with_retry(runner, "rtsp", 1000, 20);
    log_sample_caps(out, "rtsp");
    auto t = require_tensor(out, "rtsp");
    last_scores = scores_from_tensor(t, "rtsp");
    if (expected_id < 0) {
      return last_scores;
    }
    auto top = topk_with_softmax(last_scores, 1);
    if (!top.empty() && top[0].index == expected_id &&
        (min_prob <= 0.0f || top[0].prob >= min_prob)) {
      return last_scores;
    }
  }
  return last_scores;
}

int main(int argc, char** argv) {
  constexpr const char* kGoldfishUrl =
      "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
      "n01443537_goldfish.JPEG";
  constexpr int kGoldfishId = 1; // ILSVRC2012 0-based index for "goldfish"

  const bool default_no_args = (argc <= 1);
  const bool use_goldfish = sima_test::has_flag(argc, argv, "--goldfish") || default_no_args;
  const bool only_direct = sima_test::has_flag(argc, argv, "--only-direct");
  const bool only_rtsp = sima_test::has_flag(argc, argv, "--only-rtsp");
  const bool only_jpeg = sima_test::has_flag(argc, argv, "--only-jpeg");
  const bool skip_direct = sima_test::has_flag(argc, argv, "--skip-direct");
  const bool skip_jpeg = sima_test::has_flag(argc, argv, "--skip-jpeg");

  std::string image_path;
  std::string tar_gz;
  std::string goldfish_url = kGoldfishUrl;
  bool have_expected = false;
  int expected_id = -1;
  float min_prob = 0.0f;
  bool have_min_prob = false;

  std::string tmp;
  if (sima_test::get_arg(argc, argv, "--image", tmp))
    image_path = tmp;
  if (sima_test::get_arg(argc, argv, "--model", tmp))
    tar_gz = tmp;
  if (sima_test::get_arg(argc, argv, "--goldfish-url", tmp))
    goldfish_url = tmp;
  if (sima_test::parse_int_arg(argc, argv, "--expect-id", expected_id)) {
    have_expected = true;
  }
  if (sima_test::parse_float_arg(argc, argv, "--min-prob", min_prob)) {
    have_min_prob = true;
  }
  if (env_flag("SIMA_BUILD_MODE_DEBUG")) {
    std::cout << "[DBG] test RunMode Async=" << static_cast<int>(simaai::neat::RunMode::Async)
              << " Sync=" << static_cast<int>(simaai::neat::RunMode::Sync) << "\n";
  }

  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--image" || arg == "--model" || arg == "--goldfish-url" || arg == "--expect-id" ||
        arg == "--min-prob") {
      ++i;
      continue;
    }
    if (arg == "--goldfish" || arg == "--only-direct" || arg == "--only-rtsp" ||
        arg == "--only-jpeg" || arg == "--skip-direct" || arg == "--skip-jpeg") {
      continue;
    }
    if (!arg.empty() && arg[0] == '-') {
      usage(argv[0]);
      return 2;
    }
    positional.push_back(arg);
  }

  if (image_path.empty() && !use_goldfish && !positional.empty()) {
    image_path = positional[0];
  }
  if (tar_gz.empty() && positional.size() >= 2) {
    tar_gz = positional[1];
  }

  if (use_goldfish) {
    if (!have_expected) {
      expected_id = kGoldfishId;
      have_expected = true;
    }
    if (!have_min_prob) {
      min_prob = 0.2f;
    }

    const fs::path out_path = sima_test::default_goldfish_path();
    if (!sima_test::download_file(goldfish_url, out_path)) {
      std::cerr << "Failed to download goldfish image.\n";
      std::cerr << "URL was: " << goldfish_url << "\n";
      std::cerr << "Tip: supply --image <path> and --expect-id <id> instead.\n";
      return 3;
    }
    image_path = out_path.string();
    std::cout << "Using goldfish image: " << image_path << "\n";
  }

  if (image_path.empty()) {
    usage(argv[0]);
    return 2;
  }

  require(fs::exists(image_path), "Image path missing: " + image_path);

  if (tar_gz.empty()) {
    tar_gz = sima_test::resolve_resnet50_tar();
    if (tar_gz.empty()) {
      std::cerr << "Failed to resolve resnet50 tar.gz. "
                << "Set SIMA_RESNET50_TAR or run 'sima-cli modelzoo -v 2.0.0 get resnet_50'.\n";
      return 3;
    }
  }
  if (!fs::exists(tar_gz)) {
    std::cerr << "Model tar.gz missing: " << tar_gz << "\n";
    return 3;
  }

  try {
    bool run_direct = only_direct ? true : (!only_rtsp && !only_jpeg && !skip_direct);
    bool run_jpeg = only_jpeg ? true : (!only_rtsp && !only_direct && !skip_jpeg);
    bool run_rtsp = only_rtsp ? true : (!only_jpeg && !only_direct);

    const int expect = have_expected ? expected_id : -1;
    const float prob = have_expected ? min_prob : 0.0f;

    // Terminal policy validation checks.
    {
      simaai::neat::Model::Options bad_opt;
      bad_opt.media_type = "video/x-raw";
      bad_opt.format = "RGB";
      bad_opt.input_max_width = kInferWidth;
      bad_opt.input_max_height = kInferHeight;
      bad_opt.input_max_depth = 3;
      bad_opt.inference_terminal.last_stage_name = "__definitely_missing_terminal_stage__";
      bool threw = false;
      try {
        simaai::neat::Model bad_model(tar_gz, bad_opt);
        (void)bad_model.session();
      } catch (const std::exception& ex) {
        const std::string msg = ex.what();
        if (msg.find("terminal stage") != std::string::npos ||
            msg.find("infer_stages=[") != std::string::npos) {
          threw = true;
        }
      }
      require(threw, "Expected unresolved terminal policy to throw explicit error");
    }
    {
      // Precedence check: index wins over invalid name.
      simaai::neat::Model::Options idx_opt;
      idx_opt.media_type = "video/x-raw";
      idx_opt.format = "RGB";
      idx_opt.input_max_width = kInferWidth;
      idx_opt.input_max_height = kInferHeight;
      idx_opt.input_max_depth = 3;
      idx_opt.inference_terminal.last_stage_index = 0;
      idx_opt.inference_terminal.last_stage_name = "__ignored_when_index_is_set__";
      simaai::neat::Model idx_model(tar_gz, idx_opt);
      simaai::neat::Session sess;
      sess.add(idx_model.session());
      require(!sess.describe_backend().empty(),
              "Terminal index precedence model.session() returned empty gst");
    }
    {
      simaai::neat::Model::Options mla_opt;
      mla_opt.media_type = "video/x-raw";
      mla_opt.format = "RGB";
      mla_opt.input_max_width = kInferWidth;
      mla_opt.input_max_height = kInferHeight;
      mla_opt.input_max_depth = 3;
      mla_opt.inference_terminal.mla_only = true;
      simaai::neat::Model mla_model(tar_gz, mla_opt);
      simaai::neat::Session sess;
      sess.add(mla_model.session());
      require(!sess.describe_backend().empty(),
              "mla_only terminal policy produced empty session group");
    }

    if (run_direct) {
      auto model_rgb = model_from_format(tar_gz, kInferWidth, kInferHeight, "RGB");
      {
        simaai::neat::Session sess;
        sess.add(model_rgb.session());
        auto full_gst = sess.describe_backend();
        require(full_gst.find("appsrc") != std::string::npos,
                "model.session() should include appsrc by default");
        require(full_gst.find("appsink") != std::string::npos,
                "model.session() should include appsink by default");
      }
      {
        simaai::neat::Model::SessionOptions session_opt;
        session_opt.include_appsrc = false;
        session_opt.include_appsink = false;
        simaai::neat::Session sess;
        sess.add(model_rgb.session(session_opt));
        auto core_gst = sess.describe_backend();
        require(core_gst.find("appsrc") == std::string::npos,
                "model.session({false,false}) should omit appsrc");
        require(core_gst.find("appsink") == std::string::npos,
                "model.session({false,false}) should omit appsink");
      }
      cv::Mat rgb = load_rgb_resized(image_path, kInferWidth, kInferHeight);
      auto direct_t = run_direct_infer(model_rgb, rgb);
      auto direct_scores = scores_from_tensor(direct_t, "direct");
      check_top1(direct_scores, expect, prob, "direct");
    }

    if (run_jpeg) {
      auto model_nv12_jpeg =
          model_from_format(tar_gz, kJpegInputWidth, kJpegInputHeight, "NV12", kJpegDecoderName);
      auto jpeg_t = run_image_group_infer(model_nv12_jpeg, image_path, kJpegInputWidth,
                                          kJpegInputHeight, kInferFps,
                                          /*print_pipeline=*/only_jpeg);
      auto jpeg_scores = scores_from_tensor(jpeg_t, "jpeg_group");
      check_top1(jpeg_scores, expect, prob, "jpeg_group");
    }

    if (run_rtsp) {
      auto model_nv12_rtsp =
          model_from_format(tar_gz, kRtspInputWidth, kRtspInputHeight, "NV12", kRtspDecoderName);

      const int base_port_env = env_int("SIMA_RTSP_PORT_BASE", kRtspPort);
      const int max_tries = std::max(1, env_int("SIMA_RTSP_PORT_RANGE", 32));
      const int rtp_port_offset =
          std::max(0, env_int("SIMA_RTSP_RTP_PORT_OFFSET", kRtspRtpPortOffset));
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
      auto rtsp_ctx = start_rtsp_server_with_retry(
          image_path, kRtspInputWidth, kRtspInputHeight, kRtspInputWidth, kRtspInputHeight,
          kInferFps, chosen_port, rtp_port_base, rtp_ports_per_server, tries_left);
      struct RtspGuard {
        simaai::neat::RtspServerHandle* handle = nullptr;
        ~RtspGuard() {
          if (handle)
            handle->stop();
        }
      } rtsp_guard{&rtsp_ctx.handle};

      const int warmup = std::max(0, env_int("SIMA_RTSP_WARMUP", 30));
      const int attempts = std::max(1, env_int("SIMA_RTSP_ATTEMPTS", 30));
      auto rtsp_scores = run_rtsp_scores(model_nv12_rtsp, rtsp_ctx.handle.url(), kRtspInputWidth,
                                         kRtspInputHeight, kInferFps,
                                         /*print_pipeline=*/true,
                                         /*warmup_count=*/warmup,
                                         /*expected_id=*/expect,
                                         /*min_prob=*/prob,
                                         /*max_attempts=*/attempts);
      check_top1(rtsp_scores, expect, prob, "rtsp");
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 5;
  }
}

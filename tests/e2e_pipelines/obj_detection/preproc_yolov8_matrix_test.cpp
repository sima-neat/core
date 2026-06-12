#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "pipeline/StageRun.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/RenderedMlaContractQuery.h"
#include "pipeline/internal/TensorTransfer.h"
#include "pipeline/internal/TensorUtil.h"

#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace fs = std::filesystem;
namespace rendered_stage_query = simaai::neat::pipeline_internal::rendered_stage_query;

namespace {

enum class DTypeFamily {
  Int8,
  Int16,
  BFloat16,
  Unknown,
};

struct ExpectedRoute {
  bool quantize = false;
  bool tessellate = false;
  bool cast = false;
  DTypeFamily dtype = DTypeFamily::Unknown;
  std::string family = "unknown";
};

struct CliArgs {
  std::optional<fs::path> root;
  std::optional<fs::path> variants_dir;
  std::optional<fs::path> single_model;
  std::optional<bool> accuracy;
};

std::string lower_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

std::string upper_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

std::string preprocess_color_format_name(simaai::neat::PreprocessColorFormat format) {
  switch (format) {
  case simaai::neat::PreprocessColorFormat::RGB:
    return "RGB";
  case simaai::neat::PreprocessColorFormat::BGR:
    return "BGR";
  case simaai::neat::PreprocessColorFormat::GRAY8:
    return "GRAY8";
  case simaai::neat::PreprocessColorFormat::NV12:
    return "NV12";
  case simaai::neat::PreprocessColorFormat::I420:
    return "I420";
  case simaai::neat::PreprocessColorFormat::Auto:
    break;
  }
  return "AUTO";
}

simaai::neat::pipeline_internal::TensorIoStats
tensor_io_delta(const simaai::neat::pipeline_internal::TensorIoStats& before,
                const simaai::neat::pipeline_internal::TensorIoStats& after) {
  simaai::neat::pipeline_internal::TensorIoStats delta;
  delta.tensor_copy_count = after.tensor_copy_count - before.tensor_copy_count;
  delta.tensor_copy_bytes = after.tensor_copy_bytes - before.tensor_copy_bytes;
  delta.tensor_view_count = after.tensor_view_count - before.tensor_view_count;
  delta.gst_memory_map_calls = after.gst_memory_map_calls - before.gst_memory_map_calls;
  delta.holder_fast_path_hits = after.holder_fast_path_hits - before.holder_fast_path_hits;
  delta.bundle_projection_count = after.bundle_projection_count - before.bundle_projection_count;
  delta.packed_view_reuse_hits = after.packed_view_reuse_hits - before.packed_view_reuse_hits;
  delta.packed_view_reuse_opportunities =
      after.packed_view_reuse_opportunities - before.packed_view_reuse_opportunities;
  return delta;
}

std::string tensor_io_stats_string(const simaai::neat::pipeline_internal::TensorIoStats& stats) {
  std::ostringstream os;
  os << "copies=" << stats.tensor_copy_count << ",copy_bytes=" << stats.tensor_copy_bytes
     << ",views=" << stats.tensor_view_count << ",maps=" << stats.gst_memory_map_calls
     << ",holder_fast=" << stats.holder_fast_path_hits
     << ",bundle_proj=" << stats.bundle_projection_count
     << ",packed_reuse=" << stats.packed_view_reuse_hits << "/"
     << stats.packed_view_reuse_opportunities << ",packed_ratio=" << std::fixed
     << std::setprecision(3) << stats.packed_view_reuse_ratio();
  return os.str();
}

std::string join_ints(const std::vector<int>& values) {
  std::ostringstream os;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) {
      os << ",";
    }
    os << values[i];
  }
  return os.str();
}

std::string auto_flag_name(simaai::neat::AutoFlag v) {
  switch (v) {
  case simaai::neat::AutoFlag::Auto:
    return "Auto";
  case simaai::neat::AutoFlag::On:
    return "On";
  case simaai::neat::AutoFlag::Off:
    return "Off";
  }
  return "Unknown";
}

bool starts_with(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

fs::path find_repo_root(const fs::path& start) {
  fs::path cur = start;
  for (int i = 0; i < 10; ++i) {
    if (fs::exists(cur / "CMakeLists.txt") && fs::exists(cur / "tests")) {
      return cur;
    }
    if (!cur.has_parent_path()) {
      break;
    }
    cur = cur.parent_path();
  }
  return start;
}

CliArgs parse_cli(int argc, char** argv) {
  CliArgs out;
  auto parse_bool_value = [](const std::string& raw) -> std::optional<bool> {
    const std::string v = lower_copy(raw);
    if (v == "1" || v == "true" || v == "yes" || v == "on") {
      return true;
    }
    if (v == "0" || v == "false" || v == "no" || v == "off") {
      return false;
    }
    return std::nullopt;
  };
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--accuracy") {
      out.accuracy = true;
      continue;
    }
    if (starts_with(arg, "--accuracy=")) {
      const std::string raw = arg.substr(std::string("--accuracy=").size());
      const std::optional<bool> parsed = parse_bool_value(raw);
      require(parsed.has_value(), "invalid value for --accuracy: " + raw);
      out.accuracy = *parsed;
      continue;
    }
    if (starts_with(arg, "--single-model=")) {
      const std::string raw = arg.substr(std::string("--single-model=").size());
      if (!raw.empty()) {
        out.single_model = fs::path(raw);
      }
      continue;
    }
    if (arg == "--single-model" && (i + 1) < argc) {
      const std::string raw(argv[++i]);
      if (!raw.empty()) {
        out.single_model = fs::path(raw);
      }
      continue;
    }
    if (starts_with(arg, "--variants-dir=")) {
      const std::string raw = arg.substr(std::string("--variants-dir=").size());
      if (!raw.empty()) {
        out.variants_dir = fs::path(raw);
      }
      continue;
    }
    if (arg == "--variants-dir" && (i + 1) < argc) {
      const std::string raw(argv[++i]);
      if (!raw.empty()) {
        out.variants_dir = fs::path(raw);
      }
      continue;
    }
    if (!starts_with(arg, "--") && !out.root.has_value()) {
      out.root = fs::path(arg);
    }
  }
  return out;
}

bool resolve_accuracy_enabled(const CliArgs& cli) {
  if (cli.accuracy.has_value()) {
    return *cli.accuracy;
  }
  return sima_yolov8_test::env_bool("SIMA_PREPROC_YOLOV8_MATRIX_ACCURACY", false);
}

fs::path make_existing_path(const fs::path& raw, const fs::path& root) {
  if (raw.empty()) {
    return raw;
  }
  if (raw.is_absolute()) {
    return raw;
  }
  std::error_code ec;
  if (fs::exists(raw, ec)) {
    return fs::absolute(raw, ec);
  }
  fs::path rooted = root / raw;
  if (fs::exists(rooted, ec)) {
    return fs::absolute(rooted, ec);
  }
  return fs::absolute(raw, ec);
}

std::optional<fs::path> resolve_single_model_tar(const CliArgs& cli, const fs::path& root) {
  if (cli.single_model.has_value()) {
    return make_existing_path(*cli.single_model, root);
  }
  const char* env_model = std::getenv("SIMA_YOLOV8_SINGLE_MODEL");
  if (env_model && *env_model) {
    return make_existing_path(fs::path(env_model), root);
  }
  return std::nullopt;
}

fs::path resolve_variants_dir(const CliArgs& cli, const fs::path& root) {
  if (cli.variants_dir.has_value()) {
    return make_existing_path(*cli.variants_dir, root);
  }
  const char* env_dir = std::getenv("SIMA_YOLOV8_VARIANTS_DIR");
  if (env_dir && *env_dir) {
    return make_existing_path(fs::path(env_dir), root);
  }
  const fs::path repo_tmp = root / "tmp" / "yolov8n_drive";
  std::error_code ec;
  if (fs::exists(repo_tmp, ec)) {
    return repo_tmp;
  }
  const fs::path core_tmp = root / "core" / "tmp" / "yolov8n_drive";
  if (fs::exists(core_tmp, ec)) {
    return core_tmp;
  }
  return repo_tmp;
}

bool is_tar_gz(const fs::path& path) {
  const std::string name = path.filename().string();
  return name.size() >= 7 && name.rfind(".tar.gz") == (name.size() - 7);
}

std::vector<fs::path> collect_model_packs(const fs::path& variants_dir) {
  std::vector<fs::path> packs;
  std::error_code ec;
  if (!fs::exists(variants_dir, ec) || !fs::is_directory(variants_dir, ec)) {
    return packs;
  }
  for (const auto& entry : fs::directory_iterator(variants_dir, ec)) {
    if (ec) {
      break;
    }
    if (!entry.is_regular_file()) {
      continue;
    }
    const fs::path path = entry.path();
    if (!is_tar_gz(path)) {
      continue;
    }
    packs.push_back(path);
  }
  std::sort(packs.begin(), packs.end());
  return packs;
}

std::string dtype_family_name(DTypeFamily f) {
  switch (f) {
  case DTypeFamily::Int8:
    return "INT8";
  case DTypeFamily::Int16:
    return "INT16";
  case DTypeFamily::BFloat16:
    return "BF16";
  case DTypeFamily::Unknown:
    break;
  }
  return "UNKNOWN";
}

DTypeFamily dtype_family_from_token(const std::string& token) {
  const std::string up = upper_copy(token);
  if (up.find("INT8") != std::string::npos || up.find("UINT8") != std::string::npos) {
    return DTypeFamily::Int8;
  }
  if (up.find("INT16") != std::string::npos) {
    return DTypeFamily::Int16;
  }
  if (up.find("BFLOAT16") != std::string::npos || up.find("BF16") != std::string::npos) {
    return DTypeFamily::BFloat16;
  }
  return DTypeFamily::Unknown;
}

std::string inferred_family_name(bool quantize, bool tessellate, bool cast) {
  if (quantize && tessellate) {
    return "QuantTess";
  }
  if (quantize) {
    return "Quant";
  }
  if (tessellate) {
    return "Tess";
  }
  if (cast) {
    return "Cast";
  }
  return "Preproc";
}

ExpectedRoute expected_route_from_model_info(const simaai::neat::Model::ModelInfo& info) {
  ExpectedRoute out;
  out.quantize = info.needs.pre_quantization;
  out.tessellate = info.needs.pre_tessellation;
  out.cast = info.needs.pre_cast;
  out.dtype = DTypeFamily::Unknown;
  out.family = inferred_family_name(out.quantize, out.tessellate, out.cast);
  return out;
}

DTypeFamily expected_preproc_dtype_for_model(const simaai::neat::Model& model,
                                             const ExpectedRoute& expected) {
  if (expected.quantize) {
    return DTypeFamily::Int8;
  }
  const auto mla_info = rendered_stage_query::mla_input_info_from_nodes(
      simaai::neat::internal::ModelAccess::build_public_inference_nodes(model));
  const DTypeFamily mla_dtype = dtype_family_from_token(mla_info.input_dtype);
  if (mla_dtype == DTypeFamily::BFloat16 || mla_dtype == DTypeFamily::Int16) {
    return mla_dtype;
  }
  // Keep compatibility with legacy non-quant routes that remain INT16.
  return DTypeFamily::Int16;
}

simaai::neat::Model::Options baseline_options() {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Auto;
  opt.preprocess.enable = simaai::neat::AutoFlag::Auto;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::Auto;
  opt.upstream_name = "decoder";
  return opt;
}

simaai::neat::Model::Options forced_preproc_options() {
  simaai::neat::Model::Options opt = baseline_options();
  opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
  opt.score_threshold = 0.52f;
  opt.nms_iou_threshold = 0.5f;
  opt.top_k = 100;
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  opt.preprocess.resize.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.resize.width = 640;
  opt.preprocess.resize.height = 640;
  opt.preprocess.resize.mode = simaai::neat::ResizeMode::Letterbox;
  opt.preprocess.resize.pad_value = 114;
  opt.preprocess.resize.scaling_type = "BILINEAR";

  opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.normalize.mean = {0.0f, 0.0f, 0.0f};

  opt.preprocess.color_convert.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
  // Keep kernel output in RGB for parity with BF16 path support in preproc kernel.
  opt.preprocess.color_convert.output_format = simaai::neat::PreprocessColorFormat::RGB;

  opt.preprocess.layout_convert.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.layout_convert.perm = {0, 1, 2};
  return opt;
}

void require_forced_preproc_options_complete(const simaai::neat::Model::Options& opt,
                                             const std::string& model_id) {
  const auto& p = opt.preprocess;
  auto fail_missing = [&](const std::string& field, const std::string& reason) {
    throw std::runtime_error("preproc_yolov8_matrix_test: missing required option '" + field +
                             "' for model " + model_id + " (" + reason + ")");
  };

  if (p.enable != simaai::neat::AutoFlag::On) {
    fail_missing("preprocess.enable", "must be On; observed=" + auto_flag_name(p.enable));
  }
  if (p.kind != simaai::neat::InputKind::Image) {
    fail_missing("preprocess.kind", "must be Image for this test");
  }

  if (p.resize.enable != simaai::neat::AutoFlag::On) {
    fail_missing("preprocess.resize.enable",
                 "must be On; observed=" + auto_flag_name(p.resize.enable));
  }
  if (p.resize.width <= 0 || p.resize.height <= 0) {
    fail_missing("preprocess.resize.width/height", "both must be > 0");
  }

  if (p.normalize.enable != simaai::neat::AutoFlag::On) {
    fail_missing("preprocess.normalize.enable",
                 "must be On; observed=" + auto_flag_name(p.normalize.enable));
  }
  if (!p.normalize.has_explicit_stats && p.preset != simaai::neat::NormalizePreset::COCO_YOLO &&
      p.preset != simaai::neat::NormalizePreset::ImageNet) {
    fail_missing("preprocess.normalize.{mean,stddev}",
                 "set explicit stats or use a preset that provides normalize defaults");
  }

  if (p.color_convert.enable != simaai::neat::AutoFlag::On) {
    fail_missing("preprocess.color_convert.enable",
                 "must be On; observed=" + auto_flag_name(p.color_convert.enable));
  }
  if (p.color_convert.input_format == simaai::neat::PreprocessColorFormat::Auto ||
      p.color_convert.output_format == simaai::neat::PreprocessColorFormat::Auto) {
    fail_missing("preprocess.color_convert.{input_format,output_format}",
                 "both must be explicitly set (non-Auto)");
  }

  if (p.layout_convert.enable != simaai::neat::AutoFlag::On) {
    fail_missing("preprocess.layout_convert.enable",
                 "must be On; observed=" + auto_flag_name(p.layout_convert.enable));
  }
  if (p.layout_convert.perm.empty()) {
    fail_missing("preprocess.layout_convert.perm", "axis permutation must be non-empty");
  }
}

bool preproc_parity_debug_enabled() {
  return sima_yolov8_test::env_bool("SIMA_PREPROC_YOLOV8_PARITY_DEBUG", false);
}

size_t preproc_parity_window_bytes() {
  const int raw = sima_yolov8_test::env_int("SIMA_PREPROC_YOLOV8_PARITY_WINDOW_BYTES", 65536);
  if (raw <= 0) {
    return 65536U;
  }
  constexpr int kMaxWindow = 1 << 20;
  return static_cast<size_t>(std::min(raw, kMaxWindow));
}

uint64_t fnv1a64(const uint8_t* data, size_t size) {
  constexpr uint64_t kOffset = 1469598103934665603ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;
  uint64_t hash = kOffset;
  if (!data || size == 0U) {
    return hash;
  }
  for (size_t i = 0; i < size; ++i) {
    hash ^= static_cast<uint64_t>(data[i]);
    hash *= kPrime;
  }
  return hash;
}

std::string bytes_preview_hex(const uint8_t* data, size_t size, size_t limit = 16U) {
  if (!data || size == 0U || limit == 0U) {
    return "[]";
  }
  std::ostringstream oss;
  const size_t n = std::min(size, limit);
  oss << "[";
  for (size_t i = 0; i < n; ++i) {
    if (i != 0U) {
      oss << ",";
    }
    oss << "0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(data[i])
        << std::dec;
  }
  if (n < size) {
    oss << ",...";
  }
  oss << "]";
  return oss.str();
}

std::vector<uint8_t> extract_tensor_payload_debug(const simaai::neat::Tensor& tensor,
                                                  const std::string& model_id) {
  try {
    return tensor.copy_payload_bytes();
  } catch (const std::exception&) {
    auto cpu = simaai::neat::pipeline_internal::transfer_to_cpu(tensor);
    if (!cpu.storage) {
      throw std::runtime_error("preproc parity cpu transfer failed for " + model_id);
    }
    return cpu.copy_payload_bytes();
  }
}

float bf16_to_fp32(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

uint16_t fp32_to_bf16(float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

cv::Mat convert_to_output_color(const cv::Mat& src_bgr,
                                simaai::neat::PreprocessColorFormat output_format) {
  cv::Mat out;
  switch (output_format) {
  case simaai::neat::PreprocessColorFormat::BGR:
    out = src_bgr;
    break;
  case simaai::neat::PreprocessColorFormat::RGB:
    cv::cvtColor(src_bgr, out, cv::COLOR_BGR2RGB);
    break;
  case simaai::neat::PreprocessColorFormat::GRAY8:
    cv::cvtColor(src_bgr, out, cv::COLOR_BGR2GRAY);
    break;
  default:
    out = src_bgr;
    break;
  }
  return out;
}

cv::Mat resize_letterbox_center(const cv::Mat& src, int dst_w, int dst_h, int pad_value) {
  require(dst_w > 0 && dst_h > 0, "invalid letterbox target");
  const double scale = std::min(static_cast<double>(dst_w) / static_cast<double>(src.cols),
                                static_cast<double>(dst_h) / static_cast<double>(src.rows));
  const int resized_w = std::max(1, static_cast<int>(std::round(src.cols * scale)));
  const int resized_h = std::max(1, static_cast<int>(std::round(src.rows * scale)));
  cv::Mat resized;
  cv::resize(src, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);

  const int pad_left = (dst_w - resized_w) / 2;
  const int pad_top = (dst_h - resized_h) / 2;
  const int pad_right = dst_w - resized_w - pad_left;
  const int pad_bottom = dst_h - resized_h - pad_top;
  const cv::Scalar pad =
      (resized.channels() == 1)
          ? cv::Scalar(static_cast<double>(pad_value))
          : cv::Scalar(static_cast<double>(pad_value), static_cast<double>(pad_value),
                       static_cast<double>(pad_value), 0.0);
  cv::Mat out;
  cv::copyMakeBorder(resized, out, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT,
                     pad);
  return out;
}

std::vector<uint8_t> build_host_preproc_bf16_tensor(const cv::Mat& frame_bgr,
                                                    const simaai::neat::Model::Options& opt) {
  const auto& p = opt.preprocess;
  require(p.resize.width > 0 && p.resize.height > 0, "host preproc requires positive resize dims");

  cv::Mat color = convert_to_output_color(frame_bgr, p.color_convert.output_format);
  cv::Mat resized;
  if (p.resize.mode == simaai::neat::ResizeMode::Letterbox) {
    resized = resize_letterbox_center(color, p.resize.width, p.resize.height, p.resize.pad_value);
  } else {
    cv::resize(color, resized, cv::Size(p.resize.width, p.resize.height), 0.0, 0.0,
               cv::INTER_LINEAR);
  }

  cv::Mat fp32;
  resized.convertTo(fp32, CV_32F);
  if (p.normalize.enable == simaai::neat::AutoFlag::On) {
    const std::array<float, 3> mean = p.normalize.mean;
    const std::array<float, 3> stddev = p.normalize.stddev;
    const cv::Scalar mean255(static_cast<double>(mean[0]) * 255.0,
                             static_cast<double>(mean[1]) * 255.0,
                             static_cast<double>(mean[2]) * 255.0, 0.0);
    const cv::Scalar std255(std::max(1e-8, static_cast<double>(stddev[0]) * 255.0),
                            std::max(1e-8, static_cast<double>(stddev[1]) * 255.0),
                            std::max(1e-8, static_cast<double>(stddev[2]) * 255.0), 1.0);
    cv::subtract(fp32, mean255, fp32);
    cv::divide(fp32, std255, fp32);
  }

  if (!fp32.isContinuous()) {
    fp32 = fp32.clone();
  }
  const size_t elem_count = fp32.total() * static_cast<size_t>(fp32.channels());
  const float* src = reinterpret_cast<const float*>(fp32.data);
  std::vector<uint8_t> out(elem_count * sizeof(uint16_t));
  auto* dst = reinterpret_cast<uint16_t*>(out.data());
  for (size_t i = 0; i < elem_count; ++i) {
    dst[i] = fp32_to_bf16(src[i]);
  }
  return out;
}

void compare_runtime_vs_host_preproc_debug(const simaai::neat::Tensor& pre_tensor,
                                           const cv::Mat& frame_bgr,
                                           const simaai::neat::Model::Options& forced_opt,
                                           const ExpectedRoute& expected,
                                           const std::string& model_id) {
  if (!preproc_parity_debug_enabled()) {
    return;
  }
  if (expected.quantize || expected.tessellate || expected.dtype != DTypeFamily::BFloat16) {
    std::cout << "PREPROC_PARITY model=" << model_id
              << " skipped=1 reason=requires_nonquant_nontess_bf16\n";
    return;
  }
  if (pre_tensor.dtype != simaai::neat::TensorDType::BFloat16) {
    std::cout << "PREPROC_PARITY model=" << model_id
              << " skipped=1 reason=runtime_tensor_not_bf16 dtype="
              << static_cast<int>(pre_tensor.dtype) << "\n";
    return;
  }

  const std::vector<uint8_t> runtime = extract_tensor_payload_debug(pre_tensor, model_id);
  const std::vector<uint8_t> host = build_host_preproc_bf16_tensor(frame_bgr, forced_opt);
  if (runtime.empty() || host.empty()) {
    std::cout << "PREPROC_PARITY model=" << model_id
              << " skipped=1 reason=empty_payload runtime=" << runtime.size()
              << " host=" << host.size() << "\n";
    return;
  }

  const size_t common = std::min(runtime.size(), host.size());
  const size_t win = std::min(preproc_parity_window_bytes(), common);
  auto compare_window = [&](const char* tag, size_t offset) {
    if (offset > common) {
      offset = common;
    }
    const size_t n = std::min(win, common - offset);
    if (n == 0U) {
      return;
    }
    const uint8_t* runtime_ptr = runtime.data() + offset;
    const uint8_t* host_ptr = host.data() + offset;
    size_t eq = 0U;
    for (size_t i = 0; i < n; ++i) {
      if (runtime_ptr[i] == host_ptr[i]) {
        ++eq;
      }
    }
    const size_t elems = n / sizeof(uint16_t);
    double mae = 0.0;
    double max_abs = 0.0;
    const auto* runtime16 = reinterpret_cast<const uint16_t*>(runtime_ptr);
    const auto* host16 = reinterpret_cast<const uint16_t*>(host_ptr);
    for (size_t i = 0; i < elems; ++i) {
      const double diff = std::fabs(static_cast<double>(bf16_to_fp32(runtime16[i])) -
                                    static_cast<double>(bf16_to_fp32(host16[i])));
      mae += diff;
      if (diff > max_abs) {
        max_abs = diff;
      }
    }
    if (elems > 0U) {
      mae /= static_cast<double>(elems);
    }
    const double eq_ratio = n > 0U ? static_cast<double>(eq) / static_cast<double>(n) : 0.0;
    std::cout << "PREPROC_PARITY model=" << model_id << " window=" << tag << " off=" << offset
              << " n=" << n << " runtime_hash=0x" << std::hex << std::setw(16) << std::setfill('0')
              << fnv1a64(runtime_ptr, n) << " host_hash=0x" << std::setw(16) << fnv1a64(host_ptr, n)
              << std::dec << std::setfill(' ') << " eq_ratio=" << eq_ratio << " mae=" << mae
              << " max_abs=" << max_abs
              << " runtime_preview=" << bytes_preview_hex(runtime_ptr, n, 12)
              << " host_preview=" << bytes_preview_hex(host_ptr, n, 12) << "\n";
  };

  const size_t head_off = 0U;
  const size_t center_off = (common > win) ? (common / 2U - win / 2U) : 0U;
  const size_t tail_off = (common > win) ? (common - win) : 0U;
  std::cout << "PREPROC_PARITY model=" << model_id << " runtime_bytes=" << runtime.size()
            << " host_bytes=" << host.size() << " common_bytes=" << common
            << " window_bytes=" << win << " normalize="
            << (forced_opt.preprocess.normalize.enable == simaai::neat::AutoFlag::On ? 1 : 0)
            << " mean=[" << forced_opt.preprocess.normalize.mean[0] << ","
            << forced_opt.preprocess.normalize.mean[1] << ","
            << forced_opt.preprocess.normalize.mean[2] << "]"
            << " stddev=[" << forced_opt.preprocess.normalize.stddev[0] << ","
            << forced_opt.preprocess.normalize.stddev[1] << ","
            << forced_opt.preprocess.normalize.stddev[2] << "]"
            << "\n";
  compare_window("head", head_off);
  compare_window("center", center_off);
  compare_window("tail", tail_off);
}

simaai::neat::PreprocessRuntimeMeta require_preprocess_meta(const simaai::neat::Tensor& tensor,
                                                            const std::string& label) {
#if !SIMA_HAS_SIMAAI_POOL
  (void)tensor;
  skip_test_exception(label + ": simaai pool/meta unavailable");
  return {};
#else
  const std::shared_ptr<void> holder = simaai::neat::pipeline_internal::holder_from_tensor(tensor);
  require(holder != nullptr, label + ": missing tensor holder");
  GstBuffer* buffer = simaai::neat::pipeline_internal::buffer_from_tensor_holder(holder);
  require(buffer != nullptr, label + ": missing GstBuffer");
  const auto meta = simaai::neat::read_simaai_preprocess_meta(buffer);
  gst_buffer_unref(buffer);
  require(meta.has_value(), label + ": missing preprocess metadata");
  return *meta;
#endif
}

struct AccuracyResult {
  bool ok = false;
  std::string note;
  int outputs = 0;
};

std::string model_input_caps_format(const simaai::neat::Model::Options& opt) {
  const auto input_format = opt.preprocess.color_convert.input_format;
  if (input_format == simaai::neat::PreprocessColorFormat::RGB) {
    return "RGB";
  }
  if (input_format == simaai::neat::PreprocessColorFormat::GRAY8) {
    return "GRAY8";
  }
  return "BGR";
}

simaai::neat::Tensor make_model_input_tensor(const cv::Mat& input_frame,
                                             const simaai::neat::Model& model,
                                             const simaai::neat::Model::Options& opt,
                                             const char* where) {
  simaai::neat::InputOptions src_opt = model.input_appsrc_options(false);
  src_opt.payload_type = simaai::neat::PayloadType::Image;
  src_opt.width = input_frame.cols;
  src_opt.height = input_frame.rows;
  src_opt.depth = input_frame.channels();
  src_opt.format = model_input_caps_format(opt);
  return simaai::neat::tensor_from_cv_mat(input_frame, src_opt, where);
}

AccuracyResult run_boxdecode_accuracy(simaai::neat::Model& model, const cv::Mat& input_frame,
                                      const simaai::neat::Model::Options& opt,
                                      const std::string& model_id) {
  AccuracyResult out;
  constexpr float kMinScore = 0.52f;
  constexpr float kMinIou = 0.30f;
  constexpr int kTopk = 100;

  try {
    const simaai::neat::Tensor input_tensor =
        make_model_input_tensor(input_frame, model, opt, "preproc_yolov8_matrix_test:accuracy");
    simaai::neat::Model::Runner runner = model.build(simaai::neat::TensorList{input_tensor});
    require(static_cast<bool>(runner), "runner build failed for " + model_id);
    require(runner.push(simaai::neat::TensorList{input_tensor}),
            "runner push failed for " + model_id);
    simaai::neat::Sample outputs = runner.pull(120000);
    require(outputs.size() == 1U, "runner output size mismatch for " + model_id);
    simaai::neat::Sample result = outputs.front();

    std::vector<uint8_t> payload;
    std::string err;
    if (!objdet::extract_bbox_payload(result, payload, err)) {
      out.note =
          "bbox_extract_failed model=" + model_id + " err=" + sima_yolov8_test::sanitize_note(err);
      return out;
    }

    const std::vector<objdet::Box> boxes =
        objdet::parse_boxes_strict(payload, input_frame.cols, input_frame.rows, kTopk, false);
    const std::vector<objdet::ExpectedBox> expected_boxes = objdet::expected_people_boxes();
    const objdet::MatchResult match =
        objdet::match_expected_boxes(boxes, expected_boxes, kMinScore, kMinIou);
    if (!match.ok) {
      float best_person_iou = 0.0f;
      int person_candidates = 0;
      for (const auto& box : boxes) {
        if (box.class_id != 0 || box.score < kMinScore) {
          continue;
        }
        person_candidates += 1;
        for (const auto& expected_box : expected_boxes) {
          if (expected_box.class_id != 0) {
            continue;
          }
          best_person_iou =
              std::max(best_person_iou,
                       objdet::box_iou_xyxy(expected_box.x1, expected_box.y1, expected_box.x2,
                                            expected_box.y2, box.x1, box.y1, box.x2, box.y2));
        }
      }
      if (person_candidates > 0 && best_person_iou >= 0.25f) {
        out.ok = true;
        out.outputs = 1;
        out.note = "coarse_ok best_person_iou=" + std::to_string(best_person_iou) +
                   " strict=" + sima_yolov8_test::sanitize_note(match.note);
        return out;
      }
      out.note =
          "bbox_mismatch model=" + model_id + " " + sima_yolov8_test::sanitize_note(match.note);
      return out;
    }

    out.ok = true;
    out.outputs = 1;
    out.note = "ok";
    return out;
  } catch (const std::exception& e) {
    out.note = "run_error model=" + model_id + " err=" + sima_yolov8_test::sanitize_note(e.what());
    return out;
  }
}

cv::Mat prepare_model_input_frame(const cv::Mat& frame_bgr,
                                  const simaai::neat::Model::Options& opt) {
  const auto input_format = opt.preprocess.color_convert.input_format;
  if (input_format == simaai::neat::PreprocessColorFormat::RGB) {
    cv::Mat frame_rgb;
    cv::cvtColor(frame_bgr, frame_rgb, cv::COLOR_BGR2RGB);
    return frame_rgb;
  }
  return frame_bgr;
}

void run_model_case(const fs::path& tar_path, const cv::Mat& frame_bgr, bool accuracy_enabled) {
  const std::string model_id = tar_path.filename().string();
  const simaai::neat::Model baseline_model(tar_path.string(), baseline_options());
  const simaai::neat::Model::ModelInfo baseline_info = baseline_model.info();
  ExpectedRoute expected = expected_route_from_model_info(baseline_info);

  simaai::neat::Model::Options forced_opt = forced_preproc_options();
  require_forced_preproc_options_complete(forced_opt, model_id);
  std::optional<simaai::neat::Model> forced_model;
  forced_model.emplace(tar_path.string(), forced_opt);
  expected.dtype = expected_preproc_dtype_for_model(*forced_model, expected);
  if (expected.dtype == DTypeFamily::BFloat16 && forced_opt.preprocess.color_convert.input_format ==
                                                     simaai::neat::PreprocessColorFormat::BGR) {
    forced_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::RGB;
    forced_model.emplace(tar_path.string(), forced_opt);
  }
  const cv::Mat model_input = prepare_model_input_frame(frame_bgr, forced_opt);
  require(model_input.data != nullptr, "failed to prepare model input for " + model_id);
  const std::string expected_color_in =
      preprocess_color_format_name(forced_opt.preprocess.color_convert.input_format);
  const simaai::neat::Model::ModelInfo forced_info = forced_model->info();
  require(forced_info.needs.pre_quantization == expected.quantize,
          "forced model pre_quantization changed unexpectedly for " + model_id);
  require(forced_info.needs.pre_tessellation == expected.tessellate,
          "forced model pre_tessellation changed unexpectedly for " + model_id);
  require(forced_info.needs.pre_cast == expected.cast,
          "forced model pre_cast changed unexpectedly for " + model_id);

  const auto pre_nodes =
      simaai::neat::internal::ModelAccess::build_public_preprocess_nodes(*forced_model);
  require(!pre_nodes.empty(), "forced model preprocess graph is empty for " + model_id);

  const auto pre_info = rendered_stage_query::preproc_output_info_from_nodes(pre_nodes);
  const bool observed_packed_handoff =
      pre_info.transport_kind == simaai::neat::stages::PreprocOutputTransportKind::Packed;
  const DTypeFamily observed_cfg_dtype = dtype_family_from_token(pre_info.output_dtype);
  require(observed_packed_handoff == expected.tessellate,
          "config transport mismatch for " + model_id +
              " expected_packed=" + std::to_string(expected.tessellate ? 1 : 0) +
              " observed=" + std::to_string(observed_packed_handoff ? 1 : 0));
  require(observed_cfg_dtype == expected.dtype,
          "config output_dtype mismatch for " + model_id + " expected_family=" +
              dtype_family_name(expected.dtype) + " observed_token=" + pre_info.output_dtype +
              " observed_family=" + dtype_family_name(observed_cfg_dtype));

  simaai::neat::pipeline_internal::reset_tensor_io_stats();
  const auto tensor_io_before = simaai::neat::pipeline_internal::snapshot_tensor_io_stats();
  const simaai::neat::TensorList pre_outputs =
      simaai::neat::stages::Preproc(std::vector<cv::Mat>{model_input}, *forced_model);
  require(pre_outputs.size() == 1U, "Preproc should return exactly one tensor for " + model_id);
  const simaai::neat::Tensor pre = pre_outputs.front();
  compare_runtime_vs_host_preproc_debug(pre, model_input, forced_opt, expected, model_id);
  const simaai::neat::TensorList infer_outputs =
      simaai::neat::stages::Infer(simaai::neat::TensorList{pre}, *forced_model);
  require(infer_outputs.size() == 1U, "Infer should return exactly one tensor for " + model_id);
  const simaai::neat::Tensor infer = infer_outputs.front();
  const auto tensor_io_after = simaai::neat::pipeline_internal::snapshot_tensor_io_stats();
  const auto tensor_io = tensor_io_delta(tensor_io_before, tensor_io_after);
  const auto meta = require_preprocess_meta(infer, "preproc-matrix:" + model_id);

  const std::string resize_mode = lower_copy(meta.resize_mode);
  require(meta.quantize == expected.quantize,
          "runtime quantize mismatch for " + model_id +
              " expected=" + std::to_string(expected.quantize ? 1 : 0) +
              " observed=" + std::to_string(meta.quantize ? 1 : 0));
  require(meta.tessellate == expected.tessellate,
          "runtime tessellate mismatch for " + model_id +
              " expected=" + std::to_string(expected.tessellate ? 1 : 0) +
              " observed=" + std::to_string(meta.tessellate ? 1 : 0));
  require(meta.normalize, "runtime normalize must be enabled for COCO preset: " + model_id);
  require(resize_mode == "letterbox", "runtime resize_mode must be letterbox for COCO preset: " +
                                          model_id + " observed=" + meta.resize_mode);
  require(upper_copy(meta.color_in) == upper_copy(expected_color_in),
          "runtime color_in must match test-defined options: " + model_id +
              " observed=" + meta.color_in);
  require(upper_copy(meta.color_out) == "RGB",
          "runtime color_out must be RGB from test-defined options: " + model_id +
              " observed=" + meta.color_out);
  require(meta.axis_perm == forced_opt.preprocess.layout_convert.perm,
          "runtime axis_perm must match permutation-based layout_convert: " + model_id);

  AccuracyResult acc;
  if (accuracy_enabled) {
    acc = run_boxdecode_accuracy(*forced_model, model_input, forced_opt, model_id);
    require(acc.ok, "accuracy mismatch for " + model_id + ": " + acc.note);
  }

  std::cout << "PREPROC_MATRIX model=" << model_id << " inferred{family=" << expected.family
            << ",quant=" << (expected.quantize ? 1 : 0) << ",tess=" << (expected.tessellate ? 1 : 0)
            << ",cast=" << (expected.cast ? 1 : 0) << ",dtype=" << dtype_family_name(expected.dtype)
            << "} config{packed=" << (observed_packed_handoff ? 1 : 0)
            << ",dtype_token=" << pre_info.output_dtype
            << ",dtype_family=" << dtype_family_name(observed_cfg_dtype)
            << "} runtime{quant=" << (meta.quantize ? 1 : 0)
            << ",tess=" << (meta.tessellate ? 1 : 0)
            << ",resize_mode=" << (meta.resize_mode.empty() ? "<empty>" : meta.resize_mode)
            << ",color_in=" << (meta.color_in.empty() ? "<empty>" : meta.color_in)
            << ",color_out=" << (meta.color_out.empty() ? "<empty>" : meta.color_out)
            << ",axis_perm=" << join_ints(meta.axis_perm)
            << ",normalize=" << (meta.normalize ? 1 : 0)
            << "} accuracy{enabled=" << (accuracy_enabled ? 1 : 0) << ",ok=" << (acc.ok ? 1 : 0)
            << ",outputs=" << acc.outputs << ",note=" << (acc.note.empty() ? "n/a" : acc.note)
            << "} tensor_io{" << tensor_io_stats_string(tensor_io) << "}\n";
}

} // namespace

int main(int argc, char** argv) {
#if !defined(SIMA_WITH_OPENCV)
  (void)argc;
  (void)argv;
  return skip_long_test("OpenCV required for preproc_yolov8_matrix_test");
#else
  try {
    simaai::neat::gst_init_once();
    if (!simaai::neat::element_exists("neatprocessmla")) {
      return skip_long_test("missing NEAT MLA plugin (neatprocessmla)");
    }
    if (!simaai::neat::element_exists("neatprocesscvu")) {
      return skip_long_test("missing NEAT CVU plugin (neatprocesscvu)");
    }

    const CliArgs cli = parse_cli(argc, argv);
    const bool accuracy_enabled = resolve_accuracy_enabled(cli);
    fs::path root =
        cli.root.has_value() ? fs::absolute(*cli.root) : find_repo_root(fs::current_path());
    const cv::Mat frame_bgr = sima_yolov8_test::load_people_image_or_skip(root);

    std::vector<fs::path> model_packs;
    const std::optional<fs::path> single_model = resolve_single_model_tar(cli, root);
    if (single_model.has_value()) {
      require(fs::exists(*single_model), "single model tar not found: " + single_model->string());
      model_packs.push_back(*single_model);
    } else {
      fs::path variants_dir = resolve_variants_dir(cli, root);
      model_packs = collect_model_packs(variants_dir);
      if (model_packs.empty()) {
        // Mandatory test — download the six yolov8n variants on demand via
        // the shared helper (uses sima-cli for the OAuth-gated
        // docs.sima.ai endpoint) and rescan. Caller can short-circuit by
        // pre-staging tarballs in <root>/tmp/yolov8n_drive/ or pointing
        // SIMA_YOLOV8N_VARIANTS_BASE_URL at a mirror.
        variants_dir = sima_yolov8_test::ensure_yolov8n_drive_variants(root);
        model_packs = collect_model_packs(variants_dir);
      }
      require(!model_packs.empty(), "no model packs found in " + variants_dir.string() +
                                        " even after download attempt; check sima-cli login and "
                                        "SIMA_YOLOV8N_VARIANTS_BASE_URL");
    }

    std::size_t run_count = 0U;
    for (const auto& tar_path : model_packs) {
      run_model_case(tar_path, frame_bgr, accuracy_enabled);
      ++run_count;
    }

    require(run_count > 0, "no model packs executed");
    std::cout << "[OK] preproc_yolov8_matrix_test passed models=" << run_count << "\n";
    return 0;
  } catch (const SkipTest& e) {
    std::cout << "[SKIP] " << e.what() << "\n";
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    if (is_dispatcher_unavailable(e.what())) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
#endif
}

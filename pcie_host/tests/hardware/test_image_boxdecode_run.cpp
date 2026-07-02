#include <simaai/neat/pcie/SimaPCIeHost.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pcie = simaai::neat::pcie;

namespace {

std::string env_or_default(const char* name, const char* fallback) {
  if (const char* value = std::getenv(name)) {
    if (*value != '\0') {
      return value;
    }
  }
  return fallback ? fallback : "";
}

std::string env_or_default(const char* primary, const char* secondary, const char* fallback) {
  const std::string primary_value = env_or_default(primary, "");
  if (!primary_value.empty()) {
    return primary_value;
  }
  return env_or_default(secondary, fallback);
}

int env_int_or_default(const char* name, const int fallback) {
  const std::string value = env_or_default(name, "");
  if (value.empty()) {
    return fallback;
  }
  std::size_t parsed = 0;
  try {
    const int result = std::stoi(value, &parsed);
    if (parsed == value.size()) {
      return result;
    }
  } catch (const std::exception&) {
  }
  { throw std::runtime_error(std::string("invalid integer in ") + name + ": " + value); }
}

struct Args {
  std::string model = env_or_default("SIMAPCIE_YOLOV8_MODEL", DEFAULT_MODEL_PATH);
  std::string image =
      env_or_default("SIMAPCIE_BOXDECODE_IMAGE", "SIMAPCIE_TEST_IMAGE", DEFAULT_SOURCE_IMAGE);
  std::string card_host = env_or_default("SIMAPCIE_CARD_HOST", "");
  std::string user = env_or_default("SIMAPCIE_USER", "sima");
  int card_id = env_int_or_default("SIMAPCIE_CARD_ID", 0);
  int queue = env_int_or_default("SIMAPCIE_QUEUE", 0);
  int readiness_timeout_ms = env_int_or_default("SIMAPCIE_READINESS_TIMEOUT_MS", 180000);
  int pull_timeout_ms = env_int_or_default("SIMAPCIE_PULL_TIMEOUT_MS", 30000);
  float score_threshold = 0.25f;
  float nms_iou_threshold = 0.45f;
  int top_k = 100;
  pcie::BoxDecodeType decode_type = pcie::BoxDecodeType::YoloV8;
  std::string decode_type_name = "YoloV8";
  std::string card_env = env_or_default("SIMAPCIE_CARD_ENV", "");
  std::string card_gst_debug = env_or_default("SIMAPCIE_CARD_GST_DEBUG", "");
  std::string card_gst_debug_file = env_or_default("SIMAPCIE_CARD_GST_DEBUG_FILE", "");
  int iterations = 1;
  int resize_source_width = 0;
  int resize_source_height = 0;
  bool resize_alternate = false;
  bool require_detection = true;
  bool require_person = false;
  bool opencv_overload = false;
};

std::string require_value(int argc, char** argv, int& i, const char* name) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + name);
  }
  return argv[++i];
}

pcie::BoxDecodeType parse_decode_type(const std::string& value, std::string* display_name) {
  if (value == "yolov8" || value == "YoloV8" || value == "yolo8") {
    if (display_name) {
      *display_name = "YoloV8";
    }
    return pcie::BoxDecodeType::YoloV8;
  }
  throw std::runtime_error("unsupported --decode-type: " + value);
}

void parse_size_arg(const std::string& value, int* width, int* height) {
  const std::size_t sep = value.find('x');
  if (sep == std::string::npos || sep == 0 || sep + 1 >= value.size()) {
    throw std::runtime_error("size must use WxH syntax");
  }
  *width = std::stoi(value.substr(0, sep));
  *height = std::stoi(value.substr(sep + 1));
  if (*width <= 0 || *height <= 0) {
    throw std::runtime_error("size must be positive");
  }
}

void usage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " [--model model.tar.gz] [--image image.jpg] [--card-host host]"
               " [--card-id n] [--user user] [--queue n]"
               " [--readiness-timeout-ms ms] [--pull-timeout-ms ms]"
               " [--decode-type yolov8]"
               " [--score-threshold f] [--nms-iou-threshold f] [--top-k n]"
               " [--card-env 'NAME=VALUE ...']"
               " [--card-gst-debug spec] [--card-gst-debug-file path]"
               " [--iterations n] [--resize-source WxH] [--resize-alternate]"
               " [--require-detection|--allow-empty-detections]"
               " [--require-person]"
               " [--opencv-overload]\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i] ? argv[i] : "";
    if (arg == "--model") {
      args.model = require_value(argc, argv, i, "--model");
    } else if (arg == "--image") {
      args.image = require_value(argc, argv, i, "--image");
    } else if (arg == "--card-host") {
      args.card_host = require_value(argc, argv, i, "--card-host");
    } else if (arg == "--card-id") {
      args.card_id = std::stoi(require_value(argc, argv, i, "--card-id"));
    } else if (arg == "--user") {
      args.user = require_value(argc, argv, i, "--user");
    } else if (arg == "--queue") {
      args.queue = std::stoi(require_value(argc, argv, i, "--queue"));
    } else if (arg == "--readiness-timeout-ms") {
      args.readiness_timeout_ms = std::stoi(require_value(argc, argv, i, "--readiness-timeout-ms"));
    } else if (arg == "--pull-timeout-ms") {
      args.pull_timeout_ms = std::stoi(require_value(argc, argv, i, "--pull-timeout-ms"));
    } else if (arg == "--decode-type") {
      args.decode_type =
          parse_decode_type(require_value(argc, argv, i, "--decode-type"), &args.decode_type_name);
    } else if (arg == "--score-threshold") {
      args.score_threshold = std::stof(require_value(argc, argv, i, "--score-threshold"));
    } else if (arg == "--nms-iou-threshold") {
      args.nms_iou_threshold = std::stof(require_value(argc, argv, i, "--nms-iou-threshold"));
    } else if (arg == "--top-k") {
      args.top_k = std::stoi(require_value(argc, argv, i, "--top-k"));
    } else if (arg == "--card-env") {
      args.card_env = require_value(argc, argv, i, "--card-env");
    } else if (arg == "--card-gst-debug") {
      args.card_gst_debug = require_value(argc, argv, i, "--card-gst-debug");
    } else if (arg == "--card-gst-debug-file") {
      args.card_gst_debug_file = require_value(argc, argv, i, "--card-gst-debug-file");
    } else if (arg == "--iterations") {
      args.iterations = std::stoi(require_value(argc, argv, i, "--iterations"));
    } else if (arg == "--resize-source") {
      parse_size_arg(require_value(argc, argv, i, "--resize-source"), &args.resize_source_width,
                     &args.resize_source_height);
    } else if (arg == "--resize-alternate") {
      args.resize_alternate = true;
    } else if (arg == "--require-detection") {
      args.require_detection = true;
    } else if (arg == "--allow-empty-detections") {
      args.require_detection = false;
    } else if (arg == "--require-person") {
      args.require_person = true;
      args.require_detection = true;
    } else if (arg == "--opencv-overload") {
      args.opencv_overload = true;
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (!std::filesystem::is_regular_file(args.model)) {
    throw std::runtime_error("model path does not exist or is not a regular file: " + args.model);
  }
  if (!std::filesystem::is_regular_file(args.image)) {
    throw std::runtime_error("image path does not exist or is not a regular file: " + args.image);
  }
  if (args.score_threshold < 0.0f || args.score_threshold > 1.0f) {
    throw std::runtime_error("--score-threshold must be in [0, 1]");
  }
  if (args.nms_iou_threshold < 0.0f || args.nms_iou_threshold > 1.0f) {
    throw std::runtime_error("--nms-iou-threshold must be in [0, 1]");
  }
  if (args.top_k <= 0) {
    throw std::runtime_error("--top-k must be positive for boxdecode");
  }
  if (args.iterations <= 0) {
    throw std::runtime_error("--iterations must be positive");
  }
  return args;
}

std::string pipeline_state_name(const pcie::PipelineState state) {
  switch (state) {
  case pcie::PipelineState::Uninitialized:
    return "Uninitialized";
  case pcie::PipelineState::Starting:
    return "Starting";
  case pcie::PipelineState::Ready:
    return "Ready";
  case pcie::PipelineState::Failed:
    return "Failed";
  case pcie::PipelineState::Stopping:
    return "Stopping";
  case pcie::PipelineState::Exited:
    return "Exited";
  }
  return "Unknown";
}

std::string shape_string(const std::vector<std::int64_t>& shape) {
  std::string out = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      out += ", ";
    }
    out += std::to_string(shape[i]);
  }
  out += "]";
  return out;
}

std::string dtype_name(const pcie::TensorDType dtype) {
  switch (dtype) {
  case pcie::TensorDType::UInt8:
    return "UINT8";
  case pcie::TensorDType::Int8:
    return "INT8";
  case pcie::TensorDType::UInt16:
    return "UINT16";
  case pcie::TensorDType::Int16:
    return "INT16";
  case pcie::TensorDType::Int32:
    return "INT32";
  case pcie::TensorDType::BFloat16:
    return "BF16";
  case pcie::TensorDType::Float32:
    return "FP32";
  case pcie::TensorDType::Float64:
    return "FP64";
  }
  return "UNKNOWN";
}

cv::Mat load_bgr_image(const std::string& path) {
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("OpenCV failed to load image: " + path);
  }
  if (image.depth() != CV_8U || image.channels() != 3) {
    throw std::runtime_error("expected OpenCV to produce an 8-bit 3-channel BGR image");
  }
  if (!image.isContinuous()) {
    image = image.clone();
  }
  return image;
}

pcie::Tensor make_bgr_image_tensor(const cv::Mat& mat) {
  if (mat.empty() || mat.depth() != CV_8U || mat.channels() != 3) {
    throw std::runtime_error("manual BGR tensor path requires a non-empty CV_8UC3 image");
  }
  auto owner = std::make_shared<cv::Mat>(mat.isContinuous() ? mat : mat.clone());
  pcie::Tensor image;
  image.dtype = pcie::TensorDType::UInt8;
  image.layout = pcie::TensorLayout::HWC;
  image.shape = {owner->rows, owner->cols, owner->channels()};
  image.strides_bytes = {static_cast<std::int64_t>(owner->step[0]), owner->channels(), 1};
  image.owner = owner;
  image.data = owner->data;
  image.size_bytes = static_cast<std::size_t>(owner->rows) * owner->step[0];
  image.byte_offset = 0;
  image.read_only = false;
  image.image_format = pcie::PixelFormat::BGR;
  image.route.name = "input_image";
  image.route.logical_index = 0;
  image.route.physical_index = 0;
  image.route.route_slot = 0;
  return image;
}

void print_model_info(const pcie::ModelInfo& info) {
  std::cout << "model metadata\n";
  std::cout << "  has_preprocess=" << (info.has_preprocess ? "true" : "false")
            << " has_boxdecode=" << (info.has_boxdecode ? "true" : "false") << "\n";
  std::cout << "  inputs (" << info.inputs.size() << ")\n";
  for (std::size_t i = 0; i < info.inputs.size(); ++i) {
    const auto& input = info.inputs[i];
    std::cout << "    [" << i << "] name=" << (input.name.empty() ? "<unnamed>" : input.name)
              << " dtype=" << (input.dtype.empty() ? "<unknown>" : input.dtype)
              << " shape=" << shape_string(input.shape) << " size_bytes=" << input.size_bytes
              << "\n";
  }
  std::cout << "  expected outputs (" << info.outputs.size() << ")\n";
  for (std::size_t i = 0; i < info.outputs.size(); ++i) {
    const auto& output = info.outputs[i];
    std::cout << "    [" << i << "] name=" << (output.name.empty() ? "<unnamed>" : output.name)
              << " dtype=" << (output.dtype.empty() ? "<unknown>" : output.dtype)
              << " shape=" << shape_string(output.shape) << " size_bytes=" << output.size_bytes
              << "\n";
  }
}

void print_image(const pcie::Tensor& image) {
  std::cout << "constructed image input\n";
  std::cout << "  name=" << image.route.name << " dtype=" << dtype_name(image.dtype)
            << " shape=" << shape_string(image.shape)
            << " strides=" << shape_string(image.strides_bytes) << " format=BGR"
            << " size_bytes=" << image.size_bytes << "\n";
}

void print_mat(const cv::Mat& image) {
  std::cout << "loaded OpenCV image\n";
  std::cout << "  width=" << image.cols << " height=" << image.rows
            << " channels=" << image.channels() << " step=" << image.step[0] << " format=BGR"
            << " bytes=" << (static_cast<std::size_t>(image.rows) * image.step[0]) << "\n";
}

void print_outputs(const pcie::TensorList& outputs) {
  std::cout << "received outputs (" << outputs.size() << ")\n";
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& output = outputs[i];
    std::cout << "  [" << i << "] name=" << output.route.name
              << " dtype=" << dtype_name(output.dtype) << " shape=" << shape_string(output.shape)
              << " size_bytes=" << output.size_bytes << " byte_offset=" << output.byte_offset
              << "\n";
  }
}

struct BBoxRecord {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  std::int32_t class_id = -1;
};

struct BBoxSummary {
  std::uint32_t header_count = 0;
  std::size_t payload_capacity = 0;
  std::size_t valid_count = 0;
  std::size_t high_score_count = 0;
  std::size_t person_count = 0;
  float max_score = 0.0f;
};

template <typename T>
T read_le_value(const std::uint8_t* data, const std::size_t size, const std::size_t offset) {
  if (offset + sizeof(T) > size) {
    throw std::runtime_error("BBOX payload is truncated");
  }
  T value{};
  std::memcpy(&value, data + offset, sizeof(T));
  return value;
}

std::vector<BBoxRecord> parse_bbox_payload(const pcie::TensorList& outputs, const int image_width,
                                           const int image_height, const int top_k,
                                           BBoxSummary* summary) {
  if (outputs.empty()) {
    throw std::runtime_error("boxdecode produced no output tensors");
  }
  if (outputs.size() != 1) {
    throw std::runtime_error("boxdecode should produce exactly one BBOX output tensor");
  }

  const pcie::Tensor& bbox = outputs.front();
  if (!bbox.data || bbox.size_bytes < sizeof(std::uint32_t)) {
    throw std::runtime_error("first boxdecode output is missing a BBOX payload");
  }

  constexpr std::size_t kHeaderBytes = sizeof(std::uint32_t);
  constexpr std::size_t kRecordBytes = 24;
  const auto* bytes = static_cast<const std::uint8_t*>(bbox.data);
  const std::uint32_t count = read_le_value<std::uint32_t>(bytes, bbox.size_bytes, 0);
  const std::size_t max_records =
      bbox.size_bytes > kHeaderBytes ? (bbox.size_bytes - kHeaderBytes) / kRecordBytes : 0;
  if (count > max_records) {
    throw std::runtime_error("BBOX detection count exceeds payload capacity");
  }
  if (count > static_cast<std::uint32_t>(top_k)) {
    throw std::runtime_error("BBOX detection count exceeds configured top_k");
  }

  std::vector<BBoxRecord> records;
  records.reserve(count);
  BBoxSummary local;
  local.header_count = count;
  local.payload_capacity = max_records;

  for (std::uint32_t i = 0; i < count; ++i) {
    const std::size_t base = kHeaderBytes + static_cast<std::size_t>(i) * kRecordBytes;
    const std::int32_t x = read_le_value<std::int32_t>(bytes, bbox.size_bytes, base + 0);
    const std::int32_t y = read_le_value<std::int32_t>(bytes, bbox.size_bytes, base + 4);
    const std::int32_t w = read_le_value<std::int32_t>(bytes, bbox.size_bytes, base + 8);
    const std::int32_t h = read_le_value<std::int32_t>(bytes, bbox.size_bytes, base + 12);
    const float score = read_le_value<float>(bytes, bbox.size_bytes, base + 16);
    const std::int32_t class_id = read_le_value<std::int32_t>(bytes, bbox.size_bytes, base + 20);

    if (!std::isfinite(score) || score < 0.0f || score > 1.0f) {
      throw std::runtime_error("BBOX record has invalid score");
    }
    if (class_id < 0) {
      throw std::runtime_error("BBOX record has invalid class id");
    }
    if (w <= 0 || h <= 0) {
      throw std::runtime_error("BBOX record has non-positive dimensions");
    }

    constexpr float kCoordTolerance = 2.0f;
    const float x1 = static_cast<float>(x);
    const float y1 = static_cast<float>(y);
    const float x2 = static_cast<float>(x + w);
    const float y2 = static_cast<float>(y + h);
    if (x1 < -kCoordTolerance || y1 < -kCoordTolerance ||
        x2 > static_cast<float>(image_width) + kCoordTolerance ||
        y2 > static_cast<float>(image_height) + kCoordTolerance) {
      throw std::runtime_error("BBOX record coordinates exceed source image bounds");
    }

    local.valid_count += 1;
    local.max_score = std::max(local.max_score, score);
    if (class_id == 0) {
      local.person_count += 1;
    }
    records.push_back(BBoxRecord{x1, y1, x2, y2, score, class_id});
  }

  if (summary) {
    *summary = local;
  }
  return records;
}

void validate_bbox_payload(const pcie::TensorList& outputs, const int image_width,
                           const int image_height, const float score_threshold, const int top_k,
                           const bool require_detection, const bool require_person) {
  BBoxSummary summary;
  const std::vector<BBoxRecord> records =
      parse_bbox_payload(outputs, image_width, image_height, top_k, &summary);
  for (const auto& record : records) {
    if (record.score >= score_threshold) {
      summary.high_score_count += 1;
    }
  }

  if (require_detection && summary.high_score_count == 0) {
    throw std::runtime_error("BBOX payload has no detection at or above score threshold");
  }
  if (require_person && summary.person_count == 0) {
    throw std::runtime_error("BBOX payload has no person-class detection");
  }

  std::cout << "BBOX payload\n";
  std::cout << "  detections=" << summary.header_count << " valid_records=" << summary.valid_count
            << " high_score_records=" << summary.high_score_count
            << " person_records=" << summary.person_count << " max_score=" << std::fixed
            << std::setprecision(4) << summary.max_score << std::defaultfloat
            << " capacity=" << summary.payload_capacity << "\n";

  const std::size_t printed = std::min<std::size_t>(records.size(), 5);
  for (std::size_t i = 0; i < printed; ++i) {
    const BBoxRecord& record = records[i];
    std::cout << "  [" << i << "] x=" << static_cast<int>(std::round(record.x1))
              << " y=" << static_cast<int>(std::round(record.y1))
              << " w=" << static_cast<int>(std::round(record.x2 - record.x1))
              << " h=" << static_cast<int>(std::round(record.y2 - record.y1))
              << " score=" << std::fixed << std::setprecision(4) << record.score
              << " class_id=" << record.class_id << std::defaultfloat << "\n";
  }
}

void print_status(const char* label, const pcie::Status& status) {
  std::cout << label << ": state=" << pipeline_state_name(status.state)
            << " queue=" << status.queue;
  if (!status.message.empty()) {
    std::cout << " message=\"" << status.message << "\"";
  }
  if (!status.error_code.empty()) {
    std::cout << " error_code=\"" << status.error_code << "\"";
  }
  std::cout << "\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);

    pcie::ConnectionOptions conn;
    conn.card_host = args.card_host;
    conn.card_id = args.card_id;
    conn.user = args.user;
    conn.queue = args.queue;
    conn.card_env = args.card_env;
    conn.card_gst_debug = args.card_gst_debug;
    conn.card_gst_debug_file = args.card_gst_debug_file;

    pcie::ModelOptions model_options;
    model_options.preprocess.kind = pcie::InputKind::Image;

    if (args.opencv_overload) {
#if !defined(SIMA_PCIE_HAS_OPENCV_OVERLOAD)
      throw std::runtime_error("installed SimaPCIeHost headers do not provide the OpenCV overload; "
                               "rebuild/reinstall sima-pcie-host-dev and rebuild this test");
#endif
    }

    std::cout << "PCIe image boxdecode run test\n";
    std::cout << "  model=" << args.model << "\n";
    std::cout << "  image=" << args.image << "\n";
    std::cout << "  push_mode=" << (args.opencv_overload ? "opencv-overload" : "manual-bgr-tensor")
              << "\n";
    std::cout << "  card_host="
              << (conn.card_host.empty() ? ("10.0." + std::to_string(conn.card_id) + ".2")
                                         : conn.card_host)
              << " card_id=" << conn.card_id << " user=" << conn.user << " queue=" << conn.queue
              << "\n";
    std::cout << "  boxdecode=" << args.decode_type_name
              << " score_threshold=" << args.score_threshold
              << " nms_iou_threshold=" << args.nms_iou_threshold << " top_k=" << args.top_k << "\n";
    std::cout << "  iterations=" << args.iterations << " resize_source="
              << (args.resize_source_width > 0 ? (std::to_string(args.resize_source_width) + "x" +
                                                  std::to_string(args.resize_source_height))
                                               : std::string("none"))
              << " resize_alternate=" << (args.resize_alternate ? "true" : "false")
              << " require_detection=" << (args.require_detection ? "true" : "false")
              << " require_person=" << (args.require_person ? "true" : "false") << "\n";
    if (!conn.card_env.empty()) {
      std::cout << "  card_env=" << conn.card_env << "\n";
    }
    if (!conn.card_gst_debug.empty()) {
      std::cout << "  card_gst_debug=" << conn.card_gst_debug << "\n";
      std::cout << "  card_gst_debug_file="
                << (conn.card_gst_debug_file.empty()
                        ? ("/var/log/sima-neat/pcie/q" + std::to_string(conn.queue) + ".gst.log")
                        : conn.card_gst_debug_file)
                << "\n";
    }

    cv::Mat bgr = load_bgr_image(args.image);
    print_mat(bgr);
    if (args.resize_source_width > 0) {
      cv::Mat resized;
      cv::resize(bgr, resized, cv::Size(args.resize_source_width, args.resize_source_height), 0.0,
                 0.0, cv::INTER_LINEAR);
      if (!resized.isContinuous()) {
        resized = resized.clone();
      }
      bgr = resized;
      std::cout << "resized source image\n";
      print_mat(bgr);
    }

    pcie::SimaPCIeHost host(conn);
    print_status("initial status", host.status());

    std::cout << "preprocess resize\n";
    std::cout << "  enabled=true mode=letterbox target=core-inferred\n";

    model_options.preprocess.color_convert.input_format = pcie::ColorFormat::BGR;
    model_options.preprocess.input_max_width = bgr.cols;
    model_options.preprocess.input_max_height = bgr.rows;
    model_options.preprocess.input_max_depth = bgr.channels();
    model_options.preprocess.resize.enable = pcie::AutoFlag::On;
    model_options.preprocess.resize.mode = pcie::ResizeMode::Letterbox;
    model_options.decode_type = args.decode_type;
    model_options.score_threshold = args.score_threshold;
    model_options.nms_iou_threshold = args.nms_iou_threshold;
    model_options.top_k = args.top_k;

    std::cout << "loading image metadata and starting card/host pipelines...\n";
    const auto started = std::chrono::steady_clock::now();
    const pcie::ModelInfo info =
        host.init_pipeline(args.model, model_options, args.readiness_timeout_ms);
    const auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - started)
                             .count();
    std::cout << "init_pipeline completed in " << init_ms << " ms\n";
    print_status("ready status", host.status());
    print_model_info(info);

    std::vector<cv::Mat> frames;
    frames.push_back(bgr);
    if (args.resize_alternate) {
      cv::Mat resized;
      const int resized_width = std::max(96, bgr.cols / 2);
      const int resized_height = std::max(96, bgr.rows / 2);
      cv::resize(bgr, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);
      if (!resized.isContinuous()) {
        resized = resized.clone();
      }
      frames.push_back(resized);
      std::cout << "alternate resized image\n";
      print_mat(resized);
    }

    for (int iter = 0; iter < args.iterations; ++iter) {
      const cv::Mat& frame = frames[static_cast<std::size_t>(iter) % frames.size()];
      std::cout << "iteration " << (iter + 1) << "/" << args.iterations << "\n";

      if (args.opencv_overload) {
#if defined(SIMA_PCIE_HAS_OPENCV_OVERLOAD)
        std::cout << "push image with OpenCV overload...\n";
        if (!host.push(frame)) {
          throw std::runtime_error("push returned false");
        }
#endif
      } else {
        pcie::Tensor image = make_bgr_image_tensor(frame);
        print_image(image);
        std::cout << "push image tensor...\n";
        if (!host.push(image)) {
          throw std::runtime_error("push returned false");
        }
      }

      std::cout << "pull outputs with timeout_ms=" << args.pull_timeout_ms << "...\n";
      const auto result = host.pull(args.pull_timeout_ms);
      if (!result.has_value()) {
        throw std::runtime_error("pull timed out without a result");
      }
      print_outputs(*result);
      validate_bbox_payload(*result, frame.cols, frame.rows, args.score_threshold, args.top_k,
                            args.require_detection, args.require_person);
    }

    std::cout << "stopping...\n";
    host.stop();
    print_status("final status", host.status());
    std::cout << "done\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}

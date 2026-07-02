#include <simaai/neat/pcie/SimaPCIeHost.h>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
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
  std::string image = env_or_default("SIMAPCIE_TEST_IMAGE", DEFAULT_SOURCE_IMAGE);
  std::string card_host = env_or_default("SIMAPCIE_CARD_HOST", "");
  std::string user = env_or_default("SIMAPCIE_USER", "sima");
  int card_id = env_int_or_default("SIMAPCIE_CARD_ID", 0);
  int queue = env_int_or_default("SIMAPCIE_QUEUE", 0);
  int readiness_timeout_ms = env_int_or_default("SIMAPCIE_READINESS_TIMEOUT_MS", 180000);
  int pull_timeout_ms = env_int_or_default("SIMAPCIE_PULL_TIMEOUT_MS", 30000);
  std::string card_env = env_or_default("SIMAPCIE_CARD_ENV", "");
  std::string card_gst_debug = env_or_default("SIMAPCIE_CARD_GST_DEBUG", "");
  std::string card_gst_debug_file = env_or_default("SIMAPCIE_CARD_GST_DEBUG_FILE", "");
  bool opencv_overload = false;
};

std::string require_value(int argc, char** argv, int& i, const char* name) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + name);
  }
  return argv[++i];
}

void usage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " [--model model.tar.gz] [--image image.jpg] [--card-host host]"
               " [--card-id n] [--user user] [--queue n]"
               " [--readiness-timeout-ms ms] [--pull-timeout-ms ms]"
               " [--card-env 'NAME=VALUE ...']"
               " [--card-gst-debug spec] [--card-gst-debug-file path]"
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
    } else if (arg == "--card-env") {
      args.card_env = require_value(argc, argv, i, "--card-env");
    } else if (arg == "--card-gst-debug") {
      args.card_gst_debug = require_value(argc, argv, i, "--card-gst-debug");
    } else if (arg == "--card-gst-debug-file") {
      args.card_gst_debug_file = require_value(argc, argv, i, "--card-gst-debug-file");
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

    std::cout << "PCIe image run test\n";
    std::cout << "  model=" << args.model << "\n";
    std::cout << "  image=" << args.image << "\n";
    std::cout << "  push_mode=" << (args.opencv_overload ? "opencv-overload" : "manual-bgr-tensor")
              << "\n";
    std::cout << "  card_host="
              << (conn.card_host.empty() ? ("10.0." + std::to_string(conn.card_id) + ".2")
                                         : conn.card_host)
              << " card_id=" << conn.card_id << " user=" << conn.user << " queue=" << conn.queue
              << "\n";
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

    if (args.opencv_overload) {
#if defined(SIMA_PCIE_HAS_OPENCV_OVERLOAD)
      std::cout << "push image with OpenCV overload...\n";
      if (!host.push(bgr)) {
        throw std::runtime_error("push returned false");
      }
#endif
    } else {
      pcie::Tensor image = make_bgr_image_tensor(bgr);
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

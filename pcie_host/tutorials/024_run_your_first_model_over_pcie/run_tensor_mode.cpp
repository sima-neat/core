// Run YOLOv8s tensor-mode inference over PCIe.
//
// Usage:
//   tutorial_024_run_tensor_mode [--card 0]

#include <simaai/neat/pcie/Model.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pcie = simaai::neat::pcie;

namespace {

constexpr int kBuildTimeoutMs = 180000;
constexpr int kRunTimeoutMs = 30000;
constexpr char kModelPath[] = "yolo_v8s_mpk.tar.gz";
constexpr char kImagePath[] = "share/sima-pcie-host/tutorials/assets/street-scene.png";

int parse_card(const int argc, char** argv) {
  int card_id = 0;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--card" && index + 1 < argc) {
      card_id = std::stoi(argv[++index]);
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [--card 0]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
  }
  return card_id;
}

std::string shape_string(const std::vector<std::int64_t>& shape) {
  std::string text = "[";
  for (std::size_t index = 0; index < shape.size(); ++index) {
    text += (index == 0 ? "" : ", ") + std::to_string(shape[index]);
  }
  return text + "]";
}

const char* dtype_name(const pcie::TensorDType dtype) {
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

pcie::Tensor make_yolo_tensor(const cv::Mat& bgr, const pcie::TensorInfo& input) {
  if (input.shape.size() != 3 || input.shape[2] != 3) {
    throw std::runtime_error("expected a three-channel HWC YOLO input");
  }
  const int target_height = static_cast<int>(input.shape[0]);
  const int target_width = static_cast<int>(input.shape[1]);
  const double scale = std::min(static_cast<double>(target_width) / bgr.cols,
                                static_cast<double>(target_height) / bgr.rows);
  const int resized_width = std::max(1, static_cast<int>(std::round(bgr.cols * scale)));
  const int resized_height = std::max(1, static_cast<int>(std::round(bgr.rows * scale)));

  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(resized_width, resized_height));
  cv::Mat letterboxed(target_height, target_width, CV_8UC3, cv::Scalar(114, 114, 114));
  const int left = (target_width - resized_width) / 2;
  const int top = (target_height - resized_height) / 2;
  resized.copyTo(letterboxed(cv::Rect(left, top, resized_width, resized_height)));

  cv::Mat rgb;
  cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
  cv::Mat normalized;
  rgb.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
  if (!normalized.isContinuous()) {
    normalized = normalized.clone();
  }
  const auto* begin = normalized.ptr<float>();
  std::vector<float> values(begin, begin + normalized.total() * normalized.channels());
  return pcie::Tensor::from_vector(std::move(values), input.shape, input.name);
}

} // namespace

int main(int argc, char** argv) {
  try {
    const int card_id = parse_card(argc, argv);
    if (!std::filesystem::is_regular_file(kModelPath)) {
      throw std::runtime_error(std::string("model does not exist: ") + kModelPath);
    }
    const cv::Mat image = cv::imread(kImagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error(std::string("OpenCV could not decode: ") + kImagePath);
    }

    // CORE LOGIC
    // STEP tensor-mode
    pcie::ConnectionOptions connection;
    connection.card_id = card_id;
    pcie::Model model(kModelPath, {}, connection);
    const auto info = model.info();
    if (info.inputs.size() != 1) {
      throw std::runtime_error("YOLOv8s must expose one input tensor");
    }
    const pcie::Tensor input = make_yolo_tensor(image, info.inputs[0]);
    model.build(kBuildTimeoutMs);
    const auto outputs = model.run(input, kRunTimeoutMs);
    if (outputs.empty()) {
      throw std::runtime_error("tensor mode returned no outputs");
    }
    std::cout << "Tensor mode raw outputs:\n";
    for (const auto& output : outputs) {
      std::cout << "  " << output.route.name << " " << dtype_name(output.dtype) << " "
                << shape_string(output.shape) << '\n';
    }
    model.close();
    // END STEP
    // END CORE LOGIC

    std::cout << "[OK] 024_run_tensor_mode\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}

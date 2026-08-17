// Run YOLOv8s image-mode inference over PCIe.
//
// Usage:
//   tutorial_024_run_image_mode [--card 0]

#include <simaai/neat/pcie/Model.h>

#include <opencv2/imgcodecs.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
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

pcie::ModelOptions image_options() {
  pcie::ModelOptions options;
  options.preprocess.kind = pcie::InputKind::Image;
  options.preprocess.color_convert.input_format = pcie::ColorFormat::BGR;
  options.preprocess.color_convert.output_format = pcie::ColorFormat::RGB;
  options.preprocess.resize.enable = pcie::AutoFlag::On;
  options.preprocess.resize.mode = pcie::ResizeMode::Letterbox;
  options.preprocess.normalize.preset = pcie::NormalizePreset::COCO_YOLO;
  return options;
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
    // STEP image-mode
    pcie::ConnectionOptions connection;
    connection.card_id = card_id;
    pcie::Model model(kModelPath, image_options(), connection);
    model.build(kBuildTimeoutMs);
    const auto outputs = model.run(image, kRunTimeoutMs);
    if (outputs.empty()) {
      throw std::runtime_error("image mode returned no outputs");
    }
    std::cout << "Image mode raw outputs:\n";
    for (const auto& output : outputs) {
      std::cout << "  " << output.route.name << " " << dtype_name(output.dtype) << " "
                << shape_string(output.shape) << '\n';
    }
    model.close();
    // END STEP
    // END CORE LOGIC

    std::cout << "[OK] 024_run_image_mode\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}

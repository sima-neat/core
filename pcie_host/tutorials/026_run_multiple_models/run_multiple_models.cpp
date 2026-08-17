// Run ResNet-50 and YOLOv8s concurrently on two PCIe queues.
//
// Usage:
//   tutorial_026_run_multiple_models

#include <simaai/neat/pcie/Model.h>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pcie = simaai::neat::pcie;

namespace {

constexpr int kBuildTimeoutMs = 180000;
constexpr int kRunTimeoutMs = 30000;
constexpr int kResnetQueue = 0;
constexpr int kYoloQueue = 1;
constexpr char kResnetModelPath[] = "resnet_50_mpk.tar.gz";
constexpr char kYoloModelPath[] = "yolo_v8s_mpk.tar.gz";
constexpr char kResnetImagePath[] = "share/sima-pcie-host/tutorials/assets/labrador.jpg";
constexpr char kYoloImagePath[] = "share/sima-pcie-host/tutorials/assets/street-scene.png";

struct Args {
  int card_id = 0;
};

std::string require_value(int argc, char** argv, int& index, const char* option) {
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + option);
  }
  return argv[++index];
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--card") {
      args.card_id = std::stoi(require_value(argc, argv, index, "--card"));
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [--card 0]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return args;
}

pcie::ConnectionOptions connection_for(const Args& args, const int queue) {
  pcie::ConnectionOptions connection;
  connection.card_id = args.card_id;
  connection.queue = queue;
  return connection;
}

pcie::ModelOptions classification_options() {
  pcie::ModelOptions options;
  options.preprocess.kind = pcie::InputKind::Image;
  options.preprocess.color_convert.input_format = pcie::ColorFormat::BGR;
  options.preprocess.color_convert.output_format = pcie::ColorFormat::RGB;
  options.preprocess.resize.enable = pcie::AutoFlag::On;
  options.preprocess.resize.mode = pcie::ResizeMode::Stretch;
  options.preprocess.normalize.preset = pcie::NormalizePreset::ImageNet;
  return options;
}

pcie::ModelOptions detection_options() {
  pcie::ModelOptions options;
  options.preprocess.kind = pcie::InputKind::Image;
  options.preprocess.color_convert.input_format = pcie::ColorFormat::BGR;
  options.preprocess.color_convert.output_format = pcie::ColorFormat::RGB;
  options.preprocess.resize.enable = pcie::AutoFlag::On;
  options.preprocess.resize.mode = pcie::ResizeMode::Letterbox;
  options.preprocess.normalize.preset = pcie::NormalizePreset::COCO_YOLO;
  options.decode_type = pcie::BoxDecodeType::YoloV8;
  options.score_threshold = 0.25F;
  options.nms_iou_threshold = 0.45F;
  options.top_k = 100;
  return options;
}

std::string shape_string(const std::vector<std::int64_t>& shape) {
  std::string text = "[";
  for (std::size_t index = 0; index < shape.size(); ++index) {
    text += (index == 0 ? "" : ", ") + std::to_string(shape[index]);
  }
  return text + "]";
}

int top_class(const pcie::TensorList& outputs) {
  if (outputs.size() != 1 || outputs[0].dtype != pcie::TensorDType::Float32 ||
      outputs[0].data == nullptr || outputs[0].byte_offset < 0) {
    throw std::runtime_error("ResNet-50 must return one populated FP32 tensor");
  }
  const auto& output = outputs[0];
  const auto offset = static_cast<std::size_t>(output.byte_offset);
  if (offset > output.size_bytes || (output.size_bytes - offset) % sizeof(float) != 0) {
    throw std::runtime_error("ResNet-50 returned an invalid output span");
  }
  const auto* scores =
      reinterpret_cast<const float*>(static_cast<const std::uint8_t*>(output.data) + offset);
  const std::size_t count = (output.size_bytes - offset) / sizeof(float);
  return static_cast<int>(std::distance(scores, std::max_element(scores, scores + count)));
}

struct Box {
  int x;
  int y;
  int width;
  int height;
  float score;
  int class_id;
};

template <typename T> T read_value(const std::uint8_t* data) {
  T value{};
  std::memcpy(&value, data, sizeof(value));
  return value;
}

std::vector<Box> parse_boxes(const pcie::TensorList& outputs) {
  if (outputs.size() != 1 || outputs[0].data == nullptr || outputs[0].byte_offset < 0) {
    throw std::runtime_error("YOLOv8 boxdecode must return one populated BBOX tensor");
  }
  const auto& tensor = outputs[0];
  const auto offset = static_cast<std::size_t>(tensor.byte_offset);
  if (offset > tensor.size_bytes || tensor.size_bytes - offset < 4) {
    throw std::runtime_error("BBOX tensor is too small");
  }
  const auto* bytes = static_cast<const std::uint8_t*>(tensor.data) + offset;
  const std::size_t available = tensor.size_bytes - offset;
  const auto count = read_value<std::uint32_t>(bytes);
  constexpr std::size_t record_size = 24;
  if (count > (available - 4) / record_size) {
    throw std::runtime_error("BBOX detection count exceeds its payload");
  }
  std::vector<Box> boxes;
  boxes.reserve(count);
  for (std::uint32_t index = 0; index < count; ++index) {
    const auto* record = bytes + 4 + index * record_size;
    boxes.push_back({read_value<std::int32_t>(record), read_value<std::int32_t>(record + 4),
                     read_value<std::int32_t>(record + 8), read_value<std::int32_t>(record + 12),
                     read_value<float>(record + 16), read_value<std::int32_t>(record + 20)});
  }
  return boxes;
}

std::string class_name(const int class_id) {
  switch (class_id) {
  case 0:
    return "person";
  case 1:
    return "bicycle";
  case 2:
    return "car";
  case 3:
    return "motorcycle";
  case 5:
    return "bus";
  case 7:
    return "truck";
  default:
    return "class_" + std::to_string(class_id);
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    // STEP load-assets
    const Args args = parse_args(argc, argv);
    for (const auto* model : {kResnetModelPath, kYoloModelPath}) {
      if (!std::filesystem::is_regular_file(model)) {
        throw std::runtime_error(std::string("model does not exist: ") + model);
      }
    }
    const cv::Mat labrador = cv::imread(kResnetImagePath, cv::IMREAD_COLOR);
    const cv::Mat street = cv::imread(kYoloImagePath, cv::IMREAD_COLOR);
    if (labrador.empty() || street.empty()) {
      throw std::runtime_error("OpenCV could not decode one of the input images");
    }
    // END STEP

    // STEP assign-queues
    pcie::Model resnet(kResnetModelPath, classification_options(),
                       connection_for(args, kResnetQueue));
    pcie::Model yolo(kYoloModelPath, detection_options(), connection_for(args, kYoloQueue));
    try {
      resnet.build(kBuildTimeoutMs);
    } catch (const std::exception& error) {
      throw std::runtime_error("queue " + std::to_string(kResnetQueue) +
                               " failed to build ResNet-50: " + error.what());
    }
    try {
      yolo.build(kBuildTimeoutMs);
    } catch (const std::exception& error) {
      resnet.close();
      throw std::runtime_error("queue " + std::to_string(kYoloQueue) +
                               " failed to build YOLOv8s: " + error.what());
    }
    // END STEP

    // CORE LOGIC
    // STEP run-concurrently
    pcie::TensorList classification;
    pcie::TensorList detections;
    try {
      auto classification_future =
          std::async(std::launch::async, [&] { return resnet.run(labrador, kRunTimeoutMs); });
      auto detection_future =
          std::async(std::launch::async, [&] { return yolo.run(street, kRunTimeoutMs); });
      classification = classification_future.get();
      detections = detection_future.get();
    } catch (...) {
      yolo.close();
      resnet.close();
      throw;
    }
    // END STEP

    // STEP read-results
    const int top1 = top_class(classification);
    const auto boxes = parse_boxes(detections);
    std::cout << "queue=" << kResnetQueue
              << " model=resnet_50 output_shape=" << shape_string(classification[0].shape)
              << " top1=" << top1;
    if (top1 == 208) {
      std::cout << " (Labrador retriever)";
    }
    std::cout << '\n';
    std::cout << "queue=" << kYoloQueue << " model=yolo_v8s detections=" << boxes.size() << '\n';
    for (std::size_t index = 0; index < std::min<std::size_t>(boxes.size(), 5); ++index) {
      const auto& box = boxes[index];
      std::cout << "  " << class_name(box.class_id) << " score=" << std::fixed
                << std::setprecision(3) << box.score << " box=(" << box.x << ", " << box.y << ", "
                << box.width << ", " << box.height << ")\n";
    }
    if (boxes.empty()) {
      throw std::runtime_error("YOLOv8s returned no street-scene detections");
    }
    // END STEP
    // END CORE LOGIC

    yolo.close();
    resnet.close();
    std::cout << "[OK] 026_run_multiple_models\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}

// Run YOLOv8s image inference with card-side box decode over PCIe.
//
// Usage:
//   tutorial_024_run_image_boxdecode [--card 0]

#include <simaai/neat/pcie/Model.h>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
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
    throw std::runtime_error("boxdecode must return one populated BBOX tensor");
  }
  const auto& tensor = outputs[0];
  const auto offset = static_cast<std::size_t>(tensor.byte_offset);
  if (offset > tensor.size_bytes || tensor.size_bytes - offset < sizeof(std::uint32_t)) {
    throw std::runtime_error("BBOX tensor is too small");
  }
  const auto* bytes = static_cast<const std::uint8_t*>(tensor.data) + offset;
  const std::size_t available = tensor.size_bytes - offset;
  const std::uint32_t count = read_value<std::uint32_t>(bytes);
  constexpr std::size_t record_size = 24;
  if (count > (available - 4) / record_size) {
    throw std::runtime_error("BBOX detection count exceeds its payload");
  }
  std::vector<Box> boxes;
  boxes.reserve(count);
  for (std::uint32_t index = 0; index < count; ++index) {
    const auto* record = bytes + 4 + index * record_size;
    boxes.push_back({read_value<std::int32_t>(record), read_value<std::int32_t>(record + 4),
                     read_value<std::int32_t>(record + 8),
                     read_value<std::int32_t>(record + 12), read_value<float>(record + 16),
                     read_value<std::int32_t>(record + 20)});
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
    const int card_id = parse_card(argc, argv);
    if (!std::filesystem::is_regular_file(kModelPath)) {
      throw std::runtime_error(std::string("model does not exist: ") + kModelPath);
    }
    const cv::Mat image = cv::imread(kImagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error(std::string("OpenCV could not decode: ") + kImagePath);
    }

    // CORE LOGIC
    // STEP decode-boxes
    pcie::ConnectionOptions connection;
    connection.card_id = card_id;
    pcie::Model model(kModelPath, detection_options(), connection);
    model.build(kBuildTimeoutMs);
    const auto boxes = parse_boxes(model.run(image, kRunTimeoutMs));
    std::cout << "Image + boxdecode detections=" << boxes.size() << '\n';
    for (std::size_t index = 0; index < std::min<std::size_t>(boxes.size(), 10); ++index) {
      const auto& box = boxes[index];
      std::cout << "  " << class_name(box.class_id) << " score=" << std::fixed
                << std::setprecision(3) << box.score << " box=(" << box.x << ", " << box.y << ", "
                << box.width << ", " << box.height << ")\n";
    }
    if (boxes.empty()) {
      throw std::runtime_error("no detections passed the score threshold");
    }
    model.close();
    // END STEP
    // END CORE LOGIC

    std::cout << "[OK] 024_run_image_boxdecode\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}

// Run a model from a MIPI/libcamera camera source.
//
// Usage:
//   tutorial_023_run_mipi_camera_model --model /path/to/model.tar.gz [--frames 5]

#include <neat.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace neat = simaai::neat;
namespace fs = std::filesystem;

namespace {

bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i])
      return true;
  }
  return false;
}

int int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string value;
  if (!get_arg(argc, argv, key, value))
    return def;
  return std::stoi(value);
}

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

neat::BoxDecodeType decode_type_from_token(const std::string& token) {
  const std::string v = lower_copy(token);
  if (v.empty() || v == "none" || v == "raw")
    return neat::BoxDecodeType::Unspecified;
  if (v == "yolo")
    return neat::BoxDecodeType::Yolo;
  if (v == "yolov5")
    return neat::BoxDecodeType::YoloV5;
  if (v == "yolov8")
    return neat::BoxDecodeType::YoloV8;
  if (v == "yolov8seg" || v == "yolov8-seg")
    return neat::BoxDecodeType::YoloV8Seg;
  if (v == "yolov9")
    return neat::BoxDecodeType::YoloV9;
  if (v == "yolov9seg" || v == "yolov9-seg")
    return neat::BoxDecodeType::YoloV9Seg;
  throw std::runtime_error("unsupported --decode token: " + token);
}

template <typename Shape>
std::string shape_string(const Shape& shape) {
  std::string out = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    out += std::to_string(shape[i]);
    if (i + 1 < shape.size())
      out += ",";
  }
  out += "]";
  return out;
}

neat::Model::Options model_options_for_camera(const neat::CameraInputOptions& camera,
                                              neat::BoxDecodeType decode_type) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.input_max_width = static_cast<int>(camera.width);
  options.preprocess.input_max_height = static_cast<int>(camera.height);
  options.preprocess.input_max_depth = 3;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  options.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
  options.preprocess.resize.enable = neat::AutoFlag::On;
  options.preprocess.resize.width = 640;
  options.preprocess.resize.height = 640;
  options.preprocess.resize.mode = neat::ResizeMode::Letterbox;
  options.preprocess.resize.pad_value = 114;
  options.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  options.advanced_execution.preprocess_target = "EV74";
  options.decode_type = decode_type;
  if (decode_type == neat::BoxDecodeType::Unspecified) {
    options.inference_terminal.mla_only = true;
  } else {
    options.advanced_execution.postprocess_target = "EV74";
    options.score_threshold = 0.25f;
    options.nms_iou_threshold = 0.45f;
    options.top_k = 100;
  }
  return options;
}

void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " --model <model.tar.gz> [--frames 5] [--width 1920] [--height 1080] "
               "[--fps 30] [--camera-name NAME] [--decode none|yolov8|yolov9seg] "
               "[--pull-timeout-ms 15000] [--strict-zero-copy] [--print-backend]\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::string model_path;
    if (!get_arg(argc, argv, "--model", model_path)) {
      usage(argv[0]);
      return 1;
    }
    if (!fs::exists(model_path))
      throw std::runtime_error("model archive not found: " + model_path);

    const int frames = int_arg(argc, argv, "--frames", 5);
    if (frames <= 0)
      throw std::runtime_error("--frames must be positive");
    const int pull_timeout_ms = int_arg(argc, argv, "--pull-timeout-ms", 15000);
    if (pull_timeout_ms <= 0)
      throw std::runtime_error("--pull-timeout-ms must be positive");

    std::string decode_token = "none";
    get_arg(argc, argv, "--decode", decode_token);

    // CORE LOGIC
    // STEP configure-camera
    neat::CameraInputOptions camera;
    camera.width = static_cast<std::uint32_t>(int_arg(argc, argv, "--width", 1920));
    camera.height = static_cast<std::uint32_t>(int_arg(argc, argv, "--height", 1080));
    camera.framerate_num = static_cast<std::uint32_t>(int_arg(argc, argv, "--fps", 30));
    camera.framerate_den = 1;
    camera.format = "NV12";
    camera.buffer_name = "camera0";
    camera.allow_cpu_fallback = !has_flag(argc, argv, "--strict-zero-copy");
    std::string camera_name;
    if (get_arg(argc, argv, "--camera-name", camera_name)) {
      camera.camera_name = camera_name;
    }
    // END STEP

    // STEP configure-model
    const neat::BoxDecodeType decode_type = decode_type_from_token(decode_token);
    neat::Model model(model_path, model_options_for_camera(camera, decode_type));

    neat::Model::RouteOptions route;
    route.include_input = false;
    route.include_output = true;
    route.upstream_name = camera.buffer_name;
    route.buffer_name = camera.buffer_name;
    route.name_suffix = "_camera0";
    route.advanced_execution.preprocess_target = "EV74";
    if (decode_type != neat::BoxDecodeType::Unspecified) {
      route.advanced_execution.postprocess_target = "EV74";
    }
    // END STEP

    // STEP compose-graph
    neat::Graph graph("mipi_camera_model");
    graph.add(neat::nodes::CameraInput(camera));
    graph.add(model.graph(route));

    if (has_flag(argc, argv, "--print-backend")) {
      std::cout << graph.describe_backend(false) << "\n";
    }

    neat::Run run = graph.build();
    // END STEP

    // STEP pull-output
    for (int i = 0; i < frames; ++i) {
      std::optional<neat::Sample> sample = run.pull(/*timeout_ms=*/pull_timeout_ms);
      if (!sample.has_value()) {
        std::cout << "frame=" << i << " output_timeout timeout_ms=" << pull_timeout_ms;
        const std::string last_error = run.last_error();
        if (!last_error.empty())
          std::cout << " last_error=" << last_error;
        std::cout << "\n";
        return 2;
      }
      const neat::TensorList tensors = neat::tensors_from_sample(*sample, true);
      std::cout << "frame=" << i << " tensors=" << tensors.size();
      if (!tensors.empty())
        std::cout << " first_shape=" << shape_string(tensors.front().shape);
      std::cout << "\n";
    }
    // END STEP
    // END CORE LOGIC

    std::cout << "[OK] 023_run_mipi_camera_model\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

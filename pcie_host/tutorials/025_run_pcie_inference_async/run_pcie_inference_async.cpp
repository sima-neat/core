// Measure completed YOLOv8s detections with asynchronous PCIe push/pull.
//
// Usage:
//   tutorial_025_run_pcie_inference_async

#include <simaai/neat/pcie/Model.h>

#include <opencv2/imgcodecs.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace pcie = simaai::neat::pcie;

namespace {

using Clock = std::chrono::steady_clock;
constexpr int kBuildTimeoutMs = 180000;
constexpr int kPullTimeoutMs = 30000;
constexpr int kWarmupFrames = 5;
constexpr int kMeasuredFrames = 1000;
constexpr char kModelPath[] = "yolo_v8s_mpk.tar.gz";
constexpr char kImagePath[] = "share/sima-pcie-host/tutorials/assets/street-scene.png";

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

std::uint32_t detection_count(const pcie::TensorList& outputs) {
  if (outputs.size() != 1 || outputs[0].data == nullptr || outputs[0].byte_offset < 0) {
    throw std::runtime_error("boxdecode must return one populated BBOX tensor");
  }
  const auto& tensor = outputs[0];
  const auto offset = static_cast<std::size_t>(tensor.byte_offset);
  if (offset > tensor.size_bytes || tensor.size_bytes - offset < sizeof(std::uint32_t)) {
    throw std::runtime_error("BBOX tensor is too small");
  }
  std::uint32_t count = 0;
  std::memcpy(&count, static_cast<const std::uint8_t*>(tensor.data) + offset, sizeof(count));
  constexpr std::size_t record_size = 24;
  if (count > (tensor.size_bytes - offset - 4) / record_size) {
    throw std::runtime_error("BBOX detection count exceeds its payload");
  }
  return count;
}

struct BenchmarkResult {
  std::size_t completed = 0;
  double elapsed_seconds = 0.0;
  double average_latency_ms = 0.0;
  std::uint64_t total_detections = 0;
};

BenchmarkResult measure(pcie::Model& model, const cv::Mat& image, const int frame_count) {
  std::deque<Clock::time_point> submitted;
  std::mutex submitted_mutex;
  std::mutex failure_mutex;
  std::exception_ptr first_failure;
  std::atomic<bool> cancelled = false;
  std::vector<double> latency_ms;
  latency_ms.reserve(static_cast<std::size_t>(frame_count));
  std::uint64_t total_detections = 0;

  auto fail = [&](std::exception_ptr failure) {
    {
      std::lock_guard<std::mutex> lock(failure_mutex);
      if (!first_failure) {
        first_failure = std::move(failure);
      }
    }
    cancelled = true;
    model.close();
  };

  const auto benchmark_start = Clock::now();
  std::thread producer([&] {
    try {
      for (int index = 0; index < frame_count && !cancelled; ++index) {
        const auto started = Clock::now();
        {
          std::lock_guard<std::mutex> lock(submitted_mutex);
          submitted.push_back(started);
        }
        if (!model.push(image)) {
          throw std::runtime_error("push rejected frame " + std::to_string(index));
        }
      }
    } catch (...) {
      fail(std::current_exception());
    }
  });

  std::thread consumer([&] {
    try {
      for (int index = 0; index < frame_count && !cancelled; ++index) {
        auto outputs = model.pull(kPullTimeoutMs);
        if (!outputs) {
          throw std::runtime_error("pull timed out for frame " + std::to_string(index));
        }
        Clock::time_point started;
        {
          std::lock_guard<std::mutex> lock(submitted_mutex);
          if (submitted.empty()) {
            throw std::runtime_error("completion arrived without a submission record");
          }
          started = submitted.front();
          submitted.pop_front();
        }
        total_detections += detection_count(*outputs);
        latency_ms.push_back(
            std::chrono::duration<double, std::milli>(Clock::now() - started).count());
      }
    } catch (...) {
      fail(std::current_exception());
    }
  });

  producer.join();
  consumer.join();
  const auto benchmark_end = Clock::now();
  if (first_failure) {
    std::rethrow_exception(first_failure);
  }
  if (latency_ms.size() != static_cast<std::size_t>(frame_count)) {
    throw std::runtime_error("not every submitted frame completed");
  }

  BenchmarkResult result;
  result.completed = latency_ms.size();
  result.elapsed_seconds = std::chrono::duration<double>(benchmark_end - benchmark_start).count();
  result.average_latency_ms =
      std::accumulate(latency_ms.begin(), latency_ms.end(), 0.0) / latency_ms.size();
  result.total_detections = total_detections;
  return result;
}

} // namespace

int main(int argc, char** argv) {
  try {
    // STEP configure-model
    const Args args = parse_args(argc, argv);
    if (!std::filesystem::is_regular_file(kModelPath)) {
      throw std::runtime_error(std::string("model does not exist: ") + kModelPath);
    }
    const cv::Mat image = cv::imread(kImagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error(std::string("OpenCV could not decode: ") + kImagePath);
    }
    pcie::ConnectionOptions connection;
    connection.card_id = args.card_id;
    pcie::Model model(kModelPath, detection_options(), connection);
    model.build(kBuildTimeoutMs);
    // END STEP

    // STEP warm-up
    for (int index = 0; index < kWarmupFrames; ++index) {
      (void)detection_count(model.run(image, kPullTimeoutMs));
    }
    // END STEP

    // CORE LOGIC
    // STEP push-pull
    const BenchmarkResult result = measure(model, image, kMeasuredFrames);
    // END STEP

    // STEP report-results
    std::cout << "completed=" << result.completed << '\n';
    std::cout << std::fixed << std::setprecision(2) << "elapsed_seconds=" << result.elapsed_seconds
              << '\n'
              << "throughput_fps=" << result.completed / result.elapsed_seconds << '\n'
              << "average_latency_ms=" << result.average_latency_ms << '\n'
              << "total_detections=" << result.total_detections << '\n';
    // END STEP
    // END CORE LOGIC

    model.close();
    std::cout << "[OK] 025_run_pcie_inference_async\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}

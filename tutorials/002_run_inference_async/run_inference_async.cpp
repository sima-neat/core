// Async push/pull: producer thread pushes frames, main thread pulls outputs.
//
// Usage:
//   tutorial_002_run_inference_async --model /path/to/resnet_50.tar.gz [--image /path/to.jpg] [--n
//   4]

#include "neat.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

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

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string value;
  if (!get_arg(argc, argv, key, value))
    return def;
  return std::stoi(value);
}

cv::Mat load_rgb(const fs::path& image_path, int size) {
  cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  if (bgr.empty())
    throw std::runtime_error("failed to read image: " + image_path.string());
  if (bgr.cols != size || bgr.rows != size) {
    cv::resize(bgr, bgr, cv::Size(size, size), 0, 0, cv::INTER_AREA);
  }
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  if (!rgb.isContinuous())
    rgb = rgb.clone();
  return rgb;
}

simaai::neat::Model::Options build_options(int size) {
  simaai::neat::Model::Options opt;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::RGB;
  opt.preprocess.input_max_width = size;
  opt.preprocess.input_max_height = size;
  opt.preprocess.input_max_depth = 3;
  opt.preprocess.normalize.mean = {0.485f, 0.456f, 0.406f};
  opt.preprocess.normalize.stddev = {0.229f, 0.224f, 0.225f};
  return opt;
}

int top1_from_output(const simaai::neat::Sample& out) {
  if (simaai::neat::tensors_from_sample(out, true).empty())
    throw std::runtime_error("no tensor output");
  const simaai::neat::Mapping m = simaai::neat::tensors_from_sample(out, true).front().map_read();
  const size_t n = m.size_bytes / sizeof(float);
  const float* p = reinterpret_cast<const float*>(m.data);
  int best = 0;
  for (size_t i = 1; i < n && i < 1000; ++i) {
    if (p[i] > p[best])
      best = static_cast<int>(i);
  }
  return best;
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::string model_path, image;
    if (!get_arg(argc, argv, "--model", model_path)) {
      std::cerr
          << "Usage: tutorial_002_run_inference_async --model <path> [--image <path>] [--n <n>]\n";
      return 1;
    }
    get_arg(argc, argv, "--image", image);
    const int n = parse_int_arg(argc, argv, "--n", 4);
    const int size = 224;

    cv::Mat frame = image.empty() ? cv::Mat(size, size, CV_8UC3, cv::Scalar(99, 99, 99))
                                  : load_rgb(image, size);
    std::vector<cv::Mat> frames(n, frame);

    // CORE LOGIC
    // Build a Graph around the model and run it async: one producer thread pushes,
    // the main thread pulls outputs.
    // STEP load-model
    simaai::neat::Model model(model_path, build_options(size));
    simaai::neat::Model::RouteOptions route_opt;
    route_opt.include_input = true;
    route_opt.include_output = true;
    // END STEP

    // STEP build-async
    simaai::neat::Graph graph;
    graph.add(model.graph(route_opt));

    auto run = graph.build(std::vector<cv::Mat>{frames.front()});
    // END STEP

    // STEP push-frames
    std::atomic<int> pushed{0};
    std::atomic<bool> producer_done{false};
    std::thread producer([&]() {
      for (const cv::Mat& f : frames) {
        run.push(std::vector<cv::Mat>{f});
        pushed.fetch_add(1, std::memory_order_relaxed);
      }
      run.close_input();
      producer_done.store(true);
    });
    // END STEP

    // STEP pull-results
    int pulled = 0;
    while (pulled < n) {
      auto out = run.pull(/*timeout_ms=*/2000);
      if (!out.has_value()) {
        if (producer_done.load())
          break;
        continue;
      }
      std::cout << "top1=" << top1_from_output(*out) << "\n";
      ++pulled;
    }
    producer.join();
    // END STEP
    // END CORE LOGIC

    std::cout << "pushed=" << pushed.load() << " pulled=" << pulled << "\n";
    if (pulled != n)
      throw std::runtime_error("pulled=" + std::to_string(pulled) +
                               " != pushed=" + std::to_string(pushed.load()));
    std::cout << "[OK] 002_run_inference_async\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

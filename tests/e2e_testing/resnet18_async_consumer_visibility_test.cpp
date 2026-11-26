// Async consumer visibility on a real classifier: a two-thread push/pull run must hand each
// consumer the output of its own input. The reference is the synchronous run() result for the
// same image, so a swapped, dropped or repeated frame fails the test.
#include "cli_utils.h"
#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "model/Model.h"
#include "test_utils.h"

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::uint64_t duration_ns(std::chrono::steady_clock::time_point a,
                          std::chrono::steady_clock::time_point b) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
}

void burn_us(int usec) {
  if (usec <= 0)
    return;
  const auto start = std::chrono::steady_clock::now();
  const auto target = std::chrono::microseconds(usec);
  std::uint64_t sink = 0;
  while ((std::chrono::steady_clock::now() - start) < target) {
    ++sink;
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : "+r"(sink));
#endif
  }
}

float bf16_to_fp32(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

std::vector<double> tensor_to_scores(const simaai::neat::Tensor& tensor) {
  simaai::neat::Mapping map = tensor.map(simaai::neat::MapMode::Read);
  if (!map.data || map.size_bytes == 0) {
    throw std::runtime_error("output tensor map is empty");
  }
  std::vector<double> out;
  switch (tensor.dtype) {
  case simaai::neat::TensorDType::UInt8: {
    const auto* p = static_cast<const uint8_t*>(map.data);
    out.assign(p, p + map.size_bytes);
    break;
  }
  case simaai::neat::TensorDType::Int8: {
    const auto* p = static_cast<const int8_t*>(map.data);
    out.assign(p, p + map.size_bytes);
    break;
  }
  case simaai::neat::TensorDType::UInt16: {
    const auto* p = static_cast<const uint16_t*>(map.data);
    out.assign(p, p + map.size_bytes / sizeof(uint16_t));
    break;
  }
  case simaai::neat::TensorDType::Int16: {
    const auto* p = static_cast<const int16_t*>(map.data);
    out.assign(p, p + map.size_bytes / sizeof(int16_t));
    break;
  }
  case simaai::neat::TensorDType::Int32: {
    const auto* p = static_cast<const int32_t*>(map.data);
    out.assign(p, p + map.size_bytes / sizeof(int32_t));
    break;
  }
  case simaai::neat::TensorDType::BFloat16: {
    const auto* p = static_cast<const uint16_t*>(map.data);
    out.reserve(map.size_bytes / sizeof(uint16_t));
    for (std::size_t i = 0; i < map.size_bytes / sizeof(uint16_t); ++i)
      out.push_back(static_cast<double>(bf16_to_fp32(p[i])));
    break;
  }
  case simaai::neat::TensorDType::Float32: {
    const auto* p = static_cast<const float*>(map.data);
    out.assign(p, p + map.size_bytes / sizeof(float));
    break;
  }
  case simaai::neat::TensorDType::Float64: {
    const auto* p = static_cast<const double*>(map.data);
    out.assign(p, p + map.size_bytes / sizeof(double));
    break;
  }
  }
  if (out.size() < 1000) {
    throw std::runtime_error("expected 1000 ImageNet scores, got " + std::to_string(out.size()));
  }
  out.resize(1000);
  return out;
}

int argmax(const std::vector<double>& scores) {
  return static_cast<int>(std::max_element(scores.begin(), scores.end()) - scores.begin());
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double diff = 0.0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    diff = std::max(diff, std::abs(a[i] - b[i]));
  return diff;
}

#if defined(SIMA_WITH_OPENCV)
// Distinct, deterministic BGR images: seeded noise, gradients, checkerboards and solid colours.
std::vector<cv::Mat> make_images(int count) {
  std::vector<cv::Mat> images;
  images.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    cv::Mat img(224, 224, CV_8UC3);
    switch (i % 4) {
    case 0: {
      cv::RNG rng(static_cast<uint64_t>(1000 + i));
      rng.fill(img, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(256));
      break;
    }
    case 1:
      for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
          img.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<uchar>((x + 7 * i) & 0xFF),
                                              static_cast<uchar>((y * 2 + 13 * i) & 0xFF),
                                              static_cast<uchar>((x + y + 31 * i) & 0xFF));
      break;
    case 2: {
      const int cell = 8 + 4 * (i % 5);
      for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) {
          const bool on = ((x / cell) + (y / cell)) % 2 == 0;
          img.at<cv::Vec3b>(y, x) = on ? cv::Vec3b(static_cast<uchar>(40 * i % 256), 220, 30)
                                       : cv::Vec3b(10, static_cast<uchar>(17 * i % 256), 200);
        }
      break;
    }
    default:
      img.setTo(cv::Scalar((53 * i) % 256, (97 * i) % 256, (151 * i) % 256));
      cv::circle(img, cv::Point(112, 112), 30 + 5 * (i % 7), cv::Scalar(255, 255, 255), -1);
      break;
    }
    images.push_back(img);
  }
  return images;
}

struct Result {
  std::string mode;
  int total = 0;
  std::uint64_t wall_ns = 0;
};

void print_result(const Result& r) {
  const double sec = static_cast<double>(r.wall_ns) / 1e9;
  const double fps = (sec > 0.0) ? static_cast<double>(r.total) / sec : 0.0;
  std::cout << std::fixed << std::setprecision(3) << "[RESULT] mode=" << r.mode
            << " total=" << r.total << " wall_ms=" << (static_cast<double>(r.wall_ns) / 1e6)
            << " fps=" << fps << "\n";
}

Result run_sync_inline(simaai::neat::Model& model, const std::vector<cv::Mat>& images,
                       int timeout_ms, std::vector<std::vector<double>>* scores_out) {
  auto runner = model.build(std::vector<cv::Mat>{images.front()});
  Result r;
  r.mode = "sync_inline_run_read";
  const auto start = std::chrono::steady_clock::now();
  for (const cv::Mat& image : images) {
    simaai::neat::TensorList outs = runner.run(std::vector<cv::Mat>{image}, timeout_ms);
    if (outs.empty())
      throw std::runtime_error("sync run returned no output at " + std::to_string(r.total));
    scores_out->push_back(tensor_to_scores(outs.front()));
    ++r.total;
  }
  r.wall_ns = duration_ns(start, std::chrono::steady_clock::now());
  runner.close();
  return r;
}

// With read_outputs, each consumed output is compared with the synchronous result for the input
// it belongs to. Without it, only completeness is checked.
Result run_async_two_thread(simaai::neat::Model& model, const std::vector<cv::Mat>& images,
                            const std::vector<std::vector<double>>& reference, int timeout_ms,
                            int queue_depth, bool read_outputs, int consumer_work_us) {
  simaai::neat::RunOptions run_opt;
  run_opt.queue_depth = queue_depth;
  auto runner = model.build(std::vector<cv::Mat>{images.front()},
                            simaai::neat::Model::RouteOptions{}, run_opt);
  Result r;
  r.mode = read_outputs ? "two_thread_pull_read" : "two_thread_pull_no_read";
  const int count = static_cast<int>(images.size());

  std::atomic<bool> stop{false};
  std::mutex err_mu;
  std::string err;
  auto set_err = [&](const std::string& s) {
    std::lock_guard<std::mutex> lk(err_mu);
    if (err.empty())
      err = s;
    stop.store(true);
  };

  // input_seq is an internal transport counter. Model::build() warms the runner with a seed
  // frame first, which consumes sequence values before the push loop, so the counter is rebased
  // to the first value the consumer sees, which belongs to images[0].
  int64_t base_input_seq = -1;
  std::set<int> seen;
  const auto start = std::chrono::steady_clock::now();
  std::thread consumer([&]() {
    try {
      while (r.total < count && !stop.load()) {
        simaai::neat::Sample samples = runner.pull(timeout_ms);
        if (samples.empty()) {
          set_err("async pull returned empty at " + std::to_string(r.total));
          return;
        }
        for (auto& sample : samples) {
          if (r.total >= count)
            break;
          const int64_t seq = sample.input_seq >= 0 ? sample.input_seq : sample.orig_input_seq;
          if (seq < 0)
            throw std::runtime_error("async sample has no input sequence");
          if (base_input_seq < 0)
            base_input_seq = seq;
          const int64_t index = seq - base_input_seq;
          if (index < 0 || index >= count)
            throw std::runtime_error("async sample maps to input " + std::to_string(index) +
                                     " outside 0.." + std::to_string(count - 1));
          if (!seen.insert(static_cast<int>(index)).second)
            throw std::runtime_error("async output for input " + std::to_string(index) +
                                     " arrived twice");
          if (read_outputs) {
            simaai::neat::TensorList tensors = simaai::neat::tensors_from_sample(sample);
            if (tensors.empty())
              throw std::runtime_error("async sample has no tensors");
            const auto scores = tensor_to_scores(tensors.front());
            burn_us(consumer_work_us);
            const auto& expected = reference[static_cast<std::size_t>(index)];
            if (argmax(scores) != argmax(expected)) {
              throw std::runtime_error("async output for input " + std::to_string(index) +
                                       " has class " + std::to_string(argmax(scores)) +
                                       ", sync run gave " + std::to_string(argmax(expected)));
            }
            const double diff = max_abs_diff(scores, expected);
            if (diff > 1e-3 * std::max(1.0, std::abs(expected[argmax(expected)]))) {
              throw std::runtime_error("async output for input " + std::to_string(index) +
                                       " differs from the sync run by " + std::to_string(diff));
            }
          }
          ++r.total;
        }
      }
    } catch (const std::exception& e) {
      set_err(e.what());
    }
  });

  for (int i = 0; i < count && !stop.load(); ++i) {
    if (!runner.push(std::vector<cv::Mat>{images[static_cast<std::size_t>(i)]})) {
      set_err("async push returned false at " + std::to_string(i));
      break;
    }
  }
  runner.close_input();
  consumer.join();
  r.wall_ns = duration_ns(start, std::chrono::steady_clock::now());
  runner.close();

  if (!err.empty())
    throw std::runtime_error(r.mode + ": " + err);
  if (r.total != count || static_cast<int>(seen.size()) != count)
    throw std::runtime_error(r.mode + ": consumed " + std::to_string(r.total) + " of " +
                             std::to_string(count) + " outputs");
  return r;
}
#endif

} // namespace

int main(int argc, char** argv) {
#if !defined(SIMA_WITH_OPENCV)
  (void)argc;
  (void)argv;
  return fail_test("OpenCV required");
#else
  try {
    simaai::neat::gst_init_once();
    if (!simaai::neat::element_exists("neatkernelrunner") ||
        !simaai::neat::element_exists("neatprocessmla")) {
      return fail_test("missing SimaAI plugins");
    }

    constexpr int count = 32;
    constexpr int timeout_ms = 20000;
    constexpr int queue_depth = 4;
    constexpr int consumer_work_us = 2000;
    std::string model_path;
    if (const char* env = std::getenv("SIMA_RESNET18_TAR"); env && *env)
      model_path = env;
    std::string tmp;
    if (sima_test::get_arg(argc, argv, "--model", tmp))
      model_path = tmp;
    if (model_path.empty())
      throw std::runtime_error("SIMA_RESNET18_TAR is not set; point it at the Model Browser "
                               "resnet_18 w8a8 MPK (or pass --model)");
    std::error_code ec;
    if (!fs::is_regular_file(model_path, ec) || fs::file_size(model_path, ec) == 0)
      throw std::runtime_error("ResNet-18 MPK is missing or empty: " + model_path);

    simaai::neat::Model::Options model_opt;
    model_opt.preprocess.kind = simaai::neat::InputKind::Image;
    model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
    model_opt.preprocess.color_convert.enable = simaai::neat::AutoFlag::On;
    model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
    model_opt.preprocess.preset = simaai::neat::NormalizePreset::ImageNet;

    std::cout << "[CONTROL] model=" << model_path << " count=" << count
              << " queue_depth=" << queue_depth << " consumer_work_us=" << consumer_work_us << "\n";
    simaai::neat::Model model(model_path, model_opt);
    const std::vector<cv::Mat> images = make_images(count);

    std::vector<std::vector<double>> reference;
    print_result(run_sync_inline(model, images, timeout_ms, &reference));
    std::set<int> classes;
    for (const auto& scores : reference)
      classes.insert(argmax(scores));
    std::cout << "[CONTROL] distinct_sync_classes=" << classes.size() << "\n";
    if (classes.size() < 2)
      throw std::runtime_error("all inputs gave the same class; the visibility check needs "
                               "inputs with different results");

    print_result(run_async_two_thread(model, images, reference, timeout_ms, queue_depth, false,
                                      consumer_work_us));
    print_result(run_async_two_thread(model, images, reference, timeout_ms, queue_depth, true,
                                      consumer_work_us));
    std::cout << "[OK] resnet18_async_consumer_visibility_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
#endif
}

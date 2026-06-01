#include "asset_utils.h"
#include "cli_utils.h"
#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "model/Model.h"
#include "test_utils.h"

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core.hpp>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr const char* kDefaultModelName = "mnist_cnn";
constexpr const char* kDefaultImagesUrl =
    "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz";
constexpr const char* kDefaultLabelsUrl =
    "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz";

std::string env_string(const char* key, const std::string& fallback = {}) {
  const char* v = std::getenv(key);
  return (v && *v) ? std::string(v) : fallback;
}

int env_int_value(const char* key, int fallback) {
  const char* v = std::getenv(key);
  return (v && *v) ? std::atoi(v) : fallback;
}

bool local_env_flag(const char* key) {
  const char* v = std::getenv(key);
  if (!v || !*v)
    return false;
  const std::string s(v);
  return s != "0" && s != "false" && s != "FALSE" && s != "off" && s != "OFF";
}

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

std::vector<double> tensor_to_scores(const simaai::neat::Tensor& tensor,
                                     std::uint64_t* map_read_ns = nullptr) {
  const auto t0 = std::chrono::steady_clock::now();
  simaai::neat::Mapping map = tensor.map(simaai::neat::MapMode::Read);
  const auto t1 = std::chrono::steady_clock::now();
  if (map_read_ns)
    *map_read_ns += duration_ns(t0, t1);
  if (!map.data || map.size_bytes == 0) {
    throw std::runtime_error("output tensor map is empty");
  }

  std::vector<double> out;
  switch (tensor.dtype) {
  case simaai::neat::TensorDType::UInt8: {
    const auto* p = static_cast<const uint8_t*>(map.data);
    out.reserve(map.size_bytes);
    for (std::size_t i = 0; i < map.size_bytes; ++i)
      out.push_back(static_cast<double>(p[i]));
    break;
  }
  case simaai::neat::TensorDType::Int8: {
    const auto* p = static_cast<const int8_t*>(map.data);
    out.reserve(map.size_bytes);
    for (std::size_t i = 0; i < map.size_bytes; ++i)
      out.push_back(static_cast<double>(p[i]));
    break;
  }
  case simaai::neat::TensorDType::UInt16: {
    const auto* p = static_cast<const uint16_t*>(map.data);
    out.reserve(map.size_bytes / sizeof(uint16_t));
    for (std::size_t i = 0; i < map.size_bytes / sizeof(uint16_t); ++i)
      out.push_back(static_cast<double>(p[i]));
    break;
  }
  case simaai::neat::TensorDType::Int16: {
    const auto* p = static_cast<const int16_t*>(map.data);
    out.reserve(map.size_bytes / sizeof(int16_t));
    for (std::size_t i = 0; i < map.size_bytes / sizeof(int16_t); ++i)
      out.push_back(static_cast<double>(p[i]));
    break;
  }
  case simaai::neat::TensorDType::Int32: {
    const auto* p = static_cast<const int32_t*>(map.data);
    out.reserve(map.size_bytes / sizeof(int32_t));
    for (std::size_t i = 0; i < map.size_bytes / sizeof(int32_t); ++i)
      out.push_back(static_cast<double>(p[i]));
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
    out.reserve(map.size_bytes / sizeof(float));
    for (std::size_t i = 0; i < map.size_bytes / sizeof(float); ++i)
      out.push_back(static_cast<double>(p[i]));
    break;
  }
  case simaai::neat::TensorDType::Float64: {
    const auto* p = static_cast<const double*>(map.data);
    out.reserve(map.size_bytes / sizeof(double));
    for (std::size_t i = 0; i < map.size_bytes / sizeof(double); ++i)
      out.push_back(p[i]);
    break;
  }
  }
  if (out.size() < 10) {
    throw std::runtime_error("expected at least 10 MNIST scores, got " +
                             std::to_string(out.size()));
  }
  out.resize(10);
  return out;
}

int argmax10(const std::vector<double>& scores) {
  return static_cast<int>(std::max_element(scores.begin(), scores.begin() + 10) - scores.begin());
}

std::vector<uint8_t> read_binary_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("failed to open " + path.string());
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(static_cast<std::size_t>(size));
  in.read(reinterpret_cast<char*>(data.data()), size);
  if (!in)
    throw std::runtime_error("failed to read " + path.string());
  return data;
}

uint32_t read_be32(const std::vector<uint8_t>& data, std::size_t offset) {
  return (static_cast<uint32_t>(data[offset]) << 24) |
         (static_cast<uint32_t>(data[offset + 1]) << 16) |
         (static_cast<uint32_t>(data[offset + 2]) << 8) | static_cast<uint32_t>(data[offset + 3]);
}

bool ensure_decompressed_gzip(const fs::path& gz_path, const fs::path& out_path) {
  if (sima_test::is_usable_regular_file(out_path))
    return true;
  std::error_code ec;
  fs::create_directories(out_path.parent_path(), ec);
  const fs::path tmp_path = out_path.string() + ".tmp";
  fs::remove(tmp_path, ec);
  const std::string cmd = "gzip -dc " + sima_test::shell_quote(gz_path.string()) + " > " +
                          sima_test::shell_quote(tmp_path.string());
  if (std::system(cmd.c_str()) != 0 || !sima_test::is_usable_regular_file(tmp_path)) {
    fs::remove(tmp_path, ec);
    return false;
  }
  fs::rename(tmp_path, out_path, ec);
  if (ec) {
    fs::copy_file(tmp_path, out_path, fs::copy_options::overwrite_existing, ec);
    fs::remove(tmp_path, ec);
  }
  return sima_test::is_usable_regular_file(out_path);
}

fs::path ensure_mnist_file(const fs::path& cache_dir, const std::string& name,
                           const std::string& url) {
  const fs::path gz_path = cache_dir / (name + ".gz");
  const fs::path raw_path = cache_dir / name;
  if (!sima_test::is_usable_regular_file(gz_path) && !sima_test::download_file(url, gz_path)) {
    throw std::runtime_error("failed to download MNIST asset: " + url);
  }
  if (!ensure_decompressed_gzip(gz_path, raw_path)) {
    throw std::runtime_error("failed to decompress MNIST asset: " + gz_path.string());
  }
  return raw_path;
}

#if defined(SIMA_WITH_OPENCV)
struct MnistDataset {
  int rows = 0;
  int cols = 0;
  std::vector<cv::Mat> images;
  std::vector<uint8_t> labels;
};

MnistDataset load_mnist_dataset(const fs::path& images_path, const fs::path& labels_path,
                                int limit) {
  const auto image_bytes = read_binary_file(images_path);
  const auto label_bytes = read_binary_file(labels_path);
  const uint32_t image_count = read_be32(image_bytes, 4);
  const uint32_t rows = read_be32(image_bytes, 8);
  const uint32_t cols = read_be32(image_bytes, 12);
  const uint32_t label_count = read_be32(label_bytes, 4);
  const uint32_t available = std::min(image_count, label_count);
  if (limit <= 0 || static_cast<uint32_t>(limit) > available)
    limit = static_cast<int>(available);
  MnistDataset out;
  out.rows = static_cast<int>(rows);
  out.cols = static_cast<int>(cols);
  out.images.reserve(static_cast<std::size_t>(limit));
  out.labels.reserve(static_cast<std::size_t>(limit));
  const std::size_t image_size = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  for (int i = 0; i < limit; ++i) {
    cv::Mat img(out.rows, out.cols, CV_8UC1);
    std::memcpy(img.data, image_bytes.data() + 16 + static_cast<std::size_t>(i) * image_size,
                image_size);
    out.images.push_back(img);
    out.labels.push_back(label_bytes[8 + static_cast<std::size_t>(i)]);
  }
  return out;
}
#endif

struct Result {
  std::string mode;
  int total = 0;
  int correct = 0;
  std::uint64_t wall_ns = 0;
  std::uint64_t push_ns = 0;
  std::uint64_t pull_ns = 0;
  std::uint64_t map_ns = 0;
};

void print_result(const Result& r) {
  const double sec = static_cast<double>(r.wall_ns) / 1e9;
  const double fps = (sec > 0.0) ? static_cast<double>(r.total) / sec : 0.0;
  const double acc =
      r.total > 0 ? static_cast<double>(r.correct) / static_cast<double>(r.total) : 0.0;
  std::cout << std::fixed << std::setprecision(3) << "[RESULT] mode=" << r.mode
            << " total=" << r.total << " correct=" << r.correct << " acc=" << acc
            << " wall_ms=" << (static_cast<double>(r.wall_ns) / 1e6) << " fps=" << fps
            << " push_ms=" << (static_cast<double>(r.push_ns) / 1e6)
            << " pull_ms=" << (static_cast<double>(r.pull_ns) / 1e6)
            << " map_ms=" << (static_cast<double>(r.map_ns) / 1e6) << "\n";
}

#if defined(SIMA_WITH_OPENCV)
Result run_sync_inline(simaai::neat::Model& model, const MnistDataset& mnist, int timeout_ms,
                       int consumer_work_us) {
  auto runner = model.build(std::vector<cv::Mat>{mnist.images.front()});
  Result r;
  r.mode = "sync_inline_run_read";
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < mnist.images.size(); ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    simaai::neat::TensorList outs = runner.run(std::vector<cv::Mat>{mnist.images[i]}, timeout_ms);
    const auto t1 = std::chrono::steady_clock::now();
    r.pull_ns += duration_ns(t0, t1);
    if (outs.empty())
      throw std::runtime_error("sync returned no output");
    const auto scores = tensor_to_scores(outs.front(), &r.map_ns);
    burn_us(consumer_work_us);
    const int pred = argmax10(scores);
    if (pred == static_cast<int>(mnist.labels[i]))
      ++r.correct;
    ++r.total;
  }
  const auto end = std::chrono::steady_clock::now();
  r.wall_ns = duration_ns(start, end);
  runner.close();
  return r;
}

Result run_async_two_thread(simaai::neat::Model& model, const MnistDataset& mnist, int timeout_ms,
                            int queue_depth, bool read_outputs, int consumer_work_us) {
  simaai::neat::RunOptions run_opt;
  run_opt.queue_depth = queue_depth;
  run_opt.enable_metrics = true;
  auto runner = model.build(std::vector<cv::Mat>{mnist.images.front()},
                            simaai::neat::Model::RouteOptions{}, run_opt);
  Result r;
  r.mode = read_outputs ? "option_c_two_thread_pull_read" : "option_c_two_thread_pull_no_read";

  std::atomic<bool> stop{false};
  std::mutex err_mu;
  std::string err;
  auto set_err = [&](const std::string& s) {
    std::lock_guard<std::mutex> lk(err_mu);
    if (err.empty())
      err = s;
    stop.store(true);
  };

  // input_seq is an INTERNAL transport counter (RunCore re-stamps it from a
  // per-segment fetch_add(0..N)). Model::build() seeds/warms the runner with a
  // dummy frame, which consumes the first sequence value(s) BEFORE the push
  // loop, so the user's frame i arrives with input_seq = i + seed_offset (an
  // off-by-one for a single warm-up frame). We therefore rebase input_seq to the
  // first value the consumer observes, which corresponds to pushed image[0].
  // This keeps the output->input correlation (robust to reordering) while
  // absorbing the warm-up offset, instead of indexing labels by the raw,
  // offset internal counter.
  int64_t base_input_seq = -1;
  const auto start = std::chrono::steady_clock::now();
  std::thread consumer([&]() {
    try {
      while (r.total < static_cast<int>(mnist.images.size()) && !stop.load()) {
        const auto t0 = std::chrono::steady_clock::now();
        simaai::neat::Sample samples = runner.pull(timeout_ms);
        const auto t1 = std::chrono::steady_clock::now();
        r.pull_ns += duration_ns(t0, t1);
        if (samples.empty()) {
          set_err("async pull returned empty at " + std::to_string(r.total));
          return;
        }
        for (auto& sample : samples) {
          if (r.total >= static_cast<int>(mnist.images.size()))
            break;
          int sample_index = r.total;
          int64_t seq = sample.input_seq >= 0 ? sample.input_seq : sample.orig_input_seq;
          if (seq >= 0) {
            if (base_input_seq < 0)
              base_input_seq = seq; // first observed output == pushed image[0]
            const int64_t rebased = seq - base_input_seq;
            if (rebased >= 0 && rebased < static_cast<int64_t>(mnist.labels.size())) {
              sample_index = static_cast<int>(rebased);
            }
          }
          if (read_outputs) {
            simaai::neat::TensorList tensors = simaai::neat::tensors_from_sample(sample);
            if (tensors.empty())
              throw std::runtime_error("async sample has no tensors");
            const auto scores = tensor_to_scores(tensors.front(), &r.map_ns);
            burn_us(consumer_work_us);
            const int pred = argmax10(scores);
            if (pred == static_cast<int>(mnist.labels[sample_index]))
              ++r.correct;
          } else {
            ++r.correct; // no-read mode only checks transport completeness.
          }
          ++r.total;
        }
      }
    } catch (const std::exception& e) {
      set_err(e.what());
    }
  });

  for (std::size_t i = 0; i < mnist.images.size() && !stop.load(); ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = runner.push(std::vector<cv::Mat>{mnist.images[i]});
    const auto t1 = std::chrono::steady_clock::now();
    r.push_ns += duration_ns(t0, t1);
    if (!ok) {
      set_err("async push returned false at " + std::to_string(i));
      break;
    }
  }
  runner.close_input();
  consumer.join();
  const auto end = std::chrono::steady_clock::now();
  r.wall_ns = duration_ns(start, end);
  runner.close();

  if (!err.empty())
    throw std::runtime_error(err);
  return r;
}
#endif

} // namespace

int main(int argc, char** argv) {
#if !defined(SIMA_WITH_OPENCV)
  (void)argc;
  (void)argv;
  return skip_long_test("OpenCV required");
#else
  try {
    if (!sima_test::has_flag(argc, argv, "--run") &&
        !local_env_flag("SIMA_RUN_MNIST_ASYNC_CONSUMER_CONTROL")) {
      return skip_long_test("set SIMA_RUN_MNIST_ASYNC_CONSUMER_CONTROL=1 or pass --run");
    }
    simaai::neat::gst_init_once();
    if (!simaai::neat::element_exists("neatprocesscvu") ||
        !simaai::neat::element_exists("neatprocessmla")) {
      return skip_long_test("missing SimaAI plugins");
    }

    const fs::path root = sima_test::test_source_root();
    const fs::path cache_dir = root / "tmp" / "mnist_async_consumer_control";
    std::error_code ec;
    fs::create_directories(cache_dir, ec);
    fs::current_path(root, ec);

    int limit = env_int_value("SIMA_MNIST_ASYNC_CONTROL_LIMIT", 100);
    int timeout_ms = env_int_value("SIMA_MNIST_ASYNC_CONTROL_TIMEOUT_MS", 20000);
    int queue_depth = env_int_value("SIMA_MNIST_ASYNC_CONTROL_QUEUE_DEPTH", 4);
    int consumer_work_us = env_int_value("SIMA_MNIST_ASYNC_CONTROL_WORK_US", 0);
    std::string tmp;
    sima_test::parse_int_arg(argc, argv, "--limit", limit);
    sima_test::parse_int_arg(argc, argv, "--timeout-ms", timeout_ms);
    sima_test::parse_int_arg(argc, argv, "--queue-depth", queue_depth);
    sima_test::parse_int_arg(argc, argv, "--consumer-work-us", consumer_work_us);

    std::string model_name = env_string("SIMA_MNIST_MODEL_NAME", kDefaultModelName);
    std::string model_path = env_string("SIMA_MNIST_MODEL_TAR");
    if (sima_test::get_arg(argc, argv, "--model-name", tmp))
      model_name = tmp;
    if (sima_test::get_arg(argc, argv, "--model", tmp))
      model_path = tmp;

    const fs::path images_path =
        ensure_mnist_file(cache_dir, "t10k-images-idx3-ubyte",
                          env_string("SIMA_MNIST_IMAGES_URL", kDefaultImagesUrl));
    const fs::path labels_path =
        ensure_mnist_file(cache_dir, "t10k-labels-idx1-ubyte",
                          env_string("SIMA_MNIST_LABELS_URL", kDefaultLabelsUrl));
    MnistDataset mnist = load_mnist_dataset(images_path, labels_path, limit);
    if (model_path.empty())
      model_path = sima_test::resolve_modelzoo_tar(model_name, root);
    if (model_path.empty() || !fs::exists(model_path)) {
      throw std::runtime_error("failed to resolve MNIST model tar.gz");
    }

    simaai::neat::Model::Options model_opt;
    model_opt.preprocess.kind = simaai::neat::InputKind::Image;
    model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
    model_opt.preprocess.input_max_depth = 1;
    model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::GRAY8;
    model_opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
    model_opt.preprocess.normalize.mean = std::array<float, 3>{0.1307f, 0.1307f, 0.1307f};
    model_opt.preprocess.normalize.stddev = std::array<float, 3>{0.3081f, 0.3081f, 0.3081f};
    model_opt.preprocess.normalize.has_explicit_stats = true;

    std::cout << "[CONTROL] model=" << model_path << " limit=" << mnist.images.size()
              << " queue_depth=" << queue_depth << " consumer_work_us=" << consumer_work_us << "\n";
    simaai::neat::Model model(model_path, model_opt);
    std::cout << "[CONTROL] resolved preprocess plan:\n"
              << model.resolved_preprocess_plan().to_debug_string() << "\n";

    auto r0 = run_sync_inline(model, mnist, timeout_ms, consumer_work_us);
    print_result(r0);
    auto r1 = run_async_two_thread(model, mnist, timeout_ms, queue_depth, false, consumer_work_us);
    print_result(r1);
    auto r2 = run_async_two_thread(model, mnist, timeout_ms, queue_depth, true, consumer_work_us);
    print_result(r2);

    const double sync_fps = static_cast<double>(r0.total) / (static_cast<double>(r0.wall_ns) / 1e9);
    const double async_fps =
        static_cast<double>(r2.total) / (static_cast<double>(r2.wall_ns) / 1e9);
    std::cout << std::fixed << std::setprecision(3)
              << "[CONTROL] option_c_speedup_vs_sync=" << (async_fps / sync_fps)
              << " no_read_upper_bound_fps="
              << (static_cast<double>(r1.total) / (static_cast<double>(r1.wall_ns) / 1e9)) << "\n";
    return 0;
  } catch (const SkipTest& e) {
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
#endif
}

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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
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
  if (!v || !*v) {
    return fallback;
  }
  return std::string(v);
}

int env_int_value(const char* key, int fallback) {
  const char* v = std::getenv(key);
  if (!v || !*v) {
    return fallback;
  }
  return std::atoi(v);
}

double env_double_value(const char* key, double fallback) {
  const char* v = std::getenv(key);
  if (!v || !*v) {
    return fallback;
  }
  char* end = nullptr;
  const double parsed = std::strtod(v, &end);
  if (!end || end == v) {
    return fallback;
  }
  return parsed;
}

std::string dtype_name(simaai::neat::TensorDType dtype) {
  switch (dtype) {
  case simaai::neat::TensorDType::UInt8:
    return "UInt8";
  case simaai::neat::TensorDType::Int8:
    return "Int8";
  case simaai::neat::TensorDType::UInt16:
    return "UInt16";
  case simaai::neat::TensorDType::Int16:
    return "Int16";
  case simaai::neat::TensorDType::Int32:
    return "Int32";
  case simaai::neat::TensorDType::BFloat16:
    return "BFloat16";
  case simaai::neat::TensorDType::Float32:
    return "Float32";
  case simaai::neat::TensorDType::Float64:
    return "Float64";
  }
  return "Unknown";
}

std::string shape_string(const std::vector<int64_t>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      oss << ",";
    }
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

float bf16_to_fp32(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

std::vector<double> tensor_to_scores(const simaai::neat::Tensor& tensor) {
  // Read directly from the mapped output tensor. The CPU can access EV74-backed
  // SiMa memory after Tensor::map(Read) performs the required visibility sync;
  // cloning first only adds an allocation + memcpy and can hide the zero-copy
  // path this regression test is meant to exercise.
  simaai::neat::Mapping map = tensor.map(simaai::neat::MapMode::Read);
  if (!map.data || map.size_bytes == 0) {
    throw std::runtime_error("output tensor map is empty");
  }

  std::vector<double> out;
  auto reserve_count = [&](std::size_t elem_size) {
    if (elem_size == 0 || (map.size_bytes % elem_size) != 0) {
      throw std::runtime_error("output tensor byte size is incompatible with dtype " +
                               dtype_name(tensor.dtype) + ": bytes=" +
                               std::to_string(map.size_bytes));
    }
    out.reserve(map.size_bytes / elem_size);
  };

  switch (tensor.dtype) {
  case simaai::neat::TensorDType::UInt8: {
    reserve_count(sizeof(uint8_t));
    const auto* p = static_cast<const uint8_t*>(map.data);
    for (std::size_t i = 0; i < map.size_bytes; ++i) {
      out.push_back(static_cast<double>(p[i]));
    }
    break;
  }
  case simaai::neat::TensorDType::Int8: {
    reserve_count(sizeof(int8_t));
    const auto* p = static_cast<const int8_t*>(map.data);
    for (std::size_t i = 0; i < map.size_bytes; ++i) {
      out.push_back(static_cast<double>(p[i]));
    }
    break;
  }
  case simaai::neat::TensorDType::UInt16: {
    reserve_count(sizeof(uint16_t));
    const auto* p = static_cast<const uint16_t*>(map.data);
    for (std::size_t i = 0; i < map.size_bytes / sizeof(uint16_t); ++i) {
      out.push_back(static_cast<double>(p[i]));
    }
    break;
  }
  case simaai::neat::TensorDType::Int16: {
    reserve_count(sizeof(int16_t));
    const auto* p = static_cast<const int16_t*>(map.data);
    for (std::size_t i = 0; i < map.size_bytes / sizeof(int16_t); ++i) {
      out.push_back(static_cast<double>(p[i]));
    }
    break;
  }
  case simaai::neat::TensorDType::Int32: {
    reserve_count(sizeof(int32_t));
    const auto* p = static_cast<const int32_t*>(map.data);
    for (std::size_t i = 0; i < map.size_bytes / sizeof(int32_t); ++i) {
      out.push_back(static_cast<double>(p[i]));
    }
    break;
  }
  case simaai::neat::TensorDType::BFloat16: {
    reserve_count(sizeof(uint16_t));
    const auto* p = static_cast<const uint16_t*>(map.data);
    for (std::size_t i = 0; i < map.size_bytes / sizeof(uint16_t); ++i) {
      out.push_back(static_cast<double>(bf16_to_fp32(p[i])));
    }
    break;
  }
  case simaai::neat::TensorDType::Float32: {
    reserve_count(sizeof(float));
    const auto* p = static_cast<const float*>(map.data);
    for (std::size_t i = 0; i < map.size_bytes / sizeof(float); ++i) {
      out.push_back(static_cast<double>(p[i]));
    }
    break;
  }
  case simaai::neat::TensorDType::Float64: {
    reserve_count(sizeof(double));
    const auto* p = static_cast<const double*>(map.data);
    for (std::size_t i = 0; i < map.size_bytes / sizeof(double); ++i) {
      out.push_back(p[i]);
    }
    break;
  }
  }

  if (out.size() < 10) {
    throw std::runtime_error("expected at least 10 MNIST scores, got " +
                             std::to_string(out.size()) + " dtype=" + dtype_name(tensor.dtype) +
                             " shape=" + shape_string(tensor.shape));
  }
  out.resize(10);
  return out;
}

int argmax10(const std::vector<double>& scores) {
  return static_cast<int>(std::max_element(scores.begin(), scores.begin() + 10) - scores.begin());
}

std::string format_top_scores(const std::vector<double>& scores, int k = 3) {
  std::vector<int> indices(10);
  for (int i = 0; i < 10; ++i) {
    indices[i] = i;
  }
  k = std::min(k, 10);
  std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                    [&](int a, int b) { return scores[a] > scores[b]; });
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4);
  for (int i = 0; i < k; ++i) {
    if (i != 0) {
      oss << " ";
    }
    oss << indices[i] << ":" << scores[indices[i]];
  }
  return oss.str();
}

bool ensure_decompressed_gzip(const fs::path& gz_path, const fs::path& out_path) {
  if (sima_test::is_usable_regular_file(out_path)) {
    return true;
  }
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

std::vector<uint8_t> read_binary_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open " + path.string());
  }
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  in.seekg(0, std::ios::beg);
  if (size <= 0) {
    throw std::runtime_error("empty file " + path.string());
  }
  std::vector<uint8_t> data(static_cast<std::size_t>(size));
  in.read(reinterpret_cast<char*>(data.data()), size);
  if (!in) {
    throw std::runtime_error("failed to read " + path.string());
  }
  return data;
}

uint32_t read_be32(const std::vector<uint8_t>& data, std::size_t offset, const std::string& label) {
  if (offset + 4 > data.size()) {
    throw std::runtime_error("truncated IDX header in " + label);
  }
  return (static_cast<uint32_t>(data[offset]) << 24) |
         (static_cast<uint32_t>(data[offset + 1]) << 16) |
         (static_cast<uint32_t>(data[offset + 2]) << 8) |
         static_cast<uint32_t>(data[offset + 3]);
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

  const uint32_t image_magic = read_be32(image_bytes, 0, images_path.string());
  const uint32_t image_count = read_be32(image_bytes, 4, images_path.string());
  const uint32_t rows = read_be32(image_bytes, 8, images_path.string());
  const uint32_t cols = read_be32(image_bytes, 12, images_path.string());
  const uint32_t label_magic = read_be32(label_bytes, 0, labels_path.string());
  const uint32_t label_count = read_be32(label_bytes, 4, labels_path.string());

  if (image_magic != 2051) {
    throw std::runtime_error("unexpected MNIST image magic: " + std::to_string(image_magic));
  }
  if (label_magic != 2049) {
    throw std::runtime_error("unexpected MNIST label magic: " + std::to_string(label_magic));
  }
  if (rows == 0 || cols == 0) {
    throw std::runtime_error("invalid MNIST image dimensions");
  }

  const uint32_t available = std::min(image_count, label_count);
  if (available == 0) {
    throw std::runtime_error("MNIST dataset is empty");
  }
  const std::size_t image_size = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  const std::size_t required_image_bytes = 16 + static_cast<std::size_t>(image_count) * image_size;
  const std::size_t required_label_bytes = 8 + static_cast<std::size_t>(label_count);
  if (image_bytes.size() < required_image_bytes || label_bytes.size() < required_label_bytes) {
    throw std::runtime_error("truncated MNIST IDX payload");
  }

  if (limit <= 0 || static_cast<uint32_t>(limit) > available) {
    limit = static_cast<int>(available);
  }

  MnistDataset out;
  out.rows = static_cast<int>(rows);
  out.cols = static_cast<int>(cols);
  out.images.reserve(static_cast<std::size_t>(limit));
  out.labels.reserve(static_cast<std::size_t>(limit));

  for (int i = 0; i < limit; ++i) {
    cv::Mat img(out.rows, out.cols, CV_8UC1);
    const std::size_t offset = 16 + static_cast<std::size_t>(i) * image_size;
    std::memcpy(img.data, image_bytes.data() + offset, image_size);
    out.images.push_back(img);
    out.labels.push_back(label_bytes[8 + static_cast<std::size_t>(i)]);
  }
  return out;
}
#endif

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

} // namespace

int main(int argc, char** argv) {
#if !defined(SIMA_WITH_OPENCV)
  (void)argc;
  (void)argv;
  return skip_long_test("OpenCV required for MNIST Preproc channels=1 accuracy test");
#else
  try {
    const bool run_flag = sima_test::has_flag(argc, argv, "--run");
    if (!run_flag && !env_flag("SIMA_RUN_MNIST_PREPROC_CHANNELS1")) {
      return skip_long_test(
          "set SIMA_RUN_MNIST_PREPROC_CHANNELS1=1 or pass --run to execute");
    }

    simaai::neat::gst_init_once();
    if (!simaai::neat::element_exists("neatprocesscvu") ||
        !simaai::neat::element_exists("neatprocessmla")) {
      return skip_long_test("missing SimaAI plugins (neatprocesscvu/mla)");
    }

    const fs::path root = sima_test::test_source_root();
    const fs::path cache_dir = root / "tmp" / "mnist_preproc_channels1";
    std::error_code ec;
    fs::create_directories(cache_dir, ec);
    fs::current_path(root, ec);

    std::string model_name = env_string("SIMA_MNIST_MODEL_NAME", kDefaultModelName);
    std::string model_path = env_string("SIMA_MNIST_MODEL_TAR");
    std::string tmp;
    if (sima_test::get_arg(argc, argv, "--model-name", tmp)) {
      model_name = tmp;
    }
    if (sima_test::get_arg(argc, argv, "--model", tmp)) {
      model_path = tmp;
    }

    int limit = env_int_value("SIMA_MNIST_PREPROC_LIMIT", 100);
    sima_test::parse_int_arg(argc, argv, "--limit", limit);
    double min_accuracy = env_double_value("SIMA_MNIST_PREPROC_MIN_ACCURACY", 0.90);
    sima_test::parse_double_arg(argc, argv, "--min-accuracy", min_accuracy);
    const int timeout_ms = env_int_value("SIMA_MNIST_PREPROC_TIMEOUT_MS", 20000);

    const std::string images_url = env_string("SIMA_MNIST_IMAGES_URL", kDefaultImagesUrl);
    const std::string labels_url = env_string("SIMA_MNIST_LABELS_URL", kDefaultLabelsUrl);
    const fs::path images_path = ensure_mnist_file(cache_dir, "t10k-images-idx3-ubyte", images_url);
    const fs::path labels_path = ensure_mnist_file(cache_dir, "t10k-labels-idx1-ubyte", labels_url);
    MnistDataset mnist = load_mnist_dataset(images_path, labels_path, limit);
    require(!mnist.images.empty(), "MNIST dataset load returned no images");

    if (model_path.empty()) {
      model_path = sima_test::resolve_modelzoo_tar(model_name, root);
    }
    if (model_path.empty() || !fs::exists(model_path)) {
      throw std::runtime_error("failed to resolve MNIST model tar.gz; run 'sima-cli modelzoo get " +
                               model_name + "' or set SIMA_MNIST_MODEL_TAR");
    }

    simaai::neat::Model::Options model_opt;
    model_opt.preprocess.kind = simaai::neat::InputKind::Image;
    model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
    model_opt.preprocess.input_max_depth = 1;
    model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::GRAY8;
    if (env_flag("SIMA_MNIST_PREPROC_EXPLICIT_NORM")) {
      model_opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
      model_opt.preprocess.normalize.mean = std::array<float, 3>{0.1307f, 0.1307f, 0.1307f};
      model_opt.preprocess.normalize.stddev = std::array<float, 3>{0.3081f, 0.3081f, 0.3081f};
      model_opt.preprocess.normalize.has_explicit_stats = true;
    }

    std::cout << "[MNIST] model_name=" << model_name << "\n";
    std::cout << "[MNIST] model_path=" << model_path << "\n";
    std::cout << "[MNIST] samples=" << mnist.images.size() << " image=" << mnist.cols << "x"
              << mnist.rows << "x1 min_accuracy=" << min_accuracy << "\n";

    simaai::neat::Model model(model_path, model_opt);
    std::cout << "[MNIST] resolved preprocess plan:\n"
              << model.resolved_preprocess_plan().to_debug_string() << "\n";

    auto runner = model.build(std::vector<cv::Mat>{mnist.images.front()});
    require(static_cast<bool>(runner), "failed to build MNIST model runner");

    struct Mismatch {
      int index = 0;
      int label = 0;
      int pred = 0;
      std::string top;
    };
    std::vector<Mismatch> mismatches;
    int correct = 0;

    for (std::size_t i = 0; i < mnist.images.size(); ++i) {
      simaai::neat::TensorList outs = runner.run(std::vector<cv::Mat>{mnist.images[i]}, timeout_ms);
      if (outs.empty()) {
        runner.close();
        throw std::runtime_error("model returned no output for sample " + std::to_string(i));
      }
      const auto scores = tensor_to_scores(outs.front());
      const int pred = argmax10(scores);
      const int label = static_cast<int>(mnist.labels[i]);
      if (pred == label) {
        ++correct;
      } else if (mismatches.size() < 10) {
        mismatches.push_back(Mismatch{static_cast<int>(i), label, pred, format_top_scores(scores)});
      }
    }
    runner.close();

    const double accuracy = static_cast<double>(correct) / static_cast<double>(mnist.images.size());
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[MNIST] correct=" << correct << " total=" << mnist.images.size()
              << " accuracy=" << accuracy << "\n";
    for (const auto& mm : mismatches) {
      std::cout << "[MNIST][MISMATCH] index=" << mm.index << " label=" << mm.label
                << " pred=" << mm.pred << " top=" << mm.top << "\n";
    }

    if (accuracy < min_accuracy) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(4)
          << "MNIST Preproc channels=1 accuracy below threshold: accuracy=" << accuracy
          << " threshold=" << min_accuracy << " correct=" << correct
          << " total=" << mnist.images.size();
      throw std::runtime_error(oss.str());
    }

    std::cout << "[OK] mnist_preproc_channels1_accuracy_test passed\n";
    return 0;
  } catch (const SkipTest& e) {
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (is_dispatcher_unavailable(msg)) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "[FAIL] " << msg << "\n";
    return 1;
  }
#endif
}

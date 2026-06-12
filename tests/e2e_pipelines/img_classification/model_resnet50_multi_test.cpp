#include "model/Model.h"

#include "test_utils.h"

#include "asset_utils.h"
#include "cli_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static cv::Mat load_rgb_resized(const std::string& image_path, int w, int h) {
  cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("Failed to read image: " + image_path);
  }

  if (w > 0 && h > 0 && (bgr.cols != w || bgr.rows != h)) {
    cv::resize(bgr, bgr, cv::Size(w, h), 0, 0, cv::INTER_AREA);
  }

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

struct ScoredIndex {
  int index = -1;
  float value = 0.0f;
  float prob = 0.0f;
};

static std::vector<ScoredIndex> topk_with_softmax(const std::vector<float>& v, int k) {
  if (v.empty() || k <= 0)
    return {};
  const int n = static_cast<int>(v.size());
  k = std::min(k, n);

  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&v](int a, int b) { return v[a] > v[b]; });

  const float maxv = *std::max_element(v.begin(), v.end());
  double sum = 0.0;
  for (float x : v) {
    sum += std::exp(static_cast<double>(x - maxv));
  }

  std::vector<ScoredIndex> out;
  out.reserve(k);
  for (int i = 0; i < k; ++i) {
    const int id = idx[i];
    const double prob = std::exp(static_cast<double>(v[id] - maxv)) / sum;
    out.push_back(ScoredIndex{id, v[id], static_cast<float>(prob)});
  }
  return out;
}

static simaai::neat::Tensor require_tensor(const simaai::neat::TensorList& out,
                                           const std::string& label) {
  if (out.empty()) {
    throw std::runtime_error(label + ": expected tensor output");
  }
  return out.front();
}

static std::vector<float> tensor_to_floats(const simaai::neat::Tensor& t,
                                           const std::string& label) {
  if (t.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error(label + ": expected Float32 tensor output");
  }
  if (!t.is_dense()) {
    throw std::runtime_error(label + ": expected dense tensor output");
  }

  simaai::neat::Tensor cpu = t.clone();
  simaai::neat::Mapping map = cpu.map(simaai::neat::MapMode::Read);
  if (!map.data || map.size_bytes == 0) {
    throw std::runtime_error(label + ": tensor output is empty");
  }
  if (map.size_bytes % sizeof(float) != 0) {
    throw std::runtime_error(label + ": tensor size is not a multiple of float");
  }

  const size_t elems = map.size_bytes / sizeof(float);
  std::vector<float> out(elems);
  std::memcpy(out.data(), map.data, elems * sizeof(float));
  return out;
}

static std::vector<float> scores_from_tensor(const simaai::neat::Tensor& t,
                                             const std::string& label) {
  auto scores_full = tensor_to_floats(t, label);
  if (scores_full.empty()) {
    throw std::runtime_error(label + ": empty tensor output");
  }
  if (scores_full.size() < 1000) {
    throw std::runtime_error(label + ": expected at least 1000 scores, got " +
                             std::to_string(scores_full.size()));
  }
  if (scores_full.size() > 1000) {
    scores_full.resize(1000);
  }
  return scores_full;
}

static void check_top1(const std::vector<float>& scores, int expected_id, float min_prob,
                       const std::string& label) {
  const auto top = topk_with_softmax(scores, 5);
  std::cout << "[" << label << "] top1 index=" << top[0].index << " score=" << top[0].value
            << " prob=" << top[0].prob << "\n";
  std::cout << "[" << label << "] top5:";
  for (const auto& t : top) {
    std::cout << " " << t.index << ":" << t.prob;
  }
  std::cout << "\n";

  if (top[0].index != expected_id) {
    throw std::runtime_error(label + ": top-1 mismatch: expected " + std::to_string(expected_id) +
                             " got " + std::to_string(top[0].index));
  }
  if (min_prob > 0.0f && top[0].prob < min_prob) {
    throw std::runtime_error(label + ": top-1 probability too low: " + std::to_string(top[0].prob) +
                             " < " + std::to_string(min_prob));
  }
  std::cout << "[" << label << "] top-1 matches expected class " << expected_id << "\n";
}

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  constexpr const char* kGoldfishUrl =
      "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
      "n01443537_goldfish.JPEG";
  constexpr int kGoldfishId = 1; // ILSVRC2012 0-based index for "goldfish"
  constexpr int kInferWidth = 224;
  constexpr int kInferHeight = 224;
  const std::vector<float> kMean = {0.485f, 0.456f, 0.406f};
  const std::vector<float> kStd = {0.229f, 0.224f, 0.225f};

  std::string image_path;
  std::string tar_gz;
  std::string goldfish_url = kGoldfishUrl;
  int expected_id = kGoldfishId;
  float min_prob = 0.2f;

  std::string tmp;
  if (sima_test::get_arg(argc, argv, "--image", tmp))
    image_path = tmp;
  if (sima_test::get_arg(argc, argv, "--model", tmp))
    tar_gz = tmp;
  if (sima_test::get_arg(argc, argv, "--goldfish-url", tmp))
    goldfish_url = tmp;
  sima_test::parse_int_arg(argc, argv, "--expect-id", expected_id);
  sima_test::parse_float_arg(argc, argv, "--min-prob", min_prob);

  if (image_path.empty()) {
    const fs::path out_path = sima_test::default_goldfish_path();
    if (!sima_test::download_file(goldfish_url, out_path)) {
      std::cerr << "Failed to download goldfish image.\n";
      std::cerr << "URL was: " << goldfish_url << "\n";
      std::cerr << "Tip: supply --image <path> and --expect-id <id> instead.\n";
      return 3;
    }
    image_path = out_path.string();
    std::cout << "Using goldfish image: " << image_path << "\n";
  }

  require(fs::exists(image_path), "Image path missing: " + image_path);

  if (tar_gz.empty()) {
    tar_gz = sima_test::resolve_resnet50_tar();
    if (tar_gz.empty()) {
      std::cerr << "Failed to resolve resnet50 tar.gz. "
                << "Set SIMA_MODEL_TAR (or SIMA_RESNET50_TAR) or run "
                   "'sima-cli modelzoo get resnet_50'.\n";
      return 3;
    }
  }
  if (!fs::exists(tar_gz)) {
    std::cerr << "Model tar.gz missing: " << tar_gz << "\n";
    return 3;
  }

  try {
    std::cerr << "[stage] load image\n";
    cv::Mat rgb = load_rgb_resized(image_path, kInferWidth, kInferHeight);

    std::cerr << "[stage] model1 init\n";
    simaai::neat::Model::Options model1_opt;
    model1_opt.preprocess.kind = simaai::neat::InputKind::Image;
    model1_opt.preprocess.enable = simaai::neat::AutoFlag::On;
    model1_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::RGB;
    model1_opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
    model1_opt.preprocess.normalize.mean = std::array<float, 3>{kMean[0], kMean[1], kMean[2]};
    model1_opt.preprocess.normalize.stddev = std::array<float, 3>{kStd[0], kStd[1], kStd[2]};
    model1_opt.preprocess.normalize.has_explicit_stats = true;
    simaai::neat::Model model1(tar_gz, model1_opt);
    std::cerr << "[stage] model1 run\n";
    auto first_out = model1.run(std::vector<cv::Mat>{rgb});
    auto first_scores = scores_from_tensor(require_tensor(first_out, "first"), "first");
    check_top1(first_scores, expected_id, min_prob, "first");

    std::cerr << "[stage] model2 init (parallel)\n";
    simaai::neat::Model::Options model2_opt;
    model2_opt.preprocess.kind = simaai::neat::InputKind::Image;
    model2_opt.preprocess.enable = simaai::neat::AutoFlag::On;
    model2_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::RGB;
    model2_opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
    model2_opt.preprocess.normalize.mean = std::array<float, 3>{kMean[0], kMean[1], kMean[2]};
    model2_opt.preprocess.normalize.stddev = std::array<float, 3>{kStd[0], kStd[1], kStd[2]};
    model2_opt.preprocess.normalize.has_explicit_stats = true;
    simaai::neat::Model model2(tar_gz, model2_opt);
    std::cerr << "[stage] model2 run\n";
    auto second_out = model2.run(std::vector<cv::Mat>{rgb});
    auto second_scores = scores_from_tensor(require_tensor(second_out, "second"), "second");
    check_top1(second_scores, expected_id, min_prob, "second");

    std::cout << "[OK] neatmodel_resnet50_multi_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

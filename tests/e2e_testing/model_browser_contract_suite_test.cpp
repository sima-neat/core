// Contract checks for single-component Model Browser packages. Each CTest runs one entry of the
// table below on a real DevKit: the declared outputs, finite values, sync and async agreement,
// output lifetime across runs and, where the model has a known answer, that answer.
// CI prepares the packages into SIMA_MODEL_BROWSER_DIR (tools/ci/partition_devkit_e2e.py).
#include "asset_utils.h"
#include "cli_utils.h"
#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "model/Model.h"
#include "test_utils.h"

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using simaai::neat::AutoFlag;
using simaai::neat::Model;
using simaai::neat::Tensor;
using simaai::neat::TensorList;

namespace {

struct Contract {
  // Element count of every output, from the model README. Order-independent: the package
  // decides the output order and the check compares sorted counts.
  std::vector<std::int64_t> output_elements;
  std::function<void(Model::Options&)> configure;
  // Installed test images relative to the shared asset root; the first is the reference input.
  std::vector<std::string> images;
  int expected_top1 = -1; // Classifiers only: ImageNet class of images[0].
};

void rgb_image_input(Model::Options& opt, simaai::neat::PreprocessColorFormat format) {
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.enable = AutoFlag::On;
  opt.preprocess.color_convert.enable = AutoFlag::On;
  opt.preprocess.color_convert.input_format = format;
}

const std::map<std::string, Contract>& contracts() {
  static const std::map<std::string, Contract> table = {
      // README: 299x299 RGB normalized to -1..1, one `logits` output with 1000 classes.
      {"inception_v3",
       {{1000},
        [](Model::Options& opt) {
          rgb_image_input(opt, simaai::neat::PreprocessColorFormat::BGR);
          opt.preprocess.normalize.enable = AutoFlag::On;
          opt.preprocess.normalize.mean = {0.5f, 0.5f, 0.5f};
          opt.preprocess.normalize.stddev = {0.5f, 0.5f, 0.5f};
          opt.preprocess.normalize.has_explicit_stats = true;
        },
        {"pcie_host/tutorials/assets/labrador.jpg", "tests/images/people.jpg"},
        208}},
      // README: 640x640 BGR 0..255 without normalization; bbox (4), objectness (1) and class
      // logits (80) on 80x80, 40x40 and 20x20 grids.
      {"yolox_s",
       {{80 * 80 * 4, 40 * 40 * 4, 20 * 20 * 4, 80 * 80, 40 * 40, 20 * 20, 80 * 80 * 80,
         40 * 40 * 80, 20 * 20 * 80},
        [](Model::Options& opt) {
          rgb_image_input(opt, simaai::neat::PreprocessColorFormat::BGR);
          opt.preprocess.normalize.enable = AutoFlag::Off;
        },
        {"tests/images/people.jpg", "pcie_host/tutorials/assets/labrador.jpg"},
        -1}},
  };
  return table;
}

int failures = 0;

void check(bool ok, const std::string& name, const std::string& detail = {}) {
  if (!ok)
    ++failures;
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << name;
  if (!detail.empty())
    std::cout << ": " << detail;
  std::cout << "\n";
}

std::string shape_text(const std::vector<std::int64_t>& shape) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < shape.size(); ++i)
    out << (i ? "," : "") << shape[i];
  out << "]";
  return out.str();
}

std::int64_t element_count(const std::vector<std::int64_t>& shape) {
  std::int64_t n = 1;
  for (std::int64_t d : shape) {
    if (d < 0)
      return -1;
    n *= d;
  }
  return n;
}

std::vector<std::uint8_t> bytes_of(const Tensor& tensor) {
  simaai::neat::Mapping map = tensor.map(simaai::neat::MapMode::Read);
  if (!map.data || map.size_bytes == 0)
    return {};
  const auto* p = static_cast<const std::uint8_t*>(map.data);
  return std::vector<std::uint8_t>(p, p + map.size_bytes);
}

float bf16_to_fp32(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

// Float outputs as doubles; empty for integer dtypes, which are always finite.
std::vector<double> float_values(const Tensor& tensor) {
  const std::vector<std::uint8_t> raw = bytes_of(tensor);
  std::vector<double> out;
  if (tensor.dtype == simaai::neat::TensorDType::Float32) {
    out.resize(raw.size() / sizeof(float));
    for (std::size_t i = 0; i < out.size(); ++i) {
      float v = 0.0f;
      std::memcpy(&v, raw.data() + i * sizeof(float), sizeof(float));
      out[i] = v;
    }
  } else if (tensor.dtype == simaai::neat::TensorDType::BFloat16) {
    out.resize(raw.size() / sizeof(std::uint16_t));
    for (std::size_t i = 0; i < out.size(); ++i) {
      std::uint16_t v = 0;
      std::memcpy(&v, raw.data() + i * sizeof(std::uint16_t), sizeof(std::uint16_t));
      out[i] = bf16_to_fp32(v);
    }
  }
  return out;
}

std::vector<std::vector<std::uint8_t>> all_bytes(const TensorList& outputs) {
  std::vector<std::vector<std::uint8_t>> out;
  out.reserve(outputs.size());
  for (const Tensor& t : outputs)
    out.push_back(bytes_of(t));
  return out;
}

void check_outputs(const TensorList& outputs, const Contract& contract, const std::string& run) {
  check(outputs.size() == contract.output_elements.size(), run + " output count",
        std::to_string(outputs.size()) + " outputs, expected " +
            std::to_string(contract.output_elements.size()));
  std::vector<std::int64_t> seen;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const Tensor& t = outputs[i];
    seen.push_back(element_count(t.shape));
    std::cout << "[OUTPUT] " << run << " " << i << " shape=" << shape_text(t.shape)
              << " dtype=" << static_cast<int>(t.dtype) << "\n";
    const std::vector<double> values = float_values(t);
    const bool finite =
        std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); });
    check(finite, run + " output " + std::to_string(i) + " finite");
    check(!bytes_of(t).empty(), run + " output " + std::to_string(i) + " has data");
  }
  std::vector<std::int64_t> expected = contract.output_elements;
  std::sort(seen.begin(), seen.end());
  std::sort(expected.begin(), expected.end());
  std::ostringstream detail;
  for (std::int64_t n : seen)
    detail << n << " ";
  check(seen == expected, run + " output element counts match the README", detail.str());
}

int top1(const Tensor& logits) {
  const std::vector<double> values = float_values(logits);
  if (values.empty())
    return -1;
  return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

} // namespace

int main(int argc, char** argv) {
#if !defined(SIMA_WITH_OPENCV)
  (void)argc;
  (void)argv;
  std::cerr << "[FAIL] OpenCV required\n";
  return 1;
#else
  try {
    std::string id;
    std::string variant;
    if (!sima_test::get_arg(argc, argv, "--id", id) ||
        !sima_test::get_arg(argc, argv, "--variant", variant))
      throw std::runtime_error("usage: model_browser_contract_suite_test --id <id> --variant <v>");
    const auto entry = contracts().find(id);
    if (entry == contracts().end())
      throw std::runtime_error("no contract for model id " + id);
    const Contract& contract = entry->second;

    const std::string archive_name = id + "_" + variant + "_mpk.tar.gz";
    const char* dir = std::getenv("SIMA_MODEL_BROWSER_DIR");
    const fs::path archive = (dir && *dir) ? fs::path(dir) / archive_name : fs::path(archive_name);
    std::error_code ec;
    if (!dir || !*dir || !fs::is_regular_file(archive, ec) || fs::file_size(archive, ec) == 0) {
      throw std::runtime_error(
          "Model Browser package missing or empty: " + archive.string() +
          ". Set SIMA_MODEL_BROWSER_DIR to a directory holding it, prepared with "
          "'python3 scripts/ci/prepare_model_browser_fixture.py --id " +
          id + " --variant " + variant + " --output <dir>'");
    }
    std::cout << "[MODEL] " << id << " " << variant << " " << archive << "\n";

    simaai::neat::gst_init_once();
    if (!simaai::neat::element_exists("neatkernelrunner") ||
        !simaai::neat::element_exists("neatprocessmla"))
      throw std::runtime_error("missing SimaAI plugins");

    std::vector<cv::Mat> images;
    for (const std::string& rel : contract.images) {
      const fs::path path = sima_test::test_shared_asset_path(rel);
      cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
      if (img.empty())
        throw std::runtime_error("cannot read test image " + path.string());
      images.push_back(img);
    }

    Model::Options opt;
    contract.configure(opt);
    Model model(archive.string(), opt);

    for (const auto& spec : model.input_specs())
      std::cout << "[INPUT] spec shape=" << shape_text(spec.shape) << "\n";
    check(model.compiled_batch_size() == 1, "compiled batch size is 1",
          std::to_string(model.compiled_batch_size()));
    check(model.output_specs().size() == contract.output_elements.size(),
          "output_specs() count matches the README", std::to_string(model.output_specs().size()));

    constexpr int timeout_ms = 30000;
    const TensorList first = model.run(std::vector<cv::Mat>{images[0]}, timeout_ms);
    check_outputs(first, contract, "sync");
    const auto first_bytes = all_bytes(first);

    if (contract.expected_top1 >= 0 && !first.empty()) {
      const int got = top1(first.front());
      check(got == contract.expected_top1, "top-1 class on " + contract.images[0],
            std::to_string(got) + ", expected " + std::to_string(contract.expected_top1));
    }

    // Async: the same input through a built runner gives the same bytes as the sync run.
    {
      auto runner = model.build(std::vector<cv::Mat>{images[0]});
      check(runner.push(std::vector<cv::Mat>{images[0]}), "async push");
      simaai::neat::Sample sample = runner.pull(timeout_ms);
      const TensorList async_out = simaai::neat::tensors_from_sample(sample);
      runner.close();
      check(all_bytes(async_out) == first_bytes, "async output is byte-identical to sync");
    }

    // A second input changes the output, and the first output stays as it was.
    const TensorList second = model.run(std::vector<cv::Mat>{images[1]}, timeout_ms);
    check_outputs(second, contract, "second");
    check(all_bytes(second) != first_bytes, "a different input gives a different output");
    check(all_bytes(first) == first_bytes, "the first output is unchanged after a second run");

    if (failures > 0) {
      std::cerr << "[FAIL] " << id << ": " << failures << " check(s) failed\n";
      return 1;
    }
    std::cout << "[OK] model_browser_contract_" << id << " passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
#endif
}

// DETR ResNet-50 from the Model Browser: one host-preprocessed image tensor in, two outputs out
// (class logits and box logits). The test checks the output contract, decodes detections on the
// CPU, compares the async path with the sync path on alternating images, checks that retained
// outputs stay valid, and checks that malformed inputs are rejected.
#include "asset_utils.h"
#include "cli_utils.h"
#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "model/Model.h"
#include "test_utils.h"

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kInputSize = 640;
constexpr int kQueries = 100;
constexpr int kClasses = 92; // 91 COCO category slots plus "no object"
constexpr int kTimeoutMs = 120000;
constexpr int kCocoPerson = 1;
constexpr int kCocoDog = 18;
// Expected detections were read from the images, not from a reference run. A board run confirms
// or corrects them.
constexpr float kMinScore = 0.5f;
constexpr float kMinIou = 0.5f;
// Hand-labelled box of the jumping skier in tests/images/people.jpg (640x427), in pixels.
constexpr float kSkierBox[4] = {245.0f, 92.0f, 375.0f, 265.0f};
// Async and sync results of the same input must agree to this tolerance.
constexpr double kAsyncTolerance = 1e-3;

struct Detection {
  int class_id = -1;
  float score = 0.0f;
  float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
};

struct Outputs {
  std::vector<double> logits;
  std::vector<double> boxes;
};

int failures = 0;

void check(bool ok, const std::string& name, const std::string& detail = {}) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << name;
  if (!detail.empty())
    std::cout << ": " << detail;
  std::cout << "\n";
  if (!ok)
    ++failures;
}

float bf16_to_fp32(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

std::string shape_string(const std::vector<int64_t>& shape) {
  std::ostringstream os;
  os << "[";
  for (std::size_t i = 0; i < shape.size(); ++i)
    os << (i ? "," : "") << shape[i];
  os << "]";
  return os.str();
}

// Reads the first `count` elements of a float-like tensor as doubles.
std::vector<double> tensor_values(const simaai::neat::Tensor& tensor, std::size_t count) {
  simaai::neat::Mapping map = tensor.map(simaai::neat::MapMode::Read);
  if (!map.data || map.size_bytes == 0)
    throw std::runtime_error("output tensor map is empty");
  std::vector<double> out;
  out.reserve(count);
  switch (tensor.dtype) {
  case simaai::neat::TensorDType::Float32: {
    if (map.size_bytes < count * sizeof(float))
      throw std::runtime_error("float32 output is smaller than expected");
    const auto* p = static_cast<const float*>(map.data);
    out.assign(p, p + count);
    break;
  }
  case simaai::neat::TensorDType::BFloat16: {
    if (map.size_bytes < count * sizeof(uint16_t))
      throw std::runtime_error("bf16 output is smaller than expected");
    const auto* p = static_cast<const uint16_t*>(map.data);
    for (std::size_t i = 0; i < count; ++i)
      out.push_back(static_cast<double>(bf16_to_fp32(p[i])));
    break;
  }
  default:
    throw std::runtime_error("unexpected output dtype " +
                             std::to_string(static_cast<int>(tensor.dtype)));
  }
  return out;
}

Outputs read_outputs(const simaai::neat::TensorList& outs) {
  if (outs.size() != 2)
    throw std::runtime_error("expected 2 outputs, got " + std::to_string(outs.size()));
  return {tensor_values(outs[0], static_cast<std::size_t>(kQueries) * kClasses),
          tensor_values(outs[1], static_cast<std::size_t>(kQueries) * 4)};
}

bool all_finite(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); });
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double diff = a.size() == b.size() ? 0.0 : INFINITY;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    diff = std::max(diff, std::abs(a[i] - b[i]));
  return diff;
}

// The decode the model's README describes: softmax over 92 logits, drop "no object", sigmoid on
// the box logits, (cx, cy, w, h) fractions scaled to the original image.
std::vector<Detection> decode(const Outputs& out, int width, int height) {
  std::vector<Detection> dets;
  for (int q = 0; q < kQueries; ++q) {
    const double* row = out.logits.data() + static_cast<std::size_t>(q) * kClasses;
    const double peak = *std::max_element(row, row + kClasses);
    double sum = 0.0;
    for (int c = 0; c < kClasses; ++c)
      sum += std::exp(row[c] - peak);
    int best = 0;
    for (int c = 1; c < kClasses - 1; ++c)
      if (row[c] > row[best])
        best = c;
    auto sigmoid = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
    const double* box = out.boxes.data() + static_cast<std::size_t>(q) * 4;
    const double cx = sigmoid(box[0]), cy = sigmoid(box[1]);
    const double w = sigmoid(box[2]), h = sigmoid(box[3]);
    Detection d;
    d.class_id = best;
    d.score = static_cast<float>(std::exp(row[best] - peak) / sum);
    d.x1 = static_cast<float>((cx - w / 2) * width);
    d.y1 = static_cast<float>((cy - h / 2) * height);
    d.x2 = static_cast<float>((cx + w / 2) * width);
    d.y2 = static_cast<float>((cy + h / 2) * height);
    dets.push_back(d);
  }
  std::sort(dets.begin(), dets.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });
  return dets;
}

float iou(const Detection& d, const float* ref) {
  const float ix1 = std::max(d.x1, ref[0]), iy1 = std::max(d.y1, ref[1]);
  const float ix2 = std::min(d.x2, ref[2]), iy2 = std::min(d.y2, ref[3]);
  const float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
  const float area = (d.x2 - d.x1) * (d.y2 - d.y1) + (ref[2] - ref[0]) * (ref[3] - ref[1]);
  return area - inter > 0.0f ? inter / (area - inter) : 0.0f;
}

void print_top(const std::string& label, const std::vector<Detection>& dets, int count) {
  for (int i = 0; i < count && i < static_cast<int>(dets.size()); ++i) {
    const auto& d = dets[static_cast<std::size_t>(i)];
    std::cout << "[DETECTION] " << label << " class=" << d.class_id << " score=" << d.score
              << " box=(" << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << ")\n";
  }
}

#if defined(SIMA_WITH_OPENCV)
struct Image {
  std::string label;
  int width = 0;
  int height = 0;
  simaai::neat::Tensor tensor;
};

// Stretch to 640x640 with bilinear resampling, divide by 255, then ImageNet mean and std, HWC.
Image load_image(const std::string& label, const fs::path& path) {
  cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (bgr.empty())
    throw std::runtime_error("cannot read image " + path.string());
  cv::Mat rgb, resized;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  cv::resize(rgb, resized, cv::Size(kInputSize, kInputSize), 0, 0, cv::INTER_LINEAR);
  static const float mean[3] = {0.485f, 0.456f, 0.406f};
  static const float std_dev[3] = {0.229f, 0.224f, 0.225f};
  std::vector<float> data(static_cast<std::size_t>(kInputSize) * kInputSize * 3);
  for (int y = 0; y < kInputSize; ++y) {
    const auto* px = resized.ptr<cv::Vec3b>(y);
    for (int x = 0; x < kInputSize; ++x)
      for (int c = 0; c < 3; ++c)
        data[(static_cast<std::size_t>(y) * kInputSize + x) * 3 + c] =
            (static_cast<float>(px[x][c]) / 255.0f - mean[c]) / std_dev[c];
  }
  return {label, bgr.cols, bgr.rows,
          simaai::neat::Tensor::from_vector(data, {kInputSize, kInputSize, 3},
                                            simaai::neat::TensorMemory::CPU)};
}

bool throws(const std::function<void()>& fn, std::string* what) {
  try {
    fn();
  } catch (const std::exception& e) {
    *what = e.what();
    return true;
  }
  return false;
}
#endif

} // namespace

int main(int argc, char** argv) {
#if !defined(SIMA_WITH_OPENCV)
  (void)argc;
  (void)argv;
  return skip_test("OpenCV required");
#else
  try {
    simaai::neat::gst_init_once();
    if (!simaai::neat::element_exists("neatkernelrunner") ||
        !simaai::neat::element_exists("neatprocessmla")) {
      return fail_test("missing SimaAI plugins");
    }

    std::string model_path;
    if (const char* env = std::getenv("SIMA_DETR_RESNET50_TAR"); env && *env)
      model_path = env;
    std::string arg;
    if (sima_test::get_arg(argc, argv, "--model", arg))
      model_path = arg;
    std::error_code ec;
    if (model_path.empty() || !fs::is_regular_file(model_path, ec) ||
        fs::file_size(model_path, ec) == 0) {
      return fail_test("DETR ResNet-50 MPK is missing or empty: '" + model_path +
                       "'. Set SIMA_DETR_RESNET50_TAR (or pass --model) to the "
                       "detr_resnet_50_w8a8_mixed_mpk.tar.gz from 'sima-cli models download --id "
                       "detr_resnet_50 --variant w8a8_mixed', or run "
                       "scripts/ci/prepare_model_browser_fixture.py --id detr_resnet_50 --variant "
                       "w8a8_mixed --output <dir>");
    }

    std::cout << "[MODEL] detr_resnet_50 w8a8_mixed " << model_path << "\n";
    simaai::neat::Model model(model_path);

    const auto in_specs = model.input_specs();
    const auto out_specs = model.output_specs();
    for (const auto& s : in_specs)
      std::cout << "[SPEC] input rank=" << s.rank << " shape=" << shape_string(s.shape) << "\n";
    for (const auto& s : out_specs)
      std::cout << "[SPEC] output rank=" << s.rank << " shape=" << shape_string(s.shape) << "\n";
    check(in_specs.size() == 1, "one input spec", std::to_string(in_specs.size()));
    check(out_specs.size() == 2, "two output specs", std::to_string(out_specs.size()));
    check(model.compiled_batch_size() == 1, "compiled batch size is 1",
          std::to_string(model.compiled_batch_size()));

    const Image people =
        load_image("people", sima_test::test_shared_asset_path("tests/images/people.jpg"));
    const Image labrador = load_image(
        "labrador", sima_test::test_shared_asset_path("pcie_host/tutorials/assets/labrador.jpg"));

    auto runner = model.build(simaai::neat::TensorList{people.tensor});
    simaai::neat::TensorList people_outs = runner.run({people.tensor}, kTimeoutMs);
    check(people_outs.size() == 2, "sync run returns two outputs",
          std::to_string(people_outs.size()));
    for (std::size_t i = 0; i < people_outs.size(); ++i) {
      const auto& t = people_outs[i];
      std::cout << "[OUTPUT] index=" << i << " name='" << t.route.name
                << "' dtype=" << static_cast<int>(t.dtype) << " shape=" << shape_string(t.shape)
                << "\n";
    }
    if (people_outs.size() == 2) {
      if (!people_outs[0].route.name.empty() && !people_outs[1].route.name.empty())
        check(people_outs[0].route.name == "logits" && people_outs[1].route.name == "box_logits",
              "output order is logits then box_logits",
              people_outs[0].route.name + ", " + people_outs[1].route.name);
      else
        std::cout << "[INFO] route names are empty; output order checked by element count\n";
    }
    const Outputs people_ref = read_outputs(people_outs);
    check(all_finite(people_ref.logits) && all_finite(people_ref.boxes), "outputs are finite");

    const auto people_dets = decode(people_ref, people.width, people.height);
    print_top("people", people_dets, 5);
    bool skier_found = false;
    for (const auto& d : people_dets)
      if (d.class_id == kCocoPerson && d.score >= kMinScore && iou(d, kSkierBox) >= kMinIou)
        skier_found = true;
    check(skier_found, "people.jpg has a person box on the jumping skier",
          "class 1, score >= 0.5, IoU >= 0.5");

    const Outputs labrador_ref = read_outputs(runner.run({labrador.tensor}, kTimeoutMs));
    const auto labrador_dets = decode(labrador_ref, labrador.width, labrador.height);
    print_top("labrador", labrador_dets, 3);
    check(!labrador_dets.empty() && labrador_dets.front().class_id == kCocoDog &&
              labrador_dets.front().score >= kMinScore,
          "labrador.jpg top detection is a dog", "class 18, score >= 0.5");

    // Retained outputs: the first run's tensors must not change when later runs reuse buffers,
    // and each output must stay readable after the other one is released.
    simaai::neat::Tensor kept_boxes = people_outs[1];
    simaai::neat::Tensor kept_logits = people_outs[0];
    people_outs.clear();
    (void)runner.run({labrador.tensor}, kTimeoutMs);
    (void)runner.run({labrador.tensor}, kTimeoutMs);
    check(max_abs_diff(tensor_values(kept_boxes, kQueries * 4), people_ref.boxes) == 0.0,
          "retained box_logits unchanged after later runs");
    check(max_abs_diff(tensor_values(kept_logits, static_cast<std::size_t>(kQueries) * kClasses),
                       people_ref.logits) == 0.0,
          "retained logits unchanged after later runs");
    kept_logits = simaai::neat::Tensor{};
    check(max_abs_diff(tensor_values(kept_boxes, kQueries * 4), people_ref.boxes) == 0.0,
          "box_logits readable after logits is released");

    // Async on alternating images must match the sync result of the same image.
    {
      const std::vector<const Image*> order = {&people, &labrador, &people, &labrador};
      for (const Image* img : order)
        if (!runner.push(simaai::neat::TensorList{img->tensor}))
          throw std::runtime_error("async push failed");
      bool ok = true;
      std::string detail;
      for (std::size_t i = 0; i < order.size(); ++i) {
        simaai::neat::Sample sample = runner.pull(kTimeoutMs);
        const Outputs got = read_outputs(simaai::neat::tensors_from_sample(sample));
        const Outputs& want = order[i] == &people ? people_ref : labrador_ref;
        const double diff =
            std::max(max_abs_diff(got.logits, want.logits), max_abs_diff(got.boxes, want.boxes));
        if (diff > kAsyncTolerance) {
          ok = false;
          detail +=
              order[i]->label + "#" + std::to_string(i) + " diff=" + std::to_string(diff) + " ";
        }
      }
      check(ok, "async results match sync results on alternating images", detail);
    }

    // Malformed inputs must be rejected, and a valid run must still work afterwards.
    std::string what;
    check(throws(
              [&] {
                std::vector<float> small(320 * 320 * 3, 0.0f);
                (void)runner.run({simaai::neat::Tensor::from_vector(
                                     small, {320, 320, 3}, simaai::neat::TensorMemory::CPU)},
                                 kTimeoutMs);
              },
              &what),
          "wrong input shape is rejected", what);
    check(throws(
              [&] {
                std::vector<uint8_t> bytes(static_cast<std::size_t>(kInputSize) * kInputSize * 3,
                                           0);
                (void)runner.run(
                    {simaai::neat::Tensor::from_vector(bytes, {kInputSize, kInputSize, 3},
                                                       simaai::neat::TensorMemory::CPU)},
                    kTimeoutMs);
              },
              &what),
          "wrong input dtype is rejected", what);
    check(throws([&] { (void)runner.run({people.tensor, people.tensor}, kTimeoutMs); }, &what),
          "extra input is rejected", what);
    const Outputs again = read_outputs(runner.run({people.tensor}, kTimeoutMs));
    check(std::max(max_abs_diff(again.logits, people_ref.logits),
                   max_abs_diff(again.boxes, people_ref.boxes)) <= kAsyncTolerance,
          "valid run after rejected inputs matches the first run");
    runner.close();

    if (failures > 0) {
      std::cerr << "[FAIL] detr_resnet50_multi_output_test: " << failures << " check(s) failed\n";
      return 1;
    }
    std::cout << "[OK] detr_resnet50_multi_output_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
#endif
}

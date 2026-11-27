// RF-DETR Small from the Model Browser: two components with a CPU step between them. The backbone
// takes a host-preprocessed image and returns features, 1024 proposal scores and 1024 proposal
// boxes; the test keeps the 300 best proposals on the CPU and runs the decoder on the features and
// those proposals. It checks every tensor's size, decodes detections on two known images, checks
// that alternating images never mix frames, and compares async backbone results with sync ones.
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
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kInputSize = 512;
constexpr int kProposals = 1024;
constexpr int kQueries = 300;
constexpr int kClasses = 91; // COCO category slots; 11 of them are unused
constexpr int kTimeoutMs = 120000;
constexpr int kCocoPerson = 1;
constexpr int kCocoDog = 18;
// Expected detections were read from the images, not from a reference run. A board run confirms
// or corrects them.
constexpr float kMinScore = 0.5f;
constexpr float kMinIou = 0.5f;
// Hand-labelled box of the jumping skier in tests/images/people.jpg (640x427), in pixels.
constexpr float kSkierBox[4] = {245.0f, 92.0f, 375.0f, 265.0f};
// Results of the same input must agree to this tolerance.
constexpr double kTolerance = 1e-3;
// COCO category ids that are not used: background and the ten ids missing from COCO 2017.
constexpr std::array<int, 11> kUnusedClasses = {0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83};

struct Detection {
  int class_id = -1;
  float score = 0.0f;
  float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
};

struct BackboneOut {
  simaai::neat::Tensor features;
  std::vector<double> scores;
  std::vector<double> boxes;
};

constexpr std::size_t kFeatureCount = static_cast<std::size_t>(32) * 32 * 256;

struct DecoderOut {
  std::vector<double> dets;
  std::vector<double> labels;
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
    throw std::runtime_error("tensor map is empty");
  std::vector<double> out;
  out.reserve(count);
  switch (tensor.dtype) {
  case simaai::neat::TensorDType::Float32: {
    if (map.size_bytes < count * sizeof(float))
      throw std::runtime_error("float32 tensor is smaller than expected");
    const auto* p = static_cast<const float*>(map.data);
    out.assign(p, p + count);
    break;
  }
  case simaai::neat::TensorDType::BFloat16: {
    if (map.size_bytes < count * sizeof(uint16_t))
      throw std::runtime_error("bf16 tensor is smaller than expected");
    const auto* p = static_cast<const uint16_t*>(map.data);
    for (std::size_t i = 0; i < count; ++i)
      out.push_back(static_cast<double>(bf16_to_fp32(p[i])));
    break;
  }
  default:
    throw std::runtime_error("unexpected dtype " + std::to_string(static_cast<int>(tensor.dtype)));
  }
  return out;
}

bool all_finite(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); });
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double diff = a.size() == b.size() ? 0.0 : INFINITY;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    const double d = std::abs(a[i] - b[i]);
    if (!std::isfinite(d))
      return INFINITY;
    diff = std::max(diff, d);
  }
  return diff;
}

struct OutputContract {
  const char* name;
  std::vector<int64_t> shape;
};

const std::vector<OutputContract> kBackboneContract = {{"backbone_features", {1, 32, 32, 256}},
                                                       {"proposal_scores", {1, kProposals}},
                                                       {"proposal_boxes", {1, kProposals, 4}}};
const std::vector<OutputContract> kDecoderContract = {{"dets", {1, kQueries, 4}},
                                                      {"labels", {1, kQueries, kClasses}}};

// The part of a route name after the last '/': the compiler's stage prefix, such as cast_5/,
// follows its node numbering and may change.
std::string leaf_name(const std::string& name) {
  const std::size_t slash = name.rfind('/');
  return slash == std::string::npos ? name : name.substr(slash + 1);
}

// Empty when every output has the declared name, shape and a dtype from the model's output spec;
// otherwise what differs.
std::string contract_mismatch(const simaai::neat::TensorList& outs,
                              const std::vector<simaai::neat::TensorSpec>& specs,
                              const std::vector<OutputContract>& contract) {
  if (outs.size() != contract.size() || specs.size() != contract.size())
    return std::to_string(outs.size()) + " outputs and " + std::to_string(specs.size()) +
           " output specs, expected " + std::to_string(contract.size());
  std::string detail;
  for (std::size_t i = 0; i < contract.size(); ++i) {
    const simaai::neat::Tensor& t = outs[i];
    const auto& dtypes = specs[i].dtypes;
    const bool dtype_ok = std::find(dtypes.begin(), dtypes.end(), t.dtype) != dtypes.end();
    if (leaf_name(t.route.name) != contract[i].name || t.shape != contract[i].shape || !dtype_ok)
      detail += "output " + std::to_string(i) + " is '" + t.route.name + "' " +
                shape_string(t.shape) + " dtype " + std::to_string(static_cast<int>(t.dtype)) +
                ", expected '" + contract[i].name + "' " + shape_string(contract[i].shape) +
                " with a dtype from output_specs(); ";
  }
  return detail;
}

void require_contract(const simaai::neat::TensorList& outs,
                      const std::vector<simaai::neat::TensorSpec>& specs,
                      const std::vector<OutputContract>& contract) {
  const std::string mismatch = contract_mismatch(outs, specs, contract);
  if (!mismatch.empty())
    throw std::runtime_error("output contract: " + mismatch);
}

// Every backbone and decoder result goes through these, so each one is checked against the
// contract.
BackboneOut read_backbone(const simaai::neat::TensorList& outs,
                          const std::vector<simaai::neat::TensorSpec>& specs) {
  require_contract(outs, specs, kBackboneContract);
  return {outs[0], tensor_values(outs[1], kProposals),
          tensor_values(outs[2], static_cast<std::size_t>(kProposals) * 4)};
}

DecoderOut read_decoder(const simaai::neat::TensorList& outs,
                        const std::vector<simaai::neat::TensorSpec>& specs) {
  require_contract(outs, specs, kDecoderContract);
  return {tensor_values(outs[0], static_cast<std::size_t>(kQueries) * 4),
          tensor_values(outs[1], static_cast<std::size_t>(kQueries) * kClasses)};
}

// The 300 highest proposal scores, ties broken by the lower index, as the README describes.
std::vector<int> top_proposals(const std::vector<double>& scores) {
  std::vector<int> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return scores[static_cast<std::size_t>(a)] > scores[static_cast<std::size_t>(b)];
  });
  order.resize(kQueries);
  return order;
}

// Gathers the selected proposal rows into a new tensor in EV74 memory, which the decoder reads.
simaai::neat::Tensor proposals_tensor(const std::vector<double>& boxes,
                                      const std::vector<int>& order) {
  std::vector<float> data;
  data.reserve(static_cast<std::size_t>(kQueries) * 4);
  for (int idx : order)
    for (int k = 0; k < 4; ++k)
      data.push_back(static_cast<float>(boxes[static_cast<std::size_t>(idx) * 4 + k]));
  return simaai::neat::Tensor::from_vector(data, {1, kQueries, 4},
                                           simaai::neat::TensorMemory::EV74);
}

// The README's postprocessing: sigmoid on the class logits, the 300 best box-class pairs, drop
// unused classes, then (cx, cy, w, h) fractions to pixel corners clipped to the image.
std::vector<Detection> decode(const DecoderOut& out, int width, int height) {
  const std::size_t pairs = static_cast<std::size_t>(kQueries) * kClasses;
  std::vector<double> scores(pairs);
  for (std::size_t i = 0; i < pairs; ++i)
    scores[i] = 1.0 / (1.0 + std::exp(-std::clamp(out.labels[i], -80.0, 80.0)));
  std::vector<std::size_t> order(pairs);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t a, std::size_t b) { return scores[a] > scores[b]; });
  order.resize(kQueries);
  std::vector<Detection> dets;
  for (std::size_t idx : order) {
    const int cls = static_cast<int>(idx % kClasses);
    if (std::find(kUnusedClasses.begin(), kUnusedClasses.end(), cls) != kUnusedClasses.end())
      continue;
    const double* box = out.dets.data() + (idx / kClasses) * 4;
    Detection d;
    d.class_id = cls;
    d.score = static_cast<float>(scores[idx]);
    d.x1 = static_cast<float>(std::clamp((box[0] - box[2] / 2) * width, 0.0, double(width)));
    d.x2 = static_cast<float>(std::clamp((box[0] + box[2] / 2) * width, 0.0, double(width)));
    d.y1 = static_cast<float>(std::clamp((box[1] - box[3] / 2) * height, 0.0, double(height)));
    d.y2 = static_cast<float>(std::clamp((box[1] + box[3] / 2) * height, 0.0, double(height)));
    if (d.x2 > d.x1 && d.y2 > d.y1)
      dets.push_back(d);
  }
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

// Stretch to 512x512 with bilinear resampling, divide by 255, then ImageNet mean and std, NHWC
// in EV74 memory, which the backbone's device-visible input route requires.
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
          simaai::neat::Tensor::from_vector(data, {1, kInputSize, kInputSize, 3},
                                            simaai::neat::TensorMemory::EV74)};
}
#endif

std::string resolve_component(const char* var, const char* component, int argc, char** argv) {
  std::string path;
  if (const char* env = std::getenv(var); env && *env)
    path = env;
  std::string arg;
  if (sima_test::get_arg(argc, argv, std::string("--") + component, arg))
    path = arg;
  std::error_code ec;
  if (path.empty() || !fs::is_regular_file(path, ec) || fs::file_size(path, ec) == 0) {
    throw std::runtime_error(
        std::string("RF-DETR Small ") + component + " MPK is missing or empty: '" + path +
        "'. Set " + var + " (or pass --" + component + ") to the rf_detr_small_w8a16_w16a16_" +
        component + "_mpk.tar.gz from 'sima-cli models download --id rf_detr_small --variant " +
        "w8a16_w16a16', or run scripts/ci/prepare_model_browser_fixture.py --id rf_detr_small " +
        "--variant w8a16_w16a16 --component backbone --component decoder --output <dir>");
  }
  return path;
}

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

    std::string backbone_path, decoder_path;
    try {
      backbone_path = resolve_component("SIMA_RF_DETR_BACKBONE_TAR", "backbone", argc, argv);
      decoder_path = resolve_component("SIMA_RF_DETR_DECODER_TAR", "decoder", argc, argv);
    } catch (const std::exception& e) {
      return fail_test(e.what());
    }
    std::cout << "[MODEL] rf_detr_small w8a16_w16a16 backbone=" << backbone_path
              << " decoder=" << decoder_path << "\n";

    simaai::neat::Model backbone(backbone_path);
    simaai::neat::Model decoder(decoder_path);
    const auto backbone_specs = backbone.output_specs();
    const auto decoder_specs = decoder.output_specs();
    for (const auto& s : backbone.input_specs())
      std::cout << "[SPEC] backbone input shape=" << shape_string(s.shape) << "\n";
    for (const auto& s : backbone.output_specs())
      std::cout << "[SPEC] backbone output shape=" << shape_string(s.shape) << "\n";
    for (const auto& s : decoder.input_specs())
      std::cout << "[SPEC] decoder input shape=" << shape_string(s.shape) << "\n";
    for (const auto& s : decoder.output_specs())
      std::cout << "[SPEC] decoder output shape=" << shape_string(s.shape) << "\n";
    check(backbone.input_specs().size() == 1, "backbone has one input spec",
          std::to_string(backbone.input_specs().size()));
    check(backbone.output_specs().size() == 3, "backbone has three output specs",
          std::to_string(backbone.output_specs().size()));
    check(decoder.input_specs().size() == 2, "decoder has two input specs",
          std::to_string(decoder.input_specs().size()));
    check(decoder.output_specs().size() == 2, "decoder has two output specs",
          std::to_string(decoder.output_specs().size()));
    check(backbone.compiled_batch_size() == 1 && decoder.compiled_batch_size() == 1,
          "both components are compiled at batch 1",
          std::to_string(backbone.compiled_batch_size()) + ", " +
              std::to_string(decoder.compiled_batch_size()));

    const Image people =
        load_image("people", sima_test::test_shared_asset_path("tests/images/people.jpg"));
    const Image labrador = load_image(
        "labrador", sima_test::test_shared_asset_path("pcie_host/tutorials/assets/labrador.jpg"));

    auto backbone_runner = backbone.build(simaai::neat::TensorList{people.tensor});
    simaai::neat::TensorList people_bb_outs = backbone_runner.run({people.tensor}, kTimeoutMs);
    for (std::size_t i = 0; i < people_bb_outs.size(); ++i) {
      const auto& t = people_bb_outs[i];
      std::cout << "[OUTPUT] backbone index=" << i << " name='" << t.route.name
                << "' dtype=" << static_cast<int>(t.dtype) << " shape=" << shape_string(t.shape)
                << "\n";
    }
    const std::string bb_mismatch =
        contract_mismatch(people_bb_outs, backbone_specs, kBackboneContract);
    check(bb_mismatch.empty(),
          "backbone outputs are backbone_features [1,32,32,256], proposal_scores [1,1024] and "
          "proposal_boxes [1,1024,4]",
          bb_mismatch);
    const BackboneOut people_bb = read_backbone(people_bb_outs, backbone_specs);
    const std::vector<double> people_features = tensor_values(people_bb.features, kFeatureCount);
    check(all_finite(people_bb.scores) && all_finite(people_bb.boxes),
          "proposal scores and boxes are finite");
    // Read from the model's description, not seen on a board yet: the proposal boxes are
    // (cx, cy, w, h) fractions of the image.
    const bool boxes_in_unit = std::all_of(people_bb.boxes.begin(), people_bb.boxes.end(),
                                           [](double v) { return v >= -1e-3 && v <= 1.0 + 1e-3; });
    check(boxes_in_unit, "proposal boxes are fractions of the image in [0, 1]");

    const std::vector<int> people_top = top_proposals(people_bb.scores);
    {
      const BackboneOut again =
          read_backbone(backbone_runner.run({people.tensor}, kTimeoutMs), backbone_specs);
      check(top_proposals(again.scores) == people_top,
            "the top 300 proposals are the same on a second run of the same image");
    }

    auto decoder_runner = decoder.build(simaai::neat::TensorList{
        people_bb.features.cvu(), proposals_tensor(people_bb.boxes, people_top)});

    auto run_pipeline = [&](const Image& img, BackboneOut* bb_out) {
      BackboneOut bb = read_backbone(backbone_runner.run({img.tensor}, kTimeoutMs), backbone_specs);
      const auto order = top_proposals(bb.scores);
      // The decoder's EV74 input stage cannot read the backbone's output memory; cvu() copies the
      // features, as the models repository's application flow does.
      simaai::neat::TensorList outs =
          decoder_runner.run({bb.features.cvu(), proposals_tensor(bb.boxes, order)}, kTimeoutMs);
      if (bb_out)
        *bb_out = bb;
      return outs;
    };

    simaai::neat::TensorList people_dec_outs = run_pipeline(people, nullptr);
    for (std::size_t i = 0; i < people_dec_outs.size(); ++i) {
      const auto& t = people_dec_outs[i];
      std::cout << "[OUTPUT] decoder index=" << i << " name='" << t.route.name
                << "' dtype=" << static_cast<int>(t.dtype) << " shape=" << shape_string(t.shape)
                << "\n";
    }
    const std::string dec_mismatch =
        contract_mismatch(people_dec_outs, decoder_specs, kDecoderContract);
    check(dec_mismatch.empty(), "decoder outputs are dets [1,300,4] and labels [1,300,91]",
          dec_mismatch);
    const DecoderOut people_ref = read_decoder(people_dec_outs, decoder_specs);
    check(all_finite(people_ref.dets) && all_finite(people_ref.labels),
          "decoder outputs are finite");

    const auto people_dets = decode(people_ref, people.width, people.height);
    print_top("people", people_dets, 5);
    bool skier_found = false;
    for (const auto& d : people_dets)
      if (d.class_id == kCocoPerson && d.score >= kMinScore && iou(d, kSkierBox) >= kMinIou)
        skier_found = true;
    check(skier_found, "people.jpg has a person box on the jumping skier",
          "class 1, score >= 0.5, IoU >= 0.5");

    BackboneOut labrador_bb;
    const DecoderOut labrador_ref =
        read_decoder(run_pipeline(labrador, &labrador_bb), decoder_specs);
    const std::vector<double> labrador_features =
        tensor_values(labrador_bb.features, kFeatureCount);
    const auto labrador_dets = decode(labrador_ref, labrador.width, labrador.height);
    print_top("labrador", labrador_dets, 3);
    check(!labrador_dets.empty() && labrador_dets.front().class_id == kCocoDog &&
              labrador_dets.front().score >= kMinScore,
          "labrador.jpg top detection is a dog", "class 18, score >= 0.5");

    // Alternating images through the whole pipeline: each result must match the result of its
    // own image, which catches stale or mixed frames.
    {
      const std::vector<const Image*> order = {&people, &labrador, &people, &labrador};
      bool ok = true;
      std::string detail;
      for (std::size_t i = 0; i < order.size(); ++i) {
        const DecoderOut got = read_decoder(run_pipeline(*order[i], nullptr), decoder_specs);
        const DecoderOut& want = order[i] == &people ? people_ref : labrador_ref;
        const double diff =
            std::max(max_abs_diff(got.dets, want.dets), max_abs_diff(got.labels, want.labels));
        if (diff > kTolerance) {
          ok = false;
          detail +=
              order[i]->label + "#" + std::to_string(i) + " diff=" + std::to_string(diff) + " ";
        }
      }
      check(ok, "alternating images give each image its own result", detail);
    }

    // Async backbone on alternating images must match the sync backbone result of each image.
    {
      const std::vector<const Image*> order = {&people, &labrador, &people, &labrador};
      for (const Image* img : order)
        if (!backbone_runner.push(simaai::neat::TensorList{img->tensor}))
          throw std::runtime_error("async backbone push failed");
      bool ok = true;
      std::string detail;
      for (std::size_t i = 0; i < order.size(); ++i) {
        simaai::neat::Sample sample = backbone_runner.pull(kTimeoutMs);
        const BackboneOut got =
            read_backbone(simaai::neat::tensors_from_sample(sample), backbone_specs);
        const bool is_people = order[i] == &people;
        const BackboneOut& want = is_people ? people_bb : labrador_bb;
        // backbone_features is what the decoder consumes, so it is compared too.
        const double diff =
            std::max({max_abs_diff(got.scores, want.scores), max_abs_diff(got.boxes, want.boxes),
                      max_abs_diff(tensor_values(got.features, kFeatureCount),
                                   is_people ? people_features : labrador_features)});
        if (diff > kTolerance) {
          ok = false;
          detail +=
              order[i]->label + "#" + std::to_string(i) + " diff=" + std::to_string(diff) + " ";
        }
      }
      check(ok, "async backbone results match sync results on alternating images", detail);
    }

    decoder_runner.close();
    backbone_runner.close();

    if (failures > 0) {
      std::cerr << "[FAIL] rf_detr_small_pipeline_test: " << failures << " check(s) failed\n";
      return 1;
    }
    std::cout << "[OK] rf_detr_small_pipeline_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
#endif
}

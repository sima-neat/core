// Proves the mla_only route against the default route on the same input: the host quantizes,
// the card runs only the MLA, the host dequantizes, and every head must match what the card
// produces when it quantizes and dequantizes itself. It doubles as the reference for using
// ModelOptions::mla_only from an application.
#include <simaai/neat/pcie/Model.h>

#include "SignalCloseGuard.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pcie = simaai::neat::pcie;

namespace {

constexpr int kSkipExitCode = 77;
// Largest tolerated deviation from the default route, in units of each head's quantization
// scale. Correct behaviour measures 0; a wrong zero point measures >= 1, a wrong scale ~100.
constexpr double kMaxErrorScales = 0.05;

std::string env_or_default(const char* name, const char* fallback) {
  if (const char* value = std::getenv(name)) {
    if (*value != '\0') {
      return value;
    }
  }
  return fallback ? fallback : "";
}

int env_int_or_default(const char* name, const int fallback) {
  const std::string value = env_or_default(name, "");
  if (value.empty()) {
    return fallback;
  }
  std::size_t parsed = 0;
  try {
    const int result = std::stoi(value, &parsed);
    if (parsed == value.size()) {
      return result;
    }
  } catch (const std::exception&) {
  }
  throw std::runtime_error(std::string("invalid integer in ") + name + ": " + value);
}

struct Args {
  std::string model = env_or_default("SIMAPCIE_YOLOV8_MODEL", DEFAULT_MODEL_PATH);
  std::string card_host = env_or_default("SIMAPCIE_CARD_HOST", "");
  std::string user = env_or_default("SIMAPCIE_USER", "sima");
  int card_id = env_int_or_default("SIMAPCIE_CARD_ID", 0);
  int queue = env_int_or_default("SIMAPCIE_QUEUE", 0);
  int readiness_timeout_ms = env_int_or_default("SIMAPCIE_READINESS_TIMEOUT_MS", 180000);
  int pull_timeout_ms = env_int_or_default("SIMAPCIE_PULL_TIMEOUT_MS", 30000);
  std::string image = env_or_default("SIMAPCIE_TEST_IMAGE", "");
  std::string card_env = env_or_default("SIMAPCIE_CARD_ENV", "");
  std::string card_gst_debug = env_or_default("SIMAPCIE_CARD_GST_DEBUG", "");
  std::string card_gst_debug_file = env_or_default("SIMAPCIE_CARD_GST_DEBUG_FILE", "");
};

std::string require_value(int argc, char** argv, int& i, const char* name) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + name);
  }
  return argv[++i];
}

void usage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " [--model mla_int8_model.tar.gz] [--card-host host] [--card-id n] [--user user]"
               " [--queue n] [--image path] [--card-env 'NAME=VALUE ...']"
               " [--card-gst-debug spec] [--card-gst-debug-file path]\n"
               "Without --model the archive comes from SIMAPCIE_YOLOV8_MODEL; without --image the\n"
               "input is a synthetic ramp over every INT8 code. Exits 77 when no model is\n"
               "configured or the archive is not an MLA-only capable build.\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i] ? argv[i] : "";
    if (arg == "--model") {
      args.model = require_value(argc, argv, i, "--model");
    } else if (arg == "--card-host") {
      args.card_host = require_value(argc, argv, i, "--card-host");
    } else if (arg == "--card-id") {
      args.card_id = std::stoi(require_value(argc, argv, i, "--card-id"));
    } else if (arg == "--user") {
      args.user = require_value(argc, argv, i, "--user");
    } else if (arg == "--queue") {
      args.queue = std::stoi(require_value(argc, argv, i, "--queue"));
    } else if (arg == "--image") {
      args.image = require_value(argc, argv, i, "--image");
    } else if (arg == "--card-env") {
      args.card_env = require_value(argc, argv, i, "--card-env");
    } else if (arg == "--card-gst-debug") {
      args.card_gst_debug = require_value(argc, argv, i, "--card-gst-debug");
    } else if (arg == "--card-gst-debug-file") {
      args.card_gst_debug_file = require_value(argc, argv, i, "--card-gst-debug-file");
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (!args.image.empty() && !std::filesystem::is_regular_file(args.image)) {
    throw std::runtime_error("image path does not exist or is not a regular file: " + args.image);
  }
  if (!args.model.empty() && !std::filesystem::is_regular_file(args.model)) {
    throw std::runtime_error("model path does not exist or is not a regular file: " + args.model);
  }
  if (args.readiness_timeout_ms <= 0 || args.pull_timeout_ms <= 0) {
    throw std::runtime_error("timeouts must be positive");
  }
  return args;
}

std::string shape_string(const std::vector<std::int64_t>& shape) {
  std::string out = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    out += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
  }
  return out + "]";
}

std::size_t element_count(const std::vector<std::int64_t>& shape) {
  std::size_t count = shape.empty() ? 0U : 1U;
  for (const auto dim : shape) {
    if (dim <= 0) {
      throw std::runtime_error("tensor shape has a non-positive dimension: " + shape_string(shape));
    }
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

const std::uint8_t* tensor_bytes(const pcie::Tensor& tensor) {
  return static_cast<const std::uint8_t*>(tensor.data) + tensor.byte_offset;
}

// Every MLA-tessellated archive seen so far quantizes per tensor; per-axis parameters would need
// a channel lookup this reference does not implement.
const pcie::QuantParams& require_quant(const pcie::TensorInfo& info) {
  if (!info.quant.has_value() || info.quant->scales.size() != 1U ||
      info.quant->zero_points.size() != 1U) {
    throw std::runtime_error("tensor '" + info.name + "' needs per-tensor quantization parameters");
  }
  return *info.quant;
}

std::int8_t quantize(const float value, const pcie::QuantParams& quant) {
  const float code =
      std::nearbyint(value / quant.scales[0]) + static_cast<float>(quant.zero_points[0]);
  return static_cast<std::int8_t>(std::clamp(code, -128.0f, 127.0f));
}

float dequantize(const std::int8_t code, const pcie::QuantParams& quant) {
  return static_cast<float>(static_cast<std::int32_t>(code) - quant.zero_points[0]) *
         quant.scales[0];
}

void print_tensor_info(const char* kind, const std::size_t index, const pcie::TensorInfo& info) {
  std::cout << "  " << kind << "[" << index << "] name=" << info.name << " dtype=" << info.dtype
            << " shape=" << shape_string(info.shape) << " size_bytes=" << info.size_bytes;
  if (info.quant.has_value() && !info.quant->scales.empty()) {
    std::cout << " scale=" << info.quant->scales[0] << " zero_point=" << info.quant->zero_points[0];
  }
  if (info.input_range.has_value()) {
    std::cout << " input_range=[" << info.input_range->first << ", " << info.input_range->second
              << "]";
  }
  std::cout << "\n";
}

std::vector<std::int8_t> image_codes(const std::string& path, const pcie::TensorInfo& ingress,
                                     const pcie::QuantParams& quant) {
  if (ingress.shape.size() != 3U || ingress.shape[2] < 1 || ingress.shape[2] > 3) {
    throw std::runtime_error("--image needs an HWC ingress with 1..3 channels, got " +
                             shape_string(ingress.shape));
  }
  if (!ingress.input_range.has_value()) {
    throw std::runtime_error("model input '" + ingress.name + "' declares no input_range");
  }
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("failed to read image: " + path);
  }
  cv::resize(image, image,
             cv::Size(static_cast<int>(ingress.shape[1]), static_cast<int>(ingress.shape[0])));
  if (ingress.shape[2] == 3) {
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  } else {
    cv::Mat yuv;
    cv::cvtColor(image, yuv, cv::COLOR_BGR2YUV);
    std::vector<cv::Mat> planes;
    cv::split(yuv, planes);
    image = ingress.shape[2] == 1 ? planes[0] : cv::Mat();
    if (image.empty()) {
      cv::merge(std::vector<cv::Mat>{planes[1], planes[2]}, image);
    }
  }
  const auto [lo, hi] = *ingress.input_range;
  std::vector<std::int8_t> codes(image.total() * image.channels());
  for (std::size_t i = 0; i < codes.size(); ++i) {
    const double value = lo + (static_cast<double>(image.data[i]) / 255.0) * (hi - lo);
    codes[i] = quantize(static_cast<float>(value), quant);
  }
  return codes;
}

std::vector<std::int8_t> ramp_codes(const std::size_t count) {
  std::vector<std::int8_t> codes(count);
  for (std::size_t i = 0; i < count; ++i) {
    codes[i] = static_cast<std::int8_t>(static_cast<std::uint8_t>((i * 7U + 3U) & 0xffU));
  }
  return codes;
}

struct Inputs {
  pcie::TensorList int8; ///< What the mla_only route consumes: one dense INT8 tensor per input.
  std::vector<std::vector<float>> fp32; ///< The same values dequantized, for the default route.
};

Inputs prepare_inputs(const pcie::ModelInfo& info, const std::string& image) {
  std::cout << "input: " << (image.empty() ? "synthetic INT8 ramp" : image) << "\n";
  Inputs inputs;
  for (const pcie::TensorInfo& ingress : info.inputs) {
    const pcie::QuantParams& quant = require_quant(ingress);
    const std::size_t count = element_count(ingress.shape);
    if (ingress.dtype != "INT8" || ingress.size_bytes != count) {
      throw std::runtime_error("mla_only input '" + ingress.name + "' is not a dense INT8 tensor");
    }
    std::vector<std::int8_t> codes =
        image.empty() ? ramp_codes(count) : image_codes(image, ingress, quant);
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
      values[i] = dequantize(codes[i], quant);
      if (quantize(values[i], quant) != codes[i]) {
        throw std::runtime_error("host quantizer does not round-trip INT8 code " +
                                 std::to_string(static_cast<int>(codes[i])));
      }
    }
    std::cout << "  submitting '" << ingress.name << "' " << shape_string(ingress.shape) << "\n";
    inputs.int8.push_back(pcie::Tensor::from_vector(std::move(codes), ingress.shape, ingress.name));
    inputs.fp32.push_back(std::move(values));
  }
  return inputs;
}

struct Head {
  std::vector<std::int64_t> shape;
  std::vector<float> values;
};

// The reference: the same values through the default route, where the card quantizes and
// dequantizes on the CVU. Its FP32 heads are what the mla_only route has to reproduce.
std::map<std::string, Head> run_default_route(const Args& args, const pcie::ConnectionOptions& conn,
                                              const pcie::ModelInfo& mla_only_info,
                                              const std::vector<std::vector<float>>& inputs) {
  std::cout << "default route: card quantizes and dequantizes\n";
  pcie::Model model(args.model, {}, conn);
  pcie::test::SignalCloseGuard guard(model);
  model.build(args.readiness_timeout_ms);
  const pcie::ModelInfo info = model.info();
  if (info.inputs.size() != mla_only_info.inputs.size()) {
    throw std::runtime_error("default route exposes " + std::to_string(info.inputs.size()) +
                             " input(s), mla_only exposes " +
                             std::to_string(mla_only_info.inputs.size()));
  }
  pcie::TensorList submitted;
  for (std::size_t i = 0; i < info.inputs.size(); ++i) {
    const auto& shape = mla_only_info.inputs[i].shape;
    if (info.inputs[i].dtype != "FP32" || info.inputs[i].shape != shape) {
      throw std::runtime_error("default route input " + std::to_string(i) + " is not FP32 " +
                               shape_string(shape));
    }
    submitted.push_back(pcie::Tensor::from_vector(inputs[i], shape, info.inputs[i].name));
  }
  std::map<std::string, Head> heads;
  for (const auto& output : model.run(submitted, args.pull_timeout_ms)) {
    const std::size_t count = element_count(output.shape);
    if (output.dtype != pcie::TensorDType::Float32 || output.size_bytes != count * sizeof(float)) {
      throw std::runtime_error("default route output '" + output.route.name +
                               "' is not a dense FP32 tensor");
    }
    Head head{output.shape, std::vector<float>(count)};
    std::memcpy(head.values.data(), tensor_bytes(output), output.size_bytes);
    heads.emplace(output.route.name, std::move(head));
  }
  model.close();
  std::cout << "  " << heads.size() << " FP32 head(s)\n";
  return heads;
}

// What the mla_only route promises about its outputs: the heads info() described, as dense
// contiguous INT8 tensors that share one owning block - the HWC16 padding is never visible.
void check_outputs(const pcie::TensorList& outputs, const pcie::ModelInfo& info) {
  if (outputs.size() != info.outputs.size()) {
    throw std::runtime_error("output count mismatch: got " + std::to_string(outputs.size()) +
                             " expected " + std::to_string(info.outputs.size()));
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& output = outputs[i];
    const auto& spec = info.outputs[i];
    const std::string where = " at output " + std::to_string(i) + " ('" + spec.name + "')";
    if (output.route.name != spec.name || output.shape != spec.shape) {
      throw std::runtime_error("name or shape mismatch" + where);
    }
    const std::size_t dense = element_count(spec.shape);
    if (output.dtype != pcie::TensorDType::Int8 || output.size_bytes != dense ||
        spec.size_bytes != dense) {
      throw std::runtime_error("output is not dense INT8" + where);
    }
    std::int64_t stride = 1;
    for (std::size_t d = spec.shape.size(); d-- > 0; stride *= spec.shape[d]) {
      if (output.strides_bytes[d] != stride) {
        throw std::runtime_error("output is not contiguous" + where);
      }
    }
    if (!output.data || !output.owner || output.owner != outputs[0].owner) {
      throw std::runtime_error("outputs do not share one owning dense block" + where);
    }
  }
}

void require_identical(const pcie::TensorList& first, const pcie::TensorList& second) {
  for (std::size_t i = 0; i < first.size(); ++i) {
    if (first[i].size_bytes != second[i].size_bytes ||
        std::memcmp(tensor_bytes(first[i]), tensor_bytes(second[i]), first[i].size_bytes) != 0) {
      throw std::runtime_error("two runs of the same input differ at output '" +
                               first[i].route.name + "'");
    }
  }
  std::cout << "  two runs are byte-identical across " << first.size() << " head(s)\n";
}

// Dequantize on the host and compare with the reference. max_err is the largest deviation in
// quantization steps; `identical` is the fraction of elements that match bit for bit, which
// only stays at 1.0 while the published scale equals the card's own float32 reciprocal.
void compare_with_reference(const pcie::TensorList& outputs, const pcie::ModelInfo& info,
                            const std::map<std::string, Head>& reference) {
  std::cout << "accuracy vs default route (error in units of each head's scale)\n";
  bool ok = true;
  for (std::size_t i = 0; i < info.outputs.size(); ++i) {
    const auto& spec = info.outputs[i];
    const auto& quant = require_quant(spec);
    const auto it = reference.find(spec.name);
    if (it == reference.end() || it->second.shape != spec.shape) {
      throw std::runtime_error("default route has no matching output '" + spec.name + "'");
    }
    const auto* codes = reinterpret_cast<const std::int8_t*>(tensor_bytes(outputs[i]));
    const std::vector<float>& card = it->second.values;
    double max_error = 0.0;
    std::size_t identical = 0;
    for (std::size_t e = 0; e < card.size(); ++e) {
      const float host = dequantize(codes[e], quant);
      identical += host == card[e] ? 1U : 0U;
      const double error = std::fabs(static_cast<double>(host) - card[e]) / quant.scales[0];
      if (!(error <= max_error)) { // also promotes NaN
        max_error = error;
      }
    }
    const bool head_ok = max_error <= kMaxErrorScales;
    ok = ok && head_ok;
    std::cout << "  " << (head_ok ? "ok  " : "FAIL") << " max_err=" << std::setw(8) << std::fixed
              << std::setprecision(4) << max_error << " identical=" << std::setw(8)
              << std::setprecision(5) << static_cast<double>(identical) / card.size()
              << " elements=" << std::setw(8) << card.size() << " " << spec.name << "\n";
  }
  if (!ok) {
    throw std::runtime_error("mla_only outputs deviate from the default route");
  }
}

// The route accepts nothing but the INT8 ingress; the application cannot fall back to FP32.
void expect_fp32_rejected(pcie::Model& model, const pcie::ModelInfo& info,
                          const std::vector<float>& fp32) {
  try {
    (void)model.push(pcie::Tensor::from_vector(fp32, info.inputs[0].shape, info.inputs[0].name));
  } catch (const std::exception& e) {
    std::cout << "FP32 push rejected: " << e.what() << "\n";
    return;
  }
  throw std::runtime_error("FP32 push to the mla_only model was accepted");
}

bool not_mla_only_capable(const std::string& reason) {
  for (const char* needle :
       {"does not support stage", "must be INT8", "hybrid host/card quantization"}) {
    if (reason.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    if (args.model.empty()) {
      std::cout << "SKIP: SIMAPCIE_YOLOV8_MODEL is not set and no --model was given\n";
      return kSkipExitCode;
    }
    std::fesetround(FE_TONEAREST); // the card rounds to nearest; keep the host quantizer in step

    pcie::ConnectionOptions conn;
    conn.card_host = args.card_host;
    conn.card_id = args.card_id;
    conn.user = args.user;
    conn.queue = args.queue;
    conn.card_env = args.card_env;
    conn.card_gst_debug = args.card_gst_debug;
    conn.card_gst_debug_file = args.card_gst_debug_file;

    pcie::ModelOptions options;
    options.mla_only = true;
    std::unique_ptr<pcie::Model> model;
    try {
      model = std::make_unique<pcie::Model>(args.model, options, conn);
    } catch (const std::exception& e) {
      if (!not_mla_only_capable(e.what())) {
        throw;
      }
      std::cout << "SKIP: " << args.model << " is not an MLA-only capable build: " << e.what()
                << "\n";
      return kSkipExitCode;
    }
    pcie::test::SignalCloseGuard guard(*model);

    std::cout << "PCIe MLA-only tensor run test\n  model=" << args.model
              << "\n  card_host=" << conn.card_host << " card_id=" << conn.card_id
              << " queue=" << conn.queue << "\n";
    const pcie::ModelInfo info = model->info();
    std::cout << "mla_only model metadata\n";
    for (std::size_t i = 0; i < info.inputs.size(); ++i) {
      print_tensor_info("input", i, info.inputs[i]);
    }
    for (std::size_t i = 0; i < info.outputs.size(); ++i) {
      print_tensor_info("output", i, info.outputs[i]);
    }

    const Inputs inputs = prepare_inputs(info, args.image);
    const std::map<std::string, Head> reference = run_default_route(args, conn, info, inputs.fp32);

    std::cout << "mla_only route: host quantizes and dequantizes\n";
    model->build(args.readiness_timeout_ms);
    const pcie::TensorList first = model->run(inputs.int8, args.pull_timeout_ms);
    check_outputs(first, info);
    const pcie::TensorList second = model->run(inputs.int8, args.pull_timeout_ms);
    check_outputs(second, info);
    require_identical(first, second);
    compare_with_reference(first, info, reference);
    expect_fp32_rejected(*model, info, inputs.fp32[0]);

    model->close();
    std::cout << "done\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}

#include <simaai/neat/pcie/Model.h>

#include "AsyncTestRunner.h"
#include "SignalCloseGuard.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pcie = simaai::neat::pcie;

namespace {

constexpr int kSkipExitCode = 77;

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
  int max_inflight = env_int_or_default("SIMAPCIE_MAX_INFLIGHT", 10);
  int readiness_timeout_ms = env_int_or_default("SIMAPCIE_READINESS_TIMEOUT_MS", 180000);
  int pull_timeout_ms = env_int_or_default("SIMAPCIE_PULL_TIMEOUT_MS", 30000);
  int sync_iterations = env_int_or_default("SIMAPCIE_SYNC_ITERATIONS", 50);
  int iterations = env_int_or_default("SIMAPCIE_TENSOR_ITERATIONS",
                                      env_int_or_default("SIMAPCIE_TEST_ITERATIONS", 1000));
  double max_abs_error_scales = 0.05;
  std::string image = env_or_default("SIMAPCIE_TEST_IMAGE", "");
  std::string dump_raw;
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
               " [--queue n] [--max-inflight n] [--readiness-timeout-ms ms]"
               " [--pull-timeout-ms ms] [--sync-iterations n] [--iterations n]"
               " [--image path] [--max-abs-error-scales x] [--dump-raw dir]"
               " [--card-env 'NAME=VALUE ...'] [--card-gst-debug spec]"
               " [--card-gst-debug-file path]\n"
               "Without --model the archive comes from SIMAPCIE_YOLOV8_MODEL, without --image the\n"
               "input comes from SIMAPCIE_TEST_IMAGE or falls back to a synthetic ramp. Exits 77 "
               "when no\n"
               "model is configured, or when it is not an MLA-only capable build.\n";
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
    } else if (arg == "--max-inflight") {
      args.max_inflight = std::stoi(require_value(argc, argv, i, "--max-inflight"));
    } else if (arg == "--readiness-timeout-ms") {
      args.readiness_timeout_ms = std::stoi(require_value(argc, argv, i, "--readiness-timeout-ms"));
    } else if (arg == "--pull-timeout-ms") {
      args.pull_timeout_ms = std::stoi(require_value(argc, argv, i, "--pull-timeout-ms"));
    } else if (arg == "--sync-iterations") {
      args.sync_iterations = std::stoi(require_value(argc, argv, i, "--sync-iterations"));
    } else if (arg == "--iterations") {
      args.iterations = std::stoi(require_value(argc, argv, i, "--iterations"));
    } else if (arg == "--max-abs-error-scales") {
      args.max_abs_error_scales = std::stod(require_value(argc, argv, i, "--max-abs-error-scales"));
    } else if (arg == "--image") {
      args.image = require_value(argc, argv, i, "--image");
    } else if (arg == "--dump-raw") {
      args.dump_raw = require_value(argc, argv, i, "--dump-raw");
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
  if (args.max_inflight < 0 || args.max_inflight > 256 || args.readiness_timeout_ms <= 0 ||
      args.pull_timeout_ms <= 0 || args.sync_iterations < 0 || args.iterations < 0 ||
      args.max_abs_error_scales < 0.0) {
    throw std::runtime_error("max-inflight must be in range 0..256, timeouts must be positive, "
                             "iterations must be non-negative, and thresholds must be sane");
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

std::vector<std::int64_t> contiguous_int8_strides(const std::vector<std::int64_t>& shape) {
  std::vector<std::int64_t> strides(shape.size(), 0);
  std::int64_t stride = 1;
  for (std::size_t index = shape.size(); index > 0; --index) {
    strides[index - 1] = stride;
    stride *= shape[index - 1];
  }
  return strides;
}

bool model_is_not_mla_only_capable(const std::string& reason) {
  static constexpr std::string_view kIneligible[] = {
      "does not support stage", "supports exactly one model input", "must be INT8"};
  for (const std::string_view needle : kIneligible) {
    if (reason.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

const pcie::QuantParams& require_quant(const pcie::TensorInfo& info) {
  if (!info.quant.has_value() || info.quant->scales.empty() ||
      info.quant->scales.size() != info.quant->zero_points.size()) {
    throw std::runtime_error("tensor '" + info.name + "' has no usable quantization parameters");
  }
  return *info.quant;
}

std::size_t quant_channel(const pcie::QuantParams& quant, const std::vector<std::int64_t>& shape,
                          const std::size_t index) {
  if (quant.scales.size() == 1U) {
    return 0U;
  }
  const int rank = static_cast<int>(shape.size());
  const int axis = quant.axis < 0 ? quant.axis + rank : quant.axis;
  if (axis < 0 || axis >= rank || static_cast<std::size_t>(shape[axis]) != quant.scales.size()) {
    throw std::runtime_error("per-channel quantization axis does not match the tensor shape");
  }
  std::size_t inner = 1;
  for (int dim = axis + 1; dim < rank; ++dim) {
    inner *= static_cast<std::size_t>(shape[dim]);
  }
  return (index / inner) % quant.scales.size();
}

std::int8_t quantize(const float value, const float scale, const std::int32_t zero_point) {
  const float code = std::nearbyint(value / scale) + static_cast<float>(zero_point);
  return static_cast<std::int8_t>(std::clamp(code, -128.0f, 127.0f));
}

float dequantize(const std::int8_t code, const float scale, const std::int32_t zero_point) {
  return static_cast<float>(static_cast<std::int32_t>(code) - zero_point) * scale;
}

std::size_t distinct_codes(const std::vector<std::int8_t>& codes) {
  std::array<bool, 256> seen{};
  for (const std::int8_t code : codes) {
    seen[static_cast<std::uint8_t>(code)] = true;
  }
  return static_cast<std::size_t>(std::count(seen.begin(), seen.end(), true));
}

std::vector<std::int8_t> make_int8_pattern(const std::size_t count) {
  std::vector<std::int8_t> codes(count);
  for (std::size_t i = 0; i < count; ++i) {
    codes[i] = static_cast<std::int8_t>(static_cast<std::uint8_t>((i * 7U + 3U) & 0xffU));
  }
  return codes;
}

// One ingress plane from the image: luma for a single channel, chroma for two, RGB for three.
// A multi-input model splits one picture across several planes at different resolutions, so each
// ingress is resized on its own.
std::vector<std::int8_t> quantized_image(const std::string& path, const pcie::TensorInfo& ingress,
                                         const pcie::QuantParams& quant) {
  if (ingress.shape.size() != 3U || ingress.shape[2] < 1 || ingress.shape[2] > 3) {
    throw std::runtime_error("--image needs an HWC ingress with 1..3 channels, got " +
                             shape_string(ingress.shape));
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
    if (ingress.shape[2] == 1) {
      image = planes[0];
    } else {
      cv::merge(std::vector<cv::Mat>{planes[1], planes[2]}, image);
    }
  }
  // Map the image byte into the value range the model declares, so the full INT8 code range is
  // exercised whatever convention the model was compiled with: [0,1] normalized, raw [0,255], or
  // a mean/std range that does not start at zero.
  if (!ingress.input_range.has_value()) {
    throw std::runtime_error("model input '" + ingress.name + "' declares no input_range");
  }
  const auto [lo, hi] = *ingress.input_range;
  std::vector<std::int8_t> codes(image.total() * image.channels());
  for (std::size_t i = 0; i < codes.size(); ++i) {
    const float value =
        static_cast<float>(lo + (static_cast<double>(image.data[i]) / 255.0) * (hi - lo));
    codes[i] = quantize(value, quant.scales[0], quant.zero_points[0]);
  }
  return codes;
}

const std::uint8_t* tensor_bytes(const pcie::Tensor& tensor) {
  return static_cast<const std::uint8_t*>(tensor.data) + tensor.byte_offset;
}

void print_tensor_info(const char* kind, const std::size_t index, const pcie::TensorInfo& info) {
  std::cout << "  " << kind << "[" << index << "] name=" << info.name << " dtype=" << info.dtype
            << " shape=" << shape_string(info.shape) << " size_bytes=" << info.size_bytes;
  if (info.quant.has_value() && !info.quant->scales.empty()) {
    std::cout << " scale=" << info.quant->scales[0] << " zero_point=" << info.quant->zero_points[0]
              << (info.quant->scales.size() > 1U ? " (per-channel)" : "");
  }
  std::cout << "\n";
}

void validate_outputs(const pcie::TensorList& outputs,
                      const std::vector<pcie::TensorInfo>& expected) {
  if (outputs.size() != expected.size()) {
    throw std::runtime_error("output count mismatch: got " + std::to_string(outputs.size()) +
                             " expected " + std::to_string(expected.size()));
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto& output = outputs[i];
    const auto& spec = expected[i];
    const std::string where = " at output " + std::to_string(i) + " ('" + spec.name + "')";
    if (output.route.name != spec.name) {
      throw std::runtime_error("name mismatch" + where + ": got " + output.route.name);
    }
    if (output.shape != spec.shape) {
      throw std::runtime_error("shape mismatch" + where + ": got " + shape_string(output.shape));
    }
    if (output.dtype != pcie::TensorDType::Int8) {
      throw std::runtime_error("dtype is not INT8" + where);
    }
    const std::size_t dense = element_count(spec.shape);
    if (output.size_bytes != dense || spec.size_bytes != dense) {
      throw std::runtime_error("size_bytes is not the dense logical size" + where);
    }
    if (output.strides_bytes != contiguous_int8_strides(spec.shape)) {
      throw std::runtime_error("output is not contiguous" + where);
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
}

struct Head {
  std::vector<std::int64_t> shape;
  std::vector<float> values;
};

std::map<std::string, Head> run_default_route(const Args& args, const pcie::ConnectionOptions& conn,
                                              const std::vector<pcie::TensorInfo>& ingresses,
                                              const std::vector<std::vector<float>>& inputs) {
  std::cout << "phase A: default route (card quantizes and dequantizes)\n";
  pcie::Model model(args.model, {}, conn);
  pcie::test::SignalCloseGuard guard(model);
  model.build(args.readiness_timeout_ms);
  const pcie::ModelInfo info = model.info();
  if (info.inputs.size() != ingresses.size()) {
    throw std::runtime_error("default route exposes " + std::to_string(info.inputs.size()) +
                             " input(s), mla_only exposes " + std::to_string(ingresses.size()));
  }
  pcie::TensorList submitted;
  for (std::size_t i = 0; i < info.inputs.size(); ++i) {
    if (info.inputs[i].dtype != "FP32" || info.inputs[i].shape != ingresses[i].shape) {
      throw std::runtime_error("default route input " + std::to_string(i) + " is not FP32 " +
                               shape_string(ingresses[i].shape));
    }
    submitted.push_back(
        pcie::Tensor::from_vector(inputs[i], ingresses[i].shape, info.inputs[i].name));
  }
  const pcie::TensorList outputs = model.run(submitted, args.pull_timeout_ms);
  std::map<std::string, Head> heads;
  for (const auto& output : outputs) {
    const std::size_t count = element_count(output.shape);
    if (output.dtype != pcie::TensorDType::Float32 || output.size_bytes != count * sizeof(float)) {
      throw std::runtime_error("default route output '" + output.route.name +
                               "' is not a dense FP32 tensor");
    }
    Head head;
    head.shape = output.shape;
    head.values.resize(count);
    std::memcpy(head.values.data(), tensor_bytes(output), output.size_bytes);
    heads.emplace(output.route.name, std::move(head));
  }
  model.close();
  std::cout << "default route produced " << heads.size() << " FP32 head(s)\n";
  return heads;
}

void compare_with_reference(const pcie::TensorList& outputs,
                            const std::vector<pcie::TensorInfo>& specs,
                            const std::map<std::string, Head>& reference, const Args& args) {
  std::cout << "accuracy vs default route (error in units of each head's scale)\n";
  bool ok = true;
  for (std::size_t i = 0; i < specs.size(); ++i) {
    const auto& spec = specs[i];
    const auto& quant = require_quant(spec);
    const auto it = reference.find(spec.name);
    if (it == reference.end()) {
      throw std::runtime_error("default route has no output named '" + spec.name + "'");
    }
    if (it->second.shape != spec.shape) {
      throw std::runtime_error("shape differs between routes at '" + spec.name + "': " +
                               shape_string(it->second.shape) + " vs " + shape_string(spec.shape));
    }
    const auto* codes = reinterpret_cast<const std::int8_t*>(tensor_bytes(outputs[i]));
    const std::vector<float>& card = it->second.values;
    double max_error_scales = 0.0;
    std::size_t identical = 0;
    std::array<bool, 256> seen{};
    std::size_t distinct = 0;
    for (std::size_t e = 0; e < card.size(); ++e) {
      const auto code = static_cast<std::uint8_t>(codes[e]);
      distinct += seen[code] ? 0U : 1U;
      seen[code] = true;
      const std::size_t c = quant_channel(quant, spec.shape, e);
      const float host = dequantize(codes[e], quant.scales[c], quant.zero_points[c]);
      if (host == card[e]) {
        ++identical;
      }
      const double error = std::fabs(static_cast<double>(host) - card[e]) / quant.scales[c];
      if (!(error <= max_error_scales)) { // also promotes NaN
        max_error_scales = error;
      }
    }
    const double identical_fraction = static_cast<double>(identical) / card.size();
    const bool head_ok = max_error_scales <= args.max_abs_error_scales;
    ok = ok && head_ok;
    std::cout << "  " << (head_ok ? "ok  " : "FAIL") << " max_err=" << std::setw(8) << std::fixed
              << std::setprecision(4) << max_error_scales << " identical=" << std::setw(8)
              << std::setprecision(5) << identical_fraction << " elements=" << std::setw(8)
              << card.size() << " distinct=" << std::setw(3) << distinct
              << (distinct < 2U ? " vacuous " : " ") << spec.name << "\n";
  }
  if (!ok) {
    throw std::runtime_error("mla_only outputs deviate from the default route beyond "
                             "--max-abs-error-scales");
  }
}

void dump_raw(const std::filesystem::path& dir, const std::vector<std::int8_t>& input,
              const pcie::TensorList& outputs) {
  std::filesystem::create_directories(dir);
  const auto write = [&](std::string name, const void* data, const std::size_t size) {
    for (char& ch : name) {
      if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '.' && ch != '-') {
        ch = '_';
      }
    }
    std::ofstream out(dir / (name + ".int8"), std::ios::binary);
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!out) {
      throw std::runtime_error("failed to write " + (dir / name).string());
    }
  };
  write("input", input.data(), input.size());
  for (const auto& output : outputs) {
    write(output.route.name, tensor_bytes(output), output.size_bytes);
  }
  std::cout << "raw INT8 input and outputs written to " << dir << "\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    if (args.model.empty()) {
      std::cout << "SKIP: SIMAPCIE_YOLOV8_MODEL is not set and no --model was given\n";
      return kSkipExitCode;
    }
    std::fesetround(FE_TONEAREST);

    pcie::ConnectionOptions conn;
    conn.card_host = args.card_host;
    conn.card_id = args.card_id;
    conn.user = args.user;
    conn.queue = args.queue;
    conn.max_inflight = args.max_inflight;
    conn.card_env = args.card_env;
    conn.card_gst_debug = args.card_gst_debug;
    conn.card_gst_debug_file = args.card_gst_debug_file;

    pcie::ModelOptions mla_options;
    mla_options.mla_only = true;

    try {
      (void)pcie::Model(args.model, mla_options).info();
    } catch (const std::exception& e) {
      if (!model_is_not_mla_only_capable(e.what())) {
        throw;
      }
      std::cout << "SKIP: " << args.model << " is not an MLA-only capable build: " << e.what()
                << "\n";
      return kSkipExitCode;
    }

    std::cout << "PCIe MLA-only tensor run test\n  model=" << args.model << "\n  card_host="
              << (conn.card_host.empty() ? ("10.0." + std::to_string(conn.card_id) + ".2")
                                         : conn.card_host)
              << " queue=" << conn.queue << " max_inflight=" << conn.max_inflight
              << "\n  sync_iterations=" << args.sync_iterations
              << " async_iterations=" << args.iterations << "\n";

    pcie::Model model(args.model, mla_options, conn);
    pcie::test::SignalCloseGuard signal_guard(model);
    const pcie::ModelInfo info = model.info();
    std::cout << "mla_only model metadata\n";
    for (std::size_t i = 0; i < info.inputs.size(); ++i) {
      print_tensor_info("input", i, info.inputs[i]);
    }
    for (std::size_t i = 0; i < info.outputs.size(); ++i) {
      print_tensor_info("output", i, info.outputs[i]);
    }
    if (info.inputs.empty()) {
      throw std::runtime_error("mla_only model must expose at least one INT8 input");
    }
    for (const auto& input : info.inputs) {
      if (input.dtype != "INT8") {
        throw std::runtime_error("mla_only input '" + input.name + "' is not INT8");
      }
    }
    for (const auto& output : info.outputs) {
      if (output.dtype != "INT8") {
        throw std::runtime_error("mla_only output '" + output.name + "' is not INT8");
      }
      (void)require_quant(output);
    }

    std::cout << "input: " << (args.image.empty() ? "synthetic INT8 ramp" : args.image) << "\n";
    std::vector<std::vector<std::int8_t>> codes(info.inputs.size());
    std::vector<std::vector<float>> fp32(info.inputs.size());
    std::vector<std::int8_t> packed;
    for (std::size_t index = 0; index < info.inputs.size(); ++index) {
      const pcie::TensorInfo& ingress = info.inputs[index];
      const pcie::QuantParams& input_quant = require_quant(ingress);
      if (input_quant.scales.size() != 1U) {
        throw std::runtime_error("per-channel input quantization is not covered by this test");
      }
      const std::size_t count = element_count(ingress.shape);
      if (ingress.size_bytes != count) {
        throw std::runtime_error("mla_only input '" + ingress.name +
                                 "' size_bytes is not the dense INT8 size");
      }
      codes[index] = args.image.empty() ? make_int8_pattern(count)
                                        : quantized_image(args.image, ingress, input_quant);
      fp32[index].resize(count);
      for (std::size_t i = 0; i < count; ++i) {
        fp32[index][i] =
            dequantize(codes[index][i], input_quant.scales[0], input_quant.zero_points[0]);
        if (quantize(fp32[index][i], input_quant.scales[0], input_quant.zero_points[0]) !=
            codes[index][i]) {
          throw std::runtime_error("host quantizer does not round-trip INT8 code " +
                                   std::to_string(static_cast<int>(codes[index][i])));
        }
      }
      packed.insert(packed.end(), codes[index].begin(), codes[index].end());
      std::cout << "  submitting '" << ingress.name << "' " << shape_string(ingress.shape) << " "
                << count << " code(s), " << distinct_codes(codes[index]) << " distinct\n";
    }

    const std::map<std::string, Head> reference = run_default_route(args, conn, info.inputs, fp32);

    std::cout << "phase B: mla_only route (host quantizes and dequantizes)\n";
    const auto started = std::chrono::steady_clock::now();
    model.build(args.readiness_timeout_ms);
    std::cout << "build completed in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - started)
                     .count()
              << " ms\n";

    pcie::TensorList input;
    for (std::size_t index = 0; index < info.inputs.size(); ++index) {
      input.push_back(pcie::Tensor::from_vector(codes[index], info.inputs[index].shape,
                                                info.inputs[index].name));
    }
    const pcie::TensorList first = model.run(input, args.pull_timeout_ms);
    validate_outputs(first, info.outputs);
    if (!args.dump_raw.empty()) {
      dump_raw(args.dump_raw, packed, first);
    }
    const pcie::TensorList second = model.run(input, args.pull_timeout_ms);
    validate_outputs(second, info.outputs);
    require_identical(first, second);
    std::cout << "two runs are byte-identical across " << first.size() << " head(s)\n";
    compare_with_reference(first, info.outputs, reference, args);

    for (int iteration = 1; iteration <= args.sync_iterations; ++iteration) {
      if (!model.push(input)) {
        throw std::runtime_error("sync push returned false at iteration " +
                                 std::to_string(iteration));
      }
      const auto result = model.pull(args.pull_timeout_ms);
      if (!result.has_value()) {
        throw std::runtime_error("sync pull timed out at iteration " + std::to_string(iteration));
      }
      validate_outputs(*result, info.outputs);
    }
    std::cout << "completed " << args.sync_iterations << " sync push/pull iteration(s)\n";

    if (args.iterations > 0) {
      pcie::test::run_async_workers(
          [&] { model.close(); },
          [&](const std::atomic_bool& cancelled) {
            for (int iteration = 1; iteration <= args.iterations && !cancelled.load();
                 ++iteration) {
              if (!model.push(input)) {
                throw std::runtime_error("async push returned false at iteration " +
                                         std::to_string(iteration));
              }
            }
          },
          [&](const std::atomic_bool& cancelled) {
            for (int iteration = 1; iteration <= args.iterations && !cancelled.load();
                 ++iteration) {
              const auto result = model.pull(args.pull_timeout_ms);
              if (!result.has_value()) {
                throw std::runtime_error("async pull timed out at iteration " +
                                         std::to_string(iteration));
              }
              validate_outputs(*result, info.outputs);
            }
          });
      std::cout << "completed " << args.iterations << " async push/pull iteration(s)\n";
    }

    {
      bool accepted = true;
      std::string message;
      try {
        (void)model.push(
            pcie::Tensor::from_vector(fp32[0], info.inputs[0].shape, info.inputs[0].name));
      } catch (const std::exception& e) {
        accepted = false;
        message = e.what();
      }
      if (accepted) {
        throw std::runtime_error("FP32 push to the mla_only model was accepted");
      }
      std::cout << "FP32 push rejected: " << message << "\n";
    }

    model.close();
    std::cout << "done\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}

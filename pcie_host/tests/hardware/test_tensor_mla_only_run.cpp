// Proves the mla_only route against the default route on the same input: the host quantizes (or
// casts to BF16), the card runs only the MLA, the host converts back, and every head must match
// what the card produces when it converts itself. It doubles as the reference for using
// ModelOptions::mla_only from an application.
#include <simaai/neat/pcie/Model.h>

#include "SignalCloseGuard.h"

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
#include <stdexcept>
#include <string>
#include <vector>

namespace pcie = simaai::neat::pcie;

namespace {

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
  std::string model =
      env_or_default("SIMAPCIE_MLA_ONLY_MODEL",
                     env_or_default("SIMAPCIE_YOLOV8_MODEL", DEFAULT_MODEL_PATH).c_str());
  std::string card_host = env_or_default("SIMAPCIE_CARD_HOST", "");
  std::string user = env_or_default("SIMAPCIE_USER", "sima");
  int card_id = env_int_or_default("SIMAPCIE_CARD_ID", 0);
  int queue = env_int_or_default("SIMAPCIE_QUEUE", 0);
  int readiness_timeout_ms = env_int_or_default("SIMAPCIE_READINESS_TIMEOUT_MS", 180000);
  int pull_timeout_ms = env_int_or_default("SIMAPCIE_PULL_TIMEOUT_MS", 30000);
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
               " [--queue n] [--card-env 'NAME=VALUE ...']"
               " [--card-gst-debug spec] [--card-gst-debug-file path]\n"
               "Without --model the archive comes from SIMAPCIE_MLA_ONLY_MODEL, then\n"
               "SIMAPCIE_YOLOV8_MODEL. The archive has to be an MLA-only capable build.\n";
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
  if (args.model.empty()) {
    throw std::runtime_error("neither SIMAPCIE_MLA_ONLY_MODEL nor SIMAPCIE_YOLOV8_MODEL is set "
                             "and no --model was given");
  }
  if (!std::filesystem::is_regular_file(args.model)) {
    throw std::runtime_error("model path does not exist or is not a regular file: " + args.model);
  }
  if (args.readiness_timeout_ms <= 0 || args.pull_timeout_ms <= 0) {
    throw std::runtime_error("timeouts must be positive");
  }
  return args;
}

std::size_t element_count(const std::vector<std::int64_t>& shape) {
  std::size_t count = shape.empty() ? 0U : 1U;
  for (const auto dim : shape) {
    if (dim <= 0) {
      throw std::runtime_error("tensor shape has a non-positive dimension");
    }
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

const std::uint8_t* tensor_bytes(const pcie::Tensor& tensor) {
  return static_cast<const std::uint8_t*>(tensor.data) + tensor.byte_offset;
}

bool is_int8(const pcie::TensorInfo& info) {
  if (info.dtype != "INT8" && info.dtype != "BF16") {
    throw std::runtime_error("tensor '" + info.name + "' has unsupported dtype " + info.dtype);
  }
  return info.dtype == "INT8";
}

// Every MLA-tessellated INT8 archive seen so far quantizes per tensor; per-axis parameters would
// need a channel lookup this reference does not implement.
const pcie::QuantParams& require_quant(const pcie::TensorInfo& info) {
  if (!info.quant.has_value() || info.quant->scales.size() != 1U ||
      info.quant->zero_points.size() != 1U) {
    throw std::runtime_error("tensor '" + info.name + "' needs per-tensor quantization parameters");
  }
  return *info.quant;
}

float bf16_to_float(const std::uint16_t code) {
  const std::uint32_t bits = static_cast<std::uint32_t>(code) << 16;
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint16_t float_to_bf16(const float value) { // round to nearest even, like the card's cast
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
}

// Element `i` of a published head as FP32: dequantized INT8 or widened BF16.
float decode(const pcie::TensorInfo& spec, const std::uint8_t* codes, const std::size_t i) {
  if (is_int8(spec)) {
    const pcie::QuantParams& quant = require_quant(spec);
    return static_cast<float>(static_cast<std::int8_t>(codes[i]) - quant.zero_points[0]) *
           quant.scales[0];
  }
  return bf16_to_float(reinterpret_cast<const std::uint16_t*>(codes)[i]);
}

// The head's resolution at `value`: one quantization step, or one BF16 ulp.
double resolution(const pcie::TensorInfo& spec, const float value) {
  return is_int8(spec) ? spec.quant->scales[0] : std::ldexp(1.0, std::ilogb(value) - 7);
}

struct Inputs {
  pcie::TensorList codes; ///< What the mla_only route consumes: one dense tensor per input.
  std::vector<std::vector<float>> fp32; ///< The same values as FP32, for the default route.
};

// A synthetic ramp over the input codes, and its exact FP32 image for the default route.
Inputs prepare_inputs(const pcie::ModelInfo& info) {
  Inputs inputs;
  for (const pcie::TensorInfo& ingress : info.inputs) {
    const std::size_t count = element_count(ingress.shape);
    const bool int8 = is_int8(ingress);
    if (ingress.size_bytes != count * (int8 ? 1U : 2U)) {
      throw std::runtime_error("mla_only input '" + ingress.name + "' is not dense");
    }
    std::vector<float> values(count);
    pcie::Tensor tensor;
    if (int8) {
      std::vector<std::int8_t> codes(count);
      for (std::size_t i = 0; i < count; ++i) {
        codes[i] = static_cast<std::int8_t>(static_cast<std::uint8_t>((i * 7U + 3U) & 0xffU));
        values[i] = decode(ingress, reinterpret_cast<const std::uint8_t*>(codes.data()), i);
      }
      tensor = pcie::Tensor::from_vector(std::move(codes), ingress.shape, ingress.name);
    } else {
      std::vector<std::uint16_t> codes(count);
      for (std::size_t i = 0; i < count; ++i) {
        codes[i] = float_to_bf16(static_cast<float>((i * 7U + 3U) & 0xffU) / 255.0f);
        values[i] = bf16_to_float(codes[i]);
      }
      tensor = pcie::Tensor::from_vector(std::move(codes), ingress.shape, ingress.name);
      tensor.dtype = pcie::TensorDType::BFloat16;
    }
    std::cout << "  submitting '" << ingress.name << "' (" << count << " " << ingress.dtype
              << " codes)\n";
    inputs.codes.push_back(std::move(tensor));
    inputs.fp32.push_back(std::move(values));
  }
  return inputs;
}

struct Head {
  std::vector<std::int64_t> shape;
  std::vector<float> values;
};

// The reference: the same values through the default route, where the card quantizes and
// dequantizes on the CVU. Its FP32 heads are what the mla_only route has to reproduce. The two
// routes may list a multi-input model's inputs in different orders, so match them by name.
std::map<std::string, Head> run_default_route(const Args& args, const pcie::ConnectionOptions& conn,
                                              const pcie::ModelInfo& mla_only_info,
                                              const std::vector<std::vector<float>>& inputs) {
  std::cout << "default route: card quantizes and dequantizes\n";
  pcie::Model model(args.model, {}, conn);
  pcie::test::SignalCloseGuard guard(model);
  model.build(args.readiness_timeout_ms);
  pcie::TensorList submitted;
  for (const pcie::TensorInfo& ingress : model.info().inputs) {
    const auto match = std::find_if(
        mla_only_info.inputs.begin(), mla_only_info.inputs.end(),
        [&](const pcie::TensorInfo& candidate) { return candidate.name == ingress.name; });
    if (match == mla_only_info.inputs.end() || ingress.dtype != "FP32" ||
        ingress.shape != match->shape) {
      throw std::runtime_error("default route input '" + ingress.name +
                               "' does not match the mla_only ingress");
    }
    const auto index = static_cast<std::size_t>(match - mla_only_info.inputs.begin());
    submitted.push_back(pcie::Tensor::from_vector(inputs[index], match->shape, ingress.name));
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
  return heads;
}

// What the mla_only route promises about its outputs: the heads info() described, as dense
// contiguous tensors that share one owning block - the transport padding is never visible.
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
    const bool int8 = is_int8(spec);
    const std::size_t dense = element_count(spec.shape) * (int8 ? 1U : 2U);
    if (output.dtype != (int8 ? pcie::TensorDType::Int8 : pcie::TensorDType::BFloat16) ||
        output.size_bytes != dense || spec.size_bytes != dense) {
      throw std::runtime_error("output is not dense " + spec.dtype + where);
    }
    std::int64_t stride = int8 ? 1 : 2;
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

// Convert on the host and compare with the reference; max_err is the largest deviation in units
// of the head's resolution (quantization step or BF16 ulp).
void compare_with_reference(const pcie::TensorList& outputs, const pcie::ModelInfo& info,
                            const std::map<std::string, Head>& reference) {
  std::cout << "accuracy vs default route (error in units of each head's resolution)\n";
  bool ok = true;
  for (std::size_t i = 0; i < info.outputs.size(); ++i) {
    const auto& spec = info.outputs[i];
    const auto it = reference.find(spec.name);
    if (it == reference.end() || it->second.shape != spec.shape) {
      throw std::runtime_error("default route has no matching output '" + spec.name + "'");
    }
    const std::uint8_t* codes = tensor_bytes(outputs[i]);
    const std::vector<float>& card = it->second.values;
    double max_error = 0.0;
    for (std::size_t e = 0; e < card.size(); ++e) {
      const float host = decode(spec, codes, e);
      if (host == card[e]) {
        continue;
      }
      const double error =
          std::fabs(static_cast<double>(host) - card[e]) / resolution(spec, card[e]);
      if (!(error <= max_error)) { // also promotes NaN
        max_error = error;
      }
    }
    const bool head_ok = max_error <= kMaxErrorScales;
    ok = ok && head_ok;
    std::cout << "  " << (head_ok ? "ok  " : "FAIL") << " max_err=" << std::setw(8) << std::fixed
              << std::setprecision(4) << max_error << " elements=" << std::setw(8) << card.size()
              << " " << spec.name << "\n";
  }
  if (!ok) {
    throw std::runtime_error("mla_only outputs deviate from the default route");
  }
}

// The route accepts nothing but the INT8 ingress; the application cannot fall back to FP32.
void expect_fp32_rejected(pcie::Model& model, const pcie::ModelInfo& info, const Inputs& inputs) {
  pcie::TensorList submitted = inputs.codes;
  submitted[0] =
      pcie::Tensor::from_vector(inputs.fp32[0], info.inputs[0].shape, info.inputs[0].name);
  try {
    (void)model.push(submitted);
  } catch (const std::exception& e) {
    std::cout << "FP32 push rejected: " << e.what() << "\n";
    return;
  }
  throw std::runtime_error("FP32 push to the mla_only model was accepted");
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
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
    pcie::Model model(args.model, options, conn);
    pcie::test::SignalCloseGuard guard(model);

    std::cout << "PCIe MLA-only tensor run test\n  model=" << args.model
              << "\n  card_host=" << conn.card_host << " card_id=" << conn.card_id
              << " queue=" << conn.queue << "\n";
    const pcie::ModelInfo info = model.info();
    const Inputs inputs = prepare_inputs(info);
    const std::map<std::string, Head> reference = run_default_route(args, conn, info, inputs.fp32);

    std::cout << "mla_only route: host quantizes and dequantizes\n";
    model.build(args.readiness_timeout_ms);
    const pcie::TensorList outputs = model.run(inputs.codes, args.pull_timeout_ms);
    check_outputs(outputs, info);
    compare_with_reference(outputs, info, reference);
    expect_fp32_rejected(model, info, inputs);

    model.close();
    std::cout << "done\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}

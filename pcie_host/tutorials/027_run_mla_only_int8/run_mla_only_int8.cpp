// Quantize on the host, run only the MLA over PCIe, and dequantize the INT8 results.
//
// Usage:
//   tutorial_027_run_mla_only_int8 --model yolo26n-det-int8-b1.tar.gz [--card 0]

#include <simaai/neat/pcie/Model.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pcie = simaai::neat::pcie;

namespace {

constexpr int kBuildTimeoutMs = 180000;
constexpr int kRunTimeoutMs = 30000;
constexpr double kMaxErrorScales = 0.05; // one twentieth of a quantization step
constexpr char kImagePath[] = "share/sima-pcie-host/tutorials/assets/street-scene.png";

struct Args {
  std::string model;
  int card_id = 0;
};

std::string require_value(int argc, char** argv, int& index, const char* option) {
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + option);
  }
  return argv[++index];
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--model") {
      args.model = require_value(argc, argv, index, "--model");
    } else if (arg == "--card") {
      args.card_id = std::stoi(require_value(argc, argv, index, "--card"));
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " --model yolo26n-det-int8-b1.tar.gz [--card 0]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (args.model.empty()) {
    throw std::runtime_error("--model is required: yolo26n-det-int8-b1.tar.gz from the Model Zoo");
  }
  return args;
}

std::string shape_string(const std::vector<std::int64_t>& shape) {
  std::string text = "[";
  for (std::size_t index = 0; index < shape.size(); ++index) {
    text += (index == 0 ? "" : ", ") + std::to_string(shape[index]);
  }
  return text + "]";
}

std::size_t element_count(const std::vector<std::int64_t>& shape) {
  std::size_t count = 1;
  for (const auto dim : shape) {
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

// The MLA-only contract quantizes every tensor with one scale and one zero point:
//   x = (q - zero_point) * scale
//   q = clamp(round(x / scale) + zero_point, -128, 127)
const pcie::QuantParams& require_quant(const pcie::TensorInfo& info) {
  if (!info.quant.has_value()) {
    throw std::runtime_error("tensor '" + info.name + "' publishes no quantization parameters");
  }
  return *info.quant;
}

std::int8_t quantize(const float value, const pcie::QuantParams& quant) {
  const float code = std::nearbyint(value / quant.scale) + static_cast<float>(quant.zero_point);
  return static_cast<std::int8_t>(std::clamp(code, -128.0F, 127.0F));
}

float dequantize(const std::int8_t code, const pcie::QuantParams& quant) {
  return static_cast<float>(static_cast<std::int32_t>(code) - quant.zero_point) * quant.scale;
}

const std::int8_t* int8_data(const pcie::Tensor& tensor) {
  return reinterpret_cast<const std::int8_t*>(static_cast<const std::uint8_t*>(tensor.data) +
                                              tensor.byte_offset);
}

// Preprocessing of the reference model: one RGB HWC image with pixels in [0, 1]. Resize, BGR to
// RGB, divide by 255, then quantize with the ingress parameters. Another model needs its own
// recipe.
std::vector<std::int8_t> quantize_image(const cv::Mat& bgr, const pcie::TensorInfo& input) {
  if (input.shape.size() != 3 || input.shape[2] != 3) {
    throw std::runtime_error("the reference model takes a three-channel HWC input, got " +
                             shape_string(input.shape));
  }
  const pcie::QuantParams& quant = require_quant(input);
  cv::Mat rgb;
  cv::resize(bgr, rgb,
             cv::Size(static_cast<int>(input.shape[1]), static_cast<int>(input.shape[0])));
  cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
  std::vector<std::int8_t> codes(rgb.total() * rgb.channels());
  for (std::size_t index = 0; index < codes.size(); ++index) {
    codes[index] = quantize(static_cast<float>(rgb.data[index] / 255.0), quant);
  }
  return codes;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    if (!std::filesystem::is_regular_file(args.model)) {
      throw std::runtime_error("model does not exist: " + args.model);
    }
    const cv::Mat image = cv::imread(kImagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error(std::string("OpenCV could not decode: ") + kImagePath);
    }
    pcie::ConnectionOptions connection;
    connection.card_id = args.card_id;

    // CORE LOGIC
    // STEP inspect-contract
    pcie::ModelOptions options;
    options.mla_only = true;
    pcie::Model model(args.model, options, connection);
    const pcie::ModelInfo info = model.info();
    if (info.inputs.size() != 1) {
      throw std::runtime_error("the reference model has exactly one input");
    }
    std::cout << "MLA-only contract:\n";
    for (const auto& input : info.inputs) {
      const pcie::QuantParams& quant = require_quant(input);
      std::cout << "  input " << input.name << " " << input.dtype << " "
                << shape_string(input.shape) << " scale=" << quant.scale
                << " zero_point=" << quant.zero_point << '\n';
    }
    for (const auto& output : info.outputs) {
      const pcie::QuantParams& quant = require_quant(output);
      std::cout << "  output " << output.name << " " << output.dtype << " "
                << shape_string(output.shape) << " scale=" << quant.scale
                << " zero_point=" << quant.zero_point << '\n';
    }
    // END STEP

    // STEP quantize-on-host
    const pcie::TensorInfo& ingress = info.inputs.front();
    std::vector<std::int8_t> codes = quantize_image(image, ingress);
    std::vector<float> fp32_input(codes.size());
    for (std::size_t index = 0; index < codes.size(); ++index) {
      fp32_input[index] = dequantize(codes[index], *ingress.quant);
    }
    pcie::TensorList int8_inputs;
    int8_inputs.push_back(pcie::Tensor::from_vector(std::move(codes), ingress.shape, ingress.name));
    // END STEP

    // STEP run-int8
    model.build(kBuildTimeoutMs);
    const pcie::TensorList outputs = model.run(int8_inputs, kRunTimeoutMs);
    model.close();
    // END STEP

    // STEP dequantize-and-compare
    pcie::Model reference(args.model, {}, connection);
    reference.build(kBuildTimeoutMs);
    pcie::TensorList fp32_tensors;
    fp32_tensors.push_back(
        pcie::Tensor::from_vector(std::move(fp32_input), ingress.shape, ingress.name));
    const pcie::TensorList reference_outputs = reference.run(fp32_tensors, kRunTimeoutMs);
    reference.close();

    std::cout << "Dequantized MLA-only outputs vs the default route (error in scale units):\n";
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const auto& spec = info.outputs[index];
      const pcie::QuantParams& quant = *spec.quant;
      const std::int8_t* codes = int8_data(outputs[index]);
      const auto& card = reference_outputs[index];
      if (card.route.name != spec.name || card.dtype != pcie::TensorDType::Float32) {
        throw std::runtime_error("default route output '" + spec.name + "' is not FP32");
      }
      const auto* card_values = reinterpret_cast<const float*>(
          static_cast<const std::uint8_t*>(card.data) + card.byte_offset);
      double max_error = 0.0;
      for (std::size_t element = 0; element < element_count(spec.shape); ++element) {
        const double host = dequantize(codes[element], quant);
        max_error = std::max(max_error, std::fabs(host - card_values[element]) / quant.scale);
      }
      std::cout << "  " << spec.name << " " << shape_string(spec.shape) << " max_err=" << std::fixed
                << std::setprecision(4) << max_error << '\n';
      if (max_error > kMaxErrorScales) {
        throw std::runtime_error("output '" + spec.name + "' deviates from the default route");
      }
    }
    // END STEP
    // END CORE LOGIC

    std::cout << "[OK] 027_run_mla_only_int8\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}

#include <simaai/neat/pcie/Model.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pcie = simaai::neat::pcie;

namespace {

struct Args {
  std::string model;
  std::string card_host;
  std::string user = "sima";
  int card_id = 0;
  int queue = 0;
  int readiness_timeout_ms = 180000;
  int pull_timeout_ms = 30000;
  bool shared = false;
};

std::string require_value(int argc, char** argv, int& i, const char* name) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + name);
  }
  return argv[++i];
}

void usage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " [--model model.tar.gz] [--card-host host] [--user user]"
               " [--card-id n] [--queue n] [--shared]"
               " [--readiness-timeout-ms ms] [--pull-timeout-ms ms]\n\n"
               "Without --model and --card-host, this prints the constructed tensor layout only.\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i] ? argv[i] : "";
    if (arg == "--model") {
      args.model = require_value(argc, argv, i, "--model");
    } else if (arg == "--card-host") {
      args.card_host = require_value(argc, argv, i, "--card-host");
    } else if (arg == "--user") {
      args.user = require_value(argc, argv, i, "--user");
    } else if (arg == "--card-id") {
      args.card_id = std::stoi(require_value(argc, argv, i, "--card-id"));
    } else if (arg == "--queue") {
      args.queue = std::stoi(require_value(argc, argv, i, "--queue"));
    } else if (arg == "--readiness-timeout-ms") {
      args.readiness_timeout_ms = std::stoi(require_value(argc, argv, i, "--readiness-timeout-ms"));
    } else if (arg == "--pull-timeout-ms") {
      args.pull_timeout_ms = std::stoi(require_value(argc, argv, i, "--pull-timeout-ms"));
    } else if (arg == "--shared") {
      args.shared = true;
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (!args.model.empty() && !std::filesystem::is_regular_file(args.model)) {
    throw std::runtime_error("model path does not exist: " + args.model);
  }
  return args;
}

std::string shape_string(const std::vector<std::int64_t>& shape) {
  std::string out = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      out += ", ";
    }
    out += std::to_string(shape[i]);
  }
  out += "]";
  return out;
}

std::size_t dtype_bytes(const pcie::TensorDType dtype) {
  switch (dtype) {
  case pcie::TensorDType::UInt8:
  case pcie::TensorDType::Int8:
    return 1;
  case pcie::TensorDType::UInt16:
  case pcie::TensorDType::Int16:
  case pcie::TensorDType::BFloat16:
    return 2;
  case pcie::TensorDType::Int32:
  case pcie::TensorDType::Float32:
    return 4;
  case pcie::TensorDType::Float64:
    return 8;
  }
  throw std::runtime_error("unsupported tensor dtype");
}

pcie::TensorDType dtype_from_fact(std::string dtype) {
  std::transform(dtype.begin(), dtype.end(), dtype.begin(),
                 [](const unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
  if (dtype == "UINT8")
    return pcie::TensorDType::UInt8;
  if (dtype == "INT8" || dtype == "EVXX_INT8" || dtype == "EV74_INT8")
    return pcie::TensorDType::Int8;
  if (dtype == "UINT16")
    return pcie::TensorDType::UInt16;
  if (dtype == "INT16")
    return pcie::TensorDType::Int16;
  if (dtype == "INT32")
    return pcie::TensorDType::Int32;
  if (dtype == "BF16" || dtype == "BFLOAT16" || dtype == "EVXX_BFLOAT16" ||
      dtype == "EV74_BFLOAT16")
    return pcie::TensorDType::BFloat16;
  if (dtype == "FP32" || dtype == "FLOAT32")
    return pcie::TensorDType::Float32;
  if (dtype == "FP64" || dtype == "FLOAT64")
    return pcie::TensorDType::Float64;
  throw std::runtime_error("unsupported tensor dtype from model facts: " + dtype);
}

std::vector<std::int64_t> contiguous_strides(const std::vector<std::int64_t>& shape,
                                             const std::size_t element_bytes) {
  std::vector<std::int64_t> strides(shape.size(), 0);
  std::int64_t stride = static_cast<std::int64_t>(element_bytes);
  for (std::size_t index = shape.size(); index > 0; --index) {
    const std::size_t dimension = index - 1;
    strides[dimension] = stride;
    stride *= shape[dimension];
  }
  return strides;
}

std::size_t logical_payload_bytes(const pcie::Tensor& tensor) {
  std::size_t count = 1;
  for (const auto dim : tensor.shape) {
    if (dim <= 0 ||
        count > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dim)) {
      throw std::runtime_error("invalid or overflowing tensor shape");
    }
    count *= static_cast<std::size_t>(dim);
  }
  const std::size_t element_bytes = dtype_bytes(tensor.dtype);
  if (count > std::numeric_limits<std::size_t>::max() / element_bytes) {
    throw std::runtime_error("tensor payload size overflows size_t");
  }
  return count * element_bytes;
}

pcie::Tensor make_single_contiguous_tensor() {
  constexpr std::int64_t height = 640;
  constexpr std::int64_t width = 640;
  constexpr std::int64_t channels = 3;
  std::vector<float> data(height * width * channels, 0.25F);
  pcie::Tensor tensor =
      pcie::Tensor::from_vector(std::move(data), {height, width, channels}, "images");
  tensor.route.logical_index = 0;
  tensor.route.physical_index = 0;
  tensor.route.route_slot = 0;
  return tensor;
}

pcie::TensorList make_shared_packed_tensors() {
  constexpr std::size_t first_count = 8 * 8;
  constexpr std::size_t second_count = 4 * 8;
  auto storage = std::make_shared<std::vector<float>>(first_count + second_count, 1.0F);

  pcie::Tensor first =
      pcie::Tensor::from_external(storage->data(), storage->size(), storage, {8, 8}, "input_0");
  first.route.logical_index = 0;
  first.route.physical_index = 0;
  first.route.route_slot = 0;

  pcie::Tensor second =
      pcie::Tensor::from_external(storage->data(), storage->size(), storage, {4, 8}, "input_1",
                                  static_cast<std::int64_t>(first_count * sizeof(float)));
  second.route.logical_index = 1;
  second.route.physical_index = 1;
  second.route.route_slot = 1;
  return {first, second};
}

pcie::TensorList make_shared_packed_tensors(const pcie::ModelInfo& info) {
  if (info.inputs.empty()) {
    throw std::runtime_error("model metadata contains no inputs");
  }

  struct InputLayout {
    pcie::TensorDType dtype;
    std::vector<std::int64_t> shape;
    std::size_t payload_bytes;
  };

  std::vector<InputLayout> layouts;
  layouts.reserve(info.inputs.size());
  std::size_t packed_bytes = 0;
  for (const auto& input : info.inputs) {
    const pcie::TensorDType dtype = dtype_from_fact(input.dtype);
    std::vector<std::int64_t> shape = input.shape;
    if (shape.empty()) {
      const std::size_t element_bytes = dtype_bytes(dtype);
      if (input.size_bytes == 0 || input.size_bytes % element_bytes != 0) {
        throw std::runtime_error("input tensor is missing a usable shape: " + input.name);
      }
      shape = {static_cast<std::int64_t>(input.size_bytes / element_bytes)};
    }

    pcie::Tensor probe;
    probe.dtype = dtype;
    probe.shape = shape;
    const std::size_t payload_bytes = logical_payload_bytes(probe);
    if (input.size_bytes != 0 && input.size_bytes != payload_bytes) {
      throw std::runtime_error("model input size does not match its dtype and shape: " +
                               input.name);
    }
    if (payload_bytes > std::numeric_limits<std::size_t>::max() - packed_bytes) {
      throw std::runtime_error("packed model inputs are too large");
    }
    layouts.push_back({dtype, std::move(shape), payload_bytes});
    packed_bytes += payload_bytes;
  }

  auto storage = std::make_shared<std::vector<std::uint8_t>>(packed_bytes, 0);
  pcie::TensorList tensors;
  tensors.reserve(layouts.size());
  std::size_t byte_offset = 0;
  for (std::size_t index = 0; index < layouts.size(); ++index) {
    const auto& layout = layouts[index];
    pcie::Tensor tensor;
    tensor.dtype = layout.dtype;
    tensor.shape = layout.shape;
    tensor.strides_bytes = contiguous_strides(layout.shape, dtype_bytes(layout.dtype));
    tensor.owner = storage;
    tensor.data = storage->data();
    tensor.size_bytes = storage->size();
    tensor.byte_offset = static_cast<std::int64_t>(byte_offset);
    tensor.route.name = info.inputs[index].name.empty() ? "input_" + std::to_string(index)
                                                        : info.inputs[index].name;
    tensor.route.logical_index = static_cast<int>(index);
    tensor.route.physical_index = static_cast<int>(index);
    tensor.route.route_slot = static_cast<int>(index);
    tensor.read_only = false;
    tensors.push_back(std::move(tensor));
    byte_offset += layout.payload_bytes;
  }
  return tensors;
}

void print_tensors(const pcie::TensorList& tensors, const std::string& expected_path) {
  std::cout << "expected pciehost path: " << expected_path << "\n";
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    std::cout << "  [" << i << "] name=" << tensor.route.name
              << " shape=" << shape_string(tensor.shape) << " byte_offset=" << tensor.byte_offset
              << " logical_payload_bytes=" << logical_payload_bytes(tensor)
              << " backing_size_bytes=" << tensor.size_bytes
              << " owner=" << (tensor.owner ? "yes" : "no") << "\n";
  }
}

void run_model(const Args& args) {
  pcie::ConnectionOptions connection;
  connection.card_host = args.card_host;
  connection.user = args.user;
  connection.card_id = args.card_id;
  connection.queue = args.queue;

  pcie::Model model(args.model, {}, connection);
  const pcie::TensorList tensors = args.shared ? make_shared_packed_tensors(model.info())
                                               : pcie::TensorList{make_single_contiguous_tensor()};
  print_tensors(tensors, args.shared ? (tensors.size() > 1U
                                            ? "model-aware shared packed tensor-set fast path"
                                            : "model-aware shared single-tensor fast path")
                                     : "single contiguous tensor fast path");
  model.build(args.readiness_timeout_ms);
  const pcie::TensorList outputs = model.run(tensors, args.pull_timeout_ms);
  model.close();
  std::cout << "received outputs: " << outputs.size() << "\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    if (!args.model.empty() && !args.card_host.empty()) {
      run_model(args);
      return 0;
    }

    const pcie::TensorList tensors = args.shared
                                         ? make_shared_packed_tensors()
                                         : pcie::TensorList{make_single_contiguous_tensor()};
    print_tensors(tensors, args.shared ? "shared packed tensor-set fast path"
                                       : "single contiguous tensor fast path");
    std::cout << "not running: pass --model and --card-host to push through PCIe\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}

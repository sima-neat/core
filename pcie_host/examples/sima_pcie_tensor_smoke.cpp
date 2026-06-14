#include "simaai/neat/pcie/SimaPCIeHost.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pcie = simaai::neat::pcie;

namespace {

struct Args {
  std::string model;
  std::string card_host;
  int queue = 0;
  int frames = 1;
};

std::string value(int argc, char** argv, int& i, const char* name) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + name);
  }
  return argv[++i];
}

Args parse(int argc, char** argv) {
  Args out;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model") {
      out.model = value(argc, argv, i, "--model");
    } else if (arg == "--card-host") {
      out.card_host = value(argc, argv, i, "--card-host");
    } else if (arg == "--queue") {
      out.queue = std::stoi(value(argc, argv, i, "--queue"));
    } else if (arg == "--frames") {
      out.frames = std::stoi(value(argc, argv, i, "--frames"));
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (out.model.empty() || !std::filesystem::exists(out.model)) {
    throw std::runtime_error("--model must point to a model archive");
  }
  return out;
}

pcie::TensorDType dtype_from_fact(const std::string& dtype) {
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
  throw std::runtime_error("unsupported input dtype: " + dtype);
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
  return 0;
}

std::vector<std::int64_t> contiguous_strides(const std::vector<std::int64_t>& shape,
                                             const std::size_t elem_size) {
  std::vector<std::int64_t> strides(shape.size(), 0);
  std::int64_t stride = static_cast<std::int64_t>(elem_size);
  for (std::size_t index = shape.size(); index > 0; --index) {
    const std::size_t dim = index - 1;
    strides[dim] = stride;
    stride *= shape[dim];
  }
  return strides;
}

pcie::Tensor make_zero_tensor(const pcie::TensorInfo& fact, const std::size_t fallback_index) {
  auto owner = std::make_shared<std::vector<std::uint8_t>>(fact.size_bytes, 0);
  pcie::Tensor tensor;
  tensor.dtype = dtype_from_fact(fact.dtype);
  tensor.shape = fact.shape.empty() ? std::vector<std::int64_t>{static_cast<std::int64_t>(fact.size_bytes)}
                                    : fact.shape;
  tensor.strides_bytes = contiguous_strides(tensor.shape, dtype_bytes(tensor.dtype));
  tensor.owner = owner;
  tensor.data = owner->data();
  tensor.size_bytes = owner->size();
  tensor.read_only = false;
  tensor.route.name = fact.name.empty() ? "input_" + std::to_string(fallback_index) : fact.name;
  tensor.route.logical_index = static_cast<int>(fallback_index);
  tensor.route.physical_index = static_cast<int>(fallback_index);
  return tensor;
}

pcie::TensorList make_zero_inputs(const pcie::ModelInfo& model_info) {
  pcie::TensorList inputs;
  inputs.reserve(model_info.inputs.size());
  for (std::size_t i = 0; i < model_info.inputs.size(); ++i) {
    inputs.push_back(make_zero_tensor(model_info.inputs[i], i));
  }
  return inputs;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse(argc, argv);

    pcie::ConnectionOptions conn;
    conn.card_host = args.card_host;
    conn.queue = args.queue;

    pcie::SimaPCIeHost host(conn);
    const auto model_info = host.init_pipeline(args.model);
    pcie::TensorList inputs = make_zero_inputs(model_info);

    for (int i = 0; i < args.frames; ++i) {
      const auto result = host.run(inputs, 30000);
      std::cout << "received tensors=" << result.size() << "\n";
    }
    host.stop();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}

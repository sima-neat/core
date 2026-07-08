#include <simaai/neat/pcie/Model.h>

#include "SignalCloseGuard.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pcie = simaai::neat::pcie;

namespace {

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
  { throw std::runtime_error(std::string("invalid integer in ") + name + ": " + value); }
}

struct Args {
  std::string model = env_or_default("SIMAPCIE_YOLOV8_MODEL", DEFAULT_MODEL_PATH);
  std::string card_host = env_or_default("SIMAPCIE_CARD_HOST", "");
  std::string user = env_or_default("SIMAPCIE_USER", "sima");
  int card_id = env_int_or_default("SIMAPCIE_CARD_ID", 0);
  int queue = env_int_or_default("SIMAPCIE_QUEUE", 0);
  int max_inflight = env_int_or_default("SIMAPCIE_MAX_INFLIGHT", 0);
  int readiness_timeout_ms = env_int_or_default("SIMAPCIE_READINESS_TIMEOUT_MS", 180000);
  int pull_timeout_ms = env_int_or_default("SIMAPCIE_PULL_TIMEOUT_MS", 30000);
  int burst_iterations = env_int_or_default("SIMAPCIE_BURST_ITERATIONS", 0);
  int sync_iterations = env_int_or_default("SIMAPCIE_SYNC_ITERATIONS", 50);
  int iterations = env_int_or_default("SIMAPCIE_TENSOR_ITERATIONS",
                                      env_int_or_default("SIMAPCIE_TEST_ITERATIONS", 1000));
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
            << " [--model model.tar.gz] [--card-host host] [--card-id n]"
               " [--user user] [--queue n] [--max-inflight n] [--readiness-timeout-ms ms]"
               " [--pull-timeout-ms ms] [--burst-iterations n]"
               " [--sync-iterations n] [--iterations n]"
               " [--card-env 'NAME=VALUE ...'] [--card-gst-debug spec]"
               " [--card-gst-debug-file path]\n";
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
    } else if (arg == "--burst-iterations") {
      args.burst_iterations = std::stoi(require_value(argc, argv, i, "--burst-iterations"));
    } else if (arg == "--sync-iterations") {
      args.sync_iterations = std::stoi(require_value(argc, argv, i, "--sync-iterations"));
    } else if (arg == "--iterations") {
      args.iterations = std::stoi(require_value(argc, argv, i, "--iterations"));
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

  if (!std::filesystem::is_regular_file(args.model)) {
    throw std::runtime_error("model path does not exist or is not a regular file: " + args.model);
  }
  if (args.max_inflight < 0 || args.max_inflight > 256 ||
      args.readiness_timeout_ms <= 0 || args.pull_timeout_ms <= 0 ||
      args.burst_iterations < 0 || args.sync_iterations < 0 || args.iterations < 0) {
    throw std::runtime_error(
        "max-inflight must be in range 0..256, timeouts must be positive, and iterations must be "
        "non-negative");
  }
  if (args.burst_iterations == 0 && args.sync_iterations == 0 && args.iterations == 0) {
    throw std::runtime_error("at least one of burst, sync, or async iterations must be positive");
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

std::string dtype_name(const pcie::TensorDType dtype) {
  switch (dtype) {
  case pcie::TensorDType::UInt8:
    return "UINT8";
  case pcie::TensorDType::Int8:
    return "INT8";
  case pcie::TensorDType::UInt16:
    return "UINT16";
  case pcie::TensorDType::Int16:
    return "INT16";
  case pcie::TensorDType::Int32:
    return "INT32";
  case pcie::TensorDType::BFloat16:
    return "BF16";
  case pcie::TensorDType::Float32:
    return "FP32";
  case pcie::TensorDType::Float64:
    return "FP64";
  }
  return "UNKNOWN";
}

pcie::TensorDType dtype_from_fact(std::string dtype) {
  std::transform(dtype.begin(), dtype.end(), dtype.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
  if (dtype == "UINT8") {
    return pcie::TensorDType::UInt8;
  }
  if (dtype == "INT8" || dtype == "EVXX_INT8" || dtype == "EV74_INT8") {
    return pcie::TensorDType::Int8;
  }
  if (dtype == "UINT16") {
    return pcie::TensorDType::UInt16;
  }
  if (dtype == "INT16") {
    return pcie::TensorDType::Int16;
  }
  if (dtype == "INT32") {
    return pcie::TensorDType::Int32;
  }
  if (dtype == "BF16" || dtype == "BFLOAT16" || dtype == "EVXX_BFLOAT16" ||
      dtype == "EV74_BFLOAT16") {
    return pcie::TensorDType::BFloat16;
  }
  if (dtype == "FP32" || dtype == "FLOAT32") {
    return pcie::TensorDType::Float32;
  }
  if (dtype == "FP64" || dtype == "FLOAT64") {
    return pcie::TensorDType::Float64;
  }
  throw std::runtime_error("unsupported tensor dtype from model facts: " + dtype);
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

std::size_t dense_size_bytes(const std::vector<std::int64_t>& shape,
                             const pcie::TensorDType dtype) {
  const std::size_t elem_size = dtype_bytes(dtype);
  if (elem_size == 0 || shape.empty()) {
    return 0;
  }
  std::size_t bytes = elem_size;
  for (const auto dim : shape) {
    if (dim < 0) {
      return 0;
    }
    const auto u_dim = static_cast<std::size_t>(dim);
    if (u_dim != 0 && bytes > static_cast<std::size_t>(-1) / u_dim) {
      return 0;
    }
    bytes *= u_dim;
  }
  return bytes;
}

void fill_pattern(std::vector<std::uint8_t>* bytes, const pcie::TensorDType dtype,
                  const std::size_t tensor_index) {
  if (!bytes || bytes->empty()) {
    return;
  }
  if (dtype == pcie::TensorDType::Float32) {
    const std::size_t count = bytes->size() / sizeof(float);
    auto* data = reinterpret_cast<float*>(bytes->data());
    for (std::size_t i = 0; i < count; ++i) {
      data[i] = static_cast<float>((i + tensor_index) % 255U) / 255.0f;
    }
    return;
  }
  if (dtype == pcie::TensorDType::Float64) {
    const std::size_t count = bytes->size() / sizeof(double);
    auto* data = reinterpret_cast<double*>(bytes->data());
    for (std::size_t i = 0; i < count; ++i) {
      data[i] = static_cast<double>((i + tensor_index) % 255U) / 255.0;
    }
    return;
  }
  for (std::size_t i = 0; i < bytes->size(); ++i) {
    (*bytes)[i] = static_cast<std::uint8_t>((i + tensor_index * 17U) & 0xffU);
  }
}

pcie::Tensor make_input_tensor(const pcie::TensorInfo& info, const std::size_t index) {
  pcie::Tensor tensor;
  tensor.dtype = dtype_from_fact(info.dtype);
  tensor.shape = info.shape.empty()
                     ? std::vector<std::int64_t>{static_cast<std::int64_t>(info.size_bytes)}
                     : info.shape;
  tensor.strides_bytes = contiguous_strides(tensor.shape, dtype_bytes(tensor.dtype));

  const std::size_t dense_bytes = dense_size_bytes(tensor.shape, tensor.dtype);
  const std::size_t payload_bytes = info.size_bytes != 0 ? info.size_bytes : dense_bytes;
  if (payload_bytes == 0) {
    throw std::runtime_error("input tensor has zero payload size: " + info.name);
  }

  auto owner = std::make_shared<std::vector<std::uint8_t>>(payload_bytes, 0);
  fill_pattern(owner.get(), tensor.dtype, index);

  tensor.owner = owner;
  tensor.data = owner->data();
  tensor.size_bytes = owner->size();
  tensor.byte_offset = 0;
  tensor.read_only = false;
  tensor.route.name = info.name.empty() ? "input_" + std::to_string(index) : info.name;
  tensor.route.logical_index = static_cast<int>(index);
  tensor.route.physical_index = static_cast<int>(index);
  tensor.route.route_slot = static_cast<int>(index);
  return tensor;
}

pcie::TensorList make_inputs(const pcie::ModelInfo& info) {
  pcie::TensorList inputs;
  inputs.reserve(info.inputs.size());
  for (std::size_t i = 0; i < info.inputs.size(); ++i) {
    inputs.push_back(make_input_tensor(info.inputs[i], i));
  }
  return inputs;
}

void print_model_info(const pcie::ModelInfo& info) {
  std::cout << "model metadata\n";
  std::cout << "  has_preprocess=" << (info.has_preprocess ? "true" : "false")
            << " has_boxdecode=" << (info.has_boxdecode ? "true" : "false") << "\n";
  std::cout << "  inputs (" << info.inputs.size() << ")\n";
  for (std::size_t i = 0; i < info.inputs.size(); ++i) {
    const auto& input = info.inputs[i];
    std::cout << "    [" << i << "] name=" << (input.name.empty() ? "<unnamed>" : input.name)
              << " dtype=" << (input.dtype.empty() ? "<unknown>" : input.dtype)
              << " shape=" << shape_string(input.shape) << " size_bytes=" << input.size_bytes
              << "\n";
  }
  std::cout << "  expected outputs (" << info.outputs.size() << ")\n";
  for (std::size_t i = 0; i < info.outputs.size(); ++i) {
    const auto& output = info.outputs[i];
    std::cout << "    [" << i << "] name=" << (output.name.empty() ? "<unnamed>" : output.name)
              << " dtype=" << (output.dtype.empty() ? "<unknown>" : output.dtype)
              << " shape=" << shape_string(output.shape) << " size_bytes=" << output.size_bytes
              << "\n";
  }
}

void print_inputs(const pcie::TensorList& inputs) {
  std::cout << "constructed inputs\n";
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto& input = inputs[i];
    std::cout << "  [" << i << "] name=" << input.route.name << " dtype=" << dtype_name(input.dtype)
              << " shape=" << shape_string(input.shape) << " size_bytes=" << input.size_bytes
              << "\n";
  }
}

void print_outputs(const pcie::TensorList& outputs) {
  std::cout << "received outputs (" << outputs.size() << ")\n";
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& output = outputs[i];
    std::cout << "  [" << i << "] name=" << output.route.name
              << " dtype=" << dtype_name(output.dtype) << " shape=" << shape_string(output.shape)
              << " size_bytes=" << output.size_bytes << " byte_offset=" << output.byte_offset
              << "\n";
  }
}

void validate_outputs(const pcie::TensorList& outputs,
                      const std::vector<pcie::TensorInfo>& expected) {
  if (outputs.size() != expected.size()) {
    throw std::runtime_error("output count mismatch: got " + std::to_string(outputs.size()) +
                             " expected " + std::to_string(expected.size()));
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (!expected[i].name.empty() && outputs[i].route.name != expected[i].name) {
      throw std::runtime_error("output name mismatch at index " + std::to_string(i) + ": got " +
                               outputs[i].route.name + " expected " + expected[i].name);
    }
    if (!expected[i].shape.empty() && outputs[i].shape != expected[i].shape) {
      throw std::runtime_error("output shape mismatch at index " + std::to_string(i));
    }
    if (expected[i].size_bytes != 0 && outputs[i].size_bytes != expected[i].size_bytes) {
      throw std::runtime_error("output size mismatch at index " + std::to_string(i));
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);

    pcie::ConnectionOptions conn;
    conn.card_host = args.card_host;
    conn.card_id = args.card_id;
    conn.user = args.user;
    conn.queue = args.queue;
    conn.max_inflight = args.max_inflight;
    conn.card_env = args.card_env;
    conn.card_gst_debug = args.card_gst_debug;
    conn.card_gst_debug_file = args.card_gst_debug_file;

    std::cout << "PCIe tensor run test\n";
    std::cout << "  model=" << args.model << "\n";
    std::cout << "  mode=tensor\n";
    std::cout << "  card_host="
              << (conn.card_host.empty() ? ("10.0." + std::to_string(conn.card_id) + ".2")
                                         : conn.card_host)
              << " card_id=" << conn.card_id << " user=" << conn.user << " queue=" << conn.queue
              << " max_inflight=" << conn.max_inflight << "\n";
    std::cout << "  burst_iterations=" << args.burst_iterations
              << " sync_iterations=" << args.sync_iterations
              << " async_iterations=" << args.iterations << "\n";
    if (!conn.card_env.empty()) {
      std::cout << "  card_env=" << conn.card_env << "\n";
    }
    if (!conn.card_gst_debug.empty()) {
      std::cout << "  card_gst_debug=" << conn.card_gst_debug << "\n";
      std::cout << "  card_gst_debug_file="
                << (conn.card_gst_debug_file.empty()
                        ? ("/var/log/sima-neat/pcie/q" + std::to_string(conn.queue) + ".gst.log")
                        : conn.card_gst_debug_file)
                << "\n";
    }

    pcie::Model model(args.model, {}, conn);
    pcie::test::SignalCloseGuard signal_guard(model);
    std::cout << "initial running=" << (model.running() ? "true" : "false") << "\n";

    std::cout << "loading metadata and starting card/host pipelines...\n";
    const auto started = std::chrono::steady_clock::now();
    model.build(args.readiness_timeout_ms);
    const pcie::ModelInfo info = model.info();
    const auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - started)
                             .count();
    std::cout << "build completed in " << init_ms << " ms\n";
    std::cout << "ready running=" << (model.running() ? "true" : "false") << "\n";
    print_model_info(info);

    pcie::TensorList inputs = make_inputs(info);
    print_inputs(inputs);

    if (args.burst_iterations > 0) {
      std::cout << "starting parallel burst phase: " << args.burst_iterations
                << " push/pull iteration(s)\n";
      const auto burst_started = std::chrono::steady_clock::now();
      auto producer = std::async(std::launch::async, [&] {
        for (int iteration = 1; iteration <= args.burst_iterations; ++iteration) {
          if (iteration == 1 || iteration % 10 == 0 || iteration == args.burst_iterations) {
            std::cout << "burst producer " << iteration << "/" << args.burst_iterations << "\n";
          }
          if (!model.push(inputs)) {
            throw std::runtime_error("burst push returned false at iteration " +
                                     std::to_string(iteration));
          }
        }
      });
      auto consumer = std::async(std::launch::async, [&] {
        for (int iteration = 1; iteration <= args.burst_iterations; ++iteration) {
          if (iteration == 1 || iteration % 10 == 0 || iteration == args.burst_iterations) {
            std::cout << "burst consumer " << iteration << "/" << args.burst_iterations
                      << " timeout_ms=" << args.pull_timeout_ms << "\n";
          }
          const auto result = model.pull(args.pull_timeout_ms);
          if (!result.has_value()) {
            throw std::runtime_error("burst pull timed out without a result at iteration " +
                                     std::to_string(iteration));
          }
          validate_outputs(*result, info.outputs);
          if (iteration == 1) {
            print_outputs(*result);
          }
        }
      });
      producer.get();
      consumer.get();
      const auto burst_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - burst_started)
                                .count();
      std::cout << "completed burst phase in " << burst_ms << " ms\n";
    }

    if (args.sync_iterations > 0) {
      const auto sync_started = std::chrono::steady_clock::now();
      for (int iteration = 1; iteration <= args.sync_iterations; ++iteration) {
        if (args.sync_iterations == 1 || iteration == 1 || iteration % 10 == 0 ||
            iteration == args.sync_iterations) {
          std::cout << "sync iteration " << iteration << "/" << args.sync_iterations
                    << ": push inputs...\n";
        }
        if (!model.push(inputs)) {
          throw std::runtime_error("sync push returned false at iteration " +
                                   std::to_string(iteration));
        }

        if (args.sync_iterations == 1 || iteration == 1 || iteration % 10 == 0 ||
            iteration == args.sync_iterations) {
          std::cout << "sync iteration " << iteration << "/" << args.sync_iterations
                    << ": pull outputs with timeout_ms=" << args.pull_timeout_ms << "...\n";
        }
        const auto result = model.pull(args.pull_timeout_ms);
        if (!result.has_value()) {
          throw std::runtime_error("sync pull timed out without a result at iteration " +
                                   std::to_string(iteration));
        }
        validate_outputs(*result, info.outputs);
        if (iteration == 1 || args.sync_iterations == 1) {
          print_outputs(*result);
        }
      }
      const auto sync_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - sync_started)
                               .count();
      std::cout << "completed " << args.sync_iterations << " sync push/pull iteration(s) in "
                << sync_ms << " ms\n";
    }

    if (args.iterations > 0) {
      std::cout << "starting async producer/consumer phase: " << args.iterations
                << " iteration(s)\n";
      const auto async_started = std::chrono::steady_clock::now();
      auto producer = std::async(std::launch::async, [&] {
        for (int iteration = 1; iteration <= args.iterations; ++iteration) {
          if (iteration == 1 || iteration % 100 == 0 || iteration == args.iterations) {
            std::cout << "async producer " << iteration << "/" << args.iterations << "\n";
          }
          if (!model.push(inputs)) {
            throw std::runtime_error("async push returned false at iteration " +
                                     std::to_string(iteration));
          }
        }
      });
      auto consumer = std::async(std::launch::async, [&] {
        for (int iteration = 1; iteration <= args.iterations; ++iteration) {
          if (iteration == 1 || iteration % 100 == 0 || iteration == args.iterations) {
            std::cout << "async consumer " << iteration << "/" << args.iterations << "\n";
          }
          const auto result = model.pull(args.pull_timeout_ms);
          if (!result.has_value()) {
            throw std::runtime_error("async pull timed out without a result at iteration " +
                                     std::to_string(iteration));
          }
          validate_outputs(*result, info.outputs);
        }
      });
      producer.get();
      consumer.get();
      const auto async_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - async_started)
                                .count();
      std::cout << "completed " << args.iterations << " async push/pull iteration(s) in "
                << async_ms << " ms\n";
    }

    std::cout << "stopping...\n";
    model.close();
    std::cout << "final running=" << (model.running() ? "true" : "false") << "\n";
    std::cout << "done\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}

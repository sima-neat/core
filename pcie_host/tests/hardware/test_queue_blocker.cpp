#include <simaai/neat/pcie/SimaPCIeHost.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace pcie = simaai::neat::pcie;

namespace {

std::atomic<bool> g_stop_requested{false};

void handle_signal(int) {
  g_stop_requested.store(true);
}

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
  int interval_ms = 1000;
  int iterations = 0;
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
               " [--user user] [--queue n] [--readiness-timeout-ms ms]"
               " [--pull-timeout-ms ms] [--interval-ms ms] [--iterations n]"
               " [--card-env 'NAME=VALUE ...']"
               " [--card-gst-debug spec] [--card-gst-debug-file path]\n";
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
    } else if (arg == "--readiness-timeout-ms") {
      args.readiness_timeout_ms = std::stoi(require_value(argc, argv, i, "--readiness-timeout-ms"));
    } else if (arg == "--pull-timeout-ms") {
      args.pull_timeout_ms = std::stoi(require_value(argc, argv, i, "--pull-timeout-ms"));
    } else if (arg == "--interval-ms") {
      args.interval_ms = std::stoi(require_value(argc, argv, i, "--interval-ms"));
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
  if (args.readiness_timeout_ms <= 0 || args.pull_timeout_ms <= 0 || args.interval_ms < 0 ||
      args.iterations < 0) {
    throw std::runtime_error("timeouts and iteration values must be non-negative, with positive timeouts");
  }
  return args;
}

std::string pipeline_state_name(const pcie::PipelineState state) {
  switch (state) {
  case pcie::PipelineState::Uninitialized:
    return "Uninitialized";
  case pcie::PipelineState::Starting:
    return "Starting";
  case pcie::PipelineState::Ready:
    return "Ready";
  case pcie::PipelineState::Failed:
    return "Failed";
  case pcie::PipelineState::Stopping:
    return "Stopping";
  case pcie::PipelineState::Exited:
    return "Exited";
  }
  return "Unknown";
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
                  const std::size_t tensor_index, const std::uint64_t iteration) {
  if (!bytes || bytes->empty()) {
    return;
  }
  if (dtype == pcie::TensorDType::Float32) {
    const std::size_t count = bytes->size() / sizeof(float);
    auto* data = reinterpret_cast<float*>(bytes->data());
    for (std::size_t i = 0; i < count; ++i) {
      data[i] = static_cast<float>((i + tensor_index + iteration) % 255U) / 255.0f;
    }
    return;
  }
  if (dtype == pcie::TensorDType::Float64) {
    const std::size_t count = bytes->size() / sizeof(double);
    auto* data = reinterpret_cast<double*>(bytes->data());
    for (std::size_t i = 0; i < count; ++i) {
      data[i] = static_cast<double>((i + tensor_index + iteration) % 255U) / 255.0;
    }
    return;
  }
  for (std::size_t i = 0; i < bytes->size(); ++i) {
    (*bytes)[i] = static_cast<std::uint8_t>((i + tensor_index * 17U + iteration) & 0xffU);
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
  fill_pattern(owner.get(), tensor.dtype, index, 0);

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

void refresh_inputs(pcie::TensorList* inputs, const std::uint64_t iteration) {
  if (!inputs) {
    return;
  }
  for (std::size_t i = 0; i < inputs->size(); ++i) {
    auto& tensor = (*inputs)[i];
    auto owner = std::static_pointer_cast<std::vector<std::uint8_t>>(tensor.owner);
    fill_pattern(owner.get(), tensor.dtype, i, iteration);
  }
}

void print_status(const char* label, const pcie::Status& status) {
  std::cout << label << ": state=" << pipeline_state_name(status.state)
            << " queue=" << status.queue;
  if (!status.message.empty()) {
    std::cout << " message=\"" << status.message << "\"";
  }
  if (!status.error_code.empty()) {
    std::cout << " error_code=\"" << status.error_code << "\"";
  }
  std::cout << "\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    const Args args = parse_args(argc, argv);

    pcie::ConnectionOptions conn;
    conn.card_host = args.card_host;
    conn.card_id = args.card_id;
    conn.user = args.user;
    conn.queue = args.queue;
    conn.card_env = args.card_env;
    conn.card_gst_debug = args.card_gst_debug;
    conn.card_gst_debug_file = args.card_gst_debug_file;

    std::cout << "PCIe queue blocker\n";
    std::cout << "  model=" << args.model << "\n";
    std::cout << "  card_host="
              << (conn.card_host.empty() ? ("10.0." + std::to_string(conn.card_id) + ".2")
                                         : conn.card_host)
              << " card_id=" << conn.card_id << " user=" << conn.user
              << " queue=" << conn.queue << "\n";
    std::cout << "  readiness_timeout_ms=" << args.readiness_timeout_ms
              << " pull_timeout_ms=" << args.pull_timeout_ms
              << " interval_ms=" << args.interval_ms
              << " iterations=" << (args.iterations == 0 ? std::string("forever")
                                                         : std::to_string(args.iterations))
              << "\n";

    pcie::SimaPCIeHost host(conn);
    print_status("initial status", host.status());

    const auto started = std::chrono::steady_clock::now();
    const pcie::ModelInfo info = host.init_pipeline(args.model, {}, args.readiness_timeout_ms);
    const auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - started)
                             .count();
    std::cout << "init_pipeline completed in " << init_ms << " ms\n";
    print_status("ready status", host.status());
    std::cout << "inputs=" << info.inputs.size() << " outputs=" << info.outputs.size() << "\n";
    for (std::size_t i = 0; i < info.inputs.size(); ++i) {
      std::cout << "  input[" << i << "] name="
                << (info.inputs[i].name.empty() ? "<unnamed>" : info.inputs[i].name)
                << " shape=" << shape_string(info.inputs[i].shape)
                << " size_bytes=" << info.inputs[i].size_bytes << "\n";
    }

    pcie::TensorList inputs = make_inputs(info);
    std::cout << "queue is now occupied; press Ctrl-C or send SIGTERM to stop\n";

    std::uint64_t iteration = 0;
    while (!g_stop_requested.load() &&
           (args.iterations == 0 || iteration < static_cast<std::uint64_t>(args.iterations))) {
      ++iteration;
      refresh_inputs(&inputs, iteration);

      std::cout << "iteration " << iteration << ": push\n";
      if (!host.push(inputs)) {
        throw std::runtime_error("push returned false");
      }

      std::cout << "iteration " << iteration << ": pull timeout_ms=" << args.pull_timeout_ms
                << "\n";
      const auto result = host.pull(args.pull_timeout_ms);
      if (result.has_value()) {
        std::cout << "iteration " << iteration << ": received outputs=" << result->size()
                  << "\n";
      } else {
        std::cout << "iteration " << iteration << ": pull timed out; keeping queue occupied\n";
      }

      if (args.interval_ms > 0 && !g_stop_requested.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(args.interval_ms));
      }
    }

    std::cout << "stopping blocker...\n";
    host.stop();
    print_status("final status", host.status());
    std::cout << "done\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}

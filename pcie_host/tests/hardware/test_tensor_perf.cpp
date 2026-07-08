#include <simaai/neat/pcie/Model.h>

#include "SignalCloseGuard.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace pcie = simaai::neat::pcie;

namespace {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

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

enum class Mode {
  Latency,
  Throughput,
};

struct Args {
  std::string model = env_or_default("SIMAPCIE_YOLOV8_MODEL", DEFAULT_MODEL_PATH);
  std::string card_host = env_or_default("SIMAPCIE_CARD_HOST", "");
  std::string user = env_or_default("SIMAPCIE_USER", "sima");
  int card_id = env_int_or_default("SIMAPCIE_CARD_ID", 0);
  int queue = env_int_or_default("SIMAPCIE_QUEUE", 0);
  int max_inflight = env_int_or_default("SIMAPCIE_MAX_INFLIGHT", 10);
  int readiness_timeout_ms = env_int_or_default("SIMAPCIE_READINESS_TIMEOUT_MS", 180000);
  int pull_timeout_ms = env_int_or_default("SIMAPCIE_PULL_TIMEOUT_MS", 30000);
  int warmup = env_int_or_default("SIMAPCIE_PERF_WARMUP", 50);
  int iterations = env_int_or_default("SIMAPCIE_PERF_ITERATIONS", 1000);
  Mode mode = Mode::Latency;
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
            << " --mode latency|throughput [--model model.tar.gz] [--card-host host]"
               " [--card-id n] [--user user] [--queue n] [--max-inflight n]"
               " [--readiness-timeout-ms ms] [--pull-timeout-ms ms]"
               " [--warmup n] [--iterations n]"
               " [--card-env 'NAME=VALUE ...'] [--card-gst-debug spec]"
               " [--card-gst-debug-file path]\n";
}

Mode parse_mode(const std::string& mode) {
  if (mode == "latency") {
    return Mode::Latency;
  }
  if (mode == "throughput") {
    return Mode::Throughput;
  }
  throw std::runtime_error("unknown mode: " + mode);
}

std::string mode_name(const Mode mode) {
  return mode == Mode::Latency ? "latency" : "throughput";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i] ? argv[i] : "";
    if (arg == "--mode") {
      args.mode = parse_mode(require_value(argc, argv, i, "--mode"));
    } else if (arg == "--model") {
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
    } else if (arg == "--warmup") {
      args.warmup = std::stoi(require_value(argc, argv, i, "--warmup"));
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
  if (args.max_inflight < 0 || args.max_inflight > 256 || args.readiness_timeout_ms <= 0 ||
      args.pull_timeout_ms <= 0 || args.warmup < 0 || args.iterations <= 0) {
    throw std::runtime_error(
        "max-inflight must be in range 0..256, timeouts must be positive, warmup must be "
        "non-negative, and iterations must be positive");
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
    if (u_dim != 0 && bytes > std::numeric_limits<std::size_t>::max() / u_dim) {
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
  tensor.read_only = true;
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

std::size_t tensor_payload_bytes(const pcie::Tensor& tensor) {
  if (tensor.size_bytes != 0) {
    return tensor.size_bytes;
  }
  return dense_size_bytes(tensor.shape, tensor.dtype);
}

std::size_t tensor_list_payload_bytes(const pcie::TensorList& tensors) {
  std::size_t total = 0;
  for (const auto& tensor : tensors) {
    total += tensor_payload_bytes(tensor);
  }
  return total;
}

std::size_t tensor_info_payload_bytes(const std::vector<pcie::TensorInfo>& tensors) {
  std::size_t total = 0;
  for (const auto& tensor : tensors) {
    total += tensor.size_bytes;
  }
  return total;
}

void print_model_summary(const pcie::ModelInfo& info) {
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
  std::cout << "  outputs (" << info.outputs.size() << ")\n";
  for (std::size_t i = 0; i < info.outputs.size(); ++i) {
    const auto& output = info.outputs[i];
    std::cout << "    [" << i << "] name=" << (output.name.empty() ? "<unnamed>" : output.name)
              << " dtype=" << (output.dtype.empty() ? "<unknown>" : output.dtype)
              << " shape=" << shape_string(output.shape) << " size_bytes=" << output.size_bytes
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

double milliseconds_between(const TimePoint start, const TimePoint end) {
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) /
         1.0e6;
}

struct LatencyStats {
  double min_ms = 0.0;
  double p50_ms = 0.0;
  double p90_ms = 0.0;
  double p95_ms = 0.0;
  double p99_ms = 0.0;
  double max_ms = 0.0;
  double mean_ms = 0.0;
  double stddev_ms = 0.0;
};

double percentile(const std::vector<double>& sorted, const double p) {
  if (sorted.empty()) {
    return 0.0;
  }
  const double rank = (static_cast<double>(sorted.size() - 1U) * p);
  const auto lower = static_cast<std::size_t>(std::floor(rank));
  const auto upper = static_cast<std::size_t>(std::ceil(rank));
  if (lower == upper) {
    return sorted[lower];
  }
  const double weight = rank - static_cast<double>(lower);
  return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

LatencyStats summarize_latencies(std::vector<double> values) {
  LatencyStats stats;
  if (values.empty()) {
    return stats;
  }
  std::sort(values.begin(), values.end());
  stats.min_ms = values.front();
  stats.p50_ms = percentile(values, 0.50);
  stats.p90_ms = percentile(values, 0.90);
  stats.p95_ms = percentile(values, 0.95);
  stats.p99_ms = percentile(values, 0.99);
  stats.max_ms = values.back();
  stats.mean_ms =
      std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
  double sq_sum = 0.0;
  for (const auto value : values) {
    const double delta = value - stats.mean_ms;
    sq_sum += delta * delta;
  }
  stats.stddev_ms = std::sqrt(sq_sum / static_cast<double>(values.size()));
  return stats;
}

void print_latency_stats(const std::string& label, const LatencyStats& stats,
                         const std::size_t count) {
  std::cout << label << "\n";
  std::cout << "  count=" << count << "\n";
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "  min_ms=" << stats.min_ms << " p50_ms=" << stats.p50_ms
            << " p90_ms=" << stats.p90_ms << " p95_ms=" << stats.p95_ms
            << " p99_ms=" << stats.p99_ms << " max_ms=" << stats.max_ms << "\n";
  std::cout << "  mean_ms=" << stats.mean_ms << " stddev_ms=" << stats.stddev_ms << "\n";
}

void run_latency(pcie::Model& model, const pcie::TensorList& inputs,
                 const std::vector<pcie::TensorInfo>& expected_outputs, const Args& args) {
  std::cout << "latency warmup: " << args.warmup << " iteration(s)\n";
  for (int i = 0; i < args.warmup; ++i) {
    if (!model.push(inputs)) {
      throw std::runtime_error("latency warmup push returned false at iteration " +
                               std::to_string(i + 1));
    }
    const auto result = model.pull(args.pull_timeout_ms);
    if (!result.has_value()) {
      throw std::runtime_error("latency warmup pull timed out at iteration " +
                               std::to_string(i + 1));
    }
    validate_outputs(*result, expected_outputs);
  }

  std::cout << "latency measure: " << args.iterations << " iteration(s)\n";
  std::vector<double> latencies_ms;
  latencies_ms.reserve(static_cast<std::size_t>(args.iterations));
  const auto started = Clock::now();
  for (int i = 0; i < args.iterations; ++i) {
    const auto t0 = Clock::now();
    if (!model.push(inputs)) {
      throw std::runtime_error("latency push returned false at iteration " + std::to_string(i + 1));
    }
    const auto result = model.pull(args.pull_timeout_ms);
    const auto t1 = Clock::now();
    if (!result.has_value()) {
      throw std::runtime_error("latency pull timed out at iteration " + std::to_string(i + 1));
    }
    validate_outputs(*result, expected_outputs);
    latencies_ms.push_back(milliseconds_between(t0, t1));
    if ((i + 1) == 1 || (i + 1) % 100 == 0 || (i + 1) == args.iterations) {
      std::cout << "latency iteration " << (i + 1) << "/" << args.iterations << "\n";
    }
  }
  const auto elapsed_ms = milliseconds_between(started, Clock::now());
  print_latency_stats("latency results", summarize_latencies(latencies_ms), latencies_ms.size());
  std::cout << std::fixed << std::setprecision(3) << "  total_ms=" << elapsed_ms
            << " effective_fps=" << (static_cast<double>(args.iterations) * 1000.0 / elapsed_ms)
            << "\n";
}

class TimestampQueue {
public:
  void push(TimePoint value) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      values_.push(value);
    }
    cond_.notify_one();
  }

  TimePoint pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [&] { return !values_.empty(); });
    const TimePoint value = values_.front();
    values_.pop();
    return value;
  }

private:
  std::mutex mutex_;
  std::condition_variable cond_;
  std::queue<TimePoint> values_;
};

void run_throughput(pcie::Model& model, const pcie::TensorList& inputs,
                    const std::vector<pcie::TensorInfo>& expected_outputs, const Args& args,
                    const std::size_t input_bytes, const std::size_t output_bytes) {
  std::cout << "throughput warmup: " << args.warmup << " sync iteration(s)\n";
  for (int i = 0; i < args.warmup; ++i) {
    if (!model.push(inputs)) {
      throw std::runtime_error("throughput warmup push returned false at iteration " +
                               std::to_string(i + 1));
    }
    const auto result = model.pull(args.pull_timeout_ms);
    if (!result.has_value()) {
      throw std::runtime_error("throughput warmup pull timed out at iteration " +
                               std::to_string(i + 1));
    }
    validate_outputs(*result, expected_outputs);
  }

  std::cout << "throughput measure: " << args.iterations << " async iteration(s)\n";
  TimestampQueue timestamps;
  std::vector<double> latencies_ms;
  latencies_ms.reserve(static_cast<std::size_t>(args.iterations));
  std::atomic_bool failed = false;
  std::mutex error_mutex;
  std::exception_ptr error;
  const auto set_error = [&](std::exception_ptr ptr) {
    {
      std::lock_guard<std::mutex> lock(error_mutex);
      if (!error) {
        error = std::move(ptr);
      }
    }
    failed.store(true);
  };

  std::atomic<std::int64_t> producer_push_ns{0};
  std::atomic<std::int64_t> consumer_pull_ns{0};
  const auto started = Clock::now();

  auto producer = std::async(std::launch::async, [&] {
    try {
      for (int i = 0; i < args.iterations; ++i) {
        if (failed.load()) {
          return;
        }
        const auto t0 = Clock::now();
        timestamps.push(t0);
        if (!model.push(inputs)) {
          throw std::runtime_error("throughput push returned false at iteration " +
                                   std::to_string(i + 1));
        }
        producer_push_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
        if ((i + 1) == 1 || (i + 1) % 100 == 0 || (i + 1) == args.iterations) {
          std::cout << "throughput producer " << (i + 1) << "/" << args.iterations << "\n";
        }
      }
    } catch (...) {
      set_error(std::current_exception());
    }
  });

  auto consumer = std::async(std::launch::async, [&] {
    try {
      for (int i = 0; i < args.iterations; ++i) {
        if (failed.load()) {
          return;
        }
        const auto pull_start = Clock::now();
        const auto result = model.pull(args.pull_timeout_ms);
        const auto pull_end = Clock::now();
        consumer_pull_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(pull_end - pull_start).count());
        if (!result.has_value()) {
          throw std::runtime_error("throughput pull timed out at iteration " +
                                   std::to_string(i + 1));
        }
        validate_outputs(*result, expected_outputs);
        const TimePoint push_start = timestamps.pop();
        latencies_ms.push_back(milliseconds_between(push_start, pull_end));
        if ((i + 1) == 1 || (i + 1) % 100 == 0 || (i + 1) == args.iterations) {
          std::cout << "throughput consumer " << (i + 1) << "/" << args.iterations << "\n";
        }
      }
    } catch (...) {
      set_error(std::current_exception());
    }
  });

  producer.get();
  consumer.get();
  if (error) {
    std::rethrow_exception(error);
  }

  const auto elapsed_ms = milliseconds_between(started, Clock::now());
  const double elapsed_s = elapsed_ms / 1000.0;
  const double frames = static_cast<double>(args.iterations);
  const double input_mib = (frames * static_cast<double>(input_bytes)) / (1024.0 * 1024.0);
  const double output_mib = (frames * static_cast<double>(output_bytes)) / (1024.0 * 1024.0);
  const double push_ms = static_cast<double>(producer_push_ns.load()) / 1.0e6;
  const double pull_ms = static_cast<double>(consumer_pull_ns.load()) / 1.0e6;

  std::cout << "throughput results\n";
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "  frames=" << args.iterations << " total_ms=" << elapsed_ms
            << " fps=" << (frames / elapsed_s) << "\n";
  std::cout << "  input_mib=" << input_mib << " input_mib_per_s=" << (input_mib / elapsed_s)
            << "\n";
  std::cout << "  output_mib=" << output_mib << " output_mib_per_s=" << (output_mib / elapsed_s)
            << "\n";
  std::cout << "  producer_push_total_ms=" << push_ms
            << " producer_push_mean_ms=" << (push_ms / frames) << "\n";
  std::cout << "  consumer_pull_total_ms=" << pull_ms
            << " consumer_pull_mean_ms=" << (pull_ms / frames) << "\n";
  print_latency_stats("throughput fifo latency estimate", summarize_latencies(latencies_ms),
                      latencies_ms.size());
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

    std::cout << "PCIe tensor perf test\n";
    std::cout << "  model=" << args.model << "\n";
    std::cout << "  mode=" << mode_name(args.mode) << "\n";
    std::cout << "  card_host="
              << (conn.card_host.empty() ? ("10.0." + std::to_string(conn.card_id) + ".2")
                                         : conn.card_host)
              << " card_id=" << conn.card_id << " user=" << conn.user << " queue=" << conn.queue
              << " max_inflight=" << conn.max_inflight << "\n";
    std::cout << "  warmup=" << args.warmup << " iterations=" << args.iterations
              << " pull_timeout_ms=" << args.pull_timeout_ms << "\n";
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

    const auto build_started = Clock::now();
    model.build(args.readiness_timeout_ms);
    const auto build_ms = milliseconds_between(build_started, Clock::now());
    const pcie::ModelInfo info = model.info();
    std::cout << std::fixed << std::setprecision(3) << "build completed in " << build_ms << " ms\n";
    print_model_summary(info);

    const pcie::TensorList inputs = make_inputs(info);
    const std::size_t input_bytes = tensor_list_payload_bytes(inputs);
    const std::size_t output_bytes = tensor_info_payload_bytes(info.outputs);
    std::cout << "payload sizes\n";
    std::cout << "  input_bytes_per_frame=" << input_bytes << "\n";
    std::cout << "  output_bytes_per_frame=" << output_bytes << "\n";

    if (args.mode == Mode::Latency) {
      run_latency(model, inputs, info.outputs, args);
    } else {
      run_throughput(model, inputs, info.outputs, args, input_bytes, output_bytes);
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

#include <simaai/neat/pcie/Model.h>

#include "AsyncTestRunner.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace pcie = simaai::neat::pcie;

namespace {

using namespace std::chrono_literals;

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
  int lifecycle_cycles = env_int_or_default("SIMAPCIE_LIFECYCLE_CYCLES", 3);
  int lifetime_iterations = env_int_or_default("SIMAPCIE_LIFETIME_ITERATIONS", 20);
  std::string card_gst_debug = env_or_default("SIMAPCIE_CARD_GST_DEBUG", "");
  std::string card_gst_debug_file = env_or_default("SIMAPCIE_CARD_GST_DEBUG_FILE", "");
};

std::string require_value(int argc, char** argv, int& index, const char* name) {
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + name);
  }
  return argv[++index];
}

void usage(const char* executable) {
  std::cerr << "usage: " << executable
            << " [--model model.tar.gz] [--card-host host] [--card-id n] [--user user]"
               " [--queue n] [--readiness-timeout-ms ms] [--pull-timeout-ms ms]"
               " [--lifecycle-cycles n] [--lifetime-iterations n]\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index] ? argv[index] : "";
    if (arg == "--model") {
      args.model = require_value(argc, argv, index, "--model");
    } else if (arg == "--card-host") {
      args.card_host = require_value(argc, argv, index, "--card-host");
    } else if (arg == "--card-id") {
      args.card_id = std::stoi(require_value(argc, argv, index, "--card-id"));
    } else if (arg == "--user") {
      args.user = require_value(argc, argv, index, "--user");
    } else if (arg == "--queue") {
      args.queue = std::stoi(require_value(argc, argv, index, "--queue"));
    } else if (arg == "--readiness-timeout-ms") {
      args.readiness_timeout_ms =
          std::stoi(require_value(argc, argv, index, "--readiness-timeout-ms"));
    } else if (arg == "--pull-timeout-ms") {
      args.pull_timeout_ms = std::stoi(require_value(argc, argv, index, "--pull-timeout-ms"));
    } else if (arg == "--lifecycle-cycles") {
      args.lifecycle_cycles = std::stoi(require_value(argc, argv, index, "--lifecycle-cycles"));
    } else if (arg == "--lifetime-iterations") {
      args.lifetime_iterations =
          std::stoi(require_value(argc, argv, index, "--lifetime-iterations"));
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
  if (args.queue < 0 || args.queue > 5 || args.readiness_timeout_ms <= 0 ||
      args.pull_timeout_ms <= 0 || args.lifecycle_cycles < 3 || args.lifetime_iterations <= 0) {
    throw std::runtime_error(
        "queue must be in range 0..5, timeouts must be positive, lifecycle cycles must be at "
        "least 3, and lifetime iterations must be positive");
  }
  return args;
}

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
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
  throw std::runtime_error("unsupported input dtype");
}

std::vector<std::int64_t> contiguous_strides(const std::vector<std::int64_t>& shape,
                                             const std::size_t element_bytes) {
  std::vector<std::int64_t> strides(shape.size(), 0);
  std::int64_t stride = static_cast<std::int64_t>(element_bytes);
  for (std::size_t index = shape.size(); index > 0; --index) {
    const std::size_t dim = index - 1;
    strides[dim] = stride;
    stride *= shape[dim];
  }
  return strides;
}

std::size_t dense_size(const std::vector<std::int64_t>& shape, const pcie::TensorDType dtype) {
  std::size_t size = dtype_bytes(dtype);
  for (const std::int64_t dim : shape) {
    if (dim <= 0) {
      throw std::runtime_error("input shape contains a non-positive dimension");
    }
    size *= static_cast<std::size_t>(dim);
  }
  return size;
}

struct TrackedBuffer {
  TrackedBuffer(const std::size_t size, std::shared_ptr<std::atomic_int> destroyed)
      : bytes(size, 0), destroyed(std::move(destroyed)) {}

  ~TrackedBuffer() {
    if (destroyed) {
      destroyed->fetch_add(1);
    }
  }

  std::vector<std::uint8_t> bytes;
  std::shared_ptr<std::atomic_int> destroyed;
};

pcie::TensorList make_inputs(const pcie::ModelInfo& info,
                             const std::shared_ptr<std::atomic_int>& destroyed = {}) {
  pcie::TensorList inputs;
  inputs.reserve(info.inputs.size());
  for (std::size_t index = 0; index < info.inputs.size(); ++index) {
    const pcie::TensorInfo& spec = info.inputs[index];
    pcie::Tensor tensor;
    tensor.dtype = dtype_from_fact(spec.dtype);
    tensor.shape = spec.shape.empty()
                       ? std::vector<std::int64_t>{static_cast<std::int64_t>(spec.size_bytes)}
                       : spec.shape;
    tensor.strides_bytes = contiguous_strides(tensor.shape, dtype_bytes(tensor.dtype));
    const std::size_t size =
        spec.size_bytes != 0 ? spec.size_bytes : dense_size(tensor.shape, tensor.dtype);
    require(size > 0, "input tensor has an empty payload");

    auto owner = std::make_shared<TrackedBuffer>(size, destroyed);
    tensor.owner = owner;
    tensor.data = owner->bytes.data();
    tensor.size_bytes = owner->bytes.size();
    tensor.route.name = spec.name.empty() ? "input_" + std::to_string(index) : spec.name;
    tensor.route.logical_index = static_cast<int>(index);
    tensor.route.physical_index = static_cast<int>(index);
    tensor.route.route_slot = static_cast<int>(index);
    inputs.push_back(std::move(tensor));
  }
  return inputs;
}

void validate_outputs(const pcie::TensorList& outputs,
                      const std::vector<pcie::TensorInfo>& expected) {
  require(outputs.size() == expected.size(), "output count mismatch: got " +
                                                 std::to_string(outputs.size()) + " expected " +
                                                 std::to_string(expected.size()));
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (!expected[index].name.empty()) {
      require(outputs[index].route.name == expected[index].name,
              "output name mismatch at index " + std::to_string(index));
    }
    if (!expected[index].shape.empty()) {
      require(outputs[index].shape == expected[index].shape,
              "output shape mismatch at index " + std::to_string(index));
    }
    if (expected[index].size_bytes != 0) {
      require(outputs[index].size_bytes == expected[index].size_bytes,
              "output size mismatch at index " + std::to_string(index));
    }
  }
}

void infer_once(pcie::Model& model, const pcie::ModelInfo& info, const int timeout_ms) {
  pcie::TensorList inputs = make_inputs(info);
  require(model.push(inputs), "push returned false");
  const auto outputs = model.pull(timeout_ms);
  require(outputs.has_value(), "pull timed out");
  validate_outputs(*outputs, info.outputs);
}

pcie::ConnectionOptions connection(const Args& args) {
  pcie::ConnectionOptions value;
  value.card_host = args.card_host;
  value.card_id = args.card_id;
  value.user = args.user;
  value.queue = args.queue;
  value.max_inflight = 1;
  value.card_gst_debug = args.card_gst_debug;
  value.card_gst_debug_file = args.card_gst_debug_file;
  return value;
}

void test_failed_build_cleanup(const Args& args) {
  std::cout << "[scenario] invalid model archive cleanup\n";
  const std::filesystem::path invalid_model =
      std::filesystem::temp_directory_path() /
      ("sima-pcie-invalid-model-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".tar.gz");
  {
    std::ofstream output(invalid_model, std::ios::binary | std::ios::trunc);
    output << "not a model archive";
  }
  bool invalid_model_failed = false;
  try {
    pcie::Model invalid(invalid_model.string());
  } catch (const std::exception&) {
    invalid_model_failed = true;
  }
  std::error_code remove_error;
  std::filesystem::remove(invalid_model, remove_error);
  require(invalid_model_failed, "invalid model archive was accepted");

  std::cout << "[scenario] unreachable host cleanup\n";
  pcie::ConnectionOptions unreachable = connection(args);
  unreachable.card_host = "no-such-pcie-card.invalid";
  pcie::Model unreachable_model(args.model, {}, unreachable);
  bool unreachable_failed = false;
  try {
    unreachable_model.build(args.readiness_timeout_ms);
  } catch (const std::exception&) {
    unreachable_failed = true;
  }
  require(unreachable_failed, "unreachable card host did not fail build");
  require(!unreachable_model.running(), "unreachable-host model reports running");
  unreachable_model.close();
  unreachable_model.close();

  std::cout << "[scenario] invalid startup configuration cleanup\n";
  pcie::ConnectionOptions invalid_env = connection(args);
  invalid_env.card_env = "INVALID_ENV_ENTRY";
  pcie::Model invalid_env_model(args.model, {}, invalid_env);
  bool invalid_env_failed = false;
  try {
    invalid_env_model.build(args.readiness_timeout_ms);
  } catch (const std::exception& error) {
    invalid_env_failed =
        std::string(error.what()).find("card_env entries must be NAME=VALUE") != std::string::npos;
  }
  require(invalid_env_failed, "invalid card environment did not fail build as expected");
  require(!invalid_env_model.running(), "invalid-environment model reports running");
  invalid_env_model.close();
  invalid_env_model.close();

  std::cout << "[scenario] readiness-timeout recovery on the same model\n";
  pcie::Model readiness_model(args.model, {}, connection(args));
  bool readiness_failed = false;
  try {
    readiness_model.build(1);
  } catch (const std::exception&) {
    readiness_failed = true;
  }
  require(readiness_model.running() != readiness_failed,
          "short readiness attempt left the model in an inconsistent state");
  readiness_model.close();

  readiness_model.build(args.readiness_timeout_ms);
  const pcie::ModelInfo info = readiness_model.info();
  infer_once(readiness_model, info, args.pull_timeout_ms);
  readiness_model.close();
}

void test_timeout_and_backpressure(pcie::Model& model, const pcie::ModelInfo& info,
                                   const Args& args) {
  std::cout << "[scenario] pull timeout recovery\n";
  require(!model.pull(1).has_value(), "pull before push unexpectedly returned a result");
  infer_once(model, info, args.pull_timeout_ms);

  std::cout << "[scenario] synchronous run timeout requires draining\n";
  bool run_timed_out = false;
  try {
    (void)model.run(make_inputs(info), 1);
  } catch (const std::exception& error) {
    run_timed_out = std::string(error.what()).find("timed out waiting") != std::string::npos;
  }
  require(run_timed_out, "one-millisecond synchronous run unexpectedly completed");

  bool next_run_rejected = false;
  try {
    (void)model.run(make_inputs(info), args.pull_timeout_ms);
  } catch (const std::exception& error) {
    next_run_rejected = std::string(error.what()).find("call pull() to drain") != std::string::npos;
  }
  require(next_run_rejected,
          "synchronous run was accepted before the timed-out result was drained");

  bool next_push_rejected = false;
  try {
    (void)model.push(make_inputs(info));
  } catch (const std::exception& error) {
    next_push_rejected =
        std::string(error.what()).find("call pull() to drain") != std::string::npos;
  }
  require(next_push_rejected, "push was accepted before the timed-out result was drained");

  const auto late_outputs = model.pull(args.pull_timeout_ms);
  require(late_outputs.has_value(), "timed-out synchronous result could not be drained");
  validate_outputs(*late_outputs, info.outputs);
  infer_once(model, info, args.pull_timeout_ms);

  std::cout << "[scenario] max_inflight=1 backpressure and recovery\n";
  pcie::TensorList first_inputs = make_inputs(info);
  require(model.push(first_inputs), "first backpressure push returned false");

  std::promise<void> second_started;
  auto second_push = std::async(std::launch::async, [&] {
    pcie::TensorList second_inputs = make_inputs(info);
    second_started.set_value();
    return model.push(second_inputs);
  });
  second_started.get_future().wait();
  require(second_push.wait_for(2ms) == std::future_status::timeout,
          "second push did not block at max_inflight=1");
  require(second_push.wait_for(std::chrono::milliseconds(args.pull_timeout_ms)) ==
              std::future_status::ready,
          "second push did not resume when transport capacity became available");
  require(second_push.get(), "second backpressure push returned false");

  for (int result = 0; result < 2; ++result) {
    const auto outputs = model.pull(args.pull_timeout_ms);
    require(outputs.has_value(), "backpressure result pull timed out");
    validate_outputs(*outputs, info.outputs);
  }
}

void test_concurrent_queue_ownership(pcie::Model& first, pcie::Model& second,
                                     const pcie::ModelInfo& info, const Args& args) {
  std::cout << "[scenario] concurrent queue ownership\n";
  struct BuildResult {
    bool succeeded = false;
    std::string error;
  };

  std::promise<void> start;
  const std::shared_future<void> start_signal = start.get_future().share();
  auto launch = [&](pcie::Model& model) {
    start_signal.wait();
    BuildResult result;
    try {
      model.build(args.readiness_timeout_ms);
      result.succeeded = true;
    } catch (const std::exception& error) {
      result.error = error.what();
    }
    return result;
  };

  auto first_build = std::async(std::launch::async, [&] { return launch(first); });
  auto second_build = std::async(std::launch::async, [&] { return launch(second); });
  start.set_value();

  const BuildResult first_result = first_build.get();
  const BuildResult second_result = second_build.get();
  require(first_result.succeeded != second_result.succeeded,
          "concurrent queue claim did not produce exactly one owner");

  pcie::Model& winner = first_result.succeeded ? first : second;
  pcie::Model& loser = first_result.succeeded ? second : first;
  const std::string& loser_error =
      first_result.succeeded ? second_result.error : first_result.error;
  require(loser_error.find("queue_busy") != std::string::npos,
          "concurrent queue loser did not report queue_busy: " + loser_error);

  loser.close();
  require(winner.running(), "closing the concurrent queue loser stopped the winner");
  infer_once(winner, info, args.pull_timeout_ms);
  winner.close();
}

void test_queue_ownership(pcie::Model& owner, pcie::Model& contender, const pcie::ModelInfo& info,
                          const Args& args) {
  std::cout << "[scenario] queue ownership and failed contender cleanup\n";
  bool queue_busy = false;
  try {
    contender.build(args.readiness_timeout_ms);
  } catch (const std::exception& error) {
    queue_busy = std::string(error.what()).find("queue_busy") != std::string::npos;
  }
  require(queue_busy, "second model did not fail with queue_busy");
  contender.close();
  contender.close();

  require(owner.running(), "failed contender stopped the queue owner");
  std::cout << "  verifying original queue owner remains usable" << std::endl;
  infer_once(owner, info, args.pull_timeout_ms);
  std::cout << "  closing original queue owner" << std::endl;
  owner.close();

  std::cout << "  rebuilding contender after queue release" << std::endl;
  contender.build(args.readiness_timeout_ms);
  require(contender.running(), "contender could not acquire the released queue");
  std::cout << "  verifying new queue owner is usable" << std::endl;
  infer_once(contender, info, args.pull_timeout_ms);
}

void test_unique_buffer_lifetime(pcie::Model& model, const pcie::ModelInfo& info,
                                 const Args& args) {
  std::cout << "[scenario] unique input ownership for " << args.lifetime_iterations
            << " iteration(s)\n";
  auto destroyed = std::make_shared<std::atomic_int>(0);
  const int expected_destroyed = args.lifetime_iterations * static_cast<int>(info.inputs.size());

  pcie::test::run_async_workers(
      [&] { model.close(); },
      [&](const std::atomic_bool& cancelled) {
        for (int iteration = 0; iteration < args.lifetime_iterations; ++iteration) {
          if (cancelled.load()) {
            return;
          }
          pcie::TensorList inputs = make_inputs(info, destroyed);
          require(model.push(inputs), "unique-buffer push returned false");
          inputs.clear();
        }
      },
      [&](const std::atomic_bool& cancelled) {
        for (int iteration = 0; iteration < args.lifetime_iterations; ++iteration) {
          if (cancelled.load()) {
            return;
          }
          const auto outputs = model.pull(args.pull_timeout_ms);
          require(outputs.has_value(), "unique-buffer pull timed out");
          validate_outputs(*outputs, info.outputs);
        }
      });

  for (int retry = 0; retry < 100 && destroyed->load() != expected_destroyed; ++retry) {
    std::this_thread::sleep_for(10ms);
  }
  require(destroyed->load() == expected_destroyed,
          "input owners were not released after inference: got " +
              std::to_string(destroyed->load()) + " expected " +
              std::to_string(expected_destroyed));
}

void test_close_blocked_pull(pcie::Model& model) {
  std::cout << "[scenario] close while pull is blocked\n";
  std::promise<void> pull_started;
  auto blocked_pull = std::async(std::launch::async, [&] {
    pull_started.set_value();
    return model.pull(-1);
  });
  pull_started.get_future().wait();
  require(blocked_pull.wait_for(50ms) == std::future_status::timeout,
          "pull did not block before close");

  model.close();
  require(blocked_pull.wait_for(5s) == std::future_status::ready,
          "blocked pull did not wake after close");
  require(!blocked_pull.get().has_value(), "blocked pull returned a result after close");
}

void test_close_blocked_push(pcie::Model& model, const pcie::ModelInfo& info) {
  std::cout << "[scenario] close while push is blocked\n";
  pcie::TensorList first_inputs = make_inputs(info);
  require(model.push(first_inputs), "first teardown push returned false");

  std::promise<void> push_started;
  auto blocked_push = std::async(std::launch::async, [&] {
    pcie::TensorList second_inputs = make_inputs(info);
    push_started.set_value();
    return model.push(second_inputs);
  });
  push_started.get_future().wait();
  require(blocked_push.wait_for(2ms) == std::future_status::timeout,
          "push did not block at max_inflight=1 before close");

  model.close();
  require(blocked_push.wait_for(5s) == std::future_status::ready,
          "blocked push did not wake after close");
  bool stopped = false;
  try {
    (void)blocked_push.get();
  } catch (const std::exception&) {
    stopped = true;
  }
  require(stopped, "blocked push succeeded after close");
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    std::cout << "PCIe Model resilience test\n";
    std::cout << "  model=" << args.model << "\n";
    std::cout << "  card_host=" << (args.card_host.empty() ? "<derived>" : args.card_host)
              << " card_id=" << args.card_id << " user=" << args.user << " queue=" << args.queue
              << " max_inflight=1\n";
    std::cout << "  lifecycle_cycles=" << args.lifecycle_cycles
              << " lifetime_iterations=" << args.lifetime_iterations << "\n";

    test_failed_build_cleanup(args);

    pcie::ConnectionOptions conn = connection(args);
    pcie::Model owner(args.model, {}, conn);
    pcie::Model contender(args.model, {}, conn);
    const pcie::ModelInfo info = owner.info();

    test_concurrent_queue_ownership(owner, contender, info, args);

    std::cout << "[lifecycle 1/" << args.lifecycle_cycles << "] build owner\n";
    owner.build(args.readiness_timeout_ms);
    test_timeout_and_backpressure(owner, info, args);
    test_queue_ownership(owner, contender, info, args);
    test_unique_buffer_lifetime(contender, info, args);
    test_close_blocked_pull(contender);

    for (int cycle = 2; cycle <= args.lifecycle_cycles; ++cycle) {
      std::cout << "[lifecycle " << cycle << "/" << args.lifecycle_cycles
                << "] rebuild, infer, close\n";
      contender.build(args.readiness_timeout_ms);
      infer_once(contender, info, args.pull_timeout_ms);
      contender.close();
      require(!contender.running(), "model still reports running after close");
    }

    std::cout << "[teardown] rebuild for blocked-push cancellation\n";
    contender.build(args.readiness_timeout_ms);
    test_close_blocked_push(contender, info);

    std::cout << "test_model_resilience: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_model_resilience: FAIL: " << error.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}

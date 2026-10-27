#pragma once

// Public-API fixtures for the SimaVoxelize, SimaPillarScatter, SimaBEVPoolPlan and
// SimaBEVPool hardware suites. Every graph is built and run through the public
// Graph/Node/Run API; numerical truth comes from the independent oracles in
// bevfusion_kernel_reference.h, never from a provider call.
#include <neat.h>

#include "bevfusion_kernel_reference.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bevfusion_test {
namespace neat = simaai::neat;
namespace ref = ::bevfusion_test::reference;
constexpr int kTimeoutMs = 30000;

class Failure final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};
inline void require(bool condition, const std::string& message) {
  if (!condition)
    throw Failure(message);
}

// The hardware a suite executable targets, fixed at build time: CTest runs each
// executable without arguments, and every scenario is required.
inline neat::SimaKernelTarget suite_target() {
#if defined(SIMA_BEVFUSION_TARGET_EV74)
  return neat::SimaKernelTarget::EV74;
#elif defined(SIMA_BEVFUSION_TARGET_A65)
  return neat::SimaKernelTarget::A65;
#else
#error "define SIMA_BEVFUSION_TARGET_EV74 or SIMA_BEVFUSION_TARGET_A65"
#endif
}
inline bool on_ev74() {
  return suite_target() == neat::SimaKernelTarget::EV74;
}
inline std::string target_name() {
  return on_ev74() ? "EV74" : "A65";
}

// Named scenarios, required in order. A failed comparison marks its scenario
// FAIL and the suite continues; finish() fails the executable.
class Suite {
public:
  Suite(std::string name, std::vector<std::string> required)
      : name_(std::move(name)), required_(std::move(required)) {}
  template <class Fn> void run(std::string_view name, Fn&& fn) {
    require(completed_ < required_.size() && required_[completed_] == name,
            "scenario inventory/order mismatch: " + std::string(name));
    std::cout << "KERNEL_CASE kernel=" << name_ << " target=" << target_name() << " case=" << name
              << " phase=start\n";
    try {
      fn();
      std::cout << "KERNEL_CASE kernel=" << name_ << " target=" << target_name() << " case=" << name
                << " result=PASS\n";
    } catch (const Failure& error) {
      ++failed_;
      std::cerr << "KERNEL_CASE kernel=" << name_ << " target=" << target_name() << " case=" << name
                << " result=FAIL reason=" << error.what() << '\n';
    }
    ++completed_;
  }
  void finish() const {
    require(completed_ == required_.size(), "required scenario was not executed");
    std::cout << "KERNEL_SUITE kernel=" << name_ << " target=" << target_name()
              << " completed=" << completed_ << " required=" << required_.size()
              << " failed=" << failed_ << '\n';
    require(failed_ == 0, name_ + ": numerical/contract scenario failures");
  }

private:
  std::string name_;
  std::vector<std::string> required_;
  std::size_t completed_ = 0;
  std::size_t failed_ = 0;
};

template <class Fn> int main_test(int argc, Fn&& fn) {
  std::cout << std::unitbuf;
  try {
    require(argc == 1, "canonical kernel suite takes no arguments; every scenario is required");
    fn();
    return 0;
  } catch (const Failure& error) {
    std::cerr << "[ASSERTION_AFTER_CLOSE] " << error.what() << '\n';
    return 1;
  } catch (const std::exception& error) {
    // An unexpected runtime failure never becomes an expected numerical PASS.
    std::cerr << "[RUNTIME_OR_SETUP_FAILURE] " << error.what() << '\n';
    return 2;
  }
}

// ---------------------------------------------------------------- tensors ---
inline void name(neat::Tensor& tensor, std::string_view value) {
  tensor.route.name = value;
}
template <class T>
neat::Tensor tensor(const std::vector<T>& values, std::vector<std::int64_t> shape,
                    std::string_view route) {
  auto result = neat::Tensor::from_vector(values, std::move(shape), neat::TensorMemory::EV74);
  name(result, route);
  return result;
}
// BF16 bits as a BF16 tensor; from_vector has no BF16 overload.
inline neat::Tensor bf16(const std::vector<std::uint16_t>& bits, std::vector<std::int64_t> shape,
                         std::string_view route) {
  auto result = tensor(bits, std::move(shape), route);
  result.dtype = neat::TensorDType::BFloat16;
  return result;
}
inline neat::Tensor count(std::int32_t value, std::string_view route = "count") {
  return tensor(std::vector<std::int32_t>{value}, {1}, route);
}

// One read-only EV74 carrier holding several input views at 64-byte-aligned
// offsets, with 0xa5 guard bytes around each view. check() proves that neither
// a view's bytes nor the guards changed. Every frame of one run must use the
// same layout: graph ingress places each view at the seed frame's offset.
class SharedCarrier {
public:
  template <class T> std::size_t add(const std::vector<T>& values) {
    const std::size_t offset = bytes_.size() + 64;
    bytes_.resize(offset + values.size() * sizeof(T) + 64, 0xa5);
    std::memcpy(bytes_.data() + offset, values.data(), values.size() * sizeof(T));
    while (bytes_.size() % 64)
      bytes_.push_back(0xa5);
    return offset;
  }
  void seal() {
    parent_ = neat::Tensor::from_vector(bytes_, {static_cast<std::int64_t>(bytes_.size())},
                                        neat::TensorMemory::EV74);
  }
  neat::Tensor view(std::size_t offset, neat::TensorDType dtype, std::vector<std::int64_t> shape,
                    std::string_view route) const {
    auto result = parent_;
    result.dtype = dtype;
    result.shape = std::move(shape);
    result.strides_bytes.clear();
    result.axis_semantics.clear();
    result.layout = neat::TensorLayout::Unknown;
    result.byte_offset = static_cast<std::int64_t>(offset);
    name(result, route);
    return result;
  }
  void check() const {
    require(parent_.copy_dense_bytes_tight() == bytes_,
            "read-only shared carrier or its guard bytes changed");
  }

private:
  std::vector<std::uint8_t> bytes_;
  neat::Tensor parent_;
};

// ------------------------------------------------------------------- runs ---
// An ordered multi-tensor Input, as public tensor-set ingress.
inline neat::InputOptions input_options() {
  neat::InputOptions result;
  result.payload_type = neat::PayloadType::Tensor;
  result.memory_policy = neat::InputMemoryPolicy::Ev74;
  result.pool_min_buffers = result.pool_max_buffers = 4;
  result.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                         "storage=(string)tensorbuffer";
  return result;
}
inline neat::RunOptions run_options() {
  neat::RunOptions result;
  result.output_memory = neat::OutputMemory::Owned;
  result.queue_depth = 1;
  result.input_timeout_ms = kTimeoutMs;
  return result;
}
inline neat::Graph graph(const std::shared_ptr<neat::Node>& node) {
  neat::Graph result;
  result.add(neat::nodes::Input(input_options()));
  result.add(node);
  result.add(neat::nodes::Output());
  return result;
}
// Build once, run every frame in order on the same Run, close, then return the
// owned outputs. Assertions happen after the Run has closed normally.
inline std::vector<neat::TensorList> execute(const std::shared_ptr<neat::Node>& node,
                                             const std::vector<neat::TensorList>& frames) {
  require(!frames.empty(), "empty fixture sequence");
  auto pipeline = graph(node);
  auto run = pipeline.build(frames.front(), run_options());
  std::vector<neat::TensorList> results;
  try {
    for (const auto& frame : frames)
      results.push_back(run.run(frame, kTimeoutMs));
  } catch (...) {
    run.close();
    throw;
  }
  run.close();
  std::cout << "KERNEL_RUN_CLOSED frames=" << results.size() << " outputs=owned\n";
  return results;
}
// Graph admission must reject the request before any execution.
inline void expect_rejected(const std::shared_ptr<neat::Node>& node, const neat::TensorList& frame,
                            std::string_view needle, const std::string& what) {
  auto pipeline = graph(node);
  try {
    auto unexpected = pipeline.build(frame, run_options());
    unexpected.close();
  } catch (const std::exception& error) {
    require(std::string(error.what()).find(needle) != std::string::npos,
            what + ": rejected without '" + std::string(needle) + "': " + error.what());
    return;
  }
  throw Failure(what + ": graph admission accepted an invalid request");
}
// Node construction must reject structurally invalid options.
template <class Fn>
void expect_invalid_options(Fn&& fn, std::string_view needle, const std::string& what) {
  try {
    fn();
  } catch (const std::invalid_argument& error) {
    require(std::string(error.what()).find(needle) != std::string::npos,
            what + ": " + error.what());
    return;
  }
  throw Failure(what + ": invalid options were accepted");
}

// Public payload as T, after checking dtype and shape; the bytes are compared
// byte for byte with the oracle by the caller.
template <class T>
std::vector<T> values(const neat::Tensor& tensor, neat::TensorDType dtype,
                      const std::vector<std::int64_t>& shape, const std::string& what) {
  require(tensor.dtype == dtype && tensor.shape == shape,
          what + ": public output dtype/shape mismatch");
  const auto bytes = tensor.copy_dense_bytes_tight();
  std::size_t count = 1;
  for (const auto extent : shape)
    count *= static_cast<std::size_t>(extent);
  require(bytes.size() == count * sizeof(T), what + ": public output payload extent mismatch");
  std::vector<T> result(count);
  if (!bytes.empty())
    std::memcpy(result.data(), bytes.data(), bytes.size());
  return result;
}
// The stored bytes of a published tensor, for outputs whose storage is not
// dense in their logical shape (an MLA-tiled BF16 canvas).
inline std::vector<std::uint8_t> stored_bytes(const neat::Tensor& tensor, neat::TensorDType dtype,
                                              const std::vector<std::int64_t>& shape,
                                              const std::string& what) {
  require(tensor.dtype == dtype && tensor.shape == shape,
          what + ": public output dtype/shape mismatch");
  auto bytes = tensor.copy_dense_bytes_tight();
  require(bytes.size() == tensor.dense_bytes_tight(),
          what + ": public output payload extent mismatch");
  return bytes;
}
template <class T>
void expect_bytes(const std::vector<T>& actual, const std::vector<T>& expected,
                  const std::string& what) {
  if (actual == expected)
    return;
  std::size_t first = 0;
  while (first < actual.size() && first < expected.size() && actual[first] == expected[first])
    ++first;
  throw Failure(what + ": differs from the oracle at element " + std::to_string(first) + " of " +
                std::to_string(expected.size()));
}

// ------------------------------------------------------------ calibration ---
inline std::vector<std::vector<float>> calibration_values(const ref::Calibration& calibration) {
  std::vector<float> c2l, k, aug;
  for (std::size_t n = 0; n < calibration.camera2lidar.size(); ++n) {
    c2l.insert(c2l.end(), calibration.camera2lidar[n].begin(), calibration.camera2lidar[n].end());
    k.insert(k.end(), calibration.intrinsics[n].begin(), calibration.intrinsics[n].end());
    aug.insert(aug.end(), calibration.img_aug[n].begin(), calibration.img_aug[n].end());
  }
  return {c2l, k, aug, {calibration.lidar_aug.begin(), calibration.lidar_aug.end()}};
}
inline neat::TensorList calibration_tensors(const ref::Calibration& calibration) {
  const auto values = calibration_values(calibration);
  const auto cameras = static_cast<std::int64_t>(calibration.camera2lidar.size());
  return {tensor(values[0], {cameras, 4, 4}, "camera2lidar"),
          tensor(values[1], {cameras, 4, 4}, "camera_intrinsics"),
          tensor(values[2], {cameras, 4, 4}, "img_aug_matrix"),
          tensor(values[3], {4, 4}, "lidar_aug_matrix")};
}
// SimaBEVPoolPlan's public plan bytes, through its own A65 graph.
inline std::vector<std::uint8_t> plan_from_node(const neat::SimaBEVPoolPlanOptions& options,
                                                const ref::Calibration& calibration) {
  const auto outputs =
      execute(neat::nodes::SimaBEVPoolPlan(options), {calibration_tensors(calibration)});
  require(outputs.size() == 1 && outputs[0].size() == 1, "plan output arity");
  const auto& plan = outputs[0][0];
  return values<std::uint8_t>(plan, neat::TensorDType::UInt8, plan.shape, "plan");
}

} // namespace bevfusion_test

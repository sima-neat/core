// Run's TensorList verbs (H0 plan 3.3; Core's Tensor surface is TensorList-only):
// pull(milliseconds) -> Result<TensorList>, operator()(TensorList) -> TensorList, on_output with a
// TensorList callback, and the allocation behind alloc() (RunAllocation.h). H0's stats() has no
// public Run method (plan 3.3; unit_graph_api_surface_test keeps Run::stats() private):
// MeasureScope's MeasureReport carries the counters, which the Dataflow backend fills from its
// SessionReport. The TensorList push and alloc() themselves are in RunPush.cpp.
//
// Both backends: on Dataflow the Run state builds the tensors from the message (DataflowRun.cpp);
// on GStreamer the pulled Sample projects to them (SampleProjection.h), with a stream id interned
// per Run.

#include "pipeline/Run.h"
#include "RunInternal.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/SampleProjection.h"
#include "pipeline/runtime/RunAllocation.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace simaai::neat {

namespace runtime {

// ---- alloc() ------------------------------------------------------------------------------------

Mapping RunAllocationMapper::operator()(MapMode /*mode*/) const {
  Mapping mapping;
  if (!allocation) {
    return mapping;
  }
  std::lock_guard<std::mutex> lock(allocation->mu);
  if (allocation->state != RunAllocation::State::Writable) {
    return mapping; // pushed (or being pushed): no data
  }
  ++allocation->maps;
  mapping.data = allocation->data;
  mapping.size_bytes = allocation->size;
  mapping.unmap = [held = allocation] {
    std::lock_guard<std::mutex> unmap_lock(held->mu);
    if (held->maps > 0) {
      --held->maps;
    }
  };
  return mapping;
}

Tensor make_allocated_tensor(const std::shared_ptr<RunAllocation>& allocation, TensorDType dtype,
                             std::vector<std::int64_t> shape) {
  auto storage = std::make_shared<TensorBuffer>();
  storage->kind = StorageKind::CpuExternal;
  storage->device = {DeviceType::CPU, 0};
  storage->size_bytes = allocation->size;
  storage->holder = std::shared_ptr<void>(allocation, allocation.get());
  storage->data = allocation->data;
  storage->map_fn = RunAllocationMapper{allocation};
  allocation->storage = storage;
  Tensor out;
  out.storage = std::move(storage);
  out.dtype = dtype;
  out.shape = std::move(shape);
  out.device = {DeviceType::CPU, 0};
  out.read_only = false;
  return out;
}

std::shared_ptr<RunAllocation> allocation_of(const Tensor& tensor, const RunCore& core) {
  if (!tensor.storage) {
    return nullptr;
  }
  const auto* mapper = tensor.storage->map_fn.target<RunAllocationMapper>();
  if (!mapper || !mapper->allocation) {
    return nullptr;
  }
  // The same Run: the same control block. The allocation's weak reference keeps that block, so
  // another Run can never share it.
  const std::weak_ptr<const RunCore> self = core.weak_from_this();
  const std::weak_ptr<const RunCore>& owner = mapper->allocation->run;
  if (owner.owner_before(self) || self.owner_before(owner)) {
    return nullptr;
  }
  return mapper->allocation;
}

std::vector<std::int64_t> declared_alloc_shape(const InputOptions& declared,
                                               const std::string& where) {
  if (declared.shape.empty()) {
    pipeline_internal::error_util::throw_session_error(
        error_codes::kInputShape, where + ": the input declares no shape", {},
        "Declare it with Graph::input(name, shape, dtype, max_shape) to allocate its tensors.");
  }
  std::vector<std::int64_t> shape(declared.shape.size());
  for (std::size_t i = 0; i < declared.shape.size(); ++i) {
    const std::int64_t bound = declared.max_shape.empty() ? -1 : declared.max_shape[i];
    shape[i] = declared.shape[i] > 0 ? declared.shape[i] : bound;
    if (shape[i] <= 0) {
      pipeline_internal::error_util::throw_session_error(
          error_codes::kInputShape,
          where + ": axis " + std::to_string(i) +
              " of the declared shape is dynamic without a bound",
          {},
          "Give every dynamic axis a bound (Graph::input's max_shape) to allocate its tensors.");
    }
  }
  return shape;
}

namespace {

void require_pushable_locked(const RunAllocation& allocation, const char* where) {
  if (allocation.state != RunAllocation::State::Writable) {
    pipeline_internal::error_util::throw_session_error(
        error_codes::kInputShape,
        std::string(where) + ": this Tensor from Run::alloc was pushed already", {},
        "A Tensor from Run::alloc is pushed once; take another with Run::alloc.");
  }
  if (allocation.maps != 0) {
    pipeline_internal::error_util::throw_session_error(
        error_codes::kInputShape,
        std::string(where) + ": this Tensor from Run::alloc is still mapped (" +
            std::to_string(allocation.maps) + " live Mapping)",
        {}, "Release every Mapping of it before pushing it.");
  }
}

} // namespace

void require_pushable(RunAllocation& allocation, const char* where) {
  std::lock_guard<std::mutex> lock(allocation.mu);
  require_pushable_locked(allocation, where);
}

void begin_allocated_push(RunAllocation& allocation, const char* where) {
  std::lock_guard<std::mutex> lock(allocation.mu);
  require_pushable_locked(allocation, where);
  allocation.state = RunAllocation::State::Pushing;
}

void end_allocated_push(RunAllocation& allocation, bool published) {
  std::lock_guard<std::mutex> lock(allocation.mu);
  if (!published) {
    allocation.state = RunAllocation::State::Writable;
    return;
  }
  allocation.state = RunAllocation::State::Pushed;
  if (std::shared_ptr<TensorBuffer> storage = allocation.storage.lock()) {
    storage->data = nullptr;
    storage->size_bytes = 0;
  }
}

// ---- on_output ----------------------------------------------------------------------------------

struct OutputDelivery {
  std::optional<std::string> output; // the output it owns, as pulls resolve it (see owned_output)
  std::optional<std::string> name;   // as registered: nullopt for the default output
  std::thread thread;
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
};

struct OutputCallbacks {
  std::mutex mu;
  bool stopping = false;
  std::atomic<bool> stop{false};
  std::vector<std::shared_ptr<OutputDelivery>> deliveries;
};

} // namespace runtime

namespace {

namespace error_util = pipeline_internal::error_util;

// Core's int milliseconds for a chrono timeout: negative or too large to add to now is -1 (no
// limit), 0 a poll.
int timeout_ms_of(std::chrono::milliseconds timeout) {
  if (timeout.count() < 0 || timeout.count() > std::numeric_limits<int>::max()) {
    return -1;
  }
  return static_cast<int>(timeout.count());
}

// The output a pull or a callback names: its name, or for an unnamed call the Run's only named
// output (nullopt when it has none or several).
std::optional<std::string> owned_output(const runtime::RunCore& core,
                                        std::optional<std::string_view> name) {
  if (name) {
    return std::string(*name);
  }
  const std::vector<std::string> names = core.output_names();
  if (names.size() == 1U) {
    return names.front();
  }
  return std::nullopt;
}

bool delivered_to_callback(const runtime::RunCore& core, std::optional<std::string_view> name) {
  if (!core.output_callbacks_active.load(std::memory_order_acquire)) {
    return false;
  }
  const std::shared_ptr<runtime::OutputCallbacks> callbacks = core.output_callbacks;
  if (!callbacks) {
    return false;
  }
  const std::optional<std::string> output = owned_output(core, name);
  std::lock_guard<std::mutex> lock(callbacks->mu);
  return std::any_of(callbacks->deliveries.begin(), callbacks->deliveries.end(),
                     [&](const auto& delivery) { return delivery->output == output; });
}

// One pull of an item's tensors, on either backend, without the callback check (the delivery
// thread uses it too).
PullStatus pull_tensors_on_core(runtime::RunCore& core, std::optional<std::string_view> output,
                                int timeout_ms, TensorList& out, PullError* err) {
  if (core.dataflow_) {
    return core.dataflow_->pull(output, timeout_ms, out, err);
  }
  Sample sample;
  const PullStatus status = output ? core.pull_named_output(*output, timeout_ms, sample, err)
                                   : core.pull(timeout_ms, sample, err);
  if (status != PullStatus::Ok) {
    return status;
  }
  const Stamp stamp = pipeline_internal::stamp_of_pulled_sample(sample, &core.tensor_streams);
  std::optional<TensorList> tensors = pipeline_internal::tensors_from_pulled_sample(sample, stamp);
  if (!tensors) {
    error_util::set_pull_error(err, error_codes::kRuntimePull,
                               "Run::pull: the output item carries no tensor");
    return PullStatus::Error;
  }
  out = std::move(*tensors);
  return PullStatus::Ok;
}

Result<TensorList> pull_result(runtime::RunCore* core, std::optional<std::string_view> output,
                               std::chrono::milliseconds timeout) {
  const std::string where =
      output ? "Run::pull(\"" + std::string(*output) + "\")" : std::string("Run::pull");
  PullError err;
  if (!core) {
    err.code = error_codes::kRuntimePull;
    err.message = where + ": stream is closed";
    return Result<TensorList>::failure(PullStatus::Closed, std::move(err));
  }
  if (delivered_to_callback(*core, output)) {
    err.code = error_codes::kRuntimePull;
    err.message = where + ": the output is delivered to on_output; it cannot be pulled";
    return Result<TensorList>::failure(PullStatus::Error, std::move(err));
  }
  TensorList tensors;
  const PullStatus status =
      pull_tensors_on_core(*core, output, timeout_ms_of(timeout), tensors, &err);
  if (status == PullStatus::Ok) {
    return Result<TensorList>::success(std::move(tensors));
  }
  return Result<TensorList>::failure(status, std::move(err));
}

// The delivery thread of one on_output callback. It holds the Run's core, so the core outlives it
// even when a stop could not wait for it (a callback still running: leak rather than hang).
void deliver(std::shared_ptr<runtime::RunCore> core,
             std::shared_ptr<runtime::OutputCallbacks> callbacks,
             std::shared_ptr<runtime::OutputDelivery> delivery,
             std::function<void(TensorList)> callback) {
  // Each pull is bounded, so a stop that did not end a pull still ends the loop in this time.
  constexpr int kPullQuantumMs = 100;
  const std::optional<std::string_view> output =
      delivery->name ? std::optional<std::string_view>(*delivery->name) : std::nullopt;
  while (!callbacks->stop.load(std::memory_order_acquire)) {
    TensorList tensors;
    PullError err;
    const PullStatus status = pull_tensors_on_core(*core, output, kPullQuantumMs, tensors, &err);
    if (status == PullStatus::Timeout) {
      continue;
    }
    if (status == PullStatus::Closed) {
      break;
    }
    if (status == PullStatus::Error) {
      if (!callbacks->stop.load(std::memory_order_acquire)) {
        core->set_terminal_error(std::move(err));
      }
      break;
    }
    try {
      callback(std::move(tensors));
    } catch (const std::exception& e) {
      PullError failure;
      failure.code = error_codes::kRuntimePull;
      failure.message = std::string("Run::on_output: the callback threw: ") + e.what();
      core->set_terminal_error(std::move(failure));
      break;
    } catch (...) {
      PullError failure;
      failure.code = error_codes::kRuntimePull;
      failure.message = "Run::on_output: the callback threw a non-standard exception";
      core->set_terminal_error(std::move(failure));
      break;
    }
  }
  {
    std::lock_guard<std::mutex> lock(delivery->mu);
    delivery->done = true;
  }
  delivery->cv.notify_all();
}

void register_output_callback(const std::shared_ptr<runtime::RunCore>& core,
                              std::optional<std::string_view> name,
                              std::function<void(TensorList)> callback, const char* where) {
  if (!core) {
    error_util::throw_session_error(error_codes::kRuntimePull,
                                    std::string(where) + ": stream is closed");
  }
  if (!callback) {
    error_util::throw_session_error(error_codes::kPipelineShape,
                                    std::string(where) + ": the callback is empty");
  }
  if (name) {
    const std::vector<std::string> names = core->output_names();
    if (name->empty() || std::find(names.begin(), names.end(), *name) == names.end()) {
      std::string list = "[";
      for (std::size_t i = 0; i < names.size(); ++i) {
        list += (i == 0 ? "" : ", ") + names[i];
      }
      error_util::throw_session_error(error_codes::kRuntimePull,
                                      std::string(where) + "(\"" + std::string(*name) +
                                          "\"): unknown output. Available outputs: " + list + "]");
    }
  }
  if (!core->running()) {
    error_util::throw_session_error(error_codes::kRuntimePull,
                                    std::string(where) + ": the Run is stopped");
  }
  std::shared_ptr<runtime::OutputCallbacks> callbacks;
  {
    std::lock_guard<std::mutex> lock(core->output_callbacks_mu);
    if (!core->output_callbacks) {
      core->output_callbacks = std::make_shared<runtime::OutputCallbacks>();
    }
    callbacks = core->output_callbacks;
  }
  auto delivery = std::make_shared<runtime::OutputDelivery>();
  delivery->output = owned_output(*core, name);
  if (name) {
    delivery->name = std::string(*name);
  }
  std::lock_guard<std::mutex> lock(callbacks->mu);
  if (callbacks->stopping) {
    error_util::throw_session_error(error_codes::kRuntimePull,
                                    std::string(where) + ": the Run is stopped");
  }
  for (const auto& existing : callbacks->deliveries) {
    if (existing->output == delivery->output) {
      error_util::throw_session_error(error_codes::kPipelineShape,
                                      std::string(where) +
                                          ": the output already has an on_output callback");
    }
  }
  delivery->thread = std::thread(deliver, core, callbacks, delivery, std::move(callback));
  callbacks->deliveries.push_back(std::move(delivery));
  core->output_callbacks_active.store(true, std::memory_order_release);
}

} // namespace

void runtime::RunCore::stop_output_callbacks() noexcept {
  std::shared_ptr<OutputCallbacks> callbacks;
  {
    std::lock_guard<std::mutex> lock(output_callbacks_mu);
    callbacks = output_callbacks;
  }
  if (!callbacks) {
    return;
  }
  std::vector<std::shared_ptr<OutputDelivery>> deliveries;
  {
    std::lock_guard<std::mutex> lock(callbacks->mu);
    callbacks->stopping = true;
    callbacks->stop.store(true, std::memory_order_release);
    deliveries.swap(callbacks->deliveries); // one caller joins each thread
  }
  // A callback in progress finishes; a delivery that does not end within the budget is left to
  // end on its own (it holds the core): leak rather than hang (CLAUDE.md, teardown).
  const auto budget = std::chrono::milliseconds(
      std::max(1, pipeline_internal::env_int("SIMA_PIPELINE_STREAM_STOP_TIMEOUT_MS", 2000)));
  const auto deadline = std::chrono::steady_clock::now() + budget;
  for (const auto& delivery : deliveries) {
    if (!delivery->thread.joinable()) {
      continue;
    }
    if (delivery->thread.get_id() == std::this_thread::get_id()) {
      delivery->thread.detach(); // stop() from inside the callback: it ends when the callback does
      continue;
    }
    bool done = false;
    {
      std::unique_lock<std::mutex> lock(delivery->mu);
      done = delivery->cv.wait_until(lock, deadline, [&] { return delivery->done; });
    }
    if (done) {
      delivery->thread.join();
    } else {
      std::fprintf(stderr,
                   "[WARN] Run::stop: an on_output callback did not return within %lld ms; its "
                   "thread is left to finish on its own\n",
                   static_cast<long long>(budget.count()));
      delivery->thread.detach();
    }
  }
}

Result<TensorList> Run::pull(std::chrono::milliseconds timeout) {
  require_async_pull_mode("Run::pull");
  return pull_result(core_.get(), std::nullopt, timeout);
}

Result<TensorList> Run::pull(std::string_view output_name, std::chrono::milliseconds timeout) {
  require_async_pull_mode("Run::pull(name)");
  return pull_result(core_.get(), output_name, timeout);
}

TensorList Run::operator()(const TensorList& input) {
  if (!push(input)) {
    error_util::throw_session_error(
        error_codes::kRuntimePull,
        "Run::operator(): the push was refused (the input is closed or the Run is stopped)");
  }
  return pull(std::chrono::milliseconds(-1)).value();
}

void Run::on_output(std::function<void(TensorList)> callback) {
  require_async_pull_mode("Run::on_output");
  register_output_callback(core_, std::nullopt, std::move(callback), "Run::on_output");
}

void Run::on_output(std::string_view output_name, std::function<void(TensorList)> callback) {
  require_async_pull_mode("Run::on_output(name)");
  register_output_callback(core_, output_name, std::move(callback), "Run::on_output");
}

namespace run_internal {

bool output_delivered_to_callback(const runtime::RunCore& core,
                                  std::optional<std::string_view> output) {
  return delivered_to_callback(core, output);
}

} // namespace run_internal

} // namespace simaai::neat

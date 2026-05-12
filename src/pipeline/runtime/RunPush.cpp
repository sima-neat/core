#include "pipeline/Run.h"
#include "RunInternal.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/SampleUtil.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <stdexcept>

namespace simaai::neat {

namespace {

bool push_return_debug_enabled() {
  return pipeline_internal::env_bool("SIMA_PIPELINE_PUSH_RETURN_DEBUG", false);
}

bool run_push_timing_enabled() {
  return pipeline_internal::env_bool("SIMA_RUN_PUSH_TIMING", false);
}

int run_push_timing_limit() {
  static const int limit =
      std::max(0, pipeline_internal::env_int("SIMA_RUN_PUSH_TIMING_LIMIT", 32));
  return limit;
}

std::string decorate_with_error_code(const std::string& code, const std::string& message) {
  return pipeline_internal::error_util::decorate_error(code, message);
}

void validate_image_inputs(const std::vector<cv::Mat>& inputs, const char* where) {
  const char* tag = where ? where : "Run";
  if (inputs.empty()) {
    throw std::runtime_error(decorate_with_error_code(error_codes::kRuntimePull,
                                                      std::string(tag) + ": empty image list"));
  }
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].empty()) {
      throw std::runtime_error(decorate_with_error_code(
          error_codes::kRuntimePull,
          std::string(tag) + ": empty image input at index " + std::to_string(i)));
    }
  }
}

bool input_options_expect_tensor_media(const std::optional<InputOptions>& opt) {
  if (!opt.has_value()) {
    return false;
  }
  const std::string media = pipeline_internal::lower_copy(opt->media_type);
  return media == "application/vnd.simaai.tensor";
}

#if defined(SIMA_WITH_OPENCV)
std::optional<cv::Mat> try_materialize_image_sample(const Sample& msg) {
  if (!sample_has_tensor_list(msg) || msg.tensors.size() != 1U) {
    return std::nullopt;
  }
  const Tensor& tensor = msg.tensors.front();
  if (!tensor.semantic.image.has_value()) {
    return std::nullopt;
  }
  const auto fmt = tensor.semantic.image->format;
  if (auto view = tensor.map_cv_mat_view(fmt); view.has_value()) {
    return view->mat.clone();
  }
  try {
    return tensor.to_cv_mat_copy(fmt);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}
#endif

struct InputQueueAdmission {
  bool accepted = false;
  int max = 0;
  const char* reason = "";
};

template <typename StateT>
InputQueueAdmission admit_input_queue_locked(StateT& st, std::unique_lock<std::mutex>& lock,
                                             bool block) {
  const int max = st.opt.queue_depth;
  if (st.input_closed) {
    return {false, max, "input_closed"};
  }

  if (st.opt.overflow_policy == OverflowPolicy::Block) {
    if (!block) {
      if (run_internal::queue_full(st.in_queue, max)) {
        return {false, max, "queue_full_block"};
      }
    } else {
      st.in_cv.wait(lock, [&]() {
        return st.stop_requested.load() || st.input_closed ||
               !run_internal::queue_full(st.in_queue, max);
      });
    }
  } else if (st.opt.overflow_policy == OverflowPolicy::DropIncoming) {
    if (run_internal::queue_full(st.in_queue, max)) {
      st.inputs_dropped.fetch_add(1, std::memory_order_relaxed);
      return {false, max, "queue_full_drop_newest"};
    }
  } else if (st.opt.overflow_policy == OverflowPolicy::KeepLatest) {
    if (run_internal::queue_full(st.in_queue, max)) {
      st.in_queue.pop_front();
      st.inputs_dropped.fetch_add(1, std::memory_order_relaxed);
    }
  }

  if (st.stop_requested.load()) {
    return {false, max, "stop_requested"};
  }
  if (st.input_closed) {
    return {false, max, "input_closed"};
  }
  return {true, max, ""};
}

template <typename StateT>
void maybe_log_push_rejection(const StateT& st, const InputQueueAdmission& admission) {
  if (!push_return_debug_enabled() || admission.accepted || !admission.reason ||
      !admission.reason[0]) {
    return;
  }
  std::fprintf(stderr, "[PIPELINE] push_return_false reason=%s qsize=%zu max=%d drop=%d\n",
               admission.reason, st.in_queue.size(), admission.max,
               static_cast<int>(st.opt.overflow_policy));
}

} // namespace

void Run::require_async_mode(const char* where) const {
  if (state_ && state_->mode == RunMode::Sync) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull,
        std::string(where ? where : "Run") +
            ": push/try_push are not allowed in sync mode; use run(...) instead"));
  }
}

bool Run::push_impl(const cv::Mat& input, bool block) {
  if (!state_) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push: stream is closed"));
  }
  if (!state_->supports_push) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull, "Run::push: pipeline has no Input (push not supported)"));
  }
  auto st = state_;
  {
    std::unique_lock<std::mutex> lock(st->in_mu);
    const InputQueueAdmission admission = admit_input_queue_locked(*st, lock, block);
    if (!admission.accepted) {
      maybe_log_push_rejection(*st, admission);
      return false;
    }

    InputItem item;
    item.kind = QueuedInputKind::Mat;
    const bool force_copy = st->opt.advanced.copy_input;
    if (force_copy) {
      item.mat = input.clone();
    } else {
      item.mat = input;
    }
    st->in_queue.push_back(std::move(item));
    st->inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
  }
  st->in_cv.notify_one();
  return true;
}

bool Run::push_impl(const simaai::neat::Tensor& input, bool block) {
  if (!state_) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push: stream is closed"));
  }
  if (!state_->supports_push) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull, "Run::push: pipeline has no Input (push not supported)"));
  }
  auto st = state_;
  {
    std::unique_lock<std::mutex> lock(st->in_mu);
    const InputQueueAdmission admission = admit_input_queue_locked(*st, lock, block);
    if (!admission.accepted) {
      maybe_log_push_rejection(*st, admission);
      return false;
    }

    InputItem item;
    item.kind = QueuedInputKind::Tensor;
    const bool force_copy = st->opt.advanced.copy_input;
    if (force_copy && !st->stream_opt.require_device_visible_input) {
      item.tensor = input.clone();
      item.tensor.read_only = false;
    } else {
      item.tensor = input;
      if (!st->stream_opt.require_device_visible_input) {
        const std::size_t qsize = st->in_queue.size();
        run_internal::maybe_force_copy_for_backpressure(
            item.tensor, qsize, "queue_depth",
            pipeline_internal::env_bool("SIMA_GRAPH_ZERO_COPY_DEBUG", false));
      }
    }
    st->in_queue.push_back(std::move(item));
    st->inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
  }
  st->in_cv.notify_one();
  return true;
}

bool Run::push_holder_impl(const std::shared_ptr<void>& holder, bool block) {
  if (!state_) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push_holder: stream is closed"));
  }
  if (!state_->supports_push) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull, "Run::push_holder: pipeline has no Input (push not supported)"));
  }
  if (!holder) {
    throw std::invalid_argument(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push_holder: missing holder"));
  }
  auto st = state_;
  {
    std::unique_lock<std::mutex> lock(st->in_mu);
    const InputQueueAdmission admission = admit_input_queue_locked(*st, lock, block);
    if (!admission.accepted) {
      maybe_log_push_rejection(*st, admission);
      return false;
    }

    InputItem item;
    item.kind = QueuedInputKind::Holder;
    item.holder = holder;
    st->in_queue.push_back(std::move(item));
    st->inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
  }
  st->in_cv.notify_one();
  return true;
}

bool Run::push_message_impl(const Sample& msg, bool block) {
  if (!state_) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push: stream is closed"));
  }
  if (!state_->supports_push) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull, "Run::push: pipeline has no Input (push not supported)"));
  }
  const bool timing = run_push_timing_enabled();
  const auto t0 =
      timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  std::size_t q_before = 0;
  std::size_t q_after = 0;
  std::chrono::steady_clock::time_point t_admit{};
  std::chrono::steady_clock::time_point t_copy{};
  auto st = state_;
  {
    std::unique_lock<std::mutex> lock(st->in_mu);
    q_before = st->in_queue.size();
    const InputQueueAdmission admission = admit_input_queue_locked(*st, lock, block);
    if (timing) {
      t_admit = std::chrono::steady_clock::now();
    }
    if (!admission.accepted) {
      maybe_log_push_rejection(*st, admission);
      return false;
    }

    InputItem item;
    item.kind = QueuedInputKind::Message;
    Sample copy = msg;
    const bool force_copy = st->opt.advanced.copy_input;
    if (force_copy && !st->stream_opt.require_device_visible_input) {
      run_internal::force_copy_sample_if_zero_copy(copy);
    } else if (!st->stream_opt.require_device_visible_input) {
      const std::size_t qsize = st->in_queue.size();
      run_internal::maybe_force_copy_for_backpressure(
          copy, qsize, "queue_depth",
          pipeline_internal::env_bool("SIMA_GRAPH_ZERO_COPY_DEBUG", false));
    }
    item.msg = std::move(copy);
    if (timing) {
      t_copy = std::chrono::steady_clock::now();
    }
    st->in_queue.push_back(std::move(item));
    q_after = st->in_queue.size();
    st->inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
  }
  st->in_cv.notify_one();
  if (timing) {
    static std::atomic<int> printed{0};
    const int idx = printed.fetch_add(1, std::memory_order_relaxed);
    const int limit = run_push_timing_limit();
    if (idx < limit || (limit > 0 && (idx % limit) == 0)) {
      const auto t1 = std::chrono::steady_clock::now();
      const auto admit_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(t_admit - t0).count();
      const auto copy_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(t_copy - t_admit).count();
      const auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      std::fprintf(stderr,
                   "[RUN_PUSH_TIMING] Run::push_message_impl idx=%d q_before=%zu q_after=%zu "
                   "admit_ns=%lld copy_ns=%lld total_ns=%lld block=%d\n",
                   idx, q_before, q_after, static_cast<long long>(admit_ns),
                   static_cast<long long>(copy_ns), static_cast<long long>(total_ns),
                   block ? 1 : 0);
    }
  }
  return true;
}

bool Run::push_sample_impl(const Sample& msg, bool block) {
  if (state_ && !input_options_expect_tensor_media(state_->tensor_input_opt_for_cv)) {
#if defined(SIMA_WITH_OPENCV)
    if (auto mat = try_materialize_image_sample(msg); mat.has_value()) {
      return push_impl(*mat, block);
    }
#endif
  }
  return push_message_impl(msg, block);
}

bool Run::push(const std::vector<cv::Mat>& inputs) {
  require_async_mode("Run::push");
  validate_image_inputs(inputs, "Run::push");
  if (!input_options_expect_tensor_media(state_->tensor_input_opt_for_cv)) {
    if (inputs.size() != 1U) {
      throw std::runtime_error(decorate_with_error_code(
          error_codes::kRuntimePull,
          "Run::push: raw-image ingress supports exactly one cv::Mat per inference item"));
    }
    return push_impl(inputs.front(), true);
  }
  TensorList tensors;
  tensors.reserve(inputs.size());
  for (const auto& input : inputs) {
    tensors.emplace_back(
        tensor_from_cv_mat(input, *state_->tensor_input_opt_for_cv, "Run::push(inputs)"));
  }
  return push_message_impl(
      pipeline_internal::sample_from_tensors_for_input(tensors, *state_->tensor_input_opt_for_cv),
      true);
}

bool Run::try_push(const std::vector<cv::Mat>& inputs) {
  require_async_mode("Run::try_push");
  validate_image_inputs(inputs, "Run::try_push");
  if (!input_options_expect_tensor_media(state_->tensor_input_opt_for_cv)) {
    if (inputs.size() != 1U) {
      throw std::runtime_error(decorate_with_error_code(
          error_codes::kRuntimePull,
          "Run::try_push: raw-image ingress supports exactly one cv::Mat per inference item"));
    }
    return push_impl(inputs.front(), false);
  }
  TensorList tensors;
  tensors.reserve(inputs.size());
  for (const auto& input : inputs) {
    tensors.emplace_back(
        tensor_from_cv_mat(input, *state_->tensor_input_opt_for_cv, "Run::try_push(inputs)"));
  }
  return push_message_impl(
      pipeline_internal::sample_from_tensors_for_input(tensors, *state_->tensor_input_opt_for_cv),
      false);
}

bool Run::push(const TensorList& inputs) {
  require_async_mode("Run::push");
  if (inputs.empty()) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push: empty tensor list"));
  }
  if (state_ && state_->input_route_processor) {
    return push_message_impl(state_->input_route_processor->process_tensors(inputs, "Run::push"),
                             true);
  }
  if (state_ && state_->tensor_input_opt_for_cv.has_value()) {
    return push_message_impl(
        pipeline_internal::sample_from_tensors_for_input(inputs, *state_->tensor_input_opt_for_cv),
        true);
  }
  return push_message_impl(sample_from_tensors(inputs), true);
}

bool Run::try_push(const TensorList& inputs) {
  require_async_mode("Run::try_push");
  if (inputs.empty()) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::try_push: empty tensor list"));
  }
  if (state_ && state_->input_route_processor) {
    return push_message_impl(
        state_->input_route_processor->process_tensors(inputs, "Run::try_push"), false);
  }
  if (state_ && state_->tensor_input_opt_for_cv.has_value()) {
    return push_message_impl(
        pipeline_internal::sample_from_tensors_for_input(inputs, *state_->tensor_input_opt_for_cv),
        false);
  }
  return push_message_impl(sample_from_tensors(inputs), false);
}

bool Run::push_holder(const std::shared_ptr<void>& holder) {
  require_async_mode("Run::push_holder");
  return push_holder_impl(holder, true);
}

bool Run::try_push_holder(const std::shared_ptr<void>& holder) {
  require_async_mode("Run::try_push_holder");
  return push_holder_impl(holder, false);
}

bool Run::push(const SampleList& msgs) {
  require_async_mode("Run::push");
  if (msgs.empty()) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push: empty sample list"));
  }
  if (state_ && state_->input_route_processor) {
    return push_message_impl(state_->input_route_processor->process_samples(msgs, "Run::push"),
                             true);
  }
  for (const auto& msg : msgs) {
    if (!push_sample_impl(msg, true)) {
      return false;
    }
  }
  return true;
}

bool Run::try_push(const SampleList& msgs) {
  require_async_mode("Run::try_push");
  if (msgs.empty()) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::try_push: empty sample list"));
  }
  if (state_ && state_->input_route_processor) {
    return push_message_impl(state_->input_route_processor->process_samples(msgs, "Run::try_push"),
                             false);
  }
  for (const auto& msg : msgs) {
    if (!push_sample_impl(msg, false)) {
      return false;
    }
  }
  return true;
}

} // namespace simaai::neat

#include "pipeline/Run.h"
#include "RunInternal.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/EnvUtil.h"

#include <cstdio>
#include <mutex>
#include <stdexcept>

namespace simaai::neat {

namespace {

bool push_return_debug_enabled() {
  return pipeline_internal::env_bool("SIMA_PIPELINE_PUSH_RETURN_DEBUG", false);
}

std::string decorate_with_error_code(const std::string& code, const std::string& message) {
  return pipeline_internal::error_util::decorate_error(code, message);
}

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
    item.kind = InputKind::Mat;
    if (st->opt.advanced.copy_input) {
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
    item.kind = InputKind::Tensor;
    if (st->opt.advanced.copy_input) {
      item.tensor = input.clone();
    } else {
      item.tensor = input;
      const std::size_t qsize = st->in_queue.size();
      run_internal::maybe_force_copy_for_backpressure(
          item.tensor, qsize, "queue_depth",
          pipeline_internal::env_bool("SIMA_GRAPH_ZERO_COPY_DEBUG", false));
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
    item.kind = InputKind::Holder;
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
  auto st = state_;
  {
    std::unique_lock<std::mutex> lock(st->in_mu);
    const InputQueueAdmission admission = admit_input_queue_locked(*st, lock, block);
    if (!admission.accepted) {
      maybe_log_push_rejection(*st, admission);
      return false;
    }

    InputItem item;
    item.kind = InputKind::Message;
    Sample copy = msg;
    const std::size_t qsize = st->in_queue.size();
    run_internal::maybe_force_copy_for_backpressure(
        copy, qsize, "queue_depth",
        pipeline_internal::env_bool("SIMA_GRAPH_ZERO_COPY_DEBUG", false));
    item.msg = std::move(copy);
    st->in_queue.push_back(std::move(item));
    st->inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
  }
  st->in_cv.notify_one();
  return true;
}

bool Run::push(const cv::Mat& input) {
  return push_impl(input, true);
}

bool Run::try_push(const cv::Mat& input) {
  return push_impl(input, false);
}

bool Run::push(const simaai::neat::Tensor& input) {
  return push_impl(input, true);
}

bool Run::try_push(const simaai::neat::Tensor& input) {
  return push_impl(input, false);
}

bool Run::push_holder(const std::shared_ptr<void>& holder) {
  return push_holder_impl(holder, true);
}

bool Run::try_push_holder(const std::shared_ptr<void>& holder) {
  return push_holder_impl(holder, false);
}

bool Run::push(const Sample& msg) {
  return push_message_impl(msg, true);
}

bool Run::try_push(const Sample& msg) {
  return push_message_impl(msg, false);
}

} // namespace simaai::neat

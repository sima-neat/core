#include "pipeline/Run.h"
#include "RunInternal.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/RealtimeFrameCredit.h"
#include "pipeline/internal/SampleUtil.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <string_view>

namespace simaai::neat::graph {
InputOptions input_opts_from_spec(const OutputSpec& spec, bool complete);
} // namespace simaai::neat::graph

namespace simaai::neat {

namespace {

bool push_return_debug_enabled() {
  return pipeline_internal::env_bool("SIMA_PIPELINE_PUSH_RETURN_DEBUG", false);
}

bool run_push_timing_enabled() {
  return pipeline_internal::env_bool("SIMA_RUN_PUSH_TIMING", false);
}

bool allow_public_cross_run_gstsample_push() {
  return pipeline_internal::env_bool("SIMA_ALLOW_CROSS_RUN_GSTSAMPLE_PUSH", false);
}

void enforce_public_sample_push_transferable(const Sample& msg, const char* where) {
  if (!allow_public_cross_run_gstsample_push() &&
      pipeline_internal::sample_has_device_gstsample_producer_lifetime(msg,
                                                                       /*require_expired=*/false)) {
    std::string reason;
    if (!pipeline_internal::sample_has_transferable_zero_copy_loan(msg, &reason)) {
      throw std::runtime_error(pipeline_internal::cross_run_zero_copy_sample_error(where) +
                               (reason.empty() ? std::string{} : " Reason: " + reason + "."));
    }
  }
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
  const std::string media = pipeline_internal::lower_copy(resolve_input_media_type(*opt));
  return media == "application/vnd.simaai.tensor";
}

const runtime::PipelineSegmentPlan* graph_default_input_segment(const runtime::RunCore& core) {
  if (!core.graph_execution_) {
    return nullptr;
  }
  const auto& plan = core.graph_execution_->plan;
  if (!plan.default_input.has_value()) {
    return nullptr;
  }
  const auto segment_index = plan.default_input->segment;
  if (segment_index == static_cast<std::size_t>(-1) ||
      segment_index >= plan.pipeline_segments.size()) {
    return nullptr;
  }
  return &plan.pipeline_segments[segment_index];
}

const runtime::PipelineSegmentPlan*
graph_input_segment_for_endpoint(const runtime::RunCore& core, const runtime::Endpoint& endpoint) {
  if (!core.graph_execution_) {
    return nullptr;
  }
  const auto segment_index = endpoint.segment;
  if (segment_index == static_cast<std::size_t>(-1) ||
      segment_index >= core.graph_execution_->plan.pipeline_segments.size()) {
    return nullptr;
  }
  return &core.graph_execution_->plan.pipeline_segments[segment_index];
}

pipeline_internal::InputRouteProcessorPtr
graph_default_input_route_processor(const runtime::RunCore& core) {
  const runtime::PipelineSegmentPlan* segment = graph_default_input_segment(core);
  if (!segment || !segment->boundary_hints.has_value()) {
    return nullptr;
  }
  return segment->boundary_hints->input_route_processor;
}

pipeline_internal::InputRouteProcessorPtr
graph_input_route_processor_for_endpoint(const runtime::RunCore& core,
                                         const runtime::Endpoint& endpoint) {
  const runtime::PipelineSegmentPlan* segment = graph_input_segment_for_endpoint(core, endpoint);
  if (!segment || !segment->boundary_hints.has_value()) {
    return nullptr;
  }
  return segment->boundary_hints->input_route_processor;
}

std::optional<InputOptions> graph_default_ingress_input(const runtime::RunCore& core,
                                                        std::size_t index = 0) {
  const runtime::PipelineSegmentPlan* segment = graph_default_input_segment(core);
  if (!segment) {
    return std::nullopt;
  }
  if (segment->boundary_hints.has_value() &&
      index < segment->boundary_hints->ingress_inputs.size()) {
    return segment->boundary_hints->ingress_inputs[index];
  }
  if (segment->boundary_hints.has_value() && !segment->boundary_hints->ingress_inputs.empty()) {
    return segment->boundary_hints->ingress_inputs.front();
  }
  if (segment->input_complete) {
    return simaai::neat::graph::input_opts_from_spec(segment->input_spec, segment->input_complete);
  }
  return std::nullopt;
}

std::optional<InputOptions> graph_ingress_input_for_endpoint(const runtime::RunCore& core,
                                                             const runtime::Endpoint& endpoint,
                                                             std::size_t index = 0) {
  const runtime::PipelineSegmentPlan* segment = graph_input_segment_for_endpoint(core, endpoint);
  if (!segment) {
    return std::nullopt;
  }
  if (segment->boundary_hints.has_value() &&
      index < segment->boundary_hints->ingress_inputs.size()) {
    return segment->boundary_hints->ingress_inputs[index];
  }
  if (segment->boundary_hints.has_value() && !segment->boundary_hints->ingress_inputs.empty()) {
    return segment->boundary_hints->ingress_inputs.front();
  }
  if (segment->input_complete) {
    return simaai::neat::graph::input_opts_from_spec(segment->input_spec, segment->input_complete);
  }
  return std::nullopt;
}

std::string available_input_names(const runtime::RunCore& core) {
  if (!core.graph_execution_) {
    return "[]";
  }
  std::vector<std::string> names;
  names.reserve(core.graph_execution_->plan.named_inputs.size());
  for (const auto& kv : core.graph_execution_->plan.named_inputs) {
    names.push_back(kv.first);
  }
  std::sort(names.begin(), names.end());
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (i > 0)
      oss << ", ";
    oss << names[i];
  }
  oss << "]";
  return oss.str();
}

runtime::Endpoint named_input_endpoint_or_throw(const runtime::RunCore& core,
                                                std::string_view input_name) {
  if (!core.graph_execution_) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull, "Run::push(name): named inputs require a graph-backed Run"));
  }
  if (input_name.empty()) {
    throw std::runtime_error(decorate_with_error_code(error_codes::kRuntimePull,
                                                      "Run::push(name): input name is empty"));
  }
  const auto& named = core.graph_execution_->plan.named_inputs;
  auto it = named.find(std::string(input_name));
  if (it == named.end()) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull,
        "Run::push(\"" + std::string(input_name) +
            "\"): unknown input. Available inputs: " + available_input_names(core)));
  }
  return it->second;
}

runtime::EdgeRouterOptions graph_router_options_for_push(const runtime::RunCore& core, bool block) {
  runtime::EdgeRouterOptions options = core.graph_options.router_options();
  if (!block) {
    options.push_timeout_ms = 0;
  }
  return options;
}

std::string default_input_name(const runtime::RunCore& core) {
  if (!core.graph_execution_ || !core.graph_execution_->plan.default_input.has_value()) {
    return "default";
  }
  const auto& def = *core.graph_execution_->plan.default_input;
  for (const auto& [name, ep] : core.graph_execution_->plan.named_inputs) {
    if (ep.node == def.node && ep.port == def.port && ep.kind == def.kind) {
      return name;
    }
  }
  return "default";
}

Sample stamp_public_graph_ingress_sample(runtime::RunCore& core, const Sample& in) {
  Sample out = in;
  if (out.stream_id.empty()) {
    out.stream_id = "default";
  }

  // Prefer a run-local public-ingress sequence over frame_id fallback whenever sequence fields are
  // absent. frame_id can repeat across looping sources, reconnects, generated tests, and fanout
  // reuse; orig_input_seq/input_seq are the stable correlation key for this Run's measurements.
  if (out.orig_input_seq < 0 && out.input_seq < 0) {
    const auto seq = core.next_public_graph_input_seq.fetch_add(1, std::memory_order_relaxed);
    out.orig_input_seq = seq;
    out.input_seq = seq;
  } else if (out.orig_input_seq < 0 && out.input_seq >= 0) {
    out.orig_input_seq = out.input_seq;
  } else if (out.input_seq < 0 && out.orig_input_seq >= 0) {
    out.input_seq = out.orig_input_seq;
  }
  return out;
}

bool push_graph_samples_to_endpoint(runtime::RunCore& core, const runtime::Endpoint& endpoint,
                                    std::string_view endpoint_name, const Sample& msgs,
                                    bool block) {
  for (const auto& msg : msgs) {
    enforce_public_sample_push_transferable(msg, "Run::push");
    Sample stamped = stamp_public_graph_ingress_sample(core, msg);
    if (pipeline_internal::env_bool("SIMA_SAMPLE_TIMING_DEBUG", false)) {
      std::fprintf(stderr,
                   "[SAMPLE_TIMING] graph_push_endpoint node=%zu port=%u has_port=%d "
                   "kind=%d frame_id=%lld pts_ns=%lld dts_ns=%lld duration_ns=%lld "
                   "stream_id=%s block=%d\n",
                   static_cast<std::size_t>(endpoint.node), static_cast<unsigned>(endpoint.port),
                   endpoint.port != simaai::neat::graph::kInvalidPort ? 1 : 0,
                   static_cast<int>(stamped.kind), static_cast<long long>(stamped.frame_id),
                   static_cast<long long>(stamped.pts_ns), static_cast<long long>(stamped.dts_ns),
                   static_cast<long long>(stamped.duration_ns), stamped.stream_id.c_str(),
                   block ? 1 : 0);
    }
    core.inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
    const auto entry_at = std::chrono::steady_clock::now();
    runtime::trace_graph_message_event(runtime::TraceGraphMessageEventType::GraphEntry,
                                       core.graph_execution_.get(), runtime::invalid_edge_index(),
                                       stamped, endpoint_name);
    if (!core.graph_push(endpoint.node, endpoint.port,
                         endpoint.port != simaai::neat::graph::kInvalidPort, stamped,
                         graph_router_options_for_push(core, block))) {
      return false;
    }
    core.record_graph_sample_entry(endpoint_name, stamped, entry_at);
    core.inputs_pushed.fetch_add(1, std::memory_order_relaxed);
  }
  return true;
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

void release_realtime_credits_for_dropped_input(const InputItem& item, const char* mode) {
  if (item.kind != QueuedInputKind::Message) {
    return;
  }
  pipeline_internal::release_realtime_frame_credits_for_sample(item.msg, mode);
}

InputQueueAdmission admit_input_queue_locked(runtime::RunCore& core,
                                             runtime::PipelineSegmentRuntime& segment,
                                             std::unique_lock<std::mutex>& lock, bool block) {
  const int max = core.opt.queue_depth;
  if (segment.input_closed) {
    return {false, max, "input_closed"};
  }

  // A blocking caller must never silently drop. Internal inter-segment forwarding pushes with
  // block=true; honoring that by waiting for room (regardless of the user-facing overflow policy)
  // lets downstream slowness backpressure upstream instead of being mistaken for a fatal push
  // failure. The drop / keep-latest policies apply only to non-blocking callers (the user-facing
  // try_push at graph ingress, which pushes with block=false).
  if (block) {
    segment.in_cv.wait(lock, [&]() {
      return core.stop_requested.load() || segment.input_closed ||
             !run_internal::queue_full(segment.in_queue, max);
    });
  } else if (core.opt.overflow_policy == OverflowPolicy::Block) {
    if (run_internal::queue_full(segment.in_queue, max)) {
      return {false, max, "queue_full_block"};
    }
  } else if (core.opt.overflow_policy == OverflowPolicy::DropIncoming) {
    if (run_internal::queue_full(segment.in_queue, max)) {
      core.inputs_dropped.fetch_add(1, std::memory_order_relaxed);
      return {false, max, "queue_full_drop_newest"};
    }
  } else if (core.opt.overflow_policy == OverflowPolicy::KeepLatest) {
    if (run_internal::queue_full(segment.in_queue, max)) {
      release_realtime_credits_for_dropped_input(segment.in_queue.front(),
                                                 "async-input-drop-oldest");
      segment.in_queue.pop_front();
      core.inputs_dropped.fetch_add(1, std::memory_order_relaxed);
    }
  }

  if (core.stop_requested.load()) {
    return {false, max, "stop_requested"};
  }
  if (segment.input_closed) {
    return {false, max, "input_closed"};
  }
  return {true, max, ""};
}

void maybe_log_push_rejection(const runtime::RunCore& core,
                              const runtime::PipelineSegmentRuntime& segment,
                              const InputQueueAdmission& admission) {
  if (!push_return_debug_enabled() || admission.accepted || !admission.reason ||
      !admission.reason[0]) {
    return;
  }
  std::fprintf(stderr, "[PIPELINE] push_return_false reason=%s qsize=%zu max=%d drop=%d\n",
               admission.reason, segment.in_queue.size(), admission.max,
               static_cast<int>(core.opt.overflow_policy));
}

bool push_mat_to_core(runtime::RunCore& core, const cv::Mat& input, bool block) {
  if (!core.pipeline.supports_push) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull, "Run::push: pipeline has no Input (push not supported)"));
  }
  auto* st = &core;
  {
    std::unique_lock<std::mutex> lock(st->pipeline.in_mu);
    const InputQueueAdmission admission = admit_input_queue_locked(*st, st->pipeline, lock, block);
    if (!admission.accepted) {
      maybe_log_push_rejection(*st, st->pipeline, admission);
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
    st->pipeline.in_queue.push_back(std::move(item));
    st->inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
  }
  st->pipeline.in_cv.notify_one();
  return true;
}

bool push_message_to_core(runtime::RunCore& core, const Sample& msg, bool block) {
  if (core.graph_execution_) {
    return core.push_samples(Sample{msg}, block);
  }
  if (!core.pipeline.supports_push) {
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
  auto* st = &core;
  {
    std::unique_lock<std::mutex> lock(st->pipeline.in_mu);
    q_before = st->pipeline.in_queue.size();
    const InputQueueAdmission admission = admit_input_queue_locked(*st, st->pipeline, lock, block);
    if (timing) {
      t_admit = std::chrono::steady_clock::now();
    }
    if (!admission.accepted) {
      maybe_log_push_rejection(*st, st->pipeline, admission);
      return false;
    }

    InputItem item;
    item.kind = QueuedInputKind::Message;
    Sample copy = msg;
    const bool force_copy = st->opt.advanced.copy_input;
    if (force_copy && !st->pipeline.stream_opt.require_device_visible_input) {
      run_internal::force_copy_sample_if_zero_copy(copy);
    } else if (!st->pipeline.stream_opt.require_device_visible_input) {
      const std::size_t qsize = st->pipeline.in_queue.size();
      run_internal::maybe_force_copy_for_backpressure(
          copy, qsize, "queue_depth",
          pipeline_internal::env_bool("SIMA_GRAPH_ZERO_COPY_DEBUG", false));
    }
    item.msg = std::move(copy);
    if (timing) {
      t_copy = std::chrono::steady_clock::now();
    }
    st->pipeline.in_queue.push_back(std::move(item));
    q_after = st->pipeline.in_queue.size();
    st->inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
  }
  st->pipeline.in_cv.notify_one();
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

bool push_sample_to_core(runtime::RunCore& core, const Sample& msg, bool block) {
  if (core.push_sample_policy == runtime::PushSamplePolicy::PreserveSample) {
    return push_message_to_core(core, msg, block);
  }
  if (!core.pipeline.stream_opt.require_device_visible_input &&
      !input_options_expect_tensor_media(core.pipeline.tensor_input_opt_for_cv)) {
#if defined(SIMA_WITH_OPENCV)
    if (auto mat = try_materialize_image_sample(msg); mat.has_value()) {
      return push_mat_to_core(core, *mat, block);
    }
#endif
  }
  return push_message_to_core(core, msg, block);
}

} // namespace

void Run::require_async_mode(const char* where) const {
  if (core_ && core_->mode == RunMode::Sync) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull,
        std::string(where ? where : "Run") +
            ": push/try_push are not allowed in sync mode; use run(...) instead"));
  }
}

bool Run::push_impl(const cv::Mat& input, bool block) {
  if (!core_) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push: stream is closed"));
  }
  if (!core_->pipeline.supports_push) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull, "Run::push: pipeline has no Input (push not supported)"));
  }
  auto st = core_;
  {
    std::unique_lock<std::mutex> lock(st->pipeline.in_mu);
    const InputQueueAdmission admission = admit_input_queue_locked(*st, st->pipeline, lock, block);
    if (!admission.accepted) {
      maybe_log_push_rejection(*st, st->pipeline, admission);
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
    st->pipeline.in_queue.push_back(std::move(item));
    st->inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
  }
  st->pipeline.in_cv.notify_one();
  return true;
}

bool Run::push_impl(const simaai::neat::Tensor& input, bool block) {
  if (!core_) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push: stream is closed"));
  }
  if (!core_->pipeline.supports_push) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull, "Run::push: pipeline has no Input (push not supported)"));
  }
  auto st = core_;
  {
    std::unique_lock<std::mutex> lock(st->pipeline.in_mu);
    const InputQueueAdmission admission = admit_input_queue_locked(*st, st->pipeline, lock, block);
    if (!admission.accepted) {
      maybe_log_push_rejection(*st, st->pipeline, admission);
      return false;
    }

    InputItem item;
    item.kind = QueuedInputKind::Tensor;
    const bool force_copy = st->opt.advanced.copy_input;
    if (force_copy && !st->pipeline.stream_opt.require_device_visible_input) {
      item.tensor = input.clone();
      item.tensor.read_only = false;
    } else {
      item.tensor = input;
      if (!st->pipeline.stream_opt.require_device_visible_input) {
        const std::size_t qsize = st->pipeline.in_queue.size();
        run_internal::maybe_force_copy_for_backpressure(
            item.tensor, qsize, "queue_depth",
            pipeline_internal::env_bool("SIMA_GRAPH_ZERO_COPY_DEBUG", false));
      }
    }
    st->pipeline.in_queue.push_back(std::move(item));
    st->inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
  }
  st->pipeline.in_cv.notify_one();
  return true;
}

bool Run::push_holder_impl(const std::shared_ptr<void>& holder, bool block) {
  if (!core_) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push_holder: stream is closed"));
  }
  if (!core_->pipeline.supports_push) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull, "Run::push_holder: pipeline has no Input (push not supported)"));
  }
  if (!holder) {
    throw std::invalid_argument(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push_holder: missing holder"));
  }
  auto st = core_;
  {
    std::unique_lock<std::mutex> lock(st->pipeline.in_mu);
    const InputQueueAdmission admission = admit_input_queue_locked(*st, st->pipeline, lock, block);
    if (!admission.accepted) {
      maybe_log_push_rejection(*st, st->pipeline, admission);
      return false;
    }

    InputItem item;
    item.kind = QueuedInputKind::Holder;
    item.holder = holder;
    st->pipeline.in_queue.push_back(std::move(item));
    st->inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
  }
  st->pipeline.in_cv.notify_one();
  return true;
}

bool Run::push_message_impl(const Sample& msg, bool block) {
  if (!core_) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push: stream is closed"));
  }
  enforce_public_sample_push_transferable(msg, "Run::push");
  return push_message_to_core(*core_, msg, block);
}
bool Run::push_sample_impl(const Sample& msg, bool block) {
  enforce_public_sample_push_transferable(msg, "Run::push");
  if (core_ && core_->push_sample_policy == runtime::PushSamplePolicy::PreserveSample) {
    return push_message_impl(msg, block);
  }
  if (core_ && !core_->pipeline.stream_opt.require_device_visible_input &&
      !input_options_expect_tensor_media(core_->pipeline.tensor_input_opt_for_cv)) {
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
  if (core_ && core_->graph_execution_) {
    TensorList tensors;
    tensors.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const InputOptions input_opt =
          graph_default_ingress_input(*core_, i).value_or(InputOptions{});
      tensors.emplace_back(tensor_from_cv_mat(inputs[i], input_opt, "Run::push(inputs)"));
    }
    if (auto route = graph_default_input_route_processor(*core_)) {
      return core_->push_samples(Sample{route->process_tensors(tensors, "Run::push")}, true);
    }
    const std::optional<InputOptions> input_opt = graph_default_ingress_input(*core_);
    if (!input_options_expect_tensor_media(input_opt) && tensors.size() != 1U) {
      throw std::runtime_error(decorate_with_error_code(
          error_codes::kRuntimePull,
          "Run::push: raw-image graph ingress supports exactly one cv::Mat per inference item"));
    }
    Sample sample = input_opt.has_value()
                        ? pipeline_internal::sample_from_tensors_for_input(tensors, *input_opt)
                        : sample_from_tensors(tensors);
    return core_->push_samples(Sample{std::move(sample)}, true);
  }
  if (!input_options_expect_tensor_media(core_->pipeline.tensor_input_opt_for_cv)) {
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
        tensor_from_cv_mat(input, *core_->pipeline.tensor_input_opt_for_cv, "Run::push(inputs)"));
  }
  return push_message_impl(pipeline_internal::sample_from_tensors_for_input(
                               tensors, *core_->pipeline.tensor_input_opt_for_cv),
                           true);
}

bool Run::push(std::string_view input_name, const std::vector<cv::Mat>& inputs) {
  require_async_mode("Run::push(name)");
  validate_image_inputs(inputs, "Run::push(name)");
  if (!core_) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push(name): stream is closed"));
  }
  const runtime::Endpoint endpoint = named_input_endpoint_or_throw(*core_, input_name);
  TensorList tensors;
  tensors.reserve(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const InputOptions input_opt =
        graph_ingress_input_for_endpoint(*core_, endpoint, i).value_or(InputOptions{});
    tensors.emplace_back(tensor_from_cv_mat(inputs[i], input_opt, "Run::push(name, inputs)"));
  }
  if (auto route = graph_input_route_processor_for_endpoint(*core_, endpoint)) {
    return push_graph_samples_to_endpoint(
        *core_, endpoint, input_name, Sample{route->process_tensors(tensors, "Run::push(name)")},
        true);
  }
  const std::optional<InputOptions> input_opt = graph_ingress_input_for_endpoint(*core_, endpoint);
  if (!input_options_expect_tensor_media(input_opt) && tensors.size() != 1U) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull,
        "Run::push(name): raw-image graph ingress supports exactly one cv::Mat per inference "
        "item"));
  }
  Sample sample = input_opt.has_value()
                      ? pipeline_internal::sample_from_tensors_for_input(tensors, *input_opt)
                      : sample_from_tensors(tensors);
  return push_graph_samples_to_endpoint(*core_, endpoint, input_name, Sample{std::move(sample)},
                                        true);
}

bool Run::try_push(const std::vector<cv::Mat>& inputs) {
  require_async_mode("Run::try_push");
  validate_image_inputs(inputs, "Run::try_push");
  if (core_ && core_->graph_execution_) {
    TensorList tensors;
    tensors.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const InputOptions input_opt =
          graph_default_ingress_input(*core_, i).value_or(InputOptions{});
      tensors.emplace_back(tensor_from_cv_mat(inputs[i], input_opt, "Run::try_push(inputs)"));
    }
    if (auto route = graph_default_input_route_processor(*core_)) {
      return core_->push_samples(Sample{route->process_tensors(tensors, "Run::try_push")}, false);
    }
    const std::optional<InputOptions> input_opt = graph_default_ingress_input(*core_);
    if (!input_options_expect_tensor_media(input_opt) && tensors.size() != 1U) {
      throw std::runtime_error(decorate_with_error_code(
          error_codes::kRuntimePull,
          "Run::try_push: raw-image graph ingress supports exactly one cv::Mat per inference "
          "item"));
    }
    Sample sample = input_opt.has_value()
                        ? pipeline_internal::sample_from_tensors_for_input(tensors, *input_opt)
                        : sample_from_tensors(tensors);
    return core_->push_samples(Sample{std::move(sample)}, false);
  }
  if (!input_options_expect_tensor_media(core_->pipeline.tensor_input_opt_for_cv)) {
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
    tensors.emplace_back(tensor_from_cv_mat(input, *core_->pipeline.tensor_input_opt_for_cv,
                                            "Run::try_push(inputs)"));
  }
  return push_message_impl(pipeline_internal::sample_from_tensors_for_input(
                               tensors, *core_->pipeline.tensor_input_opt_for_cv),
                           false);
}

bool Run::try_push(std::string_view input_name, const std::vector<cv::Mat>& inputs) {
  require_async_mode("Run::try_push(name)");
  validate_image_inputs(inputs, "Run::try_push(name)");
  if (!core_) {
    throw std::runtime_error(decorate_with_error_code(error_codes::kRuntimePull,
                                                      "Run::try_push(name): stream is closed"));
  }
  const runtime::Endpoint endpoint = named_input_endpoint_or_throw(*core_, input_name);
  TensorList tensors;
  tensors.reserve(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const InputOptions input_opt =
        graph_ingress_input_for_endpoint(*core_, endpoint, i).value_or(InputOptions{});
    tensors.emplace_back(tensor_from_cv_mat(inputs[i], input_opt, "Run::try_push(name, inputs)"));
  }
  if (auto route = graph_input_route_processor_for_endpoint(*core_, endpoint)) {
    return push_graph_samples_to_endpoint(
        *core_, endpoint, input_name,
        Sample{route->process_tensors(tensors, "Run::try_push(name)")}, false);
  }
  const std::optional<InputOptions> input_opt = graph_ingress_input_for_endpoint(*core_, endpoint);
  if (!input_options_expect_tensor_media(input_opt) && tensors.size() != 1U) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull,
        "Run::try_push(name): raw-image graph ingress supports exactly one cv::Mat per inference "
        "item"));
  }
  Sample sample = input_opt.has_value()
                      ? pipeline_internal::sample_from_tensors_for_input(tensors, *input_opt)
                      : sample_from_tensors(tensors);
  return push_graph_samples_to_endpoint(*core_, endpoint, input_name, Sample{std::move(sample)},
                                        false);
}

bool Run::push(const TensorList& inputs) {
  require_async_mode("Run::push");
  if (inputs.empty()) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push: empty tensor list"));
  }
  if (core_ && core_->graph_execution_) {
    if (auto route = graph_default_input_route_processor(*core_)) {
      return core_->push_samples(Sample{route->process_tensors(inputs, "Run::push")}, true);
    }
    const std::optional<InputOptions> input_opt = graph_default_ingress_input(*core_);
    if (!input_options_expect_tensor_media(input_opt) && inputs.size() != 1U) {
      throw std::runtime_error(decorate_with_error_code(
          error_codes::kRuntimePull,
          "Run::push: raw-image graph ingress supports exactly one tensor per inference item"));
    }
    Sample sample = input_opt.has_value()
                        ? pipeline_internal::sample_from_tensors_for_input(inputs, *input_opt)
                        : sample_from_tensors(inputs);
    return core_->push_samples(Sample{std::move(sample)}, true);
  }
  if (core_ && core_->pipeline.input_route_processor) {
    return push_message_impl(
        core_->pipeline.input_route_processor->process_tensors(inputs, "Run::push"), true);
  }
  if (core_ && core_->pipeline.tensor_input_opt_for_cv.has_value()) {
    return push_message_impl(pipeline_internal::sample_from_tensors_for_input(
                                 inputs, *core_->pipeline.tensor_input_opt_for_cv),
                             true);
  }
  return push_message_impl(sample_from_tensors(inputs), true);
}

bool Run::push(std::string_view input_name, const TensorList& inputs) {
  require_async_mode("Run::push(name)");
  if (inputs.empty()) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push(name): empty tensor list"));
  }
  if (!core_) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push(name): stream is closed"));
  }
  const runtime::Endpoint endpoint = named_input_endpoint_or_throw(*core_, input_name);
  if (auto route = graph_input_route_processor_for_endpoint(*core_, endpoint)) {
    return push_graph_samples_to_endpoint(*core_, endpoint, input_name,
                                          Sample{route->process_tensors(inputs, "Run::push(name)")},
                                          true);
  }
  const std::optional<InputOptions> input_opt = graph_ingress_input_for_endpoint(*core_, endpoint);
  if (!input_options_expect_tensor_media(input_opt) && inputs.size() != 1U) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull,
        "Run::push(name): raw-image graph ingress supports exactly one tensor per inference item"));
  }
  Sample sample = input_opt.has_value()
                      ? pipeline_internal::sample_from_tensors_for_input(inputs, *input_opt)
                      : sample_from_tensors(inputs);
  return push_graph_samples_to_endpoint(*core_, endpoint, input_name, Sample{std::move(sample)},
                                        true);
}

bool Run::try_push(const TensorList& inputs) {
  require_async_mode("Run::try_push");
  if (inputs.empty()) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::try_push: empty tensor list"));
  }
  if (core_ && core_->graph_execution_) {
    if (auto route = graph_default_input_route_processor(*core_)) {
      return core_->push_samples(Sample{route->process_tensors(inputs, "Run::try_push")}, false);
    }
    const std::optional<InputOptions> input_opt = graph_default_ingress_input(*core_);
    if (!input_options_expect_tensor_media(input_opt) && inputs.size() != 1U) {
      throw std::runtime_error(decorate_with_error_code(
          error_codes::kRuntimePull,
          "Run::try_push: raw-image graph ingress supports exactly one tensor per inference item"));
    }
    Sample sample = input_opt.has_value()
                        ? pipeline_internal::sample_from_tensors_for_input(inputs, *input_opt)
                        : sample_from_tensors(inputs);
    return core_->push_samples(Sample{std::move(sample)}, false);
  }
  if (core_ && core_->pipeline.input_route_processor) {
    return push_message_impl(
        core_->pipeline.input_route_processor->process_tensors(inputs, "Run::try_push"), false);
  }
  if (core_ && core_->pipeline.tensor_input_opt_for_cv.has_value()) {
    return push_message_impl(pipeline_internal::sample_from_tensors_for_input(
                                 inputs, *core_->pipeline.tensor_input_opt_for_cv),
                             false);
  }
  return push_message_impl(sample_from_tensors(inputs), false);
}

bool Run::try_push(std::string_view input_name, const TensorList& inputs) {
  require_async_mode("Run::try_push(name)");
  if (inputs.empty()) {
    throw std::runtime_error(decorate_with_error_code(error_codes::kRuntimePull,
                                                      "Run::try_push(name): empty tensor list"));
  }
  if (!core_) {
    throw std::runtime_error(decorate_with_error_code(error_codes::kRuntimePull,
                                                      "Run::try_push(name): stream is closed"));
  }
  const runtime::Endpoint endpoint = named_input_endpoint_or_throw(*core_, input_name);
  if (auto route = graph_input_route_processor_for_endpoint(*core_, endpoint)) {
    return push_graph_samples_to_endpoint(
        *core_, endpoint, input_name, Sample{route->process_tensors(inputs, "Run::try_push(name)")},
        false);
  }
  const std::optional<InputOptions> input_opt = graph_ingress_input_for_endpoint(*core_, endpoint);
  if (!input_options_expect_tensor_media(input_opt) && inputs.size() != 1U) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull,
        "Run::try_push(name): raw-image graph ingress supports exactly one tensor per inference "
        "item"));
  }
  Sample sample = input_opt.has_value()
                      ? pipeline_internal::sample_from_tensors_for_input(inputs, *input_opt)
                      : sample_from_tensors(inputs);
  return push_graph_samples_to_endpoint(*core_, endpoint, input_name, Sample{std::move(sample)},
                                        false);
}

bool Run::push_holder(const std::shared_ptr<void>& holder) {
  require_async_mode("Run::push_holder");
  return push_holder_impl(holder, true);
}

bool Run::try_push_holder(const std::shared_ptr<void>& holder) {
  require_async_mode("Run::try_push_holder");
  return push_holder_impl(holder, false);
}

bool runtime::RunCore::push_samples(const Sample& msgs, bool block) {
  if (mode == RunMode::Sync) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull,
        "Run::push: push/try_push are not allowed in sync mode; use run(...) instead"));
  }
  if (msgs.empty()) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push: empty sample list"));
  }
  if (graph_execution_) {
    const auto& endpoint = graph_execution_->plan.default_input;
    if (!endpoint.has_value()) {
      throw std::runtime_error(decorate_with_error_code(
          error_codes::kRuntimePull,
          "Run::push: graph has no unambiguous default input; use push(name, ...). Available "
          "inputs: " +
              available_input_names(*this)));
    }
    return push_graph_samples_to_endpoint(*this, *endpoint, default_input_name(*this), msgs, block);
  }
  if (pipeline.input_route_processor) {
    return push_message_to_core(
        *this, pipeline.input_route_processor->process_samples(msgs, "Run::push"), block);
  }
  for (const auto& msg : msgs) {
    if (!push_sample_to_core(*this, msg, block)) {
      return false;
    }
  }
  return true;
}

bool runtime::RunCore::push_named_samples(std::string_view input_name, const Sample& msgs,
                                          bool block) {
  if (mode == RunMode::Sync) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull,
        "Run::push(name): push/try_push are not allowed in sync mode; use run(...) instead"));
  }
  if (msgs.empty()) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push(name): empty sample list"));
  }
  const runtime::Endpoint endpoint = named_input_endpoint_or_throw(*this, input_name);
  return push_graph_samples_to_endpoint(*this, endpoint, input_name, msgs, block);
}

bool Run::push(const Sample& msgs) {
  require_async_mode("Run::push");
  if (msgs.empty()) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push: empty sample list"));
  }
  if (core_ && core_->graph_execution_) {
    if (auto route = graph_default_input_route_processor(*core_)) {
      return core_->push_samples(Sample{route->process_samples(msgs, "Run::push")}, true);
    }
    return core_->push_samples(msgs, true);
  }
  if (core_ && core_->pipeline.input_route_processor) {
    return push_message_impl(
        core_->pipeline.input_route_processor->process_samples(msgs, "Run::push"), true);
  }
  for (const auto& msg : msgs) {
    if (!push_sample_impl(msg, true)) {
      return false;
    }
  }
  return true;
}

bool Run::push(std::string_view input_name, const Sample& msgs) {
  require_async_mode("Run::push(name)");
  if (msgs.empty()) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push(name): empty sample list"));
  }
  if (!core_) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::push(name): stream is closed"));
  }
  const runtime::Endpoint endpoint = named_input_endpoint_or_throw(*core_, input_name);
  if (auto route = graph_input_route_processor_for_endpoint(*core_, endpoint)) {
    return push_graph_samples_to_endpoint(*core_, endpoint, input_name,
                                          Sample{route->process_samples(msgs, "Run::push(name)")},
                                          true);
  }
  return core_->push_named_samples(input_name, msgs, true);
}

bool Run::try_push(const Sample& msgs) {
  require_async_mode("Run::try_push");
  if (msgs.empty()) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::try_push: empty sample list"));
  }
  if (core_ && core_->graph_execution_) {
    if (auto route = graph_default_input_route_processor(*core_)) {
      return core_->push_samples(Sample{route->process_samples(msgs, "Run::try_push")}, false);
    }
    return core_->push_samples(msgs, false);
  }
  if (core_ && core_->pipeline.input_route_processor) {
    return push_message_impl(
        core_->pipeline.input_route_processor->process_samples(msgs, "Run::try_push"), false);
  }
  for (const auto& msg : msgs) {
    if (!push_sample_impl(msg, false)) {
      return false;
    }
  }
  return true;
}

bool Run::try_push(std::string_view input_name, const Sample& msgs) {
  require_async_mode("Run::try_push(name)");
  if (msgs.empty()) {
    throw std::runtime_error(decorate_with_error_code(error_codes::kRuntimePull,
                                                      "Run::try_push(name): empty sample list"));
  }
  if (!core_) {
    throw std::runtime_error(decorate_with_error_code(error_codes::kRuntimePull,
                                                      "Run::try_push(name): stream is closed"));
  }
  const runtime::Endpoint endpoint = named_input_endpoint_or_throw(*core_, input_name);
  if (auto route = graph_input_route_processor_for_endpoint(*core_, endpoint)) {
    return push_graph_samples_to_endpoint(
        *core_, endpoint, input_name, Sample{route->process_samples(msgs, "Run::try_push(name)")},
        false);
  }
  return core_->push_named_samples(input_name, msgs, false);
}

} // namespace simaai::neat

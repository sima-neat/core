#include "EdgeRouter.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/RealtimeFrameCredit.h"
#include "pipeline/internal/SampleUtil.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace simaai::neat::runtime {
namespace {

std::uint64_t port_key(simaai::neat::graph::NodeId id, simaai::neat::graph::PortId port) {
  return (static_cast<std::uint64_t>(id) << 32) | static_cast<std::uint64_t>(port);
}

bool stop_requested(const EdgeRouterCallbacks& callbacks) {
  return callbacks.stop_requested && callbacks.stop_requested();
}

void request_stop(const EdgeRouterCallbacks& callbacks, const std::string& msg) {
  if (callbacks.request_stop) {
    callbacks.request_stop(msg);
  }
}

const char* graph_backpressure_timeout_explanation() {
  return " This can happen because of graph backpressure: downstream stages, appsinks, or the "
         "application are not draining outputs as fast as inputs are pushed, so an internal "
         "edge/pipeline queue filled before the timeout. Pull outputs concurrently, reduce the "
         "push rate, increase GraphRunOptions.edge_queue/push_timeout_ms, or remove/relax slow "
         "downstream stages.";
}

std::uint64_t elapsed_ns_since(std::chrono::steady_clock::time_point start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start)
          .count());
}

void atomic_add_max(std::atomic<std::uint64_t>& total, std::atomic<std::uint64_t>& max_value,
                    std::uint64_t ns) {
  total.fetch_add(ns, std::memory_order_relaxed);
  std::uint64_t cur = max_value.load(std::memory_order_relaxed);
  while (cur < ns && !max_value.compare_exchange_weak(cur, ns, std::memory_order_relaxed,
                                                      std::memory_order_relaxed)) {
  }
}

bool realtime_link_diag_enabled() {
  return pipeline_internal::env_bool("SIMA_GRAPH_REALTIME_LINK_DEBUG", false) ||
         pipeline_internal::env_bool("SIMA_GRAPH_DIAG_ON_STOP", false);
}

bool realtime_overwrite_debug_enabled() {
  return pipeline_internal::env_bool("SIMA_GRAPH_REALTIME_OVERWRITE_DEBUG", false);
}

bool realtime_credit_probe_enabled() {
  return pipeline_internal::env_bool("SIMA_GRAPH_REALTIME_CREDIT_PROBE", false) ||
         pipeline_internal::env_bool("SIMA_GRAPH_REALTIME_CREDIT_DEBUG", false) ||
         realtime_link_diag_enabled();
}

const char* target_kind_name(DownstreamTarget::Kind kind);

void log_realtime_credit_probe_basic(const char* event, const DownstreamTarget& target,
                                     std::size_t edge_index, const std::string& stream_id,
                                     std::size_t extra_value = 0) {
  const char* force = std::getenv("SIMA_GRAPH_REALTIME_CREDIT_PROBE_BASIC");
  if (!force || !*force || std::strcmp(force, "0") == 0) {
    return;
  }
  std::fprintf(
      stderr,
      "[GRAPH][credit-probe-basic] event=%s target=%s/%zu edge=%zu "
      "stream=%s extra=%zu probe=%d diag=%d env_probe=%s env_credit_debug=%s "
      "env_link_debug=%s env_diag=%s\n",
      event ? event : "<null>", target_kind_name(target.kind), target.index, edge_index,
      stream_id.empty() ? "<empty>" : stream_id.c_str(), extra_value,
      realtime_credit_probe_enabled() ? 1 : 0, realtime_link_diag_enabled() ? 1 : 0,
      std::getenv("SIMA_GRAPH_REALTIME_CREDIT_PROBE")
          ? std::getenv("SIMA_GRAPH_REALTIME_CREDIT_PROBE")
          : "<unset>",
      std::getenv("SIMA_GRAPH_REALTIME_CREDIT_DEBUG")
          ? std::getenv("SIMA_GRAPH_REALTIME_CREDIT_DEBUG")
          : "<unset>",
      std::getenv("SIMA_GRAPH_REALTIME_LINK_DEBUG") ? std::getenv("SIMA_GRAPH_REALTIME_LINK_DEBUG")
                                                    : "<unset>",
      std::getenv("SIMA_GRAPH_DIAG_ON_STOP") ? std::getenv("SIMA_GRAPH_DIAG_ON_STOP") : "<unset>");
}

int realtime_credit_probe_limit() {
  static const int value =
      std::max(0, pipeline_internal::env_int("SIMA_GRAPH_REALTIME_CREDIT_PROBE_LIMIT", 512));
  return value;
}

int realtime_credit_probe_every() {
  static const int value =
      std::max(0, pipeline_internal::env_int("SIMA_GRAPH_REALTIME_CREDIT_PROBE_EVERY", 0));
  return value;
}

constexpr int kDefaultRawCreditPerStream = 4;
constexpr int kDefaultRawCreditTotalCap = 8;

void validate_realtime_credit_option(const char* name, int value) {
  if (value == 0 || value < -1) {
    throw std::runtime_error(std::string("GraphLinkOptions::") + name +
                             " must be -1 or a positive value");
  }
}

int realtime_credit_max_inflight_per_stream(const GraphLinkOptions& options) {
  validate_realtime_credit_option("max_inflight_per_stream", options.max_inflight_per_stream);
  if (options.max_inflight_per_stream > 0) {
    return options.max_inflight_per_stream;
  }
  static const int value = [] {
    int parsed = 0;
    if (pipeline_internal::env_int("SIMA_GRAPH_REALTIME_CREDIT_MAX_INFLIGHT_PER_STREAM", &parsed)) {
      return std::max(0, parsed);
    }
    return std::max(0, pipeline_internal::env_int("SIMA_LATEST_MUX_MAX_INFLIGHT_PER_STREAM",
                                                  kDefaultRawCreditPerStream));
  }();
  return value;
}

int safe_total_credit_limit(int per_stream, std::size_t stream_count) {
  if (per_stream <= 0 || stream_count == 0U) {
    return 0;
  }
  if (stream_count > static_cast<std::size_t>(std::numeric_limits<int>::max() / per_stream)) {
    return std::numeric_limits<int>::max();
  }
  return per_stream * static_cast<int>(stream_count);
}

int realtime_credit_max_inflight_total(const GraphLinkOptions& options, int per_stream,
                                       std::size_t stream_count) {
  validate_realtime_credit_option("max_inflight_total", options.max_inflight_total);
  if (options.max_inflight_total > 0) {
    return options.max_inflight_total;
  }
  int parsed = 0;
  if (pipeline_internal::env_int("SIMA_GRAPH_REALTIME_CREDIT_MAX_INFLIGHT_GLOBAL", &parsed)) {
    return std::max(0, parsed);
  }
  /*
   * The total cap is a resource guard for all streams on this realtime link.
   * Derive it from the per-stream cap so 4-stream and 16-stream fan-in graphs
   * scale without a hidden fixed global bottleneck.
   */
  return std::min(safe_total_credit_limit(per_stream, stream_count), kDefaultRawCreditTotalCap);
}

int realtime_link_log_every() {
  static const int every =
      std::max(0, pipeline_internal::env_int("SIMA_GRAPH_REALTIME_LINK_LOG_EVERY", 0));
  return every;
}

const char* target_kind_name(DownstreamTarget::Kind kind) {
  switch (kind) {
  case DownstreamTarget::Kind::StageGroup:
    return "stage";
  case DownstreamTarget::Kind::PipelineInput:
    return "pipeline-input";
  case DownstreamTarget::Kind::GraphSink:
    return "sink";
  case DownstreamTarget::Kind::RealtimeLatestLink:
    return "realtime-link";
  }
  return "unknown";
}

const char* sample_kind_name(SampleKind kind) {
  switch (kind) {
  case SampleKind::Tensor:
    return "tensor";
  case SampleKind::TensorSet:
    return "tensor-set";
  case SampleKind::Bundle:
    return "bundle";
  case SampleKind::Unknown:
    return "unknown";
  }
  return "unknown";
}

const char* payload_type_name(PayloadType payload_type) {
  switch (payload_type) {
  case PayloadType::Auto:
    return "auto";
  case PayloadType::Image:
    return "image";
  case PayloadType::Tensor:
    return "tensor";
  case PayloadType::Encoded:
    return "encoded";
  }
  return "unknown";
}

struct SampleDebugCounts {
  int tensors = 0;
  int gst_holders = 0;
};

bool tensor_has_gstsample_holder(const simaai::neat::Tensor& tensor) {
  return tensor.storage && tensor.storage->kind == simaai::neat::StorageKind::GstSample &&
         tensor.storage->holder != nullptr;
}

bool sample_has_gstsample_holder(const simaai::neat::Sample& sample) {
  if (sample.tensor.has_value() && tensor_has_gstsample_holder(*sample.tensor)) {
    return true;
  }
  for (const auto& tensor : sample.tensors) {
    if (tensor_has_gstsample_holder(tensor)) {
      return true;
    }
  }
  if (sample.kind == simaai::neat::SampleKind::Bundle) {
    for (const auto& field : sample.fields) {
      if (sample_has_gstsample_holder(field)) {
        return true;
      }
    }
  }
  return false;
}

bool sample_looks_raw_video(const simaai::neat::Sample& sample) {
  if (sample.media_type == "video/x-raw" ||
      sample.payload_type == simaai::neat::PayloadType::Image) {
    return true;
  }
  const auto is_raw_tag = [](const std::string& tag) {
    return tag == "NV12" || tag == "RGB" || tag == "BGR" || tag == "I420" || tag == "YUV420P";
  };
  if (is_raw_tag(sample.payload_tag) || is_raw_tag(sample.format)) {
    return true;
  }
  if (sample.kind == simaai::neat::SampleKind::Bundle) {
    for (const auto& field : sample.fields) {
      if (sample_looks_raw_video(field)) {
        return true;
      }
    }
  }
  return false;
}

void collect_sample_debug_counts(const simaai::neat::Sample& sample, SampleDebugCounts* counts) {
  if (!counts) {
    return;
  }
  const auto add_tensor = [&](const simaai::neat::Tensor& tensor) {
    ++counts->tensors;
    if (tensor_has_gstsample_holder(tensor)) {
      ++counts->gst_holders;
    }
  };
  if (sample.tensor.has_value()) {
    add_tensor(*sample.tensor);
  }
  for (const auto& tensor : sample.tensors) {
    add_tensor(tensor);
  }
  for (const auto& field : sample.fields) {
    collect_sample_debug_counts(field, counts);
  }
}

void log_realtime_credit_probe(const char* event, const DownstreamTarget& downstream,
                               std::size_t edge_index, const std::string& key,
                               const simaai::neat::Sample& sample, bool raw_decoder_backed,
                               bool already_admitted, bool credit_applicable,
                               const char* note = nullptr) {
  if (!realtime_credit_probe_enabled()) {
    return;
  }
  static std::atomic<int> logs{0};
  const int seen = logs.fetch_add(1, std::memory_order_relaxed);
  const int limit = realtime_credit_probe_limit();
  const int every = realtime_credit_probe_every();
  if (seen >= limit && (every <= 0 || (seen % every) != 0)) {
    return;
  }
  SampleDebugCounts counts;
  collect_sample_debug_counts(sample, &counts);
  const auto credits = pipeline_internal::realtime_frame_credits_for_sample(sample);
  std::fprintf(stderr,
               "[GRAPH][credit-probe] event=%s target=%s/%zu edge=%zu key=%s "
               "kind=%s payload=%s media=%s tag=%s fmt=%s stream=%s label=%s "
               "frame=%lld input=%lld orig=%lld pts=%lld tensors=%d gst_holders=%d "
               "raw=%d admitted=%d applicable=%d credits=%zu note=%s\n",
               event ? event : "probe", target_kind_name(downstream.kind), downstream.index,
               edge_index, key.empty() ? "<empty>" : key.c_str(), sample_kind_name(sample.kind),
               payload_type_name(sample.payload_type),
               sample.media_type.empty() ? "<empty>" : sample.media_type.c_str(),
               sample.payload_tag.empty() ? "<empty>" : sample.payload_tag.c_str(),
               sample.format.empty() ? "<empty>" : sample.format.c_str(),
               sample.stream_id.empty() ? "<empty>" : sample.stream_id.c_str(),
               sample.stream_label.empty() ? "<empty>" : sample.stream_label.c_str(),
               static_cast<long long>(sample.frame_id), static_cast<long long>(sample.input_seq),
               static_cast<long long>(sample.orig_input_seq), static_cast<long long>(sample.pts_ns),
               counts.tensors, counts.gst_holders, raw_decoder_backed ? 1 : 0,
               already_admitted ? 1 : 0, credit_applicable ? 1 : 0, credits.size(),
               note ? note : "");
}

void apply_link_stream_id(const ExecutionGraphRuntime& runtime, std::size_t edge_index,
                          simaai::neat::Sample& sample) {
  if (edge_index == invalid_edge_index() || edge_index >= runtime.plan.edges.size()) {
    return;
  }
  const std::string& stream_id = runtime.plan.edges[edge_index].stream_id;
  if (stream_id.empty()) {
    return;
  }
  sample.stream_id = stream_id;
  if (sample.stream_label.empty()) {
    sample.stream_label = stream_id;
  }
}

} // namespace

RealtimeLatestLink::RealtimeLatestLink(DownstreamTarget downstream, GraphLinkOptions options,
                                       std::string stream_id)
    : downstream_(downstream), options_(options),
      credit_namespace_(pipeline_internal::next_realtime_frame_credit_namespace()),
      credit_limit_per_stream_(realtime_credit_max_inflight_per_stream(options_)),
      credit_limit_global_(0) {
  add_edge_stream_id(downstream_.edge_index, stream_id);
  log_realtime_credit_probe_basic("construct", downstream_, downstream_.edge_index, stream_id,
                                  static_cast<std::size_t>(options_.queue_depth));
  if (realtime_link_diag_enabled()) {
    std::fprintf(stderr,
                 "[GRAPH][credit-probe] link-created target=%s/%zu edge=%zu "
                 "stream=%s q=%d per_stream=%d global=%d env_probe=%s env_credit_debug=%s "
                 "env_link_debug=%s env_diag=%s\n",
                 target_kind_name(downstream_.kind), downstream_.index, downstream_.edge_index,
                 stream_id.empty() ? "<empty>" : stream_id.c_str(), options_.queue_depth,
                 credit_limit_per_stream_, credit_limit_global_,
                 std::getenv("SIMA_GRAPH_REALTIME_CREDIT_PROBE")
                     ? std::getenv("SIMA_GRAPH_REALTIME_CREDIT_PROBE")
                     : "<unset>",
                 std::getenv("SIMA_GRAPH_REALTIME_CREDIT_DEBUG")
                     ? std::getenv("SIMA_GRAPH_REALTIME_CREDIT_DEBUG")
                     : "<unset>",
                 std::getenv("SIMA_GRAPH_REALTIME_LINK_DEBUG")
                     ? std::getenv("SIMA_GRAPH_REALTIME_LINK_DEBUG")
                     : "<unset>",
                 std::getenv("SIMA_GRAPH_DIAG_ON_STOP") ? std::getenv("SIMA_GRAPH_DIAG_ON_STOP")
                                                        : "<unset>");
  }
}

RealtimeLatestLink::~RealtimeLatestLink() {
  close();
  join();
  pipeline_internal::release_all_registered_realtime_frame_credits(credit_namespace_,
                                                                   "graph-realtime-destroy");
}

std::string RealtimeLatestLink::key_for_(const simaai::neat::Sample& sample,
                                         std::size_t edge_index) const {
  if (!sample.stream_id.empty()) {
    return sample.stream_id;
  }
  return "edge:" + std::to_string(edge_index);
}

bool RealtimeLatestLink::offer(simaai::neat::Sample&& sample, std::size_t edge_index) {
  offered_.fetch_add(1, std::memory_order_relaxed);
  std::vector<pipeline_internal::RealtimeFrameCredit> credits_to_release;
  const char* release_mode = nullptr;
  bool accepted = false;
  std::string stream_id;
  {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = stream_id_by_edge_.find(edge_index);
    if (it != stream_id_by_edge_.end()) {
      stream_id = it->second;
    }
  }
  if (!stream_id.empty()) {
    sample.stream_id = stream_id;
    if (sample.stream_label.empty()) {
      sample.stream_label = stream_id;
    }
  }
  const std::string key = key_for_(sample, edge_index);
  log_realtime_credit_probe_basic("offer-enter", downstream_, edge_index, key);
  const auto offered_at = std::chrono::steady_clock::now();
  if (realtime_credit_probe_enabled()) {
    const bool raw_decoder_backed =
        pipeline_internal::sample_has_device_gstsample_holder(sample) ||
        (sample_looks_raw_video(sample) && sample_has_gstsample_holder(sample));
    const bool already_admitted =
        pipeline_internal::sample_has_attached_realtime_frame_credit(sample);
    log_realtime_credit_probe("offer", downstream_, edge_index, key, sample, raw_decoder_backed,
                              already_admitted, raw_decoder_backed && !already_admitted);
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_) {
      credits_to_release = pipeline_internal::realtime_frame_credits_for_sample(sample);
      release_mode = "graph-realtime-offer-closed";
    } else {
      Pending& pending = pending_[key];
      if (pending.has_sample) {
        if (realtime_overwrite_debug_enabled()) {
          std::fprintf(stderr,
                       "[GRAPH] realtime_link_overwrite key=%s old_stream=%s old_frame=%lld "
                       "new_stream=%s new_frame=%lld\n",
                       key.c_str(), pending.sample.stream_id.c_str(),
                       static_cast<long long>(pending.sample.frame_id), sample.stream_id.c_str(),
                       static_cast<long long>(sample.frame_id));
        }
        const auto overwritten_credits =
            pipeline_internal::realtime_frame_credits_for_sample(pending.sample);
        credits_to_release.insert(credits_to_release.end(), overwritten_credits.begin(),
                                  overwritten_credits.end());
        release_mode = "graph-realtime-overwrite";
        overwritten_.fetch_add(1, std::memory_order_relaxed);
      }
      pending.sample = std::move(sample);
      pending.ready_at = offered_at;
      pending.edge_index = edge_index;
      pending.has_sample = true;
      if (!pending.queued) {
        pending.queued = true;
        ready_.push_back(key);
      }
      accepted = true;
    }
  }
  if (!credits_to_release.empty()) {
    pipeline_internal::release_realtime_frame_credits_without_output(credits_to_release,
                                                                     release_mode);
  }
  if (!accepted) {
    return false;
  }
  cv_.notify_one();
  return true;
}

void RealtimeLatestLink::add_edge_stream_id(std::size_t edge_index, const std::string& stream_id) {
  if (edge_index == invalid_edge_index()) {
    return;
  }
  std::lock_guard<std::mutex> lock(mu_);
  edge_indices_.insert(edge_index);
  if (!stream_id.empty()) {
    stream_id_by_edge_[edge_index] = stream_id;
  }
  configure_global_credit_limit_locked_();
}

void RealtimeLatestLink::start(DispatchFn dispatch, StopFn stop, ErrorFn error) {
  dispatch_ = std::move(dispatch);
  stop_ = std::move(stop);
  error_ = std::move(error);
  worker_ = std::thread([this] { run_(); });
}

void RealtimeLatestLink::close() {
  std::vector<pipeline_internal::RealtimeFrameCredit> pending_credits;
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& item : pending_) {
      Pending& pending = item.second;
      if (!pending.has_sample) {
        continue;
      }
      const auto credits = pipeline_internal::realtime_frame_credits_for_sample(pending.sample);
      pending_credits.insert(pending_credits.end(), credits.begin(), credits.end());
      pending.has_sample = false;
      pending.queued = false;
    }
    ready_.clear();
    closed_ = true;
  }
  if (!pending_credits.empty()) {
    pipeline_internal::release_realtime_frame_credits_without_output(
        pending_credits, "graph-realtime-close-pending");
  }
  pipeline_internal::release_all_registered_realtime_frame_credits(credit_namespace_,
                                                                   "graph-realtime-close");
  cv_.notify_all();
}

void RealtimeLatestLink::join() {
  if (worker_.joinable()) {
    worker_.join();
  }
}

RealtimeLatestLink::Stats RealtimeLatestLink::stats() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::uint64_t credit_registered = 0;
  std::uint64_t credit_released_by_output = 0;
  std::uint64_t credit_released_without_output = 0;
  std::uint64_t credit_missing_key = 0;
  std::size_t credit_inflight = 0;
  std::size_t credit_limit = 0;
  const auto collect_lane = [&](const pipeline_internal::RealtimeFrameCreditLanePtr& lane) {
    if (!lane) {
      return;
    }
    credit_registered += lane->registered.load(std::memory_order_relaxed);
    credit_released_by_output += lane->released_by_output.load(std::memory_order_relaxed);
    credit_released_without_output += lane->released_without_output.load(std::memory_order_relaxed);
    credit_missing_key += lane->missing_key.load(std::memory_order_relaxed);
    if (lane->gate) {
      credit_inflight += static_cast<std::size_t>(std::max(0, lane->gate->inflight()));
      credit_limit += static_cast<std::size_t>(std::max(0, lane->gate->credit_limit()));
    }
  };
  for (const auto& item : credit_lanes_) {
    collect_lane(item.second);
  }
  collect_lane(global_credit_lane_);
  return Stats{.offered = offered_.load(std::memory_order_relaxed),
               .scheduled = scheduled_.load(std::memory_order_relaxed),
               .overwritten = overwritten_.load(std::memory_order_relaxed),
               .dispatch_failed = dispatch_failed_.load(std::memory_order_relaxed),
               .ready_wait_ns = ready_wait_ns_.load(std::memory_order_relaxed),
               .ready_wait_max_ns = ready_wait_max_ns_.load(std::memory_order_relaxed),
               .dispatch_ns = dispatch_ns_.load(std::memory_order_relaxed),
               .dispatch_max_ns = dispatch_max_ns_.load(std::memory_order_relaxed),
               .no_credit_skips = no_credit_skips_.load(std::memory_order_relaxed),
               .credit_registered = credit_registered,
               .credit_released_by_output = credit_released_by_output,
               .credit_released_without_output = credit_released_without_output,
               .credit_missing_key = credit_missing_key,
               .credit_inflight = credit_inflight,
               .credit_limit = credit_limit,
               .ready = ready_.size()};
}

std::string RealtimeLatestLink::debug_stream_ids() const {
  std::lock_guard<std::mutex> lock(mu_);
  if (stream_id_by_edge_.empty()) {
    return {};
  }
  std::vector<std::pair<std::size_t, std::string>> items(stream_id_by_edge_.begin(),
                                                         stream_id_by_edge_.end());
  std::sort(items.begin(), items.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  std::ostringstream oss;
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (i != 0U) {
      oss << ",";
    }
    oss << items[i].first << ":" << items[i].second;
  }
  return oss.str();
}

pipeline_internal::RealtimeFrameCreditLanePtr
RealtimeLatestLink::credit_lane_for_key_locked_(const std::string& key) {
  if (credit_limit_per_stream_ <= 0 || key.empty()) {
    return nullptr;
  }
  auto it = credit_lanes_.find(key);
  if (it != credit_lanes_.end()) {
    return it->second;
  }
  auto lane = pipeline_internal::make_realtime_frame_credit_lane(credit_limit_per_stream_,
                                                                 [this] { cv_.notify_one(); });
  credit_lanes_.emplace(key, lane);
  configure_global_credit_limit_locked_();
  return lane;
}

void RealtimeLatestLink::configure_global_credit_limit_locked_() {
  const std::size_t stream_count =
      std::max<std::size_t>(1U, std::max(edge_indices_.size(), credit_lanes_.size()));
  credit_limit_global_ =
      realtime_credit_max_inflight_total(options_, credit_limit_per_stream_, stream_count);
  if (credit_limit_global_ <= 0) {
    global_credit_lane_.reset();
    return;
  }
  if (!global_credit_lane_) {
    global_credit_lane_ = pipeline_internal::make_realtime_frame_credit_lane(
        credit_limit_global_, [this] { cv_.notify_one(); });
    return;
  }
  if (global_credit_lane_->gate) {
    global_credit_lane_->gate->configure(credit_limit_global_);
  }
}

void RealtimeLatestLink::run_() {
  while (true) {
    Sample sample;
    std::chrono::steady_clock::time_point ready_at;
    std::size_t edge_index = invalid_edge_index();
    pipeline_internal::RealtimeFrameCreditLanePtr credit_lane;
    pipeline_internal::RealtimeFrameCreditLanePtr global_credit_lane;
    pipeline_internal::RealtimeFrameCredit credit;
    bool credit_acquired = false;
    bool global_credit_acquired = false;
    bool credit_registered = false;
    bool credit_released_after_register = false;
    bool raw_decoder_backed_selected = false;
    bool already_admitted_selected = false;
    bool credit_applicable_selected = false;
    std::string selected_key;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [&] { return closed_ || (stop_ && stop_()) || !ready_.empty(); });
      if ((closed_ || (stop_ && stop_())) && ready_.empty()) {
        return;
      }

      bool selected = false;
      const std::size_t attempts = ready_.size();
      for (std::size_t attempt = 0; attempt < attempts; ++attempt) {
        std::string key = std::move(ready_.front());
        ready_.pop_front();
        auto it = pending_.find(key);
        if (it == pending_.end() || !it->second.has_sample) {
          if (it != pending_.end()) {
            it->second.queued = false;
          }
          continue;
        }
        Pending& pending = it->second;
        /*
         * The realtime credit is an admission guard for decoder-backed raw
         * frames at the first C++ realtime boundary after decode.  The scarce
         * resource is the CVU/EV74 GstSample holder, not only the model
         * PipelineInput queue: a stage inbox or graph sink can retain the same
         * decoder buffers long enough to exhaust a small native decoder pool.
         *
         * Once a raw frame is admitted, attach the private credit key to the
         * TensorBuffer sidecar.  Downstream realtime links see that marker and
         * do not acquire a second credit for the same frame; normal output,
         * drop, and close paths release the carried key.
         */
        const bool raw_decoder_backed =
            pipeline_internal::sample_has_device_gstsample_holder(pending.sample) ||
            (sample_looks_raw_video(pending.sample) && sample_has_gstsample_holder(pending.sample));
        const bool already_admitted =
            pipeline_internal::sample_has_attached_realtime_frame_credit(pending.sample);
        const bool credit_applicable = raw_decoder_backed && !already_admitted;
        log_realtime_credit_probe_basic("select-enter", downstream_, pending.edge_index, key,
                                        credit_applicable ? 1U : 0U);
        if (realtime_credit_probe_enabled()) {
          log_realtime_credit_probe("select", downstream_, pending.edge_index, key, pending.sample,
                                    raw_decoder_backed, already_admitted, credit_applicable);
        }
        credit_lane = credit_applicable ? credit_lane_for_key_locked_(key) : nullptr;
        global_credit_lane = credit_applicable ? global_credit_lane_ : nullptr;
        if (credit_applicable && global_credit_lane && global_credit_lane->gate &&
            global_credit_lane->gate->enabled() && !global_credit_lane->gate->try_acquire()) {
          if (realtime_credit_probe_enabled()) {
            log_realtime_credit_probe("gate-global-full", downstream_, pending.edge_index, key,
                                      pending.sample, raw_decoder_backed, already_admitted,
                                      credit_applicable);
          }
          no_credit_skips_.fetch_add(1, std::memory_order_relaxed);
          pending.queued = true;
          ready_.push_back(std::move(key));
          continue;
        }
        global_credit_acquired = credit_applicable && global_credit_lane &&
                                 global_credit_lane->gate && global_credit_lane->gate->enabled();
        if (credit_applicable && credit_lane && credit_lane->gate && credit_lane->gate->enabled() &&
            !credit_lane->gate->try_acquire()) {
          if (realtime_credit_probe_enabled()) {
            log_realtime_credit_probe("gate-stream-full", downstream_, pending.edge_index, key,
                                      pending.sample, raw_decoder_backed, already_admitted,
                                      credit_applicable);
          }
          if (global_credit_acquired && global_credit_lane && global_credit_lane->gate) {
            global_credit_lane->gate->release();
            global_credit_acquired = false;
          }
          no_credit_skips_.fetch_add(1, std::memory_order_relaxed);
          pending.queued = true;
          ready_.push_back(std::move(key));
          continue;
        }

        sample = std::move(pending.sample);
        ready_at = pending.ready_at;
        edge_index = pending.edge_index;
        selected_key = key;
        raw_decoder_backed_selected = raw_decoder_backed;
        already_admitted_selected = already_admitted;
        credit_applicable_selected = credit_applicable;
        pending.has_sample = false;
        pending.queued = false;
        credit_acquired =
            credit_applicable && credit_lane && credit_lane->gate && credit_lane->gate->enabled();
        selected = true;
        break;
      }

      if (!selected) {
        cv_.wait_for(lock, std::chrono::milliseconds(5));
        continue;
      }
    }
    if (ready_at.time_since_epoch().count() != 0) {
      atomic_add_max(ready_wait_ns_, ready_wait_max_ns_, elapsed_ns_since(ready_at));
    }

    if (credit_acquired) {
      const bool credit_key_available = !sample.stream_id.empty() && sample.frame_id >= 0;
      if (credit_key_available) {
        credit = pipeline_internal::RealtimeFrameCredit{credit_namespace_, sample.stream_id,
                                                        sample.frame_id};
        std::vector<pipeline_internal::RealtimeFrameCreditLanePtr> companions;
        if (global_credit_acquired && global_credit_lane) {
          companions.push_back(global_credit_lane);
        }
        credit_registered = pipeline_internal::register_realtime_frame_credit(
            credit.namespace_id, credit.stream_id, credit.frame_id, credit_lane, companions);
        if (credit_registered) {
          pipeline_internal::attach_realtime_frame_credit_to_sample(sample, credit);
          if (!pipeline_internal::sample_has_attached_realtime_frame_credit(sample)) {
            if (realtime_credit_probe_enabled()) {
              log_realtime_credit_probe("attach-failed", downstream_, edge_index, selected_key,
                                        sample, raw_decoder_backed_selected,
                                        already_admitted_selected, credit_applicable_selected);
            }
            (void)pipeline_internal::release_registered_realtime_frame_credit(
                credit, "graph-realtime-credit-attach-failed", /*by_output=*/false);
            credit_released_after_register = true;
            credit_registered = false;
          }
          if (credit_registered && realtime_credit_probe_enabled()) {
            log_realtime_credit_probe("registered", downstream_, edge_index, selected_key, sample,
                                      raw_decoder_backed_selected, already_admitted_selected,
                                      credit_applicable_selected);
          }
        }
      }
      if (!credit_registered) {
        if (realtime_credit_probe_enabled()) {
          log_realtime_credit_probe(credit_key_available ? "register-failed" : "missing-key",
                                    downstream_, edge_index, selected_key, sample,
                                    raw_decoder_backed_selected, already_admitted_selected,
                                    credit_applicable_selected);
        }
        if (!credit_released_after_register) {
          if (credit_lane) {
            if (!credit_key_available) {
              credit_lane->missing_key.fetch_add(1, std::memory_order_relaxed);
            }
            if (credit_lane->gate) {
              credit_lane->gate->release();
            }
          }
          if (global_credit_acquired && global_credit_lane && global_credit_lane->gate) {
            global_credit_lane->gate->release();
          }
        }
      }
    }
    if (credit_applicable_selected && !credit_registered && realtime_credit_probe_enabled()) {
      log_realtime_credit_probe("dispatch-uncredited-raw", downstream_, edge_index, selected_key,
                                sample, raw_decoder_backed_selected, already_admitted_selected,
                                credit_applicable_selected);
    }

    if (!dispatch_) {
      if (credit_registered) {
        (void)pipeline_internal::release_registered_realtime_frame_credit(
            credit, "graph-realtime-missing-dispatch", /*by_output=*/false);
      }
      dispatch_failed_.fetch_add(1, std::memory_order_relaxed);
      if (error_) {
        error_("RealtimeLatestLink: missing dispatch callback");
      }
      return;
    }
    const auto dispatch_start = std::chrono::steady_clock::now();
    if (!dispatch_(downstream_, std::move(sample), edge_index)) {
      atomic_add_max(dispatch_ns_, dispatch_max_ns_, elapsed_ns_since(dispatch_start));
      if (credit_registered) {
        (void)pipeline_internal::release_registered_realtime_frame_credit(
            credit, "graph-realtime-dispatch-failed", /*by_output=*/false);
      }
      dispatch_failed_.fetch_add(1, std::memory_order_relaxed);
      if (stop_ && stop_()) {
        return;
      }
      if (error_) {
        error_("RealtimeLatestLink: downstream dispatch failed");
      }
      return;
    }
    atomic_add_max(dispatch_ns_, dispatch_max_ns_, elapsed_ns_since(dispatch_start));
    const std::uint64_t scheduled = scheduled_.fetch_add(1, std::memory_order_relaxed) + 1U;
    const int log_every = realtime_link_log_every();
    if (log_every > 0 && scheduled % static_cast<std::uint64_t>(log_every) == 0U &&
        realtime_link_diag_enabled()) {
      const Stats s = stats();
      const std::string stream_ids = debug_stream_ids();
      const std::uint64_t dispatched = s.scheduled + s.dispatch_failed;
      const auto avg_ms = [](std::uint64_t total_ns, std::uint64_t count) {
        return count == 0 ? 0.0
                          : static_cast<double>(total_ns) / static_cast<double>(count) / 1.0e6;
      };
      std::fprintf(
          stderr,
          "[GRAPH] realtime_link_progress downstream_kind=%d downstream_index=%zu "
          "streams=%s offered=%llu scheduled=%llu overwritten=%llu ready=%zu "
          "avg_ready_wait_ms=%.3f max_ready_wait_ms=%.3f avg_dispatch_ms=%.3f "
          "max_dispatch_ms=%.3f no_credit_skips=%llu credit_inflight=%zu "
          "credit_limit=%zu credit_registered=%llu credit_released=%llu/%llu "
          "credit_missing_key=%llu\n",
          static_cast<int>(downstream_.kind), downstream_.index,
          stream_ids.empty() ? "<none>" : stream_ids.c_str(),
          static_cast<unsigned long long>(s.offered), static_cast<unsigned long long>(s.scheduled),
          static_cast<unsigned long long>(s.overwritten), s.ready,
          avg_ms(s.ready_wait_ns, dispatched), static_cast<double>(s.ready_wait_max_ns) / 1.0e6,
          avg_ms(s.dispatch_ns, dispatched), static_cast<double>(s.dispatch_max_ns) / 1.0e6,
          static_cast<unsigned long long>(s.no_credit_skips), s.credit_inflight, s.credit_limit,
          static_cast<unsigned long long>(s.credit_registered),
          static_cast<unsigned long long>(s.credit_released_by_output),
          static_cast<unsigned long long>(s.credit_released_without_output),
          static_cast<unsigned long long>(s.credit_missing_key));
    }
  }
}

const std::vector<DownstreamTarget>* EdgeRouter::targets(simaai::neat::graph::NodeId node,
                                                         simaai::neat::graph::PortId port) const {
  if (!runtime_)
    return nullptr;
  const auto it = runtime_->adjacency.find(port_key(node, port));
  if (it == runtime_->adjacency.end() || it->second.empty())
    return nullptr;
  return &it->second;
}

bool EdgeRouter::push_to_sink(simaai::neat::graph::NodeId sink_node, Sample&& sample,
                              std::size_t edge_index, const EdgeRouterOptions& options,
                              const EdgeRouterCallbacks& callbacks,
                              std::size_t sink_backpressure_context) const {
  if (!runtime_) {
    request_stop(callbacks, "EdgeRouter: missing runtime state");
    return false;
  }

  auto sink_it = runtime_->sinks.find(sink_node);
  const auto realtime_credits = pipeline_internal::realtime_frame_credits_for_sample(sample);
  if (sink_it == runtime_->sinks.end() || !sink_it->second) {
    pipeline_internal::release_realtime_frame_credits(realtime_credits, "graph-sink-discard");
    return true;
  }

  const std::size_t qsize = sink_it->second->size();
  if (callbacks.prepare_sink_sample) {
    callbacks.prepare_sink_sample(sink_node, sample, qsize, sink_backpressure_context);
  }

  const bool trace = graph_message_trace_enabled(runtime_) && edge_index != invalid_edge_index();
  TraceGraphMessageArgs trace_args;
  if (trace) {
    trace_args = make_trace_graph_message_args(runtime_, edge_index, sample);
    trace_graph_message_event(TraceGraphMessageEventType::EdgeSrcPush, trace_args);
  }

  if (!sink_it->second->push(RuntimeSinkQueueMsg{std::move(sample), edge_index},
                             options.push_timeout_ms)) {
    if (trace) {
      trace_graph_message_event(TraceGraphMessageEventType::Drop, trace_args);
    }
    pipeline_internal::release_realtime_frame_credits(realtime_credits, "graph-sink-drop");
    if (!stop_requested(callbacks)) {
      std::ostringstream msg;
      msg << "GraphRun: sink backpressure timeout (node=" << static_cast<std::size_t>(sink_node)
          << ", edge_queue=" << options.edge_queue
          << ", push_timeout_ms=" << options.push_timeout_ms << ")."
          << graph_backpressure_timeout_explanation();
      request_stop(callbacks, msg.str());
    }
    return false;
  }
  if (trace) {
    trace_graph_message_event(TraceGraphMessageEventType::QueueIn, trace_args);
  }
  return true;
}

bool EdgeRouter::dispatch_to_target(const DownstreamTarget& target, Sample&& sample,
                                    const EdgeRouterOptions& options,
                                    const EdgeRouterCallbacks& callbacks,
                                    const EdgeRouterDispatchOptions& dispatch_options) const {
  if (!runtime_) {
    request_stop(callbacks, "EdgeRouter: missing runtime state");
    return false;
  }

  apply_link_stream_id(*runtime_, target.edge_index, sample);
  if (realtime_credit_probe_enabled()) {
    const bool raw_decoder_backed =
        pipeline_internal::sample_has_device_gstsample_holder(sample) ||
        (sample_looks_raw_video(sample) && sample_has_gstsample_holder(sample));
    const bool already_admitted =
        pipeline_internal::sample_has_attached_realtime_frame_credit(sample);
    if (raw_decoder_backed || already_admitted) {
      log_realtime_credit_probe("dispatch-target", target, target.edge_index,
                                target.edge_index == invalid_edge_index()
                                    ? std::string{}
                                    : std::string{"edge:"} + std::to_string(target.edge_index),
                                sample, raw_decoder_backed, already_admitted,
                                raw_decoder_backed && !already_admitted);
    }
  }

  if (target.kind == DownstreamTarget::Kind::RealtimeLatestLink) {
    const auto realtime_credits = pipeline_internal::realtime_frame_credits_for_sample(sample);
    if (target.index >= runtime_->realtime_links.size() ||
        !runtime_->realtime_links[target.index]) {
      pipeline_internal::release_realtime_frame_credits(realtime_credits,
                                                        "realtime-link-target-missing");
      request_stop(callbacks, "EdgeRouter: realtime link target out of range");
      return false;
    }
    return runtime_->realtime_links[target.index]->offer(std::move(sample), target.edge_index);
  }

  if (target.kind == DownstreamTarget::Kind::StageGroup) {
    const auto realtime_credits = pipeline_internal::realtime_frame_credits_for_sample(sample);
    if (!callbacks.dispatch_to_stage_group) {
      pipeline_internal::release_realtime_frame_credits(realtime_credits, "stage-dispatch-missing");
      request_stop(callbacks, "EdgeRouter: missing stage dispatch callback");
      return false;
    }
    const bool trace =
        graph_message_trace_enabled(runtime_) && target.edge_index != invalid_edge_index();
    TraceGraphMessageArgs trace_args;
    if (trace) {
      trace_args = make_trace_graph_message_args(runtime_, target.edge_index, sample);
      trace_graph_message_event(TraceGraphMessageEventType::EdgeSrcPush, trace_args);
    }
    const bool ok = callbacks.dispatch_to_stage_group(target.index, target.port, std::move(sample),
                                                      target.edge_index);
    if (trace) {
      trace_graph_message_event(
          ok ? TraceGraphMessageEventType::QueueIn : TraceGraphMessageEventType::Drop, trace_args);
    }
    if (!ok) {
      pipeline_internal::release_realtime_frame_credits(realtime_credits, "stage-dispatch-drop");
    }
    return ok;
  }

  if (target.kind == DownstreamTarget::Kind::PipelineInput) {
    const auto realtime_credits = pipeline_internal::realtime_frame_credits_for_sample(sample);
    if (target.index >= runtime_->pipelines.size() || !runtime_->pipelines[target.index]) {
      pipeline_internal::release_realtime_frame_credits(realtime_credits,
                                                        "pipeline-input-target-missing");
      std::ostringstream msg;
      msg << "GraphRun: pipeline input target out of range (index=" << target.index << ")";
      request_stop(callbacks, msg.str());
      return false;
    }

    if (!callbacks.ensure_pipeline_built) {
      pipeline_internal::release_realtime_frame_credits(realtime_credits,
                                                        "pipeline-input-build-missing");
      request_stop(callbacks, "EdgeRouter: missing pipeline build callback");
      return false;
    }

    auto& pipe = *runtime_->pipelines[target.index];
    auto& telemetry = pipe.transport.telemetry;
    std::string build_err;
    const auto ensure_start = std::chrono::steady_clock::now();
    telemetry.router_ensure_build_calls.fetch_add(1, std::memory_order_relaxed);
    if (!callbacks.ensure_pipeline_built(target.index, sample, &build_err)) {
      atomic_add_max(telemetry.router_ensure_build_ns, telemetry.router_ensure_build_max_ns,
                     elapsed_ns_since(ensure_start));
      pipeline_internal::release_realtime_frame_credits(realtime_credits,
                                                        "pipeline-input-build-failed");
      request_stop(callbacks, build_err.empty() ? "GraphRun: pipeline build failed" : build_err);
      return false;
    }
    atomic_add_max(telemetry.router_ensure_build_ns, telemetry.router_ensure_build_max_ns,
                   elapsed_ns_since(ensure_start));

    bool sanitized_before_enqueue = false;
    if (dispatch_options.sanitize_pipeline_input_before_enqueue &&
        callbacks.sanitize_pipeline_input) {
      const auto sanitize_start = std::chrono::steady_clock::now();
      callbacks.sanitize_pipeline_input(target.index, sample);
      (void)pipeline_internal::alias_registered_realtime_frame_credits(realtime_credits, sample,
                                                                       "pipeline-input-sanitize");
      sanitized_before_enqueue = true;
      telemetry.router_sanitize_calls.fetch_add(1, std::memory_order_relaxed);
      atomic_add_max(telemetry.router_sanitize_ns, telemetry.router_sanitize_max_ns,
                     elapsed_ns_since(sanitize_start));
    }

    auto& input_queue = pipe.transport.input_queue;
    if (!input_queue) {
      pipeline_internal::release_realtime_frame_credits(realtime_credits,
                                                        "pipeline-input-no-queue");
      return true;
    }

    const auto push_start = std::chrono::steady_clock::now();
    telemetry.router_input_push_calls.fetch_add(1, std::memory_order_relaxed);
    const bool trace =
        graph_message_trace_enabled(runtime_) && target.edge_index != invalid_edge_index();
    TraceGraphMessageArgs trace_args;
    if (trace) {
      trace_args = make_trace_graph_message_args(runtime_, target.edge_index, sample);
      trace_graph_message_event(TraceGraphMessageEventType::EdgeSrcPush, trace_args);
    }
    const bool pushed =
        dispatch_options.drop_pipeline_input_when_full
            ? input_queue->try_push(RuntimePipelineQueueMsg{std::move(sample), target.edge_index,
                                                            sanitized_before_enqueue})
            : input_queue->push(RuntimePipelineQueueMsg{std::move(sample), target.edge_index,
                                                        sanitized_before_enqueue},
                                options.push_timeout_ms);
    if (!pushed) {
      if (trace) {
        trace_graph_message_event(TraceGraphMessageEventType::Drop, trace_args);
      }
      pipeline_internal::release_realtime_frame_credits(realtime_credits, "pipeline-input-drop");
      atomic_add_max(telemetry.router_input_push_ns, telemetry.router_input_push_max_ns,
                     elapsed_ns_since(push_start));
      if (dispatch_options.drop_pipeline_input_when_full) {
        return true;
      }
      if (!stop_requested(callbacks)) {
        std::ostringstream msg;
        msg << "GraphRun: pipeline input backpressure timeout (seg="
            << static_cast<std::size_t>(runtime_->pipelines[target.index]->seg.id)
            << ", edge_queue=" << options.edge_queue
            << ", push_timeout_ms=" << options.push_timeout_ms << ")."
            << graph_backpressure_timeout_explanation();
        request_stop(callbacks, msg.str());
      }
      return false;
    }
    atomic_add_max(telemetry.router_input_push_ns, telemetry.router_input_push_max_ns,
                   elapsed_ns_since(push_start));
    if (trace) {
      trace_graph_message_event(TraceGraphMessageEventType::QueueIn, trace_args);
    }
    return true;
  }

  const auto sink_node = static_cast<simaai::neat::graph::NodeId>(target.index);
  return push_to_sink(sink_node, std::move(sample), target.edge_index, options, callbacks,
                      dispatch_options.sink_backpressure_context.value_or(target.index));
}

bool EdgeRouter::dispatch_to_targets(const std::vector<DownstreamTarget>& targets, Sample&& sample,
                                     const EdgeRouterOptions& options,
                                     const EdgeRouterCallbacks& callbacks,
                                     const EdgeRouterDispatchOptions& dispatch_options) const {
  if (targets.empty())
    return false;
  if (targets.size() == 1) {
    return dispatch_to_target(targets.front(), std::move(sample), options, callbacks,
                              dispatch_options);
  }

  bool ok = true;
  for (const auto& target : targets) {
    Sample sample_copy = sample;
    ok = dispatch_to_target(target, std::move(sample_copy), options, callbacks, dispatch_options) &&
         ok;
  }
  return ok;
}

} // namespace simaai::neat::runtime

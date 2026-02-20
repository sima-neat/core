#pragma once

#include "internal/InputStream.h"
#include "pipeline/Run.h"

#include <opencv2/core/mat.hpp>

#include <atomic>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

namespace simaai::neat {

enum class InputKind {
  Mat,
  Tensor,
  Holder,
  Message,
};

struct InputItem {
  InputKind kind = InputKind::Mat;
  cv::Mat mat;
  simaai::neat::Tensor tensor;
  std::shared_ptr<void> holder;
  Sample msg;
};

struct Run::State {
  std::uint64_t latency_count = 0;
  double latency_mean_ms = 0.0;
  double latency_min_ms = 0.0;
  double latency_max_ms = 0.0;
  std::thread input_thread;
  std::atomic<std::uint64_t> inputs_enqueued{0};
  std::atomic<std::uint64_t> inputs_dropped{0};
  std::atomic<std::uint64_t> inputs_pushed{0};
  std::atomic<std::uint64_t> outputs_ready{0};
  std::atomic<std::uint64_t> outputs_pulled{0};
  std::atomic<std::uint64_t> outputs_dropped{0};
  InputStream stream;
  std::string error;
  std::string diag_sysinfo;

  std::mutex in_mu;
  std::condition_variable in_cv;
  std::mutex out_mu;
  std::condition_variable out_cv;
  std::mutex latency_mu;
  mutable std::mutex error_mu;
  RunOptions opt;
  std::deque<InputItem> in_queue;
  std::deque<Sample> out_queue;
  std::deque<std::chrono::steady_clock::time_point> pending_times;
  InputStreamOptions stream_opt;
  std::atomic<int> handle_refs{0};
  bool supports_push = false;
  bool supports_pull = false;
  bool zero_copy_fallback_enabled = false;
  std::atomic<bool> copy_output_latched{true};
  std::atomic<bool> zero_copy_warned{false};
  bool input_closed = false;
  bool latency_init = false;
  std::atomic<bool> stop_requested{false};
  std::atomic<bool> input_thread_done{false};
  bool diag_enabled = false;
  std::atomic<bool> diag_logged{false};
  std::atomic<bool> pull_timeout_logged{false};
};

namespace run_internal {

template <typename T> inline bool queue_full(const std::deque<T>& q, int max) {
  return max > 0 && static_cast<int>(q.size()) >= max;
}

inline int parse_num_buffers_for(const std::string& pipeline, const std::string& plugin) {
  const std::string key = "num-buffers=";
  size_t pos = 0;
  while ((pos = pipeline.find(plugin, pos)) != std::string::npos) {
    size_t nb = pipeline.find(key, pos);
    if (nb == std::string::npos) {
      pos += plugin.size();
      continue;
    }
    nb += key.size();
    size_t end = nb;
    while (end < pipeline.size() && std::isdigit(static_cast<unsigned char>(pipeline[end]))) {
      ++end;
    }
    if (end > nb) {
      return std::atoi(pipeline.substr(nb, end - nb).c_str());
    }
    pos = end;
  }
  return 0;
}

inline int parse_queue2_depth(const std::string& pipeline) {
  const std::string key = "queue2";
  const std::string depth_key = "max-size-buffers=";
  size_t pos = 0;
  while ((pos = pipeline.find(key, pos)) != std::string::npos) {
    size_t depth_pos = pipeline.find(depth_key, pos);
    if (depth_pos == std::string::npos) {
      pos += key.size();
      continue;
    }
    depth_pos += depth_key.size();
    size_t end = depth_pos;
    while (end < pipeline.size() && std::isdigit(static_cast<unsigned char>(pipeline[end]))) {
      ++end;
    }
    if (end > depth_pos) {
      return std::atoi(pipeline.substr(depth_pos, end - depth_pos).c_str());
    }
    pos = end;
  }
  return 0;
}

inline bool tensor_is_zero_copy(const simaai::neat::Tensor& t) {
  return t.storage && t.storage->kind == simaai::neat::StorageKind::GstSample;
}

inline int zero_copy_backpressure_cap() {
  const char* env = std::getenv("SIMA_GRAPH_ZERO_COPY_BACKPRESSURE_CAP");
  if (env && *env)
    return std::max(0, std::atoi(env));
  env = std::getenv("SIMA_GRAPH_ZERO_COPY_MAX_INFLIGHT");
  if (env && *env)
    return std::max(0, std::atoi(env));
  return 1;
}

inline void force_copy_tensor_if_zero_copy(simaai::neat::Tensor& t) {
  if (!tensor_is_zero_copy(t))
    return;
  t = t.clone();
  t.read_only = false;
}

inline void force_copy_sample_if_zero_copy(Sample& sample) {
  if (sample.kind == SampleKind::Tensor && sample.tensor.has_value()) {
    force_copy_tensor_if_zero_copy(*sample.tensor);
    sample.owned = true;
    return;
  }
  if (sample.kind != SampleKind::Bundle)
    return;
  for (auto& field : sample.fields) {
    if (field.kind == SampleKind::Tensor && field.tensor.has_value()) {
      force_copy_tensor_if_zero_copy(*field.tensor);
      field.owned = true;
    }
  }
  sample.owned = true;
}

inline bool sample_has_zero_copy_tensor(const Sample& sample) {
  if (sample.kind == SampleKind::Tensor && sample.tensor.has_value()) {
    return tensor_is_zero_copy(*sample.tensor);
  }
  if (sample.kind != SampleKind::Bundle)
    return false;
  for (const auto& field : sample.fields) {
    if (field.kind == SampleKind::Tensor && field.tensor.has_value()) {
      if (tensor_is_zero_copy(*field.tensor))
        return true;
    }
  }
  return false;
}

inline void maybe_force_copy_for_backpressure(Sample& sample, std::size_t qsize, const char* where,
                                              bool debug_enabled) {
  if (!sample_has_zero_copy_tensor(sample))
    return;
  const int cap = zero_copy_backpressure_cap();
  if (cap <= 0 || qsize < static_cast<std::size_t>(cap))
    return;
  force_copy_sample_if_zero_copy(sample);
  if (debug_enabled) {
    std::fprintf(stderr,
                 "[PIPELINE] zero_copy_backpressure where=%s qsize=%zu cap=%d stream_id=%s\n",
                 where ? where : "queue", qsize, cap, sample.stream_id.c_str());
  }
}

inline void maybe_force_copy_for_backpressure(simaai::neat::Tensor& tensor, std::size_t qsize,
                                              const char* where, bool debug_enabled) {
  if (!tensor_is_zero_copy(tensor))
    return;
  const int cap = zero_copy_backpressure_cap();
  if (cap <= 0 || qsize < static_cast<std::size_t>(cap))
    return;
  force_copy_tensor_if_zero_copy(tensor);
  if (debug_enabled) {
    std::fprintf(stderr, "[PIPELINE] zero_copy_backpressure where=%s qsize=%zu cap=%d\n",
                 where ? where : "queue", qsize, cap);
  }
}

} // namespace run_internal

} // namespace simaai::neat

#pragma once

#include "RunCore.h"

namespace simaai::neat {

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
  // Tensor creators can now place input directly in device-visible GstSample
  // storage.  Do not silently clone those tensors back to CPU under queue
  // pressure; lifetime must be owned by the caller/ring instead.
  return 0;
}

inline void force_copy_tensor_if_zero_copy(simaai::neat::Tensor& t) {
  if (!tensor_is_zero_copy(t))
    return;
  t = t.clone();
  t.read_only = false;
}

inline void force_copy_sample_if_zero_copy(Sample& sample) {
  if (sample.kind == SampleKind::TensorSet) {
    for (auto& tensor : sample.tensors) {
      force_copy_tensor_if_zero_copy(tensor);
    }
    sample.owned = true;
    return;
  }
  if (!sample_is_multi_output(sample))
    return;
  for (auto& field : sample.fields) {
    force_copy_sample_if_zero_copy(field);
  }
  sample.owned = true;
}

inline bool sample_has_zero_copy_tensor(const Sample& sample) {
  if (sample.kind == SampleKind::TensorSet) {
    for (const auto& tensor : sample.tensors) {
      if (tensor_is_zero_copy(tensor))
        return true;
    }
    return false;
  }
  if (!sample_is_multi_output(sample))
    return false;
  for (const auto& field : sample.fields) {
    if (sample_has_zero_copy_tensor(field))
      return true;
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

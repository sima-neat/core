#include "RunCore.h"

#include <algorithm>

namespace simaai::neat::runtime {
namespace {

std::string normalized_stream_id(const Sample& sample) {
  return sample.stream_id.empty() ? std::string("default") : sample.stream_id;
}

std::optional<GraphSampleIdentityKey> make_graph_sample_identity_key(const Sample& sample) {
  GraphSampleIdentityKey key;
  key.stream_id = normalized_stream_id(sample);
  if (sample.orig_input_seq >= 0) {
    key.kind = GraphSampleTimingKeyKind::OrigInputSeq;
    key.value = sample.orig_input_seq;
    return key;
  }
  if (sample.input_seq >= 0) {
    key.kind = GraphSampleTimingKeyKind::InputSeq;
    key.value = sample.input_seq;
    return key;
  }
  if (sample.frame_id >= 0) {
    key.kind = GraphSampleTimingKeyKind::FrameId;
    key.value = sample.frame_id;
    return key;
  }
  return std::nullopt;
}

GraphSampleTimingEvent make_timing_event(std::string_view endpoint,
                                         const GraphSampleIdentityKey& key,
                                         std::chrono::steady_clock::time_point at,
                                         std::chrono::steady_clock::time_point start) {
  GraphSampleTimingEvent ev;
  ev.endpoint = std::string(endpoint);
  ev.stream_id = key.stream_id;
  ev.key_kind = key.kind;
  ev.key_value = key.value;
  ev.timestamp_s = start.time_since_epoch().count() == 0
                       ? 0.0
                       : std::chrono::duration<double>(at - start).count();
  return ev;
}

} // namespace

void RunCore::record_graph_sample_entry(std::string_view endpoint, const Sample& sample,
                                        std::chrono::steady_clock::time_point at) {
  const auto key = make_graph_sample_identity_key(sample);
  if (!key.has_value()) {
    graph_sample_timing_unkeyed.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  {
    std::lock_guard<std::mutex> lock(graph_sample_timing_mu);
    const std::uint64_t gen = ++graph_sample_timing_generation;
    auto [it, inserted] = graph_sample_timing_by_key.emplace(
        *key, GraphSampleTimingState{at, std::string(endpoint), gen, false});
    if (!inserted) {
      it->second.ambiguous = true;
    } else {
      graph_sample_timing_order.push_back(GraphSampleTimingOrderEntry{*key, gen});
    }
    while (graph_sample_timing_order.size() > graph_sample_timing_capacity) {
      const GraphSampleTimingOrderEntry old = graph_sample_timing_order.front();
      graph_sample_timing_order.pop_front();
      auto old_it = graph_sample_timing_by_key.find(old.key);
      if (old_it != graph_sample_timing_by_key.end() &&
          old_it->second.generation == old.generation) {
        graph_sample_timing_by_key.erase(old_it);
      }
    }
  }

  std::lock_guard<std::mutex> lock(latency_mu);
  if (measurement_active) {
    measurement_graph_entries.push_back(
        make_timing_event(endpoint, *key, at, measurement_started_at));
  }
}

void RunCore::record_graph_sample_output(std::string_view endpoint, const Sample& sample,
                                         std::chrono::steady_clock::time_point at) {
  std::optional<GraphSampleTimingState> state;
  const auto key = make_graph_sample_identity_key(sample);
  if (!key.has_value()) {
    graph_sample_timing_misses.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  {
    std::lock_guard<std::mutex> lock(graph_sample_timing_mu);
    auto it = graph_sample_timing_by_key.find(*key);
    if (it != graph_sample_timing_by_key.end()) {
      if (it->second.ambiguous) {
        graph_sample_timing_by_key.erase(it);
        graph_sample_timing_misses.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      state = it->second;
      graph_sample_timing_by_key.erase(it);
    }
  }
  if (!state.has_value()) {
    graph_sample_timing_misses.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  const double ms = std::chrono::duration<double, std::milli>(at - state->graph_entry_at).count();
  std::lock_guard<std::mutex> lock(latency_mu);
  ++latency_count;
  if (!latency_init) {
    latency_mean_ms = latency_min_ms = latency_max_ms = ms;
    latency_init = true;
  } else {
    latency_mean_ms += (ms - latency_mean_ms) / static_cast<double>(latency_count);
    latency_min_ms = std::min(latency_min_ms, ms);
    latency_max_ms = std::max(latency_max_ms, ms);
  }
  if (measurement_active) {
    measurement_latencies_ms.push_back(ms);
    measurement_graph_pulls.push_back(
        make_timing_event(endpoint, *key, at, measurement_started_at));
  }
}

} // namespace simaai::neat::runtime

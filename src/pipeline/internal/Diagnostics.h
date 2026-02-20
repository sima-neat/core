#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/SessionReport.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <glib.h>

namespace simaai::neat::pipeline_internal {

struct BoundaryFlowCounters {
  std::string boundary_name;
  int after_node_index = -1;
  int before_node_index = -1;

  std::atomic<uint64_t> in_buffers{0}, out_buffers{0};
  std::atomic<int64_t> last_in_pts_ns{-1}, last_out_pts_ns{-1};
  std::atomic<int64_t> last_in_wall_us{0}, last_out_wall_us{0};

  BoundaryFlowStats snapshot() const {
    BoundaryFlowStats s;
    s.boundary_name = boundary_name;
    s.after_node_index = after_node_index;
    s.before_node_index = before_node_index;

    s.in_buffers = in_buffers.load(std::memory_order_relaxed);
    s.out_buffers = out_buffers.load(std::memory_order_relaxed);
    s.last_in_pts_ns = last_in_pts_ns.load(std::memory_order_relaxed);
    s.last_out_pts_ns = last_out_pts_ns.load(std::memory_order_relaxed);
    s.last_in_wall_us = last_in_wall_us.load(std::memory_order_relaxed);
    s.last_out_wall_us = last_out_wall_us.load(std::memory_order_relaxed);
    return s;
  }
};

struct StageTimingStats {
  std::string stage_name;
  uint64_t samples = 0;
  uint64_t total_us = 0;
  uint64_t max_us = 0;
};

struct StageTimingCounters {
  std::string stage_name;
  std::atomic<uint64_t> samples{0};
  std::atomic<uint64_t> total_us{0};
  std::atomic<uint64_t> max_us{0};

  StageTimingStats snapshot() const {
    StageTimingStats s;
    s.stage_name = stage_name;
    s.samples = samples.load(std::memory_order_relaxed);
    s.total_us = total_us.load(std::memory_order_relaxed);
    s.max_us = max_us.load(std::memory_order_relaxed);
    return s;
  }
};

struct ElementTimingStats {
  std::string element_name;
  uint64_t samples = 0;
  uint64_t total_us = 0;
  uint64_t max_us = 0;
  uint64_t min_us = 0;
  uint64_t missed_in = 0;
  uint64_t missed_out = 0;
};

struct ElementFlowStats {
  std::string element_name;
  uint64_t in_buffers = 0;
  uint64_t out_buffers = 0;
  uint64_t in_bytes = 0;
  uint64_t out_bytes = 0;
  uint64_t caps_changes = 0;
};

struct ElementTimingKey {
  int64_t frame_id = -1;
  uint32_t stream_hash = 0;
};

struct ElementTimingKeyHash {
  size_t operator()(const ElementTimingKey& k) const {
    const size_t h1 = std::hash<int64_t>{}(k.frame_id);
    const size_t h2 = std::hash<uint32_t>{}(k.stream_hash);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

struct ElementTimingKeyEq {
  bool operator()(const ElementTimingKey& a, const ElementTimingKey& b) const {
    return a.frame_id == b.frame_id && a.stream_hash == b.stream_hash;
  }
};

struct ElementTimingCounters {
  std::string element_name;
  std::atomic<uint64_t> samples{0};
  std::atomic<uint64_t> total_us{0};
  std::atomic<uint64_t> max_us{0};
  std::atomic<uint64_t> min_us{0};
  std::atomic<uint64_t> missed_in{0};
  std::atomic<uint64_t> missed_out{0};
  std::mutex pending_mu;
  std::unordered_map<ElementTimingKey, int64_t, ElementTimingKeyHash, ElementTimingKeyEq> pending;
  size_t max_pending = 1024;

  ElementTimingStats snapshot() const {
    ElementTimingStats s;
    s.element_name = element_name;
    s.samples = samples.load(std::memory_order_relaxed);
    s.total_us = total_us.load(std::memory_order_relaxed);
    s.max_us = max_us.load(std::memory_order_relaxed);
    s.min_us = min_us.load(std::memory_order_relaxed);
    s.missed_in = missed_in.load(std::memory_order_relaxed);
    s.missed_out = missed_out.load(std::memory_order_relaxed);
    return s;
  }
};

struct ElementFlowCounters {
  std::string element_name;
  std::atomic<uint64_t> in_buffers{0};
  std::atomic<uint64_t> out_buffers{0};
  std::atomic<uint64_t> in_bytes{0};
  std::atomic<uint64_t> out_bytes{0};
  std::atomic<uint64_t> caps_changes{0};

  ElementFlowStats snapshot() const {
    ElementFlowStats s;
    s.element_name = element_name;
    s.in_buffers = in_buffers.load(std::memory_order_relaxed);
    s.out_buffers = out_buffers.load(std::memory_order_relaxed);
    s.in_bytes = in_bytes.load(std::memory_order_relaxed);
    s.out_bytes = out_bytes.load(std::memory_order_relaxed);
    s.caps_changes = caps_changes.load(std::memory_order_relaxed);
    return s;
  }
};

struct NextCpuDecision {
  int node_index = -1;
  std::string node_kind;
  std::string node_label;
  std::string next_cpu;
  bool applied = false;
};

struct DiagCtx {
  std::string pipeline_string;
  std::vector<NodeReport> node_reports;

  bool queue2_enabled = false;
  int queue2_depth = 0;
  std::vector<NextCpuDecision> next_cpu_decisions;

  mutable std::mutex bus_mu;
  std::vector<BusMessage> bus;

  std::vector<std::unique_ptr<BoundaryFlowCounters>> boundaries;
  std::vector<std::unique_ptr<StageTimingCounters>> stage_timings;
  std::vector<std::unique_ptr<ElementTimingCounters>> element_timings;
  std::vector<std::unique_ptr<ElementFlowCounters>> element_flows;

  bool has_build_adaptation = false;
  BuildAdaptationSummary build_adaptation;

  static int64_t now_us() {
    return (int64_t)g_get_monotonic_time();
  }

  void push_bus(const std::string& type, const std::string& src, const std::string& detail) {
    std::lock_guard<std::mutex> lk(bus_mu);
    bus.push_back(BusMessage{type, src, detail, now_us()});
  }

  SessionReport snapshot_basic() const {
    SessionReport rep;
    rep.pipeline_string = pipeline_string;
    rep.nodes = node_reports;

    {
      std::lock_guard<std::mutex> lk(bus_mu);
      rep.bus = bus;
    }

    rep.boundaries.reserve(boundaries.size());
    for (const auto& b : boundaries) {
      if (!b)
        continue;
      rep.boundaries.push_back(b->snapshot());
    }

    rep.repro_gst_launch = "gst-launch-1.0 -v '" + pipeline_string + "'";
    rep.has_build_adaptation = has_build_adaptation;
    if (has_build_adaptation) {
      rep.build_adaptation = build_adaptation;
    }
    // rep.repro_env filled by caller if desired
    return rep;
  }
};

inline std::shared_ptr<DiagCtx> diag_as_ctx(const std::shared_ptr<void>& p) {
  if (!p)
    return {};
  // aliasing to the *same* DiagCtx type everywhere
  return std::shared_ptr<DiagCtx>(p, reinterpret_cast<DiagCtx*>(p.get()));
}

} // namespace simaai::neat::pipeline_internal

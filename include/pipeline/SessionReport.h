/**
 * @file
 * @ingroup diagnostics
 * @brief Structured pipeline diagnostics and reproduction helpers.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat {

struct BusMessage {
  std::string type;         // e.g. "ERROR", "WARNING", "STATE_CHANGED"
  std::string src;          // element/object name
  std::string detail;       // formatted message (incl debug if present)
  int64_t wall_time_us = 0; // monotonic
};

struct BoundaryFlowStats {
  std::string boundary_name;  // sima_b<N>
  int after_node_index = -1;  // upstream node index
  int before_node_index = -1; // downstream node index (may be -1 for terminal tap boundary)

  uint64_t in_buffers = 0;  // observed on identity:sink
  uint64_t out_buffers = 0; // observed on identity:src

  int64_t last_in_pts_ns = -1;
  int64_t last_out_pts_ns = -1;

  int64_t last_in_wall_us = 0;
  int64_t last_out_wall_us = 0;
};

struct NodeReport {
  int index = -1;
  std::string kind;                  // "FileInput", "H264Decode", ...
  std::string user_label;            // optional user-supplied label
  std::string backend_fragment;      // fragment with names
  std::vector<std::string> elements; // element names owned by this node
};

struct BuildAdaptationAction {
  std::string target; // e.g. "input_constraints", "appsrc_caps_seed"
  bool applied = false;
  std::string detail;
  std::string reason; // populated when applied == false
};

struct BuildAdaptationSummary {
  std::string shape_policy;
  std::string dynamic_capability;

  int seed_width = -1;
  int seed_height = -1;
  int seed_depth = -1;
  std::string seed_width_origin;
  std::string seed_height_origin;
  std::string seed_depth_origin;

  int max_width = -1;
  int max_height = -1;
  int max_depth = -1;
  std::string max_width_origin;
  std::string max_height_origin;
  std::string max_depth_origin;

  std::size_t max_input_bytes_guard = 0;
  std::string byte_guard_origin;
  bool allow_ingress_cvu_format_renegotiation = false;

  std::vector<BuildAdaptationAction> actions;
};

struct SessionReport {
  std::string pipeline_string;
  // Canonical machine-triage code. Prefer values from pipeline/ErrorCodes.h.
  // Empty when operation succeeds or only informational diagnostics are returned.
  std::string error_code;

  std::vector<NodeReport> nodes;
  std::vector<BusMessage> bus;
  std::vector<BoundaryFlowStats> boundaries;

  // Heavy-on-failure add-ons
  std::string caps_dump;
  std::vector<std::string> dot_paths;

  // Repro helpers
  std::string repro_gst_launch; // gst-launch-1.0 -v '...'
  std::string repro_env;        // env vars suggestion (GST_DEBUG, DOT dir)
  std::string repro_note;       // human summary + hint

  // Input-build adaptation snapshot (present for build(input) flows).
  bool has_build_adaptation = false;
  BuildAdaptationSummary build_adaptation;

  // Optional: JSON serialization for CI / support bundling
  std::string to_json() const;
};

} // namespace simaai::neat

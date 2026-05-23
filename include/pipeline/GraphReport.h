/**
 * @file
 * @ingroup diagnostics
 * @brief Structured pipeline diagnostics — what `Graph::validate()` and `NeatError::report()`
 * carry.
 *
 * `GraphReport` is the framework's primary triage record: when something fails (or even
 * when validation passes), it returns a structured snapshot of the pipeline shape, GStreamer
 * bus messages, build-adaptation history, optional caps dumps and DOT graph paths, plus
 * **reproducer information** — a standalone `gst-launch-1.0` command that re-creates the
 * pipeline outside the framework, env-var suggestions, and a human-readable summary.
 *
 * @see NeatError for the exception type that carries this report
 * @see "Validation API" (§29 of the design deep dive)
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief A single GStreamer bus message captured during pipeline build/run.
 *
 * The framework drains the GStreamer bus and accumulates messages into the report's `bus[]`
 * vector. Useful for triaging caps negotiation failures, missing plugins, and runtime errors
 * — the GStreamer-level error messages are usually more specific than NEAT's wrapper error.
 * @ingroup diagnostics
 */
struct BusMessage {
  std::string type;   ///< Message type: `"ERROR"`, `"WARNING"`, `"STATE_CHANGED"`, etc.
  std::string src;    ///< Source element/object name (typically a NEAT-deterministic name like
                      ///< `"n3_videoconvert"`).
  std::string detail; ///< Formatted message text, including GStreamer debug info when present.
  int64_t wall_time_us = 0; ///< Monotonic wall-clock timestamp at message capture (microseconds).
};

/**
 * @brief Per-boundary flow statistics — buffer counts at identity probes between Nodes.
 *
 * The framework can insert `identity` probes at boundaries between Nodes (named `sima_b<N>`)
 * to track how many buffers flow through each boundary. Useful for localizing where in the
 * pipeline a backlog or stall is occurring.
 * @ingroup diagnostics
 */
struct BoundaryFlowStats {
  std::string boundary_name;  ///< Boundary identifier (e.g., `"sima_b3"`).
  int after_node_index = -1;  ///< Upstream Node index.
  int before_node_index = -1; ///< Downstream Node index (may be -1 for a terminal tap boundary).

  uint64_t in_buffers = 0;  ///< Buffers observed on the identity element's sink pad.
  uint64_t out_buffers = 0; ///< Buffers observed on the identity element's src pad.

  int64_t last_in_pts_ns = -1;  ///< PTS of the most recent buffer seen on input.
  int64_t last_out_pts_ns = -1; ///< PTS of the most recent buffer seen on output.

  int64_t last_in_wall_us = 0;  ///< Wall-clock time of the most recent input buffer.
  int64_t last_out_wall_us = 0; ///< Wall-clock time of the most recent output buffer.
};

/**
 * @brief Per-Node entry in the report, listing what that Node produced in the GStreamer pipeline.
 * @ingroup diagnostics
 */
struct NodeReport {
  int index = -1;                    ///< Position in the Graph's Node list.
  std::string kind;                  ///< Node's kind label (e.g., `"FileInput"`, `"H264Decode"`).
  std::string user_label;            ///< Optional user-supplied label (when set on the Node).
  std::string backend_fragment;      ///< The GStreamer launch fragment this Node emitted.
  std::vector<std::string> elements; ///< Names of GStreamer elements owned by this Node.
};

/**
 * @brief One line item from the build-adaptation log.
 *
 * Each adaptation action describes what the framework attempted to adjust during build (e.g.,
 * tightening caps to match a seed input, allocating an appsrc with a specific format), whether
 * the action was applied, and if not why.
 * @ingroup diagnostics
 */
struct BuildAdaptationAction {
  std::string
      target; ///< What was being adapted (e.g., `"input_constraints"`, `"appsrc_caps_seed"`).
  bool applied = false; ///< Whether the action was successfully applied.
  std::string detail;   ///< Description of what was adapted.
  std::string reason;   ///< Reason for non-application (populated only when `applied == false`).
};

/**
 * @brief Snapshot of build-time adaptation state — present for `build(input)` flows.
 *
 * When a Graph is built with a sample input, the framework adapts the pipeline's caps to
 * match the actual input shape/format. This struct captures what was seeded, what dynamic
 * capabilities were used, and the full action log.
 * @ingroup diagnostics
 */
struct BuildAdaptationSummary {
  std::string shape_policy;       ///< Resolved shape policy (e.g., `"static"`, `"dynamic"`).
  std::string dynamic_capability; ///< Which dynamic-shape capability the planner used.

  int seed_width = -1;            ///< Effective width seeded into the pipeline (-1 if not seeded).
  int seed_height = -1;           ///< Effective height seeded into the pipeline.
  int seed_depth = -1;            ///< Effective channel depth.
  std::string seed_width_origin;  ///< Where the seed_width came from (e.g., `"input"`,
                                  ///< `"manifest_default"`).
  std::string seed_height_origin; ///< Source of the seed_height value.
  std::string seed_depth_origin;  ///< Source of the seed_depth value.

  int max_width = -1;            ///< Maximum width the pipeline was prepared to accept.
  int max_height = -1;           ///< Maximum height.
  int max_depth = -1;            ///< Maximum channel depth.
  std::string max_width_origin;  ///< Source of the max_width value.
  std::string max_height_origin; ///< Source of the max_height value.
  std::string max_depth_origin;  ///< Source of the max_depth value.

  std::size_t max_input_bytes_guard = 0; ///< Configured cap on input buffer size, if any.
  std::string byte_guard_origin;         ///< Source of the byte-guard value.
  bool allow_ingress_cvu_format_renegotiation =
      false; ///< Whether the planner allowed mid-stream CVU caps renegotiation.

  std::vector<BuildAdaptationAction> actions; ///< Full ordered log of adaptation attempts.
};

/**
 * @brief Structured pipeline diagnostics — the framework's primary triage record.
 *
 * Returned by `Graph::validate()` (always) and carried by `NeatError::report()` (when
 * thrown). On success, contains the pipeline string and any informational diagnostics; on
 * failure, also contains `error_code`, captured bus messages, optional caps dumps, and a
 * standalone reproducer command.
 *
 * **Triage workflow**: read `error_code` first (machine-bucketable), then `repro_note` (human
 * summary), then `bus[]` for first terminal errors, then `repro_gst_launch` to replay outside
 * the framework.
 *
 * @see NeatError, ValidateOptions, "Error code taxonomy" (§41)
 * @ingroup diagnostics
 */
struct GraphReport {
  std::string pipeline_string; ///< The GStreamer launch string the framework produced.
  /**
   * @brief Canonical machine-triage code (see `pipeline/ErrorCodes.h`).
   *
   * Empty when the operation succeeded or only informational diagnostics are returned.
   * Otherwise, one of the `domain.reason` constants from `simaai::neat::error_codes`.
   */
  std::string error_code;

  std::vector<NodeReport> nodes;             ///< Per-Node entries: what each Node emitted.
  std::vector<BusMessage> bus;               ///< Captured GStreamer bus messages.
  std::vector<BoundaryFlowStats> boundaries; ///< Per-boundary flow stats (if probes were inserted).

  // ── Heavy-on-failure add-ons ─────────────────────────────────────────────────────────────
  std::string caps_dump; ///< Verbose caps dump (populated on caps negotiation failures).
  std::vector<std::string> dot_paths; ///< Paths to GraphViz `.dot` files dumped for visualization.

  // ── Reproducer helpers ───────────────────────────────────────────────────────────────────
  std::string repro_gst_launch; ///< Standalone `gst-launch-1.0 -v '...'` command that reproduces
                                ///< the pipeline.
  std::string repro_env;        ///< Suggested env vars for reproduction (GST_DEBUG, DOT dir, etc.).
  std::string repro_note;       ///< Human-readable summary + actionable hint.

  /// True when `build_adaptation` is meaningful (only for `build(input)` flows).
  bool has_build_adaptation = false;
  BuildAdaptationSummary build_adaptation; ///< Build-time caps adaptation snapshot.

  /// Serialize the entire report to JSON for CI bundling, support tickets, or persistent triage.
  std::string to_json() const;
};

} // namespace simaai::neat

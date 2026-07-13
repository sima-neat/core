/**
 * @file
 * @ingroup nodes_common
 * @brief `Output` Node — terminal sink that exposes pulled samples to `Run::pull()`.
 *
 * Wraps an `appsink`-style element. Place at the end of a Graph that uses async-mode
 * (`Run::pull()`). For sync-mode pipelines that write directly to a file, use the
 * file-output Graph fragments instead.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

/**
 * @brief How a public Graph output combines multiple named producers.
 *
 * This is only meaningful when an `Output("name")` is used as a named public Graph boundary and
 * more than one upstream endpoint connects to it. With the default `None`, multiple producers are
 * rejected so accidental fan-in does not silently mix samples.
 *
 * @ingroup nodes_common
 */
enum class CombinePolicy {
  /// No combining. The Output accepts one producer. Multiple producers fail closed and ask the
  /// user to choose an explicit policy.
  None,

  /// Combine one sample from each declared input when their NEAT `Sample::frame_id` values match.
  /// There is no fallback to PTS; samples without frame IDs fail with an explicit diagnostic.
  ByFrame,

  /// Combine one sample from each declared input when their media presentation timestamp
  /// (`Sample::pts_ns`, Presentation Timestamp) values match. There is no fallback to frame IDs.
  ByPts,

  /// Fairly forward samples from multiple producers one at a time, preserving each original
  /// sample rather than bundling. This is the live-stream fan-in policy for sharing one downstream
  /// path across multiple already-independent streams.
  RoundRobin,
};

/**
 * @brief Buffering / sync options for the `Output` Node.
 *
 * The three factory presets cover the common use cases — pick one rather than
 * configuring fields manually.
 *
 * @ingroup nodes_common
 */
struct OutputOptions {
  int max_buffers = 4; ///< Cap on queued samples before back-pressure. `0` = unbounded.
  bool drop = false;   ///< If true, drop the oldest sample on overflow instead of blocking.
  bool sync = false;   ///< If true, sync to the pipeline clock (slower, but matches realtime).
  CombinePolicy combine_policy = CombinePolicy::None; ///< Multi-producer Graph combine policy.

  /// Preset: keep only the most recent sample (drop=true, max_buffers=1). Lowest latency.
  static OutputOptions Latest();
  /// Preset: keep up to `max_buffers` samples; consumer pulls every frame.
  static OutputOptions EveryFrame(int max_buffers = 30);
  /// Preset: clocked output (sync=true) — pace samples to wall-clock.
  static OutputOptions Clocked(int max_buffers = 1);
};

/**
 * @brief Pull-side terminal sink. Samples land here for `Run::pull()` to consume.
 *
 * @ingroup nodes_common
 */
class Output final : public Node {
public:
  /// Construct with default options (`max_buffers=4`, `drop=false`, `sync=false`).
  Output() = default;
  /// Construct with a public Graph output endpoint name.
  explicit Output(std::string name) : endpoint_name_(std::move(name)) {}
  /// Construct with explicit options.
  explicit Output(OutputOptions opt) : opt_(std::move(opt)) {}
  /// Construct with a public Graph output endpoint name and explicit options.
  Output(std::string name, OutputOptions opt)
      : opt_(std::move(opt)), endpoint_name_(std::move(name)) {}

  /// Inspect the Node's options.
  const OutputOptions& options() const {
    return opt_;
  }

  /// Type label for this Node kind.
  std::string kind() const override {
    return "Output";
  }
  /// Optional public Graph endpoint name used by `Run::pull(name, ...)`.
  std::string user_label() const override {
    return endpoint_name_;
  }
  /// Explicit public endpoint name, empty when unnamed.
  const std::string& endpoint_name() const noexcept {
    return endpoint_name_;
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }
  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;

private:
  OutputOptions opt_;
  std::string endpoint_name_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for an `Output` Node with optional `OutputOptions`.
std::shared_ptr<simaai::neat::Node> Output(OutputOptions opt = {});
/// Convenience factory for a named public Graph output endpoint.
std::shared_ptr<simaai::neat::Node> Output(std::string name, OutputOptions opt = {});
} // namespace simaai::neat::nodes

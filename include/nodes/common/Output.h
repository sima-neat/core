/**
 * @file
 * @ingroup nodes_common
 * @brief `Output` Node — terminal sink that exposes pulled samples to `Run::pull()`.
 *
 * Wraps an `appsink`-style element. Place at the end of a Session that uses async-mode
 * (`Run::pull()`). For sync-mode pipelines that write directly to a file, use the
 * file-output NodeGroups instead.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <utility>
#include <vector>

namespace simaai::neat {

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
  /// Construct with explicit options.
  explicit Output(OutputOptions opt) : opt_(std::move(opt)) {}

  /// Inspect the Node's options.
  const OutputOptions& options() const {
    return opt_;
  }

  /// Type label for this Node kind.
  std::string kind() const override {
    return "Output";
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
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for an `Output` Node with optional `OutputOptions`.
std::shared_ptr<simaai::neat::Node> Output(OutputOptions opt = {});
} // namespace simaai::neat::nodes

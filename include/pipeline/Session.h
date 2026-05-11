/**
 * @file
 * @ingroup pipeline
 * @brief Session — the assembly stage that takes Nodes and turns them into a runnable Run.
 *
 * `Session` is the central concept of the framework. It collects Nodes (or NodeGroups,
 * which are bundles of Nodes), validates them against built-in contracts, compiles them
 * into a deterministic GStreamer pipeline string, instantiates the pipeline, negotiates
 * caps between adjacent elements, and returns a `Run` handle for push/pull execution.
 * `Model` is internally a Session wrapper: the same composition, validation, and runtime
 * machinery powers Model underneath. New users typically use `Model::run()`; advanced
 * users compose their own Sessions with `model.session()` plus extra Nodes for input
 * sources, custom processing, side branches, or RTSP server output.
 *
 * @see Run for the runtime handle a built Session produces
 * @see RtspServerHandle for server-mode Sessions
 * @see "Sessions: the assembly contract" (§0.12 of the design deep dive)
 */
#pragma once

#include "builder/Node.h"
#include "builder/NodeGroup.h"
#include "pipeline/SessionOptions.h"
#include "pipeline/Run.h"
#include "builder/GraphPrinter.h"
#include "nodes/common/Output.h"
#include "nodes/common/Caps.h"
#include "nodes/common/FileInput.h"
#include "nodes/common/JpegDecode.h"
#include "nodes/common/VideoTrackSelect.h"
#include "nodes/common/Queue.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/common/VideoScale.h"
#include "nodes/io/StillImageInput.h"
#include "nodes/io/Input.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264Depacketize.h"
#include "nodes/sima/H264DecodeSima.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/H264Packetize.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <gst/gst.h>

namespace simaai::neat {
namespace internal {
struct ModelAccess;
}
namespace pipeline_internal {
struct InputRouteProcessor;
}

/**
 * @brief Live handle for a Session running in RTSP server mode.
 *
 * Returned by `Session::run_rtsp(opts)`. The handle owns the running RTSP server background
 * threads and the underlying GStreamer pipeline. Destroying the handle (or calling `stop()`)
 * tears down the server cleanly. Move-only: the handle uniquely owns its server.
 *
 * Use this when the Session terminates in an H.264 encoded stream that should be published as
 * an RTSP source for downstream consumers (VLC, ffmpeg, browsers via WebRTC gateway, etc.).
 *
 * @see Session::run_rtsp
 * @ingroup pipeline
 */
class RtspServerHandle {
public:
  /// Default-construct an empty handle that owns no server.
  RtspServerHandle() = default;
  /// Destructor; stops the server and tears down the underlying pipeline.
  ~RtspServerHandle();

  RtspServerHandle(const RtspServerHandle&) = delete;            ///< Non-copyable: server ownership is unique.
  RtspServerHandle& operator=(const RtspServerHandle&) = delete; ///< Non-copyable.

  RtspServerHandle(RtspServerHandle&&) noexcept;            ///< Move-constructible.
  RtspServerHandle& operator=(RtspServerHandle&&) noexcept; ///< Move-assignable.

  /// Returns the broadcast URL (e.g., `"rtsp://0.0.0.0:8554/stream"`) clients connect to.
  const std::string& url() const {
    return url_;
  }
  /// Stop the server and tear down the pipeline. Safe to call multiple times.
  void stop();
  /// Alias for `stop()`. Kept for parity with other handle types.
  void kill() {
    stop();
  }
  /// Returns `true` while the server is actively serving clients.
  bool running() const;

private:
  friend class Session;

  std::string url_;
  void* impl_ = nullptr;
  std::shared_ptr<void> guard_;
};

/**
 * @brief The assembly stage — turns a list of Nodes into a runnable, deterministic pipeline.
 *
 * A Session does five jobs when you call `build()`:
 * 1. **Composition** — collects Nodes and NodeGroups in the order you added them.
 * 2. **Validation** — runs structural contracts (no empty pipeline, no null nodes, sink last,
 *    etc.) and surfaces issues as a structured `SessionReport`.
 * 3. **Compilation** — translates the Node sequence into a deterministic GStreamer pipeline
 *    string with stable element names like `n3_videoconvert`.
 * 4. **Negotiation** — hands the pipeline to GStreamer, which negotiates caps between adjacent
 *    elements (formats, resolutions, framerates, memory layouts).
 * 5. **Materialization** — instantiates the actual GStreamer elements, transitions through
 *    NULL → READY → PAUSED state, and returns a `Run` handle.
 *
 * @code
 *   sima::Session sess;
 *   sess.add(sima::nodes::groups::RtspDecodedInput({.url = "rtsp://camera/stream"}));
 *   sess.add(model.session());
 *   sess.add(sima::Output{});
 *   auto run = sess.build();
 * @endcode
 *
 * Sessions are **non-copyable** but **movable**. They are not thread-safe — build a Session on
 * one thread, then hand the resulting `Run` to wherever it's needed.
 *
 * @see Run for the runtime handle this produces
 * @see Model — the simplified entry point that wraps a Session for users who don't need
 *      composition flexibility
 * @see SessionReport for the structured error/diagnostics surface
 * @see RtspServerHandle for server-mode Sessions
 * @ingroup pipeline
 */
class Session {
public:
  /// Callback signature used by source-mode pipelines (no input pushed; pipeline produces tensors continuously).
  using TensorCallback = std::function<bool(const simaai::neat::Tensor&)>;

  /// Construct an empty Session with the given options (or defaults).
  explicit Session(const SessionOptions& opt = {});
  /// Destroy the Session, stopping any running pipeline.
  ~Session();
  Session(const Session&) = delete;            ///< Non-copyable.
  Session& operator=(const Session&) = delete; ///< Non-copyable.
  Session(Session&&) noexcept;                  ///< Move-constructible.
  Session& operator=(Session&&) noexcept;       ///< Move-assignable.

  // ── Core: add Nodes / NodeGroups ─────────────────────────────────────────────────────────
  /**
   * @brief Append a single Node to the Session.
   * @param node Shared pointer (typically returned by a Node factory like `sima::nodes::common::Queue()`).
   * @return `*this` to allow chaining.
   */
  Session& add(std::shared_ptr<Node> node);
  /// Append a NodeGroup (a bundle of Nodes) by copy.
  Session& add(const NodeGroup& group);
  /// Append a NodeGroup by move (avoids copying internal Node vector).
  Session& add(NodeGroup&& group);

  // ── Custom GStreamer escape hatch ────────────────────────────────────────────────────────
  /**
   * @brief Splice a raw GStreamer launch fragment into the pipeline.
   *
   * Useful for one-off experiments, third-party plugins NEAT doesn't wrap, or GStreamer
   * features (`tee`, `selector`, dynamic pads) that are awkward to model as Nodes. The
   * trade-off: you lose deterministic naming for the spliced fragment. Use sparingly.
   *
   * @param fragment Raw GStreamer launch string (e.g., `"identity silent=false ! videocrop ..."`).
   * @return `*this` to allow chaining.
   */
  Session& custom(std::string fragment);
  /// Variant that declares the fragment's role (e.g., source vs. sink).
  Session& custom(std::string fragment, InputRole role);

  // ── Typed runner: last node must be Output() ────────────────────────────────────────────
  /**
   * @brief Run a source-mode pipeline (no inputs pushed; producer Nodes drive the flow).
   *
   * Used in conjunction with `set_tensor_callback()`. The pipeline runs until end-of-stream
   * or until the callback returns `false`.
   *
   * @throws SessionError on validation or runtime failure (with structured `SessionReport`).
   */
  void run();
  /// One-shot synchronous push+pull from `cv::Mat` inputs.
  TensorList run(const std::vector<cv::Mat>& inputs, const RunOptions& opt = {});
  /// One-shot synchronous push+pull from `Tensor` inputs.
  TensorList run(const TensorList& inputs, const RunOptions& opt = {});
  /// One-shot synchronous push+pull from `Sample` inputs (carries per-buffer metadata).
  SampleList run(const SampleList& inputs, const RunOptions& opt = {});
  /**
   * @brief Build a long-lived `Run` handle, seeding caps from `cv::Mat` inputs.
   * @param inputs One Mat per ingress port; used for build-time adaptation.
   * @param mode   `Async` (default; pipeline runs continuously) or `Sync`.
   * @param opt    Runtime options (queue depth, overflow policy).
   * @throws SessionError on validation or build failure.
   */
  Run build(const std::vector<cv::Mat>& inputs, RunMode mode = RunMode::Async,
            const RunOptions& opt = {});
  /// Build variant seeded with `Tensor` inputs.
  Run build(const TensorList& inputs, RunMode mode = RunMode::Async, const RunOptions& opt = {});
  /// Build variant seeded with full `Sample` inputs (with per-buffer metadata).
  Run build(const SampleList& inputs, RunMode mode = RunMode::Async, const RunOptions& opt = {});
  /**
   * @brief Validate the Session against a real input sample without running the pipeline.
   *
   * Runs structural contracts AND build-time adaptation against the input. Reports whether
   * the pipeline would accept this input shape/format and what conversions would be needed.
   * Useful in CI to catch shape mismatches before deploying.
   */
  SessionReport validate(const ValidateOptions& opt, const cv::Mat& input) const;

  // ── Server-style run ────────────────────────────────────────────────────────────────────
  /**
   * @brief Build the Session and run it as an RTSP server.
   *
   * The Session must terminate in an H.264 encoded stream. The returned handle owns a live
   * RTSP server publishing the pipeline's output to network clients. Stop the server by
   * calling `handle.stop()` or letting the handle go out of scope.
   *
   * @param opt RTSP server options (mount point, port, RTP port range).
   * @return Live `RtspServerHandle` exposing the broadcast URL.
   */
  RtspServerHandle run_rtsp(const RtspServerOptions& opt);

  /**
   * @brief Validate the Session structurally without running.
   *
   * Runs all built-in contracts (NonEmptyPipeline, NoNullNodes, SinkLastForRun, etc.) and
   * returns a structured `SessionReport`. Cheaper than `build()` because it doesn't instantiate
   * GStreamer state. Useful in unit tests and CI.
   *
   * @return `SessionReport` carrying any contract failures, with `error_code` and `repro_note`.
   */
  SessionReport validate(const ValidateOptions& opt = {}) const;

  // ── Tensor-friendly output helper ────────────────────────────────────────────────────────
  /**
   * @brief Append a tensor-friendly output (auto-inserts convert/scale/caps + sink).
   *
   * Convenience for "I want my output as a Tensor in a specific format/shape." Equivalent to
   * adding `VideoConvert`, `VideoScale`, `Caps`, and `Output` Nodes manually but encapsulated.
   *
   * @return `*this` to allow chaining.
   */
  Session& add_output_tensor(const OutputTensorOptions& opt = {});

  // ── UX helpers ───────────────────────────────────────────────────────────────────────────
  /// Returns a hierarchical, human-readable view of the Nodes added so far.
  std::string describe(const GraphPrinter::Options& opt = {}) const;
  /**
   * @brief Returns the GStreamer launch string the Session would emit at `build()`.
   *
   * Paste into `gst-launch-1.0` to reproduce the pipeline outside the framework — invaluable for
   * debugging caps issues or isolating "is this NEAT's bug or GStreamer's?"
   *
   * @param insert_boundaries If true, inserts diagnostic identity probes between Nodes.
   */
  std::string describe_backend(bool insert_boundaries = false) const;

  /// Attach an external lifetime guard (used by externally-managed runtimes).
  void set_guard(std::shared_ptr<void> guard);

  /// Set the per-tensor callback used by source-mode `run()`.
  void set_tensor_callback(TensorCallback cb);

  /// Save the Session's Node list, options, and topology to a JSON file at `path`.
  void save(const std::string& path) const;
  /// Reconstruct a Session from a previously-saved JSON file.
  static Session load(const std::string& path);

  /**
   * @brief Build a Session as an asynchronous runner without seeding inputs.
   *
   * Use this for source pipelines (Sessions whose first Node is a producer like `RTSPInput`
   * or `StillImageInput` — no `push()` from user code needed). Push pipelines should prefer
   * `build(inputs, ...)` so caps can be derived from the actual input.
   */
  Run build(const RunOptions& opt = {});

  /// Returns the GStreamer launch string from the most recent `build()` call.
  const std::string& last_pipeline() const {
    return last_pipeline_;
  }

private:
  friend struct simaai::neat::internal::ModelAccess;

  /**
   * @brief Opaque state carried by a built (not yet running) Session.
   *
   * Holds raw pointers to the materialised GStreamer pipeline and its sink element, plus an
   * opaque `diag` payload used by build-time diagnostics. Lives only between `build()` and
   * the destruction of the resulting `Run`.
   *
   * @ingroup pipeline
   */
  struct BuiltState {
    GstElement* pipeline = nullptr;     ///< Materialised GStreamer pipeline element.
    GstElement* sink = nullptr;         ///< Terminal sink element of the pipeline.
    std::shared_ptr<void> diag;         ///< Opaque diagnostics payload from build time.
  };
  struct RunCache;
  /**
   * @brief Per-NodeGroup metadata captured during build.
   *
   * Records the half-open `[start, end)` range of `nodes_` belonging to a single
   * `NodeGroup`, the group's caps-negotiation behavior, and an optional human-readable
   * label propagated to diagnostics.
   *
   * @ingroup pipeline
   */
  struct GroupMeta {
    std::size_t start = 0;                                          ///< Inclusive start index in `nodes_`.
    std::size_t end = 0;                                            ///< Exclusive end index in `nodes_`.
    NodeCapsBehavior caps_behavior = NodeCapsBehavior::Dynamic;     ///< Caps behavior for the group.
    std::string label;                                              ///< Optional group label for diagnostics.
  };

  std::vector<std::shared_ptr<Node>> nodes_;
  std::vector<GroupMeta> groups_;
  std::string last_pipeline_;
  std::shared_ptr<void> guard_;
  std::shared_ptr<void> verbose_guard_;
  SessionOptions opt_{};
  TensorCallback tensor_cb_;
  std::atomic<uint64_t> nodes_version_{0};
  std::unique_ptr<BuiltState> built_;
  std::unique_ptr<RunCache> run_cache_;
  uint64_t built_version_ = 0;
  std::shared_ptr<const pipeline_internal::InputRouteProcessor> input_route_processor_;

  void build_cached_source();
};

} // namespace simaai::neat

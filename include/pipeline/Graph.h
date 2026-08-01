/**
 * @file
 * @ingroup pipeline
 * @brief Graph — the assembly stage that takes Nodes and turns them into a runnable Run.
 *
 * `Graph` is the central concept of the framework. It collects Nodes and reusable Graph
 * fragments, validates them against built-in contracts, compiles them
 * into a deterministic GStreamer pipeline string, instantiates the pipeline, negotiates
 * caps between adjacent elements, and returns a `Run` handle for push/pull execution.
 * `Model` is internally a Graph wrapper: the same composition, validation, and runtime
 * machinery powers Model underneath. New users typically use `Model::run()`; advanced
 * users compose their own Graphs with `graph.add(model)` plus extra Nodes for input
 * sources, custom processing, side branches, or RTSP server output.
 *
 * @see Run for the runtime handle a built Graph produces
 * @see RtspServerHandle for server-mode Graphs
 * @see "Graphs: the assembly contract" (§0.12 of the design deep dive)
 */
#pragma once

#include "builder/Node.h"
#include "pipeline/GraphOptions.h"
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
#include "nodes/sima/SimaDecode.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <gst/gst.h>

namespace simaai::neat {
namespace internal {
struct ModelAccess;
}
namespace graph {
class Node;
}
namespace pipeline_internal {
struct InputRouteProcessor;
}

class Graph;
class Model;

#ifdef SIMA_NEAT_INTERNAL
namespace runtime {
struct ExecutionGraphPlan;
struct FragmentBoundaryHints;
struct FragmentPlan;
struct Provenance;
ExecutionGraphPlan compile_public_graph(const Graph& graph, const RunOptions& opt,
                                        std::optional<Sample> seed = std::nullopt);
} // namespace runtime
#endif

/**
 * @brief Live handle for a Graph running in RTSP server mode.
 *
 * Returned by `Graph::run_rtsp(opts)`. The handle owns the running RTSP server background
 * threads and the underlying GStreamer pipeline. Destroying the handle (or calling `stop()`)
 * tears down the server cleanly. Move-only: the handle uniquely owns its server.
 *
 * Use this when the Graph terminates in an H.264 encoded stream that should be published as
 * an RTSP source for downstream consumers (VLC, ffmpeg, browsers via WebRTC gateway, etc.).
 *
 * @see Graph::run_rtsp
 * @ingroup pipeline
 */
class RtspServerHandle {
public:
  /// Default-construct an empty handle that owns no server.
  RtspServerHandle() = default;
  /// Destructor; stops the server and tears down the underlying pipeline.
  ~RtspServerHandle();

  RtspServerHandle(const RtspServerHandle&) = delete; ///< Non-copyable: server ownership is unique.
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
  friend class Graph;

  std::string url_;
  void* impl_ = nullptr;
  std::shared_ptr<void> guard_;
};

/**
 * @brief The assembly stage — turns a list of Nodes into a runnable, deterministic pipeline.
 *
 * A Graph does five jobs when you call `build()`:
 * 1. **Composition** — collects Nodes and Graph fragments in the order you added them.
 * 2. **Validation** — runs structural contracts (no empty pipeline, no null nodes, sink last,
 *    etc.) and surfaces issues as a structured `GraphReport`.
 * 3. **Compilation** — translates the Node sequence into a deterministic GStreamer pipeline
 *    string with stable element names like `n3_videoconvert`.
 * 4. **Negotiation** — hands the pipeline to GStreamer, which negotiates caps between adjacent
 *    elements (formats, resolutions, framerates, memory layouts).
 * 5. **Materialization** — instantiates the actual GStreamer elements, transitions through
 *    NULL → READY → PAUSED state, and returns a `Run` handle.
 *
 * @code
 *   sima::Graph graph;
 *   graph.add(sima::nodes::groups::RtspDecodedInput({.url = "rtsp://camera/stream"}));
 *   graph.add(model);
 *   graph.add(sima::nodes::Output("classes"));
 *   auto run = graph.build();
 * @endcode
 *
 * Graphs are **non-copyable** but **movable**. They are not thread-safe — build a Graph on
 * one thread, then hand the resulting `Run` to wherever it's needed.
 *
 * @see Run for the runtime handle this produces
 * @see Model — the simplified entry point that wraps a Graph for users who don't need
 *      composition flexibility
 * @see GraphReport for the structured error/diagnostics surface
 * @see RtspServerHandle for server-mode Graphs
 * @ingroup pipeline
 */
class Graph {
public:
  /// Callback signature used by source-mode pipelines (no input pushed; pipeline produces tensors
  /// continuously).
  using TensorCallback = std::function<bool(const simaai::neat::Tensor&)>;

  /// Construct an empty Graph with the given options (or defaults).
  explicit Graph(const GraphOptions& opt = {});
  /// Construct an empty Graph with a label used for diagnostics, save/load, and visualization.
  /// Public I/O endpoints are declared by `nodes::Input("name")`,
  /// `nodes::Output("name")`, or model route boundary hints, not by the Graph label.
  explicit Graph(std::string name, const GraphOptions& opt = {});
  /// Destroy the Graph, stopping any running pipeline. Never throws — all GStreamer
  /// teardown happens via RAII handles inside `BuiltState` whose deleters swallow
  /// (and log) any rare exception so that destruction is always safe.
  ~Graph() noexcept;
  Graph(const Graph&) = delete;            ///< Non-copyable.
  Graph& operator=(const Graph&) = delete; ///< Non-copyable.
  Graph(Graph&&) noexcept;                 ///< Move-constructible.
  Graph& operator=(Graph&&) noexcept;      ///< Move-assignable.

  /// Set the Graph label used for diagnostics, save/load, and visualization.
  Graph& set_name(std::string name);
  /// Return the Graph label, if one was provided.
  const std::string& name() const noexcept;
  /// Return public logical input endpoint names declared by this Graph.
  std::vector<std::string> inputs() const;
  /// Return public logical output endpoint names declared by this Graph.
  std::vector<std::string> outputs() const;

  // ── Core: add Nodes / Graph fragments ───────────────────────────────────────────────────
  /**
   * @brief Append a single Node to the Graph.
   * @param node Shared pointer (typically returned by a Node factory like
   * `sima::nodes::common::Queue()`).
   * @return `*this` to allow chaining.
   */
  Graph& add(std::shared_ptr<Node> node);
  /// Append another Graph as a linear fragment. A connected fragment may only initialize an empty
  /// Graph; compose non-linear topology with connect().
  Graph& add(const Graph& fragment);
  /// Append another Graph as a linear fragment. A connected fragment may only initialize an empty
  /// Graph; compose non-linear topology with connect().
  Graph& add(Graph&& fragment);
  /// Append a model route as a linear graph fragment.
  Graph& add(const Model& model);

  // ── Explicit graph composition ───────────────────────────────────────────────────────────
  /**
   * @brief Connect named endpoints or reusable fragments through the public Graph compiler.
   *
   * `add()` means linear splicing. `connect()` means explicit graph topology. Endpoints are
   * declared with existing `nodes::Input("name")` / `nodes::Output("name")` boundary nodes or by
   * model route boundary hints. `Graph("name")` is only a label/debug name and does not create an
   * endpoint. All overloads lower through the same
   * ExecutionGraphPlan/RunCore path, preserving named endpoints for diagnostics, save/load,
   * visualization, and named `Run::push()` / `Run::pull()`.
   */
  Graph& connect(std::string_view from_endpoint, std::string_view to_endpoint);
  Graph& connect(const Graph& from, const Graph& to);
  Graph& connect(const Graph& from, const Graph& to, const GraphLinkOptions& options);
  Graph& connect(std::shared_ptr<Node> from, std::shared_ptr<Node> to);
  Graph& connect(const Graph& from, std::shared_ptr<Node> to);
  Graph& connect(std::shared_ptr<Node> from, const Graph& to);
  Graph& connect(const Model& from, const Model& to);
  Graph& connect(const Model& from, const Graph& to);
  Graph& connect(const Graph& from, const Model& to);
  Graph& connect(const Model& from, std::shared_ptr<Node> to);
  Graph& connect(std::shared_ptr<Node> from, const Model& to);

#ifdef SIMA_NEAT_INTERNAL
  /// Internal bridge: append a public/pipeline vertex without creating an implicit add() edge.
  std::size_t append_pipeline_vertex_for_internal_graph_(std::shared_ptr<Node> node);
  /// Internal bridge: append a runtime-stage vertex without exposing graph::Graph publicly.
  std::size_t
  append_runtime_vertex_for_internal_graph_(std::shared_ptr<simaai::neat::graph::Node> node);
  /// Internal bridge: connect two composition vertices by runtime port names.
  void connect_runtime_port_for_internal_graph_(std::size_t from, std::string_view from_port,
                                                std::size_t to, std::string_view to_port);
#endif

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
  Graph& custom(std::string fragment);
  /// Variant that declares the fragment's role (e.g., source vs. sink).
  Graph& custom(std::string fragment, InputRole role);

  // ── Typed runner: last node must be Output() ────────────────────────────────────────────
  /**
   * @brief Run a source-mode pipeline (no inputs pushed; producer Nodes drive the flow).
   *
   * Used in conjunction with `set_tensor_callback()`. The pipeline runs until end-of-stream
   * or until the callback returns `false`.
   *
   * @throws NeatError on validation or runtime failure (with structured `GraphReport`).
   */
  void run();
  /// One-shot synchronous push+pull from `cv::Mat` inputs.
  TensorList run(const std::vector<cv::Mat>& inputs, const RunOptions& opt = {});
  /// One-shot synchronous push+pull from `Tensor` inputs.
  TensorList run(const TensorList& inputs, const RunOptions& opt = {});
  /// One-shot synchronous push+pull from `Sample` inputs (carries per-buffer metadata).
  Sample run(const Sample& inputs, const RunOptions& opt = {});
  /**
   * @brief Build a long-lived async `Run` handle, seeding caps from `cv::Mat` inputs.
   * @param inputs One Mat per ingress port; used for build-time adaptation.
   * @param opt    Runtime options (queue depth, overflow policy).
   * @throws NeatError on validation or build failure.
   *
   * `build(...)` always returns a reusable push/pull runner. Use `run(...)` for one-shot
   * synchronous execution.
   */
  Run build(const std::vector<cv::Mat>& inputs, const RunOptions& opt = {});
  /// Build variant seeded with `Tensor` inputs.
  Run build(const TensorList& inputs, const RunOptions& opt = {});
  /// Build variant seeded with full `Sample` inputs (with per-buffer metadata).
  Run build(const Sample& inputs, const RunOptions& opt = {});
#ifdef SIMA_NEAT_INTERNAL
  /// Internal seeded build entry point; may create sync-mode Runs for runtime-owned caches.
  Run build_seeded_internal(const std::vector<cv::Mat>& inputs, RunMode mode,
                            const RunOptions& opt = {});
  /// Internal seeded build variant for tensors.
  Run build_seeded_internal(const TensorList& inputs, RunMode mode, const RunOptions& opt = {});
  /// Internal seeded build variant for Samples.
  Run build_seeded_internal(const Sample& inputs, RunMode mode, const RunOptions& opt = {});
#endif
  /**
   * @brief Validate the Graph against a real input sample without running the pipeline.
   *
   * Runs structural contracts AND build-time adaptation against the input. Reports whether
   * the pipeline would accept this input shape/format and what conversions would be needed.
   * Useful in CI to catch shape mismatches before deploying.
   */
  GraphReport validate(const ValidateOptions& opt, const cv::Mat& input) const;

  // ── Server-style run ────────────────────────────────────────────────────────────────────
  /**
   * @brief Build the Graph and run it as an RTSP server.
   *
   * The Graph must terminate in an H.264 encoded stream. The returned handle owns a live
   * RTSP server publishing the pipeline's output to network clients. Stop the server by
   * calling `handle.stop()` or letting the handle go out of scope.
   *
   * @param opt RTSP server options (mount point, port, RTP port range).
   * @return Live `RtspServerHandle` exposing the broadcast URL.
   */
  RtspServerHandle run_rtsp(const RtspServerOptions& opt);

  /**
   * @brief Validate the Graph structurally without running.
   *
   * Runs all built-in contracts (NonEmptyPipeline, NoNullNodes, SinkLastForRun, etc.) and
   * returns a structured `GraphReport`. Cheaper than `build()` because it doesn't instantiate
   * GStreamer state. Useful in unit tests and CI.
   *
   * @return `GraphReport` carrying any contract failures, with `error_code` and `repro_note`.
   */
  GraphReport validate(const ValidateOptions& opt = {}) const;

  // ── Tensor-friendly output helper ────────────────────────────────────────────────────────
  /**
   * @brief Append a tensor-friendly output (auto-inserts convert/scale/caps + sink).
   *
   * Convenience for "I want my output as a Tensor in a specific format/shape." Equivalent to
   * adding `VideoConvert`, `VideoScale`, `Caps`, and `Output` Nodes manually but encapsulated.
   *
   * @return `*this` to allow chaining.
   */
  Graph& add_output_tensor(const OutputTensorOptions& opt = {});

  // ── UX helpers ───────────────────────────────────────────────────────────────────────────
  /// Returns a hierarchical, human-readable view of the Nodes added so far.
  std::string describe(const GraphPrinter::Options& opt = {}) const;
  /**
   * @brief Returns the GStreamer launch string the Graph would emit at `build()`.
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

  /// Save the Graph's Node list, options, and topology to a JSON file at `path`.
  void save(const std::string& path) const;
  /// Reconstruct a Graph from a previously-saved JSON file.
  static Graph load(const std::string& path);

  /**
   * @brief Build a Graph as an asynchronous runner without seeding inputs.
   *
   * Use this for source pipelines (Graphs whose first Node is a producer like `RTSPInput`
   * or `StillImageInput` — no `push()` from user code needed). Push pipelines should prefer
   * `build(inputs, ...)` so caps can be derived from the actual input.
   */
  Run build(const RunOptions& opt = {});
  /**
   * @brief Preserve the established `build({})` default-options idiom.
   *
   * The seeded `build(...)` overloads also accept an empty braced value, so
   * `build({})` is otherwise ambiguous. This exact empty-list compatibility
   * shim forwards to the established RunOptions overload.
   */
  Run build(std::initializer_list<std::nullptr_t>) {
    return build(RunOptions{});
  }

  /// Returns the GStreamer launch string from the most recent `build()` call.
  const std::string& last_pipeline() const {
    return last_pipeline_;
  }

private:
  friend struct simaai::neat::internal::ModelAccess;

  // Opaque internal state types — the GStreamer pipeline + sink are owned via RAII
  // handles defined in the private pipeline build implementation, so the public header never
  // needs to know how they are torn down. Forward-declared here; defined alongside `RunCache` and
  // `CompositionGraph` in the internal detail header.
  struct BuiltState;
  struct CompositionGraph;
  enum class CompositionEdgeKind {
    ImplicitLinear,
    RuntimePort,
    PublicEndpoint,
  };
  struct EndpointEdgeMeta {
    std::string from_endpoint;
    std::string to_endpoint;
  };
  struct CompositionEdge {
    std::size_t from = static_cast<std::size_t>(-1);
    std::size_t to = static_cast<std::size_t>(-1);
    CompositionEdgeKind kind = CompositionEdgeKind::ImplicitLinear;
    std::string from_port;
    std::string to_port;
    std::optional<EndpointEdgeMeta> endpoint;
    GraphLinkOptions link_options;
    std::string stream_id;
  };
  struct GroupMeta;
  struct NamedFragment;
  struct RunCache;
#ifdef SIMA_NEAT_INTERNAL
  struct CompositionView {
    std::vector<std::shared_ptr<Node>> linear_nodes;
    std::vector<std::shared_ptr<Node>> vertices;
    std::vector<std::shared_ptr<simaai::neat::graph::Node>> runtime_vertices;
    std::span<const CompositionEdge> edges;
    std::span<const GroupMeta> groups;
    std::span<const runtime::FragmentPlan> fragments;
    std::span<const NamedFragment> named_fragments;
    GraphOptions options;
    std::string graph_name;
    bool graph_user_named = false;
    std::uint64_t graph_id = 0;
    std::uint64_t graph_version = 0;
    bool linear = true;
  };
#endif

  // Source-mode build helper. Body defined in the private pipeline build implementation.
  struct PreparedSource;
  enum class SinkRequirement : int { Required, OptionalIfPresent };

#ifdef SIMA_NEAT_INTERNAL
  CompositionView composition_view_for_internal_compile() const;
  friend runtime::ExecutionGraphPlan runtime::compile_public_graph(const Graph& graph,
                                                                   const RunOptions& opt,
                                                                   std::optional<Sample> seed);
#endif

  /// Drop the built pipeline (if any) and any cached runner. Tears down GStreamer
  /// resources via RAII; safe to call when there is no built state. Never throws.
  void invalidate_built_() noexcept;
  Run build_source_internal_(const RunOptions& opt);

  /// Shared "build a source-mode pipeline to PAUSED" body for `build(RunOptions)`
  /// and `build_cached_source()`. Validates, materializes, compiles, attaches all
  /// probes, resolves the sink, and drives to GST_STATE_PAUSED. Callers add the
  /// state transition step (PLAYING for live `build`, or stay at PAUSED to cache).
  PreparedSource prepare_source_(RunMode mode, const RunOptions& opt, SinkRequirement sink_req,
                                 const char* where);
  /**
   * @brief Per-fragment metadata captured during build.
   *
   * Records the half-open `[start, end)` insertion-order vertex range belonging to a
   * single reusable fragment, the fragment's caps-negotiation behavior, and an optional
   * human-readable label propagated to diagnostics.
   *
   * @ingroup pipeline
   */
  struct GroupMeta {
    std::size_t start = 0; ///< Inclusive composition vertex index.
    std::size_t end = 0;   ///< Exclusive composition vertex index.
    NodeCapsBehavior caps_behavior = NodeCapsBehavior::Dynamic; ///< Caps behavior for the group.
    std::string label; ///< Optional group label for diagnostics.
  };

  struct NamedFragment {
    std::size_t start = 0;
    std::size_t end = 0;
    std::string name;
    bool user_named = false;
  };

  std::unique_ptr<CompositionGraph> composition_;
  std::vector<GroupMeta> groups_;
  std::string last_pipeline_;
  std::shared_ptr<void> guard_;
  std::shared_ptr<void> verbose_guard_;
  GraphOptions opt_{};
  std::string endpoint_name_;
  TensorCallback tensor_cb_;
  uint64_t graph_id_ = 0;
  std::atomic<uint64_t> nodes_version_{0};
  std::unique_ptr<BuiltState> built_;
  std::unique_ptr<RunCache> run_cache_;
  uint64_t built_version_ = 0;
  std::shared_ptr<const pipeline_internal::InputRouteProcessor> input_route_processor_;

  void mark_composition_changed();
  std::pair<std::size_t, std::size_t> append_linear_fragment_(const Graph& fragment,
                                                              const char* where);
  std::pair<std::size_t, std::size_t> import_composition_fragment_(const Graph& fragment,
                                                                   const char* where);
  std::pair<std::size_t, std::size_t> import_or_reuse_composition_fragment_(const Graph& fragment,
                                                                            const char* where);
  bool is_output_collection_fragment_(const Graph& fragment) const;
  std::pair<std::size_t, std::size_t> import_output_collection_fragment_(const Graph& fragment,
                                                                         const char* where);
  std::pair<std::size_t, std::size_t>
  import_or_reuse_output_collection_fragment_(const Graph& fragment, const char* where);
  std::pair<std::size_t, std::size_t> import_or_reuse_node_fragment_(std::shared_ptr<Node> node,
                                                                     const char* where);
  std::pair<std::size_t, std::size_t> import_or_reuse_model_fragment_(const Model& model,
                                                                      const char* where);
  void connect_imported_ranges_(std::pair<std::size_t, std::size_t> from_range,
                                std::string_view from_name,
                                std::pair<std::size_t, std::size_t> to_range,
                                std::string_view to_name, const char* where);
#ifdef SIMA_NEAT_INTERNAL
  void attach_fragment_boundary_hints_(std::size_t start, std::size_t end,
                                       runtime::FragmentBoundaryHints hints);
  void attach_fragment_boundary_hints_(std::size_t start, std::size_t end,
                                       runtime::FragmentBoundaryHints hints,
                                       runtime::Provenance provenance);
#endif
  std::vector<std::shared_ptr<Node>> linear_nodes_snapshot(const char* where) const;
  void build_cached_source();
};

} // namespace simaai::neat

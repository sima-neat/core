/**
 * @file
 * @ingroup pipeline
 * @brief Graph/Run options, the `Sample` type, pull-status enums, and verbosity controls.
 *
 * Defines the option structs the framework consumes at construction/build time and the
 * `Sample` type that flows out of `Run::pull()`. Key types here:
 *   - `VerboseOptions` / `VerbosityLevel` — diagnostic verbosity controls (per topic).
 *   - `GraphOptions` — per-Graph knobs (callback timeout, naming, processor preference).
 *   - `RtspServerOptions` / `ValidateOptions` / `OutputTensorOptions` — option packs for specific
 * calls.
 *   - `RunMode` — internal Async vs Sync runtime timing mode.
 *   - `Sample` / `SampleKind` — the typed payload `pull()` returns; can be a Tensor, a
 *     TensorSet (multiple physical outputs), or a Bundle (recursive multi-logical-output).
 *   - `PullStatus` / `PullError` — structured pull results.
 *
 * @see Graph, Run, Tensor
 */
#pragma once

#include "pipeline/GraphReport.h"
#include "pipeline/FormatSpec.h"
#include "pipeline/PayloadType.h"
#include "pipeline/TensorTypes.h"
#include "pipeline/Tensor.h"
#include "pipeline/TensorCore.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cv {
class Mat;
}

namespace simaai::neat {

/**
 * @brief Runtime policy for a connection between two public Graph fragments.
 *
 * `Default` preserves lossless/backpressure semantics for ordinary one-to-one edges. If multiple
 * producers connect to the same live public input, the framework promotes those edges to
 * `RealtimeLatestByStream` automatically so users do not need app-local fan-in mutex code.
 * `RealtimeLatestByStream` keeps producers non-blocking, retains only the latest frame per
 * `Sample::stream_id` (or per source edge when the stream id is empty), and schedules ready
 * streams fairly into the downstream graph. `RealtimeEveryFrameByStream` is the opt-in fused
 * decoder-source variant: it retains one pending frame per stream and blocks only that producer
 * until the mux consumes it, preserving bursty input without another EV-memory queue. Build these
 * links with `fuse_realtime_source_branches`.
 */
enum class GraphLinkPolicy {
  Default = 0,
  RealtimeLatestByStream,
  RealtimeEveryFrameByStream,
};

/**
 * @brief Options for `Graph::connect(from, to, options)`.
 */
struct GraphLinkOptions {
  GraphLinkPolicy policy = GraphLinkPolicy::Default;
  int queue_depth = 16;
  /// Compatibility stream id to stamp on samples crossing this link before realtime scheduling.
  /// New runtime code copies this value into internal edge metadata during composition; leave it
  /// empty for automatic per-edge identity.
  std::string stream_id;
  /// Only applies to realtime-by-stream links carrying raw decoder-backed samples.
  /// -1 uses the framework default (4); positive values set the per-stream raw-frame inflight cap.
  int max_inflight_per_stream = -1;
  /// Only applies to realtime-by-stream links carrying raw decoder-backed samples.
  /// -1 keeps env/default behavior; positive values set the total cap across streams.
  int max_inflight_total = -1;
};

/**
 * @brief Coarse-grained framework verbosity selector.
 *
 * Combined with the per-topic flags in `VerboseOptions` to decide what diagnostic output the
 * framework emits at runtime. Most users pick one of the three preset factories on
 * `VerboseOptions` (`quiet()`, `production()`, `debug_all()`) rather than setting flags by hand.
 * @ingroup diagnostics
 */
enum class VerbosityLevel {
  Quiet,      ///< Suppress topic-specific messages; progress is controlled separately.
  Production, ///< Concise phase updates suitable for end users.
  Verbose,    ///< Rich human-facing diagnostics across all framework topics.
};

/**
 * @brief Per-topic framework verbosity controls.
 *
 * Each boolean enables one diagnostic topic. The factory methods `quiet()`, `production()`,
 * `debug_plugins()`, and `debug_all()` return preset bundles for common use cases.
 * @ingroup diagnostics
 */
struct VerboseOptions {
  /// Coarse verbosity level: Quiet / Production / Verbose. Defaults to production-safe output.
  VerbosityLevel level = VerbosityLevel::Production;

  /// Emit clean lifecycle/progress updates such as "Model loaded" and "Building graph...".
  bool progress = true;

  /// Force progress output even when stderr is not a TTY.
  bool progress_force = false;

  /// Surface GStreamer/loader startup details that are otherwise suppressed.
  bool gstreamer = false;

  /// Surface model route / preprocess-planner diagnostics.
  bool planner = false;

  /// Surface graph scheduling / stage / teardown diagnostics.
  bool graph = false;

  /// Surface pipeline lifecycle diagnostics such as state transitions and flow traces.
  bool pipeline = false;

  /// Surface input stream / appsink / appsrc detail.
  bool inputstream = false;

  /// Surface tensor holder / mapping / payload detail.
  bool tensor = false;

  /// Surface plugin-internal detail such as processcvu / processmla / boxdecode traces.
  bool plugins = false;

  /// Preset: silence all topics, including progress messages. For embedded/headless production.
  [[nodiscard]] static VerboseOptions quiet() {
    VerboseOptions opt;
    opt.level = VerbosityLevel::Quiet;
    opt.progress = false;
    opt.progress_force = false;
    opt.gstreamer = false;
    opt.planner = false;
    opt.graph = false;
    opt.pipeline = false;
    opt.inputstream = false;
    opt.tensor = false;
    opt.plugins = false;
    return opt;
  }

  /// Preset: concise progress messages, no topic spam. For typical production deployments.
  [[nodiscard]] static VerboseOptions production() {
    VerboseOptions opt;
    opt.level = VerbosityLevel::Production;
    opt.progress = true;
    opt.progress_force = false;
    opt.gstreamer = false;
    opt.planner = false;
    opt.graph = false;
    opt.pipeline = false;
    opt.inputstream = false;
    opt.tensor = false;
    opt.plugins = false;
    return opt;
  }

  /// Preset: production messages plus GStreamer and plugin-internal traces. For debugging plugin
  /// behavior.
  [[nodiscard]] static VerboseOptions debug_plugins() {
    VerboseOptions opt = production();
    opt.gstreamer = true;
    opt.plugins = true;
    return opt;
  }

  /// Preset: every topic enabled. For deep diagnostic work; very chatty.
  [[nodiscard]] static VerboseOptions debug_all() {
    VerboseOptions opt;
    opt.level = VerbosityLevel::Verbose;
    opt.progress = true;
    opt.progress_force = true;
    opt.gstreamer = true;
    opt.planner = true;
    opt.graph = true;
    opt.pipeline = true;
    opt.inputstream = true;
    opt.tensor = true;
    opt.plugins = true;
    return opt;
  }
};

/**
 * @brief Options for `Graph::run_rtsp()` — controls the RTSP server's mount point and ports.
 * @ingroup pipeline
 */
struct RtspServerOptions {
  std::string mount =
      "image";     ///< RTSP path component (e.g., `"image"` → `rtsp://host:port/image`).
  int port = 8554; ///< RTSP server TCP port.
  /**
   * @brief Optional RTP/RTCP UDP port range.
   *
   * When set, the RTSP server will only allocate RTP/RTCP ports within
   * `[rtp_port_base, rtp_port_base + rtp_port_count - 1]`. Useful when firewall rules require
   * a fixed port range. Leave at the defaults (`-1`, `0`) for unrestricted port allocation.
   */
  int rtp_port_base = -1; ///< First UDP port for RTP/RTCP allocation; `-1` = no restriction.
  int rtp_port_count = 0; ///< Size of the RTP/RTCP port range; `0` = no restriction.
};

/**
 * @brief Options for `Graph::validate()`.
 *
 * Validation runs structural contracts; with `parse_launch=true` it also asks GStreamer to
 * parse the generated pipeline string (catches plugin-availability issues). With
 * `enforce_names=true` it confirms no unnamed/foreign elements snuck in via raw `custom()`.
 * @ingroup diagnostics
 */
struct ValidateOptions {
  bool parse_launch = true;  ///< Build the GStreamer pipeline string and verify element naming.
  bool enforce_names = true; ///< Reject pipelines containing unnamed or non-NEAT-named elements.
};

/**
 * @brief Internal timing mode a `Run` operates in.
 *
 * Public Graph users should not pass this into `Graph::build(...)`: use `Graph::build(...)` for
 * reusable push/pull runners and `Graph::run(...)` for one-shot execution. The runtime still keeps
 * this enum to select internal optimized paths.
 * @see Graph::build, Graph::run
 * @ingroup pipeline
 */
enum class RunMode {
  Async, ///< Continuous pipeline; user pushes/pulls asynchronously.
  Sync,  ///< One frame in, one result out, synchronously.
};

/**
 * @brief Simple process-CVU backend placement for model pre/post stages.
 *
 * Use this when a model needs the pre-MLA CVU work and post-MLA CVU work on
 * different devices. Accepted tokens are `"AUTO"`, `"EV74"`, and `"A65"`
 * (case-insensitive, with compatibility aliases handled by the internal
 * resolver).
 *
 * Example:
 *
 * ```cpp
 * Model::Options opt;
 * opt.processcvu.pre_run_target = "EV74";
 * opt.processcvu.post_run_target = "A65";
 * ```
 */
struct ProcessCvuOptions {
  std::string pre_run_target = "AUTO";
  std::string post_run_target = "AUTO";

  /// Enable the prepared safe async processcvu submit path for model-managed
  /// CVU stages. Default true keeps multi-stage model routes throughput-first;
  /// set false, or use the plugin/env kill switches, to force the synchronous
  /// fallback.
  bool async = true;
};

/**
 * @brief process-MLA execution options.
 */
struct ProcessMlaOptions {
  /// Enable the safe async processmla submit/emit path. Default true uses the
  /// optimized prepared async lane when the plugin/stage is eligible; set false,
  /// or use the plugin/env kill switches, to force the synchronous fallback.
  bool async = true;

  /// Optional processmla output pool size override. A value <= 0 leaves the
  /// runtime default in place. The framework runtime default is 4, matching
  /// model-managed CVU/MLA buffering and avoiding artificial backpressure while
  /// downstream stages still hold previous tensor-set outputs.
  int output_pool_buffers = 0;

  /// For prepared MLASHM outputs, skip the immediate producer-side CPU
  /// invalidate and stamp the output metadata as device-produced/cpu-dirty.
  /// The framework runtime default is enabled so MLA->CVU/postprocess routes
  /// pay the invalidate only at the actual CPU consumer boundary. Manual
  /// low-level pipelines that expose raw MLA outputs to legacy CPU readers can
  /// still override the element property to false.
  bool defer_output_invalidate = true;
};

/**
 * @brief Experimental prepared-route runner options.
 *
 * Default mode is empty/"passthrough" and leaves the normal per-plugin
 * pipeline untouched. `mode="dequant"` replaces eligible model-managed
 * graph223 postprocess CVU stages with `neatpreparedrunner mode=dequant`;
 * `mode="route"` is reserved for the full quant->MLA->dequant fused
 * runner once all executor bodies are enabled.
 */
struct PreparedRunnerOptions {
  std::string mode;
  int ring_depth = 0;
  bool profile = false;
  /**
   * @brief Optional prepared-runner graph223/dequant optimization flags.
   *
   * These are graph dequantize metadata tokens such as
   * `"fused,half,zpfold,bitmagic"`. Empty keeps model/runtime defaults.
   * This is the model/graph option equivalent of the old diagnostic env
   * controls and is only consumed by the opt-in prepared runner.
   */
  std::string dequant_flags;
};

/**
 * @brief Intent-named, jargon-free execution controls (preferred over the raw legacy fields).
 *
 * Bundles the scattered `processcvu`/`processmla`/`prepared_runner`/`async_queue_depth` knobs
 * behind one optional-valued object. Every field is `std::optional`: an unset field changes
 * nothing (so the default is a complete no-op), and a set field is applied **unconditionally**
 * by `GraphOptions::resolve_advanced_execution()` — an explicit `false` overrides a truthy
 * legacy default (unlike the legacy truthy-OR merge). Targets use the resolver's native tokens
 * (`"AUTO"`/`"A65"`/`"EV74"` and documented aliases). This is the only execution surface bound
 * into Python (`pyneat`); the raw legacy fields below stay for C++/ABI/back-compat.
 * @ingroup pipeline
 */
struct AdvancedExecutionOptions {
  /// Backend for model-managed process-CVU pre stages. Unsupported explicit placement is an
  /// error; for example, native Preproc is EV74-only.
  std::optional<std::string> preprocess_target; ///< -> processcvu.pre_run_target.
  /// Backend for model-managed process-CVU post adapters. This does not relocate BoxDecode,
  /// which executes on A65.
  std::optional<std::string> postprocess_target;        ///< -> processcvu.post_run_target.
  std::optional<bool> preprocess_async;                 ///< -> processcvu.async.
  std::optional<bool> inference_async;                  ///< -> processmla.async.
  std::optional<int> inference_output_buffers;          ///< -> processmla.output_pool_buffers.
  std::optional<bool> defer_output_cache_sync;          ///< -> processmla.defer_output_invalidate.
  std::optional<PreparedRunnerOptions> prepared_runner; ///< -> prepared_runner (whole-object).
  std::optional<int> internal_queue_depth;              ///< -> async_queue_depth.

  /// Overlay every set (has_value) field from @p other onto this (other wins). Used to layer
  /// a route-level object over a model-level one before resolution.
  void overlay(const AdvancedExecutionOptions& other) {
    if (other.preprocess_target)
      preprocess_target = other.preprocess_target;
    if (other.postprocess_target)
      postprocess_target = other.postprocess_target;
    if (other.preprocess_async)
      preprocess_async = other.preprocess_async;
    if (other.inference_async)
      inference_async = other.inference_async;
    if (other.inference_output_buffers)
      inference_output_buffers = other.inference_output_buffers;
    if (other.defer_output_cache_sync)
      defer_output_cache_sync = other.defer_output_cache_sync;
    if (other.prepared_runner)
      prepared_runner = other.prepared_runner;
    if (other.internal_queue_depth)
      internal_queue_depth = other.internal_queue_depth;
  }
};

/**
 * @brief Per-Graph construction options. Passed to `Graph(opt)`.
 *
 * Most fields default to sensible values. Set `element_name_prefix`/`element_name_suffix`
 * when running multiple Graphs in one process to avoid GStreamer element-name collisions.
 * @ingroup pipeline
 */
struct GraphOptions {
  int callback_timeout_ms = 1000; ///< Maximum time a user callback (e.g., `set_tensor_callback`)
                                  ///< may take before the framework intervenes.
  /// Prefix prepended to every generated GStreamer element name (sanitized to valid characters).
  std::string element_name_prefix;
  /// Suffix appended to every generated GStreamer element name (sanitized to valid characters).
  std::string element_name_suffix;
  /// Diagnostic verbosity for this Graph.
  VerboseOptions verbose;
  /**
   * @brief Requested backend for model-managed `processcvu` generic-EV stages.
   *
   * `"AUTO"` (default) lets core resolve per-stage, preferring A65 when a generic-EV graph
   * has A65 support and otherwise the EV74 path. Set explicitly to `"EV74"` or `"A65"` to
   * force a backend.
   */
  std::string processcvu_requested_run_target = "AUTO";

  /// Simple pre/post process-CVU placement. These values take priority over
  /// the legacy coarse `processcvu_requested_run_target` when non-AUTO.
  ProcessCvuOptions processcvu;

  /// MLA stage execution options.
  ProcessMlaOptions processmla;

  /// Experimental prepared-route runner. Defaults off.
  PreparedRunnerOptions prepared_runner;

  /// Depth for internally inserted async queue2 elements. 0 keeps the legacy
  /// default/diagnostic environment fallback; positive values are used as-is
  /// and are the preferred production control. For fused realtime source fan-in,
  /// 0 preserves the single-chain consumer path and a positive value inserts
  /// bounded, non-leaky queues before CVU, MLA, and decode stages (never before
  /// the terminal Output).
  int async_queue_depth = 0;

  /// Preferred jargon-free execution surface. Folded into the legacy fields above by
  /// `resolve_advanced_execution()` (called from the Graph constructor). Default is all-unset,
  /// so it is a no-op unless the caller sets a field.
  AdvancedExecutionOptions advanced_execution;

  /// Apply any explicitly-set `advanced_execution` fields onto the legacy execution fields.
  /// Unconditional assignment on `has_value()` — an explicit `false`/`0`/token overrides even a
  /// truthy legacy default. No-op when nothing is set; idempotent. Call once before the options
  /// are consumed (the Graph constructor does this) so serialized/effective config reflects intent.
  void resolve_advanced_execution() {
    if (advanced_execution.preprocess_target) {
      processcvu.pre_run_target = *advanced_execution.preprocess_target;
    }
    if (advanced_execution.postprocess_target) {
      processcvu.post_run_target = *advanced_execution.postprocess_target;
    }
    if (advanced_execution.preprocess_async) {
      processcvu.async = *advanced_execution.preprocess_async;
    }
    if (advanced_execution.inference_async) {
      processmla.async = *advanced_execution.inference_async;
    }
    if (advanced_execution.inference_output_buffers) {
      processmla.output_pool_buffers = *advanced_execution.inference_output_buffers;
    }
    if (advanced_execution.defer_output_cache_sync) {
      processmla.defer_output_invalidate = *advanced_execution.defer_output_cache_sync;
    }
    if (advanced_execution.prepared_runner) {
      prepared_runner = *advanced_execution.prepared_runner;
    }
    if (advanced_execution.internal_queue_depth) {
      async_queue_depth = *advanced_execution.internal_queue_depth;
    }
  }
};

/**
 * @brief Options for `Graph::add_output_tensor()` — the tensor-friendly output helper.
 *
 * Specifies the target format/dtype/dimensions for the output tensor; the framework auto-
 * inserts videoconvert/videoscale/capsfilter as needed.
 * @ingroup pipeline
 */
struct OutputTensorOptions {
  FormatSpec format = FormatTag::RGB;     ///< Target pixel/data format (RGB, NV12, FP32, etc.).
  TensorDType dtype = TensorDType::UInt8; ///< Target dtype.

  int target_width = -1;  ///< Target output width in pixels (-1 = no resize).
  int target_height = -1; ///< Target output height in pixels (-1 = no resize).
  int target_fps = -1;    ///< Target output frame rate (-1 = no rate change).
};

/**
 * @brief What kind of payload a `Sample` carries.
 *
 * The framework's outputs come in three shapes depending on the model's output topology:
 * a single Tensor, a flat list of Tensors (TensorSet), or a recursive Bundle of Samples
 * (Bundle, used by multi-logical-output models).
 * @see Sample
 * @ingroup pipeline
 */
enum class SampleKind {
  Tensor,    ///< Single tensor payload (the `tensor` field is set).
  TensorSet, ///< Multiple tensors at one logical output index (the `tensors` field is set).
  Bundle,    ///< Recursive: payload is a vector of Samples (the `fields` field is set).
  Unknown,   ///< Default/uninitialized.
};

/**
 * @brief Result status of `Run::pull()`.
 * @ingroup pipeline
 */
enum class PullStatus {
  Ok,      ///< A sample is available in the output parameter.
  Timeout, ///< The wait elapsed without a sample arriving.
  Closed,  ///< The pipeline has reached EOS; no more samples will come.
  Error,   ///< A runtime error occurred; check the optional `PullError`.
};

/**
 * @brief Structured error returned by `Run::pull()` when status is `Error`.
 *
 * The `code` field is a machine-triage value from `pipeline/ErrorCodes.h`. The `message` is a
 * human-readable string (typically prefixed with `[code]`). The optional `report` carries the
 * full structured `GraphReport` for severe runtime/plugin failures.
 * @ingroup diagnostics
 */
struct PullError {
  std::string message; ///< Human-readable error string (often prefixed with `[code]`).
  std::string code;    ///< Canonical machine-triage code (see `pipeline/ErrorCodes.h`).
  std::optional<GraphReport> report; ///< Optional structured report for runtime/plugin failures.
};

/**
 * @brief Typed payload returned by `Run::pull()` and consumed by `Run::push()`.
 *
 * A Sample is a tagged union: depending on `kind`, exactly one of `tensor`, `tensors`, or
 * `fields` is meaningful. Includes per-buffer metadata: stream/port labels, timestamps,
 * frame IDs, and routing slot information. Use `make_tensor_sample()`, `make_image_sample()`,
 * or `make_bundle_sample()` to construct typed Samples ergonomically.
 * @ingroup pipeline
 */
struct Sample {
  SampleKind kind = SampleKind::Unknown; ///< Discriminator: which payload field is meaningful.
  bool owned =
      true; ///< If `false`, the framework holds a borrowed reference to the underlying buffer.

  std::optional<simaai::neat::Tensor> tensor; ///< Set when `kind == Tensor`.
  TensorList tensors;                         ///< Set when `kind == TensorSet`.
  std::vector<Sample> fields; ///< Set when `kind == Bundle` (recursive multi-logical-output).

  std::string
      caps_string; ///< Caps string from the source GStreamer buffer (for media-typed payloads).
  PayloadType payload_type = PayloadType::Auto; ///< Public semantic payload family.
  std::string media_type;                       ///< MIME-style media type (e.g., `"video/x-raw"`,
                                                ///< `"application/vnd.simaai.tensor"`).
  std::string payload_tag; ///< Subformat tag (e.g., `"NV12"`, `"FP32"`, `"INT8"`). Replaces
                           ///< deprecated `format`.
  /// Subformat tag for the payload.
  /// @deprecated Use `payload_tag`. Kept for transition.
  std::string format;

  int64_t frame_id = -1;    ///< Source-assigned frame ID, when carried.
  std::string stream_id;    ///< Stream identifier (multi-stream pipelines).
  std::string stream_label; ///< Human-readable stream label.
  std::string port_name;    ///< Ingress port name (multi-input models).
  /// Logical output index this sample corresponds to.
  /// @deprecated Legacy alias for `logical_output_index`.
  int output_index = -1;
  int logical_output_index = -1; ///< Logical output index this sample corresponds to.
  int memory_index = -1;    ///< Underlying memory segment index (advanced; for zero-copy routing).
  int route_slot = -1;      ///< Route-graph slot identifier (advanced).
  std::string segment_name; ///< Memory segment name (advanced).
  int64_t input_seq = -1;   ///< Input sequence number assigned at push time (lets pull match push).
  int64_t orig_input_seq =
      -1;                   ///< Original input sequence (when re-numbered through a sub-pipeline).
  int64_t pts_ns = -1;      ///< Presentation timestamp in nanoseconds (-1 if absent).
  int64_t dts_ns = -1;      ///< Decoding timestamp in nanoseconds (-1 if absent).
  int64_t duration_ns = -1; ///< Sample duration in nanoseconds (-1 if absent).

  /// Returns true when this Sample carries no payload.
  bool empty() const noexcept {
    return kind == SampleKind::Unknown && !tensor.has_value() && tensors.empty() && fields.empty();
  }

  /// Number of logical payloads represented by this Sample.
  ///
  /// A Bundle reports its number of fields. Any non-empty non-Bundle Sample reports 1.
  /// This intentionally lets Sample replace the former "list of Samples" surface while
  /// keeping a single recursive public payload type.
  std::size_t size() const noexcept {
    if (kind == SampleKind::Bundle) {
      return fields.size();
    }
    return empty() ? 0U : 1U;
  }

  /// Reserve Bundle field storage. Turns an empty Sample into a Bundle builder.
  void reserve(std::size_t n) {
    if (kind == SampleKind::Unknown && empty()) {
      kind = SampleKind::Bundle;
    }
    fields.reserve(n);
  }

  /// Append a field to this Sample. Turns an empty Sample into a Bundle builder.
  void push_back(Sample sample) {
    if (kind == SampleKind::Unknown && empty()) {
      kind = SampleKind::Bundle;
    }
    fields.push_back(std::move(sample));
    if (kind == SampleKind::Unknown) {
      kind = SampleKind::Bundle;
    }
  }

  /// First logical payload. For non-Bundle Samples, this is the Sample itself.
  Sample& front() {
    if (kind == SampleKind::Bundle) {
      return fields.front();
    }
    return *this;
  }
  const Sample& front() const {
    if (kind == SampleKind::Bundle) {
      return fields.front();
    }
    return *this;
  }

  /// Last logical payload. For non-Bundle Samples, this is the Sample itself.
  Sample& back() {
    if (kind == SampleKind::Bundle) {
      return fields.back();
    }
    return *this;
  }
  const Sample& back() const {
    if (kind == SampleKind::Bundle) {
      return fields.back();
    }
    return *this;
  }

  /// Logical field access. For non-Bundle Samples only index 0 is meaningful.
  Sample& operator[](std::size_t i) {
    if (kind == SampleKind::Bundle) {
      return fields[i];
    }
    return *this;
  }
  const Sample& operator[](std::size_t i) const {
    if (kind == SampleKind::Bundle) {
      return fields[i];
    }
    return *this;
  }

  class iterator {
  public:
    using difference_type = std::ptrdiff_t;
    using value_type = Sample;
    using pointer = Sample*;
    using reference = Sample&;
    using iterator_category = std::forward_iterator_tag;

    iterator(Sample* owner, std::size_t index) : owner_(owner), index_(index) {}
    reference operator*() const {
      return owner_->kind == SampleKind::Bundle ? owner_->fields[index_] : *owner_;
    }
    pointer operator->() const {
      return &(**this);
    }
    iterator& operator++() {
      ++index_;
      return *this;
    }
    bool operator==(const iterator& other) const {
      return owner_ == other.owner_ && index_ == other.index_;
    }
    bool operator!=(const iterator& other) const {
      return !(*this == other);
    }

  private:
    Sample* owner_ = nullptr;
    std::size_t index_ = 0;
  };

  class const_iterator {
  public:
    using difference_type = std::ptrdiff_t;
    using value_type = Sample;
    using pointer = const Sample*;
    using reference = const Sample&;
    using iterator_category = std::forward_iterator_tag;

    const_iterator(const Sample* owner, std::size_t index) : owner_(owner), index_(index) {}
    reference operator*() const {
      return owner_->kind == SampleKind::Bundle ? owner_->fields[index_] : *owner_;
    }
    pointer operator->() const {
      return &(**this);
    }
    const_iterator& operator++() {
      ++index_;
      return *this;
    }
    bool operator==(const const_iterator& other) const {
      return owner_ == other.owner_ && index_ == other.index_;
    }
    bool operator!=(const const_iterator& other) const {
      return !(*this == other);
    }

  private:
    const Sample* owner_ = nullptr;
    std::size_t index_ = 0;
  };

  iterator begin() {
    return iterator(this, 0);
  }
  iterator end() {
    return iterator(this, size());
  }
  const_iterator begin() const {
    return const_iterator(this, 0);
  }
  const_iterator end() const {
    return const_iterator(this, size());
  }
  const_iterator cbegin() const {
    return begin();
  }
  const_iterator cend() const {
    return end();
  }

#if defined(SIMA_WITH_OPENCV)
  static const char* image_format_string(ImageSpec::PixelFormat fmt) {
    switch (fmt) {
    case ImageSpec::PixelFormat::RGB:
      return "RGB";
    case ImageSpec::PixelFormat::BGR:
      return "BGR";
    case ImageSpec::PixelFormat::GRAY8:
      return "GRAY8";
    case ImageSpec::PixelFormat::NV12:
      return "NV12";
    case ImageSpec::PixelFormat::I420:
      return "I420";
    case ImageSpec::PixelFormat::UNKNOWN:
    default:
      return "UNKNOWN";
    }
  }

  static Sample from_image(const cv::Mat& image,
                           ImageSpec::PixelFormat fmt = ImageSpec::PixelFormat::BGR,
                           bool read_only = true) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    Tensor tensor = Tensor::from_cv_mat(image, fmt, read_only);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
    if (read_only && tensor.storage &&
        tensor.storage->kind == simaai::neat::StorageKind::CpuExternal) {
      tensor = tensor.clone();
      tensor.read_only = false;
    }
    Sample out;
    out.kind = SampleKind::TensorSet;
    out.tensors = TensorList{std::move(tensor)};
    out.payload_type = PayloadType::Image;
    out.media_type = "video/x-raw";
    out.format = image_format_string(fmt);
    out.payload_tag = out.format;
    return out;
  }

  static Sample from_image(const cv::Mat& image, ImageSpec::PixelFormat fmt, TensorMemory memory) {
    Tensor tensor = Tensor::from_cv_mat(image, fmt, memory);
    Sample out;
    out.kind = SampleKind::TensorSet;
    out.tensors = TensorList{std::move(tensor)};
    out.payload_type = PayloadType::Image;
    out.media_type = "video/x-raw";
    out.format = image_format_string(fmt);
    out.payload_tag = out.format;
    return out;
  }
#endif
};

/// Return a Sample's standardized payload family, falling back from legacy media_type.
inline PayloadType sample_payload_type(const Sample& sample) {
  if (sample.payload_type != PayloadType::Auto) {
    return sample.payload_type;
  }
  const PayloadType from_media = payload_type_from_media_type(sample.media_type);
  if (from_media != PayloadType::Auto) {
    return from_media;
  }
  if (sample.kind == SampleKind::Tensor && sample.tensor.has_value()) {
    if (sample.tensor->semantic.encoded.has_value() ||
        sample.tensor->semantic.byte_stream.has_value()) {
      return PayloadType::Encoded;
    }
    if (sample.tensor->semantic.image.has_value()) {
      return PayloadType::Image;
    }
    return PayloadType::Tensor;
  }
  if (sample.kind == SampleKind::TensorSet && !sample.tensors.empty()) {
    const Tensor& first = sample.tensors.front();
    if (first.semantic.encoded.has_value() || first.semantic.byte_stream.has_value()) {
      return PayloadType::Encoded;
    }
    const bool all_image =
        std::all_of(sample.tensors.begin(), sample.tensors.end(),
                    [](const Tensor& tensor) { return tensor.semantic.image.has_value(); });
    return all_image ? PayloadType::Image : PayloadType::Tensor;
  }
  return PayloadType::Auto;
}

/// Return the internal media type corresponding to a Sample's payload family.
inline std::string sample_media_type(const Sample& sample) {
  if (!sample.media_type.empty()) {
    return sample.media_type;
  }
  return media_type_from_payload_type(sample_payload_type(sample));
}

/// Construct a `TensorSet`-kind Sample wrapping a single `Tensor` for the named port.
inline Sample make_tensor_sample(const std::string& port_name, simaai::neat::Tensor tensor) {
  Sample out;
  out.kind = SampleKind::TensorSet;
  out.stream_label = port_name;
  out.tensors = TensorList{std::move(tensor)};
  out.payload_type = sample_payload_type(out);
  return out;
}

#if defined(SIMA_WITH_OPENCV)
/// Construct a `TensorSet`-kind Sample wrapping an OpenCV `cv::Mat` image.
inline Sample make_image_sample(const cv::Mat& image,
                                ImageSpec::PixelFormat fmt = ImageSpec::PixelFormat::BGR,
                                bool read_only = true) {
  return Sample::from_image(image, fmt, read_only);
}

/// Construct an image Sample whose Tensor payload is created in the requested memory.
inline Sample make_image_sample(const cv::Mat& image, ImageSpec::PixelFormat fmt,
                                TensorMemory memory) {
  return Sample::from_image(image, fmt, memory);
}
#endif

/// Construct a `Bundle`-kind Sample whose payload is the given list of inner Samples.
inline Sample make_bundle_sample(std::initializer_list<Sample> fields) {
  Sample out;
  out.kind = SampleKind::Bundle;
  out.fields = fields;
  return out;
}

/// Returns `true` if the sample carries multiple outputs (Bundle, or TensorSet with size > 1).
inline bool sample_is_multi_output(const Sample& sample) {
  return sample.kind == SampleKind::Bundle ||
         (sample.kind == SampleKind::TensorSet && sample.tensors.size() > 1U);
}

/// Returns `true` if the sample is a TensorSet with at least one tensor.
inline bool sample_has_tensor_list(const Sample& sample) {
  return sample.kind == SampleKind::TensorSet && !sample.tensors.empty();
}

/// Mutable reference to the sample's tensor list. Throws `NeatError` if `kind != TensorSet`.
TensorList& sample_tensor_list(Sample& sample, const char* where = nullptr);
/// Const reference to the sample's tensor list. Throws `NeatError` if `kind != TensorSet`.
const TensorList& sample_tensor_list(const Sample& sample, const char* where = nullptr);
/// Returns the sample's single Tensor; throws if the sample carries 0 or >1 tensors.
Tensor& require_single_tensor(Sample& sample, const char* where = nullptr);
/// Const variant of `require_single_tensor`.
const Tensor& require_single_tensor(const Sample& sample, const char* where = nullptr);
/// Extract all tensors from any sample shape (Tensor, TensorSet, Bundle); flattens recursively.
TensorList tensors_from_sample(const Sample& sample, bool require_nonempty = true);
/// Construct a TensorSet sample from a list of tensors.
Sample sample_from_tensors(const TensorList& tensors);

} // namespace simaai::neat

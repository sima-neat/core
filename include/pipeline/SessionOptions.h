/**
 * @file
 * @ingroup pipeline
 * @brief Session/Run options, the `Sample` type, pull-status enums, and verbosity controls.
 *
 * Defines the option structs the framework consumes at construction/build time and the
 * `Sample` type that flows out of `Run::pull()`. Key types here:
 *   - `VerboseOptions` / `VerbosityLevel` — diagnostic verbosity controls (per topic).
 *   - `SessionOptions` — per-Session knobs (callback timeout, naming, processor preference).
 *   - `RtspServerOptions` / `ValidateOptions` / `OutputTensorOptions` — option packs for specific calls.
 *   - `RunMode` — Async vs Sync timing mode.
 *   - `Sample` / `SampleKind` — the typed payload `pull()` returns; can be a Tensor, a
 *     TensorSet (multiple physical outputs), or a Bundle (recursive multi-logical-output).
 *   - `PullStatus` / `PullError` — structured pull results.
 *
 * @see Session, Run, Tensor
 */
#pragma once

#include "pipeline/SessionReport.h"
#include "pipeline/FormatSpec.h"
#include "pipeline/TensorTypes.h"
#include "pipeline/Tensor.h"
#include "pipeline/TensorCore.h"

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cv {
class Mat;
}

namespace simaai::neat {

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

  /// Emit clean lifecycle/progress updates such as "Model loaded" and "Building session...".
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

  /// Preset: production messages plus GStreamer and plugin-internal traces. For debugging plugin behavior.
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
 * @brief Options for `Session::run_rtsp()` — controls the RTSP server's mount point and ports.
 * @ingroup pipeline
 */
struct RtspServerOptions {
  std::string mount = "image"; ///< RTSP path component (e.g., `"image"` → `rtsp://host:port/image`).
  int port = 8554;             ///< RTSP server TCP port.
  /**
   * @brief Optional RTP/RTCP UDP port range.
   *
   * When set, the RTSP server will only allocate RTP/RTCP ports within
   * `[rtp_port_base, rtp_port_base + rtp_port_count - 1]`. Useful when firewall rules require
   * a fixed port range. Leave at the defaults (`-1`, `0`) for unrestricted port allocation.
   */
  int rtp_port_base = -1;  ///< First UDP port for RTP/RTCP allocation; `-1` = no restriction.
  int rtp_port_count = 0;  ///< Size of the RTP/RTCP port range; `0` = no restriction.
};

/**
 * @brief Options for `Session::validate()`.
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
 * @brief Timing mode a `Run` operates in.
 *
 * `Async` runs the pipeline continuously with internal worker threads; user code pushes and
 * pulls at its own pace. `Sync` runs one frame at a time on the calling thread. Choose based
 * on the input source: streaming sources → Async; one-shot/batch → Sync.
 * @see Run
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
   * This is the model/session option equivalent of the old diagnostic env
   * controls and is only consumed by the opt-in prepared runner.
   */
  std::string dequant_flags;
};

/**
 * @brief Per-Session construction options. Passed to `Session(opt)`.
 *
 * Most fields default to sensible values. Set `element_name_prefix`/`element_name_suffix`
 * when running multiple Sessions in one process to avoid GStreamer element-name collisions.
 * @ingroup pipeline
 */
struct SessionOptions {
  int callback_timeout_ms = 1000;            ///< Maximum time a user callback (e.g., `set_tensor_callback`) may take before the framework intervenes.
  /// Prefix prepended to every generated GStreamer element name (sanitized to valid characters).
  std::string element_name_prefix;
  /// Suffix appended to every generated GStreamer element name (sanitized to valid characters).
  std::string element_name_suffix;
  /// Diagnostic verbosity for this Session.
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
  /// and are the preferred production control.
  int async_queue_depth = 0;
};

/**
 * @brief Options for `Session::add_output_tensor()` — the tensor-friendly output helper.
 *
 * Specifies the target format/dtype/dimensions for the output tensor; the framework auto-
 * inserts videoconvert/videoscale/capsfilter as needed.
 * @ingroup pipeline
 */
struct OutputTensorOptions {
  FormatSpec format = FormatTag::RGB;       ///< Target pixel/data format (RGB, NV12, FP32, etc.).
  TensorDType dtype = TensorDType::UInt8;   ///< Target dtype.

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
 * full structured `SessionReport` for severe runtime/plugin failures.
 * @ingroup diagnostics
 */
struct PullError {
  std::string message;                  ///< Human-readable error string (often prefixed with `[code]`).
  std::string code;                     ///< Canonical machine-triage code (see `pipeline/ErrorCodes.h`).
  std::optional<SessionReport> report;  ///< Optional structured report for runtime/plugin failures.
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
  bool owned = true;                     ///< If `false`, the framework holds a borrowed reference to the underlying buffer.

  std::optional<simaai::neat::Tensor> tensor; ///< Set when `kind == Tensor`.
  TensorList tensors;                          ///< Set when `kind == TensorSet`.
  std::vector<Sample> fields;                  ///< Set when `kind == Bundle` (recursive multi-logical-output).

  std::string caps_string; ///< Caps string from the source GStreamer buffer (for media-typed payloads).
  std::string media_type;  ///< MIME-style media type (e.g., `"video/x-raw"`, `"application/vnd.simaai.tensor"`).
  std::string payload_tag; ///< Subformat tag (e.g., `"NV12"`, `"FP32"`, `"INT8"`). Replaces deprecated `format`.
  /// Subformat tag for the payload.
  /// @deprecated Use `payload_tag`. Kept for transition.
  std::string format;

  int64_t frame_id = -1;       ///< Source-assigned frame ID, when carried.
  std::string stream_id;       ///< Stream identifier (multi-stream pipelines).
  std::string stream_label;    ///< Human-readable stream label.
  std::string port_name;       ///< Ingress port name (multi-input models).
  /// Logical output index this sample corresponds to.
  /// @deprecated Legacy alias for `logical_output_index`.
  int output_index = -1;
  int logical_output_index = -1; ///< Logical output index this sample corresponds to.
  int memory_index = -1;       ///< Underlying memory segment index (advanced; for zero-copy routing).
  int route_slot = -1;         ///< Route-graph slot identifier (advanced).
  std::string segment_name;    ///< Memory segment name (advanced).
  int64_t input_seq = -1;      ///< Input sequence number assigned at push time (lets pull match push).
  int64_t orig_input_seq = -1; ///< Original input sequence (when re-numbered through a sub-pipeline).
  int64_t pts_ns = -1;         ///< Presentation timestamp in nanoseconds (-1 if absent).
  int64_t dts_ns = -1;         ///< Decoding timestamp in nanoseconds (-1 if absent).
  int64_t duration_ns = -1;    ///< Sample duration in nanoseconds (-1 if absent).

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
    out.media_type = "video/x-raw";
    out.format = image_format_string(fmt);
    out.payload_tag = out.format;
    return out;
  }
#endif
};

/// Convenience alias for a list of `Sample`s (multi-input/multi-output payloads).
using SampleList = std::vector<Sample>;

/// Construct a `TensorSet`-kind Sample wrapping a single `Tensor` for the named port.
inline Sample make_tensor_sample(const std::string& port_name, simaai::neat::Tensor tensor) {
  Sample out;
  out.kind = SampleKind::TensorSet;
  out.stream_label = port_name;
  out.tensors = TensorList{std::move(tensor)};
  return out;
}

#if defined(SIMA_WITH_OPENCV)
/// Construct a `TensorSet`-kind Sample wrapping an OpenCV `cv::Mat` image.
inline Sample make_image_sample(const cv::Mat& image,
                                ImageSpec::PixelFormat fmt = ImageSpec::PixelFormat::BGR,
                                bool read_only = true) {
  return Sample::from_image(image, fmt, read_only);
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

/// Mutable reference to the sample's tensor list. Throws `SessionError` if `kind != TensorSet`.
TensorList& sample_tensor_list(Sample& sample, const char* where = nullptr);
/// Const reference to the sample's tensor list. Throws `SessionError` if `kind != TensorSet`.
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

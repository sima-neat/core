/**
 * @file
 * @ingroup model
 * @brief Model — the simplified entry point for loading and running a compiled model archive on
 * Modalix.
 *
 * `Model` is the user-facing wrapper around a compiled model archive (`.tar.gz`). It loads the
 * file, extracts and validates the manifest, runs the route planner, and exposes ready-to-use
 * `Graph` fragments (preprocess, inference, postprocess) plus convenience `run()` and
 * `build()` methods that drive a one-shot inference. Internally a Model is a `Graph` wrapper:
 * the same composition, validation, and runtime machinery the Graph API exposes powers Model
 * underneath. New users start with `Model::run(input)`; advanced users compose their own
 * `Graph` from `model.graph()` plus extra Nodes.
 *
 * @see "Models" in the design deep dive (§0.7 — The main concepts)
 * @see "Graphs: the assembly contract" (§0.12) for what Model wraps
 * @see "MPK contract" (§0.16) for the inference contract embedded in model archives
 */
#pragma once

#include "model/PreprocessPlan.h"
#include "nodes/io/Input.h"
#include "pipeline/BoxDecodeType.h"
#include "pipeline/Run.h"
#include "pipeline/TensorSpec.h"

#include <cstddef>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core/mat.hpp>
#endif

namespace simaai::neat {

namespace internal {
struct ModelAccess;
} // namespace internal
class Graph;

/// Tensor specification used by the Model API.
using TensorSpec = TensorConstraint;

/**
 * @brief Loaded form of a compiled model archive; the simplified entry point to run inference on
 * Modalix.
 *
 * A `Model` owns an extracted model archive, the parsed inference contract, and the route plan
 * derived from it. Once constructed it exposes:
 *   - **`Graph` fragments** — `preprocess()`, `inference()`, `postprocess()`, `graph()` —
 *     for composing into a user-built `Graph`.
 *   - **`run(input)`** convenience methods that lazily build one internal runner on first use,
 *     then reuse it for subsequent calls. The shortest path from "I have a tensor" to
 *     "here are detections."
 *   - **`build(...)`** that returns a long-lived `Runner` for streaming use cases (push many
 *     inputs over time, pull results asynchronously).
 *   - **Introspection** (`input_spec()`, `output_spec()`, `info()`, `metadata()`) so application
 *     code can ask the Model what shape/dtype/topology it expects and produces.
 *
 * @code
 *   sima::Model model("/models/yolov8.tar.gz");
 *   auto result = model.run(input_tensor);   // shortest path
 * @endcode
 *
 * For applications that need more control (custom pre/post nodes, multiple cameras, RTSP
 * server output), graduate from `Model::run()` to composing a `Graph` that includes
 * `model.graph()` plus your own input/output nodes.
 *
 * @see Graph
 * @see "Model is a Graph in disguise" (§0.12 of the design deep dive)
 * @ingroup model
 */
class Model {
public:
  /**
   * @brief Diagnostic snapshot of how the route planner resolved the model.
   *
   * Returned by `info()`. Aggregates the planner's needs (what the model demands),
   * capabilities (what the MPK contract provides), selection (what got included in the route),
   * and the output topology (physical vs logical outputs, packed or split).
   */
  struct ModelInfo {
    /// What the model fundamentally needs in its pre/post chain (derived from manifest dtypes).
    struct RouteNeeds {
      bool pre_quantization =
          false; ///< MLA expects INT8 input; FP32→INT8 conversion required somewhere.
      bool pre_tessellation =
          false; ///< MLA expects tessellated layout; row-major→tile transform required.
      bool pre_cast = false; ///< Floating-point dtype conversion required (e.g., FP32→BF16).
      bool post_detessellation =
          false; ///< MLA produces tessellated output; tile→row-major transform required.
      bool post_dequantization =
          false;              ///< MLA produces INT8 output; INT8→FP32 conversion required.
      bool post_cast = false; ///< Output dtype conversion required (e.g., BF16→FP32).
    };

    /// What pre/post adapters the MPK contract provides (read from manifest stages).
    struct RouteCapabilities {
      bool has_pre_quantization =
          false; ///< MPK contract provides a preprocess quantize (FP→INT8) stage.
      bool has_pre_tessellation =
          false; ///< MPK contract provides a preprocess tessellate (tile-layout) stage.
      bool has_pre_cast =
          false; ///< MPK contract provides a preprocess cast (FP dtype convert) stage.
      bool has_post_detessellation =
          false; ///< MPK contract provides a post detessellate (tile→row-major) stage.
      bool has_post_dequantization =
          false;                  ///< MPK contract provides a post dequantize (INT8→FP) stage.
      bool has_post_cast = false; ///< MPK contract provides a post cast (FP dtype convert) stage.
      bool has_post_boxdecode =
          false; ///< Manifest includes a fused detection-decode stage (YOLO BoxDecode).
    };

    /// What the planner actually included in the materialized route.
    struct RouteSelection {
      bool include_preprocess_stage = true;  ///< True if a preprocess Graph fragment is attached.
      bool include_postprocess_stage = true; ///< True if a postprocess Graph fragment is attached.
      bool infer_only = false;        ///< True if only the MLA inference stage runs (no pre/post).
      std::string preprocess_graph;   ///< Name of the chosen preprocess CVU graph (e.g., `preproc`,
                                      ///< `quanttess`).
      std::string selected_post_kind; ///< Name of the chosen post stage (e.g., `detessdequant`,
                                      ///< `boxdecode`).
    };

    /// Output topology: how many tensors the model emits, and whether they're physically packed.
    struct OutputTopology {
      std::size_t physical_outputs = 0U; ///< Number of distinct memory buffers emitted.
      std::size_t logical_outputs =
          0U; ///< Number of user-facing output tensors (may exceed physical when packed).
      bool packed_outputs = false; ///< True if logical outputs share underlying memory buffers.
    };

    std::string mpk_json_path; ///< Filesystem path to the MPK contract JSON inside the extracted
                               ///< model archive.
    std::string model_name;    ///< Human-readable model name (from manifest, when present).

    RouteNeeds needs;               ///< @copydoc RouteNeeds
    RouteCapabilities capabilities; ///< @copydoc RouteCapabilities
    RouteSelection selection;       ///< @copydoc RouteSelection
    OutputTopology output_topology; ///< @copydoc OutputTopology

    std::vector<std::string> pre_kernels;  ///< Names of pre-MLA kernels in the route, in order.
    std::vector<std::string> post_kernels; ///< Names of post-MLA kernels in the route, in order.
    std::vector<std::string>
        warnings; ///< Non-fatal planner warnings (e.g., missing optional adapters).
  };

  /**
   * @brief Where the inference pipeline should terminate.
   *
   * Most pipelines run preprocess → MLA → postprocess. This policy lets advanced users stop
   * earlier — for example, to inspect the raw MLA output before postprocessing, or to chain
   * two models where the first one only needs to run through the MLA stage.
   */
  struct InferenceTerminalPolicy {
    bool mla_only = false; ///< If true, terminate at the MLA stage; skip postprocess entirely.
    std::optional<std::size_t>
        last_stage_index; ///< Stop after this stage index (0-based, inclusive).
    std::optional<std::string>
        last_stage_name; ///< Stop after the named stage (matches MPK contract stage name).
    std::optional<std::string> last_plugin_id; ///< Stop after the first stage using this plugin ID.
    std::optional<std::string>
        last_processor; ///< Stop after the first stage running on this processor (CVU/MLA/APU).
  };

  /**
   * @brief Concrete preprocess parameters resolved by the planner from the MPK contract.
   *
   * Returned by `preprocess_requirements()`. Useful when an application needs to mirror the
   * model's preprocessing in custom code (e.g., when feeding inputs from a path the framework
   * doesn't natively support and the user wants to apply identical resize/quantize/normalize).
   */
  struct PreprocessRequirements {
    bool has_preproc_stage = false; ///< True if the route includes a preprocess stage.
    bool quant_needed = false;      ///< True if FP32→INT8 quantization is part of preprocess.
    bool tess_needed = false;       ///< True if tessellation (tile layout) is part of preprocess.
    std::string input_media_type;   ///< Expected input media type (e.g., `"video/x-raw"`,
                                    ///< `"application/vnd.simaai.tensor"`).
    std::string input_format;  ///< Expected input format token (e.g., `"NV12"`, `"RGB"`, `"FP32"`).
    std::string output_format; ///< Format produced after preprocess (handed to MLA).
    std::string output_dtype;  ///< Dtype produced after preprocess (e.g., `"INT8"`, `"BF16"`).
    /// Axis permutation applied by `layout_convert`, if any (empty if no permutation).
    std::vector<int> axis_perm;
    std::vector<int> output_shape; ///< Output tensor shape after preprocess.
    std::vector<int> slice_shape;  ///< Tile geometry when `tess_needed` is true; empty otherwise.
    std::optional<double> q_scale; ///< Per-tensor quant scale when `quant_needed` is true
                                   ///< (per-channel scales live in manifest).
    std::optional<std::int64_t> q_zp; ///< Per-tensor quant zero-point.
  };

  /**
   * @brief User-provided options at Model construction time.
   *
   * Most fields have sensible defaults; the most commonly-customized are `preprocess` (image
   * preprocessing parameters: resize target, color format, normalization mean/std) and the
   * `decode_type` family (when the MPK contract describes a YOLO-style detection model). Setting
   * these upgrades the framework's basic dtype-bridge stages into fused Generic Preproc / BoxDecode
   * kernels — same kernel slot, more work done in one pass.
   *
   * @see "The dtype contract" (§0.5 of the design deep dive) for when adapters get attached
   * @see PreprocessOptions for the preprocess substructure
   * @see BoxDecodeType for the supported detection topologies
   */
  struct Options {
    /// Intent-based preprocess interface (resize, color-convert, normalize, quantize, tessellate).
    PreprocessOptions preprocess;

    // ── Postprocessing / detection ─────────────────────────────────────────────────────────
    /**
     * @brief Detection topology to decode. Required for detection models with a BoxDecode stage.
     *
     * `Unspecified` is the unset sentinel; if the route requires BoxDecode and this stays
     * `Unspecified`, the build fails fast with an actionable error. Set to one of the
     * supported topologies (YoloV5, YoloV8, DETR, EffDet, …) when loading a detection model.
     */
    BoxDecodeType decode_type = BoxDecodeType::Unspecified;
    float score_threshold = 0.0f; ///< BoxDecode score threshold; 0 keeps all candidates.
    float nms_iou_threshold =
        0.0f;      ///< BoxDecode IoU threshold for non-max suppression; 0 disables NMS.
    int top_k = 0; ///< BoxDecode top-K cap; 0 means no cap.
    /// Number of classes produced by detection heads. Set this for detection MPKs whose raw
    /// class-head depth cannot be inferred reliably (for example single-class YOLO split heads).
    /// `0` keeps legacy inference / MPK-provided metadata.
    int num_classes = 0;
    /// Original-image width hint for BoxDecode coordinate inversion.
    /// @deprecated BoxDecode original image size is now read from preprocess metadata. Kept for
    /// transition.
    int boxdecode_original_width = 0;
    /// Original-image height hint for BoxDecode coordinate inversion.
    /// @deprecated BoxDecode original image size is now read from preprocess metadata. Kept for
    /// transition.
    int boxdecode_original_height = 0;

    // ── Naming / wiring ────────────────────────────────────────────────────────────────────
    /// Name of the upstream Node feeding preprocess input (default: `"decoder"` — i.e., a video
    /// decoder).
    std::string upstream_name = "decoder";
    /// Optional suffix appended to every generated GStreamer element name (use to disambiguate
    /// concurrent Graphs).
    std::string name_suffix;

    // ── Extraction lifecycle ───────────────────────────────────────────────────────────────
    /**
     * @brief Whether to clean up the on-disk extracted model-archive directory on process exit.
     *
     * - `true` (default): per-process extracted model-archive data is removed on normal exit.
     * - `false`: extracted data stays on disk after the process ends (useful for inspection/reuse).
     */
    bool cleanup_extracted_model_data = true;

    /// Diagnostic verbosity for model construction and route planning.
    VerboseOptions verbose;

    /// Optional override for where the inference pipeline should terminate.
    InferenceTerminalPolicy inference_terminal;

    /// Simple placement for model-managed `processcvu` stages.
    /// Example: `processcvu.pre_run_target = "EV74";`
    /// and `processcvu.post_run_target = "A65";`.
    ProcessCvuOptions processcvu;

    /// MLA stage execution options.
    ProcessMlaOptions processmla;

    /// Experimental prepared-route runner. Defaults off/passthrough.
    PreparedRunnerOptions prepared_runner;

    /// Depth for internally inserted async queue2 elements. 0 keeps the
    /// framework default.
    int async_queue_depth = 0;
  };

  /**
   * @brief Options for `Model::graph()` — controls how the model assembles into a Graph.
   *
   * By default `model.graph()` returns a reusable model fragment with open named endpoints. Set
   * `include_input` and/or `include_output` when you intentionally want the returned Graph to
   * contain public `nodes::Input` / `nodes::Output` boundary nodes.
   */
  struct RouteOptions {
    bool include_input = false;  ///< Include a public `nodes::Input` Node at the head.
    bool include_output = false; ///< Include a public `nodes::Output` Node at the tail.
    /**
     * @brief Expose individual model output endpoints when the model route has multiple physical
     * outputs.
     *
     * The default public Graph contract is intentionally ML-friendly: a model produces one
     * aggregate output Sample, normally a TensorSet containing all model tensors. That remains true
     * for optimized routes where several logical tensors are packed into one physical buffer.
     *
     * Set this to true only when advanced/debug code needs to address truly separate physical model
     * outputs by name. If the route has a single physical output buffer, this flag is ignored and
     * the Graph still exposes one aggregate output endpoint.
     */
    bool expose_all_outputs = false;
    std::string upstream_name; ///< Element name of the upstream Node feeding the model (default:
                               ///< `"decoder"`).
    std::string name_suffix;   ///< Suffix appended to generated element names.
    std::string
        buffer_name;        ///< Optional buffer pool name (used by advanced memory configurations).
    VerboseOptions verbose; ///< Diagnostic verbosity during Graph construction.
    /**
     * @brief Requested backend for model-managed `processcvu` generic-EV stages.
     *
     * `"AUTO"` (default) resolves in core during manifest rendering and prefers A65 when the
     * stage supports it; otherwise EV74. Set to `"EV74"` or `"A65"` to force a specific backend.
     */
    std::string processcvu_requested_run_target = "AUTO";

    /// Simple placement for model-managed `processcvu` stages when this
    /// Graph/Runner is built. Non-AUTO values take priority over
    /// Model::Options and the legacy coarse target above.
    ProcessCvuOptions processcvu;

    /// MLA stage execution options. Non-default values take priority over
    /// Model::Options.
    ProcessMlaOptions processmla;

    /// Experimental prepared-route runner. Non-default values take priority
    /// over Model::Options.
    PreparedRunnerOptions prepared_runner;

    /// Depth for internally inserted async queue2 elements. 0 inherits from
    /// Model::Options/framework default.
    int async_queue_depth = 0;
  };

  /**
   * @brief Selector for `fragment(Stage)` and `backend_fragment(Stage)`.
   *
   * Lets advanced users grab one specific portion of the model's pipeline.
   */
  enum class Stage {
    Preprocess,  ///< Pre-MLA portion only (resize, color-convert, normalize, quant, tess).
    Inference,   ///< MLA inference stage only.
    Postprocess, ///< Post-MLA portion only (detess, dequant, cast, BoxDecode if applicable).
    Full         ///< Full pipeline: Preprocess + Inference + Postprocess.
  };

  /**
   * @brief Load and validate a compiled model archive at the given path with default options.
   *
   * Equivalent to `Model(model_path, Options{})`. The constructor extracts the `.tar.gz`,
   * validates the manifest (rejecting malformed or malicious archives), and runs the route
   * planner. Throws `NeatError` on any failure (with a structured `GraphReport`).
   *
   * @param model_path Filesystem path to a `.tar.gz` model archive.
   * @throws NeatError on archive validation failure, manifest parse error, or unsupported route.
   */
  explicit Model(const std::string& model_path);
  /**
   * @brief Load and validate a compiled model archive with explicit options.
   *
   * @param model_path Filesystem path to a `.tar.gz` model archive.
   * @param opt        Options controlling preprocess, postprocess decode, naming, lifecycle, and
   * verbosity.
   * @throws NeatError on archive validation failure, manifest parse error, or unsupported route.
   */
  explicit Model(const std::string& model_path, const Options& opt);

  Model(Model&&) noexcept;            ///< Move-constructible.
  Model& operator=(Model&&) noexcept; ///< Move-assignable. (Models are not copyable.)
  ~Model(); ///< Destructor; releases the impl and (optionally) cleans up extracted model-archive
            ///< data.

  // ── Stage composition ────────────────────────────────────────────────────────────────────
  /// Returns the preprocess portion of the model's pipeline as a `Graph`.
  simaai::neat::Graph preprocess() const;
  /// Returns the MLA inference portion of the model's pipeline as a `Graph`.
  simaai::neat::Graph inference() const;
  /// Returns the postprocess portion of the model's pipeline as a `Graph`.
  simaai::neat::Graph postprocess() const;
  /// Returns the model's full route as a public Graph fragment.
  simaai::neat::Graph graph() const;
  /// Returns the model's full route as a public Graph, optionally with explicit Input/Output
  /// boundary Nodes when requested by `RouteOptions`.
  simaai::neat::Graph graph(RouteOptions opt) const;

  // ── Introspection ────────────────────────────────────────────────────────────────────────
  /// Single-input convenience accessor for `input_specs().front()`.
  TensorSpec input_spec() const;
  /// Returns the expected input tensor specs (one per ingress port for multi-input models).
  std::vector<TensorSpec> input_specs() const;
  /// Single-output convenience accessor for `output_specs().front()`.
  TensorSpec output_spec() const;
  /// Returns the produced output tensor specs (multiple for multi-head models).
  std::vector<TensorSpec> output_specs() const;
  /**
   * @brief Returns the batch size the model was compiled with.
   *
   * Defaults to 1 when the model is not batched. Callers that drive multiple logical inferences
   * per push/pull cycle should use this to size their input batches. The same value is prepended
   * to `input_specs()[i].shape` when batch > 1.
   */
  int compiled_batch_size() const;
  /// Returns the full resolved preprocess plan (intent-driven; richer than
  /// `preprocess_requirements()`).
  ResolvedPreprocessPlan resolved_preprocess_plan() const;
  /// Returns concrete preprocess parameters (resize, color, quant, tess) the model needs.
  PreprocessRequirements preprocess_requirements() const;
  /// Returns the full diagnostic snapshot of the loaded model and its route.
  ModelInfo info() const;
  /// Returns free-form key/value metadata declared by the model author in the manifest.
  std::unordered_map<std::string, std::string> metadata() const;

  // ── Advanced / graph composition ─────────────────────────────────────────────────────────
  /// Returns one specific stage of the pipeline as a `Graph`.
  Graph fragment(Stage stage) const;
  /// Returns the GStreamer launch fragment for one specific stage (debugging / `Graph::custom()`
  /// use).
  std::string backend_fragment(Stage stage) const;

  /// Returns the appsrc input options the planner derived for a given input mode (tensor or media).
  simaai::neat::InputOptions input_appsrc_options(bool tensor_mode) const;
  /// Per-input variant of `input_appsrc_options` for multi-input models.
  std::vector<simaai::neat::InputOptions> input_appsrc_options_list(bool tensor_mode) const;
  /// Find the path to a per-stage config file in the extracted model archive by plugin ID.
  std::string find_config_path_by_plugin(const std::string& plugin_id) const;
  /// Find the path to a per-stage config file in the extracted model archive by processor name
  /// (CVU/MLA/APU).
  std::string find_config_path_by_processor(const std::string& processor) const;
  /// Returns the canonical output element name for the inference stage (used in pipeline string
  /// emission).
  std::string infer_output_name() const;

  /**
   * @brief Long-lived execution handle returned by `Model::build(...)`.
   *
   * Wraps an underlying `Run` plus the bookkeeping needed to convert user-friendly input types
   * (`cv::Mat`, `TensorList`, `Sample`) into the right Sample shape for the model's appsrc.
   * Use the Runner for streaming workloads; use the convenience `Model::run()` overloads for
   * one-shot inference.
   *
   * @code
   *   sima::Model model("yolo.tar.gz");
   *   auto runner = model.build();
   *   while (have_input) {
   *     runner.push(next_frame);
   *     auto sample = runner.pull(/ * timeout_ms = * / 100);
   *     if (!sample.empty()) handle(sample);
   *   }
   *   runner.close();
   * @endcode
   */
  class Runner {
  public:
    /// Construct an empty Runner; assign from a built Runner before use.
    Runner() = default;
    /// Wrap a Run handle with no special tensor/ingress configuration.
    explicit Runner(simaai::neat::Run run);
    /// Wrap a Run handle that was built for tensor-mode input with the given InputOptions.
    Runner(simaai::neat::Run run, const simaai::neat::InputOptions& tensor_input_opt);
    /// Wrap a Run handle with tensor input options and explicit ingress port names (multi-input
    /// models).
    Runner(simaai::neat::Run run, const simaai::neat::InputOptions& tensor_input_opt,
           std::vector<std::string> ingress_names);
    /// Wrap a Run handle with explicit ingress port names (multi-input, non-tensor mode).
    Runner(simaai::neat::Run run, std::vector<std::string> ingress_names);

    /// Returns `true` if the underlying Run is alive and ready to push/pull.
    explicit operator bool() const noexcept;
#if defined(SIMA_WITH_OPENCV)
    /**
     * @brief Push a vector of OpenCV Mat inputs (multi-input models accept one Mat per ingress).
     * @return `true` on success; `false` if the input queue rejected the push (per OverflowPolicy).
     */
    bool push(const std::vector<cv::Mat>& inputs);
#endif
    /// Push a list of `Tensor` inputs (one per ingress port).
    bool push(const simaai::neat::TensorList& inputs);
    /// Push a list of `Sample` inputs (full Samples carry per-buffer metadata).
    bool push(const simaai::neat::Sample& inputs);
    /**
     * @brief Pull the next produced output Sample list.
     * @param timeout_ms Maximum time to wait, in milliseconds; `-1` means wait forever; `0` is
     * non-blocking.
     * @return The next `Sample`. Empty if the timeout elapsed or the pipeline closed.
     */
    simaai::neat::Sample pull(int timeout_ms = -1);
#if defined(SIMA_WITH_OPENCV)
    /// One-shot synchronous push+pull from `cv::Mat` inputs.
    simaai::neat::TensorList run(const std::vector<cv::Mat>& inputs, int timeout_ms = -1);
#endif
    /// One-shot synchronous push+pull from `Tensor` inputs.
    simaai::neat::TensorList run(const simaai::neat::TensorList& inputs, int timeout_ms = -1);
    /// One-shot synchronous push+pull from `Sample` inputs.
    simaai::neat::Sample run(const simaai::neat::Sample& inputs, int timeout_ms = -1);

    /// Start observing caller-owned push/pull code without consuming outputs.
    simaai::neat::MeasureScope start_measurement(const simaai::neat::MeasureOptions& opt = {});
    /**
     * @brief Warm up the pipeline by running `warm` inferences before measurement begins.
     *
     * Useful for stable performance numbers — the first few inferences pay one-time setup costs
     * (kernel load, JIT compilation, cache fill). Pass `warm = -1` to use a sensible default.
     */
    int warmup(const simaai::neat::TensorList& inputs, int warm = -1, int timeout_ms = -1);
    /// Close input (sends EOS) and release the underlying Run.
    void close();
    /// End-to-end push/pull/latency stats (forwards to underlying Run).
    simaai::neat::RunStats stats() const;
    /// Latency, throughput, input stats, and optional power telemetry (forwards to Run).
    simaai::neat::RunMeasurementSummary measurement_summary() const;
    /// Unified latency/throughput/power/counter metrics (forwards to underlying Run).
    simaai::neat::RuntimeMetrics metrics(const simaai::neat::RuntimeMetricsOptions& opt = {}) const;
    /// Render unified runtime metrics (forwards to underlying Run).
    std::string metrics_report(
        const simaai::neat::RuntimeMetricsOptions& opt = {},
        simaai::neat::RuntimeMetricsFormat format = simaai::neat::RuntimeMetricsFormat::Text) const;
    /// Convenience overload for selecting the output format with default options.
    std::string metrics_report(simaai::neat::RuntimeMetricsFormat format) const;
    /// Per-stage / per-element / per-pad diagnostic snapshot (forwards to underlying Run).
    simaai::neat::RunDiagSnapshot diag_snapshot() const;
    /// Human-readable formatted report of stats + diagnostics (forwards to underlying Run).
    std::string report(const simaai::neat::RunReportOptions& opt = {}) const;
    /// Send EOS into the input queue without releasing the Run (lets pull drain in flight).
    void close_input();

  private:
    simaai::neat::Run run_{};
    std::optional<simaai::neat::InputOptions> tensor_input_opt_for_cv_{};
    std::vector<std::string> ingress_names_;
  };

private:
  static const RouteOptions& default_route_options();
  static const simaai::neat::RunOptions& default_run_options();

public:
  /**
   * @brief Build a long-lived `Runner` from this Model with default options.
   *
   * Use this for streaming workloads (push many inputs over time, pull asynchronously).
   * For one-shot inference, prefer `run()`.
   *
   * @return A `Runner` ready for push/pull.
   * @throws NeatError on validation or build failure (with structured `GraphReport`).
   */
  Runner build();
  /// Build a `Runner` with explicit route options.
  Runner build(const RouteOptions& opt);
  /// Build a `Runner` with explicit runtime options.
  Runner build(const simaai::neat::RunOptions& run_opt);
  /// Build a `Runner` with explicit route and runtime options.
  Runner build(const RouteOptions& opt, const simaai::neat::RunOptions& run_opt);
  /**
   * @brief Build a `Runner` and seed the build with sample input(s).
   *
   * Seeding the build with an input lets the planner perform build-time adaptation: tightening
   * caps to match the actual input shape, picking concrete pixel formats, validating that the
   * input is compatible. The `BuildAdaptationSummary` in the resulting `GraphReport` records
   * what adaptations were applied.
   *
   * @param inputs    One Tensor per ingress port. For single-input models pass a single-element
   * list.
   * @param opt       Graph options (defaults are sensible).
   * @param run_opt   Runtime options (queue depth, overflow policy, async/sync mode).
   */
  Runner build(const simaai::neat::TensorList& inputs,
               const RouteOptions& opt = default_route_options(),
               const simaai::neat::RunOptions& run_opt = default_run_options());
  /// Build variant that seeds with full `Sample` inputs (carrying per-buffer metadata).
  Runner build(const simaai::neat::Sample& inputs,
               const RouteOptions& opt = default_route_options(),
               const simaai::neat::RunOptions& run_opt = default_run_options());
#if defined(SIMA_WITH_OPENCV)
  /// Build variant that seeds with `cv::Mat` inputs (the OpenCV convenience path).
  Runner build(const std::vector<cv::Mat>& inputs,
               const RouteOptions& opt = default_route_options(),
               const simaai::neat::RunOptions& run_opt = default_run_options());
#endif

  // ── One-shot execution (synchronous) ─────────────────────────────────────────────────────
  /**
   * @brief One-shot-style inference for the simplest applications.
   *
   * The first call lazily builds and caches an internal Runner; later calls reuse that Runner
   * and only push/pull. Use this for unit tests, single-image inference, batch processing, or
   * Python frame loops where rebuilding every frame would be expensive.
   *
   * @param inputs     Input tensors (one per ingress port).
   * @param timeout_ms Maximum wait for the result, in milliseconds; `-1` waits forever.
   * @return The output tensor list.
   * @throws NeatError on build failure or pull timeout/error.
   */
  simaai::neat::TensorList run(const simaai::neat::TensorList& inputs, int timeout_ms = -1);
  /// One-shot inference with full `Sample` inputs.
  simaai::neat::Sample run(const simaai::neat::Sample& inputs, int timeout_ms = -1);
#if defined(SIMA_WITH_OPENCV)
  /// One-shot inference with `cv::Mat` inputs (OpenCV convenience).
  simaai::neat::TensorList run(const std::vector<cv::Mat>& inputs, int timeout_ms = -1);
#endif

private:
  friend struct internal::ModelAccess;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat

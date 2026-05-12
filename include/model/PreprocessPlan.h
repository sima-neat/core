/**
 * @file
 * @ingroup model
 * @brief Preprocess intent (what the user asked for) plus the resolved plan (what runs).
 *
 * `PreprocessOptions` is the **intent layer** — what the application says it wants
 * (resize to 640×640, normalize ImageNet-style, etc.). `ResolvedPreprocessPlan` is the
 * **plan layer** — what the framework actually compiled, including which preprocess
 * graph family was chosen (`Preproc` / `Quant` / `Tess` / `QuantTess`) and the
 * negotiated MLA-side contract. The route planner produces the latter from the former
 * plus the model's MPK manifest.
 *
 * @see Model — the type that produces these
 * @see "Input planner" (§82 of the design deep dive)
 */
#pragma once

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

/**
 * @brief Tri-state knob: let the planner decide, force on, or force off.
 *
 * Used pervasively across `PreprocessOptions` so the application can express "I don't
 * care, do the right thing" (`Auto`) without losing the ability to override.
 * @ingroup model
 */
enum class AutoFlag {
  Auto = 0, ///< Planner decides based on model contract and inputs.
  On = 1,   ///< Force this stage on.
  Off = 2,  ///< Force this stage off.
};

/// What kind of input the user is feeding the preprocess pipeline.
/// @ingroup model
enum class InputKind {
  Auto = 0,   ///< Planner infers from caps / sample type.
  Image = 1,  ///< Decoded image (planar/packed pixels).
  Tensor = 2, ///< Already-shaped tensor; preprocess is mostly a pass-through.
};

/// How the resize stage maps non-matching input dimensions to the model's input shape.
/// @ingroup model
enum class ResizeMode {
  Stretch = 0,   ///< Anisotropic scale to fit; aspect ratio not preserved.
  Letterbox = 1, ///< Scale to fit while preserving aspect; pad the rest with `pad_value`.
  Crop = 2,      ///< Center-crop to the target shape after isotropic scale.
};

/// Color format hint for the color-convert stage.
/// @ingroup model
enum class PreprocessColorFormat {
  Auto = 0,  ///< Planner picks based on caps/contract.
  RGB = 1,   ///< Packed 8-bit RGB.
  BGR = 2,   ///< Packed 8-bit BGR.
  GRAY8 = 3, ///< 8-bit single-channel grayscale.
  NV12 = 4,  ///< Y plane + interleaved UV plane (semi-planar 4:2:0).
  I420 = 5,  ///< Y, U, V planes (planar 4:2:0).
};

/// Common normalize presets — convenient shorthand for famous mean/stddev pairs.
/// @ingroup model
enum class NormalizePreset {
  None = 0,      ///< Use the explicit `mean`/`stddev` fields.
  ImageNet = 1,  ///< Standard ImageNet stats.
  COCO_YOLO = 2, ///< Stats commonly used by YOLO-family detectors trained on COCO.
};

/// Which preprocess graph family the planner selected. The four variants correspond to
/// the four corners of the dtype-contract decision tree (BF16/INT8 × inside/outside MLA tess).
/// @ingroup model
enum class PreprocessGraphFamily {
  Disabled = 0,  ///< No preprocess graph (model accepts raw tensors).
  Preproc = 1,   ///< Preprocess only; quant/tess happen inside the MLA.
  Quant = 2,     ///< Preprocess + quant; tess happens inside the MLA.
  Tess = 3,      ///< Preprocess + tess; quant happens inside the MLA.
  QuantTess = 4, ///< Preprocess + quant + tess; nothing happens inside the MLA.
};

/// Which transformation a `Transform` represents — used in user-supplied transform lists.
/// @ingroup model
enum class TransformType {
  Resize = 0,        ///< Resize / letterbox / crop.
  ColorConvert = 1,  ///< Color-format conversion.
  LayoutConvert = 2, ///< Axis permutation (e.g., HWC → CHW).
  Normalize = 3,     ///< Mean/stddev normalization.
  Quantize = 4,      ///< INT8 quantization.
  Tessellate = 5,    ///< MLA-tile reshuffle.
};

/// Resize / letterbox / crop parameters.
/// @ingroup model
struct ResizeSpec {
  AutoFlag enable = AutoFlag::Auto;        ///< On/off/auto.
  int width = 0;                           ///< Target width (px). `0` = inferred from contract.
  int height = 0;                          ///< Target height (px). `0` = inferred from contract.
  ResizeMode mode = ResizeMode::Letterbox; ///< How to handle aspect-ratio mismatch.
  int pad_value = 114; ///< Pad fill value when `mode == Letterbox` (default 114, YOLO convention).
  std::string scaling_type = "BILINEAR"; ///< Sampler kind: `"BILINEAR"`, `"NEAREST"`, etc.
};

/// Color-format conversion parameters.
/// @ingroup model
struct ColorConvertSpec {
  AutoFlag enable = AutoFlag::Auto; ///< On/off/auto.
  PreprocessColorFormat input_format =
      PreprocessColorFormat::Auto; ///< Source color format (or `Auto` to infer from caps).
  PreprocessColorFormat output_format =
      PreprocessColorFormat::Auto; ///< Target color format (or `Auto` to use the contract).
};

/// Axis-permutation parameters for HWC↔CHW-style layout conversions.
/// @ingroup model
struct LayoutConvertSpec {
  AutoFlag enable = AutoFlag::Auto; ///< On/off/auto.
  std::vector<int>
      perm; ///< Permutation indices applied to input axes (e.g., `{2,0,1}` for HWC→CHW).

  /// True iff `perm` was explicitly set.
  bool has_perm() const {
    return !perm.empty();
  }
};

/// Mean/stddev normalization parameters.
/// @ingroup model
struct NormalizeSpec {
  AutoFlag enable = AutoFlag::Auto;                 ///< On/off/auto.
  std::array<float, 3> mean = {0.0f, 0.0f, 0.0f};   ///< Per-channel mean to subtract.
  std::array<float, 3> stddev = {1.0f, 1.0f, 1.0f}; ///< Per-channel stddev to divide by.
  bool has_explicit_stats =
      false; ///< True iff the application set these explicitly (vs. preset/default).
};

/// INT8 (or other low-precision) quantization parameters.
/// @ingroup model
struct QuantizeSpec {
  AutoFlag enable = AutoFlag::Auto; ///< On/off/auto.
  int zero_point = 0;               ///< Zero-point offset (`q = round(x/scale) + zero_point`).
  double scale = 0.0;               ///< Scale factor; `0` means "use the model's calibration".
  std::string output_dtype;         ///< Target dtype name (e.g., `"int8"`, `"uint8"`).
};

/// MLA tessellation (tile-shuffle) parameters.
/// @ingroup model
struct TessellateSpec {
  AutoFlag enable = AutoFlag::Auto; ///< On/off/auto.
  std::vector<int> slice_shape;     ///< Tile shape (typically `{H, W, C}`); empty means "use the
                                    ///< contract's geometry".

  /// Replace `slice_shape` with the given shape.
  void set_slice_shape(std::vector<int> shape) {
    slice_shape = std::move(shape);
  }

  /// Read element `index` of `shape`, returning `0` if out of bounds.
  static int shape_dim(const std::vector<int>& shape, std::size_t index) {
    return shape.size() > index ? shape[index] : 0;
  }

  /// Slice height (first dim of `slice_shape`), or 0 if not set.
  int slice_height() const {
    return shape_dim(slice_shape, 0);
  }

  /// Slice width (second dim of `slice_shape`), or 0 if not set.
  int slice_width() const {
    return shape_dim(slice_shape, 1);
  }

  /// Slice channel count (last dim, when shape has rank ≥ 3); 0 otherwise.
  int slice_channels() const {
    return slice_shape.size() >= 3 ? slice_shape.back() : 0;
  }

  /// True iff `slice_shape` carries a complete `H × W × C` triple.
  bool has_slice_shape() const {
    return slice_height() > 0 && slice_width() > 0 && slice_channels() > 0;
  }
};

/**
 * @brief Per-stage flags recording which preprocess fields the user set explicitly.
 *
 * The route planner uses this to distinguish "user wants this" from "user left the
 * default", so it can choose a different compiled graph without overwriting an explicit
 * intent.
 * @ingroup model
 */
struct PreprocessExplicitKnobs {
  bool resize = false;              ///< User explicitly configured `ResizeSpec`.
  bool color_convert = false;       ///< User explicitly configured `ColorConvertSpec`.
  bool layout_convert = false;      ///< User explicitly configured `LayoutConvertSpec`.
  bool normalize = false;           ///< User explicitly enabled/disabled normalize.
  bool normalize_stats = false;     ///< User explicitly set `mean`/`stddev`.
  bool quantize_enable = false;     ///< User explicitly enabled/disabled quantize.
  bool quantize_params = false;     ///< User explicitly set `zero_point`/`scale`.
  bool tessellate_enable = false;   ///< User explicitly enabled/disabled tessellate.
  bool tessellate_geometry = false; ///< User explicitly set `slice_shape`.
};

/**
 * @brief Discriminated union of all per-stage transform specs.
 *
 * Used inside `PreprocessOptions::transforms` to express an ordered list of explicit
 * transformations (overrides the auto-planner). Read the field that corresponds to
 * `type`.
 * @ingroup model
 */
struct Transform {
  TransformType type = TransformType::Resize; ///< Which spec field is active.
  ResizeSpec resize;                          ///< Active when `type == Resize`.
  ColorConvertSpec color_convert;             ///< Active when `type == ColorConvert`.
  LayoutConvertSpec layout_convert;           ///< Active when `type == LayoutConvert`.
  NormalizeSpec normalize;                    ///< Active when `type == Normalize`.
  QuantizeSpec quantize;                      ///< Active when `type == Quantize`.
  TessellateSpec tessellate;                  ///< Active when `type == Tessellate`.
};

/**
 * @brief User-facing preprocess intent — what the application asks for.
 *
 * The planner takes this plus the model's MPK manifest and produces a
 * `ResolvedPreprocessPlan`. Most fields default to `Auto`/zero; set only what you need
 * to override.
 * @ingroup model
 */
struct PreprocessOptions {
  InputKind kind = InputKind::Auto; ///< Whether inputs are images or pre-shaped tensors.
  AutoFlag enable = AutoFlag::Auto; ///< Master switch — set `Off` to skip preprocess entirely.

  int input_max_width = 0;  ///< Upper bound on input width (px); `0` = no bound.
  int input_max_height = 0; ///< Upper bound on input height (px); `0` = no bound.
  int input_max_depth = 0;  ///< Upper bound on input channel count; `0` = no bound.

  ResizeSpec resize;                ///< Resize / letterbox / crop intent.
  ColorConvertSpec color_convert;   ///< Color-format conversion intent.
  LayoutConvertSpec layout_convert; ///< Axis-permutation intent.
  NormalizeSpec normalize;          ///< Normalize intent.
  QuantizeSpec quantize;            ///< Quantize intent.
  TessellateSpec tessellate;        ///< Tessellate intent.

  std::vector<Transform> transforms; ///< Ordered explicit transforms; if non-empty, overrides the
                                     ///< per-stage specs above.
  NormalizePreset preset =
      NormalizePreset::None; ///< If non-`None`, supplies `mean`/`stddev` defaults.
};

/**
 * @brief Per-input contract describing what the preprocess stage expects to receive.
 *
 * Several of these may exist if the model has multiple inputs. Used by the planner to
 * validate caps and by GStreamer caps negotiation.
 * @ingroup model
 */
struct PreprocessContract {
  std::string media_type; ///< Media type string (e.g., `"video/x-raw"`, `"application/x-tensor"`).
  std::string format;     ///< Format identifier (e.g., `"RGB"`, `"NV12"`).
  int width = 0;          ///< Required width (px), or `0` if unbounded.
  int height = 0;         ///< Required height (px), or `0` if unbounded.
  int depth = 0;          ///< Required channel count, or `0` if unbounded.
  int max_width = 0;      ///< Upper-bound width (px), or `0` if unbounded.
  int max_height = 0;     ///< Upper-bound height (px), or `0` if unbounded.
  int max_depth = 0;      ///< Upper-bound channel count, or `0` if unbounded.
};

/**
 * @brief Metadata-side contract — names and fields the preprocess stage requires.
 *
 * Gates the preprocess Node against missing GstSimaMeta fields at build time.
 * @ingroup model
 */
struct PreprocessMetaContract {
  std::string meta_name = "GstSimaMeta";    ///< Metadata struct name (defaults to GstSimaMeta).
  std::vector<std::string> required_fields; ///< Field names that must be present on every buffer.
};

/**
 * @brief Final resolved plan — what the framework actually compiled and runs.
 *
 * Produced by the route planner from `PreprocessOptions` + MPK manifest. Carries the
 * effective options, which graph family was chosen, the path to the compiled graph
 * config, and the negotiated input/MLA contracts.
 *
 * @see PreprocessOptions for the input intent
 * @see "Input planner" (§82 of the design deep dive)
 * @ingroup model
 */
struct ResolvedPreprocessPlan {
  PreprocessOptions requested; ///< Original options as supplied by the application.
  PreprocessOptions effective; ///< Post-planning options — defaults filled in, conflicts resolved.
  PreprocessExplicitKnobs explicit_knobs; ///< Which fields the application set explicitly.

  InputKind resolved_kind = InputKind::Image; ///< Concrete input kind chosen by the planner.
  bool transforms_override = false; ///< True iff `requested.transforms` was non-empty and used.
  bool enabled = true;              ///< False iff preprocess was disabled entirely.
  PreprocessGraphFamily graph_family =
      PreprocessGraphFamily::Preproc; ///< Which preprocess graph family was selected.
  std::string graph_kernel;           ///< Kernel name backing the graph (CVU/EV74 entry point).
  std::string graph_config_path;      ///< Filesystem path to the compiled graph config (inside the
                                      ///< unpacked MPK).

  std::vector<PreprocessContract>
      ingress_contracts; ///< Per-input ingress contracts (multi-input models have multiple).
  PreprocessContract mla_contract;      ///< Contract describing the tensor handed to the MLA.
  PreprocessMetaContract meta_contract; ///< Required GstSimaMeta fields.

  std::vector<std::string>
      warnings; ///< Non-fatal advisories (e.g., "preset overridden by explicit stats").

  /// Render a multi-line debug summary of the plan — used in `Session::describe()` and reports.
  std::string to_debug_string() const;
};

} // namespace simaai::neat

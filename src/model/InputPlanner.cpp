#include "model/internal/InputPlanner.h"

#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/InputPolicy.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/sima/MpkContract.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace simaai::neat::internal {
namespace {

using pipeline_internal::lower_copy;
using pipeline_internal::upper_copy;

bool quantize_params_explicit(const QuantizeSpec& q) {
  return q.zero_point != 0 || q.scale > 0.0 || !q.output_dtype.empty();
}

bool tessellate_geometry_explicit(const TessellateSpec& t) {
  return t.has_slice_shape();
}

bool layout_convert_explicit(const LayoutConvertSpec& spec) {
  return spec.enable != AutoFlag::Auto || spec.has_perm();
}

void validate_layout_convert_spec(const LayoutConvertSpec& spec) {
  if (spec.enable == AutoFlag::On && spec.perm.empty()) {
    throw std::invalid_argument(
        "preprocess.layout_convert.enable=On requires a non-empty preprocess.layout_convert.perm");
  }
  for (std::size_t i = 0; i < spec.perm.size(); ++i) {
    const int axis = spec.perm[i];
    if (axis < 0) {
      throw std::invalid_argument("preprocess.layout_convert.perm must contain only non-negative axis indices");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (spec.perm[j] == axis) {
        throw std::invalid_argument(
            "preprocess.layout_convert.perm must not contain duplicate axis indices");
      }
    }
  }
}

bool has_non_default_simple_preprocess(const PreprocessOptions& p) {
  if (p.preset != NormalizePreset::None)
    return true;
  if (p.resize.enable != AutoFlag::Auto || p.resize.width > 0 || p.resize.height > 0)
    return true;
  if (p.color_convert.enable != AutoFlag::Auto ||
      p.color_convert.input_format != PreprocessColorFormat::Auto ||
      p.color_convert.output_format != PreprocessColorFormat::Auto) {
    return true;
  }
  if (layout_convert_explicit(p.layout_convert)) {
    return true;
  }
  if (p.normalize.enable != AutoFlag::Auto || p.normalize.has_explicit_stats)
    return true;
  if (p.quantize.enable != AutoFlag::Auto || quantize_params_explicit(p.quantize))
    return true;
  if (p.tessellate.enable != AutoFlag::Auto || tessellate_geometry_explicit(p.tessellate))
    return true;
  return false;
}

std::string color_format_to_string(PreprocessColorFormat fmt) {
  switch (fmt) {
  case PreprocessColorFormat::RGB:
    return "RGB";
  case PreprocessColorFormat::BGR:
    return "BGR";
  case PreprocessColorFormat::GRAY8:
    return "GRAY8";
  case PreprocessColorFormat::NV12:
    return "NV12";
  case PreprocessColorFormat::I420:
    return "I420";
  case PreprocessColorFormat::Auto:
    return "";
  }
  return "";
}

bool format_is_gray(PreprocessColorFormat fmt) {
  return fmt == PreprocessColorFormat::GRAY8;
}

bool resolve_step_enabled(AutoFlag flag, bool default_when_auto) {
  if (flag == AutoFlag::On)
    return true;
  if (flag == AutoFlag::Off)
    return false;
  return default_when_auto;
}

bool derive_unit_pixel_normalization_from_input_range(const std::vector<double>& input_range,
                                                      std::array<float, 3>* mean,
                                                      std::array<float, 3>* stddev) {
  if (input_range.size() < 2U || !mean || !stddev) {
    return false;
  }
  const double lo = input_range[0];
  const double hi = input_range[1];
  if (!std::isfinite(lo) || !std::isfinite(hi) || hi <= lo) {
    return false;
  }

  // MPK input_range describes the tensor domain at the model ingress.  For image
  // preproc, the CVU preproc graph implements:
  //   normalized = (pixel / 255 - mean) / stddev
  // Therefore pixel=0 maps to -mean/stddev and pixel=255 maps to
  // (1-mean)/stddev.  Solve these two equations to recover the stats.
  const double derived_stddev = 1.0 / (hi - lo);
  const double derived_mean = -lo * derived_stddev;
  if (!std::isfinite(derived_mean) || !std::isfinite(derived_stddev) ||
      derived_mean < 0.0 || derived_mean > 1.0 ||
      derived_stddev <= 0.0 || derived_stddev > 1.0) {
    return false;
  }

  mean->fill(static_cast<float>(derived_mean));
  stddev->fill(static_cast<float>(derived_stddev));
  return true;
}

void apply_preset(PreprocessOptions* p) {
  if (!p)
    return;
  if (p->preset == NormalizePreset::ImageNet) {
    if (p->normalize.enable == AutoFlag::Auto) {
      p->normalize.enable = AutoFlag::On;
    }
    if (!p->normalize.has_explicit_stats) {
      p->normalize.mean = {0.485f, 0.456f, 0.406f};
      p->normalize.stddev = {0.229f, 0.224f, 0.225f};
      p->normalize.has_explicit_stats = true;
    }
    return;
  }
  if (p->preset == NormalizePreset::COCO_YOLO) {
    if (p->resize.enable == AutoFlag::Auto) {
      p->resize.enable = AutoFlag::On;
    }
    p->resize.mode = ResizeMode::Letterbox;
    p->resize.pad_value = 114;
    if (p->normalize.enable == AutoFlag::Auto) {
      p->normalize.enable = AutoFlag::On;
    }
    if (!p->normalize.has_explicit_stats) {
      p->normalize.mean = {0.0f, 0.0f, 0.0f};
      p->normalize.stddev = {1.0f, 1.0f, 1.0f};
      p->normalize.has_explicit_stats = true;
    }
  }
}

void apply_transform(PreprocessOptions* p, const Transform& t) {
  if (!p)
    return;
  switch (t.type) {
  case TransformType::Resize:
    p->resize = t.resize;
    return;
  case TransformType::ColorConvert:
    p->color_convert = t.color_convert;
    return;
  case TransformType::LayoutConvert:
    p->layout_convert = t.layout_convert;
    return;
  case TransformType::Normalize:
    p->normalize = t.normalize;
    return;
  case TransformType::Quantize:
    p->quantize = t.quantize;
    return;
  case TransformType::Tessellate:
    p->tessellate = t.tessellate;
    return;
  }
}

PipelineType pipeline_type_from_family(PreprocessGraphFamily family) {
  switch (family) {
  case PreprocessGraphFamily::Quant:
    return PipelineType::Quant;
  case PreprocessGraphFamily::Tess:
    return PipelineType::Tess;
  case PreprocessGraphFamily::QuantTess:
    return PipelineType::QuantTess;
  case PreprocessGraphFamily::Disabled:
  case PreprocessGraphFamily::Preproc:
    return PipelineType::Preproc;
  }
  return PipelineType::Preproc;
}

std::string kernel_name_from_family(PreprocessGraphFamily family) {
  switch (family) {
  case PreprocessGraphFamily::Disabled:
    return "disabled";
  case PreprocessGraphFamily::Preproc:
    return "preproc";
  case PreprocessGraphFamily::Quant:
    return "quant";
  case PreprocessGraphFamily::Tess:
    return "tess";
  case PreprocessGraphFamily::QuantTess:
    return "quanttess";
  }
  return "preproc";
}

void enforce_normalize_stats(const NormalizeSpec& spec) {
  auto positive = [](float v) { return v > 0.0f; };
  if (!positive(spec.stddev[0]) || !positive(spec.stddev[1]) || !positive(spec.stddev[2])) {
    throw std::invalid_argument("preprocess.normalize.stddev values must be > 0");
  }
}

const GraphFamilyCapabilities* capabilities_for_family(const PreprocessCapabilities& caps,
                                                       PreprocessGraphFamily family) {
  switch (family) {
  case PreprocessGraphFamily::Preproc:
    return &caps.preproc;
  case PreprocessGraphFamily::Quant:
    return &caps.quant;
  case PreprocessGraphFamily::Tess:
    return &caps.tess;
  case PreprocessGraphFamily::QuantTess:
    return &caps.quanttess;
  case PreprocessGraphFamily::Disabled:
    return nullptr;
  }
  return nullptr;
}

std::string family_name(PreprocessGraphFamily family) {
  switch (family) {
  case PreprocessGraphFamily::Preproc:
    return "Preproc";
  case PreprocessGraphFamily::Quant:
    return "Quant";
  case PreprocessGraphFamily::Tess:
    return "Tess";
  case PreprocessGraphFamily::QuantTess:
    return "QuantTess";
  case PreprocessGraphFamily::Disabled:
    return "Disabled";
  }
  return "Unknown";
}

void ensure_family_available(const PreprocessCapabilities& caps, PreprocessGraphFamily family) {
  const GraphFamilyCapabilities* fam = capabilities_for_family(caps, family);
  if (!fam || !fam->available) {
    std::string available_families;
    for (auto candidate : {PreprocessGraphFamily::Preproc, PreprocessGraphFamily::Quant,
                           PreprocessGraphFamily::Tess, PreprocessGraphFamily::QuantTess}) {
      const auto* c = capabilities_for_family(caps, candidate);
      if (c && c->available) {
        if (!available_families.empty()) available_families += ", ";
        available_families += family_name(candidate);
      }
    }
    if (available_families.empty()) available_families = "(none)";
    throw std::invalid_argument("preprocess planner: requested graph family '" +
                                family_name(family) +
                                "' is unavailable in this model pack (no fallback allowed)."
                                " Available families: " + available_families);
  }
}

void ensure_supported(const GraphFamilyCapabilities& fam, bool resize_enabled, bool color_enabled,
                      bool layout_enabled, bool normalize_enabled, bool quant_enabled,
                      bool tess_enabled, const std::string& family) {
  auto fail = [&](const char* op) {
    std::string supported;
    if (fam.supports_resize) { if (!supported.empty()) supported += ", "; supported += "resize"; }
    if (fam.supports_color_convert) { if (!supported.empty()) supported += ", "; supported += "color_convert"; }
    if (fam.supports_layout_convert) { if (!supported.empty()) supported += ", "; supported += "layout_convert"; }
    if (fam.supports_normalize) { if (!supported.empty()) supported += ", "; supported += "normalize"; }
    if (fam.supports_quantize) { if (!supported.empty()) supported += ", "; supported += "quantize"; }
    if (fam.supports_tessellate) { if (!supported.empty()) supported += ", "; supported += "tessellate"; }
    if (supported.empty()) supported = "(none)";
    throw std::invalid_argument("preprocess planner: graph family '" + family +
                                "' does not support requested op '" + op +
                                "' for this model (no fallback allowed)."
                                " Supported ops for this family: " + supported);
  };
  if (resize_enabled && !fam.supports_resize)
    fail("resize");
  if (color_enabled && !fam.supports_color_convert)
    fail("color_convert");
  if (layout_enabled && !fam.supports_layout_convert)
    fail("layout_convert");
  if (normalize_enabled && !fam.supports_normalize)
    fail("normalize");
  if (quant_enabled && !fam.supports_quantize)
    fail("quantize");
  if (tess_enabled && !fam.supports_tessellate)
    fail("tessellate");
}

GraphFamilyCapabilities defaults_for_kernel(const std::string& kernel) {
  GraphFamilyCapabilities fam;
  fam.available = true;
  if (kernel == "preproc") {
    fam.supports_resize = true;
    fam.supports_color_convert = true;
    fam.supports_layout_convert = true;
    fam.supports_normalize = true;
  } else if (kernel == "quant") {
    fam.supports_quantize = true;
  } else if (kernel == "tess") {
    fam.supports_tessellate = true;
  } else if (kernel == "quanttess") {
    fam.supports_quantize = true;
    fam.supports_tessellate = true;
  }
  return fam;
}

bool transform_has_explicit_enable(const std::vector<Transform>& transforms, TransformType type,
                                   AutoFlag flag) {
  for (const auto& t : transforms) {
    if (t.type != type)
      continue;
    switch (type) {
    case TransformType::Resize:
      if (t.resize.enable == flag)
        return true;
      break;
    case TransformType::ColorConvert:
      if (t.color_convert.enable == flag)
        return true;
      break;
    case TransformType::LayoutConvert:
      if (t.layout_convert.enable == flag)
        return true;
      break;
    case TransformType::Normalize:
      if (t.normalize.enable == flag)
        return true;
      break;
    case TransformType::Quantize:
      if (t.quantize.enable == flag)
        return true;
      break;
    case TransformType::Tessellate:
      if (t.tessellate.enable == flag)
        return true;
      break;
    }
  }
  return false;
}

std::string canonical_dtype_for_signal(std::string raw) {
  raw = lower_copy(std::move(raw));
  if (raw.find("bfloat16") != std::string::npos || raw.find("bf16") != std::string::npos) {
    return "BF16";
  }
  if (raw.find("float32") != std::string::npos || raw == "fp32") {
    return "FP32";
  }
  if (raw.find("float16") != std::string::npos || raw == "fp16") {
    return "FP16";
  }
  if (raw.find("int8") != std::string::npos) {
    return "INT8";
  }
  if (raw.find("uint8") != std::string::npos) {
    return "UINT8";
  }
  if (raw.find("int16") != std::string::npos) {
    return "INT16";
  }
  if (raw.find("uint16") != std::string::npos) {
    return "UINT16";
  }
  if (raw.find("int32") != std::string::npos) {
    return "INT32";
  }
  if (raw.find("uint32") != std::string::npos) {
    return "UINT32";
  }
  return upper_copy(std::move(raw));
}

bool dtype_is_float_like(std::string raw) {
  raw = upper_copy(std::move(raw));
  if (raw.empty()) {
    return false;
  }
  return raw.find("FP") != std::string::npos || raw.find("FLOAT") != std::string::npos ||
         raw.find("BF16") != std::string::npos || raw.find("BFLOAT16") != std::string::npos;
}

bool dtype_is_quantized_like(std::string raw) {
  raw = upper_copy(std::move(raw));
  if (raw.empty() || dtype_is_float_like(raw)) {
    return false;
  }
  return raw.find("INT8") != std::string::npos || raw.find("UINT8") != std::string::npos ||
         raw.find("INT16") != std::string::npos || raw.find("UINT16") != std::string::npos ||
         raw.find("INT32") != std::string::npos || raw.find("UINT32") != std::string::npos;
}

std::string primary_tensor_dtype(
    const std::vector<pipeline_internal::sima::MpkTensorContract>& tensors) {
  for (const auto& tensor : tensors) {
    if (!tensor.logical_dtype.empty()) {
      return canonical_dtype_for_signal(tensor.logical_dtype);
    }
    if (!tensor.dtype.empty()) {
      return canonical_dtype_for_signal(tensor.dtype);
    }
  }
  return {};
}

int primary_tensor_rank(const std::vector<pipeline_internal::sima::MpkTensorContract>& tensors) {
  for (const auto& tensor : tensors) {
    if (!tensor.logical_shape.empty()) {
      return static_cast<int>(tensor.logical_shape.size());
    }
    if (!tensor.mpk_shape.empty()) {
      return static_cast<int>(tensor.mpk_shape.size());
    }
  }
  return 0;
}

std::string canonical_mpk_kernel_kind(const std::string& raw_kernel) {
  std::string token;
  token.reserve(raw_kernel.size());
  for (const unsigned char c : raw_kernel) {
    if (std::isalnum(c)) {
      token.push_back(static_cast<char>(std::tolower(c)));
    }
  }
  if (token.empty()) {
    return {};
  }
  if (token.find("boxdecode") != std::string::npos || token.find("boxdecoder") != std::string::npos) {
    return "boxdecode";
  }
  if (token.find("detess") != std::string::npos && token.find("dequant") != std::string::npos) {
    return "detessdequant";
  }
  if (token.find("dequant") != std::string::npos) {
    return "dequantize";
  }
  if (token.find("detess") != std::string::npos) {
    return "detessellate";
  }
  if (token.find("quanttess") != std::string::npos) {
    return "quanttess";
  }
  if (token.find("quant") != std::string::npos) {
    return "quantize";
  }
  if (token.find("tess") != std::string::npos) {
    return "tessellate";
  }
  if (token.find("preproc") != std::string::npos) {
    return "preproc";
  }
  if (token.find("cast") != std::string::npos) {
    return "cast";
  }
  if (token.find("passthrough") != std::string::npos) {
    return "pass_through";
  }
  return token;
}

} // namespace

PreprocessCapabilities inspect_preprocess_capabilities(const ModelPack& pack) {
  PreprocessCapabilities out;
  out.preproc = defaults_for_kernel("preproc");
  const auto& maybe_contract = pack.mpk_contract();
  if (!maybe_contract.has_value()) {
    throw std::runtime_error(
        "preprocess planner: MPK contract is required for pre/post route selection."
        " Ensure the model pack includes an mpk.json manifest with plugin IO contracts.");
  }
  const auto& contract = *maybe_contract;
  for (const auto& ingress : contract.ingress_tensors) {
    if (derive_unit_pixel_normalization_from_input_range(
            ingress.input_range, &out.model_input_mean, &out.model_input_stddev)) {
      out.has_model_input_normalization = true;
      break;
    }
  }
  const auto* mla_stage = pipeline_internal::sima::get_mla_stage_io_contract(contract);
  if (!mla_stage) {
    throw std::runtime_error(
        "preprocess planner: MPK contract is missing an MLA stage for pre route selection."
        " Expected a plugin with processor='MLA' or kernel='infer'/'mla' in the MPK manifest.");
  }
  const auto mla_idx = pipeline_internal::sima::find_plugin_index_by_name_or_id(
      contract, !mla_stage->name.empty() ? mla_stage->name : mla_stage->plugin_id);
  if (!mla_idx.has_value()) {
    throw std::runtime_error(
        "preprocess planner: MPK contract MLA stage is not reachable in execution order.");
  }
  const auto ordered = pipeline_internal::sima::plugins_in_execution_order(contract);
  std::unordered_map<std::size_t, std::size_t> rank_by_index;
  rank_by_index.reserve(ordered.size());
  for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
    rank_by_index.emplace(ordered[rank], rank);
  }
  const auto mla_rank_it = rank_by_index.find(*mla_idx);
  if (mla_rank_it == rank_by_index.end()) {
    throw std::runtime_error(
        "preprocess planner: MPK execution order is missing the MLA stage.");
  }

  bool pre_has_quant = false;
  bool pre_has_tess = false;
  bool pre_has_preproc = false;
  for (const std::size_t plugin_idx : ordered) {
    if (plugin_idx >= contract.plugins.size() || plugin_idx == *mla_idx) {
      continue;
    }
    const auto rank_it = rank_by_index.find(plugin_idx);
    if (rank_it == rank_by_index.end() || rank_it->second >= mla_rank_it->second) {
      continue;
    }

    const auto& plugin = contract.plugins[plugin_idx];
    std::string kernel_source = plugin.kernel;
    if (kernel_source.empty()) {
      kernel_source = plugin.name;
    }
    const std::string kernel = canonical_mpk_kernel_kind(kernel_source);
    const std::string input_dtype = primary_tensor_dtype(plugin.input_tensors);
    const std::string output_dtype = primary_tensor_dtype(plugin.output_tensors);
    const int input_rank = primary_tensor_rank(plugin.input_tensors);
    const int output_rank = primary_tensor_rank(plugin.output_tensors);
    const bool dtype_transition =
        !input_dtype.empty() && !output_dtype.empty() && input_dtype != output_dtype;
    const bool hint_pre_quant =
        dtype_transition && !dtype_is_quantized_like(input_dtype) &&
        dtype_is_quantized_like(output_dtype);
    const bool hint_pre_tess = input_rank >= 3 && output_rank > 0 && output_rank < input_rank;

    if (kernel == "quanttess") {
      pre_has_quant = true;
      pre_has_tess = true;
      continue;
    }
    if (kernel == "quantize" || hint_pre_quant) {
      pre_has_quant = true;
    }
    if (kernel == "tessellate" || hint_pre_tess) {
      pre_has_tess = true;
    }
    if (kernel == "preproc") {
      pre_has_preproc = true;
    }
  }

  if (pre_has_quant) {
    out.quant = defaults_for_kernel("quant");
  }
  if (pre_has_tess) {
    out.tess = defaults_for_kernel("tess");
  }
  if (pre_has_quant && pre_has_tess) {
    out.quanttess = defaults_for_kernel("quanttess");
  }
  if (pre_has_quant && pre_has_tess) {
    out.tensor_auto_family = PreprocessGraphFamily::QuantTess;
  } else if (pre_has_tess) {
    out.tensor_auto_family = PreprocessGraphFamily::Tess;
  } else if (pre_has_quant) {
    out.tensor_auto_family = PreprocessGraphFamily::Quant;
  } else if (pre_has_preproc) {
    out.tensor_auto_family = PreprocessGraphFamily::Preproc;
  }

  return out;
}

PreprocessPlannerResult plan_preprocess(const Model::Options& options,
                                        const PreprocessCapabilities& capabilities) {
  PreprocessPlannerResult out;
  out.resolved_plan.requested = options.preprocess;
  PreprocessOptions effective = options.preprocess;
  PreprocessExplicitKnobs explicit_knobs;
  explicit_knobs.resize =
      options.preprocess.resize.enable != AutoFlag::Auto || options.preprocess.resize.width > 0 ||
      options.preprocess.resize.height > 0;
  explicit_knobs.color_convert =
      options.preprocess.color_convert.enable != AutoFlag::Auto ||
      options.preprocess.color_convert.input_format != PreprocessColorFormat::Auto ||
      options.preprocess.color_convert.output_format != PreprocessColorFormat::Auto;
  explicit_knobs.layout_convert = layout_convert_explicit(options.preprocess.layout_convert);
  explicit_knobs.normalize = options.preprocess.normalize.enable != AutoFlag::Auto;
  explicit_knobs.normalize_stats = options.preprocess.normalize.has_explicit_stats;
  explicit_knobs.quantize_enable = options.preprocess.quantize.enable != AutoFlag::Auto;
  explicit_knobs.quantize_params = quantize_params_explicit(options.preprocess.quantize);
  explicit_knobs.tessellate_enable = options.preprocess.tessellate.enable != AutoFlag::Auto;
  explicit_knobs.tessellate_geometry = tessellate_geometry_explicit(options.preprocess.tessellate);

  const bool has_transforms = !options.preprocess.transforms.empty();
  if (has_transforms) {
    if (has_non_default_simple_preprocess(options.preprocess)) {
      out.resolved_plan.warnings.push_back(
          "preprocess.transforms and simple preprocess flags were both set; using transforms.");
    }
    out.resolved_plan.transforms_override = true;
    explicit_knobs = {};

    PreprocessOptions compiled;
    compiled.kind = options.preprocess.kind;
    compiled.enable = options.preprocess.enable;
    compiled.input_max_width = options.preprocess.input_max_width;
    compiled.input_max_height = options.preprocess.input_max_height;
    compiled.input_max_depth = options.preprocess.input_max_depth;
    compiled.preset = options.preprocess.preset;
    apply_preset(&compiled);
    for (const auto& t : options.preprocess.transforms) {
      apply_transform(&compiled, t);
      switch (t.type) {
      case TransformType::Resize:
        explicit_knobs.resize =
            explicit_knobs.resize || t.resize.enable != AutoFlag::Auto || t.resize.width > 0 ||
            t.resize.height > 0;
        break;
      case TransformType::ColorConvert:
        explicit_knobs.color_convert =
            explicit_knobs.color_convert || t.color_convert.enable != AutoFlag::Auto ||
            t.color_convert.input_format != PreprocessColorFormat::Auto ||
            t.color_convert.output_format != PreprocessColorFormat::Auto;
        break;
      case TransformType::LayoutConvert:
        explicit_knobs.layout_convert =
            explicit_knobs.layout_convert || layout_convert_explicit(t.layout_convert);
        break;
      case TransformType::Normalize:
        explicit_knobs.normalize = explicit_knobs.normalize || t.normalize.enable != AutoFlag::Auto;
        explicit_knobs.normalize_stats =
            explicit_knobs.normalize_stats || t.normalize.has_explicit_stats;
        break;
      case TransformType::Quantize:
        explicit_knobs.quantize_enable =
            explicit_knobs.quantize_enable || t.quantize.enable != AutoFlag::Auto;
        explicit_knobs.quantize_params =
            explicit_knobs.quantize_params || quantize_params_explicit(t.quantize);
        break;
      case TransformType::Tessellate:
        explicit_knobs.tessellate_enable =
            explicit_knobs.tessellate_enable || t.tessellate.enable != AutoFlag::Auto;
        explicit_knobs.tessellate_geometry =
            explicit_knobs.tessellate_geometry || tessellate_geometry_explicit(t.tessellate);
        break;
      }
    }
    effective = std::move(compiled);
  } else {
    apply_preset(&effective);
  }

  // MPK input_range describes the numeric domain expected at model ingress; it is
  // metadata, not an instruction to enable CVU preproc normalization.  Keep
  // normalization user-driven: explicit mean/stddev or a preset may supply
  // stats, but input_range alone must not flip normalize from Auto to On.

  validate_layout_convert_spec(effective.layout_convert);

  if (effective.enable == AutoFlag::Off && has_transforms) {
    out.resolved_plan.warnings.push_back(
        "preprocess.enable=Off with non-empty transforms: transforms are applied.");
    effective.enable = AutoFlag::On;
  }

  bool resize_enabled =
      resolve_step_enabled(effective.resize.enable, effective.resize.width > 0 ||
                                                      effective.resize.height > 0 ||
                                                      effective.preset == NormalizePreset::COCO_YOLO);
  bool color_enabled = resolve_step_enabled(
      effective.color_convert.enable,
      effective.color_convert.input_format != PreprocessColorFormat::Auto ||
          effective.color_convert.output_format != PreprocessColorFormat::Auto);
  bool layout_enabled = resolve_step_enabled(
      effective.layout_convert.enable, effective.layout_convert.has_perm());
  const bool normalize_requested_or_preset =
      effective.normalize.has_explicit_stats || effective.preset == NormalizePreset::ImageNet ||
      effective.preset == NormalizePreset::COCO_YOLO;
  bool normalize_enabled =
      resolve_step_enabled(effective.normalize.enable, normalize_requested_or_preset);

  InputKind kind = effective.kind;
  if (kind == InputKind::Auto) {
    const bool image_like = resize_enabled || color_enabled || layout_enabled || normalize_enabled ||
                            effective.preset != NormalizePreset::None;
    const bool tensor_like =
        effective.quantize.enable == AutoFlag::On || effective.tessellate.enable == AutoFlag::On;
    if (image_like) {
      kind = InputKind::Image;
    } else if (tensor_like) {
      kind = InputKind::Tensor;
    } else {
      kind = InputKind::Image;
    }
  }

  const bool tensor_auto_quant =
      kind == InputKind::Tensor &&
      (capabilities.tensor_auto_family == PreprocessGraphFamily::Quant ||
       capabilities.tensor_auto_family == PreprocessGraphFamily::QuantTess);
  const bool tensor_auto_tess =
      kind == InputKind::Tensor &&
      (capabilities.tensor_auto_family == PreprocessGraphFamily::Tess ||
       capabilities.tensor_auto_family == PreprocessGraphFamily::QuantTess);

  bool quant_enabled = resolve_step_enabled(effective.quantize.enable, tensor_auto_quant);
  bool tess_enabled = resolve_step_enabled(effective.tessellate.enable, tensor_auto_tess);

  bool enabled = true;
  if (effective.enable == AutoFlag::Off) {
    enabled = false;
  } else if (effective.enable == AutoFlag::Auto) {
    enabled = resize_enabled || color_enabled || layout_enabled || normalize_enabled || quant_enabled ||
              tess_enabled;
  }

  if (!enabled) {
    resize_enabled = false;
    color_enabled = false;
    layout_enabled = false;
    normalize_enabled = false;
    quant_enabled = false;
    tess_enabled = false;
  }

  if (normalize_enabled) {
    enforce_normalize_stats(effective.normalize);
  }

  const AutoFlag requested_quant_flag = effective.quantize.enable;
  const AutoFlag requested_tess_flag = effective.tessellate.enable;

  effective.resize.enable = resize_enabled ? AutoFlag::On : AutoFlag::Off;
  effective.color_convert.enable = color_enabled ? AutoFlag::On : AutoFlag::Off;
  effective.layout_convert.enable = layout_enabled ? AutoFlag::On : AutoFlag::Off;
  effective.normalize.enable = normalize_enabled ? AutoFlag::On : AutoFlag::Off;
  if (requested_quant_flag == AutoFlag::Auto && !quant_enabled) {
    effective.quantize.enable = AutoFlag::Auto;
  } else {
    effective.quantize.enable = quant_enabled ? AutoFlag::On : AutoFlag::Off;
  }
  if (requested_tess_flag == AutoFlag::Auto && !tess_enabled) {
    effective.tessellate.enable = AutoFlag::Auto;
  } else {
    effective.tessellate.enable = tess_enabled ? AutoFlag::On : AutoFlag::Off;
  }

  PreprocessGraphFamily family = PreprocessGraphFamily::Disabled;
  if (enabled) {
    if (quant_enabled && tess_enabled) {
      family = PreprocessGraphFamily::QuantTess;
    } else if (quant_enabled) {
      family = PreprocessGraphFamily::Quant;
    } else if (tess_enabled) {
      family = PreprocessGraphFamily::Tess;
    } else {
      family = PreprocessGraphFamily::Preproc;
    }
  }

  if (enabled) {
    ensure_family_available(capabilities, family);
    const GraphFamilyCapabilities* fam = capabilities_for_family(capabilities, family);
    if (!fam) {
      throw std::invalid_argument("preprocess planner: internal capability lookup failure");
    }
    ensure_supported(*fam, resize_enabled, color_enabled, layout_enabled, normalize_enabled,
                     quant_enabled, tess_enabled, family_name(family));
  }

  const bool explicit_off_resize = has_transforms
                                       ? transform_has_explicit_enable(options.preprocess.transforms,
                                                                       TransformType::Resize,
                                                                       AutoFlag::Off)
                                       : options.preprocess.resize.enable == AutoFlag::Off;
  const bool explicit_off_normalize =
      has_transforms ? transform_has_explicit_enable(options.preprocess.transforms,
                                                     TransformType::Normalize, AutoFlag::Off)
                     : options.preprocess.normalize.enable == AutoFlag::Off;

  auto emit_issue = [&](const RequirementIssue& issue) {
    out.requirement_issues.push_back(issue);
    const std::string message = requirement_issue_message(issue);
    if (issue.severity == RequirementSeverity::HardError) {
      throw std::invalid_argument(message);
    }
    out.resolved_plan.warnings.push_back(message);
  };

  if (enabled) {
    if (explicit_off_normalize &&
        (effective.preset == NormalizePreset::ImageNet ||
         effective.preset == NormalizePreset::COCO_YOLO)) {
      RequirementIssue issue;
      issue.op = RequiredPreprocessOp::Normalize;
      issue.severity = RequirementSeverity::Warning;
      issue.source = RequirementSource::Preset;
      issue.code = "PREPROC_REQ_WARN_MODEL_DEFAULT";
      issue.reason =
          "normalize was explicitly disabled, but the selected preset expects normalize to stay enabled.";
      issue.fix_hint = "Use preprocess.normalize.enable=Auto/On, or set explicit mean/std.";
      emit_issue(issue);
    }
    if (explicit_off_resize && effective.preset == NormalizePreset::COCO_YOLO) {
      RequirementIssue issue;
      issue.op = RequiredPreprocessOp::Resize;
      issue.severity = RequirementSeverity::Warning;
      issue.source = RequirementSource::Preset;
      issue.code = "PREPROC_REQ_WARN_MODEL_DEFAULT";
      issue.reason =
          "resize was explicitly disabled while COCO_YOLO preset expects letterbox resize.";
      issue.fix_hint =
          "Use preprocess.resize.enable=Auto/On, or switch preset to None for manual control.";
      emit_issue(issue);
    }
  }

  if (resize_enabled && (effective.resize.width <= 0 || effective.resize.height <= 0)) {
    out.resolved_plan.warnings.push_back(
        "preprocess.resize enabled without explicit width/height; "
        "Model path will infer missing dimensions from MLA input contract.");
  }

  out.resolved_plan.effective = effective;
  out.resolved_plan.explicit_knobs = explicit_knobs;
  out.resolved_plan.resolved_kind = kind;
  out.resolved_plan.enabled = enabled;
  out.resolved_plan.graph_family = family;
  out.resolved_plan.graph_kernel = kernel_name_from_family(family);
  out.pipeline_type = pipeline_type_from_family(family);
  out.include_preprocess_stage = (family != PreprocessGraphFamily::Disabled);
  out.include_postprocess_stage = true;
  out.infer_only_route = false;
  out.mla_tessellation = false;

  const bool tensor_ingress = (kind == InputKind::Tensor) ||
                              family == PreprocessGraphFamily::Quant ||
                              family == PreprocessGraphFamily::Tess ||
                              family == PreprocessGraphFamily::QuantTess;
  out.modelpack_media_type = tensor_ingress ? "application/vnd.simaai.tensor" : "video/x-raw";
  if (tensor_ingress) {
    out.modelpack_format = "FP32";
  } else {
    const PreprocessColorFormat in_fmt = (effective.color_convert.input_format == PreprocessColorFormat::Auto)
                                             ? PreprocessColorFormat::RGB
                                             : effective.color_convert.input_format;
    out.modelpack_format = color_format_to_string(in_fmt);
  }

  if (tensor_ingress) {
    out.modelpack_input_depth = (effective.input_max_depth > 0)
                                    ? effective.input_max_depth
                                    : (pipeline_internal::mpk_no_json_path_enabled() ? 0 : 3);
  } else {
    const PreprocessColorFormat in_fmt = (effective.color_convert.input_format == PreprocessColorFormat::Auto)
                                             ? PreprocessColorFormat::RGB
                                             : effective.color_convert.input_format;
    out.modelpack_input_depth =
        format_is_gray(in_fmt) ? 1 : pipeline_internal::default_depth_for_image_format(out.modelpack_format, 3);
  }

  out.modelpack_max_width = (effective.input_max_width > 0) ? effective.input_max_width : 1920;
  out.modelpack_max_height = (effective.input_max_height > 0) ? effective.input_max_height : 1080;
  if (effective.input_max_depth > 0) {
    out.modelpack_max_depth = effective.input_max_depth;
  } else if (out.modelpack_input_depth > 0) {
    out.modelpack_max_depth = out.modelpack_input_depth;
  } else if (pipeline_internal::mpk_no_json_path_enabled()) {
    out.modelpack_max_depth = 0;
  } else {
    out.modelpack_max_depth = 3;
  }

  out.normalize = normalize_enabled;
  out.mean = std::vector<float>{effective.normalize.mean[0], effective.normalize.mean[1],
                                effective.normalize.mean[2]};
  out.stddev = std::vector<float>{effective.normalize.stddev[0], effective.normalize.stddev[1],
                                  effective.normalize.stddev[2]};

  out.resolved_plan.meta_contract.required_fields =
      default_preprocess_meta_required_fields();

  return out;
}

std::string required_preprocess_op_name(RequiredPreprocessOp op) {
  switch (op) {
  case RequiredPreprocessOp::Resize:
    return "resize";
  case RequiredPreprocessOp::ColorConvert:
    return "color_convert";
  case RequiredPreprocessOp::LayoutConvert:
    return "layout_convert";
  case RequiredPreprocessOp::Normalize:
    return "normalize";
  case RequiredPreprocessOp::Quantize:
    return "quantize";
  case RequiredPreprocessOp::Tessellate:
    return "tessellate";
  }
  return "unknown";
}

std::string requirement_severity_name(RequirementSeverity severity) {
  switch (severity) {
  case RequirementSeverity::HardError:
    return "error";
  case RequirementSeverity::Warning:
    return "warning";
  }
  return "warning";
}

std::string requirement_source_name(RequirementSource source) {
  switch (source) {
  case RequirementSource::MlaContract:
    return "MlaContract";
  case RequirementSource::GraphFamilyCapability:
    return "GraphFamilyCapability";
  case RequirementSource::GraphDefault:
    return "GraphDefault";
  case RequirementSource::Preset:
    return "Preset";
  case RequirementSource::UserOverride:
    return "UserOverride";
  case RequirementSource::RuntimeObserved:
    return "RuntimeObserved";
  }
  return "Unknown";
}

std::string requirement_issue_message(const RequirementIssue& issue) {
  std::ostringstream oss;
  oss << "preprocess requirement violation: "
      << "code=" << (issue.code.empty() ? std::string("PREPROC_REQ_UNSPECIFIED") : issue.code)
      << " op=" << required_preprocess_op_name(issue.op)
      << " severity=" << requirement_severity_name(issue.severity)
      << " source=" << requirement_source_name(issue.source)
      << " reason=" << issue.reason;
  if (!issue.fix_hint.empty()) {
    oss << " fix=" << issue.fix_hint;
  }
  return oss.str();
}

} // namespace simaai::neat::internal

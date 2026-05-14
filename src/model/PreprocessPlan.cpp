#include "model/PreprocessPlan.h"

#include <sstream>

namespace simaai::neat {
namespace {

const char* auto_flag_name(AutoFlag v) {
  switch (v) {
  case AutoFlag::Auto:
    return "Auto";
  case AutoFlag::On:
    return "On";
  case AutoFlag::Off:
    return "Off";
  }
  return "Auto";
}

const char* input_kind_name(InputKind v) {
  switch (v) {
  case InputKind::Auto:
    return "Auto";
  case InputKind::Image:
    return "Image";
  case InputKind::Tensor:
    return "Tensor";
  }
  return "Auto";
}

const char* resize_mode_name(ResizeMode v) {
  switch (v) {
  case ResizeMode::Stretch:
    return "Stretch";
  case ResizeMode::Letterbox:
    return "Letterbox";
  case ResizeMode::Crop:
    return "Crop";
  }
  return "Letterbox";
}

const char* normalize_preset_name(NormalizePreset v) {
  switch (v) {
  case NormalizePreset::None:
    return "None";
  case NormalizePreset::ImageNet:
    return "ImageNet";
  case NormalizePreset::COCO_YOLO:
    return "COCO_YOLO";
  }
  return "None";
}

const char* graph_family_name(PreprocessGraphFamily v) {
  switch (v) {
  case PreprocessGraphFamily::Disabled:
    return "Disabled";
  case PreprocessGraphFamily::Preproc:
    return "Preproc";
  case PreprocessGraphFamily::Quant:
    return "Quant";
  case PreprocessGraphFamily::Tess:
    return "Tess";
  case PreprocessGraphFamily::QuantTess:
    return "QuantTess";
  }
  return "Disabled";
}

void append_contract(std::ostringstream& ss, const char* label, const PreprocessContract& c) {
  ss << label << "{media_type=" << c.media_type << ", format=" << c.format << ", width=" << c.width
     << ", height=" << c.height << ", depth=" << c.depth << ", max_width=" << c.max_width
     << ", max_height=" << c.max_height << ", max_depth=" << c.max_depth << "}";
}

} // namespace

std::string ResolvedPreprocessPlan::to_debug_string() const {
  std::ostringstream ss;
  ss << "ResolvedPreprocessPlan{"
     << "requested.kind=" << input_kind_name(requested.kind)
     << ", requested.enable=" << auto_flag_name(requested.enable)
     << ", requested.preset=" << normalize_preset_name(requested.preset) << ", requested.resize=("
     << auto_flag_name(requested.resize.enable) << ", " << requested.resize.width << "x"
     << requested.resize.height << ", " << resize_mode_name(requested.resize.mode) << ")"
     << ", requested.normalize=" << auto_flag_name(requested.normalize.enable)
     << ", requested.quantize=" << auto_flag_name(requested.quantize.enable)
     << ", requested.tessellate=" << auto_flag_name(requested.tessellate.enable)
     << ", requested.transforms=" << requested.transforms.size()
     << ", explicit{resize=" << (explicit_knobs.resize ? "1" : "0")
     << ",color=" << (explicit_knobs.color_convert ? "1" : "0")
     << ",layout=" << (explicit_knobs.layout_convert ? "1" : "0")
     << ",normalize=" << (explicit_knobs.normalize ? "1" : "0")
     << ",normalize_stats=" << (explicit_knobs.normalize_stats ? "1" : "0")
     << ",quant_enable=" << (explicit_knobs.quantize_enable ? "1" : "0")
     << ",quant_params=" << (explicit_knobs.quantize_params ? "1" : "0")
     << ",tess_enable=" << (explicit_knobs.tessellate_enable ? "1" : "0")
     << ",tess_geom=" << (explicit_knobs.tessellate_geometry ? "1" : "0") << "}"
     << ", effective.kind=" << input_kind_name(effective.kind)
     << ", effective.enable=" << auto_flag_name(effective.enable)
     << ", effective.preset=" << normalize_preset_name(effective.preset)
     << ", graph_family=" << graph_family_name(graph_family) << ", graph_kernel=" << graph_kernel
     << ", graph_config_path=" << graph_config_path
     << ", resolved_kind=" << input_kind_name(resolved_kind)
     << ", enabled=" << (enabled ? "true" : "false")
     << ", transforms_override=" << (transforms_override ? "true" : "false")
     << ", ingress_contracts=[";
  for (size_t i = 0; i < ingress_contracts.size(); ++i) {
    if (i) {
      ss << ", ";
    }
    append_contract(ss, "", ingress_contracts[i]);
  }
  ss << "], ";
  append_contract(ss, "mla=", mla_contract);
  ss << ", warnings=[";
  for (size_t i = 0; i < warnings.size(); ++i) {
    if (i)
      ss << "; ";
    ss << warnings[i];
  }
  ss << "], meta_required=[";
  for (size_t i = 0; i < meta_contract.required_fields.size(); ++i) {
    if (i)
      ss << ", ";
    ss << meta_contract.required_fields[i];
  }
  ss << "]}";
  return ss.str();
}

} // namespace simaai::neat

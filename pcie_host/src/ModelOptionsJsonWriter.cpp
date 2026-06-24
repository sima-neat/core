#include "ModelOptionsJsonWriter.h"

#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

namespace simaai::neat::pcie::internal {
namespace {

using nlohmann::ordered_json;

const char* auto_flag_name(const AutoFlag flag) {
  switch (flag) {
  case AutoFlag::Auto:
    return "auto";
  case AutoFlag::On:
    return "on";
  case AutoFlag::Off:
    return "off";
  }
  return "auto";
}

std::string resize_mode_token(const ResizeMode mode) {
  switch (mode) {
  case ResizeMode::Stretch:
    return "stretch";
  case ResizeMode::Letterbox:
    return "letterbox";
  case ResizeMode::Crop:
    return "crop";
  }
  return "letterbox";
}

std::string color_format_token(const ColorFormat format) {
  switch (format) {
  case ColorFormat::Auto:
    return "auto";
  case ColorFormat::RGB:
    return "rgb";
  case ColorFormat::BGR:
    return "bgr";
  case ColorFormat::GRAY8:
    return "gray8";
  case ColorFormat::NV12:
    return "nv12";
  case ColorFormat::I420:
    return "i420";
  }
  return "auto";
}

std::string normalize_preset_token(const NormalizePreset preset) {
  switch (preset) {
  case NormalizePreset::None:
    return "none";
  case NormalizePreset::ImageNet:
    return "imagenet";
  case NormalizePreset::COCO_YOLO:
    return "coco_yolo";
  }
  return "none";
}

bool float_set(const float value) {
  return std::fabs(value) > 0.0f;
}

std::string box_decode_type_token(const BoxDecodeType type) {
  switch (type) {
  case BoxDecodeType::Unspecified:
    return "";
  case BoxDecodeType::YoloV5:
    return "yolov5";
  case BoxDecodeType::YoloV6:
    return "yolov6";
  case BoxDecodeType::YoloV7:
    return "yolov7";
  case BoxDecodeType::YoloV8:
    return "yolov8";
  case BoxDecodeType::YoloV26:
    return "yolo26";
  case BoxDecodeType::YoloX:
    return "yolox";
  }
  return "";
}

std::string box_decode_type_option_token(const BoxDecodeTypeOption option) {
  switch (option) {
  case BoxDecodeTypeOption::Auto:
    return "auto";
  case BoxDecodeTypeOption::Ultralytics:
    return "ultralytics";
  case BoxDecodeTypeOption::EfficientNMS:
    return "efficient_nms";
  }
  return "auto";
}

bool has_boxdecode_request(const ModelOptions& opt) {
  return opt.decode_type != BoxDecodeType::Unspecified ||
         opt.decode_type_option != BoxDecodeTypeOption::Auto || float_set(opt.score_threshold) ||
         float_set(opt.nms_iou_threshold) || opt.top_k != 0 || opt.num_classes != 0;
}

void reject(const std::string& message) {
  throw std::invalid_argument("ModelOptions cannot be represented by WP9 model-options JSON: " +
                              message);
}

void validate_supported_options(const ModelOptions& opt) {
  if (opt.preprocess.enable != AutoFlag::Auto && opt.preprocess.enable != AutoFlag::On) {
    reject(std::string("preprocess.enable=") + auto_flag_name(opt.preprocess.enable));
  }
  if (opt.preprocess.color_convert.output_format == ColorFormat::NV12 ||
      opt.preprocess.color_convert.output_format == ColorFormat::I420 ||
      opt.preprocess.color_convert.output_format == ColorFormat::GRAY8) {
    reject("preprocess.color_convert.output_format supports only auto/rgb/bgr");
  }
}

ordered_json preprocess_json(const ModelOptions& opt) {
  const auto& p = opt.preprocess;
  ordered_json pre = ordered_json::object();

  if (p.input_max_width > 0 || p.input_max_height > 0 || p.input_max_depth > 0) {
    ordered_json input_max = ordered_json::object();
    if (p.input_max_width > 0)
      input_max["width"] = p.input_max_width;
    if (p.input_max_height > 0)
      input_max["height"] = p.input_max_height;
    if (p.input_max_depth > 0)
      input_max["depth"] = p.input_max_depth;
    pre["input_max"] = std::move(input_max);
  }

  if (p.resize.enable == AutoFlag::On || p.resize.mode != ResizeMode::Letterbox ||
      p.resize.pad_value != 114 || p.resize.scaling_type != "BILINEAR") {
    ordered_json resize = ordered_json::object();
    resize["mode"] = resize_mode_token(p.resize.mode);
    resize["pad_value"] = p.resize.pad_value;
    resize["scaling_type"] = p.resize.scaling_type;
    pre["resize"] = std::move(resize);
  }

  if (p.color_convert.enable == AutoFlag::On || p.color_convert.input_format != ColorFormat::Auto ||
      p.color_convert.output_format != ColorFormat::Auto) {
    ordered_json color = ordered_json::object();
    color["input_format"] = color_format_token(p.color_convert.input_format);
    color["output_format"] = color_format_token(p.color_convert.output_format);
    pre["color_convert"] = std::move(color);
  }

  if (p.normalize.enable == AutoFlag::On || p.normalize.has_explicit_stats ||
      p.normalize.preset != NormalizePreset::None) {
    ordered_json norm = ordered_json::object();
    if (p.normalize.preset != NormalizePreset::None) {
      norm["preset"] = normalize_preset_token(p.normalize.preset);
    }
    if (p.normalize.has_explicit_stats) {
      norm["mean"] = p.normalize.mean;
      norm["stddev"] = p.normalize.stddev;
    }
    pre["normalize"] = std::move(norm);
  }

  return pre;
}

ordered_json boxdecode_json(const ModelOptions& opt) {
  ordered_json box = ordered_json::object();
  if (opt.decode_type != BoxDecodeType::Unspecified) {
    box["decode_type"] = box_decode_type_token(opt.decode_type);
  }
  if (opt.decode_type_option != BoxDecodeTypeOption::Auto) {
    box["decode_type_option"] = box_decode_type_option_token(opt.decode_type_option);
  }
  if (float_set(opt.score_threshold)) {
    box["score_threshold"] = opt.score_threshold;
  }
  if (float_set(opt.nms_iou_threshold)) {
    box["nms_iou_threshold"] = opt.nms_iou_threshold;
  }
  if (opt.top_k != 0) {
    box["top_k"] = opt.top_k;
  }
  if (opt.num_classes != 0) {
    box["num_classes"] = opt.num_classes;
  }
  return box;
}

} // namespace

ModelOptionsJson write_model_options_json(const ModelOptions& options) {
  validate_supported_options(options);

  const bool wants_boxdecode = has_boxdecode_request(options);
  if (options.preprocess.kind == InputKind::Tensor) {
    if (wants_boxdecode) {
      reject("boxdecode requires preprocess.kind=InputKind::Image");
    }
    return {};
  }
  if (options.preprocess.kind != InputKind::Image) {
    reject("unsupported preprocess.kind");
  }

  ordered_json root = ordered_json::object();
  root["schema"] = 1;
  root["preprocess"] = preprocess_json(options);
  ModelOptionsJson out;
  if (wants_boxdecode) {
    root["boxdecode"] = boxdecode_json(options);
  }
  out.json = root.dump(2) + "\n";
  return out;
}

} // namespace simaai::neat::pcie::internal

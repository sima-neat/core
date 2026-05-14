#include "pipeline/internal/sima/BoxDecodeTypeUtils.h"

#include <algorithm>
#include <cctype>

namespace simaai::neat::pipeline_internal::sima {

namespace {

std::string lower_copy_local(std::string_view raw) {
  std::string out(raw);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return out;
}

} // namespace

std::optional<BoxDecodeTypeOption> parse_box_decode_type_option_token(std::string_view token) {
  const std::string lower = lower_copy_local(token);
  if (lower.empty() || lower == "auto") {
    return BoxDecodeTypeOption::Auto;
  }
  if (lower == "packed-per-head") {
    return BoxDecodeTypeOption::PackedPerHead;
  }
  if (lower == "interleaved-by-head") {
    return BoxDecodeTypeOption::InterleavedByHead;
  }
  if (lower == "grouped-by-role") {
    return BoxDecodeTypeOption::GroupedByRole;
  }
  if (lower == "split3-interleaved") {
    return BoxDecodeTypeOption::Split3Interleaved;
  }
  if (lower == "split3-grouped") {
    return BoxDecodeTypeOption::Split3Grouped;
  }
  if (lower == "interleaved-by-head-probability") {
    return BoxDecodeTypeOption::InterleavedByHeadProbability;
  }
  if (lower == "interleaved-by-head-logit") {
    return BoxDecodeTypeOption::InterleavedByHeadLogit;
  }
  if (lower == "grouped-by-role-probability") {
    return BoxDecodeTypeOption::GroupedByRoleProbability;
  }
  if (lower == "grouped-by-role-logit") {
    return BoxDecodeTypeOption::GroupedByRoleLogit;
  }
  return std::nullopt;
}

std::optional<BoxDecodeType> parse_box_decode_type_token(std::string_view token) {
  const std::string lower = lower_copy_local(token);
  if (lower.empty()) {
    return BoxDecodeType::Unspecified;
  }
  if (lower == "yolo") {
    return BoxDecodeType::Yolo;
  }
  if (lower == "yolov5") {
    return BoxDecodeType::YoloV5;
  }
  if (lower == "yolov5-seg") {
    return BoxDecodeType::YoloV5Seg;
  }
  if (lower == "yolov7") {
    return BoxDecodeType::YoloV7;
  }
  if (lower == "yolov7-seg") {
    return BoxDecodeType::YoloV7Seg;
  }
  if (lower == "yolov8") {
    return BoxDecodeType::YoloV8;
  }
  if (lower == "yolov8-seg") {
    return BoxDecodeType::YoloV8Seg;
  }
  if (lower == "yolov8-pose") {
    return BoxDecodeType::YoloV8Pose;
  }
  if (lower == "yolov9") {
    return BoxDecodeType::YoloV9;
  }
  if (lower == "yolov9-seg") {
    return BoxDecodeType::YoloV9Seg;
  }
  if (lower == "yolov10") {
    return BoxDecodeType::YoloV10;
  }
  if (lower == "yolov10-seg") {
    return BoxDecodeType::YoloV10Seg;
  }
  if (lower == "detr") {
    return BoxDecodeType::Detr;
  }
  if (lower == "effdet") {
    return BoxDecodeType::EffDet;
  }
  if (lower == "rcnn-stage1") {
    return BoxDecodeType::RcnnStage1;
  }
  if (lower == "centernet") {
    return BoxDecodeType::Centernet;
  }
  return std::nullopt;
}

bool is_box_decode_type_specified(BoxDecodeType type) {
  return type != BoxDecodeType::Unspecified;
}

bool is_box_decode_type_option_requires_uniform_score_domain(BoxDecodeTypeOption option) {
  switch (option) {
  case BoxDecodeTypeOption::InterleavedByHeadProbability:
  case BoxDecodeTypeOption::InterleavedByHeadLogit:
  case BoxDecodeTypeOption::GroupedByRoleProbability:
  case BoxDecodeTypeOption::GroupedByRoleLogit:
    return true;
  case BoxDecodeTypeOption::Auto:
  case BoxDecodeTypeOption::PackedPerHead:
  case BoxDecodeTypeOption::InterleavedByHead:
  case BoxDecodeTypeOption::GroupedByRole:
  case BoxDecodeTypeOption::Split3Interleaved:
  case BoxDecodeTypeOption::Split3Grouped:
  default:
    return false;
  }
}

std::string box_decode_type_token_string(BoxDecodeType type) {
  if (!is_box_decode_type_specified(type)) {
    return {};
  }
  return std::string(box_decode_type_token(type));
}

std::string box_decode_type_option_token_string(BoxDecodeTypeOption option) {
  return std::string(box_decode_type_option_token(option));
}

} // namespace simaai::neat::pipeline_internal::sima

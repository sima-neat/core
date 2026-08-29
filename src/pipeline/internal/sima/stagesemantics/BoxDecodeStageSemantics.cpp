#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"

#include "pipeline/internal/sima/BoxDecodeTypeUtils.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/stagesemantics/SsdDecodeContract.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::stagesemantics {
namespace {

bool boxdecode_bypass_mla_unpack_enabled() {
  const char* raw = std::getenv("SIMA_BOXDECODE_BYPASS_MLA_UNPACK");
  return raw && *raw && std::strcmp(raw, "0") != 0;
}

bool contract_looks_like_grouped_yolov26(const BoxDecodeStaticContract& contract) {
  if (contract.tensors.empty() || (contract.tensors.size() % 2U) != 0U) {
    return false;
  }
  const std::size_t heads = contract.tensors.size() / 2U;
  for (std::size_t i = 0; i < heads; ++i) {
    const auto& bbox = contract.tensors[i];
    const auto& scores = contract.tensors[i + heads];
    if (bbox.input_shape.size() < 3U || scores.input_shape.size() < 3U) {
      return false;
    }
    if (bbox.input_shape[0] != scores.input_shape[0] ||
        bbox.input_shape[1] != scores.input_shape[1]) {
      return false;
    }
    const int bbox_depth =
        bbox.slice_shape.size() >= 3U ? bbox.slice_shape[2] : bbox.input_shape[2];
    const int score_depth =
        scores.slice_shape.size() >= 3U ? scores.slice_shape[2] : scores.input_shape[2];
    if (bbox_depth != 4 || bbox.input_shape[2] < bbox_depth || score_depth <= 4) {
      return false;
    }
  }
  return true;
}

bool contract_looks_like_grouped_yolov26_pose(const BoxDecodeStaticContract& contract) {
  if (contract.tensors.empty() || (contract.tensors.size() % 3U) != 0U) {
    return false;
  }
  const std::size_t heads = contract.tensors.size() / 3U;
  if (heads == 0U) {
    return false;
  }
  for (std::size_t i = 0; i < heads; ++i) {
    const auto& bbox = contract.tensors[i];
    const auto& scores = contract.tensors[i + heads];
    const auto& keypoints = contract.tensors[i + (2U * heads)];
    if (bbox.input_shape.size() < 3U || scores.input_shape.size() < 3U ||
        keypoints.input_shape.size() < 3U) {
      return false;
    }
    if (bbox.input_shape[0] != scores.input_shape[0] ||
        bbox.input_shape[1] != scores.input_shape[1] ||
        bbox.input_shape[0] != keypoints.input_shape[0] ||
        bbox.input_shape[1] != keypoints.input_shape[1]) {
      return false;
    }
    const int bbox_depth =
        bbox.slice_shape.size() >= 3U ? bbox.slice_shape[2] : bbox.input_shape[2];
    const int score_depth =
        scores.slice_shape.size() >= 3U ? scores.slice_shape[2] : scores.input_shape[2];
    const int keypoint_depth =
        keypoints.slice_shape.size() >= 3U ? keypoints.slice_shape[2] : keypoints.input_shape[2];
    if (bbox_depth != 4 || bbox.input_shape[2] < bbox_depth || score_depth != 1 ||
        scores.input_shape[2] < score_depth || keypoint_depth != 51 ||
        keypoints.input_shape[2] < keypoint_depth) {
      return false;
    }
  }
  return true;
}

void apply_yolov26_static_contract_overrides(BoxDecodeStaticContract* contract) {
  if (!contract || (contract->decode_type != BoxDecodeType::YoloV26 &&
                    contract->decode_type != BoxDecodeType::YoloV26Pose &&
                    contract->decode_type != BoxDecodeType::YoloV26Seg)) {
    return;
  }

  // YOLO26 emits raw l/t/r/b distance heads plus class logits. Do not inherit
  // YOLOv8 quant-probability heuristics from an auto-extracted MPK route.
  contract->decode_type_option = BoxDecodeTypeOption::GroupedByRoleLogit;
  contract->score_activation = BoxDecodeScoreActivation::Sigmoid;
  if (contract->decode_type == BoxDecodeType::YoloV26Pose && contract->num_classes <= 0) {
    contract->num_classes = 1;
  }

  if (contract->decode_type == BoxDecodeType::YoloV26Pose &&
      contract_looks_like_grouped_yolov26_pose(*contract)) {
    const std::size_t heads = contract->tensors.size() / 3U;
    for (std::size_t i = 0; i < heads; ++i) {
      const std::string bbox_name = "bbox_" + std::to_string(i);
      auto& bbox = contract->tensors[i];
      bbox.logical_name = bbox_name;
      bbox.backend_name = bbox_name;
      if (i < contract->tensor_names.size()) {
        contract->tensor_names[i] = bbox_name;
      }

      const std::string score_name = "class_logit_" + std::to_string(i);
      auto& score = contract->tensors[i + heads];
      score.logical_name = score_name;
      score.backend_name = score_name;
      if ((i + heads) < contract->tensor_names.size()) {
        contract->tensor_names[i + heads] = score_name;
      }

      const std::string keypoint_name = "keypoint_" + std::to_string(i);
      auto& keypoint = contract->tensors[i + (2U * heads)];
      keypoint.logical_name = keypoint_name;
      keypoint.backend_name = keypoint_name;
      if ((i + (2U * heads)) < contract->tensor_names.size()) {
        contract->tensor_names[i + (2U * heads)] = keypoint_name;
      }
    }
    return;
  }

  if (!contract_looks_like_grouped_yolov26(*contract)) {
    return;
  }

  const std::size_t heads = contract->tensors.size() / 2U;
  for (std::size_t i = 0; i < heads; ++i) {
    const std::string bbox_name = "bbox_" + std::to_string(i);
    auto& bbox = contract->tensors[i];
    bbox.logical_name = bbox_name;
    bbox.backend_name = bbox_name;
    if (i < contract->tensor_names.size()) {
      contract->tensor_names[i] = bbox_name;
    }

    const std::string score_name = "class_logit_" + std::to_string(i);
    auto& score = contract->tensors[i + heads];
    score.logical_name = score_name;
    score.backend_name = score_name;
    if ((i + heads) < contract->tensor_names.size()) {
      contract->tensor_names[i + heads] = score_name;
    }
  }
}

int logical_channel_depth(const BoxDecodeTensorStaticContract& tensor) {
  if (tensor.slice_shape.size() >= 3U && tensor.slice_shape.back() > 0) {
    return tensor.slice_shape.back();
  }
  if (tensor.input_shape.size() >= 3U && tensor.input_shape.back() > 0) {
    return tensor.input_shape.back();
  }
  return 0;
}

int yolov5_packed_channel_depth(const BoxDecodeTensorStaticContract& tensor) {
  if (tensor.source_storage_kind == BoxDecodeSourceStorageKind::PackedCBlock ||
      tensor.source_storage_kind == BoxDecodeSourceStorageKind::PackedHwcC16) {
    return tensor.input_shape.size() >= 3U ? tensor.input_shape.back() : 0;
  }
  return logical_channel_depth(tensor);
}

bool tensor_name_looks_objectness_logit(std::string raw) {
  for (char& ch : raw) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return raw.find("obj_logit") != std::string::npos ||
         raw.find("objectness_logit") != std::string::npos ||
         raw.find("object_logit") != std::string::npos;
}

std::string lower_string(std::string raw) {
  for (char& ch : raw) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return raw;
}

bool contains_any_token(const std::string& raw, std::initializer_list<const char*> needles) {
  const std::string name = lower_string(raw);
  for (const char* needle : needles) {
    if (needle && *needle && name.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool tensor_name_contains_any(const BoxDecodeTensorStaticContract& tensor,
                              std::initializer_list<const char*> needles) {
  return contains_any_token(tensor.logical_name, needles) ||
         contains_any_token(tensor.backend_name, needles) ||
         contains_any_token(tensor.source_segment_name, needles);
}

bool tensor_name_is_non_class_role(const BoxDecodeTensorStaticContract& tensor) {
  return tensor_name_contains_any(tensor, {"bbox", "box", "reg", "extent", "xy", "ltrb", "keypoint",
                                           "kpt", "mask", "proto", "prototype", "coef", "coeff"});
}

bool tensor_name_is_class_role(const BoxDecodeTensorStaticContract& tensor) {
  if (tensor_name_looks_objectness_logit(tensor.logical_name) ||
      tensor_name_looks_objectness_logit(tensor.backend_name) ||
      tensor_name_looks_objectness_logit(tensor.source_segment_name) ||
      tensor_name_is_non_class_role(tensor)) {
    return false;
  }
  return tensor_name_contains_any(
      tensor, {"class_logit", "class_logits", "class_prob", "class_probability", "class_score",
               "cls_logit", "cls_logits", "cls_prob", "cls_probability", "cls_score"});
}

struct TensorHwc {
  int h = 0;
  int w = 0;
  int c = 0;
  int semantic_c = 0;
};

std::optional<TensorHwc> tensor_hwc(const BoxDecodeTensorStaticContract& tensor) {
  if (tensor.input_shape.size() < 3U) {
    return std::nullopt;
  }
  const auto rank = tensor.input_shape.size();
  const int h = tensor.input_shape[rank - 3U];
  const int w = tensor.input_shape[rank - 2U];
  const int c = tensor.input_shape[rank - 1U];
  if (h <= 0 || w <= 0 || c <= 0) {
    return std::nullopt;
  }
  int semantic_c = c;
  if (tensor.slice_shape.size() >= 3U) {
    const int slice_c = tensor.slice_shape[tensor.slice_shape.size() - 1U];
    if (slice_c > 0 && slice_c <= c) {
      semantic_c = slice_c;
    }
  }
  return TensorHwc{h, w, c, semantic_c};
}

std::optional<TensorHwc> yolov5_head_hwc(const BoxDecodeTensorStaticContract& tensor) {
  auto head = tensor_hwc(tensor);
  if (!head.has_value()) {
    return std::nullopt;
  }

  if (tensor.source_storage_kind == BoxDecodeSourceStorageKind::PackedCBlock ||
      tensor.source_storage_kind == BoxDecodeSourceStorageKind::PackedHwcC16) {
    // Packed slice_shape describes a storage tile. The input shape remains the
    // logical YOLO head geometry and channel depth.
    head->semantic_c = yolov5_packed_channel_depth(tensor);
    return head;
  }

  if (tensor.source_storage_kind == BoxDecodeSourceStorageKind::DenseHwcPhysical &&
      tensor.slice_shape.size() >= 3U) {
    const auto rank = tensor.slice_shape.size();
    const int logical_h = tensor.slice_shape[rank - 3U];
    const int logical_w = tensor.slice_shape[rank - 2U];
    const int logical_c = tensor.slice_shape[rank - 1U];
    if (logical_h <= 0 || logical_w <= 0 || logical_c <= 0 || logical_h > head->h ||
        logical_w > head->w || logical_c > head->c) {
      return std::nullopt;
    }
    head->h = logical_h;
    head->w = logical_w;
    head->semantic_c = logical_c;
  }
  return head;
}

bool same_hw(const TensorHwc& lhs, const TensorHwc& rhs) {
  return lhs.h == rhs.h && lhs.w == rhs.w;
}

void apply_yolov5_static_contract_overrides(BoxDecodeStaticContract* contract) {
  if (!contract || contract->decode_type != BoxDecodeType::YoloV5) {
    return;
  }
  if (contract->decode_type_option != BoxDecodeTypeOption::Auto &&
      contract->decode_type_option != BoxDecodeTypeOption::PackedPerHead) {
    throw std::invalid_argument("BoxDecode(YOLOv5) requires packed-per-head P3/P4/P5 tensors");
  }
  contract->decode_type_option = BoxDecodeTypeOption::PackedPerHead;
  contract->score_activation = BoxDecodeScoreActivation::Sigmoid;

  if (contract->tensors.size() != 3U) {
    throw std::invalid_argument(
        "BoxDecode(YOLOv5) requires exactly three packed tensors ordered P3, P4, P5");
  }

  std::array<TensorHwc, 3> heads{};
  std::optional<int> classes;
  for (std::size_t i = 0; i < heads.size(); ++i) {
    auto head = yolov5_head_hwc(contract->tensors[i]);
    if (!head.has_value()) {
      throw std::invalid_argument("BoxDecode(YOLOv5) packed tensor[" + std::to_string(i) +
                                  "] must have valid HWC geometry and depth=3*(num_classes+5)");
    }
    if ((head->semantic_c % 3) != 0) {
      throw std::invalid_argument("BoxDecode(YOLOv5) packed tensor[" + std::to_string(i) +
                                  "] must have valid HWC geometry and depth=3*(num_classes+5)");
    }
    const int candidate_classes = (head->semantic_c / 3) - 5;
    if (candidate_classes <= 0 || (classes.has_value() && *classes != candidate_classes)) {
      throw std::invalid_argument(
          "BoxDecode(YOLOv5) packed head depths must encode one consistent positive class count");
    }
    classes = candidate_classes;
    heads[i] = *head;
  }

  if (heads[0].h != 2 * heads[1].h || heads[0].w != 2 * heads[1].w ||
      heads[1].h != 2 * heads[2].h || heads[1].w != 2 * heads[2].w) {
    throw std::invalid_argument(
        "BoxDecode(YOLOv5) tensors must be ordered P3/P4/P5 with stride-8/16/32 grids");
  }
}

std::optional<int> consistent_positive_depth(std::optional<int> current, int candidate) {
  if (candidate <= 0) {
    return current;
  }
  if (!current.has_value()) {
    return candidate;
  }
  if (*current == candidate) {
    return current;
  }
  return std::nullopt;
}

int infer_named_class_depth(const BoxDecodeStaticContract& contract) {
  std::optional<int> inferred;
  bool saw_class_tensor = false;
  for (const auto& tensor : contract.tensors) {
    if (!tensor_name_is_class_role(tensor)) {
      continue;
    }
    const int c = logical_channel_depth(tensor);
    if (c <= 0) {
      continue;
    }
    saw_class_tensor = true;
    const auto next = consistent_positive_depth(inferred, c);
    if (!next.has_value() && inferred.has_value()) {
      // Semantic class heads in supported YOLO routes are repeated with the same class depth.
      // If naming disagrees, do not guess from names; let the geometric family fallback try.
      return 0;
    }
    inferred = next;
  }
  return saw_class_tensor && inferred.has_value() ? *inferred : 0;
}

int infer_grouped_dfl_class_depth(const BoxDecodeStaticContract& contract) {
  if (contract.tensors.size() < 2U || (contract.tensors.size() % 2U) != 0U) {
    return 0;
  }
  const std::size_t heads = contract.tensors.size() / 2U;
  std::optional<int> classes;
  for (std::size_t i = 0; i < heads; ++i) {
    const auto reg = tensor_hwc(contract.tensors[i]);
    const auto cls = tensor_hwc(contract.tensors[i + heads]);
    if (!reg.has_value() || !cls.has_value() || !same_hw(*reg, *cls) || reg->semantic_c < 16 ||
        (reg->semantic_c % 4) != 0 || cls->semantic_c <= 0) {
      return 0;
    }
    const auto next = consistent_positive_depth(classes, cls->semantic_c);
    if (!next.has_value() && classes.has_value()) {
      return 0;
    }
    classes = next;
  }
  return classes.value_or(0);
}

int infer_yolov26_grouped_class_depth(const BoxDecodeStaticContract& contract) {
  if (contract_looks_like_grouped_yolov26_pose(contract)) {
    return 1;
  }
  if (contract_looks_like_grouped_yolov26(contract)) {
    const std::size_t heads = contract.tensors.size() / 2U;
    std::optional<int> classes;
    for (std::size_t i = 0; i < heads; ++i) {
      const auto cls = tensor_hwc(contract.tensors[i + heads]);
      if (!cls.has_value()) {
        return 0;
      }
      const auto next = consistent_positive_depth(classes, cls->semantic_c);
      if (!next.has_value() && classes.has_value()) {
        return 0;
      }
      classes = next;
    }
    return classes.value_or(0);
  }
  return 0;
}

int infer_yolov26_seg_grouped_class_depth(const BoxDecodeStaticContract& contract) {
  if (contract.tensors.size() < 4U || ((contract.tensors.size() - 1U) % 3U) != 0U) {
    return 0;
  }
  const std::size_t heads = (contract.tensors.size() - 1U) / 3U;
  if (heads == 0U) {
    return 0;
  }
  const auto proto = tensor_hwc(contract.tensors.back());
  if (!proto.has_value() || proto->semantic_c != 32) {
    return 0;
  }
  std::optional<int> classes;
  for (std::size_t i = 0; i < heads; ++i) {
    const auto bbox = tensor_hwc(contract.tensors[i]);
    const auto cls = tensor_hwc(contract.tensors[i + heads]);
    const auto mask = tensor_hwc(contract.tensors[i + (2U * heads)]);
    if (!bbox.has_value() || !cls.has_value() || !mask.has_value() || !same_hw(*bbox, *cls) ||
        !same_hw(*bbox, *mask) || bbox->semantic_c != 4 || cls->semantic_c <= 0 ||
        mask->semantic_c != 32) {
      return 0;
    }
    const auto next = consistent_positive_depth(classes, cls->semantic_c);
    if (!next.has_value() && classes.has_value()) {
      return 0;
    }
    classes = next;
  }
  return classes.value_or(0);
}

int infer_yolov6_interleaved_class_depth(const BoxDecodeStaticContract& contract) {
  if (contract.tensors.size() < 2U || (contract.tensors.size() % 2U) != 0U) {
    return 0;
  }
  std::optional<int> classes;
  for (std::size_t i = 0; i < contract.tensors.size(); i += 2U) {
    const auto bbox = tensor_hwc(contract.tensors[i]);
    const auto cls = tensor_hwc(contract.tensors[i + 1U]);
    if (!bbox.has_value() || !cls.has_value() || !same_hw(*bbox, *cls) || bbox->semantic_c != 4 ||
        cls->semantic_c <= 0) {
      return 0;
    }
    const auto next = consistent_positive_depth(classes, cls->semantic_c);
    if (!next.has_value() && classes.has_value()) {
      return 0;
    }
    classes = next;
  }
  return classes.value_or(0);
}

int infer_yolox_interleaved_class_depth(const BoxDecodeStaticContract& contract) {
  if (contract.tensors.size() < 3U || (contract.tensors.size() % 3U) != 0U) {
    return 0;
  }
  std::optional<int> classes;
  for (std::size_t i = 0; i < contract.tensors.size(); i += 3U) {
    const auto bbox = tensor_hwc(contract.tensors[i]);
    const auto obj = tensor_hwc(contract.tensors[i + 1U]);
    const auto cls = tensor_hwc(contract.tensors[i + 2U]);
    if (!bbox.has_value() || !obj.has_value() || !cls.has_value() || !same_hw(*bbox, *obj) ||
        !same_hw(*bbox, *cls) || bbox->semantic_c != 4 || obj->semantic_c != 1 ||
        cls->semantic_c <= 0) {
      return 0;
    }
    const auto next = consistent_positive_depth(classes, cls->semantic_c);
    if (!next.has_value() && classes.has_value()) {
      return 0;
    }
    classes = next;
  }
  return classes.value_or(0);
}

// Grouped by role: first half loc (4*A), second half conf (num_classes*A), paired by
// feature level with one class count across levels.
int infer_ssd_grouped_class_depth(const BoxDecodeStaticContract& contract) {
  if (const auto* recipe = find_ssd_recipe_descriptor(contract.ssd_recipe_id)) {
    return recipe->encoded_class_count;
  }
  if (contract.tensors.size() < 2U || (contract.tensors.size() % 2U) != 0U) {
    return 0;
  }
  const std::size_t levels = contract.tensors.size() / 2U;
  std::optional<int> classes;
  for (std::size_t i = 0; i < levels; ++i) {
    const auto loc = tensor_hwc(contract.tensors[i]);
    const auto conf = tensor_hwc(contract.tensors[i + levels]);
    if (!loc.has_value() || !conf.has_value() || !same_hw(*loc, *conf) || loc->semantic_c < 4 ||
        (loc->semantic_c % 4) != 0) {
      return 0;
    }
    const int priors_per_cell = loc->semantic_c / 4;
    if (priors_per_cell <= 0 || (conf->semantic_c % priors_per_cell) != 0) {
      return 0;
    }
    const int candidate = conf->semantic_c / priors_per_cell;
    if (candidate <= 0) {
      return 0;
    }
    const auto next = consistent_positive_depth(classes, candidate);
    if (!next.has_value() && classes.has_value()) {
      return 0;
    }
    classes = next;
  }
  return classes.value_or(0);
}

int infer_packed_yolo_class_depth(const BoxDecodeStaticContract& contract) {
  std::optional<int> classes;
  for (const auto& tensor : contract.tensors) {
    const int c = contract.decode_type == BoxDecodeType::YoloV5
                      ? yolov5_packed_channel_depth(tensor)
                      : logical_channel_depth(tensor);
    if (c <= 0 || (c % 3) != 0) {
      return 0;
    }
    const int candidate = (c / 3) - 5;
    if (candidate <= 0) {
      return 0;
    }
    const auto next = consistent_positive_depth(classes, candidate);
    if (!next.has_value() && classes.has_value()) {
      return 0;
    }
    classes = next;
  }
  return classes.value_or(0);
}

bool decode_type_is_grouped_dfl_yolo(BoxDecodeType type) {
  return type == BoxDecodeType::YoloV8 || type == BoxDecodeType::YoloV8Seg ||
         type == BoxDecodeType::YoloV8Pose || type == BoxDecodeType::YoloV9 ||
         type == BoxDecodeType::YoloV9Seg || type == BoxDecodeType::YoloV10 ||
         type == BoxDecodeType::YoloV10Seg;
}

bool decode_type_is_pose_yolo(BoxDecodeType type) {
  return type == BoxDecodeType::YoloV8Pose || type == BoxDecodeType::YoloV26Pose;
}

bool decode_type_is_yolov26_family(BoxDecodeType type) {
  return type == BoxDecodeType::YoloV26 || type == BoxDecodeType::YoloV26Pose ||
         type == BoxDecodeType::YoloV26Seg;
}

bool decode_type_is_packed_yolo(BoxDecodeType type) {
  return type == BoxDecodeType::Yolo || type == BoxDecodeType::YoloV5 ||
         type == BoxDecodeType::YoloV5Seg || type == BoxDecodeType::YoloV7 ||
         type == BoxDecodeType::YoloV7Seg;
}

int infer_raw_yolo_class_depth(const BoxDecodeStaticContract& contract);

int infer_boxdecode_num_classes_from_contract(const BoxDecodeStaticContract& contract) {
  if (box_decode_type_is_ssd_family(contract.decode_type)) {
    // SSD class count is derived from the loc/conf head geometry, not head names:
    // confidence heads pack num_classes * priors-per-cell channels, so a name-based
    // guess would over-count by the prior multiplier.
    if (const int classes = infer_ssd_grouped_class_depth(contract); classes > 0) {
      return classes;
    }
    return contract.num_classes;
  }
  if (contract.decode_type == BoxDecodeType::YoloV5) {
    // A packed raw head may legitimately have a class-like tensor name, but its
    // channel depth is 3 * (classes + 5), not the class count itself.
    if (const int classes = infer_packed_yolo_class_depth(contract); classes > 0) {
      return classes;
    }
  }
  if (const int named = infer_named_class_depth(contract); named > 0) {
    return named;
  }

  switch (contract.decode_type) {
  case BoxDecodeType::YoloV26:
  case BoxDecodeType::YoloV26Pose:
  case BoxDecodeType::YoloV26Seg:
    if (contract.decode_type == BoxDecodeType::YoloV26Seg) {
      if (const int classes = infer_yolov26_seg_grouped_class_depth(contract); classes > 0) {
        return classes;
      }
    }
    if (const int classes = infer_yolov26_grouped_class_depth(contract); classes > 0) {
      return classes;
    }
    break;
  case BoxDecodeType::YoloV6:
    if (const int classes = infer_yolov6_interleaved_class_depth(contract); classes > 0) {
      return classes;
    }
    break;
  case BoxDecodeType::YoloX:
    if (const int classes = infer_yolox_interleaved_class_depth(contract); classes > 0) {
      return classes;
    }
    break;
  default:
    if (decode_type_is_grouped_dfl_yolo(contract.decode_type)) {
      if (const int classes = infer_grouped_dfl_class_depth(contract); classes > 0) {
        return classes;
      }
    }
    if (decode_type_is_packed_yolo(contract.decode_type)) {
      if (const int classes = infer_packed_yolo_class_depth(contract); classes > 0) {
        return classes;
      }
    }
    break;
  }

  if (const int raw = infer_raw_yolo_class_depth(contract); raw > 0) {
    return raw;
  }

  return contract.num_classes;
}

int resolve_boxdecode_num_classes(const BoxDecodeStaticContract& contract, int user_num_classes,
                                  const char* context) {
  if (decode_type_is_pose_yolo(contract.decode_type)) {
    if (user_num_classes > 0 && user_num_classes != 1) {
      throw std::invalid_argument(std::string(context ? context : "BoxDecode") +
                                  " pose decode requires num_classes=1 when specified");
    }
    return 1;
  }

  const int inferred = infer_boxdecode_num_classes_from_contract(contract);
  if (user_num_classes > 0) {
    validate_ssd_num_classes(contract.decode_type, contract.ssd_recipe_id, user_num_classes,
                             inferred, context);
    const int resolved = resolve_boxdecode_num_classes_override(contract.decode_type, inferred,
                                                                user_num_classes, context);
    // Narrowing is the documented SSD contract, so it is not a mismatch there.
    const bool ssd_narrowing = box_decode_type_is_ssd_family(contract.decode_type);
    if (inferred > 0 && user_num_classes != inferred && !ssd_narrowing) {
      std::fprintf(stderr,
                   "[WARN] %s num_classes mismatch: user=%d inferred_from_mpk=%d decode_type=%s. "
                   "Using user value.\n",
                   context ? context : "BoxDecode", user_num_classes, inferred,
                   box_decode_type_token(contract.decode_type));
    }
    return resolved;
  }
  return inferred;
}

void finalize_superpoint_contract(BoxDecodeStaticContract* contract, const char* context) {
  if (!contract || contract->decode_type != BoxDecodeType::SuperPoint) {
    return;
  }
  auto& sp = contract->superpoint;
  resolve_default_superpoint_profile(&sp);
  canonicalize_schema0_superpoint_representations(&sp);
  if (const auto metadata_error =
          validate_superpoint_static_metadata(sp, /*require_resolved_profile=*/true)) {
    throw std::invalid_argument(std::string(context ? context : "BoxDecode") +
                                "(SuperPoint): " + *metadata_error);
  }
  if (sp.profile == SuperPointProfile::PaperBicubicV1) {
    throw std::invalid_argument(
        std::string(context ? context : "BoxDecode") +
        "(SuperPoint): paper-bicubic-v1 is reserved but not production-defined. "
        "Select lightglue-v1, magic-leap-demo-v1, or a65-v1.");
  }
  if (sp.nms_radius < -1 || sp.border_margin < -1) {
    throw std::invalid_argument(std::string(context ? context : "BoxDecode") +
                                "(SuperPoint): radius and border must be -1 (profile default) "
                                "or non-negative; zero is valid");
  }
  if (sp.nms_radius == -1) {
    sp.nms_radius = 4;
  }
  if (sp.border_margin == -1) {
    sp.border_margin = sp.profile == SuperPointProfile::A65V1 ? 0 : 4;
  }
  if (contract->tensors.size() != 2U) {
    throw std::invalid_argument(std::string(context ? context : "BoxDecode") +
                                "(SuperPoint): exactly two inputs are required");
  }
  auto logical_channels = [](const BoxDecodeTensorStaticContract& tensor) {
    if (tensor.source_storage_kind == BoxDecodeSourceStorageKind::PackedCBlock ||
        tensor.source_storage_kind == BoxDecodeSourceStorageKind::PackedHwcC16) {
      return tensor.input_shape.size() >= 3U ? tensor.input_shape.back() : 0;
    }
    return tensor.slice_shape.size() >= 3U
               ? tensor.slice_shape.back()
               : (tensor.input_shape.size() >= 3U ? tensor.input_shape.back() : 0);
  };
  int detector_index = -1;
  int descriptor_index = -1;
  auto name_matches = [](const BoxDecodeTensorStaticContract& tensor, const std::string& id) {
    return !id.empty() && (tensor.logical_name == id || tensor.backend_name == id ||
                           tensor.source_segment_name == id);
  };
  auto bind_authored_id = [&](const std::string& id, const char* role_name) {
    if (id.empty()) {
      return -1;
    }
    int match = -1;
    for (std::size_t i = 0; i < contract->tensors.size(); ++i) {
      if (!name_matches(contract->tensors[i], id)) {
        continue;
      }
      if (match >= 0) {
        throw std::invalid_argument(std::string(context ? context : "BoxDecode") +
                                    "(SuperPoint): authored " + role_name + " tensor ID '" + id +
                                    "' is ambiguous");
      }
      match = static_cast<int>(i);
    }
    if (match < 0) {
      throw std::invalid_argument(std::string(context ? context : "BoxDecode") +
                                  "(SuperPoint): authored " + role_name + " tensor ID '" + id +
                                  "' does not match an input tensor");
    }
    return match;
  };
  detector_index = bind_authored_id(sp.detector_tensor_id, "detector");
  descriptor_index = bind_authored_id(sp.descriptor_tensor_id, "descriptor");
  int detector_role_count = 0;
  int descriptor_role_count = 0;
  for (std::size_t i = 0; i < contract->tensors.size(); ++i) {
    if (contract->tensors[i].role == BoxDecodeTensorRole::DetectorLogits) {
      if (!sp.detector_tensor_id.empty() && detector_index >= 0 &&
          detector_index != static_cast<int>(i)) {
        throw std::invalid_argument(std::string(context ? context : "BoxDecode") +
                                    "(SuperPoint): detector tensor ID conflicts with the "
                                    "explicit tensor role");
      }
      detector_index = static_cast<int>(i);
      ++detector_role_count;
    } else if (contract->tensors[i].role == BoxDecodeTensorRole::DescriptorGrid) {
      if (!sp.descriptor_tensor_id.empty() && descriptor_index >= 0 &&
          descriptor_index != static_cast<int>(i)) {
        throw std::invalid_argument(std::string(context ? context : "BoxDecode") +
                                    "(SuperPoint): descriptor tensor ID conflicts with the "
                                    "explicit tensor role");
      }
      descriptor_index = static_cast<int>(i);
      ++descriptor_role_count;
    }
  }
  if (detector_role_count > 1 || descriptor_role_count > 1) {
    throw std::invalid_argument(std::string(context ? context : "BoxDecode") +
                                "(SuperPoint): tensor roles must contain exactly one detector "
                                "and one descriptor, not duplicate explicit roles");
  }
  if (detector_index < 0) {
    for (std::size_t i = 0; i < contract->tensors.size(); ++i) {
      if (logical_channels(contract->tensors[i]) == 65) {
        if (detector_index >= 0) {
          throw std::invalid_argument(std::string(context ? context : "BoxDecode") +
                                      "(SuperPoint): ambiguous 65-channel detector inputs");
        }
        detector_index = static_cast<int>(i);
      }
    }
  }
  if (detector_index >= 0 && descriptor_index < 0) {
    descriptor_index = 1 - detector_index;
  }
  if (detector_index < 0 || descriptor_index < 0 || detector_index == descriptor_index) {
    throw std::invalid_argument(std::string(context ? context : "BoxDecode") +
                                "(SuperPoint): detector and descriptor roles are ambiguous");
  }
  auto& detector = contract->tensors[static_cast<std::size_t>(detector_index)];
  auto& descriptor = contract->tensors[static_cast<std::size_t>(descriptor_index)];
  detector.role = BoxDecodeTensorRole::DetectorLogits;
  descriptor.role = BoxDecodeTensorRole::DescriptorGrid;
  if (logical_channels(detector) != 65 || logical_channels(descriptor) <= 0) {
    throw std::invalid_argument(std::string(context ? context : "BoxDecode") +
                                "(SuperPoint): detector must have 65 channels and descriptor "
                                "dimension must be positive");
  }
  auto logical_geometry = [](const BoxDecodeTensorStaticContract& tensor) {
    // input_shape carries the logical tensor H/W exposed by the MLA output.
    // slice_shape is the physical CBlock tile geometry and may legitimately
    // differ between detector and descriptor even when their logical coarse
    // grids are both [60,80].
    const auto& shape = tensor.input_shape;
    return shape.size() >= 3U
               ? std::pair<int, int>{shape[shape.size() - 3U], shape[shape.size() - 2U]}
               : std::pair<int, int>{0, 0};
  };
  const auto detector_geometry = logical_geometry(detector);
  const auto descriptor_geometry = logical_geometry(descriptor);
  if (detector_geometry.first <= 0 || detector_geometry.second <= 0 ||
      detector_geometry != descriptor_geometry) {
    throw std::invalid_argument(std::string(context ? context : "BoxDecode") +
                                "(SuperPoint): detector and descriptor coarse H/W geometry "
                                "must be positive and identical");
  }
  auto supported_input_dtype = [](std::string dtype) {
    std::transform(dtype.begin(), dtype.end(), dtype.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return dtype == "INT8" || dtype == "BF16" || dtype == "BFLOAT16" || dtype == "FP32" ||
           dtype == "FLOAT32";
  };
  if (!supported_input_dtype(detector.data_type) || !supported_input_dtype(descriptor.data_type)) {
    throw std::invalid_argument(std::string(context ? context : "BoxDecode") +
                                "(SuperPoint): detector and descriptor input dtypes must each "
                                "be INT8, BF16, or FP32");
  }
  if (sp.descriptor_dim > 0 && sp.descriptor_dim != logical_channels(descriptor)) {
    throw std::invalid_argument(std::string(context ? context : "BoxDecode") +
                                "(SuperPoint): descriptor dimension conflicts with tensor "
                                "geometry");
  }
  sp.descriptor_dim = logical_channels(descriptor);
  if (sp.nms_radius < 0 || sp.border_margin < 0 || sp.cell_stride <= 0 ||
      sp.descriptor_stride <= 0 || sp.descriptor_dim <= 0) {
    throw std::invalid_argument(std::string(context ? context : "BoxDecode") +
                                "(SuperPoint): invalid radius, border, stride, or descriptor "
                                "dimension");
  }
  if (sp.descriptor_output_dtype != TensorDType::Int8 &&
      sp.descriptor_output_dtype != TensorDType::BFloat16 &&
      sp.descriptor_output_dtype != TensorDType::Float32) {
    throw std::invalid_argument(std::string(context ? context : "BoxDecode") +
                                "(SuperPoint): descriptor output dtype must be INT8, BF16, or "
                                "FP32");
  }
  if (sp.output_format == SuperPointOutputFormat::LegacyA65InterleavedV0 &&
      (sp.descriptor_dim != 256 || sp.descriptor_output_dtype != TensorDType::Int8)) {
    throw std::invalid_argument(std::string(context ? context : "BoxDecode") +
                                "(SuperPoint): legacy A65 V0 output requires 256-dimensional "
                                "INT8 descriptors");
  }
}

void finalize_superpoint_decode_controls(SuperPointProfile profile, double nms_iou_threshold,
                                         double* detection_threshold, int* topk,
                                         const char* context) {
  const std::string prefix = std::string(context ? context : "BoxDecode") + "(SuperPoint): ";
  if (!detection_threshold || !topk) {
    throw std::invalid_argument(prefix + "missing decode controls");
  }
  if (nms_iou_threshold != 0.0) {
    throw std::invalid_argument(prefix + "nms_iou_threshold is not applicable; use "
                                         "BoxDecodeOptions.superpoint.nms_radius");
  }
  if (*detection_threshold == 0.0) {
    *detection_threshold = superpoint_default_detection_threshold(profile);
  }
  if (!(*detection_threshold >= 0.0 && *detection_threshold <= 1.0)) {
    throw std::invalid_argument(prefix + "detection_threshold must be in [0, 1]");
  }
  if (*topk == 0) {
    *topk = kSuperPointDefaultTopK;
  }
  if (*topk < 0) {
    throw std::invalid_argument(prefix + "top_k must be positive, or zero to use the default");
  }
}

int infer_raw_yolo_class_depth(const BoxDecodeStaticContract& contract) {
  int best = 0;
  for (const auto& tensor : contract.tensors) {
    const int c = logical_channel_depth(tensor);
    if (c <= 4) {
      continue;
    }
    if (tensor_name_looks_objectness_logit(tensor.logical_name) ||
        tensor_name_looks_objectness_logit(tensor.backend_name) ||
        tensor_name_looks_objectness_logit(tensor.source_segment_name)) {
      continue;
    }
    best = std::max(best, c);
  }
  return best;
}

void apply_raw_yolov6_yolox_static_contract_overrides(BoxDecodeStaticContract* contract) {
  if (!contract || (contract->decode_type != BoxDecodeType::YoloV6 &&
                    contract->decode_type != BoxDecodeType::YoloX)) {
    return;
  }
  contract->score_activation = BoxDecodeScoreActivation::Sigmoid;
  if (contract->decode_type_option == BoxDecodeTypeOption::Auto) {
    contract->decode_type_option = contract->decode_type == BoxDecodeType::YoloX
                                       ? BoxDecodeTypeOption::Split3Interleaved
                                       : BoxDecodeTypeOption::InterleavedByHeadLogit;
  }
  if (contract->num_classes <= 0) {
    contract->num_classes = infer_boxdecode_num_classes_from_contract(*contract);
  }
}

void apply_ssd_static_contract_overrides(BoxDecodeStaticContract* contract) {
  if (!contract || !box_decode_type_is_ssd_family(contract->decode_type)) {
    return;
  }
  // Any other layout token would pair loc/conf differently than validated here.
  if (contract->decode_type_option == BoxDecodeTypeOption::Auto) {
    contract->decode_type_option = BoxDecodeTypeOption::GroupedByRole;
  } else if (contract->decode_type_option != BoxDecodeTypeOption::GroupedByRole) {
    throw std::invalid_argument(
        std::string("SSD BoxDecode supports only the grouped-by-role head layout, but got '") +
        box_decode_type_option_token(contract->decode_type_option) +
        "'. Use BoxDecodeTypeOption::Auto or GroupedByRole.");
  }
  const SsdRecipeDescriptor& recipe = resolve_ssd_recipe_descriptor(*contract);
  if (contract->score_activation != BoxDecodeScoreActivation::Unknown &&
      contract->score_activation != recipe.activation) {
    throw std::invalid_argument(
        "SSD BoxDecode: declared score activation conflicts with resolved profile '" +
        std::string(ssd_recipe_id_token(recipe.id)) + "'");
  }
  contract->ssd_recipe_id = recipe.id;
  contract->score_activation = recipe.activation;
  const int inferred_classes = infer_ssd_grouped_class_depth(*contract);
  if (contract->num_classes <= 0) {
    contract->num_classes = inferred_classes;
  } else {
    validate_ssd_num_classes(contract->decode_type, contract->ssd_recipe_id, contract->num_classes,
                             inferred_classes, "BoxDecode");
  }
  contract->ssd_class_selection = {
      inferred_classes,
      contract->num_classes,
      recipe.class_count_policy == SsdClassCountPolicy::AllowPrefixNarrowing
          ? SsdClassSelectionKind::PrefixFromZero
          : SsdClassSelectionKind::Exact,
  };
}

std::string resolve_boxdecode_input_dtype(const plugin_contracts::BoxDecodeContractSubset& subset) {
  std::string dtype;
  for (const auto& logical : subset.logical_inputs) {
    if (logical.dtype.empty()) {
      throw std::invalid_argument("boxdecode compiled contract requires logical input dtype");
    }
    if (dtype.empty()) {
      dtype = logical.dtype;
      continue;
    }
    if (dtype != logical.dtype && subset.decode_type == BoxDecodeType::SuperPoint) {
      dtype = "MIXED";
      continue;
    }
    if (dtype != logical.dtype && dtype != "MIXED") {
      throw std::invalid_argument(
          "boxdecode compiled contract requires a homogeneous logical input dtype");
    }
  }
  return dtype;
}

void populate_boxdecode_node_contract_common(
    const std::string& node_kind, const std::string& plugin_kind, const std::string& element_name,
    const std::string& logical_stage_id, const NodeContractDefinition& definition,
    CompiledBoxDecodeContract compiled, CompiledNodeContract* out) {
  out->node_kind = node_kind;
  out->plugin_kind = plugin_kind.empty() ? "boxdecode" : plugin_kind;
  out->element_name = element_name;
  out->logical_stage_id = logical_stage_id.empty() ? element_name : logical_stage_id;
  out->definition = definition;
  compiled.runtime_contract.plugin_kind = out->plugin_kind;
  out->boxdecode = std::move(compiled);
  out->renderable = true;
}

} // namespace

SsdModelFrame ssd_expected_model_frame(const BoxDecodeStaticContract& contract) {
  if (!box_decode_type_is_ssd_family(contract.decode_type)) {
    return {};
  }
  const auto& descriptor = resolve_ssd_recipe_descriptor(contract);
  return SsdModelFrame{descriptor.model_width, descriptor.model_height};
}

void validate_ssd_num_classes(BoxDecodeType decode_type, SsdRecipeId recipe_id, int requested,
                              int encoded, const char* context) {
  if (!box_decode_type_is_ssd_family(decode_type) || requested <= 0 || encoded <= 0) {
    return;
  }
  const auto* descriptor = find_ssd_recipe_descriptor(recipe_id);
  if (descriptor && descriptor->class_count_policy == SsdClassCountPolicy::Exact) {
    if (requested == encoded) {
      return;
    }
    throw std::invalid_argument(
        std::string(context ? context : "BoxDecode") + ": SSD profile '" +
        ssd_recipe_id_token(recipe_id) + "' requires exactly " + std::to_string(encoded) +
        " classes encoded by the confidence heads, but num_classes=" + std::to_string(requested) +
        " was requested.");
  }
  if (requested <= encoded) {
    return;
  }
  throw std::invalid_argument(
      std::string(context ? context : "BoxDecode") +
      ": SSD num_classes=" + std::to_string(requested) + " exceeds the " + std::to_string(encoded) +
      " classes encoded in the confidence heads. Use a value <= " + std::to_string(encoded) +
      ", or leave it unset to infer from the head geometry.");
}

int ssd_encoded_num_classes(BoxDecodeType decode_type, const SsdClassSelection& selection,
                            int fallback_num_classes) {
  if (box_decode_type_is_ssd_family(decode_type) && selection.encoded_count > 0) {
    return selection.encoded_count;
  }
  return fallback_num_classes;
}

int resolve_boxdecode_num_classes_override(BoxDecodeType decode_type, int inferred_num_classes,
                                           int requested_num_classes, const char* context) {
  if (requested_num_classes <= 0) {
    return inferred_num_classes;
  }
  if ((decode_type_is_yolov26_family(decode_type) || decode_type == BoxDecodeType::YoloV5) &&
      inferred_num_classes > 0 && requested_num_classes != inferred_num_classes) {
    throw std::invalid_argument(
        std::string(context ? context : "BoxDecode") +
        " num_classes mismatch: configured=" + std::to_string(requested_num_classes) +
        " inferred_from_mpk=" + std::to_string(inferred_num_classes) +
        " decode_type=" + box_decode_type_token(decode_type) +
        ". Set num_classes=" + std::to_string(inferred_num_classes) +
        " to match the model class-head depth, or leave it 0 to use MPK inference.");
  }
  return requested_num_classes;
}

void resolve_grouped_yolo_dfl_score_domain(BoxDecodeStaticContract* contract) {
  if (!contract) {
    throw std::invalid_argument("YOLO BoxDecode score-domain resolution requires a contract");
  }

  switch (contract->decode_type_option) {
  case BoxDecodeTypeOption::GroupedByRoleProbability:
    contract->score_activation = BoxDecodeScoreActivation::Identity;
    return;
  case BoxDecodeTypeOption::GroupedByRoleLogit:
    contract->score_activation = BoxDecodeScoreActivation::Sigmoid;
    return;
  case BoxDecodeTypeOption::Auto:
  case BoxDecodeTypeOption::GroupedByRole:
    break;
  default:
    throw std::runtime_error(
        "YOLO BoxDecode grouped DFL outputs require a grouped-by-role decode_type_option");
  }

  switch (contract->score_activation) {
  case BoxDecodeScoreActivation::Identity:
    contract->decode_type_option = BoxDecodeTypeOption::GroupedByRoleProbability;
    return;
  case BoxDecodeScoreActivation::Sigmoid:
    contract->decode_type_option = BoxDecodeTypeOption::GroupedByRoleLogit;
    return;
  case BoxDecodeScoreActivation::Softmax:
    throw std::runtime_error(
        "YOLO BoxDecode grouped DFL does not support a softmax score domain; softmax across the "
        "class dimension is an SSD confidence-head activation, not a YOLO grouped-DFL score "
        "domain");
  case BoxDecodeScoreActivation::Unknown:
    throw std::runtime_error(
        "YOLO BoxDecode grouped DFL score domain is ambiguous; declare class_prob/class_logit "
        "tensor semantics or set an explicit probability/logit decode_type_option");
  }
}

void apply_ssd_model_managed_contract_defaults(BoxDecodeStaticContract* contract) {
  // Exported entry point for the model-managed subset extractor. Resolves the exact ordered
  // prepared-head signature and applies the descriptor's concrete type, activation, layout, and
  // class policy. No-op for non-SSD decode types.
  apply_ssd_static_contract_overrides(contract);
}

void apply_yolov5_model_managed_contract_defaults(BoxDecodeStaticContract* contract) {
  apply_yolov5_static_contract_overrides(contract);
  if (contract && contract->decode_type == BoxDecodeType::YoloV5) {
    contract->num_classes = resolve_boxdecode_num_classes(*contract, contract->num_classes,
                                                          "BoxDecode model-managed contract");
  }
}

BoxDecodeStaticContract finalize_boxdecode_static_contract(
    const BoxDecodeStaticContract& contract, BoxDecodeType decode_type,
    const std::optional<ModelBoxdecodeSemantics>& model_semantics,
    const std::optional<ModelManagedRouteFlags>& model_route_flags,
    BoxDecodeTypeOption decode_type_option, double detection_threshold, double nms_iou_threshold,
    int topk, int num_classes, const std::vector<std::string>& required_preprocess_meta_fields) {
  BoxDecodeStaticContract finalized = contract;
  finalized.decode_type = decode_type;
  finalized.decode_type_option = decode_type_option != BoxDecodeTypeOption::Auto
                                     ? decode_type_option
                                     : contract.decode_type_option;
  if (model_route_flags.has_value()) {
    const bool direct_packed_superpoint =
        finalized.decode_type == BoxDecodeType::SuperPoint &&
        std::any_of(finalized.tensors.begin(), finalized.tensors.end(), [](const auto& tensor) {
          return tensor.source_storage_kind == BoxDecodeSourceStorageKind::PackedCBlock ||
                 tensor.source_storage_kind == BoxDecodeSourceStorageKind::PackedHwcC16;
        });
    if (!boxdecode_bypass_mla_unpack_enabled() && !direct_packed_superpoint) {
      finalized.tess_needed = model_route_flags->tess_needed;
      finalized.quant_needed = model_route_flags->quant_needed;
    }
    finalized.quant_contract_required = model_route_flags->quant_contract_required;
    finalized.model_owned_flags = true;
  } else if (model_semantics.has_value()) {
    finalized.tess_needed = model_semantics->tess_needed;
    finalized.quant_needed = model_semantics->quant_needed;
    finalized.quant_contract_required = model_semantics->quant_contract_required;
    finalized.model_owned_flags = true;
  }
  finalized.detection_threshold = detection_threshold;
  finalized.nms_iou_threshold = nms_iou_threshold;
  finalized.topk = topk;
  finalized.required_preprocess_meta_fields = required_preprocess_meta_fields;
  apply_yolov5_static_contract_overrides(&finalized);
  apply_yolov26_static_contract_overrides(&finalized);
  apply_raw_yolov6_yolox_static_contract_overrides(&finalized);
  apply_ssd_static_contract_overrides(&finalized);
  if (finalized.decode_type == BoxDecodeType::SuperPoint) {
    finalize_superpoint_contract(&finalized, "BoxDecode");
    finalize_superpoint_decode_controls(finalized.superpoint.profile, finalized.nms_iou_threshold,
                                        &finalized.detection_threshold, &finalized.topk,
                                        "BoxDecode");
    finalized.num_classes = 0;
  } else {
    finalized.num_classes = resolve_boxdecode_num_classes(finalized, num_classes, "BoxDecode");
    if (box_decode_type_is_ssd_family(finalized.decode_type)) {
      finalized.ssd_class_selection.selected_count = finalized.num_classes;
    }
  }
  return finalized;
}

CompiledBoxDecodeContract build_boxdecode_compiled_contract_from_subset(
    const plugin_contracts::BoxDecodeContractSubset& subset,
    const BoxDecodeCompiledContractOptions& options) {
  plugin_contracts::validate_boxdecode_contract_subset(subset);

  CompiledBoxDecodeContract compiled;
  if (subset.decode_type == BoxDecodeType::Ssd && subset.ssd_recipe_id == SsdRecipeId::Unknown) {
    throw std::invalid_argument(
        "boxdecode compiled SSD subset is unresolved; validate the complete ordered head "
        "signature and carry a concrete SSD profile before subset lowering");
  }
  if (is_box_decode_type_specified(options.decode_type) &&
      is_box_decode_type_specified(subset.decode_type) &&
      !box_decode_type_matches_requested_contract(subset.decode_type, options.decode_type)) {
    throw std::invalid_argument("boxdecode compiled contract decode_type conflict: subset='" +
                                box_decode_type_token_string(subset.decode_type) +
                                "', requested='" +
                                box_decode_type_token_string(options.decode_type) + "'");
  }
  compiled.payload.decode_type = subset.decode_type;
  compiled.payload.ssd_recipe_id = subset.ssd_recipe_id;
  compiled.payload.ssd_class_selection = subset.ssd_class_selection;
  if (!is_box_decode_type_specified(compiled.payload.decode_type) ||
      (options.decode_type != BoxDecodeType::Ssd &&
       is_box_decode_type_specified(options.decode_type))) {
    compiled.payload.decode_type = options.decode_type;
  }
  if (!is_box_decode_type_specified(compiled.payload.decode_type)) {
    throw std::invalid_argument("boxdecode compiled contract requires an explicit decode_type");
  }
  compiled.payload.decode_type_option = options.decode_type_option.has_value()
                                            ? options.decode_type_option
                                            : subset.decode_type_option;
  compiled.payload.score_activation = options.score_activation != BoxDecodeScoreActivation::Unknown
                                          ? options.score_activation
                                          : subset.score_activation;
  if (box_decode_type_is_ssd_family(compiled.payload.decode_type)) {
    const auto* descriptor = find_ssd_recipe_descriptor(compiled.payload.ssd_recipe_id);
    if (!descriptor) {
      throw std::invalid_argument(
          "boxdecode compiled SSD contract is unresolved; the exact ordered head signature must "
          "resolve BoxDecodeType::Ssd before subset lowering");
    }
    if (compiled.payload.score_activation != BoxDecodeScoreActivation::Unknown &&
        compiled.payload.score_activation != descriptor->activation) {
      throw std::invalid_argument(
          "boxdecode compiled SSD contract activation conflicts with resolved profile '" +
          std::string(ssd_recipe_id_token(descriptor->id)) + "'");
    }
    compiled.payload.score_activation = descriptor->activation;
    if (!compiled.payload.decode_type_option.has_value() ||
        *compiled.payload.decode_type_option == BoxDecodeTypeOption::Auto) {
      compiled.payload.decode_type_option = BoxDecodeTypeOption::GroupedByRole;
    } else if (*compiled.payload.decode_type_option != BoxDecodeTypeOption::GroupedByRole) {
      throw std::invalid_argument(
          std::string("SSD BoxDecode supports only the grouped-by-role head layout, but got '") +
          box_decode_type_option_token(*compiled.payload.decode_type_option) + "'.");
    }
  }
  compiled.payload.input_dtype = resolve_boxdecode_input_dtype(subset);
  compiled.payload.tess_needed = subset.tess_needed;
  compiled.payload.quant_needed = subset.quant_needed;
  compiled.payload.model_owned_flags = options.model_owned_flags;
  compiled.payload.quant_contract_required =
      options.quant_contract_required.value_or(subset.quant_needed);
  compiled.payload.detection_threshold = options.detection_threshold;
  compiled.payload.nms_iou_threshold = options.nms_iou_threshold;
  compiled.payload.topk = options.topk;
  validate_ssd_num_classes(compiled.payload.decode_type, compiled.payload.ssd_recipe_id,
                           options.num_classes,
                           box_decode_type_is_ssd_family(compiled.payload.decode_type)
                               ? subset.ssd_class_selection.encoded_count
                               : subset.num_classes,
                           "BoxDecode");
  compiled.payload.num_classes = options.num_classes > 0 ? options.num_classes : subset.num_classes;
  if (box_decode_type_is_ssd_family(compiled.payload.decode_type)) {
    compiled.payload.ssd_class_selection.selected_count = compiled.payload.num_classes;
  }
  compiled.payload.slice_shapes = subset.slice_shapes;
  compiled.payload.tensor_storage_kind = subset.tensor_storage_kind;
  compiled.payload.superpoint = subset.superpoint;
  if (compiled.payload.decode_type == BoxDecodeType::SuperPoint) {
    auto& resolved = compiled.payload.superpoint;
    if (options.superpoint.has_value()) {
      const auto& requested = *options.superpoint;
      const bool profile_changed = apply_superpoint_profile_override(&resolved, requested.profile);
      SuperPointOptions spatial_overrides;
      spatial_overrides.nms_radius = requested.nms_radius;
      spatial_overrides.border_margin = requested.border_margin;
      apply_superpoint_spatial_overrides(&resolved, spatial_overrides);
      resolved.descriptor_output_dtype = requested.descriptor_output_dtype;
      resolved.output_format = requested.output_format;
      if (requested.cell_stride > 0) {
        resolved.cell_stride = requested.cell_stride;
      }
      if (requested.descriptor_stride > 0) {
        resolved.descriptor_stride = requested.descriptor_stride;
      }
      if (requested.descriptor_dim > 0) {
        resolved.descriptor_dim = requested.descriptor_dim;
      }
      if (!requested.profile_fingerprint.empty()) {
        resolved.profile_fingerprint = requested.profile_fingerprint;
        resolved.fingerprint_profile = requested.fingerprint_profile;
      }
      if (!requested.detector_tensor_id.empty()) {
        resolved.detector_tensor_id = requested.detector_tensor_id;
      }
      if (!requested.descriptor_tensor_id.empty()) {
        resolved.descriptor_tensor_id = requested.descriptor_tensor_id;
      }
      if (!requested.detector_representation.empty()) {
        resolved.detector_representation = requested.detector_representation;
      }
      if (!requested.descriptor_representation.empty()) {
        resolved.descriptor_representation = requested.descriptor_representation;
      }
      if (requested.schema_version != 0) {
        resolved.schema_version = requested.schema_version;
      }
      compiled.payload.detection_threshold = rebase_superpoint_detection_threshold(
          profile_changed, resolved.profile, options.detection_threshold,
          compiled.payload.detection_threshold);
    }
    resolve_default_superpoint_profile(&resolved);
    canonicalize_schema0_superpoint_representations(&resolved);
    if (const auto metadata_error =
            validate_superpoint_static_metadata(resolved, /*require_resolved_profile=*/true)) {
      throw std::invalid_argument("boxdecode compiled SuperPoint contract: " + *metadata_error);
    }
    if (resolved.profile == SuperPointProfile::PaperBicubicV1) {
      throw std::invalid_argument(
          "boxdecode compiled SuperPoint contract: paper-bicubic-v1 is reserved but not "
          "production-defined");
    }
    if (resolved.nms_radius < -1 || resolved.border_margin < -1) {
      throw std::invalid_argument(
          "boxdecode compiled SuperPoint contract radius and border must be -1 (profile "
          "default) or non-negative; zero is valid");
    }
    if (resolved.nms_radius == -1) {
      resolved.nms_radius = 4;
    }
    if (resolved.border_margin == -1) {
      resolved.border_margin = resolved.profile == SuperPointProfile::A65V1 ? 0 : 4;
    }
    finalize_superpoint_decode_controls(resolved.profile, compiled.payload.nms_iou_threshold,
                                        &compiled.payload.detection_threshold,
                                        &compiled.payload.topk, "boxdecode compiled contract");
    if (subset.logical_inputs.size() != 2U || subset.tensor_roles.size() != 2U ||
        std::count(subset.tensor_roles.begin(), subset.tensor_roles.end(),
                   BoxDecodeTensorRole::DetectorLogits) != 1 ||
        std::count(subset.tensor_roles.begin(), subset.tensor_roles.end(),
                   BoxDecodeTensorRole::DescriptorGrid) != 1) {
      throw std::invalid_argument(
          "boxdecode compiled SuperPoint contract requires exactly one detector-logits and one "
          "descriptor-grid input role");
    }
    if (resolved.cell_stride <= 0 || resolved.descriptor_stride <= 0 ||
        resolved.descriptor_dim <= 0) {
      throw std::invalid_argument(
          "boxdecode compiled SuperPoint contract has unresolved radius, border, stride, or "
          "descriptor dimension");
    }
    if (resolved.output_format == SuperPointOutputFormat::LegacyA65InterleavedV0 &&
        (resolved.descriptor_dim != 256 || resolved.descriptor_output_dtype != TensorDType::Int8)) {
      throw std::invalid_argument(
          "boxdecode compiled SuperPoint contract legacy A65 V0 output requires "
          "256-dimensional INT8 descriptors");
    }
    compiled.payload.num_classes = 0;
  }
  compiled.payload.tensor_roles = subset.tensor_roles;
  compiled.runtime_contract.plugin_kind = "boxdecode";
  compiled.runtime_contract.logical_inputs = subset.logical_inputs;
  compiled.runtime_contract.input_bindings = subset.input_bindings;
  compiled.runtime_contract.required_preprocess_meta_fields =
      options.required_preprocess_meta_fields;
  return compiled;
}

CompiledBoxDecodeContract
build_boxdecode_compiled_contract(const BoxDecodeStaticContract& contract) {
  BoxDecodeStaticContract normalized = contract;
  apply_yolov5_static_contract_overrides(&normalized);
  apply_yolov26_static_contract_overrides(&normalized);
  apply_raw_yolov6_yolox_static_contract_overrides(&normalized);
  apply_ssd_static_contract_overrides(&normalized);
  if (normalized.decode_type == BoxDecodeType::SuperPoint) {
    finalize_superpoint_contract(&normalized, "BoxDecode");
    normalized.num_classes = 0;
  } else if (normalized.num_classes <= 0) {
    normalized.num_classes =
        resolve_boxdecode_num_classes(normalized, /*user_num_classes=*/0, "BoxDecode");
  }
  const auto subset =
      plugin_contracts::extract_boxdecode_contract_subset_from_static_contract(normalized);
  BoxDecodeCompiledContractOptions options;
  options.decode_type = normalized.decode_type;
  if (normalized.decode_type_option != BoxDecodeTypeOption::Auto) {
    options.decode_type_option = normalized.decode_type_option;
  }
  options.score_activation = normalized.score_activation;
  options.detection_threshold = normalized.detection_threshold;
  options.nms_iou_threshold = normalized.nms_iou_threshold;
  options.topk = normalized.topk;
  options.num_classes = normalized.num_classes;
  options.superpoint = normalized.superpoint;
  options.model_owned_flags = normalized.model_owned_flags;
  options.quant_contract_required = normalized.quant_contract_required;
  options.required_preprocess_meta_fields = normalized.required_preprocess_meta_fields;
  return build_boxdecode_compiled_contract_from_subset(subset, options);
}

bool build_boxdecode_node_contract(const std::string& node_kind, const std::string& plugin_kind,
                                   const std::string& element_name,
                                   const std::string& logical_stage_id,
                                   const NodeContractDefinition& definition,
                                   const CompiledBoxDecodeContract& compiled,
                                   CompiledNodeContract* out, std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = node_kind + " contract compile: output is null";
    }
    return false;
  }
  populate_boxdecode_node_contract_common(node_kind, plugin_kind, element_name, logical_stage_id,
                                          definition, compiled, out);
  if (error_message) {
    error_message->clear();
  }
  return true;
}

} // namespace simaai::neat::pipeline_internal::sima::stagesemantics

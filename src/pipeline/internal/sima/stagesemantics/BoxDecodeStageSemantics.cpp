#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"

#include "pipeline/internal/sima/BoxDecodeTypeUtils.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <optional>
#include <stdexcept>

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

bool same_hw(const TensorHwc& lhs, const TensorHwc& rhs) {
  return lhs.h == rhs.h && lhs.w == rhs.w;
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

int infer_packed_yolo_class_depth(const BoxDecodeStaticContract& contract) {
  std::optional<int> classes;
  for (const auto& tensor : contract.tensors) {
    const int c = logical_channel_depth(tensor);
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

bool decode_type_is_packed_yolo(BoxDecodeType type) {
  return type == BoxDecodeType::Yolo || type == BoxDecodeType::YoloV5 ||
         type == BoxDecodeType::YoloV5Seg || type == BoxDecodeType::YoloV7 ||
         type == BoxDecodeType::YoloV7Seg;
}

int infer_raw_yolo_class_depth(const BoxDecodeStaticContract& contract);

int infer_boxdecode_num_classes_from_contract(const BoxDecodeStaticContract& contract) {
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
    if (inferred > 0 && user_num_classes != inferred) {
      std::fprintf(stderr,
                   "[WARN] %s num_classes mismatch: user=%d inferred_from_mpk=%d decode_type=%s. "
                   "Using user value.\n",
                   context ? context : "BoxDecode", user_num_classes, inferred,
                   box_decode_type_token(contract.decode_type));
    }
    return user_num_classes;
  }
  return inferred;
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
    if (dtype != logical.dtype) {
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
  case BoxDecodeScoreActivation::Unknown:
    throw std::runtime_error(
        "YOLO BoxDecode grouped DFL score domain is ambiguous; declare class_prob/class_logit "
        "tensor semantics or set an explicit probability/logit decode_type_option");
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
    if (!boxdecode_bypass_mla_unpack_enabled()) {
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
  apply_yolov26_static_contract_overrides(&finalized);
  apply_raw_yolov6_yolox_static_contract_overrides(&finalized);
  finalized.num_classes = resolve_boxdecode_num_classes(finalized, num_classes, "BoxDecode");
  return finalized;
}

CompiledBoxDecodeContract build_boxdecode_compiled_contract_from_subset(
    const plugin_contracts::BoxDecodeContractSubset& subset,
    const BoxDecodeCompiledContractOptions& options) {
  plugin_contracts::validate_boxdecode_contract_subset(subset);

  CompiledBoxDecodeContract compiled;
  compiled.payload.decode_type =
      is_box_decode_type_specified(options.decode_type) ? options.decode_type : subset.decode_type;
  if (!is_box_decode_type_specified(compiled.payload.decode_type)) {
    throw std::invalid_argument("boxdecode compiled contract requires an explicit decode_type");
  }
  compiled.payload.decode_type_option = options.decode_type_option.has_value()
                                            ? options.decode_type_option
                                            : subset.decode_type_option;
  compiled.payload.score_activation = options.score_activation != BoxDecodeScoreActivation::Unknown
                                          ? options.score_activation
                                          : subset.score_activation;
  compiled.payload.input_dtype = resolve_boxdecode_input_dtype(subset);
  compiled.payload.tess_needed = subset.tess_needed;
  compiled.payload.quant_needed = subset.quant_needed;
  compiled.payload.model_owned_flags = options.model_owned_flags;
  compiled.payload.quant_contract_required =
      options.quant_contract_required.value_or(subset.quant_needed);
  compiled.payload.detection_threshold = options.detection_threshold;
  compiled.payload.nms_iou_threshold = options.nms_iou_threshold;
  compiled.payload.topk = options.topk;
  compiled.payload.num_classes = options.num_classes > 0 ? options.num_classes : subset.num_classes;
  compiled.payload.slice_shapes = subset.slice_shapes;
  compiled.payload.tensor_storage_kind = subset.tensor_storage_kind;
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
  apply_yolov26_static_contract_overrides(&normalized);
  apply_raw_yolov6_yolox_static_contract_overrides(&normalized);
  if (normalized.num_classes <= 0) {
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

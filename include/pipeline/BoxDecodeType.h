/**
 * @file
 * @ingroup pipeline
 * @brief Typed decode-family selection for BoxDecode stages.
 *
 * The `BoxDecodeType` enum picks the head-decoding contract used by the
 * `SimaBoxDecode` postprocess stage: which YOLO variant, DETR, EffDet, RCNN
 * stage-1 or CenterNet head layout the planner should expect. The companion
 * `BoxDecodeTypeOption` enum selects the tensor-packing variant within a
 * family. Helpers in this header stringify the enums and summarise each
 * decode family's tensor contract for diagnostics and docs.
 *
 * @see nodes/sima/SimaBoxDecode.h for the consuming postprocess node.
 * @see DetectionTypes.h for the decoded detection payload.
 */
#pragma once

#include <cstdint>

namespace simaai::neat {

/**
 * @brief Decode families accepted by the BoxDecode backend.
 *
 * `Unspecified` is an internal unset sentinel and must fail fast before runtime decode.
 * Most YOLO-family variants share the same class-inference contract in `genericboxdecode_v2`:
 * - decoupled heads: repeated class-depth tensors, class depth > 4
 * - packed heads: depth = 3 * (num_classes + 5), consistent across heads
 * `YoloV26` uses decoupled 4-channel raw l/t/r/b bbox heads paired with class heads.
 * `YoloV26Seg` uses the same raw l/t/r/b bbox heads, class-score heads,
 * 32-channel mask-coefficient heads, and a trailing mask prototype.
 * `YoloV26Pose` uses the same raw l/t/r/b bbox heads, 1-channel pose scores,
 * and 51-channel keypoint heads.
 *
 * @ingroup pipeline
 */
enum class BoxDecodeType : std::int32_t {
  // Internal sentinel; runtime decode requires a concrete value.
  Unspecified = 0, ///< Sentinel: no decode family selected (fails fast at runtime).
  // YOLO-family (generic token).
  Yolo = 1,         ///< Generic YOLO family token.
  YoloV5 = 2,       ///< YOLOv5 detection.
  YoloV5Seg = 3,    ///< YOLOv5 segmentation.
  YoloV7 = 4,       ///< YOLOv7 detection.
  YoloV7Seg = 5,    ///< YOLOv7 segmentation.
  YoloV8 = 6,       ///< YOLOv8 detection.
  YoloV8Seg = 7,    ///< YOLOv8 segmentation.
  YoloV8Pose = 8,   ///< YOLOv8 pose-estimation heads.
  YoloV9 = 9,       ///< YOLOv9 detection.
  YoloV9Seg = 10,   ///< YOLOv9 segmentation.
  YoloV10 = 11,     ///< YOLOv10 detection.
  YoloV10Seg = 12,  ///< YOLOv10 segmentation.
  Detr = 13,        ///< DETR-style transformer detection.
  EffDet = 14,      ///< EfficientDet detection.
  RcnnStage1 = 15,  ///< Region-proposal stage of two-stage R-CNN models.
  Centernet = 16,   ///< CenterNet keypoint-style detection.
  YoloV26 = 17,     ///< YOLO26 detection (raw l/t/r/b distance heads).
  YoloV26Pose = 18, ///< YOLO26 pose-estimation heads.
  YoloV26Seg = 19,  ///< YOLO26 segmentation heads.
  YoloV6 = 20,      ///< YOLOv6 raw l/t/r/b distance heads.
  YoloX = 21,       ///< YOLOX raw xywh heads with separate objectness and class logits.
};

/**
 * @brief Tensor packing/layout option within a decode family.
 *
 * Some families admit multiple equivalent head layouts (packed vs interleaved,
 * grouped by role, probability vs logit class scores). `Auto` lets the planner
 * pick from observed tensor geometry; the explicit values force a particular
 * decoding contract for ambiguous models.
 *
 * @ingroup pipeline
 */
enum class BoxDecodeTypeOption : std::int32_t {
  Auto = 0,                         ///< Backend infers the layout from tensor shapes.
  PackedPerHead = 1,                ///< Each head holds a single packed tensor (box+obj+cls).
  InterleavedByHead = 2,            ///< Heads interleaved within tensors.
  GroupedByRole = 3,                ///< Tensors grouped by role (box, score, class).
  Split3Interleaved = 4,            ///< Three split tensors, head-interleaved.
  Split3Grouped = 5,                ///< Three split tensors, grouped by role.
  InterleavedByHeadProbability = 6, ///< Interleaved-by-head, class scores as probabilities.
  InterleavedByHeadLogit = 7,       ///< Interleaved-by-head, class scores as logits.
  GroupedByRoleProbability = 8,     ///< Grouped-by-role, class scores as probabilities.
  GroupedByRoleLogit = 9,           ///< Grouped-by-role, class scores as logits.
};

/// @brief Stable lower-case token for a decode family (used in caps, manifests, logs).
constexpr const char* box_decode_type_token(BoxDecodeType type) {
  switch (type) {
  case BoxDecodeType::Yolo:
    return "yolo";
  case BoxDecodeType::YoloV5:
    return "yolov5";
  case BoxDecodeType::YoloV5Seg:
    return "yolov5-seg";
  case BoxDecodeType::YoloV7:
    return "yolov7";
  case BoxDecodeType::YoloV7Seg:
    return "yolov7-seg";
  case BoxDecodeType::YoloV8:
    return "yolov8";
  case BoxDecodeType::YoloV8Seg:
    return "yolov8-seg";
  case BoxDecodeType::YoloV8Pose:
    return "yolov8-pose";
  case BoxDecodeType::YoloV9:
    return "yolov9";
  case BoxDecodeType::YoloV9Seg:
    return "yolov9-seg";
  case BoxDecodeType::YoloV10:
    return "yolov10";
  case BoxDecodeType::YoloV10Seg:
    return "yolov10-seg";
  case BoxDecodeType::YoloV26:
    return "yolo26";
  case BoxDecodeType::YoloV26Pose:
    return "yolo26-pose";
  case BoxDecodeType::YoloV26Seg:
    return "yolo26-seg";
  case BoxDecodeType::YoloV6:
    return "yolov6";
  case BoxDecodeType::YoloX:
    return "yolox";
  case BoxDecodeType::Detr:
    return "detr";
  case BoxDecodeType::EffDet:
    return "effdet";
  case BoxDecodeType::RcnnStage1:
    return "rcnn-stage1";
  case BoxDecodeType::Centernet:
    return "centernet";
  case BoxDecodeType::Unspecified:
  default:
    return "unspecified";
  }
}

/// @brief Stable lower-case token for a decode-layout option.
constexpr const char* box_decode_type_option_token(BoxDecodeTypeOption option) {
  switch (option) {
  case BoxDecodeTypeOption::Auto:
    return "auto";
  case BoxDecodeTypeOption::PackedPerHead:
    return "packed-per-head";
  case BoxDecodeTypeOption::InterleavedByHead:
    return "interleaved-by-head";
  case BoxDecodeTypeOption::GroupedByRole:
    return "grouped-by-role";
  case BoxDecodeTypeOption::Split3Interleaved:
    return "split3-interleaved";
  case BoxDecodeTypeOption::Split3Grouped:
    return "split3-grouped";
  case BoxDecodeTypeOption::InterleavedByHeadProbability:
    return "interleaved-by-head-probability";
  case BoxDecodeTypeOption::InterleavedByHeadLogit:
    return "interleaved-by-head-logit";
  case BoxDecodeTypeOption::GroupedByRoleProbability:
    return "grouped-by-role-probability";
  case BoxDecodeTypeOption::GroupedByRoleLogit:
    return "grouped-by-role-logit";
  default:
    return "auto";
  }
}

/// @brief True iff @p type is one of the YOLO-family detection or segmentation variants.
constexpr bool box_decode_type_is_yolo_family(BoxDecodeType type) {
  switch (type) {
  case BoxDecodeType::Yolo:
  case BoxDecodeType::YoloV5:
  case BoxDecodeType::YoloV5Seg:
  case BoxDecodeType::YoloV7:
  case BoxDecodeType::YoloV7Seg:
  case BoxDecodeType::YoloV8:
  case BoxDecodeType::YoloV8Seg:
  case BoxDecodeType::YoloV8Pose:
  case BoxDecodeType::YoloV9:
  case BoxDecodeType::YoloV9Seg:
  case BoxDecodeType::YoloV10:
  case BoxDecodeType::YoloV10Seg:
  case BoxDecodeType::YoloV26:
  case BoxDecodeType::YoloV26Pose:
  case BoxDecodeType::YoloV26Seg:
  case BoxDecodeType::YoloV6:
  case BoxDecodeType::YoloX:
    return true;
  case BoxDecodeType::Detr:
  case BoxDecodeType::EffDet:
  case BoxDecodeType::RcnnStage1:
  case BoxDecodeType::Centernet:
  case BoxDecodeType::Unspecified:
  default:
    return false;
  }
}

/// @brief True iff @p type is a segmentation variant (carries a mask head).
constexpr bool box_decode_type_is_segmentation(BoxDecodeType type) {
  switch (type) {
  case BoxDecodeType::YoloV5Seg:
  case BoxDecodeType::YoloV7Seg:
  case BoxDecodeType::YoloV8Seg:
  case BoxDecodeType::YoloV9Seg:
  case BoxDecodeType::YoloV10Seg:
  case BoxDecodeType::YoloV26Seg:
    return true;
  case BoxDecodeType::Yolo:
  case BoxDecodeType::YoloV5:
  case BoxDecodeType::YoloV7:
  case BoxDecodeType::YoloV8:
  case BoxDecodeType::YoloV8Pose:
  case BoxDecodeType::YoloV9:
  case BoxDecodeType::YoloV10:
  case BoxDecodeType::YoloV26:
  case BoxDecodeType::YoloV26Pose:
  case BoxDecodeType::YoloV6:
  case BoxDecodeType::YoloX:
  case BoxDecodeType::Detr:
  case BoxDecodeType::EffDet:
  case BoxDecodeType::RcnnStage1:
  case BoxDecodeType::Centernet:
  case BoxDecodeType::Unspecified:
  default:
    return false;
  }
}

/// @brief Human-readable summary of @p type's tensor contract, used in API/docs/error surfaces.
constexpr const char* box_decode_type_contract_summary(BoxDecodeType type) {
  switch (type) {
  case BoxDecodeType::Yolo:
  case BoxDecodeType::YoloV5:
  case BoxDecodeType::YoloV7:
  case BoxDecodeType::YoloV8:
  case BoxDecodeType::YoloV8Pose:
  case BoxDecodeType::YoloV9:
  case BoxDecodeType::YoloV10:
    return "YOLO tensor contract: decoupled class heads (>4 channels, repeated across heads) or "
           "packed heads (depth=3*(num_classes+5), consistent across heads).";
  case BoxDecodeType::YoloV26:
    return "YOLO26 tensor contract: grouped raw l/t/r/b bbox heads (4 channels) paired "
           "with repeated class-score heads (>4 channels).";
  case BoxDecodeType::YoloV26Pose:
    return "YOLO26-pose contract: grouped raw l/t/r/b bbox heads (4 channels), "
           "1-channel pose score heads, and 51-channel keypoint heads.";
  case BoxDecodeType::YoloV26Seg:
    return "YOLO26-seg contract: grouped raw l/t/r/b bbox heads (4 channels), "
           "class-score heads, 32-channel mask-coefficient heads, and a trailing "
           "32-channel mask prototype.";
  case BoxDecodeType::YoloV6:
    return "YOLOv6 raw-head contract: interleaved [bbox_ltrb_i, class_logit_i] "
           "heads with 4-channel l/t/r/b distances and class-logit heads.";
  case BoxDecodeType::YoloX:
    return "YOLOX raw-head contract: interleaved [bbox_i, obj_logit_i, class_logit_i] "
           "heads with raw xywh boxes, objectness logits, and class logits.";
  case BoxDecodeType::YoloV5Seg:
  case BoxDecodeType::YoloV7Seg:
  case BoxDecodeType::YoloV8Seg:
  case BoxDecodeType::YoloV9Seg:
  case BoxDecodeType::YoloV10Seg:
    return "YOLO-seg contract: same class-depth rules as YOLO plus segmentation decode path.";
  case BoxDecodeType::Detr:
    return "DETR contract: class channels inferred from max depth across heads; max depth must be "
           ">4.";
  case BoxDecodeType::EffDet:
  case BoxDecodeType::RcnnStage1:
  case BoxDecodeType::Centernet:
    return "Non-YOLO fallback contract: class channels inferred from max depth across heads; "
           "max depth must be >4.";
  case BoxDecodeType::Unspecified:
  default:
    return "Unset decode type; runtime decode must fail fast until a concrete decode family is "
           "selected.";
  }
}

} // namespace simaai::neat

/**
 * @file
 * @ingroup pipeline
 * @brief Named, source-compatible options shared by BoxDecode nodes and stage entry points.
 */
#pragma once

#include "pipeline/BoxDecodeType.h"
#include "pipeline/SuperPointTypes.h"
#include "pipeline/YoloXSegPoseTypes.h"

namespace simaai::neat {

struct BoxDecodeOptions {
  explicit BoxDecodeOptions(BoxDecodeType type) : decode_type(type) {}
  BoxDecodeOptions() = delete;

  BoxDecodeType decode_type;
  double detection_threshold = 0.0;
  double nms_iou_threshold = 0.0;
  /// Maximum output count. Bounding-box decoding requires a positive resolved cap:
  /// zero is unspecified for raw heads, or inherits the Model cap on model-bound nodes.
  /// SuperPoint resolves zero to its profile default.
  int top_k = 0;
  SuperPointOptions superpoint;
  YoloXSegPoseOptions yolox_seg_pose;
};

} // namespace simaai::neat

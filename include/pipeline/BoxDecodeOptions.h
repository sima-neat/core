/**
 * @file
 * @ingroup pipeline
 * @brief Named, source-compatible options shared by BoxDecode nodes and stage entry points.
 */
#pragma once

#include "pipeline/BoxDecodeType.h"
#include "pipeline/SuperPointTypes.h"

#include <vector>

namespace simaai::neat {

struct BoxDecodeOptions {
  explicit BoxDecodeOptions(BoxDecodeType type) : decode_type(type) {}
  BoxDecodeOptions() = delete;

  BoxDecodeType decode_type;
  double detection_threshold = 0.0;
  double nms_iou_threshold = 0.0;
  int top_k = 0;
  SuperPointOptions superpoint;
  /// Class indices that carry keypoints; empty treats every class as pose-bearing.
  std::vector<int> pose_classes;
};

} // namespace simaai::neat

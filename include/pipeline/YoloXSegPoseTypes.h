#pragma once

#include <vector>

namespace simaai::neat {

/// YOLOX segmentation/pose settings; other BoxDecode types reject a nonempty class list.
struct YoloXSegPoseOptions {
  /// Classes with keypoints. Empty means all classes on Model, or inherit on a node.
  std::vector<int> pose_classes;
};

} // namespace simaai::neat

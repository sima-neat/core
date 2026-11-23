#pragma once

#include "pipeline/BoxDecodeType.h"
#include "pipeline/DetectionTypes.h"
#include "pipeline/FeatureTypes.h"
#include "pipeline/SuperPointTypes.h"

namespace simaai::neat::contracts {

// Core's application wire formats, not numerical kernel output layouts.
enum class BoxDecodeWireFormat {
  Boxes,
  OrientedBoxesV1,
  Pose,
  Masks,
  MasksPose,
  RcnnProposals,
  FeaturePointsV1,
  LegacyFeaturePointsV0,
  // SegmentationMasksHeaderV1 prefix (header, records, padding). The fifth
  // input spec describes masks the producer writes in place after the prefix.
  MasksV1
};

constexpr const char* boxdecode_output_format(BoxDecodeWireFormat wire) {
  switch (wire) {
  case BoxDecodeWireFormat::Boxes:
  case BoxDecodeWireFormat::RcnnProposals:
    return kDetectionFormatBbox;
  case BoxDecodeWireFormat::OrientedBoxesV1:
    return kDetectionFormatBboxObbV1;
  case BoxDecodeWireFormat::Pose:
    return kDetectionFormatBboxPose;
  case BoxDecodeWireFormat::Masks:
  case BoxDecodeWireFormat::MasksV1:
    return kDetectionFormatBboxSegmentation;
  case BoxDecodeWireFormat::MasksPose:
    return kDetectionFormatBboxSegmentationPose;
  case BoxDecodeWireFormat::FeaturePointsV1:
    return kFeatureFormatPointsV1;
  case BoxDecodeWireFormat::LegacyFeaturePointsV0:
    return kFeatureFormatLegacyA65V0;
  }
  return nullptr;
}

constexpr BoxDecodeWireFormat boxdecode_wire_hint(BoxDecodeType type,
                                                  SuperPointOutputFormat features) {
  if (type == BoxDecodeType::SuperPoint)
    return features == SuperPointOutputFormat::LegacyA65InterleavedV0
               ? BoxDecodeWireFormat::LegacyFeaturePointsV0
               : BoxDecodeWireFormat::FeaturePointsV1;
  if (box_decode_type_is_oriented(type))
    return BoxDecodeWireFormat::OrientedBoxesV1;
  if (box_decode_type_is_segmentation(type))
    return box_decode_type_is_pose(type) ? BoxDecodeWireFormat::MasksPose
                                         : BoxDecodeWireFormat::Masks;
  return box_decode_type_is_pose(type) ? BoxDecodeWireFormat::Pose : BoxDecodeWireFormat::Boxes;
}

} // namespace simaai::neat::contracts

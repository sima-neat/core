#include "pipeline/DetectionTypes.h"
#include "pipeline/TensorCore.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace {

struct RawBox {
  int32_t x = 0;
  int32_t y = 0;
  int32_t w = 0;
  int32_t h = 0;
  float score = 0.0f;
  int32_t cls = 0;
};

struct RawPosePoint {
  uint32_t x = 0;
  uint32_t y = 0;
  float visible = 0.0f;
};

struct RawPoseOut {
  RawPosePoint points[static_cast<std::size_t>(simaai::neat::kDecodedPoseKeypoints)];
};

static_assert(sizeof(RawBox) == 24);
static_assert(sizeof(RawPoseOut) == 204);

simaai::neat::Tensor make_wire_tensor(const std::vector<uint8_t>& payload,
                                      const std::string& format) {
  using namespace simaai::neat;
  auto storage = make_cpu_owned_storage(payload.size());
  auto map = storage->map(MapMode::Write);
  if (map.data && map.size_bytes >= payload.size()) {
    std::memcpy(map.data, payload.data(), payload.size());
  }

  Tensor tensor;
  tensor.storage = storage;
  tensor.dtype = TensorDType::UInt8;
  tensor.layout = TensorLayout::Unknown;
  tensor.shape = {static_cast<int64_t>(payload.size())};
  tensor.device = {DeviceType::CPU, 0};
  tensor.read_only = true;
  tag_detection_format(tensor, format);
  return tensor;
}

std::vector<uint8_t> make_pose_payload(uint32_t count, std::size_t capacity) {
  std::vector<uint8_t> out(sizeof(uint32_t) + capacity * sizeof(RawBox) +
                           capacity * sizeof(RawPoseOut));
  std::memcpy(out.data(), &count, sizeof(count));

  std::vector<RawBox> boxes(capacity);
  boxes[0] = RawBox{.x = 10, .y = 20, .w = 30, .h = 40, .score = 0.75f, .cls = 3};
  if (capacity > 1) {
    boxes[1] = RawBox{.x = 1, .y = 2, .w = 3, .h = 4, .score = 0.5f, .cls = 4};
  }
  std::memcpy(out.data() + sizeof(uint32_t), boxes.data(), capacity * sizeof(RawBox));

  const std::size_t pose_base = sizeof(uint32_t) + capacity * sizeof(RawBox);
  for (std::size_t box = 0; box < capacity; ++box) {
    RawPoseOut pose{};
    for (std::size_t point = 0;
         point < static_cast<std::size_t>(simaai::neat::kDecodedPoseKeypoints); ++point) {
      pose.points[point].x = static_cast<uint32_t>(1000 + box * 100 + point);
      pose.points[point].y = static_cast<uint32_t>(2000 + box * 100 + point);
      pose.points[point].visible = 0.01f * static_cast<float>(point + 1);
    }
    std::memcpy(out.data() + pose_base + box * sizeof(RawPoseOut), &pose, sizeof(pose));
  }
  return out;
}

std::vector<uint8_t> make_segmentation_payload(uint32_t count, std::size_t capacity) {
  const std::size_t mask_bytes = static_cast<std::size_t>(simaai::neat::kDecodedMaskWidth) *
                                 static_cast<std::size_t>(simaai::neat::kDecodedMaskHeight);
  std::vector<uint8_t> out(sizeof(uint32_t) + capacity * sizeof(RawBox) + capacity * mask_bytes);
  std::memcpy(out.data(), &count, sizeof(count));

  std::vector<RawBox> boxes(capacity);
  boxes[0] = RawBox{.x = 5, .y = 6, .w = 7, .h = 8, .score = 0.25f, .cls = 9};
  if (capacity > 1) {
    boxes[1] = RawBox{.x = 15, .y = 16, .w = 17, .h = 18, .score = 0.5f, .cls = 10};
  }
  std::memcpy(out.data() + sizeof(uint32_t), boxes.data(), capacity * sizeof(RawBox));

  const std::size_t mask_base = sizeof(uint32_t) + capacity * sizeof(RawBox);
  for (std::size_t box = 0; box < capacity; ++box) {
    uint8_t* mask = out.data() + mask_base + box * mask_bytes;
    for (std::size_t i = 0; i < mask_bytes; ++i) {
      mask[i] = static_cast<uint8_t>((box + i) & 0xffU);
    }
  }
  return out;
}

// Combined seg+pose is box-leading with three regions, each strided by the same slot
// count: [count][boxes][masks][poses]. Every slot is filled with a value that identifies
// it, so a region read at the wrong base or stride produces a recognisably wrong number
// rather than plausible data.
std::vector<uint8_t> make_segmentation_pose_payload(uint32_t count, std::size_t capacity) {
  const std::size_t mask_bytes = static_cast<std::size_t>(simaai::neat::kDecodedMaskWidth) *
                                 static_cast<std::size_t>(simaai::neat::kDecodedMaskHeight);
  std::vector<uint8_t> out(sizeof(uint32_t) +
                           capacity * (sizeof(RawBox) + mask_bytes + sizeof(RawPoseOut)));
  std::memcpy(out.data(), &count, sizeof(count));

  std::vector<RawBox> boxes(capacity);
  for (std::size_t slot = 0; slot < capacity; ++slot) {
    boxes[slot] = RawBox{.x = static_cast<int32_t>(100 + slot),
                         .y = static_cast<int32_t>(200 + slot),
                         .w = 30,
                         .h = 40,
                         .score = 0.5f,
                         .cls = static_cast<int32_t>(7 + slot)};
  }
  std::memcpy(out.data() + sizeof(uint32_t), boxes.data(), capacity * sizeof(RawBox));

  // Slot i's mask is filled with (i + 1). Reading the pose region by mistake would land on
  // keypoint bytes instead, which are never 1 or 2.
  const std::size_t mask_base = sizeof(uint32_t) + capacity * sizeof(RawBox);
  for (std::size_t slot = 0; slot < capacity; ++slot) {
    std::memset(out.data() + mask_base + slot * mask_bytes, static_cast<int>(slot + 1), mask_bytes);
  }

  // Poses start after the whole mask region, not immediately after the boxes as in a
  // pose-only payload. That extra capacity * mask_bytes offset is the part unique to this
  // format, so the keypoint values below are what catches it being dropped.
  const std::size_t pose_base = mask_base + capacity * mask_bytes;
  for (std::size_t slot = 0; slot < capacity; ++slot) {
    RawPoseOut pose{};
    for (std::size_t point = 0;
         point < static_cast<std::size_t>(simaai::neat::kDecodedPoseKeypoints); ++point) {
      pose.points[point].x = static_cast<uint32_t>(1000 * (slot + 1) + point);
      pose.points[point].y = static_cast<uint32_t>(2000 * (slot + 1) + point);
      pose.points[point].visible = 0.01f * static_cast<float>(point + 1);
    }
    std::memcpy(out.data() + pose_base + slot * sizeof(RawPoseOut), &pose, sizeof(pose));
  }
  return out;
}

std::vector<float> tensor_float_values(const simaai::neat::Tensor& tensor) {
  std::vector<uint8_t> bytes = tensor.copy_payload_bytes();
  std::vector<float> values(bytes.size() / sizeof(float));
  if (!values.empty()) {
    std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
  }
  return values;
}

bool throws_with(const std::function<void()>& fn, const std::string& needle) {
  try {
    fn();
  } catch (const std::exception& e) {
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

} // namespace

RUN_TEST(
    "unit_detection_types_boxdecode_payloads_test", ([] {
      using namespace simaai::neat;

      require(detection_format_is_bbox(kDetectionFormatBbox), "BBOX predicate failed");
      require(detection_format_is_pose(kDetectionFormatBboxPose), "pose predicate failed");
      require(detection_format_is_segmentation(kDetectionFormatBboxSegmentation),
              "segmentation predicate failed");
      require(detection_format_is_bbox_family(kDetectionFormatBboxPose),
              "pose should be BBOX-family");

      const Tensor pose_tensor =
          make_wire_tensor(make_pose_payload(2, 2), kDetectionFormatBboxPose);
      const PoseDecodeTensorList pose = decode_pose(TensorList{pose_tensor});
      require(pose.size() == 1, "decode_pose should be positional 1:1");
      require(pose[0].boxes.dtype == TensorDType::Float32, "pose boxes dtype mismatch");
      require(pose[0].boxes.shape == std::vector<int64_t>({2, kDecodedBoxColumns}),
              "pose boxes shape mismatch");
      require(pose[0].keypoints.dtype == TensorDType::Float32, "pose keypoints dtype mismatch");
      require(pose[0].keypoints.shape ==
                  std::vector<int64_t>({2, kDecodedPoseKeypoints, kDecodedPoseColumns}),
              "pose keypoints shape mismatch");
      const std::vector<float> pose_boxes = tensor_float_values(pose[0].boxes);
      require(pose_boxes[0] == 10.0f && pose_boxes[2] == 40.0f && pose_boxes[5] == 3.0f,
              "pose boxes values mismatch");
      const std::vector<float> keypoints = tensor_float_values(pose[0].keypoints);
      require(keypoints[0] == 1000.0f && keypoints[1] == 2000.0f, "first pose keypoint mismatch");
      require(keypoints[static_cast<std::size_t>(kDecodedPoseColumns) *
                        static_cast<std::size_t>(kDecodedPoseKeypoints)] == 1100.0f,
              "second detection pose keypoint mismatch");

      const TensorList pose_as_bbox = decode_bbox(TensorList{pose_tensor});
      require(pose_as_bbox.size() == 1 && pose_as_bbox[0].shape[1] == kDecodedBoxColumns,
              "decode_bbox should keep working on pose payloads");

      const Tensor pose_tensor_with_bbox_tag =
          make_wire_tensor(make_pose_payload(1, 1), kDetectionFormatBbox);
      require(decode_pose(TensorList{pose_tensor_with_bbox_tag}).front().keypoints.shape[0] == 1,
              "decode_pose should accept current BBOX-caps pose payloads");

      const Tensor seg_tensor =
          make_wire_tensor(make_segmentation_payload(2, 2), kDetectionFormatBboxSegmentation);
      const SegmentationDecodeTensorList seg =
          decode_segmentation(TensorList{seg_tensor}, 0, 0, 1, false);
      require(seg.size() == 1, "decode_segmentation should be positional 1:1");
      require(seg[0].boxes.shape == std::vector<int64_t>({1, kDecodedBoxColumns}),
              "segmentation top_k should cap boxes");
      require(seg[0].masks.dtype == TensorDType::UInt8, "segmentation masks dtype mismatch");
      require(seg[0].masks.shape ==
                  std::vector<int64_t>({1, kDecodedMaskHeight, kDecodedMaskWidth}),
              "segmentation masks shape mismatch");
      const std::vector<uint8_t> mask_bytes = seg[0].masks.copy_payload_bytes();
      require(mask_bytes.size() == static_cast<std::size_t>(kDecodedMaskWidth) *
                                       static_cast<std::size_t>(kDecodedMaskHeight),
              "segmentation mask payload size mismatch");
      require(mask_bytes[0] == 0 && mask_bytes[255] == 255, "segmentation mask values mismatch");

      require(throws_with([&]() { (void)decode_pose(TensorList{seg_tensor}); }, "format mismatch"),
              "decode_pose should reject segmentation-tagged tensors");

      Tensor malformed_pose =
          make_wire_tensor(std::vector<uint8_t>{0, 0, 0, 0, 1}, kDetectionFormatBboxPose);
      require(throws_with([&]() { (void)decode_pose(TensorList{malformed_pose}); }, "payload size"),
              "decode_pose should reject malformed pose payload sizes");

      // ---- combined segmentation + pose ----

      require(detection_format_is_segmentation_pose(kDetectionFormatBboxSegmentationPose),
              "segmentation-pose predicate failed");
      require(detection_format_is_bbox_family(kDetectionFormatBboxSegmentationPose),
              "segmentation-pose should be BBOX-family");
      // The aliases are part of the accepted wire contract, not incidental.
      require(detection_format_is_segmentation_pose("SEG_POSE") &&
                  detection_format_is_segmentation_pose("BBOX_SEG_POSE"),
              "segmentation-pose aliases should be accepted");
      // A combined payload is neither of the single-extension formats; treating it as one
      // would parse the wrong region as the extension.
      require(!detection_format_is_pose(kDetectionFormatBboxSegmentationPose) &&
                  !detection_format_is_segmentation(kDetectionFormatBboxSegmentationPose),
              "segmentation-pose must not satisfy the single-extension predicates");

      const Tensor seg_pose_tensor = make_wire_tensor(make_segmentation_pose_payload(2, 2),
                                                      kDetectionFormatBboxSegmentationPose);
      const SegmentationPoseDecodeTensorList seg_pose =
          decode_segmentation_pose(TensorList{seg_pose_tensor});
      require(seg_pose.size() == 1, "decode_segmentation_pose should be positional 1:1");
      require(seg_pose[0].boxes.dtype == TensorDType::Float32 &&
                  seg_pose[0].boxes.shape == std::vector<int64_t>({2, kDecodedBoxColumns}),
              "segmentation-pose boxes shape/dtype mismatch");
      require(seg_pose[0].masks.dtype == TensorDType::UInt8 &&
                  seg_pose[0].masks.shape ==
                      std::vector<int64_t>({2, kDecodedMaskHeight, kDecodedMaskWidth}),
              "segmentation-pose masks shape/dtype mismatch");
      require(seg_pose[0].keypoints.dtype == TensorDType::Float32 &&
                  seg_pose[0].keypoints.shape ==
                      std::vector<int64_t>({2, kDecodedPoseKeypoints, kDecodedPoseColumns}),
              "segmentation-pose keypoints shape/dtype mismatch");

      const std::vector<float> sp_boxes = tensor_float_values(seg_pose[0].boxes);
      require(sp_boxes[0] == 100.0f && sp_boxes[5] == 7.0f,
              "segmentation-pose first box values mismatch");
      require(sp_boxes[static_cast<std::size_t>(kDecodedBoxColumns)] == 101.0f,
              "segmentation-pose second box values mismatch");

      // Slot 0 and slot 1 are checked separately: a wrong mask base breaks both, a wrong
      // mask stride breaks only the second.
      const std::vector<uint8_t> sp_masks = seg_pose[0].masks.copy_payload_bytes();
      const std::size_t sp_mask_bytes = static_cast<std::size_t>(kDecodedMaskWidth) *
                                        static_cast<std::size_t>(kDecodedMaskHeight);
      require(sp_masks.size() == 2U * sp_mask_bytes,
              "segmentation-pose mask payload size mismatch");
      require(sp_masks[0] == 1 && sp_masks[sp_mask_bytes - 1] == 1,
              "segmentation-pose slot 0 mask should be filled with 1");
      require(sp_masks[sp_mask_bytes] == 2 && sp_masks[2U * sp_mask_bytes - 1] == 2,
              "segmentation-pose slot 1 mask should be filled with 2");

      // The keypoint region starts after the entire mask region. If that offset were the
      // pose-only one, slot 0 would decode mask fill bytes instead of 1000.
      const std::vector<float> sp_keypoints = tensor_float_values(seg_pose[0].keypoints);
      require(sp_keypoints[0] == 1000.0f && sp_keypoints[1] == 2000.0f,
              "segmentation-pose slot 0 keypoint mismatch: pose region offset is wrong");
      const std::size_t sp_pose_stride = static_cast<std::size_t>(kDecodedPoseKeypoints) *
                                         static_cast<std::size_t>(kDecodedPoseColumns);
      require(sp_keypoints[sp_pose_stride] == 2000.0f &&
                  sp_keypoints[sp_pose_stride + 1] == 4000.0f,
              "segmentation-pose slot 1 keypoint mismatch: pose region stride is wrong");

      // top_k caps every region together; the regions must stay slot-aligned with each other.
      const SegmentationPoseDecodeTensorList seg_pose_topk =
          decode_segmentation_pose(TensorList{seg_pose_tensor}, 0, 0, 1, false);
      require(seg_pose_topk[0].boxes.shape == std::vector<int64_t>({1, kDecodedBoxColumns}),
              "segmentation-pose top_k should cap boxes");
      require(seg_pose_topk[0].masks.shape ==
                  std::vector<int64_t>({1, kDecodedMaskHeight, kDecodedMaskWidth}),
              "segmentation-pose top_k should cap masks");
      require(seg_pose_topk[0].keypoints.shape ==
                  std::vector<int64_t>({1, kDecodedPoseKeypoints, kDecodedPoseColumns}),
              "segmentation-pose top_k should cap keypoints");
      require(seg_pose_topk[0].masks.copy_payload_bytes()[0] == 1,
              "segmentation-pose top_k should keep slot 0's mask");

      // Box-leading is the reason the existing bbox helpers stay valid on this payload.
      const TensorList seg_pose_as_bbox = decode_bbox(TensorList{seg_pose_tensor});
      require(seg_pose_as_bbox.size() == 1 && seg_pose_as_bbox[0].shape[1] == kDecodedBoxColumns,
              "decode_bbox should keep working on segmentation-pose payloads");

      require(throws_with([&]() { (void)decode_segmentation_pose(TensorList{seg_tensor}); },
                          "format mismatch"),
              "decode_segmentation_pose should reject segmentation-tagged tensors");
      require(
          throws_with([&]() { (void)decode_pose(TensorList{seg_pose_tensor}); }, "format mismatch"),
          "decode_pose should reject segmentation-pose-tagged tensors");

      // One byte short of a whole slot: the combined stride is boxes + masks + poses, so a
      // payload that would divide evenly for a pose-only or seg-only layout must still fail.
      std::vector<uint8_t> short_seg_pose = make_segmentation_pose_payload(1, 1);
      short_seg_pose.pop_back();
      Tensor malformed_seg_pose =
          make_wire_tensor(short_seg_pose, kDetectionFormatBboxSegmentationPose);
      require(throws_with([&]() { (void)decode_segmentation_pose(TensorList{malformed_seg_pose}); },
                          "payload size"),
              "decode_segmentation_pose should reject malformed payload sizes");

      // A BBOX-tagged combined payload is accepted, matching the pose case above.
      const Tensor seg_pose_with_bbox_tag =
          make_wire_tensor(make_segmentation_pose_payload(1, 1), kDetectionFormatBbox);
      require(
          decode_segmentation_pose(TensorList{seg_pose_with_bbox_tag}).front().keypoints.shape[0] ==
              1,
          "decode_segmentation_pose should accept current BBOX-caps combined payloads");

      // Empty output: a full-capacity buffer with a zero count yields zero rows in all
      // three regions, not one row of slot-0 garbage.
      const Tensor empty_seg_pose = make_wire_tensor(make_segmentation_pose_payload(0, 8),
                                                     kDetectionFormatBboxSegmentationPose);
      const auto empty_decoded = decode_segmentation_pose(TensorList{empty_seg_pose}).front();
      require(empty_decoded.boxes.shape == std::vector<int64_t>({0, kDecodedBoxColumns}),
              "an empty segmentation-pose payload must decode to zero boxes");
      require(empty_decoded.masks.shape ==
                  std::vector<int64_t>({0, kDecodedMaskHeight, kDecodedMaskWidth}),
              "an empty segmentation-pose payload must decode to zero masks");
      require(empty_decoded.keypoints.shape ==
                  std::vector<int64_t>({0, kDecodedPoseKeypoints, kDecodedPoseColumns}),
              "an empty segmentation-pose payload must decode to zero keypoints");
      // Pre-existing and cross-cutting, not specific to this family: Tensor::copy_payload_bytes
      // has no zero-byte path, so reading an empty result throws "mapping failed" rather than
      // returning an empty vector. Pinned here so a future fix is a deliberate change; a
      // consumer must currently check shape[0] before copying.
      require(
          throws_with([&]() { (void)empty_decoded.masks.copy_payload_bytes(); }, "mapping failed"),
          "empty-result copy_payload_bytes behaviour changed");

      // Partially filled: capacity sets the region strides, the count sets the row count.
      // Reading slot 3's mask proves the stride still came from capacity, not the count.
      const Tensor partial_seg_pose = make_wire_tensor(make_segmentation_pose_payload(4, 8),
                                                       kDetectionFormatBboxSegmentationPose);
      const auto partial = decode_segmentation_pose(TensorList{partial_seg_pose}).front();
      require(partial.boxes.shape[0] == 4 && partial.masks.shape[0] == 4 &&
                  partial.keypoints.shape[0] == 4,
              "a partially filled payload must decode exactly count rows");
      const std::vector<uint8_t> partial_masks = partial.masks.copy_payload_bytes();
      require(partial_masks[3U * sp_mask_bytes] == 4,
              "slot 3's mask must be reached with a capacity-derived stride");
      const std::vector<float> partial_kpts = tensor_float_values(partial.keypoints);
      require(partial_kpts[3U * static_cast<std::size_t>(kDecodedPoseKeypoints) *
                           static_cast<std::size_t>(kDecodedPoseColumns)] == 4000.0f,
              "slot 3's keypoints must be reached past the full mask region");

      // Capacity limit. The generic bbox bound (body / sizeof(RawBox)) is far too loose for
      // this format - it would admit 2152 rows in a 2-slot payload - so the guard that
      // matters is the capacity derived from the combined stride. Non-strict clamps to it,
      // strict rejects; either way the decoder never reads past the pose region.
      std::vector<uint8_t> overrun = make_segmentation_pose_payload(2, 2);
      const uint32_t impossible = 9;
      std::memcpy(overrun.data(), &impossible, sizeof(impossible));
      const Tensor overrun_tensor = make_wire_tensor(overrun, kDetectionFormatBboxSegmentationPose);
      const auto clamped = decode_segmentation_pose(TensorList{overrun_tensor}).front();
      require(clamped.boxes.shape[0] == 2 && clamped.masks.shape[0] == 2 &&
                  clamped.keypoints.shape[0] == 2,
              "a count header beyond capacity must clamp to capacity, not read past it");
      require(throws_with(
                  [&]() {
                    (void)decode_segmentation_pose(TensorList{overrun_tensor}, 0, 0, 0,
                                                   /*strict=*/true);
                  },
                  "exceeds"),
              "strict mode must reject a count header beyond capacity");

      // top_k caps the rows returned without changing the region strides.
      const Tensor capped_tensor = make_wire_tensor(make_segmentation_pose_payload(4, 8),
                                                    kDetectionFormatBboxSegmentationPose);
      const auto capped =
          decode_segmentation_pose(TensorList{capped_tensor}, 0, 0, /*top_k=*/2, /*strict=*/false)
              .front();
      require(capped.boxes.shape[0] == 2 && capped.masks.shape[0] == 2 &&
                  capped.keypoints.shape[0] == 2,
              "top_k must cap the decoded row count");
      require(capped.masks.copy_payload_bytes()[sp_mask_bytes] == 2,
              "a top_k-capped decode must keep the capacity-derived mask stride");
    }));

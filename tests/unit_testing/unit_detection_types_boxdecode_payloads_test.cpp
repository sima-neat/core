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
    }));

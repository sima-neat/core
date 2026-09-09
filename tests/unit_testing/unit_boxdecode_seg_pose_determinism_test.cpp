#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
/**
 * @example yolox_seg_pose_determinism
 * Repeat, interleave and parallelize the two yolox-seg-pose paths Core owns and require
 * bit-identical results every time.
 *
 * Equality is exact, not toleranced, and that is a requirement rather than a convenience:
 * Core runs no floating-point accumulation on this path. Boxes are int32 corners cast to
 * float, masks and keypoint records are memcpy'd verbatim, and contract resolution is
 * integer and enum work. Any drift at all would mean state leaked between calls, so a
 * tolerance here would hide the only bug the test can find.
 *
 * Ordering is likewise exact: the backend has already scored, suppressed and ordered the
 * detections, and Core must publish that order unchanged. The tied-score and overlapping-box
 * cases exist to catch a sort or a dedup being introduced on this side.
 */
#include "pipeline/BoxDecodeType.h"
#include "pipeline/DetectionTypes.h"
#include "pipeline/TensorCore.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "test_main.h"
#include "test_utils.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace simaai::neat;
using namespace simaai::neat::pipeline_internal::sima;
using namespace simaai::neat::pipeline_internal::sima::stagesemantics;

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
  RawPosePoint points[static_cast<std::size_t>(kDecodedPoseKeypoints)];
};

static_assert(sizeof(RawBox) == 24);
static_assert(sizeof(RawPoseOut) == 204);

const std::size_t kMaskBytes =
    static_cast<std::size_t>(kDecodedMaskWidth) * static_cast<std::size_t>(kDecodedMaskHeight);

// One payload shape used by every case, parameterized by the box records so a case can
// pin ties or overlaps without a second builder.
std::vector<uint8_t> make_payload(uint32_t count, std::size_t capacity,
                                  const std::vector<RawBox>& boxes) {
  std::vector<uint8_t> out(sizeof(uint32_t) +
                           capacity * (sizeof(RawBox) + kMaskBytes + sizeof(RawPoseOut)));
  std::memcpy(out.data(), &count, sizeof(count));

  std::vector<RawBox> slots(capacity);
  for (std::size_t slot = 0; slot < capacity; ++slot) {
    slots[slot] = slot < boxes.size() ? boxes[slot]
                                      : RawBox{.x = static_cast<int32_t>(10 + slot),
                                               .y = static_cast<int32_t>(20 + slot),
                                               .w = 30,
                                               .h = 40,
                                               .score = 0.5f,
                                               .cls = static_cast<int32_t>(slot)};
  }
  std::memcpy(out.data() + sizeof(uint32_t), slots.data(), capacity * sizeof(RawBox));

  const std::size_t mask_base = sizeof(uint32_t) + capacity * sizeof(RawBox);
  for (std::size_t slot = 0; slot < capacity; ++slot) {
    std::memset(out.data() + mask_base + slot * kMaskBytes, static_cast<int>(slot + 1), kMaskBytes);
  }

  const std::size_t pose_base = mask_base + capacity * kMaskBytes;
  for (std::size_t slot = 0; slot < capacity; ++slot) {
    RawPoseOut pose{};
    for (std::size_t point = 0; point < static_cast<std::size_t>(kDecodedPoseKeypoints); ++point) {
      pose.points[point].x = static_cast<uint32_t>(1000 * (slot + 1) + point);
      pose.points[point].y = static_cast<uint32_t>(2000 * (slot + 1) + point);
      pose.points[point].visible = 0.01f * static_cast<float>(point + 1);
    }
    std::memcpy(out.data() + pose_base + slot * sizeof(RawPoseOut), &pose, sizeof(pose));
  }
  return out;
}

Tensor make_tensor(const std::vector<uint8_t>& payload) {
  auto storage = make_cpu_owned_storage(payload.size());
  Mapping dst = storage->map(MapMode::Write);
  std::memcpy(dst.data, payload.data(), payload.size());
  Tensor t;
  t.storage = std::move(storage);
  t.device = {DeviceType::CPU, 0};
  t.dtype = TensorDType::UInt8;
  t.layout = TensorLayout::Unknown;
  t.shape = {static_cast<int64_t>(payload.size())};
  tag_detection_format(t, kDetectionFormatBboxSegmentationPose);
  return t;
}

// Flatten a decode result to bytes so comparison is exact and covers all three regions
// plus their shapes.
struct DecodedBytes {
  std::vector<int64_t> shapes;
  std::vector<uint8_t> boxes;
  std::vector<uint8_t> masks;
  std::vector<uint8_t> keypoints;

  bool operator==(const DecodedBytes& other) const {
    return shapes == other.shapes && boxes == other.boxes && masks == other.masks &&
           keypoints == other.keypoints;
  }
};

std::vector<uint8_t> region_bytes(const Tensor& t) {
  if (t.shape.empty() || t.shape[0] == 0) {
    return {};
  }
  return t.copy_payload_bytes();
}

DecodedBytes decode_once(const Tensor& tensor, int top_k = 0) {
  const auto result = decode_segmentation_pose(TensorList{tensor}, 0, 0, top_k).front();
  DecodedBytes out;
  for (const Tensor* region : {&result.boxes, &result.masks, &result.keypoints}) {
    out.shapes.insert(out.shapes.end(), region->shape.begin(), region->shape.end());
  }
  out.boxes = region_bytes(result.boxes);
  out.masks = region_bytes(result.masks);
  out.keypoints = region_bytes(result.keypoints);
  return out;
}

BoxDecodeStaticContract make_contract(int classes, const std::vector<int>& grid_w) {
  const std::vector<std::pair<const char*, int>> roles = {
      {"bbox", 4}, {"cls_score", classes + 1}, {"mask_coeff", 32}, {"kpt", 39}};
  BoxDecodeStaticContract contract;
  contract.decode_type = BoxDecodeType::YoloXSegPose;
  for (const auto& [role, depth] : roles) {
    for (std::size_t level = 0; level < grid_w.size(); ++level) {
      BoxDecodeTensorStaticContract t;
      t.logical_name = std::string(role) + "_" + std::to_string(level);
      t.data_type = "BF16";
      t.layout = "HWC";
      t.input_shape = {grid_w[level], grid_w[level], depth};
      t.slice_shape = t.input_shape;
      t.source_storage_kind = BoxDecodeSourceStorageKind::DenseHwcPhysical;
      contract.tensors.push_back(std::move(t));
    }
  }
  BoxDecodeTensorStaticContract proto;
  proto.logical_name = "mask_proto";
  proto.data_type = "BF16";
  proto.layout = "HWC";
  proto.input_shape = {160, 160, 32};
  proto.slice_shape = proto.input_shape;
  proto.source_storage_kind = BoxDecodeSourceStorageKind::DenseHwcPhysical;
  contract.tensors.push_back(std::move(proto));
  return contract;
}

// The compiled payload fields a stale contract would corrupt, as a comparable string.
std::string compiled_fingerprint(const BoxDecodeStaticContract& contract) {
  const auto finalized = finalize_boxdecode_static_contract(
      contract, BoxDecodeType::YoloXSegPose, std::nullopt, std::nullopt, BoxDecodeTypeOption::Auto,
      0.30, 0.60, 100, /*num_classes=*/0, {});
  const auto compiled = build_boxdecode_compiled_contract(finalized);
  std::string out = std::to_string(static_cast<int>(compiled.payload.decode_type)) + "|" +
                    std::to_string(compiled.payload.num_classes) + "|" +
                    std::to_string(static_cast<int>(*compiled.payload.decode_type_option)) + "|" +
                    std::to_string(static_cast<int>(compiled.payload.score_activation)) + "|" +
                    compiled.payload.input_dtype + "|" +
                    std::to_string(compiled.payload.slice_shapes.size());
  for (const int cls : compiled.payload.pose_classes) {
    out += "," + std::to_string(cls);
  }
  return out;
}

} // namespace

RUN_TEST(
    "unit_boxdecode_seg_pose_determinism_test", ([] {
      constexpr int kRepeats = 64;

      // A populated frame, an empty frame, and a frame whose header exceeds capacity.
      const Tensor full = make_tensor(make_payload(3, 4, {}));
      const Tensor empty = make_tensor(make_payload(0, 4, {}));
      std::vector<uint8_t> over = make_payload(2, 2, {});
      const uint32_t impossible = 9;
      std::memcpy(over.data(), &impossible, sizeof(impossible));
      const Tensor capped = make_tensor(over);

      const DecodedBytes full_ref = decode_once(full);
      const DecodedBytes empty_ref = decode_once(empty);
      const DecodedBytes capped_ref = decode_once(capped);
      require(full_ref.shapes[0] == 3, "the populated fixture must decode three detections");
      require(empty_ref.shapes[0] == 0, "the empty fixture must decode no detections");
      require(capped_ref.shapes[0] == 2, "the over-capacity fixture must clamp to two");

      // Repeated decoding of identical input.
      for (int i = 0; i < kRepeats; ++i) {
        require(decode_once(full) == full_ref, "repeated decode of one payload must be identical");
        require(decode_once(empty) == empty_ref, "repeated decode of an empty frame must match");
        require(decode_once(capped) == capped_ref, "repeated clamped decode must match");
      }

      // Alternating inputs: a decoder holding stale state would carry the previous frame's
      // capacity or row count into the next one.
      for (int i = 0; i < kRepeats; ++i) {
        require(decode_once(full) == full_ref, "alternating: populated frame drifted");
        require(decode_once(empty) == empty_ref, "alternating: empty frame drifted");
        require(decode_once(capped) == capped_ref, "alternating: clamped frame drifted");
        require(decode_once(empty) == empty_ref, "alternating: empty after clamped drifted");
      }

      // top_k changes the row count but must not change the bytes of the rows kept.
      const DecodedBytes capped_topk_ref = decode_once(full, /*top_k=*/2);
      require(capped_topk_ref.shapes[0] == 2, "top_k must cap the row count");
      for (int i = 0; i < kRepeats; ++i) {
        require(decode_once(full, 2) == capped_topk_ref, "repeated top_k decode must match");
        require(decode_once(full) == full_ref, "an uncapped decode after a capped one must match");
      }

      // Concurrent decoding. Core holds no state on this path; this fails loudly if any is
      // introduced.
      {
        std::atomic<int> failures{0};
        std::vector<std::thread> workers;
        for (int t = 0; t < 8; ++t) {
          workers.emplace_back([&]() {
            for (int i = 0; i < kRepeats; ++i) {
              if (!(decode_once(full) == full_ref) || !(decode_once(empty) == empty_ref) ||
                  !(decode_once(capped) == capped_ref)) {
                failures.fetch_add(1);
              }
            }
          });
        }
        for (auto& worker : workers) {
          worker.join();
        }
        require(failures.load() == 0, "concurrent decoding must match the single-threaded result");
      }

      // Tied scores must keep payload order. The backend has already ordered these; a sort
      // introduced in Core would reorder equal keys and silently relabel every detection.
      {
        std::vector<RawBox> tied(4);
        for (std::size_t i = 0; i < tied.size(); ++i) {
          tied[i] = RawBox{.x = static_cast<int32_t>(10 * (i + 1)),
                           .y = 20,
                           .w = 30,
                           .h = 40,
                           .score = 0.75f,
                           .cls = static_cast<int32_t>(40 - i)};
        }
        const Tensor tensor = make_tensor(make_payload(4, 4, tied));
        const auto result = decode_segmentation_pose(TensorList{tensor}).front();
        const std::vector<uint8_t> raw = result.boxes.copy_payload_bytes();
        std::vector<float> values(raw.size() / sizeof(float));
        std::memcpy(values.data(), raw.data(), raw.size());
        for (std::size_t i = 0; i < tied.size(); ++i) {
          const std::size_t row = i * static_cast<std::size_t>(kDecodedBoxColumns);
          require(values[row] == static_cast<float>(tied[i].x),
                  "tied scores must preserve payload order");
          require(values[row + 5] == static_cast<float>(tied[i].cls),
                  "tied scores must keep each row's own class");
        }
        const DecodedBytes tied_ref = decode_once(tensor);
        for (int i = 0; i < kRepeats; ++i) {
          require(decode_once(tensor) == tied_ref, "tied-score decode must be repeatable");
        }
      }

      // Equal scores cannot catch a stable sort, so pin the stronger property too: the
      // backend's order is authoritative even when it is not score-descending.
      {
        std::vector<RawBox> ascending(4);
        for (std::size_t i = 0; i < ascending.size(); ++i) {
          ascending[i] = RawBox{.x = static_cast<int32_t>(10 * (i + 1)),
                                .y = 20,
                                .w = 30,
                                .h = 40,
                                .score = 0.1f * static_cast<float>(i + 1),
                                .cls = static_cast<int32_t>(i)};
        }
        const Tensor tensor = make_tensor(make_payload(4, 4, ascending));
        const auto result = decode_segmentation_pose(TensorList{tensor}).front();
        const std::vector<uint8_t> raw = result.boxes.copy_payload_bytes();
        std::vector<float> values(raw.size() / sizeof(float));
        std::memcpy(values.data(), raw.data(), raw.size());
        for (std::size_t i = 0; i < ascending.size(); ++i) {
          const std::size_t row = i * static_cast<std::size_t>(kDecodedBoxColumns);
          require(values[row + 5] == static_cast<float>(ascending[i].cls),
                  "ascending payload scores must not be reordered by Core");
        }
        // The mask region is indexed by payload slot, so a reordering that moved boxes
        // without moving masks would mismatch a detection with another's mask.
        const std::vector<uint8_t> masks = result.masks.copy_payload_bytes();
        for (std::size_t i = 0; i < ascending.size(); ++i) {
          require(masks[i * kMaskBytes] == static_cast<uint8_t>(i + 1),
                  "each row's mask must still belong to its own payload slot");
        }
      }

      // Fully overlapping boxes must all survive: suppression already happened on device.
      {
        std::vector<RawBox> overlapping(4);
        for (std::size_t i = 0; i < overlapping.size(); ++i) {
          overlapping[i] = RawBox{.x = 100,
                                  .y = 100,
                                  .w = 50,
                                  .h = 50,
                                  .score = 0.9f - 0.1f * static_cast<float>(i),
                                  .cls = static_cast<int32_t>(i)};
        }
        const Tensor tensor = make_tensor(make_payload(4, 4, overlapping));
        const auto result = decode_segmentation_pose(TensorList{tensor}).front();
        require(result.boxes.shape[0] == 4,
                "identical boxes must not be deduplicated on the Core side");
        require(result.masks.shape[0] == 4 && result.keypoints.shape[0] == 4,
                "every retained box must keep its mask and keypoint rows");
      }

      // Contract resolution: same heads in, same compiled payload out, including when
      // interleaved with a different geometry.
      {
        const BoxDecodeStaticContract a = make_contract(36, {80, 40, 20});
        const BoxDecodeStaticContract b = make_contract(3, {64, 32, 16});
        const std::string a_ref = compiled_fingerprint(a);
        const std::string b_ref = compiled_fingerprint(b);
        require(a_ref != b_ref, "the two contract fixtures must differ");
        for (int i = 0; i < kRepeats; ++i) {
          require(compiled_fingerprint(a) == a_ref, "repeated contract resolution must match");
          require(compiled_fingerprint(b) == b_ref, "interleaved contract resolution must match");
        }

        std::atomic<int> failures{0};
        std::vector<std::thread> workers;
        for (int t = 0; t < 8; ++t) {
          workers.emplace_back([&]() {
            for (int i = 0; i < kRepeats; ++i) {
              if (compiled_fingerprint(a) != a_ref || compiled_fingerprint(b) != b_ref) {
                failures.fetch_add(1);
              }
            }
          });
        }
        for (auto& worker : workers) {
          worker.join();
        }
        require(failures.load() == 0, "concurrent contract resolution must be deterministic");
      }
    }));

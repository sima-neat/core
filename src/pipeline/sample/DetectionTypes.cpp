#include "pipeline/DetectionTypes.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/internal/TensorMath.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>

namespace simaai::neat {
using pipeline_internal::upper_copy;
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
  RawPosePoint points[static_cast<std::size_t>(kDecodedPoseKeypoints)];
};

static_assert(sizeof(RawBox) == 24, "BoxDecode BBOX wire record must stay 24 bytes");
static_assert(sizeof(RawPoseOut) == 204, "BoxDecode pose wire record must stay 204 bytes");

struct ParsedBoxRecords {
  std::vector<Box> boxes;
  std::vector<std::size_t> record_indices;
};

std::string normalize_detection_format(std::string value) {
  value = upper_copy(std::move(value));
  for (char& c : value) {
    if (c == '-' || c == ' ')
      c = '_';
  }
  return value;
}

ParsedBoxRecords parse_bbox_records(const std::vector<uint8_t>& bytes, int img_w, int img_h,
                                    int expected_topk, bool strict) {
  ParsedBoxRecords out;
  if (bytes.size() < sizeof(uint32_t)) {
    if (strict)
      throw std::runtime_error("bbox buffer too small");
    return out;
  }

  const size_t payload = bytes.size() - sizeof(uint32_t);
  if (payload < sizeof(RawBox)) {
    if (strict)
      throw std::runtime_error("bbox buffer payload too small");
    return out;
  }

  uint32_t header = 0;
  std::memcpy(&header, bytes.data(), sizeof(header));
  const size_t max_boxes = payload / sizeof(RawBox);
  if (header > max_boxes) {
    if (strict)
      throw std::runtime_error("bbox header exceeds payload count");
    header = static_cast<uint32_t>(max_boxes);
  }

  if (expected_topk > 0) {
    if (strict && static_cast<size_t>(header) > static_cast<size_t>(expected_topk)) {
      throw std::runtime_error("bbox header exceeds expected topk");
    }
    if (!strict) {
      header =
          static_cast<uint32_t>(std::min<std::size_t>(header, static_cast<size_t>(expected_topk)));
    }
  }

  out.boxes.reserve(header);
  out.record_indices.reserve(header);
  const uint8_t* base = bytes.data() + sizeof(uint32_t);
  for (size_t i = 0; i < header; ++i) {
    RawBox r{};
    std::memcpy(&r, base + i * sizeof(RawBox), sizeof(RawBox));

    float x1 = static_cast<float>(r.x);
    float y1 = static_cast<float>(r.y);
    float x2 = static_cast<float>(r.x + r.w);
    float y2 = static_cast<float>(r.y + r.h);

    if (img_w > 0 && img_h > 0) {
      x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_w)));
      y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_h)));
      x2 = std::max(0.0f, std::min(x2, static_cast<float>(img_w)));
      y2 = std::max(0.0f, std::min(y2, static_cast<float>(img_h)));
    }

    if (!strict && (x2 <= x1 || y2 <= y1))
      continue;
    out.boxes.push_back(Box{x1, y1, x2, y2, r.score, static_cast<int>(r.cls)});
    out.record_indices.push_back(i);
  }

  return out;
}

std::vector<uint8_t> copy_detection_payload(const simaai::neat::Tensor& tensor, const char* kind) {
  if (!tensor.storage) {
    throw std::runtime_error(std::string(kind) + " tensor missing storage");
  }
  if (!tensor.is_dense()) {
    throw std::runtime_error(std::string(kind) + " tensor must be dense");
  }
  try {
    return tensor.copy_payload_bytes();
  } catch (const std::exception& ex) {
    throw std::runtime_error(std::string(kind) + " tensor copy failed: " + ex.what());
  }
}

void validate_extended_detection_format(const simaai::neat::Tensor& tensor, const char* kind,
                                        bool (*predicate)(const std::string&)) {
  const std::string fmt = read_detection_format(tensor);
  if (!fmt.empty() && !predicate(fmt) && !detection_format_is_bbox(fmt)) {
    throw std::runtime_error(std::string(kind) +
                             " tensor format mismatch: " + normalize_detection_format(fmt));
  }
  if (fmt.empty() &&
      (tensor.shape.size() != 1U || tensor.dtype != simaai::neat::TensorDType::UInt8)) {
    throw std::runtime_error(std::string(kind) + " tensor is not a " + kind +
                             " tensor (no detection tag and not a rank-1 UInt8 buffer)");
  }
}

std::size_t infer_extended_capacity(const std::vector<uint8_t>& bytes,
                                    std::size_t extension_record_bytes, const char* kind,
                                    bool strict) {
  if (bytes.size() < sizeof(uint32_t)) {
    if (strict)
      throw std::runtime_error(std::string(kind) + " buffer too small");
    return 0;
  }
  const std::size_t body = bytes.size() - sizeof(uint32_t);
  const std::size_t stride = sizeof(RawBox) + extension_record_bytes;
  if (stride == 0 || body % stride != 0) {
    throw std::runtime_error(std::string(kind) + " payload size does not match BoxDecode wire "
                                                 "layout");
  }
  return body / stride;
}

simaai::neat::Tensor keypoints_to_tensor(const std::vector<uint8_t>& bytes,
                                         const std::vector<std::size_t>& record_indices,
                                         std::size_t capacity) {
  const int64_t n = static_cast<int64_t>(record_indices.size());
  const int64_t rows = kDecodedPoseKeypoints;
  const int64_t cols = kDecodedPoseColumns;
  const std::size_t bytes_out = static_cast<std::size_t>(n) * static_cast<std::size_t>(rows) *
                                static_cast<std::size_t>(cols) * sizeof(float);

  auto storage = simaai::neat::make_cpu_owned_storage(bytes_out);
  if (bytes_out > 0) {
    simaai::neat::Mapping dst = storage->map(simaai::neat::MapMode::Write);
    auto* out = static_cast<float*>(dst.data);
    const std::size_t pose_base = sizeof(uint32_t) + capacity * sizeof(RawBox);
    for (std::size_t i = 0; i < record_indices.size(); ++i) {
      const std::size_t source_index = record_indices[i];
      RawPoseOut pose{};
      std::memcpy(&pose, bytes.data() + pose_base + source_index * sizeof(RawPoseOut),
                  sizeof(RawPoseOut));
      for (std::size_t point = 0; point < static_cast<std::size_t>(kDecodedPoseKeypoints);
           ++point) {
        float* row = out + (i * static_cast<std::size_t>(kDecodedPoseKeypoints) + point) *
                               static_cast<std::size_t>(kDecodedPoseColumns);
        row[0] = static_cast<float>(pose.points[point].x);
        row[1] = static_cast<float>(pose.points[point].y);
        row[2] = pose.points[point].visible;
      }
    }
  }

  simaai::neat::Tensor t;
  t.storage = std::move(storage);
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = false;
  t.byte_offset = 0;
  t.dtype = simaai::neat::TensorDType::Float32;
  t.layout = simaai::neat::TensorLayout::Unknown;
  t.shape = {n, rows, cols};
  t.strides_bytes = {rows * cols * static_cast<int64_t>(sizeof(float)),
                     cols * static_cast<int64_t>(sizeof(float)),
                     static_cast<int64_t>(sizeof(float))};
  return t;
}

simaai::neat::Tensor masks_to_tensor(const std::vector<uint8_t>& bytes,
                                     const std::vector<std::size_t>& record_indices,
                                     std::size_t capacity) {
  const int64_t n = static_cast<int64_t>(record_indices.size());
  const std::size_t mask_bytes =
      static_cast<std::size_t>(kDecodedMaskWidth) * static_cast<std::size_t>(kDecodedMaskHeight);
  const std::size_t bytes_out = static_cast<std::size_t>(n) * mask_bytes;

  auto storage = simaai::neat::make_cpu_owned_storage(bytes_out);
  if (bytes_out > 0) {
    simaai::neat::Mapping dst = storage->map(simaai::neat::MapMode::Write);
    auto* out = static_cast<uint8_t*>(dst.data);
    const std::size_t mask_base = sizeof(uint32_t) + capacity * sizeof(RawBox);
    for (std::size_t i = 0; i < record_indices.size(); ++i) {
      const std::size_t source_index = record_indices[i];
      std::memcpy(out + i * mask_bytes, bytes.data() + mask_base + source_index * mask_bytes,
                  mask_bytes);
    }
  }

  simaai::neat::Tensor t;
  t.storage = std::move(storage);
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = false;
  t.byte_offset = 0;
  t.dtype = simaai::neat::TensorDType::UInt8;
  t.layout = simaai::neat::TensorLayout::Unknown;
  t.shape = {n, kDecodedMaskHeight, kDecodedMaskWidth};
  t.strides_bytes = {kDecodedMaskHeight * kDecodedMaskWidth, kDecodedMaskWidth, 1};
  return t;
}

} // namespace

bool detection_format_is_bbox(const std::string& format) {
  return normalize_detection_format(format) == kDetectionFormatBbox;
}

bool detection_format_is_pose(const std::string& format) {
  const std::string fmt = normalize_detection_format(format);
  return fmt == kDetectionFormatBboxPose || fmt == "POSE";
}

bool detection_format_is_segmentation(const std::string& format) {
  const std::string fmt = normalize_detection_format(format);
  return fmt == kDetectionFormatBboxSegmentation || fmt == "BBOX_SEG" || fmt == "SEGMENTATION" ||
         fmt == "SEG";
}

bool detection_format_is_bbox_family(const std::string& format) {
  return detection_format_is_bbox(format) || detection_format_is_pose(format) ||
         detection_format_is_segmentation(format);
}

std::vector<Box> parse_bbox_bytes(const std::vector<uint8_t>& bytes, int img_w, int img_h,
                                  int expected_topk, bool strict) {
  return parse_bbox_records(bytes, img_w, img_h, expected_topk, strict).boxes;
}

BoxDecodeResult decode_bbox_tensor(const simaai::neat::Tensor& tensor, int img_w, int img_h,
                                   int expected_topk, bool strict) {
  if (!tensor.storage) {
    throw std::runtime_error("bbox tensor missing storage");
  }
  if (!tensor.is_dense()) {
    throw std::runtime_error("bbox tensor must be dense");
  }

  const std::string fmt = read_detection_format(tensor);
  if (!fmt.empty() && !detection_format_is_bbox_family(fmt)) {
    throw std::runtime_error("bbox tensor format mismatch: " + fmt);
  }

  BoxDecodeResult out;
  out.raw = copy_detection_payload(tensor, "bbox");
  out.boxes = parse_bbox_bytes(out.raw, img_w, img_h, expected_topk, strict);
  return out;
}

simaai::neat::Tensor boxes_to_tensor(const std::vector<Box>& boxes) {
  const int64_t n = static_cast<int64_t>(boxes.size());
  const std::size_t bytes =
      static_cast<std::size_t>(n) * static_cast<std::size_t>(kDecodedBoxColumns) * sizeof(float);

  auto storage = simaai::neat::make_cpu_owned_storage(bytes);
  if (bytes > 0) {
    simaai::neat::Mapping dst = storage->map(simaai::neat::MapMode::Write);
    auto* out = static_cast<float*>(dst.data);
    for (int64_t i = 0; i < n; ++i) {
      const Box& b = boxes[static_cast<std::size_t>(i)];
      float* row = out + i * kDecodedBoxColumns;
      row[0] = b.x1;
      row[1] = b.y1;
      row[2] = b.x2;
      row[3] = b.y2;
      row[4] = b.score;
      row[5] = static_cast<float>(b.class_id);
    }
  }

  simaai::neat::Tensor t;
  t.storage = std::move(storage);
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = false;
  t.byte_offset = 0;
  t.dtype = simaai::neat::TensorDType::Float32;
  t.layout = simaai::neat::TensorLayout::Unknown;
  t.shape = {n, kDecodedBoxColumns};
  t.strides_bytes = {kDecodedBoxColumns * static_cast<int64_t>(sizeof(float)),
                     static_cast<int64_t>(sizeof(float))};
  return t;
}

simaai::neat::TensorList decode_bbox(const simaai::neat::TensorList& bbox_tensors, int img_w,
                                     int img_h, int top_k, bool strict) {
  simaai::neat::TensorList out;
  out.reserve(bbox_tensors.size());
  for (std::size_t i = 0; i < bbox_tensors.size(); ++i) {
    const simaai::neat::Tensor& in = bbox_tensors[i];
    const std::string fmt = read_detection_format(in);
    // Prefer the type-honest detection tag. The producer (EV74 box-decode
    // plugin) does not yet stamp it, so fall back to recognizing the canonical
    // BBOX wire shape (rank-1 UInt8) — the same criteria the runtime already
    // uses internally in try_decode_bbox_payload_tensor_sample. A tensor tagged
    // as some *other* detection format is rejected outright.
    const bool looks_like_bbox_wire =
        in.shape.size() == 1U && in.dtype == simaai::neat::TensorDType::UInt8;
    if (!fmt.empty()) {
      if (!detection_format_is_bbox_family(fmt)) {
        throw std::runtime_error("decode_bbox: input tensor " + std::to_string(i) +
                                 " has detection format '" + fmt +
                                 "', expected a BBOX-family format");
      }
    } else if (!looks_like_bbox_wire) {
      throw std::runtime_error(
          "decode_bbox: input tensor " + std::to_string(i) +
          " is not a BBOX tensor (no detection tag and not a rank-1 UInt8 buffer)");
    }
    BoxDecodeResult decoded = decode_bbox_tensor(in, img_w, img_h, top_k, strict);
    out.push_back(boxes_to_tensor(decoded.boxes));
  }
  return out;
}

PoseDecodeTensors decode_pose_tensor(const simaai::neat::Tensor& tensor, int img_w, int img_h,
                                     int top_k, bool strict) {
  validate_extended_detection_format(tensor, "pose", detection_format_is_pose);
  std::vector<uint8_t> bytes = copy_detection_payload(tensor, "pose");
  const std::size_t capacity = infer_extended_capacity(bytes, sizeof(RawPoseOut), "pose", strict);
  const int parse_topk =
      top_k > 0 ? static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(top_k), capacity))
                : static_cast<int>(std::min<std::size_t>(
                      capacity, static_cast<std::size_t>(std::numeric_limits<int>::max())));
  ParsedBoxRecords parsed = parse_bbox_records(bytes, img_w, img_h, parse_topk, strict);
  PoseDecodeTensors out;
  out.boxes = boxes_to_tensor(parsed.boxes);
  out.keypoints = keypoints_to_tensor(bytes, parsed.record_indices, capacity);
  return out;
}

PoseDecodeTensorList decode_pose(const simaai::neat::TensorList& pose_tensors, int img_w, int img_h,
                                 int top_k, bool strict) {
  PoseDecodeTensorList out;
  out.reserve(pose_tensors.size());
  for (std::size_t i = 0; i < pose_tensors.size(); ++i) {
    try {
      out.push_back(decode_pose_tensor(pose_tensors[i], img_w, img_h, top_k, strict));
    } catch (const std::runtime_error& e) {
      throw std::runtime_error("decode_pose: input tensor " + std::to_string(i) + ": " + e.what());
    }
  }
  return out;
}

SegmentationDecodeTensors decode_segmentation_tensor(const simaai::neat::Tensor& tensor, int img_w,
                                                     int img_h, int top_k, bool strict) {
  validate_extended_detection_format(tensor, "segmentation", detection_format_is_segmentation);
  std::vector<uint8_t> bytes = copy_detection_payload(tensor, "segmentation");
  const std::size_t mask_bytes =
      static_cast<std::size_t>(kDecodedMaskWidth) * static_cast<std::size_t>(kDecodedMaskHeight);
  const std::size_t capacity = infer_extended_capacity(bytes, mask_bytes, "segmentation", strict);
  const int parse_topk =
      top_k > 0 ? static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(top_k), capacity))
                : static_cast<int>(std::min<std::size_t>(
                      capacity, static_cast<std::size_t>(std::numeric_limits<int>::max())));
  ParsedBoxRecords parsed = parse_bbox_records(bytes, img_w, img_h, parse_topk, strict);
  SegmentationDecodeTensors out;
  out.boxes = boxes_to_tensor(parsed.boxes);
  out.masks = masks_to_tensor(bytes, parsed.record_indices, capacity);
  return out;
}

SegmentationDecodeTensorList
decode_segmentation(const simaai::neat::TensorList& segmentation_tensors, int img_w, int img_h,
                    int top_k, bool strict) {
  SegmentationDecodeTensorList out;
  out.reserve(segmentation_tensors.size());
  for (std::size_t i = 0; i < segmentation_tensors.size(); ++i) {
    try {
      out.push_back(
          decode_segmentation_tensor(segmentation_tensors[i], img_w, img_h, top_k, strict));
    } catch (const std::runtime_error& e) {
      throw std::runtime_error("decode_segmentation: input tensor " + std::to_string(i) + ": " +
                               e.what());
    }
  }
  return out;
}

void tag_detection_format(simaai::neat::Tensor& tensor, std::string format) {
  if (!tensor.semantic.detection.has_value()) {
    tensor.semantic.detection = simaai::neat::DetectionSpec{};
  }
  tensor.semantic.detection->format = std::move(format);
}

std::string read_detection_format(const simaai::neat::Tensor& tensor) {
  if (tensor.semantic.detection.has_value() && !tensor.semantic.detection->format.empty()) {
    return tensor.semantic.detection->format;
  }
  // Back-compat: legacy producers tagged the BBOX wire format on TessSpec::format.
  // TessSpec semantically describes MLA tile geometry, not detection payloads — so this
  // fallback exists only to bridge the producer migration and will be removed once every
  // BBOX-emitting site calls `tag_detection_format` instead.
  if (tensor.semantic.tess.has_value()) {
    return tensor.semantic.tess->format;
  }
  return {};
}

namespace {

std::string sample_payload_format(const simaai::neat::Sample& sample) {
  if (!sample.payload_tag.empty())
    return sample.payload_tag;
  if (!sample.format.empty())
    return sample.format;
  return {};
}

} // namespace

void tag_detection_format_in_sample(simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::TensorSet && sample.tensors.size() == 1U) {
    simaai::neat::Tensor& tensor = sample.tensors.front();
    std::string fmt = sample_payload_format(sample);
    if (fmt.empty()) {
      fmt = read_detection_format(tensor); // may pick up legacy tess-tagged BBOX
    }
    if (detection_format_is_bbox(fmt)) {
      tag_detection_format(tensor, kDetectionFormatBbox);
    } else if (detection_format_is_pose(fmt)) {
      tag_detection_format(tensor, kDetectionFormatBboxPose);
    } else if (detection_format_is_segmentation(fmt)) {
      tag_detection_format(tensor, kDetectionFormatBboxSegmentation);
    }
  } else if (sample.kind == simaai::neat::SampleKind::Bundle) {
    for (auto& field : sample.fields) {
      tag_detection_format_in_sample(field);
    }
  }
}

} // namespace simaai::neat

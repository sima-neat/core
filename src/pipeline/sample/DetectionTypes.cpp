#include "pipeline/DetectionTypes.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/internal/TensorMath.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
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

} // namespace

std::vector<Box> parse_bbox_bytes(const std::vector<uint8_t>& bytes, int img_w, int img_h,
                                  int expected_topk, bool strict) {
  std::vector<Box> out;
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
      header = static_cast<uint32_t>(
          std::min<std::size_t>(header, static_cast<size_t>(expected_topk)));
    }
  }

  out.reserve(header);
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
    out.push_back(Box{x1, y1, x2, y2, r.score, static_cast<int>(r.cls)});
  }

  return out;
}

BoxDecodeResult decode_bbox_tensor(const simaai::neat::Tensor& tensor, int img_w, int img_h,
                                   int expected_topk, bool strict) {
  if (!tensor.storage) {
    throw std::runtime_error("bbox tensor missing storage");
  }
  if (!tensor.is_dense()) {
    throw std::runtime_error("bbox tensor must be dense");
  }

  const std::string fmt = upper_copy(read_detection_format(tensor));
  if (!fmt.empty() && fmt != "BBOX") {
    throw std::runtime_error("bbox tensor format mismatch: " + fmt);
  }

  BoxDecodeResult out;
  try {
    out.raw = tensor.copy_payload_bytes();
  } catch (const std::exception& ex) {
    throw std::runtime_error(std::string("bbox tensor copy failed: ") + ex.what());
  }
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
    const std::string fmt = upper_copy(read_detection_format(in));
    // Prefer the type-honest detection tag. The producer (EV74 box-decode
    // plugin) does not yet stamp it, so fall back to recognizing the canonical
    // BBOX wire shape (rank-1 UInt8) — the same criteria the runtime already
    // uses internally in try_decode_bbox_payload_tensor_sample. A tensor tagged
    // as some *other* detection format is rejected outright.
    const bool looks_like_bbox_wire =
        in.shape.size() == 1U && in.dtype == simaai::neat::TensorDType::UInt8;
    if (!fmt.empty()) {
      if (fmt != "BBOX") {
        throw std::runtime_error(
            "decode_bbox: input tensor " + std::to_string(i) +
            " has detection format '" + fmt + "', expected 'BBOX'");
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
  if (!sample.payload_tag.empty()) return sample.payload_tag;
  if (!sample.format.empty())      return sample.format;
  return {};
}

} // namespace

void tag_detection_format_in_sample(simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::TensorSet && sample.tensors.size() == 1U) {
    simaai::neat::Tensor& tensor = sample.tensors.front();
    std::string fmt = sample_payload_format(sample);
    if (fmt.empty()) {
      fmt = read_detection_format(tensor);  // may pick up legacy tess-tagged BBOX
    }
    if (upper_copy(fmt) == "BBOX") {
      tag_detection_format(tensor, "BBOX");
    }
  } else if (sample.kind == simaai::neat::SampleKind::Bundle) {
    for (auto& field : sample.fields) {
      tag_detection_format_in_sample(field);
    }
  }
}

} // namespace simaai::neat

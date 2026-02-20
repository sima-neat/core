#include "pipeline/DetectionTypes.h"
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

  std::string fmt;
  if (tensor.semantic.tess.has_value()) {
    fmt = upper_copy(tensor.semantic.tess->format);
  }
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

} // namespace simaai::neat

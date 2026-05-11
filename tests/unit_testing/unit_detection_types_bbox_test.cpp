#include "pipeline/DetectionTypes.h"
#include "pipeline/TensorCore.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstdint>
#include <cstring>
#include <functional>
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

std::vector<uint8_t> make_bbox_payload(uint32_t count, const std::vector<RawBox>& boxes) {
  std::vector<uint8_t> out(sizeof(uint32_t) + boxes.size() * sizeof(RawBox), 0);
  std::memcpy(out.data(), &count, sizeof(uint32_t));
  if (!boxes.empty()) {
    std::memcpy(out.data() + sizeof(uint32_t), boxes.data(), boxes.size() * sizeof(RawBox));
  }
  return out;
}

simaai::neat::Tensor make_bbox_tensor(const std::vector<uint8_t>& payload,
                                      const std::string& format = "BBOX") {
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
  tensor.semantic.tess =
      TessSpec{.slice_shape = {}, .format = format};
  return tensor;
}

bool throws_with(const std::function<void()>& fn, const std::string& needle) {
  try {
    fn();
  } catch (const std::exception& e) {
    if (needle.empty())
      return true;
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

} // namespace

RUN_TEST("unit_detection_types_bbox_test", ([] {
           using namespace simaai::neat;

           const std::vector<RawBox> boxes = {
               RawBox{.x = 10, .y = 5, .w = 20, .h = 10, .score = 0.9f, .cls = 2},
               RawBox{.x = -3, .y = -4, .w = 5, .h = 6, .score = 0.7f, .cls = 1},
           };

           const auto payload = make_bbox_payload(2, boxes);
           const auto parsed = parse_bbox_bytes(payload, 64, 32, 2, true);
           require(parsed.size() == 2, "parse_bbox_bytes should decode two boxes");
           require(parsed[0].x1 == 10.0f && parsed[0].y1 == 5.0f, "first bbox top-left mismatch");
           require(parsed[0].x2 == 30.0f && parsed[0].y2 == 15.0f,
                   "first bbox bottom-right mismatch");

           // Second box should be clamped to image bounds.
           require(parsed[1].x1 == 0.0f && parsed[1].y1 == 0.0f,
                   "second bbox top-left should be clamped to zero");

           const auto malformed = std::vector<uint8_t>{1, 2, 3};
           require(parse_bbox_bytes(malformed, 64, 32, 1, false).empty(),
                   "non-strict parse should ignore malformed bbox buffers");
           require(throws_with([&]() { (void)parse_bbox_bytes(malformed, 64, 32, 1, true); },
                               "too small"),
                   "strict parse should reject malformed bbox buffers");

           const Tensor tensor = make_bbox_tensor(payload, "BBOX");
           const BoxDecodeResult decoded = decode_bbox_tensor(tensor, 64, 32, 2, true);
           require(decoded.boxes.size() == 2, "decode_bbox_tensor should return two boxes");
           require(decoded.raw.size() == payload.size(),
                   "decode_bbox_tensor raw payload size mismatch");

           require(throws_with(
                       [&]() {
                         Tensor wrong_fmt = make_bbox_tensor(payload, "FP32");
                         (void)decode_bbox_tensor(wrong_fmt, 64, 32, 2, true);
                       },
                       "format mismatch"),
                   "decode_bbox_tensor should reject non-BBOX tess formats");
         }));

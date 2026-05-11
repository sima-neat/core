#pragma once

#include "pipeline/TensorCore.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace sima_test::optiview {

struct RawBox {
  int32_t x = 0;
  int32_t y = 0;
  int32_t w = 0;
  int32_t h = 0;
  float score = 0.0f;
  int32_t cls = 0;
};

inline std::vector<uint8_t> make_bbox_payload(uint32_t count, const std::vector<RawBox>& boxes) {
  std::vector<uint8_t> out(sizeof(uint32_t) + boxes.size() * sizeof(RawBox), 0);
  std::memcpy(out.data(), &count, sizeof(uint32_t));
  if (!boxes.empty()) {
    std::memcpy(out.data() + sizeof(uint32_t), boxes.data(), boxes.size() * sizeof(RawBox));
  }
  return out;
}

inline simaai::neat::Sample make_bbox_tensor_sample(const std::vector<uint8_t>& payload,
                                                    const std::string& fmt = "BBOX") {
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
      TessSpec{.slice_shape = {}, .format = fmt};

  Sample sample = sample_from_tensors(TensorList{std::move(tensor)});
  sample.frame_id = 7;
  sample.stream_id = "stream0";
  sample.pts_ns = 123000000;
  return sample;
}

} // namespace sima_test::optiview

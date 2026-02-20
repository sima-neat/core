#include "pipeline/TensorCore.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstring>
#include <functional>
#include <string>

namespace {

simaai::neat::Tensor make_i420_tensor(int w, int h, uint8_t value = 0x11) {
  using namespace simaai::neat;
  const std::size_t y_size = static_cast<std::size_t>(w * h);
  const std::size_t u_size = static_cast<std::size_t>(w * h / 4);
  const std::size_t v_size = u_size;

  auto storage = make_cpu_owned_storage(y_size + u_size + v_size);
  auto map = storage->map(MapMode::Write);
  if (map.data && map.size_bytes > 0) {
    std::memset(map.data, value, map.size_bytes);
  }

  Tensor t;
  t.storage = storage;
  t.dtype = TensorDType::UInt8;
  t.layout = TensorLayout::HW;
  t.shape = {h, w};
  t.device = {DeviceType::CPU, 0};
  t.read_only = true;
  t.semantic.image = ImageSpec{ImageSpec::PixelFormat::I420, ""};

  Plane y;
  y.role = PlaneRole::Y;
  y.shape = {h, w};
  y.strides_bytes = {w, 1};
  y.byte_offset = 0;

  Plane u;
  u.role = PlaneRole::U;
  u.shape = {h / 2, w / 2};
  u.strides_bytes = {w / 2, 1};
  u.byte_offset = static_cast<int64_t>(y_size);

  Plane v;
  v.role = PlaneRole::V;
  v.shape = {h / 2, w / 2};
  v.strides_bytes = {w / 2, 1};
  v.byte_offset = static_cast<int64_t>(y_size + u_size);

  t.planes = {y, u, v};
  return t;
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

RUN_TEST("unit_tensor_image_mapping_test", ([] {
           using namespace simaai::neat;

           const Tensor nv12 = make_nv12_tensor(8, 6, 0x22);
           require(nv12.is_nv12(), "expected NV12 tensor");
           require(!nv12.is_i420(), "NV12 tensor should not report I420");

           auto nv12_mapped = nv12.map_nv12_read();
           require(nv12_mapped.has_value(),
                   "map_nv12_read should return mapped view for NV12 tensor");
           require(nv12_mapped->view.width == 8, "NV12 mapped width mismatch");
           require(nv12_mapped->view.height == 6, "NV12 mapped height mismatch");
           require(nv12_mapped->view.y != nullptr, "NV12 Y plane pointer missing");
           require(nv12_mapped->view.uv != nullptr, "NV12 UV plane pointer missing");

           const Tensor i420 = make_i420_tensor(8, 6, 0x33);
           require(i420.is_i420(), "expected I420 tensor");
           auto i420_mapped = i420.map_i420_read();
           require(i420_mapped.has_value(),
                   "map_i420_read should return mapped view for I420 tensor");
           require(i420_mapped->view.width == 8, "I420 mapped width mismatch");
           require(i420_mapped->view.height == 6, "I420 mapped height mismatch");
           require(i420_mapped->view.y != nullptr, "I420 Y plane pointer missing");
           require(i420_mapped->view.u != nullptr, "I420 U plane pointer missing");
           require(i420_mapped->view.v != nullptr, "I420 V plane pointer missing");

           const Tensor rgb = make_color_tensor(8, 6, ImageSpec::PixelFormat::RGB, 0x44);
           require(!rgb.map_nv12_read().has_value(), "RGB tensor should not map as NV12");
           require(!rgb.map_i420_read().has_value(), "RGB tensor should not map as I420");

           // Invalid NV12 geometry should throw.
           Tensor invalid_nv12 = make_nv12_tensor(5, 5, 0x55);
           require(throws_with([&]() { (void)invalid_nv12.map_nv12_read(); }, "even"),
                   "map_nv12_read should reject odd NV12 dimensions");
         }));

#include "pipeline/TensorCore.h"

#include "test_utils.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

int main() {
  try {
    auto storage = simaai::neat::make_cpu_owned_storage(16);
    require(storage && storage->size_bytes == 16, "storage size mismatch");

    simaai::neat::Tensor t;
    t.storage = storage;
    t.dtype = simaai::neat::TensorDType::UInt8;
    t.shape = {4, 4};
    t.strides_bytes = {4, 1};
    t.device = {simaai::neat::DeviceType::CPU, 0};
    t.read_only = false;

    require(t.is_dense(), "expected dense tensor");
    require(t.is_contiguous(), "expected contiguous tensor");

    {
      auto map = t.map(simaai::neat::MapMode::ReadWrite);
      require(map.data != nullptr, "map failed");
      std::memset(map.data, 0xAB, map.size_bytes);
    }

    simaai::neat::Tensor copy = t.clone();
    require(copy.is_contiguous(), "clone not contiguous");
    require(copy.storage && copy.storage->size_bytes == 16, "clone size mismatch");
    {
      auto map = copy.map(simaai::neat::MapMode::Read);
      require(map.data != nullptr, "clone map failed");
      const uint8_t* bytes = static_cast<const uint8_t*>(map.data);
      require(bytes[0] == 0xAB, "clone data mismatch");
    }

    simaai::neat::Tensor non;
    auto storage_nc = simaai::neat::make_cpu_owned_storage(8);
    {
      auto map = storage_nc->map(simaai::neat::MapMode::Write);
      require(map.data != nullptr, "non-contig map failed");
      uint8_t* buf = static_cast<uint8_t*>(map.data);
      buf[0] = 1;
      buf[1] = 2;
      buf[2] = 0;
      buf[3] = 0;
      buf[4] = 3;
      buf[5] = 4;
      buf[6] = 0;
      buf[7] = 0;
    }
    non.storage = storage_nc;
    non.dtype = simaai::neat::TensorDType::UInt8;
    non.shape = {2, 2};
    non.strides_bytes = {4, 1};
    non.device = {simaai::neat::DeviceType::CPU, 0};
    non.read_only = false;
    require(!non.is_contiguous(), "expected non-contiguous");
    {
      simaai::neat::Tensor non_copy = non.clone();
      require(non_copy.is_contiguous(), "non-contiguous clone not contiguous");
      auto map = non_copy.map(simaai::neat::MapMode::Read);
      require(map.data != nullptr, "non-contig clone map failed");
      const uint8_t* buf = static_cast<const uint8_t*>(map.data);
      require(buf[0] == 1 && buf[1] == 2 && buf[2] == 3 && buf[3] == 4,
              "non-contig clone data mismatch");
    }

    {
      const int w = 4;
      const int h = 2;
      auto storage_i420 = simaai::neat::make_cpu_owned_storage(12);
      auto map = storage_i420->map(simaai::neat::MapMode::Write);
      require(map.data != nullptr, "i420 map failed");
      uint8_t* buf = static_cast<uint8_t*>(map.data);
      for (int i = 0; i < 8; ++i)
        buf[i] = static_cast<uint8_t>(i);
      buf[8] = 100;
      buf[9] = 101;
      buf[10] = 200;
      buf[11] = 201;

      simaai::neat::Tensor i420;
      i420.storage = storage_i420;
      i420.dtype = simaai::neat::TensorDType::UInt8;
      i420.layout = simaai::neat::TensorLayout::HW;
      i420.shape = {h, w};
      i420.device = {simaai::neat::DeviceType::CPU, 0};
      i420.read_only = true;
      i420.semantic.image = simaai::neat::ImageSpec{simaai::neat::ImageSpec::PixelFormat::I420, ""};
      simaai::neat::Plane y;
      y.role = simaai::neat::PlaneRole::Y;
      y.shape = {h, w};
      y.strides_bytes = {w, 1};
      y.byte_offset = 0;
      simaai::neat::Plane u;
      u.role = simaai::neat::PlaneRole::U;
      u.shape = {h / 2, w / 2};
      u.strides_bytes = {w / 2, 1};
      u.byte_offset = 8;
      simaai::neat::Plane v;
      v.role = simaai::neat::PlaneRole::V;
      v.shape = {h / 2, w / 2};
      v.strides_bytes = {w / 2, 1};
      v.byte_offset = 10;
      i420.planes = {y, u, v};

      require(i420.i420_required_bytes() == 12, "i420 required bytes mismatch");
      auto mapped = i420.map_i420_read();
      require(mapped.has_value(), "i420 map_i420_read failed");
      require(mapped->view.width == w && mapped->view.height == h, "i420 dims mismatch");

      std::vector<uint8_t> out = i420.copy_i420_contiguous();
      require(out.size() == 12, "i420 copy size mismatch");
      require(out[0] == 0 && out[7] == 7, "i420 Y data mismatch");
      require(out[8] == 100 && out[9] == 101, "i420 U data mismatch");
      require(out[10] == 200 && out[11] == 201, "i420 V data mismatch");
      std::string err;
      require(i420.validate(&err), "i420 validate failed: " + err);
    }

    std::cout << "[OK] unit_tensortensor_core_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

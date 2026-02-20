#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/TensorCore.h"
#include "pipeline/SessionOptions.h"

#include "test_utils.h"

#include <cstring>
#include <iostream>
#include <vector>

namespace {

simaai::neat::Tensor make_rgb_tensor(int w, int h) {
  const std::size_t bytes = static_cast<std::size_t>(w * h * 3);
  auto storage = simaai::neat::make_cpu_owned_storage(bytes);
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes > 0) {
    std::memset(map.data, 0x11, map.size_bytes);
  }

  simaai::neat::Tensor t;
  t.storage = storage;
  t.dtype = simaai::neat::TensorDType::UInt8;
  t.layout = simaai::neat::TensorLayout::HWC;
  t.shape = {h, w, 3};
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  t.semantic.image = simaai::neat::ImageSpec{simaai::neat::ImageSpec::PixelFormat::RGB, ""};
  return t;
}

simaai::neat::Tensor make_gray_tensor(int w, int h) {
  const std::size_t bytes = static_cast<std::size_t>(w * h);
  auto storage = simaai::neat::make_cpu_owned_storage(bytes);
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes > 0) {
    std::memset(map.data, 0x22, map.size_bytes);
  }

  simaai::neat::Tensor t;
  t.storage = storage;
  t.dtype = simaai::neat::TensorDType::UInt8;
  t.layout = simaai::neat::TensorLayout::HW;
  t.shape = {h, w};
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  t.semantic.image = simaai::neat::ImageSpec{simaai::neat::ImageSpec::PixelFormat::GRAY8, ""};
  return t;
}

simaai::neat::Tensor make_i420_tensor(int w, int h) {
  const std::size_t y_size = static_cast<std::size_t>(w * h);
  const std::size_t uv_size = static_cast<std::size_t>(w * h / 4);
  auto storage = simaai::neat::make_cpu_owned_storage(y_size + uv_size * 2);
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes > 0) {
    std::memset(map.data, 0x44, map.size_bytes);
  }

  simaai::neat::Tensor t;
  t.storage = storage;
  t.dtype = simaai::neat::TensorDType::UInt8;
  t.layout = simaai::neat::TensorLayout::HW;
  t.shape = {h, w};
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  t.semantic.image = simaai::neat::ImageSpec{simaai::neat::ImageSpec::PixelFormat::I420, ""};

  simaai::neat::Plane y;
  y.role = simaai::neat::PlaneRole::Y;
  y.shape = {h, w};
  y.strides_bytes = {w, 1};
  y.byte_offset = 0;

  simaai::neat::Plane u;
  u.role = simaai::neat::PlaneRole::U;
  u.shape = {h / 2, w / 2};
  u.strides_bytes = {w / 2, 1};
  u.byte_offset = static_cast<int64_t>(y_size);

  simaai::neat::Plane v;
  v.role = simaai::neat::PlaneRole::V;
  v.shape = {h / 2, w / 2};
  v.strides_bytes = {w / 2, 1};
  v.byte_offset = static_cast<int64_t>(y_size + uv_size);

  t.planes = {y, u, v};
  return t;
}

simaai::neat::Tensor make_tensor_hwc(int w, int h, int c) {
  const std::size_t bytes = static_cast<std::size_t>(w * h * c) * 4;
  auto storage = simaai::neat::make_cpu_owned_storage(bytes);
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes > 0) {
    std::memset(map.data, 0x55, map.size_bytes);
  }

  simaai::neat::Tensor t;
  t.storage = storage;
  t.dtype = simaai::neat::TensorDType::Float32;
  t.layout = simaai::neat::TensorLayout::HWC;
  t.shape = {h, w, c};
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  return t;
}

simaai::neat::Tensor make_tensor_chw(int w, int h, int c) {
  const std::size_t bytes = static_cast<std::size_t>(w * h * c) * 4;
  auto storage = simaai::neat::make_cpu_owned_storage(bytes);
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes > 0) {
    std::memset(map.data, 0x66, map.size_bytes);
  }

  simaai::neat::Tensor t;
  t.storage = storage;
  t.dtype = simaai::neat::TensorDType::Float32;
  t.layout = simaai::neat::TensorLayout::CHW;
  t.shape = {c, h, w};
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  return t;
}

} // namespace

int main() {
  try {
    using namespace simaai::neat;

    {
      const int w = 8;
      const int h = 6;
      Sample s;
      s.kind = SampleKind::Tensor;
      s.media_type = "video/x-raw";
      s.tensor = make_rgb_tensor(w, h);

      SampleSpec spec = derive_sample_spec_or_throw(s);
      require(spec.kind == SampleMediaKind::RawVideo, "RGB spec kind mismatch");
      require(spec.format == "RGB", "RGB format mismatch");
      require(spec.width == w && spec.height == h, "RGB dims mismatch");
      require(spec.depth == 3, "RGB depth mismatch");
      require(spec.required_bytes_actual == static_cast<std::size_t>(w * h * 3),
              "RGB bytes mismatch");
      require(spec.planes.size() == 1, "RGB plane count mismatch");

      CapKey key = capkey_from_spec(spec);
      require(key == spec.caps_key, "CapKey mismatch for RGB spec");
      SampleSpec other = spec;
      other.width = w + 1;
      other.caps_key = capkey_from_spec(other);
      require(other.caps_key != spec.caps_key, "CapKey should differ on width change");
    }

    {
      const int w = 7;
      const int h = 5;
      Sample s;
      s.kind = SampleKind::Tensor;
      s.media_type = "video/x-raw";
      s.tensor = make_gray_tensor(w, h);

      SampleSpec spec = derive_sample_spec_or_throw(s);
      require(spec.kind == SampleMediaKind::RawVideo, "GRAY spec kind mismatch");
      require(spec.format == "GRAY8", "GRAY format mismatch");
      require(spec.width == w && spec.height == h, "GRAY dims mismatch");
      require(spec.depth == 1, "GRAY depth mismatch");
      require(spec.required_bytes_actual == static_cast<std::size_t>(w * h), "GRAY bytes mismatch");
      require(spec.planes.size() == 1, "GRAY plane count mismatch");
    }

    {
      const int w = 8;
      const int h = 4;
      Sample s;
      s.kind = SampleKind::Tensor;
      s.media_type = "video/x-raw";
      s.tensor = make_nv12_tensor(w, h, 0x33);

      SampleSpec spec = derive_sample_spec_or_throw(s);
      require(spec.kind == SampleMediaKind::RawVideo, "NV12 spec kind mismatch");
      require(spec.format == "NV12", "NV12 format mismatch");
      require(spec.width == w && spec.height == h, "NV12 dims mismatch");
      require(spec.planes.size() == 2, "NV12 plane count mismatch");
      require(spec.planes[0].offset_bytes == 0, "NV12 Y offset mismatch");
      require(spec.planes[1].offset_bytes == static_cast<int64_t>(w * h),
              "NV12 UV offset mismatch");
      require(spec.required_bytes_actual == static_cast<std::size_t>(w * h + (w * h) / 2),
              "NV12 bytes mismatch");
    }

    {
      const int w = 8;
      const int h = 4;
      Sample s;
      s.kind = SampleKind::Tensor;
      s.media_type = "video/x-raw";
      s.tensor = make_i420_tensor(w, h);

      SampleSpec spec = derive_sample_spec_or_throw(s);
      require(spec.kind == SampleMediaKind::RawVideo, "I420 spec kind mismatch");
      require(spec.format == "I420", "I420 format mismatch");
      require(spec.width == w && spec.height == h, "I420 dims mismatch");
      require(spec.planes.size() == 3, "I420 plane count mismatch");
      require(spec.planes[0].offset_bytes == 0, "I420 Y offset mismatch");
      require(spec.planes[1].offset_bytes == static_cast<int64_t>(w * h), "I420 U offset mismatch");
      require(spec.planes[2].offset_bytes == static_cast<int64_t>(w * h + (w * h) / 4),
              "I420 V offset mismatch");
      require(spec.required_bytes_actual == static_cast<std::size_t>(w * h + (w * h) / 2),
              "I420 bytes mismatch");
    }

    {
      const int w = 5;
      const int h = 4;
      const int c = 2;
      Sample s;
      s.kind = SampleKind::Tensor;
      s.media_type = "application/vnd.simaai.tensor";
      s.tensor = make_tensor_hwc(w, h, c);

      SampleSpec spec = derive_sample_spec_or_throw(s);
      require(spec.kind == SampleMediaKind::Tensor, "HWC spec kind mismatch");
      require(spec.format == "FP32", "HWC format mismatch");
      require(spec.width == w && spec.height == h && spec.depth == c, "HWC dims mismatch");
      require(spec.layout == TensorLayout::HWC, "HWC layout mismatch");
    }

    {
      const int w = 5;
      const int h = 4;
      const int c = 3;
      Sample s;
      s.kind = SampleKind::Tensor;
      s.media_type = "application/vnd.simaai.tensor";
      s.tensor = make_tensor_chw(w, h, c);

      SampleSpec spec = derive_sample_spec_or_throw(s);
      require(spec.kind == SampleMediaKind::Tensor, "CHW spec kind mismatch");
      require(spec.format == "FP32", "CHW format mismatch");
      require(spec.width == w && spec.height == h && spec.depth == c, "CHW dims mismatch");
      require(spec.layout == TensorLayout::CHW, "CHW layout mismatch");
    }

    {
      std::vector<uint8_t> bytes(16, 0xAB);
      Sample enc = make_encoded_sample(bytes, "video/x-h264");
      enc.media_type = "video/x-h264";
      SampleSpec spec = derive_sample_spec_or_throw(enc);
      require(spec.kind == SampleMediaKind::Encoded, "encoded spec kind mismatch");
      require(spec.caps_string == enc.caps_string, "encoded caps mismatch");
      require(spec.required_bytes_actual == bytes.size(), "encoded bytes mismatch");
    }

    {
      std::vector<uint8_t> bytes(8, 0xCD);
      Sample enc = make_encoded_sample(bytes, "video/x-h264");
      enc.caps_string.clear();
      bool threw = false;
      try {
        (void)derive_sample_spec_or_throw(enc);
      } catch (const std::exception&) {
        threw = true;
      }
      require(threw, "expected encoded sample caps_string error");
    }

    std::cout << "[OK] unit_samplespec_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

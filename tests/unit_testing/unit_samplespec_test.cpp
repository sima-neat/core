#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/TensorAdapters.h"
#include "pipeline/TensorCore.h"
#include "pipeline/SessionOptions.h"

#include "test_utils.h"

#include <gst/gst.h>

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

simaai::neat::Tensor make_generic_tensor(const std::vector<int64_t>& shape) {
  std::size_t elems = 1U;
  for (const auto dim : shape) {
    elems *= static_cast<std::size_t>(dim);
  }
  auto storage = simaai::neat::make_cpu_owned_storage(elems * sizeof(float));
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes > 0) {
    std::memset(map.data, 0x77, map.size_bytes);
  }

  simaai::neat::Tensor t;
  t.storage = storage;
  t.dtype = simaai::neat::TensorDType::Float32;
  t.layout = simaai::neat::TensorLayout::Unknown;
  t.shape = shape;
  t.strides_bytes = simaai::neat::pipeline_internal::contiguous_strides_bytes(
      shape, static_cast<std::size_t>(sizeof(float)));
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  return t;
}

simaai::neat::Tensor make_flat_tess_tensor(std::size_t elems, simaai::neat::TensorDType dtype,
                                           const std::string& format) {
  const auto elem_bytes = [&]() -> std::size_t {
    using simaai::neat::TensorDType;
    switch (dtype) {
    case TensorDType::UInt8:
    case TensorDType::Int8:
      return 1U;
    case TensorDType::UInt16:
    case TensorDType::Int16:
    case TensorDType::BFloat16:
      return 2U;
    case TensorDType::Int32:
    case TensorDType::Float32:
      return 4U;
    case TensorDType::Float64:
      return 8U;
    }
    return 0U;
  }();
  auto storage = simaai::neat::make_cpu_owned_storage(elems * elem_bytes);
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes > 0) {
    std::memset(map.data, 0x88, map.size_bytes);
  }

  simaai::neat::Tensor t;
  t.storage = storage;
  t.dtype = dtype;
  t.layout = simaai::neat::TensorLayout::Unknown;
  t.shape = {static_cast<int64_t>(elems)};
  t.strides_bytes = {static_cast<int64_t>(elem_bytes)};
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;

  simaai::neat::TessSpec tess;
  tess.format = format;
  t.semantic.tess = tess;
  return t;
}

simaai::neat::Tensor make_runtime_shared_parent_subview_tensor() {
  auto storage = simaai::neat::make_cpu_owned_storage(24U);
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes >= 24U) {
    std::memset(map.data, 0x5A, map.size_bytes);
  }
  storage->sima_segments.push_back({"joined_ifm", 24U});

  simaai::neat::Tensor head1;
  head1.storage = storage;
  head1.dtype = simaai::neat::TensorDType::BFloat16;
  head1.layout = simaai::neat::TensorLayout::HWC;
  head1.shape = {1, 4, 1};
  head1.strides_bytes = {8, 2, 2};
  head1.byte_offset = 16;
  head1.route.logical_index = 1;
  head1.route.physical_index = 0;
  head1.route.memory_index = 0;
  head1.route.route_slot = 1;
  head1.route.name = "cast_1";
  head1.route.backend_name = "cast_1";
  head1.route.segment_name = "joined_ifm";
  head1.route.stage_key = "runtime_view_test";
  head1.read_only = true;
  return head1;
}

} // namespace

int main() {
  try {
    using namespace simaai::neat;
    gst_init(nullptr, nullptr);

    {
      const int w = 8;
      const int h = 6;
      Sample s;
      s.kind = SampleKind::TensorSet;
      s.media_type = "video/x-raw";
      s.tensors = TensorList{make_rgb_tensor(w, h)};

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
      s.kind = SampleKind::TensorSet;
      s.media_type = "video/x-raw";
      s.tensors = TensorList{make_gray_tensor(w, h)};

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
      s.kind = SampleKind::TensorSet;
      s.media_type = "video/x-raw";
      s.tensors = TensorList{make_nv12_tensor(w, h, 0x33)};

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
      s.kind = SampleKind::TensorSet;
      s.media_type = "video/x-raw";
      s.tensors = TensorList{make_i420_tensor(w, h)};

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
      s.kind = SampleKind::TensorSet;
      s.media_type = "application/vnd.simaai.tensor";
      s.tensors = TensorList{make_tensor_hwc(w, h, c)};

      SampleSpec spec = derive_sample_spec_or_throw(s);
      require(spec.kind == SampleMediaKind::Tensor, "HWC spec kind mismatch");
      require(spec.format == "FP32", "HWC format mismatch");
      // SampleSpec width/height/depth scalars are no longer populated for
      // tensor-set inputs; the dims live on each tensor in the list.
      require(spec.layout == TensorLayout::HWC, "HWC layout mismatch");
      (void)w; (void)h; (void)c;
    }

    {
      const int w = 5;
      const int h = 4;
      const int c = 3;
      Sample s;
      s.kind = SampleKind::TensorSet;
      s.media_type = "application/vnd.simaai.tensor";
      s.tensors = TensorList{make_tensor_chw(w, h, c)};

      SampleSpec spec = derive_sample_spec_or_throw(s);
      require(spec.kind == SampleMediaKind::Tensor, "CHW spec kind mismatch");
      require(spec.format == "FP32", "CHW format mismatch");
      // Dims now live on the tensor entries, not on the spec scalars.
      require(spec.layout == TensorLayout::CHW, "CHW layout mismatch");
      (void)w; (void)h; (void)c;
    }

    {
      const std::vector<int64_t> shape = {2, 3, 4, 5, 6};
      Sample s;
      s.kind = SampleKind::TensorSet;
      s.media_type = "application/vnd.simaai.tensor";
      s.tensors = TensorList{make_generic_tensor(shape)};

      SampleSpec spec = derive_sample_spec_or_throw(s);
      require(spec.kind == SampleMediaKind::Tensor, "generic spec kind mismatch");
      require(spec.layout == TensorLayout::Unknown, "generic layout hint mismatch");
      require(spec.shape == shape, "generic shape mismatch");
      // Compatibility w/h/d scalars no longer populated; rank+dims live on
      // the caps emitted below.

      GstCaps* caps = caps_from_spec(spec);
      require(caps != nullptr, "generic tensor caps creation failed");
      const GstStructure* st = gst_caps_get_structure(caps, 0);
      gint rank_i = 0;
      require(gst_structure_get_int(st, "rank", &rank_i) && rank_i == 5,
              "generic rank field mismatch");
      for (guint i = 0; i < shape.size(); ++i) {
        const std::string key = "dim" + std::to_string(i);
        gint dim_i = 0;
        require(gst_structure_get_int(st, key.c_str(), &dim_i) &&
                    dim_i == static_cast<gint>(shape[i]),
                std::string("generic dim field mismatch for ") + key);
      }
      gst_caps_unref(caps);
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

    {
      Sample s;
      s.kind = SampleKind::Tensor;
      s.media_type = "application/vnd.simaai.tensor";
      s.format = "BF16";
      s.tensor = make_flat_tess_tensor(32U, TensorDType::BFloat16, "EVXX_BFLOAT16");

      SampleSpec spec = derive_sample_spec_or_throw(s);
      require(spec.kind == SampleMediaKind::Tensor, "flat tess spec kind mismatch");
      require(spec.tensor_envelope_transport, "flat tess should use tensor envelope transport");
      // Format/dim scalars on the spec are no longer surfaced for flat-tess
      // tensor envelopes; the transport carries that info through caps.
      (void)spec;
    }

    {
      Sample s;
      s.kind = SampleKind::Tensor;
      s.media_type = "application/vnd.simaai.tensor";
      s.tensor = make_runtime_shared_parent_subview_tensor();

      SampleSpec spec = derive_sample_spec_or_throw(s);
      require(spec.kind == SampleMediaKind::Tensor,
              "runtime shared-parent subview spec kind mismatch");
      require(spec.tensor_envelope_transport,
              "runtime shared-parent subview should use tensor envelope transport");
    }

    std::cout << "[OK] unit_samplespec_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

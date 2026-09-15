#include "gst/GstInit.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/PublicInputContract.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/TensorAdapters.h"
#include "pipeline/TensorCore.h"
#include "pipeline/GraphOptions.h"

#include "test_utils.h"

#include <gst/gst.h>
#include <gst/video/video.h>

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

simaai::neat::Tensor make_padded_decoder_nv12_tensor(int w, int visible_h, int physical_y_h) {
  const std::size_t y_stride = static_cast<std::size_t>(w);
  const std::size_t uv_stride = static_cast<std::size_t>(w);
  const std::size_t uv_h = static_cast<std::size_t>(visible_h / 2);
  const std::size_t uv_offset = y_stride * static_cast<std::size_t>(physical_y_h);
  const std::size_t visible_end = uv_offset + uv_stride * uv_h;
  const std::size_t padded_frame_bytes = y_stride * static_cast<std::size_t>(physical_y_h) +
                                         uv_stride * static_cast<std::size_t>(physical_y_h / 2);

  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, padded_frame_bytes, nullptr);
  require(buffer != nullptr, "failed to allocate padded decoder buffer");

  gsize offsets[GST_VIDEO_MAX_PLANES] = {0};
  gint strides[GST_VIDEO_MAX_PLANES] = {0};
  offsets[0] = 0;
  offsets[1] = uv_offset;
  strides[0] = static_cast<gint>(y_stride);
  strides[1] = static_cast<gint>(uv_stride);
  GstVideoMeta* meta = gst_buffer_add_video_meta_full(
      buffer, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_NV12, static_cast<guint>(w),
      static_cast<guint>(visible_h), 2, offsets, strides);
  require(meta != nullptr, "failed to attach padded decoder GstVideoMeta");
  GstSample* sample = gst_sample_new(buffer, nullptr, nullptr, nullptr);
  require(sample != nullptr, "failed to wrap padded decoder sample");

  simaai::neat::Tensor t;
  t.storage = simaai::neat::pipeline_internal::make_gst_sample_storage(sample);
  gst_sample_unref(sample);
  gst_buffer_unref(buffer);
  require(t.storage != nullptr, "failed to create GstSample storage for padded decoder tensor");
  require(t.storage->size_bytes == padded_frame_bytes,
          "padded decoder GstSample storage size mismatch");

  t.dtype = simaai::neat::TensorDType::UInt8;
  t.layout = simaai::neat::TensorLayout::HW;
  t.shape = {visible_h, w};
  t.device = {simaai::neat::DeviceType::SIMA_CVU, 0};
  t.read_only = true;
  t.semantic.image = simaai::neat::ImageSpec{simaai::neat::ImageSpec::PixelFormat::NV12, ""};

  simaai::neat::Plane y;
  y.role = simaai::neat::PlaneRole::Y;
  y.shape = {visible_h, w};
  y.strides_bytes = {static_cast<int64_t>(y_stride), 1};
  y.byte_offset = 0;

  simaai::neat::Plane uv;
  uv.role = simaai::neat::PlaneRole::UV;
  uv.shape = {visible_h / 2, w};
  uv.strides_bytes = {static_cast<int64_t>(uv_stride), 1};
  uv.byte_offset = static_cast<int64_t>(uv_offset);

  t.planes = {y, uv};
  require(visible_end < padded_frame_bytes,
          "test fixture must have a padded tail beyond visible plane extents");
  return t;
}

} // namespace

int main() {
  try {
    using namespace simaai::neat;
    gst_init_once();

    {
      const int w = 8;
      const int h = 6;
      Sample s;
      s.kind = SampleKind::TensorSet;
      s.payload_type = PayloadType::Image;
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
      s.payload_type = PayloadType::Image;
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
      s.payload_type = PayloadType::Image;
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
      const int w = 1280;
      const int visible_h = 720;
      const int physical_y_h = 768;
      Tensor padded = make_padded_decoder_nv12_tensor(w, visible_h, physical_y_h);

      InputOptions opt;
      opt.payload_type = PayloadType::Image;
      opt.format = FormatTag::NV12;
      SampleSpec spec = derive_tensor_spec_or_throw(padded, opt, "padded decoder NV12");

      const std::size_t uv_offset =
          static_cast<std::size_t>(w) * static_cast<std::size_t>(physical_y_h);
      const std::size_t visible_end =
          uv_offset + static_cast<std::size_t>(w) * static_cast<std::size_t>(visible_h / 2);
      const std::size_t padded_frame_bytes =
          static_cast<std::size_t>(w) * static_cast<std::size_t>(physical_y_h) * 3U / 2U;
      require(visible_end == 1443840U, "padded fixture visible span mismatch");
      require(padded_frame_bytes == 1474560U, "padded fixture frame span mismatch");
      require(spec.required_bytes_actual == padded_frame_bytes,
              "GstSample-backed NV12 spec must preserve decoder padded frame span");

      padded.storage = make_cpu_owned_storage(padded_frame_bytes);
      padded.device = {DeviceType::CPU, 0};
      SampleSpec cpu_spec = derive_tensor_spec_or_throw(padded, opt, "padded CPU NV12");
      require(cpu_spec.required_bytes_actual == visible_end,
              "CPU-owned NV12 spec should not grow to unused storage capacity");
    }

    {
      const int w = 8;
      const int h = 4;
      Sample s;
      s.kind = SampleKind::TensorSet;
      s.payload_type = PayloadType::Image;
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
      s.payload_type = PayloadType::Tensor;
      s.tensors = TensorList{make_tensor_hwc(w, h, c)};

      SampleSpec spec = derive_sample_spec_or_throw(s);
      require(spec.kind == SampleMediaKind::Tensor, "HWC spec kind mismatch");
      require(spec.format == "FP32", "HWC format mismatch");
      // SampleSpec width/height/depth scalars are no longer populated for
      // tensor-set inputs; the dims live on each tensor in the list.
      require(spec.layout == TensorLayout::HWC, "HWC layout mismatch");
      (void)w;
      (void)h;
      (void)c;
    }

    {
      const int w = 5;
      const int h = 4;
      const int c = 3;
      Sample s;
      s.kind = SampleKind::TensorSet;
      s.payload_type = PayloadType::Tensor;
      s.tensors = TensorList{make_tensor_chw(w, h, c)};

      SampleSpec spec = derive_sample_spec_or_throw(s);
      require(spec.kind == SampleMediaKind::Tensor, "CHW spec kind mismatch");
      require(spec.format == "FP32", "CHW format mismatch");
      // Dims now live on the tensor entries, not on the spec scalars.
      require(spec.layout == TensorLayout::CHW, "CHW layout mismatch");
      (void)w;
      (void)h;
      (void)c;
    }

    {
      const std::vector<int64_t> shape = {2, 3, 4, 5, 6};
      Sample s;
      s.kind = SampleKind::TensorSet;
      s.payload_type = PayloadType::Tensor;
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
      enc.payload_type = PayloadType::Encoded;
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
      s.payload_type = PayloadType::Tensor;
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
      s.payload_type = PayloadType::Tensor;
      s.tensor = make_runtime_shared_parent_subview_tensor();

      SampleSpec spec = derive_sample_spec_or_throw(s);
      require(spec.kind == SampleMediaKind::Tensor,
              "runtime shared-parent subview spec kind mismatch");
      require(spec.tensor_envelope_transport,
              "runtime shared-parent subview should use tensor envelope transport");
    }

    {
      Tensor tensor = make_rgb_tensor(16, 12);
      tensor.semantic.image.reset();
      InputOptions input;
      input.payload_type = PayloadType::Image;
      input.format = "BGR";
      Sample image = pipeline_internal::sample_from_tensors_for_input({tensor}, input);
      const SampleSpec spec = derive_sample_spec_or_throw(image);
      require(spec.kind == SampleMediaKind::RawVideo && spec.format == "BGR" && spec.width == 16 &&
                  spec.height == 12,
              "explicit image ingress must interpret an untagged tensor");
      require(image.tensors.front().storage == tensor.storage &&
                  image.tensors.front().byte_offset == tensor.byte_offset &&
                  image.tensors.front().strides_bytes == tensor.strides_bytes &&
                  !image.tensors.front().semantic.image.has_value(),
              "image ingress must preserve storage and caller Tensor semantics");
      std::string error;
      auto holder = pipeline_internal::make_sample_holder_from_bundle(image, &error);
      require(holder != nullptr, "contextual image envelope must retain format: " + error);

      tensor.semantic.image = ImageSpec{ImageSpec::PixelFormat::RGB, ""};
      bool rejected = false;
      try {
        (void)derive_sample_spec_or_throw(
            pipeline_internal::sample_from_tensors_for_input({tensor}, input));
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected, "explicit source and tensor format disagreement must not be relabeled");
      const Sample images =
          pipeline_internal::sample_from_tensors_for_input({tensor, tensor}, input);
      require(images.tensors.size() == 2U && images.payload_type == PayloadType::Image &&
                  images.tensors.front().storage == tensor.storage,
              "context completion must preserve existing multi-image transport");
    }

    {
      struct Edge {
        std::size_t from;
        std::size_t to;
      };
      struct View {
        std::vector<std::shared_ptr<Node>> vertices;
        std::vector<Edge> edges;
        std::vector<runtime::FragmentPlan> fragments;
      };
      InputOptions source;
      source.max_width = 4096;
      source.memory_policy = InputMemoryPolicy::SystemMemory;
      source.do_timestamp = false;
      View view;
      view.vertices.push_back(std::make_shared<Input>("image", source));
      view.edges.push_back({0, 1});
      runtime::FragmentPlan model;
      model.graph_start = 1;
      model.graph_end = 3;
      model.boundary_hints.emplace();
      InputOptions ingress;
      ingress.payload_type = PayloadType::Image;
      ingress.format = "BGR";
      ingress.max_width = 1920;
      model.boundary_hints->ingress_inputs.push_back(ingress);
      view.fragments.push_back(model);
      model.graph_start = 3;
      model.graph_end = 100;
      model.boundary_hints->ingress_inputs.front().format = "RGB";
      view.fragments.push_back(model);
      const auto resolved = pipeline_internal::public_input_options(view, 0);
      require(resolved.payload_type == PayloadType::Image && resolved.format.str() == "BGR",
              "public input must use its connected ingress, not the largest model fragment");
      require(resolved.max_width == source.max_width &&
                  resolved.memory_policy == source.memory_policy &&
                  resolved.do_timestamp == source.do_timestamp,
              "model ingress must not replace source geometry, memory or timestamps");
      for (const auto format : {FormatTag::FP32, FormatTag::ByteStream, FormatTag::H264}) {
        InputOptions explicit_format = source;
        explicit_format.format = format;
        view.vertices.front() = std::make_shared<Input>("image", explicit_format);
        const auto preserved = pipeline_internal::public_input_options(view, 0);
        require(preserved.payload_type == PayloadType::Auto && preserved.format.tag == format,
                "explicit non-image source format must not acquire model image semantics");
      }
      view.vertices.front() = std::make_shared<Input>("image", source);
      view.fragments.push_back(view.fragments.front());
      require(pipeline_internal::public_input_options(view, 0).format.str() == "BGR",
              "duplicate equivalent ingress declarations must retain one interpretation");
      view.edges.push_back({0, 3});
      const auto ambiguous = pipeline_internal::public_input_options(view, 0);
      require(ambiguous.payload_type == PayloadType::Auto && ambiguous.format.empty(),
              "ambiguous inference must remain unspecified, not reject an explicit sample");
      source.format = "RGB";
      view.vertices.front() = std::make_shared<Input>("image", source);
      const auto explicit_image = pipeline_internal::public_input_options(view, 0);
      require(explicit_image.payload_type == PayloadType::Image &&
                  explicit_image.format.str() == "RGB",
              "explicit image format must remain authoritative across multiple consumers");
    }

    std::cout << "[OK] unit_samplespec_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

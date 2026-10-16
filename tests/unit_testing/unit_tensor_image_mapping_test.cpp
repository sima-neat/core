#include "pipeline/TensorCore.h"
#include "pipeline/TensorAdapters.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/HolderLoanGate.h"
#include "pipeline/internal/TensorUtil.h"
#include "dmabuf_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <cstring>
#include <memory>
#include <functional>
#include <string>
#include <limits>
#include <array>

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

void test_dmabuf_logical_read_range(bool shared_memory) {
  using namespace simaai::neat;
  GstBuffer* allocation = sima_test::allocate_cma_dmabuf(512U);
  GstBuffer* buffer = allocation;
  if (shared_memory) {
    buffer = gst_buffer_new();
    auto* memory = gst_memory_share(gst_buffer_peek_memory(allocation, 0U), 64, 256);
    require(memory != nullptr, "shared DMA fixture range missing");
    gst_buffer_append_memory(buffer, memory);
  }
  GstCaps* caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "GRAY8", "width",
                                    G_TYPE_INT, 8, "height", G_TYPE_INT, 8, nullptr);
  GstSample* sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  gst_caps_unref(caps);
  if (shared_memory) gst_buffer_unref(buffer);
  gst_buffer_unref(allocation);
  Tensor tensor = from_gst_sample(sample);
  gst_sample_unref(sample);
  tensor.byte_offset = 7;
  tensor.shape = {3, 5};
  tensor.strides_bytes = {12, 1};
  const auto original_span = sima_test::dmabuf_span(
      gst_sample_get_buffer(static_cast<GstSample*>(tensor.storage->holder.get())));
  {
    auto whole = tensor.storage->map(MapMode::Write);
    require(whole.data && whole.size_bytes == (shared_memory ? 256U : 512U),
            "whole-storage mapping changed size");
    for (std::size_t i = 0; i < whole.size_bytes; ++i)
      static_cast<uint8_t*>(whole.data)[i] = static_cast<uint8_t>(i);
  }
  {
    auto logical = pipeline_internal::map_dense_tensor_read(tensor);
    require(logical.data && logical.size_bytes == 29U,
            "dense DMA READ did not use the exact checked physical span");
    require(static_cast<const uint8_t*>(logical.data)[0] == 7U &&
                static_cast<const uint8_t*>(logical.data)[28] == 35U,
            "nested tensor/GstMemory offsets were not composed exactly once");
    require(!pipeline_internal::map_dense_tensor_read(tensor).data,
            "same mapper accepted an overlapping active mapping");
  }
  // Public map and view preserve the historically accessible storage tail.
  for (bool view : {false, true}) {
    auto mapped = view ? tensor.view_read() : tensor.map_read();
    require(mapped.data && mapped.size_bytes == tensor.storage->size_bytes - 7U,
            "public DMA map/view compatibility extent changed");
    require(static_cast<const uint8_t*>(mapped.data)[0] == 7U,
            "public DMA mapping applied its offset incorrectly");
  }
  tensor.shape = {2, 6};
  tensor.strides_bytes = {20, 1};
  {
    auto dynamic = pipeline_internal::map_dense_tensor_read(tensor);
    require(dynamic.data && dynamic.size_bytes == 26U,
            "current tensor geometry did not determine the new READ extent");
  }
  tensor.byte_offset = static_cast<int64_t>(tensor.storage->size_bytes - 2U);
  require(!pipeline_internal::map_dense_tensor_read(tensor).data,
          "out-of-bounds logical tensor range was partially mapped");
  tensor.byte_offset = 7;
  {
    auto whole = tensor.storage->map(MapMode::Read);
    require(whole.data && whole.size_bytes == tensor.storage->size_bytes,
            "explicit whole-storage READ lost its original semantics");
  }
  // Unresolved/unsupported stride geometry must not fabricate a narrow
  // permission range. Preserve the established conservative storage mapping.
  tensor.strides_bytes = {-20, 1};
  require(!pipeline_internal::map_dense_tensor_read(tensor).data,
          "logical READ accepted negative strides");
  {
    auto unresolved = pipeline_internal::map_tensor_view(tensor, MapMode::Read);
    require(unresolved.data && unresolved.size_bytes == tensor.storage->size_bytes - 7U,
            "negative strides were treated as a qualified dense range");
  }
  tensor.strides_bytes = {1};
  require(!pipeline_internal::map_dense_tensor_read(tensor).data,
          "logical READ accepted incomplete strides");
  {
    auto unresolved = pipeline_internal::map_tensor_view(tensor, MapMode::Read);
    require(unresolved.data && unresolved.size_bytes == tensor.storage->size_bytes - 7U,
            "incomplete strides were treated as a qualified dense range");
  }
  tensor.strides_bytes = {20, 1};
  // An application-owned wrapper is not bypassed or presumed range-aware.
  auto original = std::move(tensor.storage->map_fn);
  unsigned callback_calls = 0;
  tensor.storage->map_fn = [original = std::move(original), &callback_calls](MapMode mode) {
    ++callback_calls;
    return original(mode);
  };
  {
    auto custom = tensor.map_read();
    require(callback_calls == 1U && custom.data &&
                custom.size_bytes == tensor.storage->size_bytes - 7U,
            "custom storage callback authority changed");
  }
  {
    auto custom = pipeline_internal::map_dense_tensor_read(tensor);
    require(callback_calls == 2U && custom.data && custom.size_bytes == 26U,
            "logical READ bypassed a custom callback or exposed the backing tail");
    require(static_cast<const uint8_t*>(custom.data)[0] == 7U &&
                static_cast<const uint8_t*>(custom.data)[25] == 32U,
            "custom logical READ offset/span changed");
  }
  require(sima_test::dmabuf_span(
              gst_sample_get_buffer(static_cast<GstSample*>(tensor.storage->holder.get()))) ==
              original_span,
          "exact logical READ replaced or copied its producer carrier");
}

void test_custom_logical_read_contract() {
  using namespace simaai::neat;
  std::array<uint8_t, 64> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<uint8_t>(i);
  unsigned calls = 0, ends = 0;
  std::size_t exposed = bytes.size();
  auto storage = std::make_shared<Storage>();
  storage->size_bytes = bytes.size();
  storage->map_fn = [&](MapMode mode) {
    require(mode == MapMode::Read, "logical helper requested writable access");
    ++calls;
    Mapping result;
    result.data = exposed ? bytes.data() : nullptr;
    result.size_bytes = exposed;
    result.unmap = [&] { ++ends; };
    return result;
  };
  Tensor tensor;
  tensor.storage = storage;
  tensor.dtype = TensorDType::UInt8;
  tensor.shape = {2, 3};
  tensor.strides_bytes = {8, 1};
  tensor.byte_offset = 5;
  {
    auto mapped = pipeline_internal::map_dense_tensor_read(tensor);
    require(mapped.data == bytes.data() + 5 && mapped.size_bytes == 11U && calls == 1U,
            "custom logical READ copied storage or changed its checked span");
    require(static_cast<const uint8_t*>(mapped.data)[10] == 15U,
            "custom logical READ did not preserve reachable strided data");
    require(ends == 0U, "logical helper ended its mapping before the reader");
  }
  require(ends == 1U, "logical helper did not end the accepted custom mapping");
  exposed = 15; // Offset5 plus span11 is one byte larger than the mapper grants.
  require(!pipeline_internal::map_dense_tensor_read(tensor).data && calls == 2U && ends == 2U,
          "short callback mapping was accepted or its cleanup was lost");
  exposed = 0;
  require(!pipeline_internal::map_dense_tensor_read(tensor).data && calls == 3U && ends == 3U,
          "failed callback extent was accepted or cleanup was lost");
  tensor.strides_bytes = {std::numeric_limits<int64_t>::max(), 1};
  tensor.shape = {std::numeric_limits<int64_t>::max(), 3};
  require(!pipeline_internal::map_dense_tensor_read(tensor).data && calls == 3U,
          "overflowing geometry reached the callback");
  tensor.shape = {2, 3};
  tensor.strides_bytes = {8, 1};
  tensor.byte_offset = -1;
  require(!pipeline_internal::map_dense_tensor_read(tensor).data && calls == 3U,
          "negative offset reached the callback");
}

// Composite images retain the complete backing view: narrowing to shape {H,W}
// alone would exclude chroma planes even though the image reader accesses them.
void test_dmabuf_composite_read(bool nv12) {
  using namespace simaai::neat;
  GstBuffer* buffer = sima_test::allocate_cma_dmabuf(512U);
  const auto backing = sima_test::dmabuf_span(buffer);
  GstCaps* caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING,
                                    nv12 ? "NV12" : "I420", "width", G_TYPE_INT, 8,
                                    "height", G_TYPE_INT, 6, nullptr);
  GstSample* sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  gst_caps_unref(caps);
  gst_buffer_unref(buffer);
  Tensor tensor = from_gst_sample(sample);
  gst_sample_unref(sample);
  require(tensor.is_composite(), "DMA image fixture lost its composite plane records");
  require(nv12 ? tensor.is_nv12() : tensor.is_i420(), "DMA image format changed");
  {
    auto write = tensor.storage->map(MapMode::Write);
    require(write.data && write.size_bytes == 512U, "DMA image initialization map failed");
    for (std::size_t i = 0; i < write.size_bytes; ++i)
      static_cast<uint8_t*>(write.data)[i] = static_cast<uint8_t>(i % 251U);
  }
  auto check_plane = [&](const Mapping& mapping, const uint8_t* bytes, PlaneRole role) {
    const Plane* plane = tensor.try_plane(role);
    require(plane && plane->shape.size() == 2U && plane->strides_bytes.size() == 2U,
            "DMA image plane geometry is incomplete");
    const auto offset = static_cast<std::size_t>(plane->byte_offset);
    const auto last = offset + static_cast<std::size_t>(plane->shape[0] - 1) *
                                   static_cast<std::size_t>(plane->strides_bytes[0]) +
                     static_cast<std::size_t>(plane->shape[1] - 1);
    require(bytes == static_cast<const uint8_t*>(mapping.data) + offset,
            "DMA plane reader copied storage or changed its byte offset");
    require(last < mapping.size_bytes && bytes[0] == offset % 251U &&
                bytes[last - offset] == last % 251U,
            "DMA image plane reader lost first/last pixel values");
  };
  auto check_read = [&] {
    if (nv12) {
      auto mapped = tensor.map_nv12_read();
      require(mapped && mapped->mapping.size_bytes == 512U,
              "NV12 READ was incorrectly narrowed to its luma shape");
      check_plane(mapped->mapping, mapped->view.y, PlaneRole::Y);
      check_plane(mapped->mapping, mapped->view.uv, PlaneRole::UV);
    } else {
      auto mapped = tensor.map_i420_read();
      require(mapped && mapped->mapping.size_bytes == 512U,
              "I420 READ was incorrectly narrowed to its luma shape");
      check_plane(mapped->mapping, mapped->view.y, PlaneRole::Y);
      check_plane(mapped->mapping, mapped->view.u, PlaneRole::U);
      check_plane(mapped->mapping, mapped->view.v, PlaneRole::V);
    }
  };
  for (unsigned i = 0; i < 3U; ++i) check_read();
  auto original = std::move(tensor.storage->map_fn);
  unsigned callback_calls = 0;
  tensor.storage->map_fn = [original = std::move(original), &callback_calls](MapMode mode) {
    ++callback_calls;
    return original(mode);
  };
  for (unsigned i = 0; i < 3U; ++i) check_read();
  require(callback_calls == 3U, "composite DMA READ bypassed the custom storage callback");
  require(sima_test::dmabuf_span(
              gst_sample_get_buffer(static_cast<GstSample*>(tensor.storage->holder.get()))) ==
              backing,
          "composite DMA READ replaced its producer carrier");
}

void test_dmabuf_mapping_lifetime(bool projected, bool logical_read) {
  using namespace simaai::neat;
  namespace internal = pipeline_internal;
  GstBuffer* buffer = sima_test::allocate_cma_dmabuf(128U);
  const auto backing = sima_test::dmabuf_span(buffer);
  GstCaps* caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "GRAY8", "width",
                                      G_TYPE_INT, 8, "height", G_TYPE_INT, 8, nullptr);
  GstSample* gst_sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  gst_caps_unref(caps);
  gst_buffer_unref(buffer);
  Tensor tensor = from_gst_sample(gst_sample);
  gst_sample_unref(gst_sample);
  if (projected) {
    tensor = internal::tensor_view_from_sample_memory(tensor, 0, true);
    tensor.byte_offset = 16;
    tensor.shape = {4, 8};
    tensor.strides_bytes = {8, 1};
  }
  require(internal::tensor_has_dmabuf_memory(tensor), "DMA storage classification missing");
  require(tensor.device.type == DeviceType::CPU && tensor.storage->sima_mem_target_flags == 0U,
          "shared DMA storage must not acquire fake device placement or legacy flags");
  {
    // Fixture initialization uses the same synchronized mapping authority as app reads.
    auto write = tensor.storage->map(MapMode::Write);
    require(write.data && write.size_bytes >= 64U, "real DMA-BUF writable mapping failed");
    std::memset(write.data, 0x5A, write.size_bytes);
  }
  require(throws_with([&] { (void)tensor.map(MapMode::Write); }, "read-only"),
          "read-only DMA Tensor accepted writable app access");

  struct FixtureState {
    bool end_completed = false;
    bool producer_released = false;
    bool end_before_release = false;
  };
  auto state = std::make_shared<FixtureState>();
  auto original_map = std::move(tensor.storage->map_fn);
  tensor.storage->map_fn = [original_map = std::move(original_map), state](MapMode mode) {
    Mapping mapped = original_map(mode);
    mapped.unmap = [finish = std::move(mapped.unmap), state]() {
      if (finish) {
        finish();
      }
      state->end_completed = true;
    };
    return mapped;
  };
  auto producer = std::shared_ptr<void>(new int(0), [state](void* value) {
    state->end_before_release = state->end_completed;
    state->producer_released = true;
    delete static_cast<int*>(value);
  });
  auto gate = std::make_shared<internal::HolderLoanGate>(1);
  Sample sample = sample_from_tensors(TensorList{tensor});
  internal::mark_sample_producer_stream_lifetime(sample, producer);
  require(internal::attach_zero_copy_loan_to_sample(sample, gate), "DMA loan attachment failed");
  producer.reset();
  require(gate->inflight() == 1, "DMA map fixture did not acquire one output loan");
  Mapping mapping = logical_read ? internal::map_dense_tensor_read(tensor) : tensor.map_read();
  require(mapping.data && mapping.size_bytes >= 32U, "real DMA-BUF read mapping failed");
  require(static_cast<const uint8_t*>(mapping.data)[0] == 0x5A,
          "DMA mapping did not expose the initialized producer allocation");
  auto* retained = static_cast<GstSample*>(tensor.storage->holder.get());
  require(sima_test::dmabuf_span(gst_sample_get_buffer(retained)) == backing,
          "mapping replaced producer backing allocation");
  std::weak_ptr<Storage> storage = tensor.storage;
  sample = {};
  tensor = {};
  require(!storage.expired() && gate->inflight() == 1 && !state->producer_released,
          "Mapping outlived its Tensor but lost the current storage/loan/producer guard");
  require(!state->end_completed, "DMA CPU epoch ended before Mapping release");
  mapping = {};
  require(storage.expired() && gate->inflight() == 0 && gate->released() == 1U,
          "DMA Mapping did not release its final loan exactly once");
  require(state->producer_released && state->end_before_release,
          "producer loan released before the real DMA CPU epoch finished");
}

} // namespace

RUN_TEST("unit_tensor_image_mapping_test", ([] {
           using namespace simaai::neat;
           gst_init(nullptr, nullptr);
           test_dmabuf_logical_read_range(false);
           test_dmabuf_logical_read_range(true);
           test_dmabuf_composite_read(true);
           test_dmabuf_composite_read(false);
           for (bool projected : {false, true}) {
             test_dmabuf_mapping_lifetime(projected, false);
             test_dmabuf_mapping_lifetime(projected, true);
           }
           test_custom_logical_read_contract();

           // Reject malformed offsets before choosing direct or ordinary storage mapping.
           Tensor invalid_offset = make_i420_tensor(8, 6);
           invalid_offset.byte_offset = -1;
           require(!pipeline_internal::map_tensor_view(invalid_offset, MapMode::Read).data,
                   "negative tensor offset must not alias the start of its allocation");
           invalid_offset.byte_offset = 73;
           require(!pipeline_internal::map_tensor_view(invalid_offset, MapMode::Read).data,
                   "tensor offset beyond the allocation must fail before pointer arithmetic");

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

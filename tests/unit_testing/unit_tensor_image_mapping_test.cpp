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

void test_dmabuf_mapping_lifetime(bool projected) {
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
  Mapping mapping = tensor.map_read();
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
           test_dmabuf_mapping_lifetime(false);
           test_dmabuf_mapping_lifetime(true);

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

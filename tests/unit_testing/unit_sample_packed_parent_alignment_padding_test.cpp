#include "pipeline/Tensor.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "gst/SimaTensorSetMetaAbi.h"
#include "test_main.h"

#include <gst/gst.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

// Regression for issue #538 (device-tensor path): a pushed multi-input tensor
// list whose FIRST tensor is a device-backed buffer padded for alignment
// (runtime segment > logical-tight bytes, not tessellated) must be laid out in
// the packed parent by its LOGICAL-TIGHT span, not its padded transport span.
// Otherwise every subsequent tensor's byte offset drifts past what the pre-MLA
// casttess/quanttess consumer reads (it reads at logical-tight offsets), which
// on RF-DETR fed the gather bytes from feature padding -> scrambled IFM.
//
// Here the first tensor has 16 logical bytes but 24 bytes of backing storage
// (8 bytes of alignment padding). The second tensor must therefore start at
// offset 16 (tight), not 24 (padded).

using namespace simaai::neat;

namespace {

void ensure_gst_ready() {
  int argc = 0;
  char** argv = nullptr;
  gst_init(&argc, &argv);
  const gchar* tags[] = {nullptr};
  if (gst_meta_get_info("GstSimaSampleMeta") == nullptr) {
    (void)gst_meta_register_custom("GstSimaSampleMeta", tags, nullptr, nullptr, nullptr);
  }
  if (gst_meta_get_info(SIMA_TENSOR_SET_META_NAME) == nullptr) {
    (void)gst_meta_register_custom(SIMA_TENSOR_SET_META_NAME, tags, nullptr, nullptr, nullptr);
  }
}

// Logical shape yields `logical_bytes`; storage is allocated `storage_bytes`
// (>= logical) to model device alignment padding.
Tensor make_padded_tensor(int logical_index, std::size_t logical_bytes, std::size_t storage_bytes,
                          const std::string& name) {
  Tensor t;
  t.dtype = TensorDType::BFloat16;
  t.layout = TensorLayout::HWC;
  t.shape = {1, static_cast<int64_t>(logical_bytes / 2), 1};
  t.strides_bytes = {static_cast<int64_t>(logical_bytes), 2, 2};
  t.storage = make_cpu_owned_storage(storage_bytes);
  t.route.logical_index = logical_index;
  t.route.physical_index = logical_index;
  t.route.route_slot = logical_index;
  t.route.name = name;
  t.route.backend_name = name;
  t.route.segment_name = name;
  t.route.stage_key = "mla_padded_join_test";
  return t;
}

Tensor make_strided_uint8_tensor(const std::vector<std::uint8_t>& storage_bytes,
                                 std::vector<int64_t> shape, std::vector<int64_t> strides_bytes) {
  Tensor t;
  t.dtype = TensorDType::UInt8;
  t.layout = TensorLayout::HW;
  t.shape = std::move(shape);
  t.strides_bytes = std::move(strides_bytes);
  t.storage = make_cpu_owned_storage(storage_bytes.size());
  t.byte_offset = 0;
  t.read_only = false;

  Mapping map = t.storage->map(MapMode::Write);
  require(map.data != nullptr, "failed to map strided tensor storage");
  require(map.size_bytes >= storage_bytes.size(), "strided tensor storage map too small");
  std::memcpy(map.data, storage_bytes.data(), storage_bytes.size());
  return t;
}

std::vector<std::uint8_t> read_gst_buffer_bytes(GstBuffer* buffer) {
  require(buffer != nullptr, "missing GstBuffer for payload readback");
  GstMapInfo map{};
  require(gst_buffer_map(buffer, &map, GST_MAP_READ), "failed to map packed parent buffer");
  std::vector<std::uint8_t> out(static_cast<const std::uint8_t*>(map.data),
                                static_cast<const std::uint8_t*>(map.data) + map.size);
  gst_buffer_unmap(buffer, &map);
  return out;
}

void test_alignment_padding_offsets() {
  ensure_gst_ready();

  // feature: 16 logical bytes, 24 bytes backing (8 pad); gather: 8 bytes tight.
  Tensor feature = make_padded_tensor(0, 16U, 24U, "cast_0");
  Tensor gather = make_padded_tensor(1, 8U, 8U, "cast_1");
  const Sample sample = sample_from_tensors(TensorList{feature, gather});

  std::string err;
  auto holder = pipeline_internal::make_sample_holder_from_bundle(sample, &err);
  require(holder != nullptr, std::string("failed to materialize tensor-set holder: ") + err);

  auto* gst_sample = static_cast<GstSample*>(holder.get());
  require(gst_sample != nullptr, "materialized holder should be a GstSample");

  simaai::neat::pipeline_internal::TensorBufferView view;
  err.clear();
  require(simaai::neat::pipeline_internal::tensor_buffer_descriptor_from_sample(gst_sample, &view,
                                                                                &err),
          std::string("failed to extract tensorbuffer descriptor: ") + err);
  require(view.tensors.size() == 2U, "materialized tensor-set should expose two tensors");
  require(view.tensors[0].byte_offset == 0U, "first tensor should start at offset zero");

  // Only the single-parent (cumulative-offset) materialization is affected by
  // the packed layout; the fallback (one memory per tensor) uses child-relative
  // offsets where padding is irrelevant.
  if (view.tensors[0].memory_index == 0 && view.tensors[1].memory_index == 0) {
    require(view.tensors[1].byte_offset == 16U,
            "packed parent must lay out the second tensor at the first tensor's LOGICAL-TIGHT "
            "offset (16), dropping the 8 bytes of alignment padding — got " +
                std::to_string(view.tensors[1].byte_offset));
  }
}

void test_strided_dense_payload_compaction() {
  ensure_gst_ready();

  Tensor head = make_strided_uint8_tensor({'G', 'H'}, {1, 2}, {2, 1});
  Tensor strided =
      make_strided_uint8_tensor({'A', 'B', 'C', 0xEE, 'D', 'E', 'F', 0xEE}, {2, 3}, {4, 1});

  const Sample sample = sample_from_tensors(TensorList{head, strided});

  std::string err;
  auto holder = pipeline_internal::make_sample_holder_from_bundle(sample, &err,
                                                                  /*allow_zero_copy=*/false);
  require(holder != nullptr, std::string("failed to materialize tensor-set holder: ") + err);

  auto* gst_sample = static_cast<GstSample*>(holder.get());
  require(gst_sample != nullptr, "materialized holder should be a GstSample");

  simaai::neat::pipeline_internal::TensorBufferView view;
  err.clear();
  require(simaai::neat::pipeline_internal::tensor_buffer_descriptor_from_sample(gst_sample, &view,
                                                                                &err),
          std::string("failed to extract tensorbuffer descriptor: ") + err);
  require(view.tensors.size() == 2U, "packed tensor-set should expose two tensors");
  require(view.tensors[0].memory_index == 0 && view.tensors[1].memory_index == 0,
          "strided regression must exercise the packed-parent path");
  require(view.tensors[0].byte_offset == 0U, "first tensor should start at offset zero");
  require(view.tensors[0].size_bytes == 2U, "first tensor logical size should be tight");
  require(view.tensors[1].byte_offset == 2U,
          "second tensor should start after the first tensor's tight logical bytes");
  require(view.tensors[1].size_bytes == 6U, "second tensor logical size should be tight");
  require(view.tensors[1].stride_bytes == std::vector<int64_t>({3, 1}),
          "second tensor descriptor should describe the compacted packed-parent view");

  const std::vector<std::uint8_t> payload = read_gst_buffer_bytes(view.buffer);
  const std::vector<std::uint8_t> expected = {'G', 'H', 'A', 'B', 'C', 'D', 'E', 'F'};
  require(payload.size() >= expected.size(),
          "packed parent buffer is smaller than expected logical payload");
  require(std::equal(expected.begin(), expected.end(), payload.begin()),
          "packed parent must compact dense strided input bytes before appending the next tensor");
}

} // namespace

int main() {
  int failures = 0;
  failures += sima_test::run_test("unit_sample_packed_parent_alignment_padding_test",
                                  [] { test_alignment_padding_offsets(); });
  failures += sima_test::run_test("unit_sample_packed_parent_strided_dense_payload_test",
                                  [] { test_strided_dense_payload_compaction(); });
  return failures == 0 ? 0 : 1;
}

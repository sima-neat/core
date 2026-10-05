#include "pipeline/Tensor.h"
#include "gst/GstInit.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/SimaaiGstCompat.h"
#include "pipeline/internal/TensorTransfer.h"
#include <simaai/simaai_memory.h>
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

void test_bundled_input_rejects_invalid_backing_before_allocation() {
  ensure_gst_ready();
  Tensor tensor = make_padded_tensor(0, 16U, 16U, "y_src");
  GstBuffer* output = nullptr;
  std::string error;
  tensor.byte_offset = -1;
  require(!pipeline_internal::build_bundled_input_gst_buffer({tensor}, &output, &error),
          "negative input byte offset must be rejected");
  require(output == nullptr, "rejected input must not return an allocated buffer");
  tensor.byte_offset = 2;
  require(!pipeline_internal::build_bundled_input_gst_buffer({tensor}, &output, &error),
          "input range beyond backing must be rejected");
  tensor.byte_offset = 0;
  tensor.storage->data = nullptr;
  require(!pipeline_internal::build_bundled_input_gst_buffer({tensor}, &output, &error),
          "non-Metoak device carrier must keep its existing materialized path");
  require(error.find("materialized fallback") != std::string::npos,
          "generic device rejection must occur before allocation");
  tensor.storage.reset();
  require(!pipeline_internal::build_bundled_input_gst_buffer({tensor}, &output, &error),
          "missing input storage must be rejected");
  require(output == nullptr, "failure must preserve null output");
}

// Regression: the packed parent used to be a fresh contiguous device allocation on every
// push, which fails under CMA pressure even when the pipeline is otherwise steady. Repeated
// pushes of one layout must draw from a single cached segment pool and carry each push's bytes.
void test_packed_parent_reuses_pooled_buffer() {
  ensure_gst_ready();

  GstBufferPool* first_pool = nullptr;
  std::size_t misses_after_first = 0U;
  for (int push = 0; push < 9; ++push) {
    const auto base = static_cast<std::uint8_t>(push * 16);
    std::vector<std::uint8_t> head_bytes(12);
    std::vector<std::uint8_t> tail_bytes(20);
    for (std::size_t i = 0; i < head_bytes.size(); ++i)
      head_bytes[i] = static_cast<std::uint8_t>(base + i);
    for (std::size_t i = 0; i < tail_bytes.size(); ++i)
      tail_bytes[i] = static_cast<std::uint8_t>(base + 100 + i);
    Tensor head = make_strided_uint8_tensor(head_bytes, {3, 4}, {4, 1});
    Tensor tail = make_strided_uint8_tensor(tail_bytes, {4, 5}, {5, 1});
    const Sample sample = sample_from_tensors(TensorList{head, tail});

    std::string err;
    auto holder = pipeline_internal::make_sample_holder_from_bundle(sample, &err,
                                                                    /*allow_zero_copy=*/false);
    require(holder != nullptr, std::string("failed to materialize tensor-set holder: ") + err);
    {
      pipeline_internal::TensorBufferView view;
      require(pipeline_internal::tensor_buffer_descriptor_from_sample(
                  static_cast<GstSample*>(holder.get()), &view, &err),
              std::string("failed to extract tensorbuffer descriptor: ") + err);
      require(view.tensors.size() == 2U && view.tensors[0].memory_index == 0 &&
                  view.tensors[1].memory_index == 0,
              "pool regression must exercise the packed-parent path");
      require(view.buffer != nullptr && view.buffer->pool != nullptr,
              "packed parent must be acquired from the segment pool");
      if (push == 0) {
        first_pool = view.buffer->pool;
        misses_after_first = pipeline_internal::tensor_transfer_pool_stats().misses;
      }
      require(view.buffer->pool == first_pool,
              "every push of one layout must reuse the same segment pool");

      std::vector<std::uint8_t> expected = head_bytes;
      expected.insert(expected.end(), tail_bytes.begin(), tail_bytes.end());
      const std::vector<std::uint8_t> payload = read_gst_buffer_bytes(view.buffer);
      require(payload.size() >= expected.size() &&
                  std::equal(expected.begin(), expected.end(), payload.begin()),
              "pooled packed parent must carry this push's bytes, not a previous push's");
    }
    holder.reset();
  }
  require(pipeline_internal::tensor_transfer_pool_stats().misses == misses_after_first,
          "repeated pushes of one layout must not create additional segment pools");
}

// Explicit target-only gate. No CVU dispatch, but real EV74-addressable allocation
// is mandatory: unavailable allocator/device fails rather than silently skipping.
void test_device_bundle_subviews() {
  simaai::neat::gst_init_once();
  ensure_gst_ready();
  gst_simaai_segment_memory_init_once();
  const char* names[] = {"y_src", "u_src", "v_src", "disp_src", "bf_mm_src", "proj_src"};
  const TensorDType types[] = {TensorDType::UInt8,  TensorDType::UInt8,   TensorDType::UInt8,
                               TensorDType::UInt16, TensorDType::Float32, TensorDType::Float32};
  const std::size_t sizes[] = {140, 35, 35, 280, 4, 12};
  const std::vector<std::int64_t> shapes[] = {{14, 10}, {7, 5}, {7, 5}, {14, 10}, {1}, {3}};
  TensorList tensors;
  std::vector<std::vector<std::uint8_t>> expected;
  for (std::size_t i = 0; i < 6; ++i) {
    std::vector<std::uint8_t> data(sizes[i] + 4, 0xee);
    for (std::size_t j = 0; j < sizes[i]; ++j)
      data.at(j + 4) = static_cast<std::uint8_t>((i * 37 + j) & 255);
    expected.emplace_back(data.begin() + 4, data.end());
    Tensor tensor =
        Tensor::from_vector(data, {static_cast<std::int64_t>(data.size())}, TensorMemory::EV74);
    require(tensor.storage && !tensor.storage->data, "must exercise device-backed read path");
    tensor.dtype = types[i];
    tensor.shape = shapes[i];
    tensor.strides_bytes.clear();
    tensor.byte_offset = 4;
    tensor.layout = i < 4 ? TensorLayout::HW : TensorLayout::Unknown;
    tensor.route.name = tensor.route.backend_name = tensor.route.segment_name = names[i];
    tensor.route.logical_index = tensor.route.physical_index = static_cast<int>(i);
    tensor.route.memory_index = 0;
    tensors.push_back(std::move(tensor));
  }
  GstBuffer* output = nullptr;
  std::string error;
  require(pipeline_internal::build_bundled_input_gst_buffer(tensors, &output, &error),
          "device bundle: " + error);
  std::unique_ptr<GstBuffer, decltype(&gst_buffer_unref)> owned(output, &gst_buffer_unref);
  require(gst_buffer_n_memory(output) == 1, "bundle must use one named-segment carrier");
  auto* memory = gst_buffer_peek_memory(output, 0);
  for (std::size_t i = 0; i < 6; ++i) {
    auto* segment = static_cast<simaai_memory_t*>(gst_simaai_memory_get_segment(memory, names[i]));
    require(segment != nullptr, "missing named segment");
    require(simaai_memory_get_bus(segment) % 4 == 0, "mixed input segment is not 4-byte aligned");
    require(simaai_memory_get_size(segment) >= sizes[i], "segment too small");
    auto* mapped = static_cast<std::uint8_t*>(simaai_memory_map(segment));
    require(mapped != nullptr, "cannot map bundle segment");
    simaai_memory_invalidate_cache(segment);
    bool equal = std::equal(expected[i].begin(), expected[i].end(), mapped);
    simaai_memory_unmap(segment);
    require(equal, "nonzero source offset copy mismatch");
  }
  GstSample* sample = gst_sample_new(output, nullptr, nullptr, nullptr);
  pipeline_internal::TensorBufferView view;
  const bool described =
      pipeline_internal::tensor_buffer_descriptor_from_sample(sample, &view, &error);
  gst_sample_unref(sample);
  require(described && view.tensors.size() == 6, "missing copied descriptor: " + error);
  for (std::size_t i = 0; i < 6; ++i) {
    // The Internals view normalizes memory_index to the named segment-table
    // index, not GstBuffer memory[0]. Validate the effective bus span instead
    // of assuming one representation for that normalized view.
    const auto& descriptor = view.tensors[i];
    auto* resolved = static_cast<simaai_memory_t*>(
        gst_simaai_memory_get_segment(memory, descriptor.segment_name.c_str()));
    auto* canonical =
        static_cast<simaai_memory_t*>(gst_simaai_memory_get_segment(memory, names[i]));
    require(resolved && canonical && descriptor.byte_offset >= 0, "invalid resolved named span");
    const auto offset = static_cast<std::uint64_t>(descriptor.byte_offset);
    require(offset <= simaai_memory_get_size(resolved) &&
                sizes[i] <= simaai_memory_get_size(resolved) - offset,
            "resolved metadata exceeds segment backing");
    require(simaai_memory_get_bus(resolved) + offset == simaai_memory_get_bus(canonical),
            "source offset or tight-layout padding leaked into resolved metadata address");
    require(descriptor.size_bytes == sizes[i], "padding leaked into logical size");
  }
  owned.reset();
  // Force source map failure after destination allocation. The original input
  // remains alive, while each failed call must leave no output ownership behind.
  TensorList failed = tensors;
  failed[3].storage = std::make_shared<TensorBuffer>(*tensors[3].storage);
  failed[3].storage->kind = StorageKind::DeviceHandle;
  failed[3].storage->data = nullptr;
  int attempts = 0;
  failed[3].storage->map_fn = [&](MapMode) {
    ++attempts;
    return Mapping{};
  };
  for (int repetition = 0; repetition < 8; ++repetition) {
    output = nullptr;
    error.clear();
    require(!pipeline_internal::build_bundled_input_gst_buffer(failed, &output, &error),
            "forced read failure succeeded");
    require(output == nullptr && error.find("device tensor read failed") != std::string::npos,
            "failure must return null with read error");
  }
  require(attempts == 8, "failure injection did not exercise source mapper");
  output = nullptr;
  error.clear();
  require(pipeline_internal::build_bundled_input_gst_buffer(tensors, &output, &error),
          "bundle did not recover after read failure: " + error);
  gst_buffer_unref(output);
}

} // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--device-bundle")
    return sima_test::run_test("device_bundle_subviews_alignment_and_read_failure",
                               test_device_bundle_subviews) == 0
               ? 0
               : 1;
  int failures = 0;
  failures += sima_test::run_test("unit_sample_packed_parent_alignment_padding_test",
                                  [] { test_alignment_padding_offsets(); });
  failures += sima_test::run_test("unit_sample_packed_parent_strided_dense_payload_test",
                                  [] { test_strided_dense_payload_compaction(); });
  failures += sima_test::run_test("unit_bundled_input_invalid_backing_test", [] {
    test_bundled_input_rejects_invalid_backing_before_allocation();
  });
  failures += sima_test::run_test("unit_sample_packed_parent_pooled_reuse_test",
                                  [] { test_packed_parent_reuses_pooled_buffer(); });
  return failures == 0 ? 0 : 1;
}

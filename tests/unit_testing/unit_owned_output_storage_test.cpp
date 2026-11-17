#include "gst/GstInit.h"
#include "pipeline/gst/InputStreamInternal.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/SimaaiGstCompat.h"
#include "dmabuf_test_utils.h"
#include "test_main.h"
#include "owned_output_test_utils.h"

#include <cstring>

namespace {
using namespace simaai::neat;
using namespace simaai::neat::pipeline_internal;
using namespace simaai::gst;

template <class Action> void require_rejected(Action action, const std::string& context) {
  bool rejected = false;
  try {
    action();
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, context + ": invalid Owned sample was accepted");
}

void exercise(bool dma_source, bool compact) {
  bool released = false;
  auto buffer = std::shared_ptr<GstBuffer>(gst_buffer_new(), gst_buffer_unref);
  for (unsigned i = 0; i < 2; ++i) {
    auto allocation =
        std::shared_ptr<GstBuffer>(dma_source ? sima_test::allocate_cma_dmabuf(64)
                                              : gst_buffer_new_allocate(nullptr, 64, nullptr),
                                   gst_buffer_unref);
    auto sample = std::shared_ptr<GstSample>(
        gst_sample_new(allocation.get(), nullptr, nullptr, nullptr), gst_sample_unref);
    Tensor reference;
    reference.storage = make_gst_sample_storage(sample.get());
    auto storage = tensor_view_from_sample_memory(reference, 0).storage;
    {
      auto map = storage->map(MapMode::Write);
      require(map.data && map.size_bytes >= 64, "fixture mapping");
      for (unsigned j = 0; j < 64; ++j)
        static_cast<unsigned char*>(map.data)[j] = static_cast<unsigned char>(j + 64 * i);
    }
    auto* memory = gst_buffer_peek_memory(allocation.get(), 0);
    gst_buffer_append_memory(buffer.get(), gst_memory_ref(memory));
  }
  TensorBufferPublishContract contract;
  contract.stage_key = "owned-output";
  contract.physical_outputs = {{0, 64, "arena0"}, {1, 64, "arena1"}};
  for (int i = 0; i < 3; ++i) {
    TensorBufferPublishLogicalOutput head;
    head.logical_index = i;
    head.memory_index = head.physical_index = i / 2;
    head.backend_output_index = head.route_slot = i;
    head.logical_name = head.backend_name = "head" + std::to_string(i);
    head.segment_name = "arena" + std::to_string(i / 2);
    head.byte_offset = 4 + 16 * (i % 2);
    head.size_bytes = 4; // Logical bytes; the padded physical span is 10.
    head.dtype = SIMA_TENSOR_SET_DTYPE_INT8_V1;
    head.layout = SIMA_TENSOR_SET_LAYOUT_HW_V1;
    head.shape = {2, 2};
    head.stride_bytes = {8, 1};
    head.quant = TensorBufferQuantView{1, 1, {0.25, 0.5}, {-3, 7}};
    contract.logical_outputs.push_back(head);
  }
  std::string error;
  TensorBufferPreparedMetaTemplate prepared;
  require(tensor_buffer_prepare_meta_template_from_contract(contract, {{0, 0, 64}, {1, 0, 64}},
                                                            &prepared, &error),
          error);
  if (compact) {
    auto publication = tensor_buffer_freeze_publication(prepared, &error);
    require(publication && tensor_buffer_attach_publication(buffer.get(), publication, &error),
            error);
  } else {
    require(tensor_buffer_attach_prepared_meta(buffer.get(), prepared, &error), error);
  }
  InputOptions options;
  options.preprocess_meta.emplace();
  options.preprocess_meta->enabled = true;
  options.preprocess_meta->target_width = options.preprocess_meta->target_height = 640;
  options.preprocess_meta->resize_mode = "letterbox";
  require(apply_simaai_preprocess_meta_template(buffer.get(), options, 1280, 720),
          "preprocess fixture");
  GST_BUFFER_PTS(buffer.get()) = 1234;
  gst_mini_object_weak_ref(
      GST_MINI_OBJECT(buffer.get()),
      [](gpointer flag, GstMiniObject*) { *static_cast<bool*>(flag) = true; }, &released);
  auto caps = std::shared_ptr<GstCaps>(gst_caps_new_empty_simple("application/vnd.simaai.tensor"),
                                       gst_caps_unref);
  auto source = std::shared_ptr<GstSample>(
      gst_sample_new(buffer.get(), caps.get(), nullptr, nullptr), gst_sample_unref);
  reset_tensor_io_stats();
  InputStream::State state;
  state.opt.copy_output = true;
  state.lifetime_token = std::make_shared<int>(0);
  auto owned = decode_sample_from_inputstream_state(state, source.get(), "owned output test");
  state.lifetime_token.reset();
  require(!sample_has_device_gstsample_producer_lifetime(owned),
          "independent output must not depend on producer Run lifetime");
  require(owned.owned && owned.tensors.size() == 3, "owned logical outputs");
  require(snapshot_tensor_io_stats().tensor_copy_count == 2 &&
              snapshot_tensor_io_stats().tensor_copy_bytes == 128,
          "copy each physical carrier once, not once per logical head");
  auto* output =
      gst_sample_get_buffer(static_cast<GstSample*>(owned.tensors[0].storage->holder.get()));
  require(gst_buffer_n_memory(output) == 2, "preserve physical memory slots");
  for (unsigned i = 0; i < 2; ++i) {
    auto* memory = gst_buffer_peek_memory(output, i);
    require(sima_test::is_system_memory(memory), "Owned output must be system memory");
    require(memory != gst_buffer_peek_memory(buffer.get(), i),
            "Owned must not reuse producer allocation");
  }
  (void)sima_test::require_owned_output(owned, "independent Owned control");
  {
    auto not_owned = owned;
    not_owned.owned = false;
    require_rejected([&] { sima_test::require_owned_output(not_owned, "not Owned"); },
                     "Owned flag control");
    // A borrowed CPU producer is system memory too; the release check below
    // proves independence for that source.
    if (dma_source) {
      auto borrowed = sample_from_gst_envelope(source.get(), "borrowed control", false, nullptr);
      borrowed.owned = true; // A flag alone must not disguise a producer allocation.
      require_rejected([&] { sima_test::require_owned_output(borrowed, "claimed Owned producer"); },
                       "borrowed carrier control");
    }
    // An independent backing allocation still must not retain a producer parent.
    auto parent_buffer = std::shared_ptr<GstBuffer>(gst_buffer_new(), gst_buffer_unref);
    gst_buffer_append_memory(parent_buffer.get(),
                             gst_memory_ref(gst_buffer_peek_memory(output, 0)));
    require(gst_buffer_add_parent_buffer_meta(parent_buffer.get(), buffer.get()) != nullptr,
            "parent-meta control fixture");
    auto parent_sample = std::shared_ptr<GstSample>(
        gst_sample_new(parent_buffer.get(), caps.get(), nullptr, nullptr), gst_sample_unref);
    Sample parent_owned;
    parent_owned.kind = SampleKind::Tensor;
    parent_owned.tensor.emplace();
    parent_owned.tensor->storage = make_gst_sample_storage(parent_sample.get());
    parent_owned.owned = true;
    require_rejected([&] { sima_test::require_owned_output(parent_owned, "Owned parent"); },
                     "producer parent control");
  }
  // Drop every producer reference, including metadata. The output must not pin
  // its pool via a parent-buffer meta, mapping callback or shared holder.
  for (int i = 0; i < 2; ++i) {
    Tensor reference;
    reference.storage = make_gst_sample_storage(source.get());
    auto selected = tensor_view_from_sample_memory(reference, i);
    auto mapping = selected.storage->map(MapMode::Write);
    require(mapping.data && mapping.size_bytes >= 64, "producer reuse mapping");
    std::memset(mapping.data, 0, 64);
  }
  source.reset();
  buffer.reset();
  require(released, "Owned output retained producer loan");
  for (int i = 0; i < 3; ++i) {
    const auto& tensor = owned.tensors[i];
    require(tensor.shape == std::vector<int64_t>({2, 2}) &&
                tensor.strides_bytes == std::vector<int64_t>({8, 1}) &&
                tensor.byte_offset == 4 + 16 * (i % 2) && tensor.route.memory_index == i / 2,
            "preserve logical shape, padded strides, offset and head order");
    require(tensor.semantic.quant &&
                tensor.semantic.quant->scales == std::vector<float>({0.25F, 0.5F}) &&
                tensor.semantic.quant->zero_points == std::vector<int32_t>({-3, 7}),
            "quantization");
    require(tensor.semantic.preprocess && tensor.semantic.preprocess->original_width == 1280,
            "preprocess geometry");
    auto view = tensor.view_read();
    const auto* bytes = static_cast<const unsigned char*>(view.data);
    require(bytes && view.size_bytes >= 10 && bytes[0] == 64 * (i / 2) + 4 + 16 * (i % 2) &&
                bytes[8] == bytes[0] + 8,
            "copied head payload and padding");
  }
  const auto before = snapshot_tensor_io_stats();
  auto reentry = sample_to_gst_envelope_holder(owned, &error, true);
  require(bool(reentry), "reentry: " + error);
  auto* input = gst_sample_get_buffer(static_cast<GstSample*>(reentry.get()));
  require(gst_buffer_n_memory(input) == 2, "reentry carrier count");
  const auto root = [](GstMemory* memory) {
    while (memory && memory->parent)
      memory = memory->parent;
    return memory;
  };
  for (unsigned i = 0; i < 2; ++i)
    require(root(gst_buffer_peek_memory(input, i)) == root(gst_buffer_peek_memory(output, i)),
            "reentry must not copy again");
  require(snapshot_tensor_io_stats().tensor_copy_count == before.tensor_copy_count,
          "no reentry staging copy");
  sima_test::require_explicit_host_copy(owned, "explicit host conversion");
  auto retained = sample_from_gst_envelope(static_cast<GstSample*>(reentry.get()), "retain output",
                                           false, nullptr);
  materialize_owned_output(retained, true);
  require(retained.owned, "Auto copies CPU memory that no framework pool certified");
}
void exercise_packed_runtime_whole() {
  // The installed Runtime allocator admits one synchronized ownership root for
  // the complete packed Y/UV allocation, not unrelated child allocations.
  gst_simaai_segment_memory_init_once();
  auto allocator = std::shared_ptr<GstAllocator>(gst_simaai_memory_get_segment_allocator(),
                                                 [](GstAllocator* p) { gst_object_unref(p); });
  require(allocator != nullptr, "packed Runtime allocator is unavailable");
  GstSimaaiAllocationParams parameters;
  gst_simaai_memory_allocation_params_init(&parameters);
  parameters.parent.flags = static_cast<GstMemoryFlags>(GST_SIMAAI_MEMORY_FLAG_DEFAULT);
  require(gst_simaai_memory_allocation_params_add_segment(&parameters, 16U, "Y") &&
              gst_simaai_memory_allocation_params_add_segment(&parameters, 8U, "UV"),
          "packed Runtime plane parameters");
  auto* memory = gst_allocator_alloc(allocator.get(), 24U, &parameters.parent);
  require(memory != nullptr, "packed Runtime plane allocation failed");
  {
    // Initialize the uniquely owned memory before the Sample adds buffer refs.
    // Runtime's allocator brackets this complete ownership-root CPU write.
    GstMapInfo map{};
    require(gst_memory_map(memory, &map, GST_MAP_WRITE),
            "packed Runtime synchronized root initialization");
    const bool complete = map.data && map.size == 24U;
    if (complete) {
      for (unsigned i = 0; i < 24U; ++i) {
        map.data[i] = static_cast<unsigned char>(i + 1U);
      }
    }
    gst_memory_unmap(memory, &map);
    require(complete, "packed Runtime root initialization extent");
  }

  bool released = false;
  auto buffer = std::shared_ptr<GstBuffer>(gst_buffer_new(), gst_buffer_unref);
  gst_buffer_append_memory(buffer.get(), memory);
  gst_mini_object_weak_ref(
      GST_MINI_OBJECT(buffer.get()),
      [](gpointer flag, GstMiniObject*) { *static_cast<bool*>(flag) = true; }, &released);
  auto source = std::shared_ptr<GstSample>(gst_sample_new(buffer.get(), nullptr, nullptr, nullptr),
                                           gst_sample_unref);
  Tensor tensor;
  tensor.storage = make_gst_sample_storage(source.get());
  tensor.shape = {4, 4};
  tensor.strides_bytes = {4, 1};
  tensor.dtype = TensorDType::UInt8;
  tensor.layout = TensorLayout::HW;
  tensor.planes = {{PlaneRole::Y, {4, 4}, {4, 1}, 0}, {PlaneRole::UV, {2, 4}, {4, 1}, 16}};
  tensor.route.physical_index = 7;
  require(tensor.storage && tensor.storage->size_bytes == 24U, "packed Runtime root extent");
  {
    // A partial GstMemory view still publishes its original typed segment table.
    // It is not authority for inventing a new packed plane layout or offsets.
    auto* partial = gst_memory_share(memory, 1, 23);
    require(partial != nullptr, "packed Runtime partial-view fixture");
    auto partial_buffer = std::shared_ptr<GstBuffer>(gst_buffer_new(), gst_buffer_unref);
    gst_buffer_append_memory(partial_buffer.get(), partial);
    auto partial_sample = std::shared_ptr<GstSample>(
        gst_sample_new(partial_buffer.get(), nullptr, nullptr, nullptr), gst_sample_unref);
    Sample invalid;
    invalid.kind = SampleKind::Tensor;
    invalid.tensor = tensor;
    invalid.tensor->storage = make_gst_sample_storage(partial_sample.get());
    reset_tensor_io_stats();
    require_rejected([&] { materialize_owned_output(invalid); },
                     "partial packed Runtime root authority");
    require(snapshot_tensor_io_stats().tensor_copy_count == 0U,
            "invalid packed Runtime whole view allocated/copied before rejection");
  }
  Sample owned;
  owned.kind = SampleKind::Tensor;
  owned.tensor = tensor;
  reset_tensor_io_stats();
  materialize_owned_output(owned);
  require(owned.owned && owned.tensor && snapshot_tensor_io_stats().tensor_copy_count == 1U &&
              snapshot_tensor_io_stats().tensor_copy_bytes == 24U,
          "copy packed Runtime ownership root once, including all planes");
  require(owned.tensor->route.memory_index == 0 && owned.tensor->route.physical_index == 7 &&
              owned.tensor->planes.size() == 2U && owned.tensor->planes[1].byte_offset == 16 &&
              owned.tensor->planes[1].strides_bytes == std::vector<int64_t>({4, 1}),
          "preserve packed Runtime plane offsets, strides and physical provenance");
  (void)sima_test::require_owned_output(owned, "packed Runtime independent output");
  tensor.storage.reset();
  source.reset();
  buffer.reset();
  require(released, "packed Runtime output retained its producer root");
  auto map = owned.tensor->storage->map(MapMode::Read);
  require(map.data && map.size_bytes == 24U, "packed Runtime independent mapping");
  for (unsigned i = 0; i < 24U; ++i) {
    require(static_cast<const unsigned char*>(map.data)[i] == i + 1U,
            "packed Runtime Y/UV copied payload");
  }
}

} // namespace

RUN_TEST("unit_owned_output_storage_test", ([] {
           gst_init_once();
           exercise_packed_runtime_whole();
           for (bool dma_source : {false, true})
             for (bool compact : {false, true})
               exercise(dma_source, compact);
         }));

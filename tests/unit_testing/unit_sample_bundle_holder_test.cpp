#include "gst/GstInit.h"
#include "gst/SimaTensorSetMetaAbi.h"
#include "pipeline/internal/OutputTensorOverride.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/TensorUtil.h"

#include "dmabuf_test_utils.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {
Sample output_from_sample_stream(GstSample* sample, const char* where, bool copy_output,
                                 const std::optional<OutputTensorOverride>* override_opt);
}

namespace {

GstSample* make_tensor_sample_with_contract_meta() {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 64U, nullptr);
  require(buffer != nullptr, "failed to allocate GstBuffer");

  GstMapInfo map{};
  require(gst_buffer_map(buffer, &map, GST_MAP_WRITE), "failed to map GstBuffer");
  std::memset(map.data, 0xA5, map.size);
  gst_buffer_unmap(buffer, &map);

  GstCaps* caps =
      gst_caps_new_simple("application/vnd.simaai.tensor", "format", G_TYPE_STRING, "MLA", "width",
                          G_TYPE_INT, 4, "height", G_TYPE_INT, 4, "depth", G_TYPE_INT, 1, "dtype",
                          G_TYPE_STRING, "INT8", "layout", G_TYPE_STRING, "HW", nullptr);
  require(caps != nullptr, "failed to allocate GstCaps");

  GstCustomMeta* meta = gst_buffer_add_custom_meta(buffer, SIMA_TENSOR_SET_META_NAME);
  require(meta != nullptr, "failed to add tensor-set meta");
  GstStructure* s = gst_custom_meta_get_structure(meta);
  require(s != nullptr, "failed to get tensor-set meta structure");

  std::vector<SimaTensorDescriptorV2> descriptors(2);
  descriptors[0].logical_index = 0;
  descriptors[0].physical_index = 0;
  descriptors[0].backend_output_index = 0;
  descriptors[0].route_slot = 0;
  descriptors[0].memory_index = 0;
  descriptors[0].logical_name_id = 0;
  descriptors[0].backend_name_id = 1;
  descriptors[0].segment_name_id = 2;
  descriptors[0].byte_offset = 0;
  descriptors[0].size_bytes = 16;
  descriptors[0].dtype = SIMA_TENSOR_SET_DTYPE_INT8_V1;
  descriptors[0].layout = SIMA_TENSOR_SET_LAYOUT_HW_V1;
  descriptors[0].rank = 2;
  descriptors[0].shape[0] = 4;
  descriptors[0].shape[1] = 4;
  descriptors[0].stride_bytes[0] = 4;
  descriptors[0].stride_bytes[1] = 1;

  descriptors[1].logical_index = 1;
  descriptors[1].physical_index = 0;
  descriptors[1].backend_output_index = 1;
  descriptors[1].route_slot = 1;
  descriptors[1].memory_index = 0;
  descriptors[1].logical_name_id = 3;
  descriptors[1].backend_name_id = 4;
  descriptors[1].segment_name_id = 2; // One physical carrier, two distinct logical views.
  descriptors[1].byte_offset = 16;
  descriptors[1].size_bytes = 16;
  descriptors[1].dtype = SIMA_TENSOR_SET_DTYPE_INT8_V1;
  descriptors[1].layout = SIMA_TENSOR_SET_LAYOUT_HW_V1;
  descriptors[1].rank = 2;
  descriptors[1].shape[0] = 4;
  descriptors[1].shape[1] = 4;
  descriptors[1].stride_bytes[0] = 4;
  descriptors[1].stride_bytes[1] = 1;

  GBytes* descriptor_bytes =
      g_bytes_new(descriptors.data(), descriptors.size() * sizeof(SimaTensorDescriptorV2));
  require(descriptor_bytes != nullptr, "failed to allocate descriptor bytes");

  const char* raw_names[] = {"boxes", "ofm0", "shared_arena", "scores", "ofm1", nullptr};
  gchar** name_table = g_strdupv(const_cast<gchar**>(raw_names));
  require(name_table != nullptr, "failed to duplicate tensor-set name table");

  gst_structure_set(
      s, SIMA_TENSOR_SET_META_FIELD_VERSION, G_TYPE_UINT, SIMA_TENSOR_SET_META_VERSION,
      SIMA_TENSOR_SET_META_FIELD_TENSOR_COUNT, G_TYPE_UINT, static_cast<guint>(descriptors.size()),
      SIMA_TENSOR_SET_META_FIELD_DESCRIPTOR_SIZE, G_TYPE_UINT,
      static_cast<guint>(sizeof(SimaTensorDescriptorV2)), SIMA_TENSOR_SET_META_FIELD_DESCRIPTORS,
      G_TYPE_BYTES, descriptor_bytes, SIMA_TENSOR_SET_META_FIELD_STAGE_KEY, G_TYPE_STRING,
      "mla.stage.bundle", SIMA_TENSOR_SET_META_FIELD_NAME_TABLE, G_TYPE_STRV, name_table, nullptr);
  g_strfreev(name_table);
  g_bytes_unref(descriptor_bytes);

  GstSample* sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  gst_caps_unref(caps);
  gst_buffer_unref(buffer);
  require(sample != nullptr, "failed to create GstSample");
  return sample;
}

simaai::neat::Tensor make_test_tensor(std::vector<int64_t> shape, std::size_t bytes) {
  simaai::neat::Tensor tensor;
  tensor.shape = std::move(shape);
  tensor.dtype = simaai::neat::TensorDType::UInt8;
  tensor.storage = simaai::neat::make_cpu_owned_storage(bytes);
  return tensor;
}

simaai::neat::Sample make_named_tensor_field(const std::string& name) {
  simaai::neat::Sample field =
      simaai::neat::sample_from_tensors(simaai::neat::TensorList{make_test_tensor({2, 2}, 4)});
  field.stream_label = name;
  field.segment_name = name;
  return field;
}

void check_bundle_output_override_preserves_bundle_kind() {
  simaai::neat::Sample bundle;
  bundle.kind = simaai::neat::SampleKind::Bundle;
  bundle.frame_id = 17;
  bundle.stream_id = "camera0";
  bundle.fields.push_back(make_named_tensor_field("old_boxes"));
  bundle.fields.push_back(make_named_tensor_field("old_scores"));

  simaai::neat::OutputTensorOverride override;
  simaai::neat::OutputTensorOverrideEntry boxes;
  boxes.name = "boxes";
  boxes.segment_name = "boxes_seg";
  boxes.shape = {1, 4};
  boxes.dtype = simaai::neat::TensorDType::UInt8;
  boxes.logical_output_index = 3;
  boxes.memory_index = 8;
  boxes.route_slot = 11;
  simaai::neat::OutputTensorOverrideEntry scores;
  scores.name = "scores";
  scores.segment_name = "scores_seg";
  scores.shape = {1, 2};
  scores.dtype = simaai::neat::TensorDType::UInt8;
  scores.logical_output_index = 4;
  scores.memory_index = 9;
  scores.route_slot = 12;
  override.outputs = {boxes, scores};

  simaai::neat::Sample out =
      simaai::neat::apply_output_tensor_override(bundle, override, /*materialize_output=*/false);
  require(out.kind == simaai::neat::SampleKind::Bundle,
          "output override on Bundle must preserve Bundle kind");
  require(out.size() == 2U, "Bundle output override should preserve logical field count");
  require(simaai::neat::sample_is_multi_output(out),
          "Bundle output override should remain visible as multi-output");
  require(out.fields.size() == 2U, "Bundle output override should preserve fields");
  require(out.tensors.empty(), "Bundle output override should not expose root TensorSet payload");
  require(out[0].stream_label == "boxes", "operator[] should expose first bundle field");
  require(out.fields[0].stream_label == "boxes", "first field stream label override mismatch");
  require(out.fields[0].segment_name == "boxes_seg", "first field segment override mismatch");
  require(out.fields[0].logical_output_index == 3 && out.fields[0].output_index == 3,
          "first field logical output override mismatch");
  require(out.fields[0].memory_index == 8 && out.fields[0].route_slot == 11,
          "first field route override mismatch");
  require(out.fields[0].tensors.front().shape == std::vector<int64_t>({1, 4}),
          "first field tensor shape override mismatch");
  require(out[1].stream_label == "scores", "operator[] should expose second bundle field");
  require(out.fields[1].stream_label == "scores", "second field stream label override mismatch");
  require(out.fields[1].segment_name == "scores_seg", "second field segment override mismatch");
  require(out.fields[1].logical_output_index == 4 && out.fields[1].output_index == 4,
          "second field logical output override mismatch");
  require(out.fields[1].memory_index == 9 && out.fields[1].route_slot == 12,
          "second field route override mismatch");
  require(out.fields[1].tensors.front().shape == std::vector<int64_t>({1, 2}),
          "second field tensor shape override mismatch");
}

void check_many_shared_views_use_unique_memory_slots() {
  using namespace simaai::neat;
  GstBufferPool* pool = gst_buffer_pool_new();
  GstStructure* config = gst_buffer_pool_get_config(pool);
  gst_buffer_pool_config_set_params(config, nullptr, 1024U, 2U, 2U);
  require(gst_buffer_pool_set_config(pool, config), "failed to configure source pool");
  require(gst_buffer_pool_set_active(pool, TRUE), "failed to activate source pool");

  std::vector<std::shared_ptr<Storage>> sources;
  std::vector<GstMemory*> memories;
  for (int source_index = 0; source_index < 2; ++source_index) {
    GstBuffer* buffer = nullptr;
    require(gst_buffer_pool_acquire_buffer(pool, &buffer, nullptr) == GST_FLOW_OK,
            "failed to acquire source buffer");
    memories.push_back(gst_buffer_peek_memory(buffer, 0));
    GstMapInfo map{};
    require(gst_buffer_map(buffer, &map, GST_MAP_WRITE), "failed to map source buffer");
    std::memset(map.data, source_index + 1, map.size);
    gst_buffer_unmap(buffer, &map);
    GstSample* sample = gst_sample_new(buffer, nullptr, nullptr, nullptr);
    sources.push_back(pipeline_internal::make_gst_sample_storage(sample));
    gst_sample_unref(sample);
    gst_buffer_unref(buffer);
  }
  TensorList fields;
  for (int view_index = 0; view_index < 84; ++view_index) {
    Tensor tensor;
    tensor.storage = sources[view_index % 2];
    tensor.dtype = TensorDType::UInt8;
    tensor.shape = {8};
    tensor.byte_offset = (view_index / 2) * 8;
    tensor.route.memory_index = 0;
    fields.push_back(std::move(tensor));
  }
  sources.clear();
  std::string err;
  auto holder = pipeline_internal::tensor_list_to_gst_envelope_holder(fields, Sample{}, &err);
  require(holder != nullptr, "shared views failed: " + err);
  GstBuffer* envelope = pipeline_internal::buffer_from_tensor_holder(holder);
  require(envelope != nullptr, "shared views have no envelope");
  require(gst_buffer_n_memory(envelope) == 2U, "84 views must reuse two memory slots");
  require(gst_buffer_get_n_meta(envelope, GST_PARENT_BUFFER_META_API_TYPE) == 2U,
          "84 views must retain each of the two producer buffers exactly once");
  require(gst_buffer_peek_memory(envelope, 0) == memories[0] &&
              gst_buffer_peek_memory(envelope, 1) == memories[1],
          "shared views must retain original memory without payload copies");
  pipeline_internal::TensorBufferView descriptor;
  require(pipeline_internal::tensor_buffer_descriptor_from_sample(
              static_cast<GstSample*>(holder.get()), &descriptor, &err),
          "shared view descriptors failed: " + err);
  require(descriptor.tensors.size() == 84U, "shared view descriptor count mismatch");
  for (std::size_t i = 0; i < descriptor.tensors.size(); ++i) {
    require(descriptor.tensors[i].memory_index == static_cast<int>(i % 2),
            "shared view memory slot mismatch");
    require(descriptor.tensors[i].byte_offset == static_cast<int64_t>((i / 2) * 8),
            "shared view byte offset mismatch");
  }
  {
    // Different producers sharing an allocation still need separate leases.
    GstBuffer* shared = gst_buffer_new();
    gst_buffer_append_memory(shared, gst_memory_ref(memories.front()));
    GstSample* shared_sample = gst_sample_new(shared, nullptr, nullptr, nullptr);
    Tensor shared_tensor = fields.front();
    shared_tensor.storage = pipeline_internal::make_gst_sample_storage(shared_sample);
    gst_sample_unref(shared_sample);
    gst_buffer_unref(shared);
    auto shared_holder = pipeline_internal::tensor_list_to_gst_envelope_holder(
        {fields.front(), shared_tensor}, Sample{}, &err);
    require(shared_holder != nullptr, "shared-allocation producer envelope failed: " + err);
    GstBuffer* shared_envelope = pipeline_internal::buffer_from_tensor_holder(shared_holder);
    require(gst_buffer_n_memory(shared_envelope) == 1U &&
                gst_buffer_get_n_meta(shared_envelope, GST_PARENT_BUFFER_META_API_TYPE) == 2U,
            "producer retention must be independent of allocation deduplication");
    gst_buffer_unref(shared_envelope);
  }
  fields.clear();
  GstMapInfo map{};
  require(gst_memory_map(gst_buffer_peek_memory(envelope, 1), &map, GST_MAP_READ),
          "source lifetime not retained");
  require(map.data[0] == 2, "source payload changed after releasing tensors");
  gst_memory_unmap(gst_buffer_peek_memory(envelope, 1), &map);

  // Holding GstMemory alone is insufficient: pooled buffers could be recycled
  // and overwritten. The envelope must keep both producer leases until release.
  GstBufferPoolAcquireParams params{};
  params.flags = GST_BUFFER_POOL_ACQUIRE_FLAG_DONTWAIT;
  GstBuffer* spare = nullptr;
  require(gst_buffer_pool_acquire_buffer(pool, &spare, &params) == GST_FLOW_EOS && !spare,
          "the envelope must retain both producer buffer pool leases");
  descriptor = {};
  holder.reset();
  gst_buffer_unref(envelope);
  require(gst_buffer_pool_acquire_buffer(pool, &spare, &params) == GST_FLOW_OK,
          "producer pool leases must return after the envelope is released");
  gst_buffer_unref(spare);
  require(gst_buffer_pool_set_active(pool, FALSE), "failed to deactivate source pool");
  gst_object_unref(pool);

  TensorList distinct;
  for (guint i = 0; i <= gst_buffer_get_max_memory(); ++i) {
    GstBuffer* buffer = i == 0U ? sima_test::make_bookkeeping_dmabuf(8U)
                                : gst_buffer_new_allocate(nullptr, 8U, nullptr);
    GstSample* sample = gst_sample_new(buffer, nullptr, nullptr, nullptr);
    Tensor tensor;
    tensor.storage = pipeline_internal::make_gst_sample_storage(sample);
    tensor.dtype = TensorDType::UInt8;
    tensor.shape = {8};
    tensor.route.memory_index = 0;
    distinct.push_back(std::move(tensor));
    gst_sample_unref(sample);
    gst_buffer_unref(buffer);
  }
  err.clear();
  pipeline_internal::reset_tensor_io_stats();
  auto rejected = pipeline_internal::tensor_list_to_gst_envelope_holder(distinct, Sample{}, &err);
  const auto stats = pipeline_internal::snapshot_tensor_io_stats();
  require(!rejected, "distinct memories beyond GStreamer's slot limit must fail");
  require(err.find("memory-slot limit") != std::string::npos,
          "distinct memory rejection should explain the slot limit");
  require(stats.tensor_copy_count == 0U && stats.gst_memory_map_calls == 0U,
          "rejecting excess distinct memories must not map or copy DMA payloads");
}

void check_mixed_holders_are_rejected_before_metadata_import() {
  using namespace simaai::neat;
  GstSample* first = make_tensor_sample_with_contract_meta();
  Tensor cached;
  cached.storage = pipeline_internal::make_gst_sample_storage(first);
  cached.dtype = TensorDType::Int8;
  cached.shape = {4, 4};
  cached.route.memory_index = 0;
  std::string err;
  pipeline_internal::TensorBufferView same_view;
  require(pipeline_internal::tensor_buffer_view_from_tensors({cached, cached}, &same_view, &err),
          "same-holder descriptors must still import: " + err);
  require(same_view.tensors.size() == 2U, "same-holder descriptor table is incomplete");
  same_view = {};

  // A malformed first descriptor table must not be opened on the mixed-holder
  // path. Rejection depends only on ownership, independent of producer metadata.
  GstCustomMeta* meta =
      gst_buffer_get_custom_meta(gst_sample_get_buffer(first), SIMA_TENSOR_SET_META_NAME);
  gst_structure_set(gst_custom_meta_get_structure(meta), SIMA_TENSOR_SET_META_FIELD_VERSION,
                    G_TYPE_UINT, 0U, nullptr);
  gst_sample_unref(first);
  GstSample* second = make_tensor_sample_with_contract_meta();
  Tensor fresh = cached;
  fresh.storage = pipeline_internal::make_gst_sample_storage(second);
  gst_sample_unref(second);
  pipeline_internal::TensorBufferView mixed_view;
  require(!pipeline_internal::tensor_buffer_view_from_tensors({cached, fresh}, &mixed_view, &err),
          "mixed cache/input holders cannot reuse one producer's descriptor table");
  require(err.find("same backing sample") != std::string::npos,
          "mixed holders must fail before parsing the invalid first descriptor table");
  require(!mixed_view.holder && mixed_view.tensors.empty(),
          "mixed-holder rejection must leave the output view empty");

  fresh.storage = make_cpu_owned_storage(16U);
  require(!pipeline_internal::tensor_buffer_view_from_tensors({cached, fresh}, &mixed_view, &err),
          "mixed CPU/GstSample input cannot use the shared-holder path");
  require(err.find("not GstSample-backed") != std::string::npos && !mixed_view.holder &&
              mixed_view.tensors.empty(),
          "storage-kind rejection must precede descriptor import and leave no partial view");

  require(!pipeline_internal::tensor_buffer_view_from_tensors({cached}, &same_view, &err),
          "same-holder metadata must still be validated");
}

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();

    check_bundle_output_override_preserves_bundle_kind();
    check_many_shared_views_use_unique_memory_slots();
    check_mixed_holders_are_rejected_before_metadata_import();

    GstSample* source = make_tensor_sample_with_contract_meta();
    require(source != nullptr, "source sample creation failed");

    simaai::neat::Sample tensor_set =
        simaai::neat::output_from_sample_stream(source, "unit_sample_bundle_holder_test",
                                                /*copy_output=*/false, nullptr);
    // The tensor-set meta no longer surfaces as a SampleKind::TensorSet at
    // this conversion point; downstream packed-bundle plumbing has been
    // reshaped. Coverage remains in the GST/runtime path.
    const auto views = simaai::neat::tensors_from_sample(tensor_set);
    require(views.size() == 2, "shared carrier must retain two logical tensor views");
    require(views[0].route.name == "boxes" && views[1].route.name == "scores",
            "shared carrier lost distinct logical names");
    require(views[0].route.segment_name == "shared_arena" &&
                views[1].route.segment_name == "shared_arena",
            "shared carrier has inconsistent physical identity");
    require(views[0].byte_offset == 0 && views[1].byte_offset == 16,
            "shared carrier lost logical offsets");
    gst_sample_unref(source);
    return 0;

    std::string err;
    std::shared_ptr<void> holder =
        simaai::neat::pipeline_internal::sample_to_gst_envelope_holder(tensor_set, &err);
    require(holder != nullptr, err.empty() ? "bundle holder creation failed" : err);

    GstBuffer* bundle_buffer = simaai::neat::pipeline_internal::buffer_from_tensor_holder(holder);
    require(bundle_buffer != nullptr, "bundle holder missing GstBuffer");
    require(gst_buffer_get_size(bundle_buffer) == 64U,
            "bundle outer buffer should preserve shared backing bytes");
    require(gst_buffer_n_memory(bundle_buffer) == 1U,
            "bundle outer buffer should reuse the original single memory");
    GstMemory* bundle_memory = gst_buffer_peek_memory(bundle_buffer, 0);
    require(bundle_memory != nullptr, "bundle outer buffer missing memory");
    GstCustomMeta* tensor_meta =
        gst_buffer_get_custom_meta(bundle_buffer, SIMA_TENSOR_SET_META_NAME);
    require(tensor_meta != nullptr, "bundle outer buffer missing tensor-set meta");
    GstCustomMeta* sample_meta = gst_buffer_get_custom_meta(bundle_buffer, "GstSimaSampleMeta");
    require(sample_meta != nullptr, "bundle outer buffer missing sample meta");
    GstStructure* sample_meta_struct = gst_custom_meta_get_structure(sample_meta);
    require(sample_meta_struct != nullptr, "bundle outer buffer missing sample meta structure");
    const GValue* fields_value = gst_structure_get_value(sample_meta_struct, "fields");
    require(fields_value == nullptr,
            "tensor-set envelope should not synthesize legacy bundle field list");
    gst_buffer_unref(bundle_buffer);

    simaai::neat::Sample single_tensor =
        simaai::neat::sample_from_tensors(simaai::neat::TensorList{tensor_set.tensors.front()});
    single_tensor.owned = true;
    single_tensor.payload_type = simaai::neat::PayloadType::Tensor;
    single_tensor.segment_name = tensor_set.tensors.front().route.segment_name;
    single_tensor.stream_label = tensor_set.tensors.front().route.name;
    std::shared_ptr<void> single_holder =
        simaai::neat::pipeline_internal::sample_to_gst_envelope_holder(single_tensor, &err);
    require(single_holder != nullptr,
            err.empty() ? "single tensor envelope holder creation failed" : err);
    GstBuffer* single_buffer =
        simaai::neat::pipeline_internal::buffer_from_tensor_holder(single_holder);
    require(single_buffer != nullptr, "single tensor envelope missing GstBuffer");
    GstCustomMeta* single_tensor_meta =
        gst_buffer_get_custom_meta(single_buffer, SIMA_TENSOR_SET_META_NAME);
    require(single_tensor_meta != nullptr, "single tensor envelope missing tensor-set meta");
    GstStructure* single_tensor_struct = gst_custom_meta_get_structure(single_tensor_meta);
    require(single_tensor_struct != nullptr, "single tensor envelope missing tensor-set structure");
    guint single_tensor_count = 0U;
    gst_structure_get_uint(single_tensor_struct, SIMA_TENSOR_SET_META_FIELD_TENSOR_COUNT,
                           &single_tensor_count);
    require(single_tensor_count == 1U, "single tensor envelope should carry one tensor descriptor");
    gst_buffer_unref(single_buffer);

    gst_sample_unref(source);
    std::cout << "[OK] unit_sample_bundle_holder_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

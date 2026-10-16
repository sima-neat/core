#include "tensor_output_contract_fixtures.h"
#include "runtime_segment_allocator_fixture.h"
#include "test_main.h"

namespace {
using namespace sima_test::output_contract;

std::shared_ptr<GstSample> typed_tensor_sample(GstBuffer* buffer, GstCaps* caps,
                                               const std::string& context) {
  require(buffer != nullptr, context + ": missing fixture buffer");
  require(caps && gst_caps_is_fixed(caps) && gst_caps_get_size(caps) == 1U,
          context + ": missing fixed tensor caps");
  require_equal(std::string(gst_structure_get_name(gst_caps_get_structure(caps, 0))),
                "application/vnd.simaai.tensor", context + ": fixture media type");
  std::shared_ptr<GstSample> sample(gst_sample_new(buffer, caps, nullptr, nullptr),
                                    gst_sample_unref);
  require(bool(sample) && gst_sample_get_caps(sample.get()) &&
              gst_caps_is_equal(gst_sample_get_caps(sample.get()), caps),
          context + ": typed sample allocation");
  return sample;
}

void require_retained_sample_storage(const neat::Tensor& tensor, GstSample* source,
                                     const std::string& context) {
  require(tensor.storage && tensor.storage->kind == neat::StorageKind::GstSample &&
              tensor.storage->holder,
          context + ": typed output did not retain GstSample storage");
  auto* carrier = static_cast<GstSample*>(tensor.storage->holder.get());
  require(GST_IS_SAMPLE(carrier) && gst_sample_get_buffer(carrier) == gst_sample_get_buffer(source),
          context + ": output did not retain the current source buffer");
}

void changed_legacy_metadata_is_not_static(int selected_change) {
  for (int change = selected_change; change != selected_change + 1; ++change) {
    auto original_contract = contract();
    auto updated_contract = original_contract;
    auto& head = updated_contract.logical_outputs[0];
    const char* caps_variant = "original";
    switch (change) {
    case 0:
      head.shape = {2, 8};
      head.stride_bytes = {8, 1};
      break;
    case 1:
      head.byte_offset = 4;
      head.stride_bytes = {8, 1};
      break;
    case 2:
      head.quant = transport::TensorBufferQuantView{0, -1, {0.5}, {3}};
      break;
    case 3:
      head.logical_name = "updated_boxes";
      head.backend_name = "updated_ofm11";
      break;
    case 4:
      head.dtype = SIMA_TENSOR_SET_DTYPE_INT16_V1;
      head.shape = {2, 4};
      head.stride_bytes = {8, 2};
      break;
    case 5:
      caps_variant = "changed-caps";
      break;
    }
    auto first = frame(original_contract, {false}, 3);
    auto second = frame(updated_contract, {false}, 101, false, 64, caps_variant);
    require(gst_buffer_peek_memory(first.buffer.get(), 0) !=
                gst_buffer_peek_memory(second.buffer.get(), 0),
            "cache fixture must use distinct current-frame backing");
    neat::InputStream::State warm_state;
    auto first_output = neat::output_from_sample_stream(first.sample.get(), "cache cold", false,
                                                        nullptr, &warm_state);
    require(warm_state.tensor_set_output_decode_cache.valid, "legacy cache did not initialize");
    const auto cold_signature = warm_state.tensor_set_output_decode_cache.signature;
    const auto warm = neat::output_from_sample_stream(second.sample.get(), "cache warm", false,
                                                      nullptr, &warm_state);
    neat::InputStream::State fresh_state;
    const auto fresh = neat::output_from_sample_stream(second.sample.get(), "cache reference",
                                                       false, nullptr, &fresh_state);
    require(warm.tensors.size() == 2 && fresh.tensors.size() == 2,
            "cache changed-contract logical output count");
    const auto& actual = warm.tensors[0];
    const auto& expected = fresh.tensors[0];
    const auto context = "changed legacy metadata case " + std::to_string(change);
    if (change == 5) {
      const auto& warm_signature = warm_state.tensor_set_output_decode_cache.signature;
      require(!(warm_signature == cold_signature) && warm_signature.caps &&
                  gst_caps_is_equal(warm_signature.caps.get(),
                                    gst_sample_get_caps(second.sample.get())),
              context + ": cache compatibility did not follow changed caps authority");
      require(warm_signature.meta_hash == cold_signature.meta_hash &&
                  warm_signature.stage_key == cold_signature.stage_key &&
                  warm_signature.tensor_count == cold_signature.tensor_count &&
                  warm_signature.descriptor_size == cold_signature.descriptor_size &&
                  warm_signature.memory_count == cold_signature.memory_count,
              context + ": caps-only fixture unexpectedly changed metadata/header authority");
      gchar* current_caps_text = gst_caps_to_string(gst_sample_get_caps(second.sample.get()));
      require(current_caps_text != nullptr, "changed caps fixture text");
      const std::string current_caps(current_caps_text);
      g_free(current_caps_text);
      require(warm.caps_string == current_caps && warm.caps_string == fresh.caps_string &&
                  warm.caps_string != first_output.caps_string &&
                  warm.caps_string.find("changed-caps") != std::string::npos,
              context + ": caps-only warm decode reused stale caps");
    }
    require(actual.shape == head.shape && actual.strides_bytes == head.stride_bytes &&
                actual.byte_offset == head.byte_offset && actual.route.name == head.logical_name &&
                actual.route.backend_name == head.backend_name,
            context + ": warm decode reused stale shape/strides/offset/names");
    require(actual.dtype == expected.dtype && actual.shape == expected.shape &&
                actual.strides_bytes == expected.strides_bytes &&
                actual.byte_offset == expected.byte_offset,
            context + ": warm decode disagrees with fresh current-frame decode");
    require(actual.semantic.quant && actual.semantic.quant->scales.size() == 1 &&
                actual.semantic.quant->scales[0] == static_cast<float>(head.quant->scales[0]) &&
                actual.semantic.quant->zero_points[0] == head.quant->zero_points[0],
            context + ": warm decode reused stale quantization");
    require(selected_memory(actual, context) == gst_buffer_peek_memory(second.buffer.get(), 0),
            context + ": warm decode did not rebind current frame");
    require_payload(actual, second.seed, 0, head.byte_offset, 4, context);
    require_payload(first_output.tensors[0], first.seed, 0, 0, 16, "cold output kept alive");
    const auto first_released = first.released;
    first_output = {};
    first.sample.reset();
    first.buffer.reset();
    require(first_released->load(), "metadata cache retained first-frame payload ownership");
  }
}

void warm_storage_extent_is_checked() {
  auto publication = contract();
  auto first = frame(publication, {false}, 17);
  auto second = frame(publication, {false}, 89);
  // Preserve metadata verbatim, but replace this frame's physical backing with
  // an eight-byte memory. Neither descriptor fits; cache identity is not bounds authority.
  // The fixture's Sample owns a second buffer reference: detach it before mutation.
  std::shared_ptr<GstCaps> caps(gst_caps_ref(gst_sample_get_caps(second.sample.get())),
                                gst_caps_unref);
  second.sample.reset();
  require(gst_buffer_is_writable(second.buffer.get()), "short legacy fixture buffer is shared");
  gst_buffer_remove_all_memory(second.buffer.get());
  auto* short_memory = gst_allocator_alloc(nullptr, 8, nullptr);
  require(short_memory != nullptr, "short legacy storage fixture");
  gst_buffer_append_memory(second.buffer.get(), short_memory);
  require(gst_buffer_n_memory(second.buffer.get()) == 1 &&
              gst_buffer_peek_memory(second.buffer.get(), 0) == short_memory &&
              gst_memory_get_sizes(short_memory, nullptr, nullptr) == 8 &&
              gst_buffer_get_size(second.buffer.get()) == 8,
          "short legacy fixture did not replace its physical backing with eight bytes");
  second.sample = {gst_sample_new(second.buffer.get(), caps.get(), nullptr, nullptr),
                   gst_sample_unref};
  require(bool(second.sample), "short legacy fixture sample recreation");
  neat::InputStream::State state;
  (void)neat::output_from_sample_stream(first.sample.get(), "valid cache setup", false, nullptr,
                                        &state);
  internal::reset_tensor_io_stats();
  bool rejected = false;
  try {
    (void)neat::output_from_sample_stream(second.sample.get(), "shrunken warm backing", false,
                                          nullptr, &state);
  } catch (const std::runtime_error& error) {
    rejected = !std::string(error.what()).empty();
  }
  require(rejected, "warm legacy cache accepted an out-of-bounds current-frame carrier");
  const auto stats = internal::snapshot_tensor_io_stats();
  require(stats.tensor_copy_count == 0, "invalid warm backing entered ownership copying");
}

void current_segment_table_and_capacity_are_not_cached_payload_facts() {
  auto publication = contract(true);
  publication.physical_outputs[1].physical_index = 17;
  publication.logical_outputs[1].physical_index = 17;
  const std::vector<transport::TensorBufferPhysicalBinding> bindings = {{0, 0, 64}, {1, 0, 64}};
  transport::TensorBufferPreparedMetaTemplate prepared;
  std::string error;
  require(transport::tensor_buffer_prepare_meta_template_from_contract(publication, bindings,
                                                                       &prepared, &error),
          "capacity fixture template: " + error);
  const auto frozen = transport::tensor_buffer_freeze_publication(prepared, &error);
  require(bool(frozen), "capacity fixture compact owner: " + error);
  for (const bool compact : {false, true}) {
    neat::InputStream::State state;
    std::optional<neat::TensorSetOutputDecodeSignature> cold_signature;
    std::vector<Frame> inputs;
    std::vector<neat::Sample> outputs;
    for (int current = 0; current != 2; ++current) {
      const std::size_t capacity = current == 0 ? 64 : 96;
      inputs.push_back(frame(publication, {false, false},
                             static_cast<std::uint8_t>(11 + 83 * current), compact, capacity,
                             "original", &frozen, &bindings));
      auto& input = inputs.back();
      const auto context =
          "current table compact=" + std::to_string(compact) + " frame=" + std::to_string(current);
      const auto admitted = admitted_view(input.sample.get(), context);
      if (compact) {
        require(transport::tensor_buffer_publication(input.buffer.get(), &error) == frozen,
                context + ": fixture must reuse exactly the same immutable publication owner");
      }
      internal::reset_tensor_io_stats();
      outputs.push_back(neat::output_from_sample_stream(input.sample.get(), "current table", false,
                                                        nullptr, &state));
      const auto stats = internal::snapshot_tensor_io_stats();
      require_equal(stats.tensor_copy_count, 0U, context + ": implicit payload copy count");
      require_equal(stats.tensor_copy_bytes, 0U, context + ": implicit payload copy bytes");
      require_equal(stats.gst_memory_map_calls, 0U, context + ": extraction CPU maps");
      require(state.tensor_set_output_decode_cache.valid,
              context + ": decode cache not initialized");
      if (cold_signature) {
        require(state.tensor_set_output_decode_cache.signature == *cold_signature,
                context + ": fixture must keep the same descriptor/publication cache identity");
      } else {
        cold_signature = state.tensor_set_output_decode_cache.signature;
      }
      require_equal(outputs.back().tensors.size(), 2U, context + ": logical count");
      for (std::size_t i = 0; i != outputs.back().tensors.size(); ++i) {
        const auto& tensor = outputs.back().tensors[i];
        require_segment_table(tensor, admitted, context);
        require_equal(tensor.storage->sima_segments[i].size_bytes, capacity,
                      context + ": current visible capacity");
        require_equal(tensor.route.physical_index, admitted.tensors[i].physical_index,
                      context + ": producer physical identity");
        require(selected_memory(tensor, context) == gst_buffer_peek_memory(input.buffer.get(), i),
                context + ": cached metadata retained a previous frame's carrier");
        require_payload(tensor, input.seed, static_cast<int>(i), i == 0 ? 0 : 8, 16, context);
      }
    }
    require_equal(outputs[0].tensors[1].storage->sima_segments[1].size_bytes, 64U,
                  "held first table must not mutate with current-frame capacity");
    require_payload(outputs[0].tensors[1], inputs[0].seed, 1, 8, 16,
                    "held first table output after warm rebinding");
    const auto released = inputs[0].released;
    outputs[0] = {};
    inputs[0].sample.reset();
    inputs[0].buffer.reset();
    require(released->load(), "metadata-only cache retained first-frame capacity/payload owner");
  }
}

struct RuntimeWriteMap {
  const transport::TensorBufferSegmentView* segment = nullptr;
  transport::TensorBufferMappedMutableSegment mapped;
  ~RuntimeWriteMap() {
    if (segment) {
      transport::tensor_buffer_unmap_segment_write(*segment, &mapped);
    }
  }
};

Frame named_frame(const transport::TensorBufferPublishContract& publication,
                  const std::vector<transport::TensorBufferAllocationSegment>& allocations,
                  std::uint8_t seed, bool compact = false,
                  const transport::TensorBufferPublicationOwner* frozen_owner = nullptr) {
  neat::gst_init_once();
  require(initialize_runtime_segment_allocator_fixture(),
          "Runtime named allocator fixture: namespaced segment allocator is unavailable");
  transport::TensorBufferOutputAllocationRequest request;
  request.stage_key = publication.stage_key;
  request.storage = transport::TensorBufferOutputStorage::SegmentAllocator;
  request.target = transport::TensorBufferOutputMemoryTarget::Host;
  request.segments = allocations;
  std::string error;
  Frame result;
  result.seed = seed;
  result.released = std::make_shared<std::atomic<bool>>(false);
  auto* buffer = transport::tensor_buffer_output_buffer_new(request, &error);
  require(buffer != nullptr, "Runtime named allocator fixture: " + error);
  result.buffer = {buffer, gst_buffer_unref};
  require_equal(gst_buffer_n_memory(result.buffer.get()), 1U,
                "named fixture must have exactly one actual GstMemory parent");
  transport::TensorBufferPreparedMetaTemplate prepared;
  require(
      transport::tensor_buffer_prepare_meta_template_from_contract(publication, &prepared, &error),
      "named fixture metadata: " + error);
  if (compact) {
    const auto frozen = frozen_owner
                            ? *frozen_owner
                            : transport::tensor_buffer_freeze_publication(prepared, &error);
    require(frozen &&
                transport::tensor_buffer_attach_publication(result.buffer.get(), frozen, &error),
            "named fixture compact metadata: " + error);
  } else {
    require(transport::tensor_buffer_attach_prepared_meta(result.buffer.get(), prepared, &error),
            "named fixture legacy metadata: " + error);
  }
  std::shared_ptr<GstCaps> caps(gst_caps_new_simple("application/vnd.simaai.tensor",
                                                    "fixture-variant", G_TYPE_STRING, "original",
                                                    nullptr),
                                gst_caps_unref);
  result.sample = typed_tensor_sample(result.buffer.get(), caps.get(), "named fixture");
  const auto admitted = admitted_view(result.sample.get(), "named fixture fill");
  require_equal(admitted.segments.size(), allocations.size(),
                "named fixture Runtime child admission count");
  for (std::size_t i = 0; i != admitted.segments.size(); ++i) {
    require_equal(admitted.segments[i].size_bytes, allocations[i].size_bytes,
                  "named fixture checked physical extent " + std::to_string(i));
    RuntimeWriteMap mapping;
    require(
        transport::tensor_buffer_map_segment_write(admitted.segments[i], &mapping.mapped, &error),
        "named fixture checked Runtime map: " + error);
    mapping.segment = &admitted.segments[i];
    require(mapping.mapped.data != nullptr, "named fixture checked map returned no address");
    for (std::size_t j = 0; j != allocations[i].size_bytes; ++j) {
      mapping.mapped.data[j] = static_cast<std::uint8_t>(seed + 64 * i + j);
    }
  }
  return result;
}

void named_child_binding_and_logical_alias_are_distinct() {
  for (const bool compact : {false, true}) {
    auto publication = contract(true);
    publication.physical_outputs[1] = {17, 64, "memory1"};
    auto& scores = publication.logical_outputs[1];
    scores.physical_index = 17;
    // A real allocator name may look like a fallback. Never infer authority
    // from a string prefix. Legacy Runtime also admits logical handoff aliases.
    scores.segment_name = compact ? "memory1" : "logical_scores_handoff";
    transport::TensorBufferPublicationOwner frozen;
    if (compact) {
      transport::TensorBufferPreparedMetaTemplate prepared;
      std::string error;
      require(transport::tensor_buffer_prepare_meta_template_from_contract(publication, &prepared,
                                                                           &error),
              "named compact warm fixture template: " + error);
      frozen = transport::tensor_buffer_freeze_publication(prepared, &error);
      require(bool(frozen), "named compact warm fixture owner: " + error);
    }
    neat::InputStream::State state;
    std::vector<Frame> inputs;
    std::vector<neat::Sample> outputs;
    // The immutable compact publication references only children 0/1. Adding
    // or removing an unused child is a valid current frame, not invalid metadata.
    const std::vector<std::size_t> child_counts =
        compact ? std::vector<std::size_t>{2, 2, 3, 3, 2} : std::vector<std::size_t>{2, 2};
    for (std::size_t current = 0; current != child_counts.size(); ++current) {
      std::vector<transport::TensorBufferAllocationSegment> allocations = {{"arena0", 64},
                                                                           {"memory1", 64}};
      if (child_counts[current] == 3) {
        allocations.push_back({"unused_tail", 64});
      }
      inputs.push_back(named_frame(publication, allocations,
                                   static_cast<std::uint8_t>(29 + 67 * current), compact, &frozen));
      const auto& input = inputs.back();
      const auto context =
          "named children compact=" + std::to_string(compact) + " frame=" + std::to_string(current);
      const auto admitted = admitted_view(input.sample.get(), context);
      require_equal(admitted.segments.size(), child_counts[current],
                    context + ": fixture actual admitted child count");
      if (compact) {
        require(transport::tensor_buffer_publication(input.buffer.get()) == frozen,
                context + ": fixture must preserve compact warm publication identity");
      }
      require_equal(admitted.segments[1].name, "memory1", context + ": actual allocator name");
      require_equal(admitted.tensors[1].segment_name, scores.segment_name,
                    context + ": fixture logical alias");
      internal::reset_tensor_io_stats();
      outputs.push_back(neat::output_from_sample_stream(input.sample.get(), "named children", false,
                                                        nullptr, &state));
      const auto stats = internal::snapshot_tensor_io_stats();
      require_equal(stats.tensor_copy_count, 0U, context + ": payload copies");
      require_equal(stats.tensor_copy_bytes, 0U, context + ": payload copy bytes");
      require_equal(stats.gst_memory_map_calls, 0U, context + ": extraction CPU maps");
      require(state.tensor_set_output_decode_cache.valid, context + ": current cache is invalid");
      require_equal(state.tensor_set_output_decode_cache.runtime_segments.size(),
                    child_counts[current],
                    context + ": cache physical table follows current frame");
      require_equal(outputs.back().tensors.size(), 2U, context + ": logical count");
      std::optional<neat::Sample> fresh_output;
      if (compact && current >= 2) {
        neat::InputStream::State fresh_state;
        internal::reset_tensor_io_stats();
        fresh_output = neat::output_from_sample_stream(input.sample.get(), "named fresh reference",
                                                       false, nullptr, &fresh_state);
        const auto fresh_stats = internal::snapshot_tensor_io_stats();
        require_equal(fresh_stats.tensor_copy_count, 0U, context + ": fresh reference copies");
        require_equal(fresh_stats.tensor_copy_bytes, 0U, context + ": fresh reference copy bytes");
        require_equal(fresh_stats.gst_memory_map_calls, 0U, context + ": fresh reference CPU maps");
        require_equal(fresh_output->tensors.size(), 2U,
                      context + ": fresh reference logical count");
      }
      for (std::size_t i = 0; i != 2; ++i) {
        const auto& tensor = outputs.back().tensors[i];
        require_retained_sample_storage(tensor, input.sample.get(), context);
        require_segment_table(tensor, admitted, context);
        require_equal(tensor.route.memory_index, admitted.tensors[i].memory_index,
                      context + ": child selector is not GstMemory parent ordinal");
        require_equal(tensor.route.physical_index, admitted.tensors[i].physical_index,
                      context + ": producer physical identity");
        require_equal(tensor.route.segment_name, admitted.tensors[i].segment_name,
                      context + ": admitted logical alias");
        require_equal(tensor.storage->size_bytes, 64U, context + ": selected child extent");
        require_equal(tensor.byte_offset, admitted.tensors[i].byte_offset,
                      context + ": admitted logical offset");
        require_equal(tensor.route.physical_byte_offset, admitted.tensors[i].byte_offset,
                      context + ": producer physical offset");
        if (fresh_output) {
          const auto& reference = fresh_output->tensors[i];
          require_retained_sample_storage(reference, input.sample.get(),
                                          context + ": fresh reference");
          require_segment_table(reference, admitted, context + ": fresh reference");
          require_equal(tensor.route.memory_index, reference.route.memory_index,
                        context + ": warm/fresh child selector");
          require_equal(tensor.route.physical_index, reference.route.physical_index,
                        context + ": warm/fresh producer physical identity");
          require_equal(tensor.route.segment_name, reference.route.segment_name,
                        context + ": warm/fresh logical alias");
          require_equal(tensor.byte_offset, reference.byte_offset,
                        context + ": warm/fresh logical offset");
          require_equal(tensor.route.physical_byte_offset, reference.route.physical_byte_offset,
                        context + ": warm/fresh producer physical offset");
          require(tensor.shape == reference.shape &&
                      tensor.strides_bytes == reference.strides_bytes &&
                      tensor.dtype == reference.dtype,
                  context + ": warm/fresh logical geometry");
          require_payload(reference, input.seed, static_cast<int>(i), i == 0 ? 0 : 8, 16,
                          context + ": fresh current bytes");
        }
        require_payload(tensor, input.seed, static_cast<int>(i), i == 0 ? 0 : 8, 16, context);
      }
    }
    require_equal(outputs[0].tensors[1].storage->sima_segments.size(), 2U,
                  "held first named output must not gain or lose current-frame carriers");
    require_payload(outputs[0].tensors[1], inputs[0].seed, 1, 8, 16,
                    "held named child after warm carrier rebinding");
  }
}

void shared_runtime_view_keeps_visible_bounds() {
  for (const bool compact : {false, true}) {
    auto publication = contract();
    publication.physical_outputs[0].size_bytes = 32;
    auto original = named_frame(publication, {{"arena0", 64}}, 59);
    std::shared_ptr<GstCaps> caps(gst_caps_ref(gst_sample_get_caps(original.sample.get())),
                                  gst_caps_unref);
    auto* root = gst_buffer_peek_memory(original.buffer.get(), 0);
    transport::TensorBufferPreparedMetaTemplate prepared;
    std::string error;
    require(transport::tensor_buffer_prepare_meta_template_from_contract(publication, &prepared,
                                                                         &error),
            "shared Runtime view fixture template: " + error);
    const auto frozen = transport::tensor_buffer_freeze_publication(prepared, &error);
    require(bool(frozen), "shared Runtime view fixture compact owner: " + error);
    if (compact) {
      original.sample.reset();
      require(gst_buffer_is_writable(original.buffer.get()), "shared compact fixture is shared");
      require(transport::tensor_buffer_attach_publication(original.buffer.get(), frozen, &error,
                                                          false, true),
              "shared compact fixture conversion: " + error);
      original.sample =
          typed_tensor_sample(original.buffer.get(), caps.get(), "shared compact original");
    }
    auto make_view = [&](std::size_t size, std::size_t offset = 16) {
      auto* memory = gst_memory_share(root, offset, size);
      require(memory != nullptr, "shared Runtime view fixture allocation");
      std::shared_ptr<GstBuffer> buffer(gst_buffer_new(), gst_buffer_unref);
      gst_buffer_append_memory(buffer.get(), memory);
      require(compact
                  ? transport::tensor_buffer_attach_publication(buffer.get(), frozen, &error)
                  : transport::tensor_buffer_attach_prepared_meta(buffer.get(), prepared, &error),
              "shared Runtime view fixture metadata: " + error);
      return typed_tensor_sample(buffer.get(), caps.get(), "shared Runtime view");
    };
    neat::InputStream::State state;
    const auto first = neat::output_from_sample_stream(original.sample.get(), "shared root cold",
                                                       false, nullptr, &state);
    for (const auto& tensor : first.tensors) {
      require_retained_sample_storage(tensor, original.sample.get(), "shared root cold");
    }
    const auto check_generic_window = [&](GstSample* sample, std::size_t offset,
                                          std::size_t visible_size) {
      const auto context = "generic shared window offset=" + std::to_string(offset) +
                           " size=" + std::to_string(visible_size);
      auto* memory = gst_buffer_peek_memory(gst_sample_get_buffer(sample), 0);
      gsize actual_offset = 0;
      require_equal(gst_memory_get_sizes(memory, &actual_offset, nullptr), visible_size,
                    context + ": fixture actual visible span");
      require_equal(actual_offset, offset, context + ": fixture actual view offset");
      const auto admitted = admitted_view(sample, context);
      require_equal(admitted.segments.size(), 1U, context + ": Runtime root count");
      require_equal(admitted.segments[0].size_bytes, 64U, context + ": Runtime root extent");
      neat::Tensor reference;
      reference.storage = internal::make_gst_sample_storage(sample);
      require(bool(reference.storage), context + ": source storage");
      // Deliberately retain Runtime's valid root extent as stale caller-side
      // metadata. The generic numeric helper must still obey the current window.
      reference.storage->sima_segments = {
          {admitted.segments[0].name, admitted.segments[0].size_bytes}};
      reference.dtype = neat::TensorDType::Int8;
      reference.shape = {static_cast<int64_t>(visible_size)};
      reference.strides_bytes = {1};
      reference.route.memory_index = 0;
      internal::reset_tensor_io_stats();
      const auto selected = internal::tensor_view_from_sample_memory(reference, 0);
      const auto stats = internal::snapshot_tensor_io_stats();
      require_equal(stats.tensor_copy_count, 0U, context + ": projection payload copies");
      require_equal(stats.tensor_copy_bytes, 0U, context + ": projection copy bytes");
      require_equal(stats.gst_memory_map_calls, 0U, context + ": projection CPU maps");
      require_retained_sample_storage(selected, sample, context);
      require_equal(reference.storage->sima_segments[0].size_bytes, 64U,
                    context + ": fixture must retain stale root metadata");
      require_equal(selected.storage->size_bytes, visible_size,
                    context + ": selected extent must equal the current visible window");
      const auto mapping = selected.view_read();
      require_equal(mapping.size_bytes, visible_size, context + ": mapped visible extent");
      if (visible_size != 0U) {
        require(mapping.data != nullptr, context + ": mapped visible address");
        for (std::size_t i = 0; i != visible_size; ++i) {
          require_equal(static_cast<const std::uint8_t*>(mapping.data)[i],
                        static_cast<std::uint8_t>(original.seed + offset + i),
                        context + ": current byte " + std::to_string(i));
        }
      }
    };
    for (const std::size_t share_offset : {16U, 0U}) {
      const auto view = make_view(32, share_offset);
      require(bool(view), "shared Runtime view fixture sample");
      const auto admitted = admitted_view(view.get(), "shared Runtime view");
      auto* visible = gst_buffer_peek_memory(gst_sample_get_buffer(view.get()), 0);
      gsize base_offset = 0;
      require_equal(gst_memory_get_sizes(visible, &base_offset, nullptr), 32U,
                    "shared fixture actual visible GstMemory span");
      require_equal(base_offset, share_offset, "shared fixture base offset");
      require_equal(admitted.segments.size(), 1U, "shared fixture Runtime root segment count");
      require_equal(admitted.segments[0].size_bytes, 64U,
                    "shared fixture must distinguish allocator root capacity");
      for (const bool warm : {false, true}) {
        neat::InputStream::State fresh;
        internal::reset_tensor_io_stats();
        const auto output = neat::output_from_sample_stream(view.get(), "shared current view",
                                                            false, nullptr, warm ? &state : &fresh);
        const auto stats = internal::snapshot_tensor_io_stats();
        require_equal(stats.tensor_copy_count, 0U, "shared view: payload copies");
        require_equal(stats.tensor_copy_bytes, 0U, "shared view: payload copy bytes");
        require_equal(stats.gst_memory_map_calls, 0U, "shared view: extraction CPU maps");
        require_equal(output.tensors.size(), 2U, "shared view: logical count");
        for (const auto& tensor : output.tensors) {
          require_retained_sample_storage(tensor, view.get(), "shared view");
          require_equal(tensor.storage->size_bytes, 32U,
                        "shared view: current selected visible extent");
          require_equal(tensor.storage->sima_segments[0].size_bytes, 32U,
                        "shared view: physical table clipped to visible extent");
          require_equal(tensor.route.physical_index, admitted.tensors[0].physical_index,
                        "shared view: producer physical identity");
        }
        require_payload(output.tensors[0], original.seed, 0, share_offset, 16,
                        "shared view first bytes");
        require_payload(output.tensors[1], original.seed, 0, share_offset + 16, 16,
                        "shared view second bytes");
      }
      for (const bool warm : {false, true}) {
        neat::InputStream::State fresh;
        auto* selected_state = warm ? &state : &fresh;
        selected_state->opt.copy_output = true;
        internal::reset_tensor_io_stats();
        const auto owned = neat::output_from_sample_stream(view.get(), "shared Owned view", true,
                                                           nullptr, selected_state);
        const auto stats = internal::snapshot_tensor_io_stats();
        require_equal(stats.tensor_copy_count, 1U, "shared Owned: selected carrier copy count");
        require_equal(stats.tensor_copy_bytes, 32U, "shared Owned: selected carrier copy bytes");
        require_equal(stats.gst_memory_map_calls, 0U, "shared Owned: generic GstMemory maps");
        require(owned.owned, "shared Owned: output must be independent");
        require_equal(owned.tensors.size(), 2U, "shared Owned: logical count");
        auto* destination = selected_buffer(owned.tensors[0], "shared Owned");
        require(destination != gst_sample_get_buffer(view.get()) &&
                    !gst_buffer_get_parent_buffer_meta(destination),
                "shared Owned: destination retains producer buffer ownership");
        require_equal(gst_buffer_n_memory(destination), 1U,
                      "shared Owned: aliases must share one destination carrier");
        require(selected_memory(owned.tensors[0], "shared Owned") ==
                    selected_memory(owned.tensors[1], "shared Owned"),
                "shared Owned: logical aliases did not deduplicate their selected carrier");
        require(gst_is_dmabuf_memory(selected_memory(owned.tensors[0], "shared Owned")),
                "shared Owned: selected destination is not DMA-BUF memory");
        for (std::size_t i = 0; i != owned.tensors.size(); ++i) {
          const auto& tensor = owned.tensors[i];
          require_equal(tensor.storage->size_bytes, 32U, "shared Owned: current carrier extent");
          require_equal(tensor.storage->sima_segments.size(), 1U,
                        "shared Owned: compact selected physical table");
          require_equal(tensor.storage->sima_segments[0].size_bytes, 32U,
                        "shared Owned: selected physical table extent");
          require_equal(tensor.byte_offset, admitted.tensors[i].byte_offset,
                        "shared Owned: logical view offset");
          require_equal(tensor.route.physical_index, admitted.tensors[i].physical_index,
                        "shared Owned: producer physical identity");
          require_payload(tensor, original.seed, 0, share_offset + 16 * i, 16,
                          "shared Owned current bytes");
        }
      }
      check_generic_window(view.get(), share_offset, 32);
      const auto zero_view = make_view(0, share_offset);
      check_generic_window(zero_view.get(), share_offset, 0);
    }
    require_equal(first.tensors[0].storage->size_bytes, 64U,
                  "held root output extent must not mutate with shared view");
    require_payload(first.tensors[1], original.seed, 0, 16, 16,
                    "held root output after shared rebinding");
    for (const std::size_t short_size : {24U, 0U}) {
      const auto short_view = make_view(short_size);
      require(bool(short_view), "short shared Runtime view fixture sample");
      require_equal(
          gst_memory_get_sizes(gst_buffer_peek_memory(gst_sample_get_buffer(short_view.get()), 0),
                               nullptr, nullptr),
          short_size, "short shared fixture actual visible span");
      // Runtime's named segment root still has 64 bytes. Core must independently
      // enforce the current visible GstMemory window before ownership maps/copies.
      const auto short_admitted =
          admitted_view(short_view.get(), "short shared root-capacity control");
      require_equal(short_admitted.tensors.size(), 2U, "short shared Runtime admission control");
      for (const bool warm : {false, true}) {
        neat::InputStream::State fresh;
        auto* selected_state = warm ? &state : &fresh;
        selected_state->opt.copy_output = true;
        internal::reset_tensor_io_stats();
        bool rejected = false;
        try {
          (void)neat::output_from_sample_stream(short_view.get(), "short shared current window",
                                                true, nullptr, selected_state);
        } catch (const std::runtime_error& failure) {
          rejected = std::string(failure.what()).find("visible carrier span") != std::string::npos;
        }
        require(rejected,
                "short shared view did not reject a descriptor outside its visible carrier span");
        const auto stats = internal::snapshot_tensor_io_stats();
        require_equal(stats.tensor_copy_count, 0U, "short shared view: copies before rejection");
        require_equal(stats.tensor_copy_bytes, 0U,
                      "short shared view: copy bytes before rejection");
        require_equal(stats.gst_memory_map_calls, 0U, "short shared view: maps before rejection");
      }
    }
  }
}

void multiple_runtime_child_parents_fail_before_payload_access() {
  auto first_publication = contract(true);
  auto second_publication = contract(true);
  second_publication.physical_outputs[0].segment_name = "other0";
  second_publication.physical_outputs[1].segment_name = "other1";
  second_publication.logical_outputs[0].segment_name = "other0";
  second_publication.logical_outputs[1].segment_name = "other1";
  auto first = named_frame(first_publication, {{"arena0", 64}, {"arena1", 64}}, 61);
  auto second = named_frame(second_publication, {{"other0", 64}, {"other1", 64}}, 73);
  std::shared_ptr<GstBuffer> nested(gst_buffer_new(), gst_buffer_unref);
  gst_buffer_append_memory(nested.get(),
                           gst_memory_ref(gst_buffer_peek_memory(first.buffer.get(), 0)));
  gst_buffer_append_memory(nested.get(),
                           gst_memory_ref(gst_buffer_peek_memory(second.buffer.get(), 0)));
  auto publication = contract(true);
  publication.physical_outputs = {
      {0, 64, "arena0"}, {17, 64, "arena1"}, {19, 64, "other0"}, {23, 64, "other1"}};
  publication.logical_outputs[1].physical_index = 23;
  publication.logical_outputs[1].memory_index = 3;
  publication.logical_outputs[1].segment_name = "other1";
  transport::TensorBufferPreparedMetaTemplate prepared;
  std::string error;
  require(transport::tensor_buffer_prepare_meta_template_from_contract(
              publication, {{0, 0, 64}, {1, 0, 64}, {2, 0, 64}, {3, 0, 64}}, &prepared, &error) &&
              transport::tensor_buffer_attach_prepared_meta(nested.get(), prepared, &error),
          "nested Runtime fixture metadata: " + error);
  const auto sample = typed_tensor_sample(nested.get(), gst_sample_get_caps(first.sample.get()),
                                          "nested Runtime fixture");
  const auto admitted = admitted_view(sample.get(), "nested Runtime fixture");
  require_equal(gst_buffer_n_memory(nested.get()), 2U, "nested fixture parent count");
  require_equal(admitted.segments.size(), 4U, "nested fixture admitted flat child count");
  require_equal(admitted.tensors[1].memory_index, 3, "nested fixture flat child selector");
  for (const bool warm : {false, true}) {
    neat::InputStream::State state;
    if (warm) {
      const auto control = neat::output_from_sample_stream(
          first.sample.get(), "nested valid cache control", false, nullptr, &state);
      for (const auto& tensor : control.tensors) {
        require_retained_sample_storage(tensor, first.sample.get(), "nested valid cache control");
      }
    }
    internal::reset_tensor_io_stats();
    bool rejected = false;
    try {
      (void)neat::output_from_sample_stream(sample.get(), "unsupported nested children", true,
                                            nullptr, &state);
    } catch (const std::runtime_error& failure) {
      rejected =
          std::string(failure.what()).find("multiple GstMemory parents") != std::string::npos;
    }
    require(rejected, "multiple Runtime child parents did not fail with the topology diagnostic");
    const auto stats = internal::snapshot_tensor_io_stats();
    require_equal(stats.tensor_copy_count, 0U,
                  "nested topology: copied before admission rejection");
    require_equal(stats.gst_memory_map_calls, 0U,
                  "nested topology: mapped before admission rejection");
  }
  // An offset or truncated GstMemory share cannot be assigned across
  // independently named children without an explicit topology binding. Even
  // offset zero must not expose the second 64-byte child beyond a 96-byte view.
  for (const gssize share_offset : {16, 0}) {
    std::shared_ptr<GstBuffer> offset_buffer(gst_buffer_new(), gst_buffer_unref);
    auto* offset_memory =
        gst_memory_share(gst_buffer_peek_memory(first.buffer.get(), 0), share_offset, 96);
    require(offset_memory != nullptr, "partial multi-child fixture memory share");
    gst_buffer_append_memory(offset_buffer.get(), offset_memory);
    gsize actual_offset = 0;
    require_equal(gst_memory_get_sizes(offset_memory, &actual_offset, nullptr), 96U,
                  "partial multi-child fixture visible span");
    require_equal(actual_offset, static_cast<gsize>(share_offset),
                  "partial multi-child fixture share offset");
    transport::TensorBufferPreparedMetaTemplate offset_prepared;
    require(transport::tensor_buffer_prepare_meta_template_from_contract(
                first_publication, &offset_prepared, &error) &&
                transport::tensor_buffer_attach_prepared_meta(offset_buffer.get(), offset_prepared,
                                                              &error),
            "partial multi-child fixture metadata: " + error);
    const auto offset_sample =
        typed_tensor_sample(offset_buffer.get(), gst_sample_get_caps(first.sample.get()),
                            "partial multi-child fixture");
    const auto root_admitted =
        admitted_view(offset_sample.get(), "partial multi-child Runtime root admission");
    require_equal(root_admitted.segments.size(), 2U,
                  "partial multi-child Runtime admitted child count");
    require_equal(root_admitted.segments[0].size_bytes, 64U,
                  "partial multi-child Runtime first root child extent");
    require_equal(root_admitted.segments[1].size_bytes, 64U,
                  "partial multi-child Runtime second root child extent");
    for (const bool warm : {false, true}) {
      neat::InputStream::State state;
      if (warm) {
        const auto control = neat::output_from_sample_stream(
            first.sample.get(), "partial valid cache control", false, nullptr, &state);
        for (const auto& tensor : control.tensors) {
          require_retained_sample_storage(tensor, first.sample.get(),
                                          "partial valid cache control");
        }
      }
      internal::reset_tensor_io_stats();
      bool rejected = false;
      try {
        (void)neat::output_from_sample_stream(offset_sample.get(), "unsupported partial children",
                                              true, nullptr, &state);
      } catch (const std::runtime_error& failure) {
        rejected = std::string(failure.what()).find("partial GstMemory view") != std::string::npos;
      }
      require(rejected, "partial Runtime child topology did not fail with its topology diagnostic");
      const auto stats = internal::snapshot_tensor_io_stats();
      require_equal(stats.tensor_copy_count, 0U, "partial child topology: copies before rejection");
      require_equal(stats.tensor_copy_bytes, 0U,
                    "partial child topology: copy bytes before rejection");
      require_equal(stats.gst_memory_map_calls, 0U,
                    "partial child topology: maps before rejection");
    }
  }
}
} // namespace

int main(int argc, char** argv) {
  return sima_test::run_test("unit_output_metadata_cache_contract_test", [&] {
    neat::gst_init_once();
    const std::string selected = argc > 1 ? argv[1] : "0";
    if (selected == "bounds") {
      warm_storage_extent_is_checked();
    } else if (selected == "table") {
      current_segment_table_and_capacity_are_not_cached_payload_facts();
    } else if (selected == "named-children") {
      named_child_binding_and_logical_alias_are_distinct();
    } else if (selected == "shared-span") {
      shared_runtime_view_keeps_visible_bounds();
    } else if (selected == "nested-negative") {
      multiple_runtime_child_parents_fail_before_payload_access();
    } else {
      const int change = std::stoi(selected);
      require(change >= 0 && change < 6, "unknown metadata cache contract case");
      changed_legacy_metadata_is_not_static(change);
    }
  });
}

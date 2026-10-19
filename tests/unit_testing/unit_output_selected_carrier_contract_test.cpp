#include "tensor_output_contract_fixtures.h"
#include "owned_output_test_utils.h"
#include "pipeline/internal/HolderLoanGate.h"
#include "pipeline/internal/OutputTensorOverride.h"
#include "test_main.h"

namespace {
using namespace sima_test::output_contract;

void unselected_dma_sibling_is_never_mapped() {
  auto input = frame(contract(true), {true, false}, 23);
  // Deliberately non-exporter bookkeeping memory is safe ONLY when unselected:
  // no DmaBufView CPU map or hardware submission may touch it.
  std::shared_ptr<GstBuffer> poisoned(sima_test::make_bookkeeping_dmabuf(64), gst_buffer_unref);
  std::shared_ptr<GstCaps> caps(gst_caps_ref(gst_sample_get_caps(input.sample.get())),
                                gst_caps_unref);
  input.sample.reset();
  require(gst_buffer_is_writable(input.buffer.get()), "CPU subset fixture buffer is shared");
  auto* cpu = gst_memory_ref(gst_buffer_peek_memory(input.buffer.get(), 1));
  gst_buffer_remove_all_memory(input.buffer.get());
  gst_buffer_append_memory(input.buffer.get(),
                           gst_memory_ref(gst_buffer_peek_memory(poisoned.get(), 0)));
  gst_buffer_append_memory(input.buffer.get(), cpu);
  require(gst_buffer_n_memory(input.buffer.get()) == 2 &&
              gst_buffer_peek_memory(input.buffer.get(), 0) ==
                  gst_buffer_peek_memory(poisoned.get(), 0) &&
              gst_buffer_peek_memory(input.buffer.get(), 1) == cpu &&
              gst_buffer_get_size(input.buffer.get()) == 128,
          "CPU subset fixture did not install its actual unmappable DMA sibling");
  input.sample = {gst_sample_new(input.buffer.get(), caps.get(), nullptr, nullptr),
                  gst_sample_unref};
  require(bool(input.sample), "CPU subset fixture sample recreation");
  const auto admitted = admitted_view(input.sample.get(), "CPU subset Runtime authority");
  const auto* admitted_scores = find_logical(admitted, 7);
  require(admitted_scores != nullptr, "CPU subset Runtime authority: missing scores identity");
  require_equal(admitted_scores->memory_index, 1, "CPU subset Runtime authority: selected carrier");
  require_equal(admitted.segment_name(1), "arena1", "CPU subset Runtime authority: physical name");
  neat::OutputTensorOverride override;
  neat::OutputTensorOverrideEntry entry;
  entry.shape = {4, 4};
  entry.strides_bytes = {4, 1};
  entry.byte_offset = 8;
  entry.memory_index = 1;
  entry.logical_output_index = 7;
  entry.route_slot = 9;
  entry.dtype = neat::TensorDType::Int8;
  entry.layout = neat::TensorLayout::HW;
  entry.name = "scores";
  entry.segment_name = "arena1";
  override.outputs.push_back(entry);
  const std::optional<neat::OutputTensorOverride> selected(override);
  neat::InputStream::State state;
  state.opt.copy_output = true;
  internal::reset_tensor_io_stats();
  const auto output = neat::output_from_sample_stream(input.sample.get(), "selected CPU only", true,
                                                      &selected, &state);
  const auto stats = internal::snapshot_tensor_io_stats();
  require(stats.tensor_copy_count == 1 && stats.tensor_copy_bytes == 64,
          "CPU subset copied an unselected DMA sibling");
  require(output.owned && output.tensors.size() == 1, "CPU subset ownership/logical count");
  const auto& tensor = output.tensors[0];
  auto* buffer = selected_buffer(tensor, "CPU subset");
  require_equal(gst_buffer_n_memory(buffer), 1U, "CPU subset: destination memory count");
  require_equal(tensor.route.memory_index, 0, "CPU subset: destination memory index");
  require_equal(tensor.route.physical_index, 1, "CPU subset: producer physical index");
  require_equal(tensor.route.physical_byte_offset, 8, "CPU subset: physical offset");
  require_equal(tensor.byte_offset, 8, "CPU subset: logical offset");
  require_equal(tensor.storage->sima_segments.size(), 1U, "CPU subset: segment count");
  require_equal(tensor.storage->sima_segments[0].name, "arena1", "CPU subset: segment name");
  (void)sima_test::require_dma_owned_output(output, {}, "CPU subset independent output");
  require_payload(tensor, input.seed, 1, 8, 16, "CPU subset");
}

std::optional<neat::OutputTensorOverride> scores_projection(std::int64_t offset,
                                                            bool route_identity_only = false) {
  neat::OutputTensorOverride override;
  neat::OutputTensorOverrideEntry entry;
  entry.shape = {4, 4};
  entry.strides_bytes = {4, 1};
  entry.byte_offset = offset;
  // These selectors deliberately describe a previous transport generation.
  // An exact admitted logical/route identity must not let them reroot storage.
  entry.memory_index = 0;
  entry.segment_name = "old_arena";
  entry.logical_output_index = route_identity_only ? -1 : 7;
  entry.route_slot = 9;
  entry.dtype = neat::TensorDType::Int8;
  entry.layout = neat::TensorLayout::HW;
  entry.name = "public_scores";
  override.outputs.push_back(std::move(entry));
  return override;
}

void admitted_binding_survives_projection(bool route_identity_only, bool compact = false) {
  // Two representative layouts, not a per-head Cartesian matrix: sparse
  // producer id 17 in local slot 1, and two logical heads packed in slot 0.
  for (const bool packed : {false, true}) {
    if ((route_identity_only || compact) && packed) {
      continue;
    }
    auto publication = contract(!packed);
    publication.physical_outputs[packed ? 0 : 1].physical_index = 17;
    for (auto& head : publication.logical_outputs) {
      if (packed || head.logical_index == 7) {
        head.physical_index = 17;
      }
    }
    const auto projection = scores_projection(packed ? 16 : 8, route_identity_only);
    transport::TensorBufferPublicationOwner frozen;
    if (compact) {
      transport::TensorBufferPreparedMetaTemplate prepared;
      std::string error;
      require(transport::tensor_buffer_prepare_meta_template_from_contract(
                  publication, {{0, 0, 64}, {1, 0, 64}}, &prepared, &error),
              "compact authority fixture template: " + error);
      frozen = transport::tensor_buffer_freeze_publication(prepared, &error);
      require(bool(frozen), "compact authority fixture owner: " + error);
    }
    // ZeroCopy, Auto DMA, Owned DMA, and Auto CPU. Stats are captured before
    // any payload oracle; preparation/verification maps do not mask extra work.
    for (int policy = 0; policy != 4; ++policy) {
      if (compact && (policy == 1 || policy == 3)) {
        continue; // Representative compact retain/copy paths; legacy covers all policies.
      }
      const bool target_dma = policy != 3;
      const bool copy = policy != 0;
      const bool preserve = policy == 1 || policy == 3;
      const bool retained = !copy || (preserve && target_dma);
      neat::InputStream::State state;
      state.opt.copy_output = copy;
      state.opt.preserve_dmabuf_output = preserve;
      std::vector<Frame> inputs;
      std::vector<neat::Sample> outputs;
      for (int current = 0; current != 2; ++current) {
        inputs.push_back(
            frame(publication,
                  packed ? std::vector<bool>{target_dma} : std::vector<bool>{true, target_dma},
                  static_cast<std::uint8_t>(19 + 79 * current), compact, 64, "original", &frozen));
        const auto& input = inputs.back();
        const auto context = "admitted authority packed=" + std::to_string(packed) +
                             " route-only=" + std::to_string(route_identity_only) +
                             " compact=" + std::to_string(compact) +
                             " policy=" + std::to_string(policy) +
                             " frame=" + std::to_string(current);
        const auto admitted = admitted_view(input.sample.get(), context);
        const auto* head = find_logical(admitted, 7);
        require(head != nullptr, context + ": missing admitted scores identity");
        require_equal(head->physical_index, 17, context + ": fixture producer physical id");
        require_equal(head->memory_index, packed ? 0 : 1,
                      context + ": fixture local carrier selector");
        if (compact) {
          require(transport::tensor_buffer_publication(input.buffer.get()) == frozen,
                  context + ": fixture must preserve compact warm publication identity");
        }
        internal::reset_tensor_io_stats();
        outputs.push_back(neat::output_from_sample_stream(input.sample.get(), "binding authority",
                                                          copy, &projection, &state));
        const auto stats = internal::snapshot_tensor_io_stats();
        require_equal(stats.tensor_copy_count, retained ? 0U : 1U,
                      context + ": selected carrier copy count");
        require_equal(stats.tensor_copy_bytes, retained ? 0U : 64U,
                      context + ": selected carrier copy bytes");
        if (retained) {
          require_equal(stats.gst_memory_map_calls, 0U,
                        context + ": extraction mapped a retained payload");
        }
        const auto& output = outputs.back();
        require_equal(output.tensors.size(), 1U, context + ": projected logical count");
        require_equal(output.owned, !retained, context + ": ownership flag");
        const auto& tensor = output.tensors[0];
        require_equal(tensor.route.physical_index, head->physical_index,
                      context + ": producer physical identity");
        require_equal(tensor.route.physical_byte_offset, head->byte_offset,
                      context + ": producer physical offset");
        require_equal(tensor.byte_offset, head->byte_offset, context + ": composed logical offset");
        require_equal(tensor.route.logical_index, route_identity_only ? 0 : 7,
                      context + ": public logical overlay");
        require_equal(tensor.route.route_slot, 9, context + ": public route overlay");
        require_equal(tensor.route.name, "public_scores", context + ": public name overlay");
        require_equal(tensor.route.backend_name, head->backend_name,
                      context + ": producer backend name");
        require_equal(tensor.route.segment_name, head->segment_name,
                      context + ": admitted route segment name");
        require_equal(tensor.route.memory_index, retained ? head->memory_index : 0,
                      context + ": current destination selector");
        if (retained) {
          require_segment_table(tensor, admitted, context);
          require(selected_memory(tensor, context) ==
                      gst_buffer_peek_memory(input.buffer.get(), head->memory_index),
                  context + ": retained payload replaced its selected memory");
        } else {
          require_equal(gst_buffer_n_memory(selected_buffer(tensor, context)), 1U,
                        context + ": destination carrier count");
          require_equal(tensor.storage->sima_segments.size(), 1U,
                        context + ": destination segment count");
          require_equal(tensor.storage->sima_segments[0].name,
                        admitted.segment_name(head->memory_index),
                        context + ": selected physical segment name");
          (void)sima_test::require_dma_owned_output(output, {}, context);
        }
        require_payload(tensor, input.seed, packed ? 0 : 1, head->byte_offset, 16, context);
      }
      require_payload(outputs[0].tensors[0], inputs[0].seed, packed ? 0 : 1, packed ? 16 : 8, 16,
                      "held first authority output after warm rebinding");
    }
  }
}

void unmatched_projection_keeps_actual_name_precedence() {
  auto input = frame(contract(true), {false, false}, 47);
  auto projection = scores_projection(8);
  auto& entry = projection->outputs[0];
  entry.logical_output_index = 99; // No exact admitted logical or route identity.
  entry.route_slot = -1;
  entry.segment_name = "arena1";
  entry.memory_index = 0; // Actual name, not this obsolete local selector, wins.
  const auto admitted = admitted_view(input.sample.get(), "unmatched actual-name control");
  for (const bool copy : {false, true}) {
    neat::InputStream::State state;
    state.opt.copy_output = copy;
    internal::reset_tensor_io_stats();
    const auto output = neat::output_from_sample_stream(input.sample.get(), "unmatched projection",
                                                        copy, &projection, &state);
    const auto stats = internal::snapshot_tensor_io_stats();
    require_equal(stats.tensor_copy_count, copy ? 1U : 0U,
                  "unmatched actual name: selected copy count");
    require_equal(stats.tensor_copy_bytes, copy ? 64U : 0U,
                  "unmatched actual name: selected copy bytes");
    if (!copy) {
      require_equal(stats.gst_memory_map_calls, 0U, "unmatched actual name: extraction map count");
    }
    require_equal(output.tensors.size(), 1U, "unmatched actual name: logical count");
    const auto& tensor = output.tensors[0];
    require_equal(tensor.route.memory_index, copy ? 0 : 1,
                  "unmatched actual name: destination selector");
    require_equal(tensor.route.segment_name, admitted.segment_name(1),
                  "unmatched actual name: selected segment name");
    require_payload(tensor, input.seed, 1, 8, 16, "unmatched actual name: selected bytes");
  }
}

void single_head_expansion_keeps_matching_authority_and_copy_dedup() {
  auto publication = contract(true);
  publication.physical_outputs[1].physical_index = 17;
  auto scores = publication.logical_outputs[1];
  scores.physical_index = 17;
  publication.logical_outputs = {scores};
  auto projection = scores_projection(8);
  auto expansion = projection->outputs[0];
  expansion.logical_output_index = 19;
  expansion.route_slot = 19;
  expansion.segment_name = "arena1";
  expansion.name = "expanded_scores";
  expansion.byte_offset = 32;
  projection->outputs.push_back(std::move(expansion));
  for (const bool copy : {false, true}) {
    neat::InputStream::State state;
    state.opt.copy_output = copy;
    state.opt.preserve_dmabuf_output = false;
    std::vector<Frame> inputs;
    std::vector<neat::Sample> outputs;
    for (int current = 0; current != 2; ++current) {
      inputs.push_back(
          frame(publication, {true, true}, static_cast<std::uint8_t>(31 + 73 * current)));
      const auto& input = inputs.back();
      const auto admitted = admitted_view(input.sample.get(), "single-head expansion authority");
      require_equal(admitted.tensors().size(), 1U, "single-head expansion: canonical logical count");
      require_equal(admitted.tensors()[0].logical_index, 7,
                    "single-head expansion: canonical identity");
      internal::reset_tensor_io_stats();
      outputs.push_back(neat::output_from_sample_stream(input.sample.get(), "single-head expansion",
                                                        copy, &projection, &state));
      const auto stats = internal::snapshot_tensor_io_stats();
      require_equal(stats.tensor_copy_count, copy ? 1U : 0U,
                    "single-head expansion: selected shared carrier copy count");
      require_equal(stats.tensor_copy_bytes, copy ? 64U : 0U,
                    "single-head expansion: selected shared carrier copy bytes");
      if (!copy) {
        require_equal(stats.gst_memory_map_calls, 0U, "single-head expansion: projection CPU maps");
      }
      const auto& output = outputs.back();
      require_equal(output.tensors.size(), 2U, "single-head expansion: projected logical count");
      require_equal(output.owned, copy, "single-head expansion: ownership flag");
      const auto& matching = output.tensors[0];
      require_equal(matching.route.physical_index, 17,
                    "single-head expansion: matched producer physical identity");
      require_equal(matching.route.physical_byte_offset, 8,
                    "single-head expansion: matched producer physical offset");
      require_equal(matching.byte_offset, 8, "single-head expansion: matched composed offset");
      require_equal(matching.route.logical_index, 7,
                    "single-head expansion: matched logical identity");
      require_equal(matching.route.segment_name, admitted.tensors()[0].segment_name,
                    "single-head expansion: matched segment authority");
      for (const auto& tensor : output.tensors) {
        require_equal(tensor.route.memory_index, copy ? 0 : 1,
                      "single-head expansion: selected current carrier");
      }
      require(selected_memory(output.tensors[0], "single-head first") ==
                  selected_memory(output.tensors[1], "single-head second"),
              "single-head expansion did not share/deduplicate its same selected carrier");
      require_payload(matching, input.seed, 1, 8, 16, "single-head matching current bytes");
      require_payload(output.tensors[1], input.seed, 1, 32, 16,
                      "single-head unmatched actual-name current bytes");
    }
    require_payload(outputs[0].tensors[0], inputs[0].seed, 1, 8, 16,
                    "held single-head matching output after warm rebinding");
    require_payload(outputs[0].tensors[1], inputs[0].seed, 1, 32, 16,
                    "held single-head expanded output after warm rebinding");
  }
}

void copied_cpu_does_not_keep_retained_dma_loan() {
  auto input = frame(contract(true), {true, false}, 37);
  neat::InputStream::State state;
  state.lifetime_token = std::make_shared<int>(0);
  auto output = neat::output_from_sample_stream(input.sample.get(), "mixed source loan", false,
                                                nullptr, &state);
  auto gate = std::make_shared<internal::HolderLoanGate>(2);
  std::string error;
  require(internal::attach_zero_copy_loan_to_sample(output, gate, &error),
          "mixed source loan setup: " + error);
  const auto acquired = gate->inflight();
  require(acquired > 0, "source loan did not acquire producer credits");
  internal::materialize_owned_output(output, true);
  require(!output.owned && output.tensors.size() == 2, "mixed Auto ownership");
  auto copied_cpu = neat::sample_from_tensors(neat::TensorList{output.tensors[1]});
  copied_cpu.owned = true;
  require(!internal::sample_has_zero_copy_loans(copied_cpu) &&
              !internal::sample_has_device_gstsample_producer_lifetime(copied_cpu),
          "independent CPU output inherited producer loan/lifetime");
  const auto producer_released = input.released;
  input.sample.reset();
  input.buffer.reset();
  require(!producer_released->load(), "retained DMA head prematurely released producer buffer");
  output = {};
  state.lifetime_token.reset();
  require(producer_released->load(), "copied CPU head pins its released retained DMA sibling");
  require(gate->inflight() == 0 && gate->released() == static_cast<std::uint64_t>(acquired) &&
              gate->overrelease() == 0,
          "final retained head did not return source loan exactly once");
  (void)sima_test::require_dma_owned_output(copied_cpu, {}, "held copied CPU output");
  require_payload(copied_cpu.tensors[0], input.seed, 1, 8, 16, "copied CPU after producer release");
}

void aliases_across_envelopes_copy_checked_span_once(bool different_span) {
  auto input = frame(contract(), {true}, 43);
  const auto decoded =
      neat::output_from_sample_stream(input.sample.get(), "alias source", false, nullptr, nullptr);
  auto* original = gst_buffer_peek_memory(input.buffer.get(), 0);
  std::shared_ptr<GstBuffer> wrapper(gst_buffer_new(), gst_buffer_unref);
  auto* memory = different_span ? gst_memory_share(original, 32, 32) : gst_memory_ref(original);
  require(memory != nullptr, "alias wrapper selected-memory fixture");
  gst_buffer_append_memory(wrapper.get(), memory);
  std::shared_ptr<GstSample> wrapper_sample(
      gst_sample_new(wrapper.get(), gst_sample_get_caps(input.sample.get()), nullptr, nullptr),
      gst_sample_unref);
  auto second = decoded.tensors[1];
  second.storage = internal::make_gst_sample_storage(wrapper_sample.get());
  second.route.memory_index = 0;
  if (different_span) {
    second.byte_offset = 0;
    second.route.physical_byte_offset = 32;
  }
  auto output = neat::sample_from_tensors(neat::TensorList{decoded.tensors[0], second});
  internal::reset_tensor_io_stats();
  internal::materialize_owned_output(output, false);
  const auto stats = internal::snapshot_tensor_io_stats();
  require(output.owned && output.tensors.size() == 2, "Owned alias logical count");
  require(stats.tensor_copy_count == (different_span ? 2U : 1U) &&
              stats.tensor_copy_bytes == (different_span ? 96U : 64U),
          "DMA alias dedup ignored checked backing identity+offset+length");
  const auto first_span = sima_test::dmabuf_span(selected_memory(output.tensors[0], "first alias"));
  const auto second_span =
      sima_test::dmabuf_span(selected_memory(output.tensors[1], "second alias"));
  require((first_span == second_span) == !different_span,
          "alias output merged different spans or lost shared span");
  require(first_span.backing != input.dma_spans[0].backing &&
              second_span.backing != input.dma_spans[0].backing,
          "Owned alias output reused producer backing");
  (void)sima_test::require_dma_owned_output(output, {input.dma_spans[0].backing},
                                            "separate valid Owned carriers");
  require_payload(output.tensors[0], input.seed, 0, 0, 16, "first alias");
  require_payload(output.tensors[1], input.seed, 0, different_span ? 32 : 16, 16, "second alias");
}
void broadcast_stride_publication_preserves_aliasing() {
  auto publication = contract();
  publication.logical_outputs.resize(1);
  auto& head = publication.logical_outputs[0];
  head.shape = {2, 2};
  head.stride_bytes = {0, 1};
  head.byte_offset = 4;
  head.size_bytes = 4; // Four logical elements use only two physical bytes.
  auto input = frame(publication, {true}, 53);
  client::TensorBufferView admitted;
  std::string admission_error;
  require(client::tensor_buffer_create_view(input.sample.get(), &admitted, &admission_error) &&
              admitted.tensors().size() == 1 && admitted.tensors()[0].physical_span_bytes == 2,
          "Runtime broadcast admission must distinguish four logical from two physical bytes: " +
              admission_error);
  neat::InputStream::State state;
  state.opt.copy_output = true;
  internal::reset_tensor_io_stats();
  const auto output = neat::output_from_sample_stream(input.sample.get(), "broadcast publication",
                                                      true, nullptr, &state);
  const auto stats = internal::snapshot_tensor_io_stats();
  require(stats.tensor_copy_count == 1 && stats.tensor_copy_bytes == 64,
          "broadcast publication must copy its selected physical carrier once");
  require(output.owned && output.tensors.size() == 1, "broadcast output ownership/count");
  const auto& tensor = output.tensors[0];
  require(tensor.shape == std::vector<int64_t>({2, 2}) &&
              tensor.strides_bytes == std::vector<int64_t>({0, 1}) && tensor.byte_offset == 4 &&
              tensor.route.physical_byte_offset == 4 && tensor.route.physical_index == 0,
          "Owned broadcast view changed admitted strides/offset/provenance");
  const auto mapping = tensor.view_read();
  require(mapping.data && mapping.size_bytes >= 2,
          "broadcast view must expose its two-byte physical span");
  require(static_cast<const std::uint8_t*>(mapping.data)[0] == input.seed + 4 &&
              static_cast<const std::uint8_t*>(mapping.data)[1] == input.seed + 5,
          "broadcast view copied the wrong physical bytes");
  const std::vector<std::uint8_t> expected = {
      static_cast<std::uint8_t>(input.seed + 4), static_cast<std::uint8_t>(input.seed + 5),
      static_cast<std::uint8_t>(input.seed + 4), static_cast<std::uint8_t>(input.seed + 5)};
  std::vector<std::uint8_t> logical;
  const auto* bytes = static_cast<const std::uint8_t*>(mapping.data);
  for (int row = 0; row != 2; ++row) {
    for (int column = 0; column != 2; ++column) {
      logical.push_back(bytes[row * tensor.strides_bytes[0] + column * tensor.strides_bytes[1]]);
    }
  }
  require(logical == expected, "broadcast logical rows no longer alias the same physical elements");
  (void)sima_test::require_dma_owned_output(output, {input.dma_spans[0].backing},
                                            "independent broadcast carrier");
}
void whole_fragmented_encoded_policy(bool all_dma) {
  auto input = frame(contract(true), {true, all_dma}, 67);
  neat::Tensor tensor;
  tensor.storage = internal::make_gst_sample_storage(input.sample.get());
  tensor.dtype = neat::TensorDType::UInt8;
  tensor.shape = {128};
  tensor.strides_bytes = {1};
  tensor.semantic.encoded.emplace();
  tensor.semantic.encoded->codec = neat::EncodedSpec::Codec::H264;
  const auto original_storage = tensor.storage;
  auto output = neat::sample_from_tensors(neat::TensorList{tensor});
  internal::reset_tensor_io_stats();
  internal::materialize_owned_output(output, true);
  const auto stats = internal::snapshot_tensor_io_stats();
  require(stats.tensor_copy_count == (all_dma ? 0U : 1U) &&
              stats.tensor_copy_bytes == (all_dma ? 0U : 128U),
          "whole encoded Auto applied policy only to its first fragment");
  require(output.owned == !all_dma && output.tensors.size() == 1,
          "whole encoded sample ownership flag/logical count");
  if (all_dma) {
    require(output.tensors[0].storage == original_storage &&
                output.tensors[0].route.memory_index == -1,
            "all-DMA encoded whole storage was replaced");
  } else {
    require(gst_buffer_n_memory(selected_buffer(output.tensors[0], "whole encoded")) == 1,
            "copied whole encoded payload did not coalesce its fragments");
    const auto mapping = output.tensors[0].storage->map(neat::MapMode::Read);
    require(mapping.data && mapping.size_bytes == 128, "whole encoded independent payload extent");
    for (std::size_t i = 0; i != 128; ++i) {
      require(static_cast<const std::uint8_t*>(mapping.data)[i] ==
                  static_cast<std::uint8_t>(input.seed + i),
              "whole encoded copy lost a fragment byte");
    }
    (void)sima_test::require_dma_owned_output(output, {input.dma_spans[0].backing},
                                              "whole mixed independent payload");
  }
}

void whole_multimemory_nv12_preserves_global_planes() {
  for (const bool row_only : {false, true}) {
    std::shared_ptr<GstBuffer> buffer(gst_buffer_new(), gst_buffer_unref);
    for (const std::size_t size : {32U, 16U}) {
      auto* memory = gst_allocator_alloc(nullptr, size, nullptr);
      require(memory != nullptr, "ordinary whole NV12 memory fixture");
      GstMapInfo mapping{};
      require(gst_memory_map(memory, &mapping, GST_MAP_WRITE), "ordinary whole NV12 fixture map");
      require(mapping.data && mapping.size >= size, "ordinary whole NV12 fixture mapped extent");
      const std::size_t base = gst_buffer_get_size(buffer.get());
      for (std::size_t i = 0; i != size; ++i) {
        mapping.data[i] = static_cast<std::uint8_t>(base + i + 1);
      }
      gst_memory_unmap(memory, &mapping);
      gst_buffer_append_memory(buffer.get(), memory);
    }
    std::shared_ptr<GstSample> source(gst_sample_new(buffer.get(), nullptr, nullptr, nullptr),
                                      gst_sample_unref);
    neat::Tensor tensor;
    tensor.storage = internal::make_gst_sample_storage(source.get());
    tensor.dtype = neat::TensorDType::UInt8;
    tensor.shape = {4, 4};
    tensor.strides_bytes = {8, 1};
    tensor.layout = neat::TensorLayout::HW;
    tensor.semantic.image.emplace();
    tensor.semantic.image->format = neat::ImageSpec::PixelFormat::NV12;
    tensor.planes = {{neat::PlaneRole::Y, {4, 4}, {8, 1}, 0},
                     {neat::PlaneRole::UV, {2, 4}, {8, 1}, 32}};
    if (row_only) {
      tensor.planes[0].strides_bytes = {8};
      tensor.planes[1].strides_bytes = {8};
    }
    auto output = neat::sample_from_tensors(neat::TensorList{tensor});
    internal::reset_tensor_io_stats();
    internal::materialize_owned_output(output, true);
    const auto stats = internal::snapshot_tensor_io_stats();
    require(output.owned && output.tensors.size() == 1 && stats.tensor_copy_count == 1 &&
                stats.tensor_copy_bytes == 48,
            "ordinary whole NV12 must copy both physical plane memories once");
    const auto& copied = output.tensors[0];
    require(copied.planes.size() == 2 && copied.planes[1].byte_offset == 32 &&
                copied.planes[0].strides_bytes == tensor.planes[0].strides_bytes &&
                copied.planes[1].strides_bytes == tensor.planes[1].strides_bytes,
            "ordinary whole NV12 changed global plane offsets or padded strides");
    const auto mapping = copied.storage->map(neat::MapMode::Read);
    require(mapping.data && mapping.size_bytes == 48, "whole NV12 independent backing extent");
    for (std::size_t i = 0; i != 48; ++i) {
      require(static_cast<const std::uint8_t*>(mapping.data)[i] == i + 1,
              "whole NV12 physical payload or padding changed");
    }
    (void)sima_test::require_dma_owned_output(output, {}, "whole NV12 independent output");
  }
}

void counterfeit_owned_helper_rejects_bad_metadata() {
  auto input = frame(contract(), {true}, 71);
  auto owned = neat::output_from_sample_stream(input.sample.get(), "valid Owned helper control",
                                               true, nullptr, nullptr);
  (void)sima_test::require_dma_owned_output(owned, {input.dma_spans[0].backing}, "valid control");
  for (int defect = 0; defect != 3; ++defect) {
    auto counterfeit = owned;
    if (defect == 0) {
      counterfeit.tensors[0].byte_offset =
          60; // 16-byte head ends beyond its actual 64-byte memory.
    } else {
      counterfeit.tensors[0].route.memory_index = defect == 1 ? 1 : -2;
    }
    bool rejected = false;
    try {
      (void)sima_test::require_dma_owned_output(counterfeit, {}, "counterfeit Owned metadata");
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected, "Owned flag/helper accepted invalid selected extent or memory binding");
  }
}

void invalid_selected_indices_reject_before_copy() {
  auto input = frame(contract(), {true}, 79);
  const auto decoded = neat::output_from_sample_stream(input.sample.get(), "binding control", false,
                                                       nullptr, nullptr);
  for (const int index : {-2, 1}) {
    auto invalid = decoded;
    invalid.tensors[0].route.memory_index = index;
    internal::reset_tensor_io_stats();
    bool rejected = false;
    try {
      internal::materialize_owned_output(invalid, false);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected && internal::snapshot_tensor_io_stats().tensor_copy_count == 0,
            "invalid selected memory index fell back or copied before rejection");
  }
}
} // namespace

int main(int argc, char** argv) {
  return sima_test::run_test("unit_output_selected_carrier_contract_test", [&] {
    neat::gst_init_once();
    const std::string selected = argc > 1 ? argv[1] : "subset";
    if (selected == "subset") {
      unselected_dma_sibling_is_never_mapped();
    } else if (selected == "authority") {
      admitted_binding_survives_projection(false);
      single_head_expansion_keeps_matching_authority_and_copy_dedup();
    } else if (selected == "compact-authority") {
      admitted_binding_survives_projection(false, true);
    } else if (selected == "route-authority") {
      admitted_binding_survives_projection(true);
    } else if (selected == "name-precedence") {
      unmatched_projection_keeps_actual_name_precedence();
    } else if (selected == "loan") {
      copied_cpu_does_not_keep_retained_dma_loan();
    } else if (selected == "alias") {
      aliases_across_envelopes_copy_checked_span_once(false);
    } else if (selected == "different-span") {
      aliases_across_envelopes_copy_checked_span_once(true);
    } else if (selected == "broadcast") {
      broadcast_stride_publication_preserves_aliasing();
    } else if (selected == "whole-retain") {
      whole_fragmented_encoded_policy(true);
    } else if (selected == "whole-mixed") {
      whole_fragmented_encoded_policy(false);
    } else if (selected == "whole-planes") {
      whole_multimemory_nv12_preserves_global_planes();
    } else if (selected == "helper-negative") {
      counterfeit_owned_helper_rejects_bad_metadata();
    } else if (selected == "binding-negative") {
      invalid_selected_indices_reject_before_copy();
    } else {
      throw std::runtime_error("unknown selected-carrier test case");
    }
  });
}

#include "tensor_output_contract_fixtures.h"
#include "owned_output_test_utils.h"
#include "pipeline/internal/OutputTensorOverride.h"
#include "test_main.h"

namespace {
using namespace sima_test::output_contract;

std::optional<neat::OutputTensorOverride> projection(bool mixed, bool enabled) {
  if (!enabled) {
    return std::nullopt;
  }
  neat::OutputTensorOverride result;
  for (int i = 0; i != 2; ++i) {
    neat::OutputTensorOverrideEntry entry;
    entry.shape = {4, 4};
    entry.strides_bytes = {4, 1};
    entry.byte_offset = i == 0 ? 0 : mixed ? 8 : 16;
    entry.memory_index = mixed ? i : 0;
    entry.logical_output_index = 2 + 5 * i;
    entry.route_slot = 5 + 4 * i;
    entry.dtype = neat::TensorDType::Int8;
    entry.layout = neat::TensorLayout::HW;
    entry.name = i == 0 ? "boxes" : "scores";
    entry.segment_name = "arena" + std::to_string(entry.memory_index);
    result.outputs.push_back(std::move(entry));
  }
  return result;
}

void policy_matrix(int selected_case) {
  for (const bool mixed : {bool(selected_case & 8)}) {
    for (const bool override_enabled : {bool(selected_case & 4)}) {
      for (const bool copy : {bool(selected_case & 2)}) {
        for (const bool preserve : {bool(selected_case & 1)}) {
          auto publication = contract(mixed);
          const auto override = projection(mixed, override_enabled);
          neat::InputStream::State state;
          state.opt.copy_output = copy;
          state.opt.preserve_framework_output = preserve;
          std::vector<Frame> inputs;
          std::vector<neat::Sample> outputs;
          for (int frame_index = 0; frame_index != 2; ++frame_index) {
            inputs.push_back(frame(publication,
                                   mixed ? std::vector<bool>{true, false} : std::vector<bool>{true},
                                   static_cast<std::uint8_t>(17 + 83 * frame_index)));
            auto& input = inputs.back();
            internal::reset_tensor_io_stats();
            outputs.push_back(neat::output_from_sample_stream(input.sample.get(), "public policy",
                                                              copy, &override, &state));
            const auto stats = internal::snapshot_tensor_io_stats();
            const std::uint64_t copied = !copy ? 0 : preserve ? (mixed ? 1 : 0) : mixed ? 2 : 1;
            const auto context =
                "policy mixed=" + std::to_string(mixed) +
                " override=" + std::to_string(override_enabled) + " copy=" + std::to_string(copy) +
                " preserve=" + std::to_string(preserve) + " frame=" + std::to_string(frame_index);
            // Measure before payload oracles and explicit conversions touch the counters.
            require(stats.tensor_copy_count == copied && stats.tensor_copy_bytes == 64 * copied,
                    context + ": copied nonselected/retained sibling carriers; actual copies=" +
                        std::to_string(stats.tensor_copy_count) +
                        " bytes=" + std::to_string(stats.tensor_copy_bytes));
            auto& output = outputs.back();
            require(state.tensor_set_output_decode_cache.valid,
                    context + ": metadata-only decode cache must not depend on output policy");
            require(output.tensors.size() == 2, context + ": logical output count");
            bool any_retained = false;
            for (int i = 0; i != 2; ++i) {
              const auto& tensor = output.tensors[i];
              const bool source_dma = !mixed || i == 0;
              const bool retained = !copy || (preserve && source_dma);
              any_retained |= retained;
              auto* selected = selected_memory(tensor, context);
              auto* producer = gst_buffer_peek_memory(input.buffer.get(), mixed ? i : 0);
              require(gst_is_dmabuf_memory(selected) == (source_dma || !retained),
                      context + ": selected payload DMA classification");
              if (retained) {
                require(selected == producer, context + ": replaced retained producer memory");
                if (source_dma) {
                  require(sima_test::dmabuf_span(selected) == input.dma_spans[0],
                          context + ": changed retained DMA span");
                }
              } else {
                require(selected != producer &&
                            !gst_buffer_get_parent_buffer_meta(selected_buffer(tensor, context)),
                        context + ": copied payload retains producer memory/parent");
                if (source_dma) {
                  require(sima_test::dmabuf_identity(selected) !=
                              sima_test::dmabuf_identity(producer),
                          context + ": copied payload aliases producer DMA allocation");
                }
              }
              const auto offset = i == 0 ? 0 : mixed ? 8 : 16;
              require(tensor.byte_offset == offset && tensor.route.physical_byte_offset == offset &&
                          tensor.route.physical_index == (mixed ? i : 0) &&
                          tensor.route.logical_index == 2 + 5 * i &&
                          tensor.strides_bytes == std::vector<int64_t>({4, 1}),
                      context + ": ownership changed logical layout or source provenance");
              require_payload(tensor, input.seed, mixed ? i : 0, offset, 16, context);
            }
            require(output.owned == !any_retained, context + ": public sample ownership flag");
            if (!mixed && copy && !preserve) {
              require(selected_memory(output.tensors[0], context) ==
                          selected_memory(output.tensors[1], context),
                      context + ": shared logical heads lost physical aliasing");
            }
          }
          require_payload(outputs[0].tensors[0], inputs[0].seed, 0, 0, 16,
                          "held cold output must not change after warm rebinding");
          if (copy && !preserve) {
            const auto released = inputs[0].released;
            inputs[0].sample.reset();
            inputs[0].buffer.reset();
            require(released->load(), "Owned output/cache retained first producer buffer");
            require_payload(outputs[0].tensors[0], inputs[0].seed, 0, 0, 16,
                            "Owned payload survives producer destruction");
          }
        }
      }
    }
  }
}
} // namespace

int main(int argc, char** argv) {
  return sima_test::run_test("unit_output_ownership_policy_contract_test", [&] {
    const int selected_case = argc > 1 ? std::stoi(argv[1]) : 11;
    require(selected_case >= 0 && selected_case < 16, "unknown ownership policy case");
    policy_matrix(selected_case);
  });
}

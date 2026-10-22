#include "model/internal/ModelInternal.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/TensorUtil.h"
#include "gst/GstInit.h"
#include "test_main.h"

#include <cstring>
#include <cstdint>
#include <memory>
#include <vector>

using namespace simaai::neat;

namespace {

void require_published_spans(const TensorList& tensors,
                             const std::vector<std::uint64_t>& physical_bytes) {
  std::string error;
  const auto holder =
      pipeline_internal::make_sample_holder_from_bundle(sample_from_tensors(tensors), &error);
  require(holder != nullptr, "public input publication: " + error);
  auto* sample = static_cast<GstSample*>(holder.get());
  auto* buffer = gst_sample_get_buffer(sample);
  pipeline_internal::TensorBufferView view;
  require(pipeline_internal::tensor_buffer_descriptor_from_sample(sample, &view, &error),
          "public input descriptor: " + error);
  require(view.tensors.size() == tensors.size(), "publication must preserve every logical input");
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto& logical = view.tensors[i];
    simaai::gst::TensorBufferPhysicalBinding binding;
    if (physical_bytes[i] > logical.size_bytes) {
      require(simaai::gst::tensor_buffer_physical_binding_at(buffer, i, &binding, &error),
              "public input physical binding: " + error);
      require(binding.size_bytes == physical_bytes[i],
              "public input binding must preserve the complete selected carrier");
    }
    require(logical.shape == tensors[i].shape &&
                logical.size_bytes == tensors[i].dense_bytes_tight(),
            "physical padding must not enlarge the public logical descriptor");
    auto* memory = gst_buffer_peek_memory(buffer, logical.memory_index);
    require(memory && gst_memory_get_sizes(memory, nullptr, nullptr) == physical_bytes[i],
            "physical padding must be backed by visible GstMemory, not only metadata");
  }
}

} // namespace

RUN_TEST(
    "unit_model_ingress_tail_padding_test", ([] {
      gst_init_once();
      {
        Tensor input;
        input.dtype = TensorDType::Int8;
        input.shape = {1, 8};
        input.storage = make_cpu_owned_storage(8U);
        std::memset(input.storage->map(MapMode::Write).data, 0x5A, 8U);
        const auto original_storage = input.storage;
        const Tensor padded = internal::materialize_dense_ingress_tail(input, 16U);
        require(padded.shape == input.shape && padded.dtype == input.dtype,
                "allocation tail must preserve the logical input shape and dtype");
        require(padded.storage != original_storage && padded.storage->size_bytes == 16U &&
                    original_storage->size_bytes == 8U,
                "allocation tail must provide real storage without resizing the caller buffer");
        const auto mapping = padded.map(MapMode::Read);
        const auto* bytes = static_cast<const std::uint8_t*>(mapping.data);
        for (std::size_t index = 0U; index < 16U; ++index) {
          require(bytes[index] == (index < 8U ? 0x5A : 0),
                  "allocation tail must copy the payload and zero the remaining bytes");
        }
        require(internal::materialize_dense_ingress_tail(padded, 16U).storage == padded.storage,
                "an existing complete input carrier must be reused");
        require(internal::materialize_dense_ingress_tail(input, 8U).storage == original_storage,
                "an input without an allocation tail must keep its existing storage");
        require_published_spans(TensorList{padded}, {16U});

        Tensor selector;
        selector.dtype = TensorDType::BFloat16;
        selector.shape = {1, 77};
        selector.storage = make_cpu_owned_storage(154U);
        const auto padded_selector = internal::materialize_dense_ingress_tail(selector, 160U);
        require_published_spans(TensorList{padded_selector}, {160U});
        require_published_spans(TensorList{padded, padded_selector}, {16U, 160U});

        Tensor ordinary = input;
        ordinary.storage = make_cpu_owned_storage(32U);
        require_published_spans(TensorList{ordinary}, {8U});

        auto external_bytes = std::make_shared<std::vector<std::uint8_t>>(8U, 0x44);
        Tensor external = input;
        external.read_only = true;
        external.storage = make_cpu_external_storage(external_bytes->data(), 8U, external_bytes);
        const auto padded_external = internal::materialize_dense_ingress_tail(external, 16U);
        require_published_spans(TensorList{padded_external}, {16U});
        require(*external_bytes == std::vector<std::uint8_t>(8U, 0x44),
                "padding external CPU memory must not modify caller-owned bytes");

        input.storage = make_cpu_owned_storage(12U);
        input.byte_offset = 4;
        std::memset(input.storage->map(MapMode::Write).data, 0x33, 12U);
        const Tensor sliced = internal::materialize_dense_ingress_tail(input, 16U);
        require(sliced.byte_offset == 0 && sliced.storage->size_bytes == 16U &&
                    static_cast<const std::uint8_t*>(sliced.map(MapMode::Read).data)[0] == 0x33,
                "a sliced input must materialize its selected payload into a complete carrier");
        require_published_spans(TensorList{sliced}, {16U});

        input.storage = make_cpu_owned_storage(20U);
        std::memset(input.storage->map(MapMode::Write).data, 0x77, 20U);
        const auto storage = input.storage;
        const auto reused = internal::materialize_dense_ingress_tail(input, 16U);
        const auto reused_mapping = reused.map(MapMode::Read);
        require(reused_mapping.data == static_cast<const std::uint8_t*>(storage->data) + 4U,
                "a sufficient sliced CPU allocation must be wrapped without copying");
        require(reused.storage->size_bytes == 16U && input.storage == storage &&
                    input.byte_offset == 4,
                "the published carrier must apply the slice once and preserve the caller view");
        require_published_spans(TensorList{reused}, {16U});
      }
      {
        gst_init_once();
        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 16U, nullptr);
        GstSample* sample = gst_sample_new(buffer, nullptr, nullptr, nullptr);
        gst_buffer_unref(buffer);
        Tensor tensor;
        tensor.dtype = TensorDType::Int8;
        tensor.shape = {1, 8};
        tensor.storage = pipeline_internal::make_gst_sample_storage(sample);
        gst_sample_unref(sample);
        tensor.route.logical_index = 0;
        tensor.route.physical_index = 0;
        tensor.route.memory_index = 0;
        tensor.route.name = "mask";
        tensor.route.segment_name = "mask";
        const Sample bundle = sample_from_tensors(TensorList{tensor});
        std::string error;
        const auto holder = pipeline_internal::make_sample_holder_from_bundle(bundle, &error);
        require(holder != nullptr, "padded public input holder: " + error);
        auto* published = gst_sample_get_buffer(static_cast<GstSample*>(holder.get()));
        simaai::gst::TensorBufferPhysicalBinding binding;
        require(simaai::gst::tensor_buffer_physical_binding_at(published, 0U, &binding, &error),
                "padded public input physical binding: " + error);
        require(binding.size_bytes == 16U, "physical binding must cover the actual input carrier");
        pipeline_internal::TensorBufferView view;
        require(pipeline_internal::tensor_buffer_descriptor_from_sample(
                    static_cast<GstSample*>(holder.get()), &view, &error),
                "padded public input descriptor: " + error);
        require(view.tensors.size() == 1U && view.tensors.front().size_bytes == 8U &&
                    view.tensors.front().shape == std::vector<std::int64_t>({1, 8}),
                "physical binding must not enlarge the logical tensor descriptor");
      }
    }));

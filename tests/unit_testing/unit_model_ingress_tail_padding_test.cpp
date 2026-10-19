#include "model/internal/ModelInternal.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/TensorUtil.h"
#include "gst/GstInit.h"
#include "test_main.h"

#include <cstring>
#include <cstdint>

using namespace simaai::neat;

RUN_TEST(
    "unit_model_ingress_tail_padding_test", ([] {
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

        input.storage = make_cpu_owned_storage(12U);
        input.byte_offset = 4;
        std::memset(input.storage->map(MapMode::Write).data, 0x33, 12U);
        const Tensor sliced = internal::materialize_dense_ingress_tail(input, 16U);
        require(sliced.byte_offset == 0 && sliced.storage->size_bytes == 16U &&
                    static_cast<const std::uint8_t*>(sliced.map(MapMode::Read).data)[0] == 0x33,
                "a sliced input must materialize its selected payload into a complete carrier");
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

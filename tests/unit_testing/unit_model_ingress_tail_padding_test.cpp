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
    require(simaai::gst::tensor_buffer_physical_binding_at(buffer, i, &binding, &error),
            "public input physical binding: " + error);
    require(binding.memory_index == logical.memory_index &&
                binding.byte_offset == static_cast<std::uint64_t>(logical.byte_offset) &&
                binding.size_bytes == physical_bytes[i],
            "public input binding must preserve the complete selected carrier");
    require(logical.shape == tensors[i].shape &&
                logical.size_bytes == tensors[i].dense_bytes_tight(),
            "physical padding must not enlarge the public logical descriptor");
    auto* memory = gst_buffer_peek_memory(buffer, logical.memory_index);
    require(memory && gst_memory_get_sizes(memory, nullptr, nullptr) == physical_bytes[i],
            "physical padding must be backed by visible GstMemory, not only metadata");
  }
}

Tensor cpu_mask(std::size_t storage_bytes = 8U, std::int64_t offset = 0) {
  Tensor tensor;
  tensor.dtype = TensorDType::Int8;
  tensor.shape = {1, 8};
  tensor.storage = make_cpu_owned_storage(storage_bytes);
  tensor.byte_offset = offset;
  std::memset(tensor.storage->map(MapMode::Write).data, 0x5A, storage_bytes);
  return tensor;
}

void test_owned_cpu_inputs() {
  struct Case {
    TensorDType dtype;
    std::int64_t elements;
    std::size_t payload;
    std::size_t reservation;
  };
  TensorList padded_inputs;
  for (const auto test :
       {Case{TensorDType::Int8, 8, 8U, 16U}, Case{TensorDType::BFloat16, 77, 154U, 160U}}) {
    Tensor input;
    input.dtype = test.dtype;
    input.shape = {1, test.elements};
    input.storage = make_cpu_owned_storage(test.payload);
    std::memset(input.storage->map(MapMode::Write).data, 0x5A, test.payload);
    const auto storage = input.storage;
    const auto padded = internal::materialize_dense_ingress_tail(input, test.reservation);
    require(padded.shape == input.shape && padded.dtype == input.dtype,
            "allocation tails must preserve logical shape and dtype");
    require(padded.storage != storage && padded.storage->size_bytes == test.reservation &&
                storage->size_bytes == test.payload,
            "missing tail storage must be materialized without resizing the caller buffer");
    const auto mapping = padded.map(MapMode::Read);
    const auto* bytes = static_cast<const std::uint8_t*>(mapping.data);
    for (std::size_t index = 0; index < test.reservation; ++index) {
      require(bytes[index] == (index < test.payload ? 0x5A : 0),
              "materialization must copy the payload and zero its tail");
    }
    require(internal::materialize_dense_ingress_tail(padded, test.reservation).storage ==
                padded.storage,
            "an already published complete carrier must be reused");
    require(internal::materialize_dense_ingress_tail(input, test.payload).storage == storage,
            "an input without an allocation tail must keep its storage");
    require_published_spans({padded}, {test.reservation});
    padded_inputs.push_back(padded);
  }
  require_published_spans(padded_inputs, {16U, 160U});
  require_published_spans({padded_inputs.front(), cpu_mask(32U)}, {16U, 8U});
  require_published_spans({cpu_mask(32U)}, {8U});
}

void test_external_cpu_input() {
  auto bytes = std::make_shared<std::vector<std::uint8_t>>(8U, 0x44);
  Tensor input = cpu_mask();
  input.read_only = true;
  input.storage = make_cpu_external_storage(bytes->data(), bytes->size(), bytes);
  const auto padded = internal::materialize_dense_ingress_tail(input, 16U);
  require_published_spans({padded}, {16U});
  require(*bytes == std::vector<std::uint8_t>(8U, 0x44),
          "padding external CPU memory must preserve caller-owned bytes");
}

void test_sliced_cpu_inputs() {
  for (const std::size_t allocation : {12U, 20U}) {
    const Tensor input = cpu_mask(allocation, 4);
    {
      const auto mapping = input.storage->map(MapMode::Write);
      auto* bytes = static_cast<std::uint8_t*>(mapping.data);
      for (std::size_t index = 0; index < allocation; ++index) {
        bytes[index] = static_cast<std::uint8_t>(0x20U + index);
      }
    }
    const auto padded = internal::materialize_dense_ingress_tail(input, 16U);
    require(padded.byte_offset == 0 && padded.storage->size_bytes == 16U &&
                input.byte_offset == 4 && input.storage->size_bytes == allocation,
            "publication must apply the slice once and preserve the caller view");
    const auto mapping = padded.map(MapMode::Read);
    const auto* bytes = static_cast<const std::uint8_t*>(mapping.data);
    for (std::size_t index = 0; index < 8U; ++index) {
      require(bytes[index] == 0x24U + index,
              "a sliced carrier must retain the selected logical payload");
    }
    if (allocation == 20U) {
      require(bytes == static_cast<const std::uint8_t*>(input.storage->data) + 4U,
              "sufficient sliced CPU storage must be wrapped without copying");
    }
    require_published_spans({padded}, {16U});
  }
}

void test_existing_gst_carrier() {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 16U, nullptr);
  GstSample* sample = gst_sample_new(buffer, nullptr, nullptr, nullptr);
  gst_buffer_unref(buffer);
  Tensor tensor;
  tensor.dtype = TensorDType::Int8;
  tensor.shape = {1, 8};
  tensor.storage = pipeline_internal::make_gst_sample_storage(sample);
  gst_sample_unref(sample);
  tensor.route.name = tensor.route.segment_name = "mask";
  tensor.route.logical_index = tensor.route.physical_index = tensor.route.memory_index = 0;
  require_published_spans({tensor}, {16U});
}

void test_padded_batches_are_rejected() {
  for (const int batch : {2, 4}) {
    const auto payload = static_cast<std::size_t>(batch * 14);
    const auto reservation = static_cast<std::size_t>((batch - 1) * 16 + 14);
    for (const auto allocation : {payload, reservation}) {
      for (const bool strided : {false, true}) {
        Tensor input;
        input.dtype = TensorDType::BFloat16;
        input.shape = {batch, 7};
        input.storage = make_cpu_owned_storage(allocation);
        if (strided)
          input.strides_bytes = {16, 2};
        bool rejected = false;
        try {
          (void)internal::materialize_dense_ingress_tail(input, reservation, batch);
        } catch (const std::runtime_error& error) {
          rejected =
              std::string(error.what()).find("per-sample storage strides") != std::string::npos;
        }
        require(rejected,
                "padded batches must be rejected before tight copying or oversized storage reuse");
      }
    }
    Tensor dense;
    dense.dtype = TensorDType::BFloat16;
    dense.shape = {batch, 8};
    dense.storage = make_cpu_owned_storage(batch * 16U);
    const auto unchanged = internal::materialize_dense_ingress_tail(dense, batch * 16U, batch);
    require(unchanged.storage == dense.storage,
            "dense batches with no padding must retain their existing carrier");
    require_published_spans({unchanged}, {batch * 16U});
  }
}

} // namespace

RUN_TEST("unit_model_ingress_tail_padding_test", ([] {
           gst_init_once();
           test_owned_cpu_inputs();
           test_external_cpu_input();
           test_sliced_cpu_inputs();
           test_existing_gst_carrier();
           test_padded_batches_are_rejected();
         }));

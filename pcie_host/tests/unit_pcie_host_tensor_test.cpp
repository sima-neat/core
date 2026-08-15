#include "simaai/neat/pcie/Model.h"

#include "HostPcieTensorPayload.h"
#include "HostPcieTensorSetMeta.h"
#include "gst/SimaTensorSetMetaAbi.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pcie = simaai::neat::pcie;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Fn> void require_throws(Fn&& fn, const std::string& message) {
  bool threw = false;
  try {
    fn();
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, message);
}

} // namespace

int main() {
  gst_init(nullptr, nullptr);
  try {
    {
      std::vector<float> data(2 * 3 * 4, 1.0F);
      pcie::Tensor tensor = pcie::Tensor::from_vector(std::move(data), {2, 3, 4}, "images");
      require(tensor.dtype == pcie::TensorDType::Float32, "from_vector<float> dtype");
      require(tensor.layout == pcie::TensorLayout::HWC, "from_vector<float> layout");
      require(tensor.shape == std::vector<std::int64_t>({2, 3, 4}), "from_vector<float> shape");
      require(tensor.strides_bytes == std::vector<std::int64_t>({48, 16, 4}),
              "from_vector<float> dense strides");
      require(tensor.owner != nullptr, "from_vector<float> owner");
      require(tensor.data != nullptr, "from_vector<float> data");
      require(tensor.size_bytes == 2U * 3U * 4U * sizeof(float), "from_vector<float> size");
      require(tensor.byte_offset == 0, "from_vector<float> offset");
      require(tensor.route.name == "images", "from_vector<float> route");
      require(tensor.read_only, "from_vector<float> read_only");
    }

    {
      std::vector<std::uint8_t> data(4 * 5 * 3, 7);
      pcie::Tensor tensor = pcie::Tensor::from_vector(std::move(data), {4, 5, 3}, "input_image",
                                                      pcie::PixelFormat::BGR);
      require(tensor.dtype == pcie::TensorDType::UInt8, "from_vector<uint8_t> dtype");
      require(tensor.image.has_value(), "from_vector<uint8_t> image metadata");
      require(tensor.image_format == pcie::PixelFormat::BGR, "from_vector<uint8_t> format");
      require(tensor.strides_bytes == std::vector<std::int64_t>({15, 3, 1}),
              "from_vector<uint8_t> strides");
    }

    {
      auto storage = std::make_shared<std::vector<float>>(2 * 8, 0.25F);
      pcie::Tensor tensor =
          pcie::Tensor::from_external(storage->data(), storage->size(), storage, {2, 4}, "view",
                                      static_cast<std::int64_t>(4 * sizeof(float)));
      require(tensor.owner == storage, "from_external owner");
      require(tensor.data == storage->data(), "from_external base pointer");
      require(tensor.size_bytes == storage->size() * sizeof(float), "from_external backing size");
      require(tensor.byte_offset == static_cast<std::int64_t>(4 * sizeof(float)),
              "from_external offset");
      require(tensor.strides_bytes == std::vector<std::int64_t>({16, 4}),
              "from_external inferred strides");
    }

    {
      auto storage = std::make_shared<std::vector<float>>(4, 0.75F);
      const float* data = storage->data();
      pcie::Tensor tensor =
          pcie::Tensor::from_external(data, storage->size(), storage, {2, 2}, "const_view");
      require(tensor.data == data, "from_external const pointer data");
      require(tensor.read_only, "from_external const pointer read_only");
    }

    {
      auto storage = std::make_shared<std::vector<float>>(64 + 32, 0.5F);
      pcie::Tensor first =
          pcie::Tensor::from_external(storage->data(), storage->size(), storage, {8, 8}, "input_0");
      pcie::Tensor second =
          pcie::Tensor::from_external(storage->data(), storage->size(), storage, {4, 8}, "input_1",
                                      static_cast<std::int64_t>(64 * sizeof(float)));
      require(first.owner == second.owner, "shared packed tensors owner");
      require(first.data == second.data, "shared packed tensors base pointer");
      require(first.byte_offset == 0, "shared packed tensors first offset");
      require(second.byte_offset == static_cast<std::int64_t>(64 * sizeof(float)),
              "shared packed tensors second offset");
    }

    {
      auto storage = std::make_shared<std::vector<std::uint8_t>>(
          std::initializer_list<std::uint8_t>{'A', 'B', 'C', 0xEE, 'D', 'E', 'F', 0xEE});
      pcie::Tensor tensor = pcie::Tensor::from_external(storage->data(), storage->size(), storage,
                                                        {2, 3}, "strided", 0, {4, 1});
      auto payload = simaai::neat::pcie::internal::prepare_tensor_payload({tensor});
      require(payload.size_bytes == 6U, "staged strided tensor compacted size");
      require(std::vector<std::uint8_t>(payload.data, payload.data + payload.size_bytes) ==
                  std::vector<std::uint8_t>({'A', 'B', 'C', 'D', 'E', 'F'}),
              "staged strided tensor compacted bytes");
      require(payload.spans.size() == 1U, "staged strided tensor metadata span");
      require(payload.spans.front().strides_bytes_override == std::vector<std::int64_t>({3, 1}),
              "staged strided tensor must advertise contiguous strides");
      require(tensor.strides_bytes == std::vector<std::int64_t>({4, 1}),
              "staging must not mutate caller tensor strides");
    }

    {
      pcie::TensorList tensors;
      tensors.push_back(pcie::Tensor::from_vector(std::vector<float>(4), {2, 2}, "input_0"));
      tensors.push_back(pcie::Tensor::from_vector(std::vector<float>(4), {2, 2}, "input_1"));
      auto payload = simaai::neat::pcie::internal::prepare_tensor_payload(tensors);

      std::vector<simaai::neat::pcie::internal::PcieTensorFact> facts(2);
      facts[0].name = "input_0";
      facts[0].dtype = "FP32";
      facts[0].shape = {2, 2};
      facts[0].physical_index = 1;
      facts[1].name = "input_1";
      facts[1].dtype = "FP32";
      facts[1].shape = {2, 2};
      facts[1].physical_index = 0;

      GstBuffer* buffer = gst_buffer_new();
      require(buffer != nullptr, "tensor-set metadata test buffer");
      simaai::neat::pcie::internal::attach_tensor_set_meta(buffer, payload.spans, facts);
      GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, SIMA_TENSOR_SET_META_NAME);
      require(meta != nullptr, "tensor-set metadata attached");
      const GstStructure* structure = gst_custom_meta_get_structure(meta);
      const GValue* value =
          gst_structure_get_value(structure, SIMA_TENSOR_SET_META_FIELD_DESCRIPTORS);
      require(value != nullptr, "tensor-set descriptors present");
      GBytes* bytes = static_cast<GBytes*>(g_value_get_boxed(value));
      gsize descriptor_bytes = 0;
      const auto* descriptors =
          static_cast<const SimaTensorDescriptorV2*>(g_bytes_get_data(bytes, &descriptor_bytes));
      require(descriptor_bytes == 2U * sizeof(SimaTensorDescriptorV2),
              "tensor-set descriptor count");
      require(descriptors[0].physical_index == 1 && descriptors[1].physical_index == 0,
              "MPK physical input routing must override logical submission order");
      gst_buffer_unref(buffer);
    }

    {
      pcie::Tensor tensor = pcie::Tensor::from_vector(std::vector<float>(4), {2, 2}, "input");
      auto payload = simaai::neat::pcie::internal::prepare_tensor_payload({tensor});
      simaai::neat::pcie::internal::PcieTensorFact fact;
      fact.name = "input";
      fact.dtype = "INT8";
      fact.shape = {2, 2};
      GstBuffer* buffer = gst_buffer_new();
      require_throws(
          [&] {
            simaai::neat::pcie::internal::attach_tensor_set_meta(buffer, payload.spans, {fact});
          },
          "tensor-set metadata must reject a mismatched dtype");
      fact.dtype = "FP32";
      fact.shape = {4, 1};
      require_throws(
          [&] {
            simaai::neat::pcie::internal::attach_tensor_set_meta(buffer, payload.spans, {fact});
          },
          "tensor-set metadata must reject a mismatched shape");
      gst_buffer_unref(buffer);
    }

    require_throws([] { (void)pcie::Tensor::from_vector(std::vector<float>(3), {2, 2}); },
                   "from_vector must reject shape/data mismatch");
    require_throws(
        [] {
          std::vector<float> data(4);
          (void)pcie::Tensor::from_external(data.data(), data.size(), {}, {2, 2});
        },
        "from_external must reject missing owner");
    require_throws(
        [] {
          auto storage = std::make_shared<std::vector<float>>(4);
          (void)pcie::Tensor::from_external(storage->data(), storage->size(), storage, {2, 2}, "",
                                            static_cast<std::int64_t>(sizeof(float)));
        },
        "from_external must reject view past backing buffer");

    std::cout << "[PASS] tensor constructors\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

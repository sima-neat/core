#include "simaai/neat/pcie/Model.h"

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

    require_throws(
        [] {
          (void)pcie::Tensor::from_vector(std::vector<float>(3), {2, 2});
        },
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

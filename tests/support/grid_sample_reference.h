#pragma once

// Test-owned ordered oracle; no Kernels or Runner development API is used.
#include <neat.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace grid_sample_test {
namespace neat = simaai::neat;

inline void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

inline std::uint16_t bf16(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  const auto upper = bits >> 16, lower = bits & 65535;
  if ((bits & 0x7fffffffU) > 0x7f800000U)
    return static_cast<std::uint16_t>(upper | 64U);
  return static_cast<std::uint16_t>(upper + (lower > 32768 || (lower == 32768 && (upper & 1))));
}

struct Case {
  const char* name;
  int height, width, channels, output_height, output_width;
  bool align;
};
inline constexpr Case comparison_cases[]{
    {"square", 16, 16, 16, 32, 8, false},
    {"wide", 8, 16, 16, 32, 8, false},
    {"tall", 16, 8, 16, 32, 8, true},
    {"square_streamed", 80, 80, 32, 300, 3, false},
    {"wide_streamed", 40, 96, 16, 300, 2, false},
    {"tall_streamed", 100, 64, 17, 300, 3, true},
};

struct Frame {
  Case geometry;
  int batch;
  bool bfloat;
  neat::SimaGridSample2DOptions options;
  std::vector<float> features, grid;

  Frame(Case c, bool bf, int generation = 0, int batches = 1,
        neat::SimaGridSample2DPadding padding = neat::SimaGridSample2DPadding::Zeros)
      : geometry(c), batch(batches), bfloat(bf), options{padding, c.align},
        features(static_cast<std::size_t>(batches) * c.height * c.width * c.channels),
        grid(static_cast<std::size_t>(batches) * c.output_height * c.output_width * 2) {
    for (std::size_t i = 0; i < features.size(); ++i) {
      const float value = static_cast<float>(static_cast<int>((i * 71 + 19) % 1021) - 510) / 32;
      features[i] = bf ? std::bit_cast<float>(std::uint32_t(bf16(value)) << 16) : value;
    }
    std::uint32_t rng = 0x4ad16953U + static_cast<std::uint32_t>(generation);
    for (auto& value : grid) {
      rng = rng * 1664525U + 1013904223U;
      value = static_cast<float>(static_cast<int>((rng >> 8) % 2663) - 1331) / 1024;
    }
  }

  neat::TensorList tensors(neat::TensorMemory memory = neat::TensorMemory::CPU) const {
    const auto& c = geometry;
    const std::vector<std::int64_t> shape{batch, c.height, c.width, c.channels};
    neat::Tensor feature;
    if (bfloat) {
      std::vector<std::uint16_t> bits;
      bits.reserve(features.size());
      for (auto value : features)
        bits.push_back(bf16(value));
      feature = neat::Tensor::from_vector(bits, shape, memory);
      feature.dtype = neat::TensorDType::BFloat16;
    } else {
      feature = neat::Tensor::from_vector(features, shape, memory);
    }
    feature.layout = neat::TensorLayout::HWC;
    feature.route.name = "features";
    auto coordinates =
        neat::Tensor::from_vector(grid, {batch, c.output_height, c.output_width, 2}, memory);
    coordinates.route.name = "grid";
    return {std::move(feature), std::move(coordinates)};
  }

  float expected(std::size_t point, int channel) const {
    const auto& c = geometry;
    const int sizes[]{c.width, c.height};
    double coordinate[2];
    const bool border = options.padding == neat::SimaGridSample2DPadding::Border;
    for (int axis = 0; axis < 2; ++axis) {
      float value = grid[2 * point + axis];
      if (std::isnan(value))
        value = -1;
      if (border)
        value = std::clamp(value, -1.0f, 1.0f);
      else if (!std::isfinite(value))
        return 0;
      const float added = value + 1;
      float mapped = added * static_cast<float>(sizes[axis] - (options.align_corners ? 1 : 0));
      if (!options.align_corners)
        mapped = mapped - 1;
      mapped = mapped * 0.5f;
      coordinate[axis] = mapped;
      if (border)
        coordinate[axis] = std::clamp(coordinate[axis], 0.0, double(sizes[axis] - 1));
      else if (!(coordinate[axis] > -1 && coordinate[axis] < sizes[axis]))
        return 0;
    }
    const int left = static_cast<int>(std::floor(coordinate[0]));
    const int top = static_cast<int>(std::floor(coordinate[1]));
    const float dx = static_cast<float>(coordinate[0] - left);
    const float dy = static_cast<float>(coordinate[1] - top);
    const float weights[]{(1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy};
    float result = 0;
    for (int tap = 0; tap < 4; ++tap) {
      int x = left + tap % 2, y = top + tap / 2;
      if (border) {
        x = std::clamp(x, 0, c.width - 1);
        y = std::clamp(y, 0, c.height - 1);
      }
      float value = 0;
      if (x >= 0 && x < c.width && y >= 0 && y < c.height) {
        const auto n = point / (c.output_height * c.output_width);
        value = features[((n * c.height + y) * c.width + x) * c.channels + channel];
      }
      result = tap ? std::fma(value, weights[tap], result) : value * weights[tap];
    }
    return result;
  }

  void check(const neat::TensorList& tensors) const {
    const auto& c = geometry;
    require(tensors.size() == 1, "GridSample must publish one output");
    const auto& tensor = tensors.front();
    require(tensor.shape ==
                std::vector<std::int64_t>{batch, c.output_height, c.output_width, c.channels},
            "GridSample output shape");
    require(tensor.dtype == (bfloat ? neat::TensorDType::BFloat16 : neat::TensorDType::Float32),
            "GridSample output dtype");
    require(tensor.layout == neat::TensorLayout::HWC && tensor.route.name == "sampled",
            "GridSample output layout and route");
    const auto bytes = tensor.copy_dense_bytes_tight();
    const auto points = static_cast<std::size_t>(batch) * c.output_height * c.output_width;
    require(bytes.size() == points * c.channels * (bfloat ? 2 : 4), "GridSample output bytes");
    for (std::size_t point = 0; point < points; ++point)
      for (int channel = 0; channel < c.channels; ++channel) {
        const float ref = expected(point, channel);
        const auto i = point * c.channels + channel;
        if (bfloat) {
          std::uint16_t value;
          std::memcpy(&value, bytes.data() + i * 2, 2);
          require(std::isnan(ref) ? (value & 0x7fff) > 0x7f80 : value == bf16(ref),
                  std::string(c.name) + ": BF16 numerical mismatch at " + std::to_string(i));
        } else {
          float value;
          std::memcpy(&value, bytes.data() + i * 4, 4);
          require(std::isnan(ref)
                      ? std::isnan(value)
                      : std::bit_cast<std::uint32_t>(value) == std::bit_cast<std::uint32_t>(ref),
                  std::string(c.name) + ": FP32 numerical mismatch at " + std::to_string(i));
        }
      }
  }
};

inline neat::Graph graph(neat::SimaGridSample2DOptions options,
                         neat::InputMemoryPolicy memory = neat::InputMemoryPolicy::SystemMemory) {
  neat::InputOptions input;
  input.payload_type = neat::PayloadType::Tensor;
  input.memory_policy = memory;
  input.pool_min_buffers = input.pool_max_buffers = 6;
  input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                        "storage=(string)tensorbuffer";
  neat::Graph result;
  result.add(neat::nodes::Input(input));
  result.add(neat::nodes::SimaGridSample2D(options));
  result.add(neat::nodes::Output());
  return result;
}
} // namespace grid_sample_test

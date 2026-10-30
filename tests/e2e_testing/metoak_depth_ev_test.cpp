// Synthetic public Graph qualification. No camera, model, provider headers or
// production numerical helpers. The scalar oracle runs only after EV completion.
#include <neat.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace n = simaai::neat;
namespace {
void require(bool value, const std::string& message) {
  if (!value)
    throw std::runtime_error(message);
}
struct Frame {
  int width, height;
  std::vector<std::uint8_t> y, u, v;
  std::vector<std::uint16_t> disparity;
  float bf;
  std::array<float, 3> projection;
};
Frame fixture(int w, int h, unsigned seed) {
  const auto pixels = std::size_t(w) * h;
  Frame result{w,
               h,
               std::vector<std::uint8_t>(pixels),
               std::vector<std::uint8_t>(pixels / 4),
               std::vector<std::uint8_t>(pixels / 4),
               std::vector<std::uint16_t>(pixels),
               seed % 2 ? 12000.0f : 10000.0f,
               {seed % 2 ? 200.0f : 100.0f, w / 2.0f + 0.25f, h / 2.0f - 0.75f}};
  constexpr std::array<std::uint16_t, 8> values{0, 1, 7, 32, 96, 320, 64000, 65535};
  for (std::size_t i = 0; i < pixels; ++i) {
    result.y[i] = std::uint8_t(i * 73 + seed * 11);
    result.disparity[i] = values[(i + seed) % values.size()];
  }
  for (std::size_t i = 0; i < pixels / 4; ++i) {
    result.u[i] = std::uint8_t(i * 37 + seed * 19);
    result.v[i] = std::uint8_t(255 - i * 51 - seed * 13);
  }
  return result;
}
template <class T>
n::Tensor input(const std::vector<T>& values, const std::vector<std::int64_t>& shape,
                const std::string& name, bool offset) {
  // Include a nonzero source offset; padding is not part of the logical tensor.
  auto data = values;
  if (offset)
    data.insert(data.begin(), T{});
  auto tensor = n::Tensor::from_vector(data, {std::int64_t(data.size())}, n::TensorMemory::EV74);
  tensor.shape = shape;
  tensor.strides_bytes.clear();
  tensor.byte_offset = offset ? sizeof(T) : 0;
  tensor.route.name = name;
  return tensor;
}
n::TensorList inputs(const Frame& f, bool offset) {
  return {
      input(f.y, {f.height, f.width}, "y_src", offset),
      input(f.u, {f.height / 2, f.width / 2}, "u_src", offset),
      input(f.v, {f.height / 2, f.width / 2}, "v_src", offset),
      input(f.disparity, {f.height, f.width}, "disp_src", offset),
      input(std::vector<float>{f.bf}, {1}, "bf_mm_src", offset),
      input(std::vector<float>(f.projection.begin(), f.projection.end()), {3}, "proj_src", offset)};
}
std::array<n::TensorList, 3> shared_inputs(const std::array<Frame, 3>& frames) {
  std::vector<std::uint8_t> bytes;
  std::array<std::array<std::size_t, 6>, 3> offsets{};
  const auto append = [&]<class T>(const T* source, std::size_t count) {
    // Keep a zero guard before each view and only require its natural alignment.
    const auto offset = (bytes.size() + sizeof(T) + alignof(T) - 1) / alignof(T) * alignof(T);
    bytes.resize(offset + count * sizeof(T), 0);
    std::memcpy(bytes.data() + offset, source, count * sizeof(T));
    return offset;
  };
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const auto& f = frames[i];
    offsets[i] = {append(f.y.data(), f.y.size()),
                  append(f.u.data(), f.u.size()),
                  append(f.v.data(), f.v.size()),
                  append(f.disparity.data(), f.disparity.size()),
                  append(&f.bf, 1),
                  append(f.projection.data(), f.projection.size())};
  }
  // Upload once. Every submission reads this same immutable allocation; only
  // the six selected views change, so cached first-frame offsets cannot pass.
  const auto allocation =
      n::Tensor::from_vector(bytes, {std::int64_t(bytes.size())}, n::TensorMemory::EV74);
  constexpr std::array types{n::TensorDType::UInt8,   n::TensorDType::UInt8,
                             n::TensorDType::UInt8,   n::TensorDType::UInt16,
                             n::TensorDType::Float32, n::TensorDType::Float32};
  constexpr std::array names{"y_src", "u_src", "v_src", "disp_src", "bf_mm_src", "proj_src"};
  std::array<n::TensorList, 3> result;
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const auto& f = frames[i];
    const std::array<std::vector<std::int64_t>, 6> shapes{{{f.height, f.width},
                                                           {f.height / 2, f.width / 2},
                                                           {f.height / 2, f.width / 2},
                                                           {f.height, f.width},
                                                           {1},
                                                           {3}}};
    for (std::size_t port = 0; port < names.size(); ++port) {
      auto tensor = allocation;
      tensor.dtype = types[port];
      tensor.shape = shapes[port];
      tensor.strides_bytes.clear();
      tensor.byte_offset = static_cast<std::int64_t>(offsets[i][port]);
      tensor.route = {};
      tensor.route.name = names[port];
      tensor.route.memory_index = 0;
      result[i].push_back(std::move(tensor));
    }
  }
  return result;
}
template <class T>
std::vector<T> values(const n::Tensor& tensor, n::TensorDType dtype,
                      const std::vector<std::int64_t>& shape, const std::string& name) {
  require(tensor.dtype == dtype && tensor.shape == shape, name + ": dtype/shape");
  require(tensor.route.name == name, name + ": published name");
  const auto bytes = tensor.copy_dense_bytes_tight();
  std::size_t size = sizeof(T);
  for (auto extent : shape)
    size *= std::size_t(extent);
  require(bytes.size() == size, name + ": byte extent");
  std::vector<T> result(bytes.size() / sizeof(T));
  std::memcpy(result.data(), bytes.data(), bytes.size());
  return result;
}
void validate(const Frame& f, const n::TensorList& outputs) {
  require(outputs.size() == 3, "RGB, depth and XYZ must all be published");
  const auto rgb =
      values<std::uint8_t>(outputs[0], n::TensorDType::UInt8, {f.height, f.width, 3}, "rgb_dst");
  const auto depth =
      values<std::uint16_t>(outputs[1], n::TensorDType::UInt16, {f.height, f.width}, "depth_dst");
  const auto xyz =
      values<float>(outputs[2], n::TensorDType::Float32, {f.height, f.width, 3}, "points_dst");
  const float scale = f.bf * 32.0f;
  const float reciprocal_focal = 1.0f / f.projection[0];
  for (int row = 0; row < f.height; ++row) {
    for (int col = 0; col < f.width; ++col) {
      const auto i = std::size_t(row) * f.width + col;
      const auto uv = std::size_t(row / 2) * (f.width / 2) + col / 2;
      const int y = int(f.y[i]) - 16, u = int(f.u[uv]) - 128, v = int(f.v[uv]) - 128;
      const std::array channels{(298 * y + 409 * v + 128) >> 8,
                                (298 * y - 100 * u - 208 * v + 128) >> 8,
                                (298 * y + 516 * u + 128) >> 8};
      for (std::size_t c = 0; c < 3; ++c)
        require(rgb[3 * i + c] == std::clamp(channels[c], 0, 255),
                "RGB differs from scalar oracle");
      if (!f.disparity[i]) {
        require(depth[i] == 0 && std::isnan(xyz[3 * i]) && std::isnan(xyz[3 * i + 1]) &&
                    std::isnan(xyz[3 * i + 2]),
                "invalid disparity output");
        continue;
      }
      const float mm = scale / float(f.disparity[i]);
      const auto expected_depth = std::uint16_t(std::min(65535.0f, mm) + 0.5f);
      require(depth[i] == expected_depth, "depth differs from scalar oracle");
      const float z = mm / 1000.0f;
      const std::array expected{(float(col) - f.projection[1]) * z * reciprocal_focal,
                                (float(row) - f.projection[2]) * z * reciprocal_focal, z};
      for (std::size_t c = 0; c < 3; ++c)
        require(std::isfinite(xyz[3 * i + c]) &&
                    std::abs(xyz[3 * i + c] - expected[c]) <= 1e-5f + 1e-5f * std::abs(expected[c]),
                "XYZ differs from scalar oracle");
    }
  }
}
void run_scenario(const std::array<Frame, 3>& data, const std::array<n::TensorList, 3>& frames,
                  bool offset, int count, bool shared_allocation = false) {
  const auto width = data.front().width, height = data.front().height;
  n::InputOptions source;
  source.payload_type = n::PayloadType::Tensor;
  source.memory_policy = n::InputMemoryPolicy::Ev74;
  source.buffer_name = "y_src";
  source.width = source.max_width = width;
  source.height = source.max_height = height;
  source.pool_min_buffers = source.pool_max_buffers = 4;
  source.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                         "storage=(string)tensorbuffer";
  n::MetoakDepthOptions depth;
  depth.width = width;
  depth.height = height;
  depth.num_buffers = 4;
  n::Graph graph;
  graph.add(n::nodes::Input(source));
  graph.add(n::nodes::MetoakDepth(depth));
  graph.add(n::nodes::Output());
  n::RunOptions options;
  options.output_memory = n::OutputMemory::Owned;
  options.queue_depth = 1;
  std::vector<n::TensorList> outputs;
  auto run = graph.build(frames.front(), options);
  const auto start = std::chrono::steady_clock::now();
  try {
    for (int frame = 0; frame < count; ++frame)
      outputs.push_back(run.run(frames[std::size_t(frame) % frames.size()], 30000));
  } catch (...) {
    run.close();
    throw;
  }
  run.close();
  const auto elapsed =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  std::cout << "EV_RUN_CLOSED frames=" << outputs.size() << " outputs=owned\n";
  // Compare retained results after all input/output pool reuse and normal close.
  for (std::size_t frame = 0; frame < outputs.size(); ++frame)
    validate(data[frame % data.size()], outputs[frame]);
  std::cout << "METOAK_CASE width=" << width << " height=" << height << " offset=" << offset
            << " frames=" << count << " elapsed_ms_including_close=" << elapsed;
  if (shared_allocation)
    std::cout << " shared_allocation=1";
  std::cout << " PASS\n";
}
void scenario(int width, int height, bool offset, int count) {
  const std::array data{fixture(width, height, 0), fixture(width, height, 1),
                        fixture(width, height, 2)};
  const std::array frames{inputs(data[0], offset), inputs(data[1], offset),
                          inputs(data[2], offset)};
  run_scenario(data, frames, offset, count);
}
void shared_allocation_scenario() {
  const std::array data{fixture(10, 14, 0), fixture(10, 14, 1), fixture(10, 14, 2)};
  const auto frames = shared_inputs(data);
  run_scenario(data, frames, true, 12, true);
}
} // namespace
int main() {
  std::cout << std::unitbuf;
  try {
    scenario(8, 8, false, 12);
    scenario(10, 14, true, 12); // 35-byte chroma planes expose alignment/padding mistakes.
    scenario(18, 10, true, 12); // SIMD row tail and odd chroma extent.
    scenario(640, 360, false, 24);
    scenario(640, 360, true, 24);
    scenario(8, 8, true, 12);     // Stop and reopen at small geometry after the sensor workload.
    shared_allocation_scenario(); // Same allocation, six views, changing offsets A/B/C/A.
    std::cout << "METOAK_SUITE completed=7 frames=108 failed=0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "METOAK_FAILURE " << error.what() << '\n';
    return 1;
  }
}

#pragma once

// Public-API fixtures and independent numerical oracles. No provider headers,
// physical route IDs, native numerical calls, or compatibility-copy paths.
#include <neat.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace visual_test {
namespace neat = simaai::neat;
constexpr int kTimeoutMs = 30000;
using Image = std::vector<std::uint8_t>;

class Failure final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};
inline void require(bool condition, const std::string& message) {
  if (!condition)
    throw Failure(message);
}

class Suite {
public:
  Suite(std::string name, std::vector<std::string> required)
      : name_(std::move(name)), required_(std::move(required)) {}
  template <class Fn> void run(std::string_view name, Fn&& fn) {
    require(completed_ < required_.size() && required_[completed_] == name,
            "scenario inventory/order mismatch: " + std::string(name));
    std::cout << "EV_CASE kernel=" << name_ << " case=" << name << " phase=start\n";
    try {
      fn();
      std::cout << "EV_CASE kernel=" << name_ << " case=" << name << " result=PASS\n";
    } catch (const Failure& error) {
      // Assertions are made only after each public Run has closed normally.
      ++failed_;
      std::cerr << "EV_CASE kernel=" << name_ << " case=" << name
                << " result=FAIL reason=" << error.what() << '\n';
    }
    ++completed_;
  }
  void finish() const {
    require(completed_ == required_.size(), "required scenario was not executed");
    std::cout << "EV_SUITE kernel=" << name_ << " completed=" << completed_
              << " required=" << required_.size() << " failed=" << failed_ << '\n';
    require(failed_ == 0, name_ + ": numerical/contract scenario failures");
  }

private:
  std::string name_;
  std::vector<std::string> required_;
  std::size_t completed_ = 0;
  std::size_t failed_ = 0;
};

template <class Fn> int main_test(int argc, Fn&& fn) {
  std::cout << std::unitbuf;
  try {
    require(argc == 1, "canonical EV test takes no arguments; every scenario is required");
    fn();
    return 0;
  } catch (const Failure& error) {
    std::cerr << "[ASSERTION_AFTER_CLOSE] " << error.what() << '\n';
    return 1;
  } catch (const std::exception& error) {
    // An unexpected runtime failure never becomes an expected numerical PASS.
    std::cerr << "[RUNTIME_OR_SETUP_FAILURE] " << error.what() << '\n';
    return 2;
  }
}

inline void name(neat::Tensor& tensor, std::string_view value) {
  tensor.route.name = value;
}
inline neat::Tensor image(const Image& pixels, int width, int height, int batch,
                          std::string_view route = "input_image") {
  require(pixels.size() == static_cast<std::size_t>(width) * height * batch,
          "fixture image byte count");
  auto result = neat::Tensor::from_vector(pixels, {batch, height, width}, neat::TensorMemory::EV74);
  result.layout = neat::TensorLayout::HW;
  result.axis_semantics = {neat::TensorAxisSemantic::N, neat::TensorAxisSemantic::H,
                           neat::TensorAxisSemantic::W};
  name(result, route);
  return result;
}
inline neat::Tensor seeds(const std::vector<std::int32_t>& points, int count, int batch = 1) {
  require(points.size() == static_cast<std::size_t>(2 * count * batch), "fixture seed count");
  auto result = neat::Tensor::from_vector(points, {batch, count, 2}, neat::TensorMemory::EV74);
  name(result, "input_points");
  return result;
}
inline neat::InputOptions input_options(int width, int height, int batch = 1,
                                        bool tensor_set = false) {
  neat::InputOptions result;
  result.payload_type = neat::PayloadType::Tensor;
  result.format = neat::FormatTag::UINT8;
  result.width = result.max_width = width;
  result.height = height;
  result.max_height = height * batch;
  result.depth = result.max_depth = 1;
  result.memory_policy = neat::InputMemoryPolicy::Ev74;
  result.buffer_name = tensor_set ? "prev_image" : "input_image";
  result.pool_min_buffers = result.pool_max_buffers = 4;
  if (tensor_set)
    result.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                           "storage=(string)tensorbuffer";
  return result;
}
inline neat::RunOptions run_options() {
  neat::RunOptions result;
  result.output_memory = neat::OutputMemory::Owned;
  result.queue_depth = 1;
  result.input_timeout_ms = kTimeoutMs;
  return result;
}
inline std::vector<neat::TensorList> execute(neat::Graph& graph,
                                             const std::vector<neat::TensorList>& frames) {
  require(!frames.empty(), "empty fixture sequence");
  auto run = graph.build(frames.front(), run_options());
  std::vector<neat::TensorList> results;
  try {
    for (const auto& frame : frames)
      results.push_back(run.run(frame, kTimeoutMs));
  } catch (...) {
    run.close();
    throw;
  }
  run.close();
  std::cout << "EV_RUN_CLOSED frames=" << results.size() << " outputs=owned\n";
  return results;
}

template <class T>
std::vector<T> values(const neat::Tensor& tensor, neat::TensorDType dtype,
                      const std::vector<std::int64_t>& shape) {
  require(tensor.dtype == dtype && tensor.shape == shape, "public output dtype/shape mismatch");
  const auto bytes = tensor.copy_dense_bytes_tight();
  std::size_t count = 1;
  for (const auto extent : shape)
    count *= static_cast<std::size_t>(extent);
  require(bytes.size() == count * sizeof(T), "public output payload extent mismatch");
  std::vector<T> result(count);
  if (!bytes.empty())
    std::memcpy(result.data(), bytes.data(), bytes.size());
  return result;
}

struct Feature {
  int x;
  int y;
  int score;
  bool operator==(const Feature&) const = default;
};
using Features = std::vector<Feature>;
inline void rank(Features& features) {
  std::sort(features.begin(), features.end(), [](const Feature& a, const Feature& b) {
    if (a.score != b.score)
      return a.score > b.score;
    return a.y != b.y ? a.y < b.y : a.x < b.x;
  });
}
struct MarkerImage {
  int width = 320;
  int height = 240;
  Image pixels;
  Features corners;
  explicit MarkerImage(int w = 320, int h = 240) : width(w), height(h), pixels(w * h, 64) {}
  void add(int x, int y, int score = 127) {
    require(x >= 3 && x <= width - 4 && y >= 3 && y <= height - 4 && score > 30 && score <= 191,
            "invalid literal FAST marker");
    for (const auto& other : corners)
      require((x - other.x) * (x - other.x) + (y - other.y) * (y - other.y) >= 64,
              "literal FAST markers overlap");
    corners.push_back({x, y, score});
    pixels[y * width + x] = static_cast<std::uint8_t>(64 + score);
  }
  void descriptor_texture(unsigned generation) {
    // Sub-threshold background variation yields distinct BRIEF words without
    // creating FAST corners. Restore each radius-3 neighborhood so its literal
    // ring contrast remains exactly the authored marker score.
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {
        unsigned v = unsigned(x) * 73856093U ^ unsigned(y) * 19349663U ^ generation * 83492791U;
        v ^= v >> 13;
        v *= 1274126177U;
        pixels[y * width + x] = static_cast<std::uint8_t>(57 + (v % 15));
      }
    for (const auto& p : corners)
      for (int dy = -3; dy <= 3; ++dy)
        for (int dx = -3; dx <= 3; ++dx)
          pixels[(p.y + dy) * width + p.x + dx] = 64;
    for (const auto& p : corners)
      pixels[p.y * width + p.x] = static_cast<std::uint8_t>(64 + p.score);
  }
};
struct DetectorConfig {
  int capacity = 96;
  int columns = 8;
  int rows = 6;
  int distance = 0;
};
struct DetectorCase {
  std::string name;
  DetectorConfig config;
  std::vector<MarkerImage> frames;
  int batch = 1;
};
inline MarkerImage grid_markers() {
  MarkerImage image;
  for (int cy = 0; cy < 6; ++cy)
    for (int cx = 0; cx < 8; ++cx) {
      const int score = cy == 0 || cx == 0 ? 180 : 100;
      image.add(cx * 40 + 16, cy * 40 + 16, score);
      image.add(cx * 40 + 24, cy * 40 + 16, score - 20);
      image.add(cx * 40 + 16, cy * 40 + 24, score - 40);
    }
  return image;
}
inline Features selected_markers(const MarkerImage& image, DetectorConfig config,
                                 int legal_border = 3) {
  // Independent selection on authored literal corners, not a FAST or native
  // provider call. This follows the documented quota/tie/no-refill contract.
  auto candidates = image.corners;
  std::erase_if(candidates, [&](const auto& p) {
    return p.x < legal_border || p.x >= image.width - legal_border || p.y < legal_border ||
           p.y >= image.height - legal_border;
  });
  rank(candidates);
  const int cells = config.columns * config.rows;
  std::vector<int> counts(cells, 0);
  Features frontier;
  for (const auto& p : candidates) {
    const int cell =
        (p.y * config.rows / image.height) * config.columns + p.x * config.columns / image.width;
    const int quota = config.capacity / cells + (cell < config.capacity % cells ? 1 : 0);
    if (counts[cell]++ < quota)
      frontier.push_back(p);
  }
  Features result;
  for (const auto& p : frontier) {
    const bool close = std::any_of(result.begin(), result.end(), [&](const auto& q) {
      const auto dx = std::int64_t(p.x) - q.x, dy = std::int64_t(p.y) - q.y;
      return dx * dx + dy * dy < std::int64_t(config.distance) * config.distance;
    });
    if (!close)
      result.push_back(p);
  }
  return result;
}
inline std::vector<std::string> detector_names(bool descriptor) {
  return {"grid_8x6",
          "grid_x",
          "grid_y",
          "global_rank",
          "quota_remainder",
          "quota_small",
          "minimum_distance",
          "cross_cell",
          "distance_equality",
          "sparse_ties_no_refill",
          descriptor ? "brief_border" : "first_last_valid_edges",
          "frame_reuse",
          "batch_two"};
}
inline std::vector<DetectorCase> detector_cases(bool descriptor) {
  const auto grid = grid_markers();
  std::vector<DetectorCase> result{
      {"grid_8x6", {96, 8, 6, 0}, {grid}},         {"grid_x", {96, 8, 1, 0}, {grid}},
      {"grid_y", {96, 1, 6, 0}, {grid}},           {"global_rank", {96, 1, 1, 0}, {grid}},
      {"quota_remainder", {100, 8, 6, 0}, {grid}}, {"quota_small", {23, 8, 6, 0}, {grid}},
      {"minimum_distance", {96, 8, 6, 10}, {grid}}};
  MarkerImage cross;
  cross.add(156, 100, 120);
  cross.add(164, 100, 100);
  cross.add(40, 100, 90);
  cross.add(280, 100, 80);
  result.push_back({"cross_cell", {4, 2, 1, 10}, {cross}});
  MarkerImage equality;
  equality.add(30, 80, 120);
  equality.add(40, 80, 100);
  equality.add(70, 80, 80);
  result.push_back({"distance_equality", {8, 1, 1, 10}, {equality}});
  MarkerImage sparse;
  sparse.add(16, 40, 100);
  sparse.add(32, 40, 100);
  sparse.add(48, 40, 100);
  result.push_back({"sparse_ties_no_refill", {8, 4, 1, 0}, {sparse}});
  MarkerImage boundary(321, 241);
  if (descriptor) {
    boundary.add(50, 14);
    boundary.add(100, 15);
    boundary.add(14, 100);
    boundary.add(15, 150);
    boundary.add(305, 180);
    boundary.add(306, 210);
  } else {
    boundary.add(3, 3);
    boundary.add(317, 3);
    boundary.add(3, 237);
    boundary.add(317, 237);
    boundary.add(100, 4);
    boundary.add(200, 100);
  }
  result.push_back(
      {descriptor ? "brief_border" : "first_last_valid_edges", {16, 1, 1, 0}, {boundary}});
  MarkerImage empty, changed;
  changed.add(50, 50, 180);
  changed.add(250, 150, 100);
  result.push_back({"frame_reuse", {96, 8, 6, 0}, {grid, empty, changed}});
  MarkerImage other;
  other.add(100, 100, 180);
  other.add(200, 180, 100);
  result.push_back({"batch_two", {8, 1, 1, 0}, {changed, other}, 2});
  if (descriptor) {
    unsigned generation = 1;
    for (auto& scenario : result)
      for (auto& frame : scenario.frames)
        frame.descriptor_texture(generation++);
  }
  return result;
}
inline std::vector<Features> read_features(const neat::Tensor& output, int capacity, int batch) {
  auto words = values<std::int32_t>(output, neat::TensorDType::Int32, {batch, 1 + 3 * capacity});
  std::vector<Features> result(batch);
  for (int b = 0; b < batch; ++b) {
    const auto base = static_cast<std::size_t>(b) * (1 + 3 * capacity);
    const auto count = words[base];
    require(count >= 0 && count <= capacity, "invalid feature count prefix");
    for (int i = 0; i < count; ++i)
      result[b].push_back(
          {words[base + 1 + 3 * i], words[base + 2 + 3 * i], words[base + 3 + 3 * i]});
  }
  return result;
}
inline void compare_features(const Features& actual, const Features& expected) {
  require(actual.size() == expected.size(),
          "feature count expected=" + std::to_string(expected.size()) +
              " actual=" + std::to_string(actual.size()));
  for (std::size_t i = 0; i < expected.size(); ++i)
    require(actual[i] == expected[i], "ordered feature mismatch at index " + std::to_string(i));
}
inline std::vector<neat::TensorList> detector_frames(const DetectorCase& test) {
  if (test.batch == 2) {
    auto pixels = test.frames[0].pixels;
    pixels.insert(pixels.end(), test.frames[1].pixels.begin(), test.frames[1].pixels.end());
    return {{image(pixels, test.frames[0].width, test.frames[0].height, 2)}};
  }
  std::vector<neat::TensorList> result;
  for (const auto& frame : test.frames)
    result.push_back({image(frame.pixels, frame.width, frame.height, 1)});
  return result;
}
inline void check_detector_output(const DetectorCase& test,
                                  const std::vector<neat::TensorList>& results,
                                  std::size_t feature_slot, std::size_t output_count,
                                  int legal_border = 3) {
  require(results.size() == (test.batch == 2 ? 1 : test.frames.size()), "frame count mismatch");
  for (std::size_t frame = 0; frame < results.size(); ++frame) {
    require(results[frame].size() == output_count, "public output tuple arity");
    const auto features =
        read_features(results[frame][feature_slot], test.config.capacity, test.batch);
    for (int b = 0; b < test.batch; ++b)
      compare_features(features[b], selected_markers(test.frames[test.batch == 2 ? b : frame],
                                                     test.config, legal_border));
  }
}

inline std::array<std::uint32_t, 8> brief(const MarkerImage& image, const Feature& feature) {
  require(feature.x >= 15 && feature.x < image.width - 15 && feature.y >= 15 &&
              feature.y < image.height - 15,
          "BRIEF oracle patch outside image");
  std::array<std::uint32_t, 8> result{};
  std::uint32_t state = 0x9E3779B9U;
  auto offset = [&]() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<int>(state % 31U) - 15;
  };
  for (int bit = 0; bit < 256; ++bit) {
    const int ax = offset(), ay = offset(), bx = offset(), by = offset();
    if (image.pixels[(feature.y + ay) * image.width + feature.x + ax] <
        image.pixels[(feature.y + by) * image.width + feature.x + bx])
      result[bit / 32] |= std::uint32_t{1} << (bit % 32);
  }
  return result;
}

inline Image texture(int width, int height, unsigned seed) {
  Image result(width * height);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      unsigned value = unsigned(x) * 73856093U ^ unsigned(y) * 19349663U ^ seed * 83492791U;
      value ^= value >> 13;
      value *= 1274126177U;
      result[y * width + x] = static_cast<std::uint8_t>(value >> 24);
    }
  for (int pass = 0; pass < 3; ++pass) {
    const auto source = result;
    for (int y = 1; y + 1 < height; ++y)
      for (int x = 1; x + 1 < width; ++x) {
        unsigned sum = 0;
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx)
            sum += source[(y + dy) * width + x + dx] * (dx == 0 ? 2U : 1U) * (dy == 0 ? 2U : 1U);
        result[y * width + x] = static_cast<std::uint8_t>((sum + 8U) / 16U);
      }
  }
  return result;
}
inline Image shifted(const Image& source, int width, int height, int dx, int dy) {
  Image result(source.size());
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      result[y * width + x] =
          source[std::clamp(y - dy, 0, height - 1) * width + std::clamp(x - dx, 0, width - 1)];
  return result;
}
} // namespace visual_test

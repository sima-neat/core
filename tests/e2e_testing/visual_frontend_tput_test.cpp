#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "gst/GstHelpers.h"
#include "test_utils.h"

#include <neat.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

int env_int_local(const char* key, int def) {
  const char* v = std::getenv(key);
  if (!v || !*v) {
    return def;
  }
  return std::atoi(v);
}

std::string env_string(const char* key, std::string def = {}) {
  const char* v = std::getenv(key);
  if (!v || !*v) {
    return def;
  }
  return std::string(v);
}

std::string shape_string(const std::vector<int64_t>& shape) {
  std::ostringstream os;
  os << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i) {
      os << ",";
    }
    os << shape[i];
  }
  os << "]";
  return os.str();
}

std::string dtype_string(simaai::neat::TensorDType dtype) {
  switch (dtype) {
  case simaai::neat::TensorDType::UInt8:
    return "UInt8";
  case simaai::neat::TensorDType::Int8:
    return "Int8";
  case simaai::neat::TensorDType::UInt16:
    return "UInt16";
  case simaai::neat::TensorDType::Int16:
    return "Int16";
  case simaai::neat::TensorDType::Int32:
    return "Int32";
  case simaai::neat::TensorDType::BFloat16:
    return "BFloat16";
  case simaai::neat::TensorDType::Float32:
    return "Float32";
  case simaai::neat::TensorDType::Float64:
    return "Float64";
  }
  return "Unknown";
}

void stamp_route(simaai::neat::Tensor& tensor, const std::string& name, int logical, int physical) {
  tensor.route.name = name;
  tensor.route.backend_name = name;
  tensor.route.segment_name = name;
  tensor.route.logical_index = logical;
  tensor.route.physical_index = physical;
  tensor.route.memory_index = 0;
  tensor.route.route_slot = logical;
  tensor.route.backend_output_index = logical;
}

int positive_mod(int value, int mod) {
  if (mod <= 0) {
    return 0;
  }
  int out = value % mod;
  return out < 0 ? out + mod : out;
}

std::uint8_t patterned_u8_pixel(int width, int height, int batch, int x, int y, std::uint8_t salt) {
  int value = 92 + ((x * 3 + y * 5 + static_cast<int>(salt) + batch * 11) & 0x0F);

  const int patch = std::max(3, std::min({9, std::max(3, width / 10), std::max(3, height / 10)}));
  const int margin = std::min(std::max(18, patch + 16), std::max(1, std::min(width, height) / 2));
  const int usable_w = std::max(1, width - 2 * margin);
  const int usable_h = std::max(1, height - 2 * margin);
  const int marker_count = std::max(1, std::min(6, (width * height) / (128 * 128) + 1));
  for (int marker = 0; marker < marker_count; ++marker) {
    const int cx =
        margin + positive_mod(static_cast<int>(salt) * 29 + batch * 53 + marker * 101, usable_w);
    const int cy =
        margin + positive_mod(static_cast<int>(salt) * 31 + batch * 47 + marker * 83, usable_h);
    const int dx = std::abs(x - cx);
    const int dy = std::abs(y - cy);
    if (dx <= patch && dy <= patch) {
      const bool inner = dx <= std::max(1, patch / 2) && dy <= std::max(1, patch / 2);
      value = inner ? 235 : ((dx == patch || dy == patch) ? 15 : 35);
    }
  }
  return static_cast<std::uint8_t>(value & 0xFF);
}

simaai::neat::Tensor make_u8_batch_tensor(int width, int height, int batch_size,
                                          const std::string& name, int logical, int physical,
                                          std::uint8_t salt) {
  require(width > 0 && height > 0 && batch_size > 0, "u8 batch tensor dimensions must be positive");
  std::vector<std::uint8_t> data(static_cast<std::size_t>(width) *
                                 static_cast<std::size_t>(height) *
                                 static_cast<std::size_t>(batch_size));
  const std::size_t image_bytes =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  for (int b = 0; b < batch_size; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        data[static_cast<std::size_t>(b) * image_bytes +
             static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
             static_cast<std::size_t>(x)] = patterned_u8_pixel(width, height, b, x, y, salt);
      }
    }
  }
  // Native visual EV kernels consume the same contiguous bytes as packed batches, but Neat
  // exposes batch as a first-class leading dimension for user-facing tensor shapes.
  auto tensor = simaai::neat::Tensor::from_vector(data, {batch_size, height, width},
                                                  simaai::neat::TensorMemory::EV74);
  tensor.layout = simaai::neat::TensorLayout::HW;
  tensor.axis_semantics = {simaai::neat::TensorAxisSemantic::N, simaai::neat::TensorAxisSemantic::H,
                           simaai::neat::TensorAxisSemantic::W};
  stamp_route(tensor, name, logical, physical);
  return tensor;
}

simaai::neat::Tensor make_u8_batch_tensor_from_data(int width, int height, int batch_size,
                                                    const std::string& name, int logical,
                                                    int physical,
                                                    const std::vector<std::uint8_t>& data) {
  require(width > 0 && height > 0 && batch_size > 0, "u8 batch tensor dimensions must be positive");
  require(data.size() == static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                             static_cast<std::size_t>(batch_size),
          name + ": u8 batch tensor data size mismatch");
  auto tensor = simaai::neat::Tensor::from_vector(data, {batch_size, height, width},
                                                  simaai::neat::TensorMemory::EV74);
  tensor.layout = simaai::neat::TensorLayout::HW;
  tensor.axis_semantics = {simaai::neat::TensorAxisSemantic::N, simaai::neat::TensorAxisSemantic::H,
                           simaai::neat::TensorAxisSemantic::W};
  stamp_route(tensor, name, logical, physical);
  return tensor;
}

simaai::neat::Tensor make_u8_tensor(int width, int height, const std::string& name, int logical,
                                    int physical, std::uint8_t salt) {
  return make_u8_batch_tensor(width, height, 1, name, logical, physical, salt);
}

std::uint8_t klt_texture_pixel(int width, int height, int batch, int x, int y, std::uint8_t salt) {
  (void)width;
  (void)height;
  const int checker = (((x / 7) + (y / 5) + batch) & 1) ? 58 : -47;
  const int cross = ((((x + 3 * batch) ^ (y * 5 + static_cast<int>(salt))) & 15) - 8) * 7;
  const int fine = ((x * 13 + y * 19 + x * y + batch * 37 + static_cast<int>(salt) * 11) & 63) - 31;
  const int value = std::clamp(128 + checker + cross + fine, 0, 255);
  return static_cast<std::uint8_t>(value);
}

std::pair<simaai::neat::Tensor, simaai::neat::Tensor>
make_klt_image_pair_tensors(int width, int height, int batch_size, const std::string& prev_name,
                            const std::string& cur_name, int dx, int dy) {
  const std::size_t image_bytes =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  std::vector<std::uint8_t> prev(image_bytes * static_cast<std::size_t>(batch_size));
  std::vector<std::uint8_t> cur(prev.size(), 0U);
  for (int b = 0; b < batch_size; ++b) {
    const std::size_t batch_offset = static_cast<std::size_t>(b) * image_bytes;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        prev[batch_offset + static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
             static_cast<std::size_t>(x)] = klt_texture_pixel(width, height, b, x, y, 47);
      }
    }
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int sx = x - dx;
        const int sy = y - dy;
        const std::size_t dst = batch_offset +
                                static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                static_cast<std::size_t>(x);
        if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
          cur[dst] =
              prev[batch_offset + static_cast<std::size_t>(sy) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(sx)];
        } else {
          cur[dst] = klt_texture_pixel(width, height, b, x, y, 101);
        }
      }
    }
  }
  return {make_u8_batch_tensor_from_data(width, height, batch_size, prev_name, 0, 0, prev),
          make_u8_batch_tensor_from_data(width, height, batch_size, cur_name, 1, 1, cur)};
}

simaai::neat::Tensor make_i32_points_batch_tensor(int width, int height, int batch_size,
                                                  int num_points, int win_half, int max_level,
                                                  int dx, int dy, const std::string& name,
                                                  int logical, int physical) {
  require(num_points > 0 && batch_size > 0, "KLT points tensor dimensions must be positive");
  std::vector<std::int32_t> data(static_cast<std::size_t>(batch_size) *
                                 static_cast<std::size_t>(num_points) * 2U);
  const int level_scale = 1 << std::max(0, max_level);
  const int margin =
      std::max(8, (win_half + 6) * level_scale + std::max(std::abs(dx), std::abs(dy)) + 4);
  const int usable_w = std::max(1, width - 2 * margin);
  const int usable_h = std::max(1, height - 2 * margin);
  const int cols =
      std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(num_points) *
                                                       std::max(1, width) / std::max(1, height)))));
  const int rows = std::max(1, (num_points + cols - 1) / cols);
  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < num_points; ++i) {
      const int col = i % cols;
      const int row = i / cols;
      const int jitter_x = positive_mod(b * 7 + i * 3, std::max(1, usable_w / std::max(1, cols)));
      const int jitter_y = positive_mod(b * 5 + i * 2, std::max(1, usable_h / std::max(1, rows)));
      int x = margin + ((2 * col + 1) * usable_w) / (2 * cols) + jitter_x / 3;
      int y = margin + ((2 * row + 1) * usable_h) / (2 * rows) + jitter_y / 3;
      x = std::clamp(x, 0, width - 1);
      y = std::clamp(y, 0, height - 1);
      const std::size_t offset =
          (static_cast<std::size_t>(b) * static_cast<std::size_t>(num_points) +
           static_cast<std::size_t>(i)) *
          2U;
      data[offset + 0U] = x;
      data[offset + 1U] = y;
    }
  }
  auto tensor = simaai::neat::Tensor::from_vector(data, {batch_size, num_points, 2},
                                                  simaai::neat::TensorMemory::EV74);
  tensor.layout = simaai::neat::TensorLayout::HW;
  tensor.axis_semantics = {simaai::neat::TensorAxisSemantic::N, simaai::neat::TensorAxisSemantic::H,
                           simaai::neat::TensorAxisSemantic::W};
  stamp_route(tensor, name, logical, physical);
  return tensor;
}

struct CpuFeature {
  int x = 0;
  int y = 0;
  int score = 0;
};

bool operator==(const CpuFeature& a, const CpuFeature& b) {
  return a.x == b.x && a.y == b.y && a.score == b.score;
}

std::string feature_string(const CpuFeature& f) {
  return "(" + std::to_string(f.x) + "," + std::to_string(f.y) + "," + std::to_string(f.score) +
         ")";
}

constexpr std::array<int, 16> kFastDx = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
constexpr std::array<int, 16> kFastDy = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};
constexpr int kFastRing = 16;
constexpr int kFastArc = 9;
constexpr int kFastRadius = 3;
constexpr int kBriefPatch = 15;
constexpr int kBriefWords = 8;
constexpr int kBriefTests = 256;

int cpu_fast_corner_score(int ip, const std::array<int, kFastRing>& ring) {
  std::array<int, kFastRing> diff{};
  for (int k = 0; k < kFastRing; ++k) {
    diff[static_cast<std::size_t>(k)] = ring[static_cast<std::size_t>(k)] - ip;
  }

  int bright_max = -1000;
  int dark_max = -1000;
  for (int start = 0; start < kFastRing; ++start) {
    int min_bright = 1000;
    int min_dark = 1000;
    for (int j = 0; j < kFastArc; ++j) {
      const int value = diff[static_cast<std::size_t>((start + j) & 15)];
      min_bright = std::min(min_bright, value);
      min_dark = std::min(min_dark, -value);
    }
    bright_max = std::max(bright_max, min_bright);
    dark_max = std::max(dark_max, min_dark);
  }
  return std::max(bright_max, dark_max);
}

int cpu_fast_score_at(const std::vector<std::uint8_t>& image, int width, int height, int threshold,
                      int x, int y) {
  if (x < kFastRadius || y < kFastRadius || x > width - 1 - kFastRadius ||
      y > height - 1 - kFastRadius) {
    return 0;
  }
  const int ip = image[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                       static_cast<std::size_t>(x)];
  const int high = ip + threshold;
  const int low = ip - threshold;
  std::array<int, kFastRing> ring{};
  int brighter = 0;
  int darker = 0;
  for (int k = 0; k < kFastRing; ++k) {
    const int rx = x + kFastDx[static_cast<std::size_t>(k)];
    const int ry = y + kFastDy[static_cast<std::size_t>(k)];
    const int rv = image[static_cast<std::size_t>(ry) * static_cast<std::size_t>(width) +
                         static_cast<std::size_t>(rx)];
    ring[static_cast<std::size_t>(k)] = rv;
    if (rv > high) {
      brighter |= 1 << k;
    }
    if (rv < low) {
      darker |= 1 << k;
    }
  }

  int bright_run = 0;
  int max_bright_run = 0;
  int dark_run = 0;
  int max_dark_run = 0;
  for (int k = 0; k < 32; ++k) {
    const int bit = 1 << (k & 15);
    if ((brighter & bit) != 0) {
      max_bright_run = std::max(max_bright_run, ++bright_run);
    } else {
      bright_run = 0;
    }
    if ((darker & bit) != 0) {
      max_dark_run = std::max(max_dark_run, ++dark_run);
    } else {
      dark_run = 0;
    }
  }
  return (max_bright_run >= kFastArc || max_dark_run >= kFastArc) ? cpu_fast_corner_score(ip, ring)
                                                                  : 0;
}

std::vector<CpuFeature> cpu_fast_reference(const std::vector<std::uint8_t>& image, int width,
                                           int height, int threshold, int emit_radius) {
  require(width > 0 && height > 0, "CPU FAST reference requires positive geometry");
  require(image.size() == static_cast<std::size_t>(width) * static_cast<std::size_t>(height),
          "CPU FAST reference image size mismatch");
  std::vector<int> scores(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
  for (int y = kFastRadius; y <= height - 1 - kFastRadius; ++y) {
    for (int x = kFastRadius; x <= width - 1 - kFastRadius; ++x) {
      scores[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
             static_cast<std::size_t>(x)] =
          cpu_fast_score_at(image, width, height, threshold, x, y);
    }
  }

  std::vector<CpuFeature> features;
  if (width <= 2 * emit_radius || height <= 2 * emit_radius) {
    return features;
  }
  for (int y = emit_radius; y <= height - 1 - emit_radius; ++y) {
    for (int x = emit_radius; x <= width - 1 - emit_radius; ++x) {
      const auto idx = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                       static_cast<std::size_t>(x);
      const int score = scores[idx];
      if (score <= 0) {
        continue;
      }
      bool suppressed = false;
      for (int dy = -1; dy <= 1 && !suppressed; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0) {
            continue;
          }
          const int nscore =
              scores[static_cast<std::size_t>(y + dy) * static_cast<std::size_t>(width) +
                     static_cast<std::size_t>(x + dx)];
          // The EV kernels keep equal-score ties; only a strictly larger neighbor suppresses.
          if (score < nscore) {
            suppressed = true;
            break;
          }
        }
      }
      if (!suppressed) {
        features.push_back({x, y, score});
      }
    }
  }
  return features;
}

const std::array<int, kBriefTests * 4>& brief_pattern() {
  static const std::array<int, kBriefTests * 4> pattern = [] {
    std::array<int, kBriefTests * 4> out{};
    std::uint32_t state = 0x9E3779B9u;
    for (int i = 0; i < kBriefTests * 4; ++i) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      out[static_cast<std::size_t>(i)] =
          static_cast<int>(state % static_cast<std::uint32_t>(2 * kBriefPatch + 1)) - kBriefPatch;
    }
    return out;
  }();
  return pattern;
}

std::array<std::int32_t, kBriefWords> cpu_brief_descriptor(const std::vector<std::uint8_t>& image,
                                                           int width, int height,
                                                           const CpuFeature& feature,
                                                           const std::string& label) {
  require(feature.x >= kBriefPatch && feature.x <= width - 1 - kBriefPatch &&
              feature.y >= kBriefPatch && feature.y <= height - 1 - kBriefPatch,
          label + ": BRIEF feature is outside descriptor patch envelope " +
              feature_string(feature));
  const auto& pattern = brief_pattern();
  std::array<std::int32_t, kBriefWords> descriptor{};
  for (int word_idx = 0; word_idx < kBriefWords; ++word_idx) {
    std::uint32_t word = 0U;
    for (int bit = 0; bit < 32; ++bit) {
      const int k = word_idx * 32 + bit;
      const int ax = feature.x + pattern[static_cast<std::size_t>(4 * k + 0)];
      const int ay = feature.y + pattern[static_cast<std::size_t>(4 * k + 1)];
      const int bx = feature.x + pattern[static_cast<std::size_t>(4 * k + 2)];
      const int by = feature.y + pattern[static_cast<std::size_t>(4 * k + 3)];
      const int va = image[static_cast<std::size_t>(ay) * static_cast<std::size_t>(width) +
                           static_cast<std::size_t>(ax)];
      const int vb = image[static_cast<std::size_t>(by) * static_cast<std::size_t>(width) +
                           static_cast<std::size_t>(bx)];
      if (va < vb) {
        word |= (std::uint32_t{1} << bit);
      }
    }
    std::memcpy(&descriptor[static_cast<std::size_t>(word_idx)], &word, sizeof(word));
  }
  return descriptor;
}

struct CpuKltReference {
  std::vector<float> points;
  std::vector<std::int32_t> status;
};

std::vector<std::uint8_t> cpu_klt_pyrdown(const std::vector<std::uint8_t>& src, int width,
                                          int height) {
  const int dst_w = (width + 1) / 2;
  const int dst_h = (height + 1) / 2;
  std::vector<std::uint8_t> dst(static_cast<std::size_t>(dst_w) * static_cast<std::size_t>(dst_h));
  const auto at = [&](int x, int y) -> int {
    x = std::clamp(x, 0, width - 1);
    y = std::clamp(y, 0, height - 1);
    return src[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
               static_cast<std::size_t>(x)];
  };
  for (int oy = 0; oy < dst_h; ++oy) {
    const int rc = std::min(2 * oy, height - 1);
    const int rm = std::max(0, rc - 1);
    const int rp = std::min(height - 1, rc + 1);
    for (int ox = 0; ox < dst_w; ++ox) {
      const int xc = std::min(2 * ox, width - 1);
      const int xm = std::max(0, xc - 1);
      const int xp = std::min(width - 1, xc + 1);
      const int h0 = at(xm, rm) + 2 * at(xc, rm) + at(xp, rm);
      const int h1 = at(xm, rc) + 2 * at(xc, rc) + at(xp, rc);
      const int h2 = at(xm, rp) + 2 * at(xc, rp) + at(xp, rp);
      dst[static_cast<std::size_t>(oy) * static_cast<std::size_t>(dst_w) +
          static_cast<std::size_t>(ox)] = static_cast<std::uint8_t>((h0 + 2 * h1 + h2 + 8) >> 4);
    }
  }
  return dst;
}

float cpu_klt_pixel(const std::vector<std::uint8_t>& image, int width, int x, int y) {
  return static_cast<float>(image[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                  static_cast<std::size_t>(x)]);
}

bool cpu_klt_lk_compute(const std::vector<std::uint8_t>& prev, const std::vector<std::uint8_t>& cur,
                        int width, int pyp, int pxp, int pyc, int pxc, int win_half, int max_iters,
                        float rx_in, float ry_in, float* rx_out, float* ry_out) {
  constexpr float kLkEps2 = 0.0009f;
  const int margin = 4;
  const int maxb = 2 * margin;
  const int wwin = 2 * win_half + 1;
  std::vector<float> ix(static_cast<std::size_t>(wwin) * static_cast<std::size_t>(wwin), 0.0f);
  std::vector<float> iy(ix.size(), 0.0f);
  std::vector<float> pv(ix.size(), 0.0f);

  float gxx = 0.0f;
  float gxy = 0.0f;
  float gyy = 0.0f;
  for (int r = 0; r < wwin; ++r) {
    const int y = pyp + 1 + r;
    for (int c = 0; c < wwin; ++c) {
      const int x = pxp + 1 + c;
      const float gx =
          (cpu_klt_pixel(prev, width, x + 1, y) - cpu_klt_pixel(prev, width, x - 1, y)) * 0.5f;
      const float gy =
          (cpu_klt_pixel(prev, width, x, y + 1) - cpu_klt_pixel(prev, width, x, y - 1)) * 0.5f;
      const std::size_t idx = static_cast<std::size_t>(r) * static_cast<std::size_t>(wwin) +
                              static_cast<std::size_t>(c);
      ix[idx] = gx;
      iy[idx] = gy;
      pv[idx] = cpu_klt_pixel(prev, width, x, y);
      gxx += gx * gx;
      gxy += gx * gy;
      gyy += gy * gy;
    }
  }
  const float det = gxx * gyy - gxy * gxy;
  if (det < 1e-6f) {
    *rx_out = rx_in;
    *ry_out = ry_in;
    return false;
  }
  const float inv = 1.0f / det;
  float vx = rx_in;
  float vy = ry_in;
  for (int it = 0; it < max_iters; ++it) {
    const float fvx = std::floor(vx);
    const float fvy = std::floor(vy);
    int bxb = margin + static_cast<int>(fvx);
    int byb = margin + static_cast<int>(fvy);
    bxb = std::clamp(bxb, 0, maxb);
    byb = std::clamp(byb, 0, maxb);
    const float fx = vx - fvx;
    const float fy = vy - fvy;
    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w01 = fx * (1.0f - fy);
    const float w10 = (1.0f - fx) * fy;
    const float w11 = fx * fy;

    float bx = 0.0f;
    float by = 0.0f;
    for (int r = 0; r < wwin; ++r) {
      const int y0 = pyc + byb + r;
      for (int c = 0; c < wwin; ++c) {
        const int x0 = pxc + bxb + c;
        const float a = cpu_klt_pixel(cur, width, x0, y0);
        const float b = cpu_klt_pixel(cur, width, x0 + 1, y0);
        const float cc = cpu_klt_pixel(cur, width, x0, y0 + 1);
        const float d = cpu_klt_pixel(cur, width, x0 + 1, y0 + 1);
        const std::size_t idx = static_cast<std::size_t>(r) * static_cast<std::size_t>(wwin) +
                                static_cast<std::size_t>(c);
        const float itv = pv[idx] - (a * w00 + b * w01 + cc * w10 + d * w11);
        bx += itv * ix[idx];
        by += itv * iy[idx];
      }
    }
    const float dvx = (gyy * bx - gxy * by) * inv;
    const float dvy = (gxx * by - gxy * bx) * inv;
    vx += dvx;
    vy += dvy;
    if (dvx * dvx + dvy * dvy < kLkEps2) {
      break;
    }
  }
  *rx_out = vx;
  *ry_out = vy;
  return true;
}

CpuKltReference cpu_klt_reference(const std::vector<std::uint8_t>& prev,
                                  const std::vector<std::uint8_t>& cur,
                                  const std::vector<std::int32_t>& points, int width, int height,
                                  int num_points, int win_half, int max_iters, int max_level,
                                  const std::string& label) {
  require(points.size() >= static_cast<std::size_t>(num_points) * 2U,
          label + ": CPU KLT points size mismatch");
  std::vector<std::vector<std::uint8_t>> prev_levels;
  std::vector<std::vector<std::uint8_t>> cur_levels;
  std::vector<int> widths;
  std::vector<int> heights;
  prev_levels.push_back(prev);
  cur_levels.push_back(cur);
  widths.push_back(width);
  heights.push_back(height);
  for (int level = 1; level <= max_level; ++level) {
    prev_levels.push_back(cpu_klt_pyrdown(prev_levels.back(), widths.back(), heights.back()));
    cur_levels.push_back(cpu_klt_pyrdown(cur_levels.back(), widths.back(), heights.back()));
    widths.push_back((widths.back() + 1) / 2);
    heights.push_back((heights.back() + 1) / 2);
  }

  CpuKltReference ref;
  ref.points.resize(static_cast<std::size_t>(num_points) * 2U, 0.0f);
  ref.status.resize(static_cast<std::size_t>(num_points), 0);
  const int margin = 4;
  const int wwin = 2 * win_half + 1;
  const int wpad = ((wwin + 15) / 16) * 16;
  const int pw = wwin + 2;
  const int cw = 2 * win_half + 2 * margin + 2;
  (void)wpad;
  for (int p = 0; p < num_points; ++p) {
    const int px = points[static_cast<std::size_t>(p) * 2U + 0U];
    const int py = points[static_cast<std::size_t>(p) * 2U + 1U];
    ref.points[static_cast<std::size_t>(p) * 2U + 0U] = static_cast<float>(px);
    ref.points[static_cast<std::size_t>(p) * 2U + 1U] = static_cast<float>(py);
    float vx = 0.0f;
    float vy = 0.0f;
    bool lost = false;
    for (int level = max_level; level >= 0; --level) {
      const int wl = widths[static_cast<std::size_t>(level)];
      const int hl = heights[static_cast<std::size_t>(level)];
      const int ix = px >> level;
      const int iy = py >> level;
      const int gxi = static_cast<int>(std::floor(vx));
      const int gyi = static_cast<int>(std::floor(vy));
      const int pxp = ix - win_half - 1;
      const int pyp = iy - win_half - 1;
      const int pxc = ix - win_half - margin + gxi;
      const int pyc = iy - win_half - margin + gyi;
      if (pxp < 0 || pyp < 0 || pxp + pw > wl || pyp + pw > hl || pxc < 0 || pyc < 0 ||
          pxc + cw > wl || pyc + cw > hl) {
        lost = true;
        break;
      }
      float rx = 0.0f;
      float ry = 0.0f;
      const bool ok = cpu_klt_lk_compute(
          prev_levels[static_cast<std::size_t>(level)], cur_levels[static_cast<std::size_t>(level)],
          wl, pyp, pxp, pyc, pxc, win_half, max_iters, vx - static_cast<float>(gxi),
          vy - static_cast<float>(gyi), &rx, &ry);
      if (!ok) {
        lost = true;
        break;
      }
      vx = static_cast<float>(gxi) + rx;
      vy = static_cast<float>(gyi) + ry;
      if (level > 0) {
        const int bx = (px >> (level - 1)) & 1;
        const int by = (py >> (level - 1)) & 1;
        vx = 2.0f * vx - static_cast<float>(bx);
        vy = 2.0f * vy - static_cast<float>(by);
      }
    }
    if (!lost) {
      const float ox = static_cast<float>(px) + vx;
      const float oy = static_cast<float>(py) + vy;
      ref.points[static_cast<std::size_t>(p) * 2U + 0U] = ox;
      ref.points[static_cast<std::size_t>(p) * 2U + 1U] = oy;
      ref.status[static_cast<std::size_t>(p)] =
          (ox >= 0.0f && oy >= 0.0f && ox <= static_cast<float>(width - 1) &&
           oy <= static_cast<float>(height - 1))
              ? 1
              : 0;
    }
  }
  return ref;
}

struct ExpectedOutput {
  std::string name;
  std::vector<int64_t> shape;
  simaai::neat::TensorDType dtype = simaai::neat::TensorDType::UInt8;
};

struct BenchCase {
  std::string name;
  std::string graph_name;
  int graph_id = 0;
  int width = 0;
  int height = 0;
  int batch_size = 1;
  int max_features = 0;
  int fast_threshold = 30;
  int descriptor_words = kBriefWords;
  int cpu_feature_input_index = 0;
  int cpu_feature_emit_radius = kFastRadius;
  bool cpu_fast_golden = false;
  bool cpu_descriptor_golden = false;
  bool cpu_klt_golden = false;
  std::vector<std::vector<CpuFeature>> cpu_feature_refs;
  std::vector<CpuKltReference> cpu_klt_refs;
  std::vector<std::string> input_names;
  simaai::neat::TensorList inputs;
  std::shared_ptr<simaai::neat::Node> node;
  std::vector<ExpectedOutput> expected_outputs;
  int klt_detect_new_features = 0;
  int klt_num_points = 0;
  int klt_win_half = 10;
  int klt_max_iters = 30;
  int klt_max_level = 3;
  bool expect_runtime_failure = false;
  std::string expected_error_substring;
};

struct TimingSummary {
  double avg_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double fps = 0.0;
};

TimingSummary summarize(const std::vector<double>& samples_ms) {
  TimingSummary out;
  if (samples_ms.empty()) {
    return out;
  }
  out.min_ms = *std::min_element(samples_ms.begin(), samples_ms.end());
  out.max_ms = *std::max_element(samples_ms.begin(), samples_ms.end());
  out.avg_ms = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) /
               static_cast<double>(samples_ms.size());
  out.fps = out.avg_ms > 0.0 ? 1000.0 / out.avg_ms : 0.0;
  return out;
}

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

struct ElementTimingLookup {
  double avg_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  std::uint64_t samples = 0;
  std::string element_name;
};

bool timing_name_matches(const std::string& candidate, const std::string& requested) {
  const std::string c = lower_copy(candidate);
  const std::string r = lower_copy(requested);
  return c == r || c.find(r) != std::string::npos || r.find(c) != std::string::npos ||
         c.find("processcvu") != std::string::npos || c.find("visual") != std::string::npos;
}

ElementTimingLookup timing_lookup_from_element(const simaai::neat::GraphElementMetrics& element) {
  ElementTimingLookup out;
  out.samples = element.latency.samples;
  out.element_name = element.name;
  out.avg_ms = element.latency.avg_ms;
  out.min_ms = element.latency.min_ms;
  out.max_ms = element.latency.max_ms;
  return out;
}

ElementTimingLookup lookup_element_timing_report(const simaai::neat::MeasureReport& report,
                                                 const std::string& element_name) {
  for (const auto& node : report.node_metrics) {
    for (const auto& element : node.elements) {
      if (element.name == element_name) {
        return timing_lookup_from_element(element);
      }
    }
  }
  for (const auto& node : report.node_metrics) {
    for (const auto& element : node.elements) {
      if (element.latency.samples > 0U && timing_name_matches(element.name, element_name)) {
        return timing_lookup_from_element(element);
      }
    }
  }
  for (const auto& node : report.node_metrics) {
    for (const auto& element : node.elements) {
      if (timing_name_matches(element.name, element_name)) {
        return timing_lookup_from_element(element);
      }
    }
  }
  return {};
}

void dump_timing_diag_if_requested(const simaai::neat::MeasureReport& report,
                                   const std::string& element_name,
                                   const ElementTimingLookup& lookup) {
  const bool dump = env_flag("SIMA_VISUAL_TPUT_DUMP_TIMINGS", false) || lookup.samples == 0U;
  if (!dump) {
    return;
  }
  std::size_t element_rows = 0;
  for (const auto& node : report.node_metrics) {
    element_rows += node.elements.size();
  }
  std::cout << " timing_diag requested=" << element_name
            << " node_rows=" << report.node_metrics.size() << " element_rows=" << element_rows
            << " source=measure_report\n";
  for (const auto& node : report.node_metrics) {
    for (const auto& element : node.elements) {
      std::cout << "  element_timing name=" << element.name << " node=" << node.label
                << " samples=" << element.latency.samples << " avg_ms=" << element.latency.avg_ms
                << " min_ms=" << element.latency.min_ms << " max_ms=" << element.latency.max_ms
                << "\n";
    }
  }
}

template <typename T>
std::vector<T> tensor_values(const simaai::neat::Tensor& tensor, const std::string& label) {
  const auto bytes = tensor.copy_dense_bytes_tight();
  require(bytes.size() % sizeof(T) == 0U,
          label + ": tensor byte size is not a multiple of value size");
  std::vector<T> out(bytes.size() / sizeof(T));
  if (!out.empty()) {
    std::memcpy(out.data(), bytes.data(), bytes.size());
  }
  return out;
}

int input_width(const BenchCase& c) {
  if (c.width > 0) {
    return c.width;
  }
  require(!c.inputs.front().shape.empty(), c.name + ": input shape must include width");
  return static_cast<int>(c.inputs.front().shape.back());
}

int input_height(const BenchCase& c) {
  if (c.height > 0) {
    return c.height;
  }
  const auto& shape = c.inputs.front().shape;
  require(shape.size() >= 2U, c.name + ": input shape must include height");
  if (shape.size() >= 3U) {
    return static_cast<int>(shape[shape.size() - 2U]);
  }
  const int batch = std::max(1, c.batch_size);
  return static_cast<int>(shape.front()) / batch;
}

int input_batch(const BenchCase& c) {
  return std::max(1, c.batch_size);
}

std::size_t image_bytes_per_batch(const BenchCase& c) {
  return static_cast<std::size_t>(input_width(c)) * static_cast<std::size_t>(input_height(c));
}

std::vector<std::uint8_t> input_image_batch(const BenchCase& c, int input_index, int batch_index,
                                            const std::string& label) {
  require(input_index >= 0 && static_cast<std::size_t>(input_index) < c.inputs.size(),
          label + ": input index out of range");
  require(batch_index >= 0 && batch_index < input_batch(c), label + ": batch index out of range");
  const auto bytes = c.inputs.at(static_cast<std::size_t>(input_index)).copy_dense_bytes_tight();
  const std::size_t image_bytes = image_bytes_per_batch(c);
  const std::size_t offset = static_cast<std::size_t>(batch_index) * image_bytes;
  require(offset <= bytes.size() && image_bytes <= bytes.size() - offset,
          label + ": packed input batch is smaller than width*height*batch");
  return std::vector<std::uint8_t>(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                                   bytes.begin() +
                                       static_cast<std::ptrdiff_t>(offset + image_bytes));
}

std::vector<std::int32_t> input_points_batch(const BenchCase& c, int input_index, int batch_index,
                                             int num_points, const std::string& label) {
  require(input_index >= 0 && static_cast<std::size_t>(input_index) < c.inputs.size(),
          label + ": input index out of range");
  require(batch_index >= 0 && batch_index < input_batch(c), label + ": batch index out of range");
  require(num_points > 0, label + ": num_points must be positive");
  const auto values = tensor_values<std::int32_t>(
      c.inputs.at(static_cast<std::size_t>(input_index)), label + ": input_points");
  const std::size_t stride = static_cast<std::size_t>(num_points) * 2U;
  const std::size_t offset = static_cast<std::size_t>(batch_index) * stride;
  require(offset <= values.size() && stride <= values.size() - offset,
          label + ": packed points batch is smaller than batch*num_points*2");
  return std::vector<std::int32_t>(values.begin() + static_cast<std::ptrdiff_t>(offset),
                                   values.begin() + static_cast<std::ptrdiff_t>(offset + stride));
}

std::size_t feature_list_stride(int max_features) {
  require(max_features > 0, "feature-list max_features must be positive");
  return 1U + static_cast<std::size_t>(max_features) * 3U;
}

int feature_batch_count(const BenchCase& c) {
  return input_batch(c);
}

int max_features_from_feature_tensor(const BenchCase& c, const simaai::neat::Tensor& tensor) {
  if (c.max_features > 0) {
    return c.max_features;
  }
  require(!tensor.shape.empty(), c.name + ": feature tensor shape is empty");
  const auto feature_width = tensor.shape.back();
  require(feature_width > 1 && ((feature_width - 1) % 3) == 0,
          c.name + ": feature tensor shape does not match [count][x,y,score]* ABI");
  return static_cast<int>((feature_width - 1) / 3);
}

std::vector<CpuFeature> read_feature_list_batch_values(const std::vector<std::int32_t>& values,
                                                       int batch_index, int max_features,
                                                       const std::string& label) {
  const std::size_t stride = feature_list_stride(max_features);
  const std::size_t offset = static_cast<std::size_t>(batch_index) * stride;
  require(offset <= values.size() && stride <= values.size() - offset,
          label + ": feature-list batch storage is smaller than max_features envelope");
  const int count = values[offset];
  require(count >= 0 && count <= max_features,
          label + ": feature count out of range batch=" + std::to_string(batch_index) +
              " count=" + std::to_string(count) + " max=" + std::to_string(max_features));
  std::vector<CpuFeature> out;
  out.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    const std::size_t base = offset + 1U + static_cast<std::size_t>(i) * 3U;
    out.push_back({values[base + 0U], values[base + 1U], values[base + 2U]});
  }
  return out;
}

void validate_feature_list_payload(const BenchCase& c, const simaai::neat::Tensor& tensor,
                                   int max_features, const std::string& label) {
  const auto values = tensor_values<std::int32_t>(tensor, label);
  require(!values.empty(), label + ": empty feature list");
  const int batches = feature_batch_count(c);
  require(values.size() >= feature_list_stride(max_features) * static_cast<std::size_t>(batches),
          label + ": feature-list storage is smaller than batched max_features envelope");
  const int width = input_width(c);
  const int height = input_height(c);
  for (int b = 0; b < batches; ++b) {
    const auto features = read_feature_list_batch_values(values, b, max_features, label);
    for (const auto& feature : features) {
      require(feature.x >= 0 && feature.x < width && feature.y >= 0 && feature.y < height,
              label + ": feature coordinate out of image bounds batch=" + std::to_string(b) +
                  " feature=" + feature_string(feature));
      require(feature.score >= 0,
              label + ": feature score must be non-negative batch=" + std::to_string(b));
    }
  }
}

void prepare_cpu_fast_golden(BenchCase* c, int input_index, int emit_radius,
                             bool validate_descriptors) {
  require(c != nullptr, "prepare_cpu_fast_golden requires a case");
  c->cpu_fast_golden = env_flag("SIMA_VISUAL_TPUT_CPU_GOLDEN", true);
  c->cpu_descriptor_golden = validate_descriptors && c->cpu_fast_golden &&
                             env_flag("SIMA_VISUAL_TPUT_CPU_DESCRIPTOR_GOLDEN", true);
  c->cpu_feature_input_index = input_index;
  c->cpu_feature_emit_radius = emit_radius;
  c->cpu_feature_refs.clear();
  if (!c->cpu_fast_golden) {
    return;
  }

  const int batches = feature_batch_count(*c);
  c->cpu_feature_refs.reserve(static_cast<std::size_t>(batches));
  for (int b = 0; b < batches; ++b) {
    const auto image =
        input_image_batch(*c, input_index, b, c->name + ": CPU FAST reference input");
    c->cpu_feature_refs.push_back(cpu_fast_reference(image, input_width(*c), input_height(*c),
                                                     c->fast_threshold, emit_radius));
  }
}

void validate_cpu_fast_golden(const BenchCase& c, const simaai::neat::Tensor& tensor,
                              int max_features, const std::string& label) {
  if (!c.cpu_fast_golden) {
    return;
  }
  const int batches = feature_batch_count(c);
  require(c.cpu_feature_refs.size() >= static_cast<std::size_t>(batches),
          label + ": CPU feature references were not prepared");
  const auto values = tensor_values<std::int32_t>(tensor, label + ": CPU golden features");
  for (int b = 0; b < batches; ++b) {
    const auto ev_features = read_feature_list_batch_values(values, b, max_features, label);
    const auto& cpu_features = c.cpu_feature_refs[static_cast<std::size_t>(b)];
    if (cpu_features.size() <= static_cast<std::size_t>(max_features)) {
      require(ev_features.size() == cpu_features.size(),
              label + ": CPU FAST count mismatch batch=" + std::to_string(b) +
                  " expected=" + std::to_string(cpu_features.size()) +
                  " actual=" + std::to_string(ev_features.size()));
    } else {
      require(ev_features.size() <= static_cast<std::size_t>(max_features),
              label + ": EV feature count exceeds max_features in saturated CPU case");
      require(!ev_features.empty(),
              label + ": CPU FAST found features but EV returned none in saturated case batch=" +
                  std::to_string(b));
    }

    std::vector<bool> matched(cpu_features.size(), false);
    for (const auto& ev_feature : ev_features) {
      bool found = false;
      for (std::size_t i = 0; i < cpu_features.size(); ++i) {
        if (!matched[i] && cpu_features[i] == ev_feature) {
          matched[i] = true;
          found = true;
          break;
        }
      }
      require(found, label + ": EV feature not present in CPU FAST reference batch=" +
                         std::to_string(b) + " feature=" + feature_string(ev_feature));
    }
    if (cpu_features.size() <= static_cast<std::size_t>(max_features)) {
      for (std::size_t i = 0; i < matched.size(); ++i) {
        require(matched[i], label + ": CPU FAST feature missing from EV output batch=" +
                                std::to_string(b) + " feature=" + feature_string(cpu_features[i]));
      }
    }
  }
}

void validate_cpu_descriptor_golden(const BenchCase& c, const simaai::neat::Tensor& features_tensor,
                                    const simaai::neat::Tensor& descriptors_tensor,
                                    int max_features, const std::string& label) {
  if (!c.cpu_descriptor_golden) {
    return;
  }
  require(c.descriptor_words == kBriefWords,
          label + ": CPU descriptor golden only supports the current BRIEF-256 ABI");
  const int batches = feature_batch_count(c);
  const auto feature_values =
      tensor_values<std::int32_t>(features_tensor, label + ": descriptor features");
  const auto descriptors = tensor_values<std::int32_t>(descriptors_tensor, label + ": descriptors");
  const std::size_t descriptor_stride = static_cast<std::size_t>(max_features) * kBriefWords;
  require(descriptors.size() >= descriptor_stride * static_cast<std::size_t>(batches),
          label + ": descriptor storage is smaller than batched max_features envelope");
  for (int b = 0; b < batches; ++b) {
    const auto image =
        input_image_batch(c, c.cpu_feature_input_index, b, label + ": descriptor CPU input");
    const auto ev_features = read_feature_list_batch_values(feature_values, b, max_features, label);
    for (std::size_t i = 0; i < ev_features.size(); ++i) {
      const auto expected =
          cpu_brief_descriptor(image, input_width(c), input_height(c), ev_features[i], label);
      const std::size_t offset = static_cast<std::size_t>(b) * descriptor_stride + i * kBriefWords;
      for (int word = 0; word < kBriefWords; ++word) {
        const auto actual = descriptors[offset + static_cast<std::size_t>(word)];
        require(actual == expected[static_cast<std::size_t>(word)],
                label + ": descriptor mismatch batch=" + std::to_string(b) +
                    " feature_index=" + std::to_string(i) +
                    " feature=" + feature_string(ev_features[i]) + " word=" + std::to_string(word) +
                    " expected=" + std::to_string(expected[static_cast<std::size_t>(word)]) +
                    " actual=" + std::to_string(actual));
      }
    }
  }
}

void prepare_cpu_klt_golden(BenchCase* c) {
  require(c != nullptr, "prepare_cpu_klt_golden requires a case");
  c->cpu_klt_golden = env_flag("SIMA_VISUAL_TPUT_CPU_GOLDEN", true) &&
                      env_flag("SIMA_VISUAL_TPUT_CPU_KLT_GOLDEN", true);
  c->cpu_klt_refs.clear();
  if (!c->cpu_klt_golden) {
    return;
  }
  const int batches = input_batch(*c);
  c->cpu_klt_refs.reserve(static_cast<std::size_t>(batches));
  for (int b = 0; b < batches; ++b) {
    const auto prev = input_image_batch(*c, 0, b, c->name + ": CPU KLT prev input");
    const auto cur = input_image_batch(*c, 1, b, c->name + ": CPU KLT cur input");
    const auto points =
        input_points_batch(*c, 2, b, c->klt_num_points, c->name + ": CPU KLT points input");
    c->cpu_klt_refs.push_back(cpu_klt_reference(
        prev, cur, points, input_width(*c), input_height(*c), c->klt_num_points, c->klt_win_half,
        c->klt_max_iters, c->klt_max_level, c->name + ": CPU KLT reference"));
  }
}

void validate_cpu_klt_golden(const BenchCase& c, const simaai::neat::Tensor& points_tensor,
                             const simaai::neat::Tensor& status_tensor, const std::string& label) {
  if (!c.cpu_klt_golden) {
    return;
  }
  const int batches = input_batch(c);
  require(c.cpu_klt_refs.size() >= static_cast<std::size_t>(batches),
          label + ": CPU KLT references were not prepared");
  const auto points = tensor_values<float>(points_tensor, label + ": KLT points");
  const auto status = tensor_values<std::int32_t>(status_tensor, label + ": KLT status");
  const std::size_t point_stride = static_cast<std::size_t>(c.klt_num_points) * 2U;
  const std::size_t status_stride = static_cast<std::size_t>(c.klt_num_points);
  require(points.size() >= point_stride * static_cast<std::size_t>(batches),
          label + ": KLT output point storage is smaller than batch*num_points*2");
  require(status.size() >= status_stride * static_cast<std::size_t>(batches),
          label + ": KLT output status storage is smaller than batch*num_points");
  const float tolerance =
      static_cast<float>(std::max(0, env_int_local("SIMA_VISUAL_TPUT_KLT_TOLERANCE_MILLI", 150))) /
      1000.0f;
  int tracked_total = 0;
  for (int b = 0; b < batches; ++b) {
    const auto& ref = c.cpu_klt_refs[static_cast<std::size_t>(b)];
    require(ref.points.size() == point_stride && ref.status.size() == status_stride,
            label + ": CPU KLT reference size mismatch");
    const std::size_t point_offset = static_cast<std::size_t>(b) * point_stride;
    const std::size_t status_offset = static_cast<std::size_t>(b) * status_stride;
    for (int i = 0; i < c.klt_num_points; ++i) {
      const auto actual_status = status[status_offset + static_cast<std::size_t>(i)];
      const auto expected_status = ref.status[static_cast<std::size_t>(i)];
      require(actual_status == expected_status,
              label + ": KLT status mismatch batch=" + std::to_string(b) +
                  " point=" + std::to_string(i) + " expected=" + std::to_string(expected_status) +
                  " actual=" + std::to_string(actual_status));
      tracked_total += actual_status != 0 ? 1 : 0;
      for (int axis = 0; axis < 2; ++axis) {
        const std::size_t idx =
            point_offset + static_cast<std::size_t>(i) * 2U + static_cast<std::size_t>(axis);
        const float expected =
            ref.points[static_cast<std::size_t>(i) * 2U + static_cast<std::size_t>(axis)];
        const float actual = points[idx];
        require(std::fabs(actual - expected) <= tolerance,
                label + ": KLT point mismatch batch=" + std::to_string(b) +
                    " point=" + std::to_string(i) + " axis=" + std::to_string(axis) +
                    " expected=" + std::to_string(expected) + " actual=" + std::to_string(actual) +
                    " tol=" + std::to_string(tolerance));
      }
    }
  }
  if (env_flag("SIMA_VISUAL_TPUT_KLT_REQUIRE_TRACKED", true)) {
    require(tracked_total > 0, label + ": CPU KLT golden did not track any points");
  }
}

void validate_semantic_payload(const BenchCase& c, const simaai::neat::TensorList& outputs) {
  if (c.graph_name == "feature_histogram") {
    const int batch = input_batch(c);
    const auto values = tensor_values<std::int32_t>(outputs.at(0), c.name + ": histogram");
    require(values.size() == 256U * static_cast<std::size_t>(batch),
            c.name + ": histogram must contain exactly 256 bins per batch");
    const auto input_bytes = c.inputs.front().copy_dense_bytes_tight();
    const std::size_t image_bytes = image_bytes_per_batch(c);
    require(input_bytes.size() >= image_bytes * static_cast<std::size_t>(batch),
            c.name + ": packed histogram input is smaller than width*height*batch");
    for (int b = 0; b < batch; ++b) {
      std::array<std::int32_t, 256> expected_bins{};
      const std::size_t input_offset = static_cast<std::size_t>(b) * image_bytes;
      for (std::size_t i = 0; i < image_bytes; ++i) {
        ++expected_bins[input_bytes[input_offset + i]];
      }
      std::int64_t sum = 0;
      const std::size_t output_offset = static_cast<std::size_t>(b) * 256U;
      for (std::size_t i = 0; i < 256U; ++i) {
        const auto value = values[output_offset + i];
        require(value >= 0, c.name + ": histogram bin must be non-negative");
        require(value == expected_bins[i],
                c.name + ": histogram bin mismatch batch=" + std::to_string(b) +
                    " bin=" + std::to_string(i) + " expected=" + std::to_string(expected_bins[i]) +
                    " actual=" + std::to_string(value));
        sum += static_cast<std::int64_t>(value);
      }
      const std::int64_t expected_pixels =
          static_cast<std::int64_t>(input_width(c)) * static_cast<std::int64_t>(input_height(c));
      require(sum == expected_pixels,
              c.name + ": histogram sum mismatch batch=" + std::to_string(b) + " expected=" +
                  std::to_string(expected_pixels) + " actual=" + std::to_string(sum));
    }
    return;
  }

  if (c.graph_name == "grider_fast") {
    const int max_features = max_features_from_feature_tensor(c, outputs.at(0));
    validate_feature_list_payload(c, outputs.at(0), max_features, c.name + ": features");
    validate_cpu_fast_golden(c, outputs.at(0), max_features, c.name + ": features");
    return;
  }

  if (c.graph_name == "track_descriptor") {
    const int max_features = max_features_from_feature_tensor(c, outputs.at(0));
    validate_feature_list_payload(c, outputs.at(0), max_features, c.name + ": features");
    validate_cpu_fast_golden(c, outputs.at(0), max_features, c.name + ": features");
    const auto descriptors = tensor_values<std::int32_t>(outputs.at(1), c.name + ": descriptors");
    require(descriptors.size() == static_cast<std::size_t>(input_batch(c)) *
                                      static_cast<std::size_t>(max_features) * kBriefWords,
            c.name + ": descriptor storage size mismatch");
    validate_cpu_descriptor_golden(c, outputs.at(0), outputs.at(1), max_features,
                                   c.name + ": descriptors");
    return;
  }

  if (c.graph_name == "track_klt") {
    const auto points = tensor_values<float>(outputs.at(0), c.name + ": output_points");
    for (const auto value : points) {
      require(std::isfinite(value), c.name + ": output point must be finite");
    }
    const auto status = tensor_values<std::int32_t>(outputs.at(1), c.name + ": output_status");
    for (const auto value : status) {
      require(value == 0 || value == 1, c.name + ": status must be 0 or 1");
    }
    validate_cpu_klt_golden(c, outputs.at(0), outputs.at(1), c.name + ": KLT CPU golden");
    if (c.klt_detect_new_features != 0) {
      require(outputs.size() == 3U,
              c.name + ": detect_new_features=1 should publish output_features");
      const int max_features = max_features_from_feature_tensor(c, outputs.at(2));
      validate_feature_list_payload(c, outputs.at(2), max_features, c.name + ": detected_features");
      validate_cpu_fast_golden(c, outputs.at(2), max_features, c.name + ": detected_features");
    }
  }
}

template <typename T>
void append_values(std::vector<std::uint8_t>* dst, const std::vector<T>& values, std::size_t offset,
                   std::size_t count, const std::string& label) {
  require(dst != nullptr, label + ": null destination");
  require(offset <= values.size() && count <= values.size() - offset,
          label + ": signature slice out of bounds");
  if (count == 0U) {
    return;
  }
  const auto* ptr = reinterpret_cast<const std::uint8_t*>(values.data() + offset);
  dst->insert(dst->end(), ptr, ptr + count * sizeof(T));
}

template <typename T>
void append_scalar(std::vector<std::uint8_t>* dst, T value, const std::string& label) {
  require(dst != nullptr, label + ": null destination");
  const auto* ptr = reinterpret_cast<const std::uint8_t*>(&value);
  dst->insert(dst->end(), ptr, ptr + sizeof(T));
}

std::vector<int> append_feature_list_signature(const BenchCase& c,
                                               const simaai::neat::Tensor& tensor, int max_features,
                                               const std::string& label,
                                               std::vector<std::uint8_t>* signature) {
  const auto values = tensor_values<std::int32_t>(tensor, label);
  require(!values.empty(), label + ": empty feature list");
  const int batches = feature_batch_count(c);
  std::vector<int> counts;
  counts.reserve(static_cast<std::size_t>(batches));
  const std::size_t stride = feature_list_stride(max_features);
  for (int b = 0; b < batches; ++b) {
    const auto features = read_feature_list_batch_values(values, b, max_features, label);
    const int count = static_cast<int>(features.size());
    counts.push_back(count);
    const std::size_t offset = static_cast<std::size_t>(b) * stride;
    append_scalar(signature, count, label + ": count batch=" + std::to_string(b));
    append_values(signature, values, offset + 1U, static_cast<std::size_t>(count) * 3U,
                  label + ": active features batch=" + std::to_string(b));
  }
  return counts;
}

std::vector<std::uint8_t> semantic_signature(const BenchCase& c,
                                             const simaai::neat::TensorList& outputs) {
  std::vector<std::uint8_t> sig;
  append_scalar(&sig, static_cast<std::uint32_t>(outputs.size()), c.name + ": output count");

  if (c.graph_name == "feature_histogram") {
    const auto values = tensor_values<std::int32_t>(outputs.at(0), c.name + ": histogram");
    append_values(&sig, values, 0U, values.size(), c.name + ": histogram signature");
    return sig;
  }

  if (c.graph_name == "grider_fast") {
    const int max_features = max_features_from_feature_tensor(c, outputs.at(0));
    (void)append_feature_list_signature(c, outputs.at(0), max_features,
                                        c.name + ": features signature", &sig);
    return sig;
  }

  if (c.graph_name == "track_descriptor") {
    const int max_features = max_features_from_feature_tensor(c, outputs.at(0));
    const auto counts = append_feature_list_signature(c, outputs.at(0), max_features,
                                                      c.name + ": features signature", &sig);
    const auto descriptors =
        tensor_values<std::int32_t>(outputs.at(1), c.name + ": descriptors signature");
    const std::size_t descriptor_stride = static_cast<std::size_t>(max_features) * kBriefWords;
    require(descriptors.size() >= descriptor_stride * counts.size(),
            c.name + ": descriptor payload out of bounds while building signature");
    for (std::size_t b = 0; b < counts.size(); ++b) {
      const std::size_t active_descriptor_words = static_cast<std::size_t>(counts[b]) * kBriefWords;
      append_values(&sig, descriptors, b * descriptor_stride, active_descriptor_words,
                    c.name + ": active descriptors signature batch=" + std::to_string(b));
    }
    return sig;
  }

  if (c.graph_name == "track_klt") {
    const auto points = tensor_values<float>(outputs.at(0), c.name + ": output_points signature");
    append_values(&sig, points, 0U, points.size(), c.name + ": output points signature");
    const auto status =
        tensor_values<std::int32_t>(outputs.at(1), c.name + ": output_status signature");
    append_values(&sig, status, 0U, status.size(), c.name + ": output status signature");
    if (c.klt_detect_new_features != 0 && outputs.size() >= 3U) {
      const int max_features = max_features_from_feature_tensor(c, outputs.at(2));
      (void)append_feature_list_signature(c, outputs.at(2), max_features,
                                          c.name + ": detected features signature", &sig);
    }
    return sig;
  }

  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto bytes = outputs.at(i).copy_dense_bytes_tight();
    sig.insert(sig.end(), bytes.begin(), bytes.end());
  }
  return sig;
}

void validate_outputs(const BenchCase& c, const simaai::neat::TensorList& outputs) {
  require(outputs.size() == c.expected_outputs.size(),
          c.name + ": expected " + std::to_string(c.expected_outputs.size()) +
              " output tensors, got " + std::to_string(outputs.size()));
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& tensor = outputs[i];
    const auto& expected = c.expected_outputs[i];
    require(tensor.shape == expected.shape,
            c.name + ": output[" + std::to_string(i) + "] shape mismatch expected=" +
                shape_string(expected.shape) + " actual=" + shape_string(tensor.shape));
    require(tensor.dtype == expected.dtype,
            c.name + ": output[" + std::to_string(i) + "] dtype mismatch expected=" +
                dtype_string(expected.dtype) + " actual=" + dtype_string(tensor.dtype));
    require(tensor.dense_bytes_tight() > 0U,
            c.name + ": output[" + std::to_string(i) + "] has zero payload bytes");
  }
  validate_semantic_payload(c, outputs);
}

BenchCase make_feature_histogram_case(int width, int height, int batch_size, int num_buffers) {
  simaai::neat::FeatureHistogramOptions opt;
  opt.width = width;
  opt.height = height;
  opt.batch_size = batch_size;
  opt.num_buffers = num_buffers;
  opt.element_name = "feature_histogram";
  opt.input_name = "input_image";
  opt.output_name = "output_hist";

  BenchCase c;
  c.name = batch_size > 1 ? "FeatureHistogramBatch" : "FeatureHistogram";
  c.graph_name = "feature_histogram";
  c.graph_id = 235;
  c.width = width;
  c.height = height;
  c.batch_size = batch_size;
  c.input_names = {opt.input_name};
  c.inputs = {make_u8_batch_tensor(width, height, batch_size, opt.input_name, 0, 0, 13)};
  c.node = simaai::neat::nodes::FeatureHistogram(opt);
  c.expected_outputs = {{opt.output_name, {batch_size, 256}, simaai::neat::TensorDType::Int32}};
  return c;
}

BenchCase make_feature_histogram_batch_guard_case(int width, int height, int num_buffers) {
  simaai::neat::FeatureHistogramOptions opt;
  opt.width = width;
  opt.height = height;
  opt.batch_size = 2;
  opt.num_buffers = num_buffers;
  opt.element_name = "feature_histogram_batch_guard";
  opt.input_name = "input_image";
  opt.output_name = "output_hist";

  BenchCase c;
  c.name = "FeatureHistogramBatchGuard";
  c.graph_name = "feature_histogram";
  c.graph_id = 235;
  c.width = width;
  c.height = height;
  c.batch_size = opt.batch_size;
  c.input_names = {opt.input_name};
  // Intentionally provide a single-image tensor while the native config requests batch_size=2.
  // The processcvu/native-visual predispatch guard must reject this as undersized before EV74.
  c.inputs = {make_u8_tensor(width, height, opt.input_name, 0, 0, 17)};
  c.node = simaai::neat::nodes::FeatureHistogram(opt);
  c.expected_outputs = {{opt.output_name, {opt.batch_size, 256}, simaai::neat::TensorDType::Int32}};
  c.expect_runtime_failure = true;
  c.expected_error_substring = "not-negotiated";
  return c;
}

BenchCase make_grider_fast_case(int width, int height, int batch_size, int max_features,
                                int num_buffers) {
  simaai::neat::GriderFastOptions opt;
  opt.width = width;
  opt.height = height;
  opt.batch_size = batch_size;
  opt.max_features = max_features;
  opt.grid_x = 8;
  opt.grid_y = 6;
  opt.threshold = 30;
  opt.min_px_dist = 10;
  opt.num_buffers = num_buffers;
  opt.element_name = "grider_fast";
  opt.input_name = "input_image";
  opt.output_name = "output_features";

  BenchCase c;
  c.name = batch_size > 1 ? "GriderFastBatch" : "GriderFast";
  c.graph_name = "grider_fast";
  c.graph_id = 236;
  c.width = width;
  c.height = height;
  c.batch_size = batch_size;
  c.max_features = max_features;
  c.fast_threshold = opt.threshold;
  c.input_names = {opt.input_name};
  c.inputs = {make_u8_batch_tensor(width, height, batch_size, opt.input_name, 0, 0, 23)};
  c.node = simaai::neat::nodes::GriderFast(opt);
  c.expected_outputs = {
      {opt.output_name, {batch_size, 1 + max_features * 3}, simaai::neat::TensorDType::Int32}};
  prepare_cpu_fast_golden(&c, /*input_index=*/0, kFastRadius, /*validate_descriptors=*/false);
  return c;
}

BenchCase make_track_descriptor_case(int width, int height, int batch_size, int max_features,
                                     int num_buffers) {
  simaai::neat::TrackDescriptorOptions opt;
  opt.width = width;
  opt.height = height;
  opt.batch_size = batch_size;
  opt.max_features = max_features;
  opt.grid_x = 8;
  opt.grid_y = 6;
  opt.threshold = 30;
  opt.min_px_dist = 10;
  opt.descriptor_words = kBriefWords;
  opt.num_buffers = num_buffers;
  opt.element_name = "track_descriptor";
  opt.input_name = "input_image";
  opt.features_output_name = "output_features";
  opt.descriptors_output_name = "output_descriptors";

  BenchCase c;
  c.name = batch_size > 1 ? "TrackDescriptorBatch" : "TrackDescriptor";
  c.graph_name = "track_descriptor";
  c.graph_id = 237;
  c.width = width;
  c.height = height;
  c.batch_size = batch_size;
  c.max_features = max_features;
  c.fast_threshold = opt.threshold;
  c.descriptor_words = opt.descriptor_words;
  c.input_names = {opt.input_name};
  c.inputs = {make_u8_batch_tensor(width, height, batch_size, opt.input_name, 0, 0, 37)};
  c.node = simaai::neat::nodes::TrackDescriptor(opt);
  c.expected_outputs = {{opt.features_output_name,
                         {batch_size, 1 + max_features * 3},
                         simaai::neat::TensorDType::Int32},
                        {opt.descriptors_output_name,
                         {batch_size, max_features, kBriefWords},
                         simaai::neat::TensorDType::Int32}};
  prepare_cpu_fast_golden(&c, /*input_index=*/0, kBriefPatch, /*validate_descriptors=*/true);
  return c;
}

BenchCase make_track_klt_case(int width, int height, int batch_size, int num_points,
                              int max_features, int num_buffers) {
  const int detect_new_features = env_flag("SIMA_VISUAL_TPUT_KLT_DETECT", false) ? 1 : 0;
  const int klt_dx = env_int_local("SIMA_VISUAL_TPUT_KLT_DX", 1);
  const int klt_dy = env_int_local("SIMA_VISUAL_TPUT_KLT_DY", 1);
  simaai::neat::TrackKLTOptions opt;
  opt.width = width;
  opt.height = height;
  opt.batch_size = batch_size;
  opt.num_points = num_points;
  opt.max_features = max_features;
  opt.win_half = env_int_local("SIMA_VISUAL_TPUT_KLT_WIN_HALF", 8);
  opt.max_iters = env_int_local("SIMA_VISUAL_TPUT_KLT_MAX_ITERS", 20);
  opt.max_level = env_int_local("SIMA_VISUAL_TPUT_KLT_MAX_LEVEL", 2);
  opt.detect_new_features = detect_new_features;
  opt.grid_x = 8;
  opt.grid_y = 6;
  opt.fast_threshold = 30;
  opt.min_px_dist = 10;
  opt.num_buffers = num_buffers;
  opt.element_name = "track_klt";
  opt.prev_image_name = "prev_image";
  opt.cur_image_name = "cur_image";
  opt.input_points_name = "input_points";
  opt.output_points_name = "output_points";
  opt.output_status_name = "output_status";
  opt.output_features_name = "output_features";

  BenchCase c;
  c.name = batch_size > 1 ? "TrackKLTBatch" : "TrackKLT";
  c.graph_name = "track_klt";
  c.graph_id = 238;
  c.width = width;
  c.height = height;
  c.batch_size = batch_size;
  c.max_features = max_features;
  c.fast_threshold = opt.fast_threshold;
  c.klt_num_points = num_points;
  c.klt_win_half = opt.win_half;
  c.klt_max_iters = opt.max_iters;
  c.klt_max_level = opt.max_level;
  c.input_names = {opt.prev_image_name, opt.cur_image_name, opt.input_points_name};
  auto image_pair = make_klt_image_pair_tensors(width, height, batch_size, opt.prev_image_name,
                                                opt.cur_image_name, klt_dx, klt_dy);
  c.inputs = {std::move(image_pair.first), std::move(image_pair.second),
              make_i32_points_batch_tensor(width, height, batch_size, num_points, opt.win_half,
                                           opt.max_level, klt_dx, klt_dy, opt.input_points_name, 2,
                                           2)};
  c.node = simaai::neat::nodes::TrackKLT(opt);
  c.expected_outputs = {
      {opt.output_points_name, {batch_size, num_points, 2}, simaai::neat::TensorDType::Float32},
      {opt.output_status_name, {batch_size, num_points, 1}, simaai::neat::TensorDType::Int32}};
  if (detect_new_features != 0) {
    c.name = batch_size > 1 ? "TrackKLTDetectBatch" : "TrackKLTDetect";
    c.expected_outputs.push_back({opt.output_features_name,
                                  {batch_size, 1 + max_features * 3},
                                  simaai::neat::TensorDType::Int32});
    prepare_cpu_fast_golden(&c, /*input_index=*/1, kFastRadius, /*validate_descriptors=*/false);
  }
  c.klt_detect_new_features = detect_new_features;
  prepare_cpu_klt_golden(&c);
  return c;
}

std::vector<BenchCase> make_cases() {
  const int width = env_int_local("SIMA_VISUAL_TPUT_WIDTH", 640);
  const int height = env_int_local("SIMA_VISUAL_TPUT_HEIGHT", 480);
  const int batch_size = std::max(1, env_int_local("SIMA_VISUAL_TPUT_BATCH", 1));
  const int max_features = env_int_local("SIMA_VISUAL_TPUT_MAX_FEATURES", 256);
  const int num_points = env_int_local("SIMA_VISUAL_TPUT_NUM_POINTS", 128);
  const int num_buffers = env_int_local("SIMA_VISUAL_TPUT_NUM_BUFFERS", 4);

  std::vector<BenchCase> cases;
  cases.push_back(make_feature_histogram_case(width, height, batch_size, num_buffers));
  cases.push_back(make_grider_fast_case(width, height, batch_size, max_features, num_buffers));
  cases.push_back(make_track_descriptor_case(width, height, batch_size, max_features, num_buffers));
  cases.push_back(
      make_track_klt_case(width, height, batch_size, num_points, max_features, num_buffers));

  const std::string only = env_string("SIMA_VISUAL_TPUT_CASE");
  if (only == "feature_histogram_batch_guard" || only == "FeatureHistogramBatchGuard") {
    return {make_feature_histogram_batch_guard_case(width, height, num_buffers)};
  }
  if (only.empty() || only == "all") {
    return cases;
  }

  std::vector<BenchCase> filtered;
  for (auto& c : cases) {
    if (c.graph_name == only || c.name == only) {
      filtered.push_back(std::move(c));
    }
  }
  if (filtered.empty()) {
    throw std::runtime_error("unknown SIMA_VISUAL_TPUT_CASE: " + only);
  }
  return filtered;
}

void print_tensor_summary(const simaai::neat::TensorList& outputs) {
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& t = outputs[i];
    std::cout << " output[" << i
              << "] name=" << (t.route.segment_name.empty() ? "<empty>" : t.route.segment_name)
              << " shape=" << shape_string(t.shape) << " dtype=" << dtype_string(t.dtype)
              << " bytes=" << t.dense_bytes_tight() << "\n";
  }
}

bool run_case(const BenchCase& c, int warmup, int iterations, int timeout_ms, int queue_depth) {
  std::cout << "\n[case] " << c.name << " graph=" << c.graph_name << " id=" << c.graph_id
            << " inputs=" << c.inputs.size() << " warmup=" << warmup << " iterations=" << iterations
            << "\n";

  simaai::neat::Graph graph;

  simaai::neat::InputOptions src_opt;
  src_opt.payload_type = simaai::neat::PayloadType::Tensor;
  src_opt.format = simaai::neat::FormatTag::UINT8;
  src_opt.width = input_width(c);
  src_opt.height = input_height(c);
  src_opt.depth = 1;
  src_opt.max_width = input_width(c);
  src_opt.max_height = input_height(c) * input_batch(c);
  src_opt.max_depth = src_opt.depth;
  src_opt.is_live = true;
  src_opt.do_timestamp = true;
  src_opt.block = true;
  src_opt.pool_min_buffers = std::max(2, env_int_local("SIMA_VISUAL_TPUT_NUM_BUFFERS", 4));
  src_opt.pool_max_buffers = src_opt.pool_min_buffers;
  src_opt.memory_policy = simaai::neat::InputMemoryPolicy::Ev74;
  src_opt.buffer_name = c.input_names.empty() ? "input_tensor" : c.input_names.front();
  src_opt.max_bytes = 0;
  if (c.inputs.size() > 1U) {
    src_opt.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                            "storage=(string)tensorbuffer";
  }

  graph.add(simaai::neat::nodes::Input(src_opt));
  graph.add(c.node);

  simaai::neat::OutputOptions sink_opt;
  sink_opt.sync = false;
  sink_opt.drop = false;
  sink_opt.max_buffers = std::max(2, queue_depth);
  graph.add(simaai::neat::nodes::Output(sink_opt));

  simaai::neat::RunOptions run_opt;
  run_opt.output_memory = simaai::neat::OutputMemory::Owned;
  run_opt.queue_depth = queue_depth;
  run_opt.input_timeout_ms = timeout_ms;
  run_opt.startup_preflight = !c.expect_runtime_failure;

  auto build_start = Clock::now();
  simaai::neat::Run run = graph.build(c.inputs, run_opt);
  auto build_end = Clock::now();
  const double build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_start).count();
  std::cout << " build_ms=" << std::fixed << std::setprecision(3) << build_ms << "\n";

  if (c.expect_runtime_failure) {
    try {
      (void)run.run(c.inputs, timeout_ms);
      run.close();
      throw std::runtime_error(c.name + ": expected runtime failure containing '" +
                               c.expected_error_substring + "' but run succeeded");
    } catch (const std::exception& e) {
      run.close();
      const std::string msg = e.what();
      require(!c.expected_error_substring.empty() &&
                  msg.find(c.expected_error_substring) != std::string::npos,
              c.name + ": expected failure containing '" + c.expected_error_substring +
                  "' got: " + msg);
      std::cout << " result=PASS expected_runtime_failure=checked error=\"" << msg << "\"\n";
      return true;
    }
  }

  simaai::neat::TensorList last_outputs;
  for (int i = 0; i < warmup; ++i) {
    last_outputs = run.run(c.inputs, timeout_ms);
    validate_outputs(c, last_outputs);
  }

  const bool require_deterministic = env_flag("SIMA_VISUAL_TPUT_REQUIRE_DETERMINISTIC", true);
  bool have_signature = false;
  std::vector<std::uint8_t> reference_signature;

  simaai::neat::MeasureOptions measure_opt;
  measure_opt.title = c.name + " visual frontend throughput";
  measure_opt.include_plugin_latency = false;
  measure_opt.include_edge_latency = false;
  measure_opt.include_power = false;
  auto measure_scope = run.start_measurement(measure_opt);

  std::vector<double> samples_ms;
  samples_ms.reserve(static_cast<std::size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    const auto start = Clock::now();
    last_outputs = run.run(c.inputs, timeout_ms);
    const auto end = Clock::now();
    validate_outputs(c, last_outputs);
    if (require_deterministic) {
      const auto sig = semantic_signature(c, last_outputs);
      if (!have_signature) {
        reference_signature = sig;
        have_signature = true;
      } else {
        require(sig == reference_signature,
                c.name + ": semantic output signature changed for identical input at iteration " +
                    std::to_string(i));
      }
    }
    samples_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }

  const simaai::neat::MeasureReport measure_report = measure_scope.stop();
  const TimingSummary timing = summarize(samples_ms);
  const ElementTimingLookup plugin_timing =
      lookup_element_timing_report(measure_report, c.graph_name);
  dump_timing_diag_if_requested(measure_report, c.graph_name, plugin_timing);
  if (env_flag("SIMA_VISUAL_TPUT_REQUIRE_PROCESSCVU_TIMING", true)) {
    require(plugin_timing.samples > 0U && plugin_timing.avg_ms > 0.0,
            c.name + ": processcvu element timing was not captured");
  }
  std::cout << " result=PASS avg_ms=" << timing.avg_ms << " min_ms=" << timing.min_ms
            << " max_ms=" << timing.max_ms << " fps=" << timing.fps
            << " processcvu_avg_ms=" << plugin_timing.avg_ms << " processcvu_element="
            << (plugin_timing.element_name.empty() ? std::string("<none>")
                                                   : plugin_timing.element_name)
            << " processcvu_samples=" << plugin_timing.samples
            << " deterministic=" << (require_deterministic ? "checked" : "disabled")
            << " pushed=" << measure_report.input.push_count
            << " pulled=" << measure_report.input.pull_count
            << " outputs_ready=" << measure_report.counters.outputs_ready
            << " outputs_pulled=" << measure_report.counters.outputs_pulled << "\n";
  print_tensor_summary(last_outputs);
  run.close();
  return true;
}

} // namespace

int main() {
  try {
    require(simaai::neat::element_exists("neatprocesscvu"),
            "Missing SIMA processcvu plugin (neatprocesscvu).");

    const int warmup = env_int_local("SIMA_VISUAL_TPUT_WARMUP", 2);
    const int iterations = env_int_local("SIMA_VISUAL_TPUT_ITERS", 20);
    const int timeout_ms = env_int_local("SIMA_VISUAL_TPUT_TIMEOUT_MS", 30000);
    const int queue_depth = std::max(1, env_int_local("SIMA_VISUAL_TPUT_QUEUE_DEPTH", 1));
    const bool continue_on_fail = env_flag("SIMA_VISUAL_TPUT_CONTINUE_ON_FAIL", true);

    std::vector<BenchCase> cases = make_cases();
    int failures = 0;
    for (const auto& c : cases) {
      try {
        (void)run_case(c, warmup, iterations, timeout_ms, queue_depth);
      } catch (const std::exception& e) {
        ++failures;
        const std::string msg = e.what();
        std::cerr << "[case-fail] " << c.name << ": " << msg << "\n";
        if (!continue_on_fail) {
          throw;
        }
      }
    }

    if (failures != 0) {
      std::cerr << "[FAIL] visual_frontend_tput_test failures=" << failures << "/" << cases.size()
                << "\n";
      return 1;
    }
    std::cout << "\n[OK] visual_frontend_tput_test passed cases=" << cases.size() << "\n";
    return 0;
  } catch (const SkipTest& e) {
    return skip_test(e.what());
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (is_dispatcher_unavailable(msg)) {
      return skip_test("dispatcher unavailable: " + msg);
    }
    std::cerr << "[FAIL] " << msg << "\n";
    return 1;
  }
}

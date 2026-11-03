// Independent numerical oracles and known-answer fixtures for the SimaVoxelize,
// SimaPillarScatter, SimaBEVPoolPlan and SimaBEVPool Nodes.
//
// Each oracle follows the kernels library's documented arithmetic (the
// operation pages and headers of sima-neat/kernels), written without the
// kernels' code or headers. INT8 fixtures use small dyadic rationals, so every
// binary32 step is exact. BF16 BEVPool sums are order-dependent by contract;
// their oracle follows the bound plan's own point order, v1 or v2. Compile
// with -ffp-contract=off.
#pragma once

#include "nodes/sima/SimaBEVPool.h"
#include "nodes/sima/SimaBEVPoolPlan.h"
#include "nodes/sima/SimaPillarScatter.h"
#include "nodes/sima/SimaVoxelize.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

// Test-only oracles; not part of the Neat API.
namespace bevfusion_test::reference {
using namespace ::simaai::neat;

// ops::quant: product rounded to binary32, integer rounding, zero point, then
// INT8 saturation. NaN maps to the zero point.
inline std::int8_t quantize(float value, const SimaKernelQuantization& q) {
  const float product = value * q.multiplier;
  if (std::isnan(product))
    return static_cast<std::int8_t>(std::clamp(q.zero_point, -128, 127));
  double rounded = 0;
  const double p = product;
  switch (q.rounding) {
  case SimaQuantRounding::NearestEven: {
    const double floor = std::floor(p), fraction = p - floor;
    rounded = fraction > 0.5 || (fraction == 0.5 && std::fmod(floor, 2.0) != 0) ? floor + 1 : floor;
    break;
  }
  case SimaQuantRounding::TowardZero:
    rounded = std::trunc(p);
    break;
  case SimaQuantRounding::TowardPositive:
    rounded = std::ceil(p);
    break;
  case SimaQuantRounding::TowardNegative:
    rounded = std::floor(p);
    break;
  }
  const double shifted = std::clamp(rounded, -1.0e6, 1.0e6) + q.zero_point;
  return static_cast<std::int8_t>(std::clamp(shifted, -128.0, 127.0));
}

// Cast's FP32-to-BF16 rule: round the upper 16 bits to nearest-even, keeping
// infinities, signed zeros and subnormals. Every NaN stores the canonical quiet
// NaN 0x7FC0, as the BF16 outputs of Voxelize and BEVPool do.
inline std::uint16_t to_bf16(float value) {
  if (std::isnan(value))
    return 0x7FC0;
  const auto bits = std::bit_cast<std::uint32_t>(value);
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
}

// BF16 widens to binary32 exactly.
inline float from_bf16(std::uint16_t bits) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

// ----------------------------------------------------------------------------
// SimaVoxelize
// ----------------------------------------------------------------------------
struct VoxelizeResult {
  std::vector<std::int8_t> decorated;        // [1,M,T,10], INT8 output
  std::vector<std::uint16_t> decorated_bf16; // [1,M,T,10], BF16 output bits
  std::vector<std::int32_t> coordinates;     // [M,2]
  std::int32_t count = 0;
};

inline VoxelizeResult voxelize(const SimaVoxelizeOptions& opt, const std::vector<float>& points,
                               std::int32_t point_count) {
  const auto m = static_cast<std::size_t>(opt.max_pillars);
  const auto t = static_cast<std::size_t>(opt.max_points_per_pillar);
  const auto& vs = opt.voxel_size;
  const auto& range = opt.point_cloud_range;
  std::array<std::int64_t, 3> grid{};
  for (int a = 0; a < 3; ++a)
    grid[a] = static_cast<std::int64_t>(std::lround((range[a + 3] - range[a]) / vs[a]));
  struct Pillar {
    std::int64_t x = 0, y = 0;
    std::vector<std::array<float, 4>> points;
    std::array<float, 3> sum{};
  };
  std::vector<Pillar> pillars;
  std::map<std::pair<std::int64_t, std::int64_t>, std::size_t> index;
  for (std::int32_t i = 0; i < point_count; ++i) {
    const float* p = &points[static_cast<std::size_t>(i) * 4];
    std::array<std::int64_t, 3> cell{};
    bool inside = true;
    for (int a = 0; a < 3; ++a) {
      const float c = std::floor((p[a] - range[a]) / vs[a]);
      inside = inside && c >= 0 && c < static_cast<float>(grid[a]);
      cell[a] = inside ? static_cast<std::int64_t>(c) : 0;
    }
    if (!inside)
      continue;
    const auto key = std::make_pair(cell[1], cell[0]);
    auto found = index.find(key);
    if (found == index.end()) {
      if (pillars.size() == m)
        continue; // pillar capacity drops new pillars
      found = index.emplace(key, pillars.size()).first;
      pillars.push_back({cell[0], cell[1], {}, {}});
    }
    auto& pillar = pillars[found->second];
    if (pillar.points.size() == t)
      continue; // point capacity drops further points
    pillar.points.push_back({p[0], p[1], p[2], p[3]});
    for (int a = 0; a < 3; ++a)
      pillar.sum[a] = pillar.sum[a] + p[a];
  }
  VoxelizeResult result;
  const bool bf16 = opt.output_dtype == TensorDType::BFloat16;
  // An empty INT8 slot stores the quantization of 0.0f; an empty BF16 slot +0.0.
  if (bf16)
    result.decorated_bf16.assign(m * t * 10, 0);
  else
    result.decorated.assign(m * t * 10, quantize(0.0f, opt.quantization));
  result.coordinates.assign(m * 2, 0);
  result.count = static_cast<std::int32_t>(pillars.size());
  for (std::size_t j = 0; j < pillars.size(); ++j) {
    const auto& pillar = pillars[j];
    result.coordinates[j * 2 + 0] = static_cast<std::int32_t>(pillar.y);
    result.coordinates[j * 2 + 1] = static_cast<std::int32_t>(pillar.x);
    const float n = static_cast<float>(pillar.points.size());
    const std::array<float, 3> mean{pillar.sum[0] / n, pillar.sum[1] / n, pillar.sum[2] / n};
    const std::array<float, 3> centre{(static_cast<float>(pillar.x) + 0.5f) * vs[0] + range[0],
                                      (static_cast<float>(pillar.y) + 0.5f) * vs[1] + range[1],
                                      0.5f * vs[2] + range[2]};
    for (std::size_t s = 0; s < pillar.points.size(); ++s) {
      const auto& p = pillar.points[s];
      const std::array<float, 10> decorated{p[0],
                                            p[1],
                                            p[2],
                                            p[3],
                                            p[0] - mean[0],
                                            p[1] - mean[1],
                                            p[2] - mean[2],
                                            p[0] - centre[0],
                                            p[1] - centre[1],
                                            p[2] - centre[2]};
      for (std::size_t c = 0; c < 10; ++c) {
        const auto at = (j * t + s) * 10 + c;
        if (bf16)
          result.decorated_bf16[at] = to_bf16(decorated[c]);
        else
          result.decorated[at] = quantize(decorated[c], opt.quantization);
      }
    }
  }
  return result;
}

struct VoxelizeCase {
  SimaVoxelizeOptions options;
  std::vector<float> points; // [N,4]
  std::int32_t point_count = 0;
};

// 4x4 grid of 1 m cells, one 4 m vertical cell. Exercises first-appearance
// pillar order, an out-of-range point, the point-per-pillar cap, the pillar
// cap, and a trailing point beyond point_count.
inline VoxelizeCase voxelize_case(SimaKernelTarget target = SimaKernelTarget::A65) {
  VoxelizeCase c;
  c.options.voxel_size = {1.0f, 1.0f, 4.0f};
  c.options.point_cloud_range = {0.0f, 0.0f, -2.0f, 4.0f, 4.0f, 2.0f};
  c.options.quantization = {4.0f, -3, SimaQuantRounding::NearestEven};
  c.options.max_pillars = 3;
  c.options.max_points_per_pillar = 8; // EV74's bucketed passes need a multiple of 8
  c.options.target = target;
  const std::vector<std::array<float, 4>> points{
      {2.25f, 1.5f, 0.5f, 0.0f},    // cell (2,1): pillar 0
      {0.5f, 3.75f, -1.0f, 0.25f},  // cell (0,3): pillar 1
      {2.75f, 1.25f, 1.0f, 0.5f},   // pillar 0
      {5.0f, 1.0f, 0.0f, 0.0f},     // x outside the range: dropped
      {3.5f, 0.5f, 1.5f, 0.0f},     // cell (3,0): pillar 2
      {1.5f, 1.5f, 0.0f, 0.0f},     // cell (1,1): pillar cap, dropped
      {2.5f, 1.75f, -1.5f, 0.125f}, // pillar 0
      {0.25f, 3.25f, 0.25f, 0.0f},  // pillar 1
      {0.75f, 0.75f, 0.0f, 0.0f},   // beyond point_count: never read
  };
  for (const auto& p : points)
    c.points.insert(c.points.end(), p.begin(), p.end());
  // Fill pillar 0 past the eight-point cap.
  for (int i = 0; i < 6; ++i) {
    const std::array<float, 4> extra{2.5f, 1.5f, 0.125f * static_cast<float>(i), 0.0f};
    c.points.insert(c.points.end() - 4, extra.begin(), extra.end());
  }
  c.point_count = static_cast<std::int32_t>(c.points.size() / 4) - 1;
  return c;
}

// The same sweep with a BF16 decorated tensor and the identity quantization.
// Some decorations need more than BF16's eight significant bits, so rounding
// is exercised.
inline VoxelizeCase voxelize_case_bf16(SimaKernelTarget target = SimaKernelTarget::A65) {
  auto c = voxelize_case(target);
  c.options.output_dtype = TensorDType::BFloat16;
  c.options.quantization = {};
  c.points[0] = 2.00390625f;
  c.points[1] = 1.50390625f;
  return c;
}

// ----------------------------------------------------------------------------
// SimaPillarScatter
// ----------------------------------------------------------------------------
// Canvas [1,H,W,C] in logical NHWC order: INT8 codes or BF16 bits. Elements are
// copied unchanged; empty cells hold zero_point (0, that is +0.0, for BF16).
template <class T>
std::vector<T> pillar_scatter(const SimaPillarScatterOptions& opt, const std::vector<T>& features,
                              const std::vector<std::int32_t>& coordinates, std::int32_t count,
                              std::size_t channels) {
  const auto h = static_cast<std::size_t>(opt.rows), w = static_cast<std::size_t>(opt.columns);
  std::vector<T> canvas(h * w * channels, static_cast<T>(opt.zero_point));
  for (std::int32_t j = 0; j < count; ++j) {
    const auto row = static_cast<std::size_t>(coordinates[2 * j]);
    const auto column = static_cast<std::size_t>(coordinates[2 * j + 1]);
    if (row >= h || column >= w)
      throw std::out_of_range("pillar coordinate outside the canvas");
    std::memcpy(&canvas[(row * w + column) * channels], &features[j * channels],
                channels * sizeof(T));
  }
  return canvas;
}

struct PillarScatterCase {
  SimaPillarScatterOptions options;
  std::size_t pillars = 0, channels = 0;
  std::vector<std::int8_t> features;        // [M,C], INT8
  std::vector<std::uint16_t> features_bf16; // [M,C], BF16 bits
  std::vector<std::int32_t> coordinates;    // [M,2]
  std::int32_t count = 0;
};

// The MlaTiled BF16 payload of one tile spanning the canvas: per cell, the low
// bytes of its C channels, then their high bytes (kernels tensor.h).
inline std::vector<std::uint8_t> whole_tile_bf16_bytes(const std::vector<std::uint16_t>& nhwc,
                                                       std::size_t channels) {
  std::vector<std::uint8_t> bytes(nhwc.size() * 2);
  for (std::size_t cell = 0; cell < nhwc.size() / channels; ++cell)
    for (std::size_t c = 0; c < channels; ++c) {
      const auto value = nhwc[cell * channels + c];
      bytes[cell * 2 * channels + c] = static_cast<std::uint8_t>(value & 0xFF);
      bytes[cell * 2 * channels + channels + c] = static_cast<std::uint8_t>(value >> 8);
    }
  return bytes;
}

// The bytes a BF16 SimaPillarScatter publishes: the oracle canvas in the MlaTiled
// BF16 payload of its one whole-canvas tile, not interleaved NHWC elements.
inline std::vector<std::uint8_t> pillar_scatter_bf16_bytes(
    const SimaPillarScatterOptions& opt, const std::vector<std::uint16_t>& features,
    const std::vector<std::int32_t>& coordinates, std::int32_t count, std::size_t channels) {
  return whole_tile_bf16_bytes(pillar_scatter(opt, features, coordinates, count, channels),
                               channels);
}

inline PillarScatterCase pillar_scatter_case(SimaKernelTarget target = SimaKernelTarget::A65) {
  PillarScatterCase c;
  c.options.rows = 4;
  c.options.columns = 6;
  c.options.zero_point = -7;
  c.options.target = target;
  c.pillars = 5;
  c.channels = 16;
  c.features.resize(c.pillars * c.channels);
  for (std::size_t i = 0; i < c.features.size(); ++i)
    c.features[i] = static_cast<std::int8_t>(static_cast<int>(i * 7 % 251) - 125);
  // Three real pillars; the remaining rows hold garbage the kernel must ignore.
  c.coordinates = {0, 5, 3, 0, 2, 2, 9, 9, -1, 40};
  c.count = 3;
  return c;
}

// BF16 features, zero_point 0. The bits include NaN payloads, infinities,
// signed zero and a subnormal, which must all pass through unchanged.
inline PillarScatterCase pillar_scatter_case_bf16(SimaKernelTarget target = SimaKernelTarget::A65) {
  auto c = pillar_scatter_case(target);
  c.options.zero_point = 0;
  c.features.clear();
  c.features_bf16.resize(c.pillars * c.channels);
  constexpr std::array<std::uint16_t, 8> special{0x7FC1, 0xFF81, 0x7F80, 0xFF80,
                                                 0x8000, 0x0001, 0x3F81, 0xC2F7};
  for (std::size_t i = 0; i < c.features_bf16.size(); ++i)
    c.features_bf16[i] =
        i % 5 == 0 ? special[i / 5 % special.size()]
                   : to_bf16(static_cast<float>(static_cast<int>(i * 7 % 251) - 125) / 8.0f);
  return c;
}

// ----------------------------------------------------------------------------
// SimaBEVPoolPlan (plan layout v1)
// ----------------------------------------------------------------------------
using Matrix4 = std::array<float, 16>; // row-major

struct Calibration {
  std::vector<Matrix4> camera2lidar, intrinsics, img_aug;
  Matrix4 lidar_aug{};
};

inline Matrix4 identity() {
  return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
}

namespace detail {
using Matrix3 = std::array<float, 9>;
inline Matrix3 rotation(const Matrix4& m) {
  return {m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]};
}
inline std::array<float, 3> translation(const Matrix4& m) {
  return {m[3], m[7], m[11]};
}
// Inverse in binary64, rounded once to binary32.
inline Matrix3 inverse(const Matrix3& f) {
  std::array<double, 9> m{};
  std::copy(f.begin(), f.end(), m.begin());
  const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
                     m[2] * (m[3] * m[7] - m[4] * m[6]);
  if (det == 0)
    throw std::invalid_argument("singular calibration matrix");
  const std::array<double, 9> adj{
      m[4] * m[8] - m[5] * m[7], m[2] * m[7] - m[1] * m[8], m[1] * m[5] - m[2] * m[4],
      m[5] * m[6] - m[3] * m[8], m[0] * m[8] - m[2] * m[6], m[2] * m[3] - m[0] * m[5],
      m[3] * m[7] - m[4] * m[6], m[1] * m[6] - m[0] * m[7], m[0] * m[4] - m[1] * m[3]};
  Matrix3 result{};
  for (int i = 0; i < 9; ++i)
    result[i] = static_cast<float>(adj[i] / det);
  return result;
}
// Each three-term sum is taken left to right in binary32.
inline Matrix3 multiply(const Matrix3& a, const Matrix3& b) {
  Matrix3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      r[i * 3 + j] = (a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j]) + a[i * 3 + 2] * b[6 + j];
  return r;
}
inline std::array<float, 3> mat_vec(const Matrix3& m, const std::array<float, 3>& v) {
  return {(m[0] * v[0] + m[1] * v[1]) + m[2] * v[2], (m[3] * v[0] + m[4] * v[1]) + m[5] * v[2],
          (m[6] * v[0] + m[7] * v[1]) + m[8] * v[2]};
}
inline void put_u32(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint32_t value) {
  for (int i = 0; i < 4; ++i)
    bytes[offset + i] = static_cast<std::uint8_t>(value >> (8 * i));
}
} // namespace detail

// The used bytes of a format-1 plan: header, cell_start[cells + 1], point[K].
inline std::vector<std::uint8_t> bev_pool_plan(const SimaBEVPoolPlanOptions& opt,
                                               const Calibration& calibration) {
  using namespace detail;
  const auto cameras = calibration.camera2lidar.size();
  const auto d_bins = opt.ds.size(), rows = opt.ys.size(), columns = opt.xs.size();
  const std::uint32_t nx = opt.nx[0], ny = opt.nx[1], nz = opt.nx[2];
  const std::size_t cells = std::size_t(nx) * ny * nz;
  std::vector<std::vector<std::uint32_t>> by_cell(cells);
  const auto extra = rotation(calibration.lidar_aug);
  const auto extra_t = translation(calibration.lidar_aug);
  std::array<float, 3> lower{};
  for (int a = 0; a < 3; ++a)
    lower[a] = opt.bx[a] - opt.dx[a] / 2.0f;
  for (std::size_t n = 0; n < cameras; ++n) {
    const auto post_inverse = inverse(rotation(calibration.img_aug[n]));
    const auto post_t = translation(calibration.img_aug[n]);
    const auto combine = multiply(rotation(calibration.camera2lidar[n]),
                                  inverse(rotation(calibration.intrinsics[n])));
    const auto c2l_t = translation(calibration.camera2lidar[n]);
    for (std::size_t d = 0; d < d_bins; ++d)
      for (std::size_t h = 0; h < rows; ++h)
        for (std::size_t w = 0; w < columns; ++w) {
          const std::array<float, 3> q{opt.xs[w] - post_t[0], opt.ys[h] - post_t[1],
                                       opt.ds[d] - post_t[2]};
          const auto r = mat_vec(post_inverse, q);
          const std::array<float, 3> s{r[0] * r[2], r[1] * r[2], r[2]};
          auto g = mat_vec(combine, s);
          for (int a = 0; a < 3; ++a)
            g[a] = g[a] + c2l_t[a];
          g = mat_vec(extra, g);
          for (int a = 0; a < 3; ++a)
            g[a] = g[a] + extra_t[a];
          std::array<float, 3> v{};
          bool kept = true;
          for (int a = 0; a < 3; ++a) {
            v[a] = (g[a] - lower[a]) / opt.dx[a];
            kept = kept && v[a] > -1.0f && v[a] < static_cast<float>(opt.nx[a]);
          }
          if (!kept)
            continue;
          const auto cx = static_cast<std::uint32_t>(std::trunc(v[0]));
          const auto cy = static_cast<std::uint32_t>(std::trunc(v[1]));
          const auto cz = static_cast<std::uint32_t>(std::trunc(v[2]));
          const auto p = static_cast<std::uint32_t>(((n * d_bins + d) * rows + h) * columns + w);
          by_cell[(std::size_t(cx) * ny + cy) * nz + cz].push_back(p);
        }
  }
  std::uint32_t kept = 0, max_run = 0;
  for (const auto& run : by_cell) {
    kept += static_cast<std::uint32_t>(run.size());
    max_run = std::max<std::uint32_t>(max_run, static_cast<std::uint32_t>(run.size()));
  }
  std::vector<std::uint8_t> bytes(64 + 4 * (cells + 1) + 4 * std::size_t(kept), 0);
  const std::array<std::uint32_t, 11> header{0x4C505642U,
                                             1,
                                             static_cast<std::uint32_t>(cameras),
                                             static_cast<std::uint32_t>(d_bins),
                                             static_cast<std::uint32_t>(rows),
                                             static_cast<std::uint32_t>(columns),
                                             nx,
                                             ny,
                                             nz,
                                             kept,
                                             max_run};
  for (std::size_t i = 0; i < header.size(); ++i)
    put_u32(bytes, 4 * i, header[i]);
  std::uint32_t start = 0;
  std::size_t point = 64 + 4 * (cells + 1);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    put_u32(bytes, 64 + 4 * cell, start);
    for (const auto p : by_cell[cell]) {
      put_u32(bytes, point, p);
      point += 4;
    }
    start += static_cast<std::uint32_t>(by_cell[cell].size());
  }
  put_u32(bytes, 64 + 4 * cells, start);
  return bytes;
}

struct BEVPoolPlanCase {
  SimaBEVPoolPlanOptions options;
  Calibration calibration;
};

// Two cameras, 3x2 image samples, 2 depth bins, a 4x4x1 grid of 1 m cells. The
// intrinsics and image augmentation have exact binary32 inverses; camera 1 is
// shifted by one cell in x, so some of its points leave the grid.
inline BEVPoolPlanCase bev_pool_plan_case() {
  BEVPoolPlanCase c;
  auto& o = c.options;
  o.xs = {0.0f, 2.0f, 4.0f};
  o.ys = {0.0f, 2.0f};
  o.ds = {1.0f, 2.0f};
  o.bx = {0.5f, 0.5f, 0.0f};
  o.dx = {1.0f, 1.0f, 8.0f};
  o.nx = {4, 4, 1};
  Matrix4 k = identity();
  k[0] = 2;
  k[2] = 1; // fx = 2, cx = 1
  k[5] = 2;
  k[6] = 1; // fy = 2, cy = 1
  Matrix4 aug = identity();
  aug[0] = 0.5f;
  aug[5] = 0.5f;
  aug[3] = -0.5f;
  aug[7] = -0.5f; // resize by 1/2, then shift
  for (int n = 0; n < 2; ++n) {
    Matrix4 c2l = identity();
    c2l[3] = static_cast<float>(n); // camera 1 one metre along x
    c.calibration.camera2lidar.push_back(c2l);
    c.calibration.intrinsics.push_back(k);
    c.calibration.img_aug.push_back(aug);
  }
  c.calibration.lidar_aug = identity();
  return c;
}

// Each cell's frustum points in the plan's own order, for plan layout v1 or v2
// (kernels bev_pool.h). v1 lists cell_start[] and point[]; v2 holds per-worker
// streams of tiles, chunks, slot loads, depth windows and runs, decoded here
// back to frustum indices p = ((n*D + d)*fH + h)*fW + w. Malformed plans throw.
inline std::vector<std::vector<std::uint32_t>>
plan_cell_points(const std::vector<std::uint8_t>& plan) {
  const auto word = [&](std::size_t index) {
    if (4 * index + 4 > plan.size())
      throw std::out_of_range("plan word beyond the plan tensor");
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i)
      v |= std::uint32_t(plan[4 * index + i]) << (8 * i);
    return v;
  };
  if (word(0) != 0x4C505642U)
    throw std::invalid_argument("plan magic");
  const auto version = word(1), n_cam = word(2), depth = word(3), rows = word(4), columns = word(5),
             nx = word(6), ny = word(7), nz = word(8), kept = word(9);
  const std::size_t cells = std::size_t(nx) * ny * nz;
  std::vector<std::vector<std::uint32_t>> result(cells);
  if (version == 1) {
    const std::size_t points = 16 + cells + 1;
    for (std::size_t cell = 0; cell < cells; ++cell)
      for (auto i = word(16 + cell); i < word(16 + cell + 1); ++i)
        result[cell].push_back(word(points + i));
    return result;
  }
  if (version != 2)
    throw std::invalid_argument("plan version");
  const auto workers = word(11), tile_h = word(13), tile_w = word(14);
  std::size_t total = 0;
  for (std::uint32_t worker = 0; worker < workers; ++worker) {
    std::size_t at = word(16 + 2 * worker);
    const auto tiles = word(16 + 2 * worker + 1);
    std::vector<std::uint32_t> slot_column(256, UINT32_MAX);
    for (std::uint32_t t = 0; t < tiles; ++t) {
      const auto tile = word(at++), chunks = word(at++);
      const auto tx = tile & 0xFFFF, ty = tile >> 16;
      for (std::uint32_t chunk = 0; chunk < chunks; ++chunk) {
        const auto header = word(at++);
        const auto loads = header & 0xFFFF, windows = header >> 16;
        const auto runs = word(at++);
        (void)word(at++); // E, the chunk's point count
        for (std::uint32_t i = 0; i < loads; ++i) {
          const auto load = word(at++);
          slot_column[load & 0xFF] = load >> 8;
        }
        struct Window {
          std::uint32_t slot, d0, dn, base;
        };
        std::vector<Window> window;
        std::uint32_t base = 0;
        for (std::uint32_t j = 0; j < windows; ++j) {
          const auto w = word(at++);
          window.push_back({w & 0xFF, (w >> 8) & 0xFFF, w >> 20, base});
          base += rows * (w >> 20);
        }
        for (std::uint32_t r = 0; r < runs; ++r) {
          const auto run = word(at++);
          const auto local = run & 0x3FF, count = run >> 10;
          const auto z = local % nz, xy = local / nz;
          const auto x = tx * tile_h + xy / tile_w, y = ty * tile_w + xy % tile_w;
          const std::size_t cell = (std::size_t(x) * ny + y) * nz + z;
          if (x >= nx || y >= ny || cell >= cells)
            throw std::out_of_range("v2 run cell outside the grid");
          for (std::uint32_t i = 0; i < count; ++i) {
            const auto point = word(at++);
            const auto offset = point & 0x3FFF, row = point >> 14;
            const auto found = std::find_if(window.begin(), window.end(), [&](const Window& w) {
              return offset >= w.base && offset < w.base + rows * w.dn;
            });
            if (found == window.end())
              throw std::out_of_range("v2 point outside its chunk's windows");
            const auto inner = offset - found->base, h = inner / found->dn,
                       d = found->d0 + inner % found->dn;
            const auto column = slot_column[found->slot];
            if (row != found->slot * rows + h || column == UINT32_MAX || d >= depth ||
                column >= n_cam * columns)
              throw std::invalid_argument("v2 point does not name one frustum point");
            const auto n = column / columns, w = column % columns;
            result[cell].push_back(((n * depth + d) * rows + h) * columns + w);
            ++total;
          }
        }
      }
    }
  }
  if (total != kept)
    throw std::invalid_argument("v2 streams do not hold the plan's K points");
  return result;
}

// ----------------------------------------------------------------------------
// SimaBEVPool
// ----------------------------------------------------------------------------
struct BEVPoolCase {
  SimaBEVPoolOptions options;
  std::int64_t cameras = 0, depth_bins = 0, rows = 0, columns = 0, channels = 0;
  // Logical [N,D,fH,fW] and [N,C,fH,fW], in logical order: INT8 codes, or BF16
  // bits when bf16 is set.
  std::vector<std::int8_t> depth, context;
  bool bf16 = false;
  std::vector<std::uint16_t> depth_bf16, context_bf16;
  std::vector<std::uint8_t> plan;
};

// bev, logical [1,nZ*C,nX,nY], in logical order.
inline std::vector<std::int8_t> bev_pool(const BEVPoolCase& c) {
  const auto& o = c.options;
  const auto nx = o.cells[0], ny = o.cells[1], nz = o.cells[2];
  const auto cells = std::size_t(nx) * ny * nz;
  const auto plan = plan_cell_points(c.plan);
  const auto k = static_cast<std::size_t>(nz * c.channels);
  std::vector<std::int8_t> bev(k * nx * ny);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const auto x = cell / (std::size_t(ny) * nz), y = cell / nz % ny, z = cell % nz;
    for (std::int64_t channel = 0; channel < c.channels; ++channel) {
      std::int32_t acc = 0;
      for (const auto p : plan[cell]) {
        const auto w = p % c.columns, h = p / c.columns % c.rows;
        const auto n = p / (c.columns * c.rows * c.depth_bins);
        const auto ctx = c.context[((n * c.channels + channel) * c.rows + h) * c.columns + w];
        acc += (c.depth[p] - o.depth_zero_point) * (ctx - o.context_zero_point);
      }
      const auto out_channel = z * c.channels + channel;
      bev[(out_channel * nx + x) * ny + y] = quantize(static_cast<float>(acc), o.quantization);
    }
  }
  return bev;
}

// BF16 inputs: per cell and channel, a binary32 sum from +0.0 over the plan's
// points in plan order, each product rounded to binary32 before the addition.
// Returns the sums in logical [1,nZ*C,nX,nY] order.
inline std::vector<float> bev_pool_bf16_sums(const BEVPoolCase& c) {
  const auto& o = c.options;
  const auto nx = o.cells[0], ny = o.cells[1], nz = o.cells[2];
  const auto cells = std::size_t(nx) * ny * nz;
  const auto plan = plan_cell_points(c.plan);
  const auto k = static_cast<std::size_t>(nz * c.channels);
  std::vector<float> sums(k * nx * ny);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const auto x = cell / (std::size_t(ny) * nz), y = cell / nz % ny, z = cell % nz;
    for (std::int64_t channel = 0; channel < c.channels; ++channel) {
      float sum = +0.0f;
      for (const auto p : plan[cell]) {
        const auto w = p % c.columns, h = p / c.columns % c.rows;
        const auto n = p / (c.columns * c.rows * c.depth_bins);
        const float product =
            from_bf16(c.depth_bf16[p]) *
            from_bf16(c.context_bf16[((n * c.channels + channel) * c.rows + h) * c.columns + w]);
        sum = sum + product;
      }
      sums[((z * c.channels + channel) * nx + x) * ny + y] = sum;
    }
  }
  return sums;
}

// A BF16 bev: each sum rounded to BF16, NaN as 0x7FC0.
inline std::vector<std::uint16_t> bev_pool_bf16(const BEVPoolCase& c) {
  std::vector<std::uint16_t> bev;
  for (const auto sum : bev_pool_bf16_sums(c))
    bev.push_back(to_bf16(sum));
  return bev;
}

// An INT8 bev from BF16 inputs: Quant of each binary32 sum.
inline std::vector<std::int8_t> bev_pool_bf16_to_int8(const BEVPoolCase& c) {
  std::vector<std::int8_t> bev;
  for (const auto sum : bev_pool_bf16_sums(c))
    bev.push_back(quantize(sum, c.options.quantization));
  return bev;
}

// Logical [A,K,H,W] values to dense channels-last [A,H,W,K] bytes.
template <class T>
std::vector<T> to_channels_last(const std::vector<T>& logical, std::int64_t a, std::int64_t k,
                                std::int64_t h, std::int64_t w) {
  std::vector<T> out(logical.size());
  for (std::int64_t i = 0; i < a; ++i)
    for (std::int64_t c = 0; c < k; ++c)
      for (std::int64_t y = 0; y < h; ++y)
        for (std::int64_t x = 0; x < w; ++x)
          out[((i * h + y) * w + x) * k + c] = logical[((i * k + c) * h + y) * w + x];
  return out;
}

// The SimaBEVPoolPlan fixture's frustum with 16 context channels (EV74 needs a
// multiple of 16), pooled from that fixture's reference plan.
inline BEVPoolCase bev_pool_case(SimaKernelTarget target = SimaKernelTarget::A65) {
  const auto plan_case = bev_pool_plan_case();
  BEVPoolCase c;
  c.cameras = 2;
  c.depth_bins = static_cast<std::int64_t>(plan_case.options.ds.size());
  c.rows = static_cast<std::int64_t>(plan_case.options.ys.size());
  c.columns = static_cast<std::int64_t>(plan_case.options.xs.size());
  c.channels = 16;
  c.options.cells = plan_case.options.nx;
  c.options.depth_zero_point = -128;
  c.options.context_zero_point = -5;
  c.options.quantization = {0.0625f, 3, SimaQuantRounding::NearestEven};
  c.options.target = target;
  const auto frustum = c.cameras * c.depth_bins * c.rows * c.columns;
  c.depth.resize(static_cast<std::size_t>(frustum));
  for (std::size_t p = 0; p < c.depth.size(); ++p)
    c.depth[p] = static_cast<std::int8_t>(static_cast<int>(p * 37 % 256) - 128);
  c.context.resize(static_cast<std::size_t>(c.cameras * c.channels * c.rows * c.columns));
  for (std::size_t i = 0; i < c.context.size(); ++i)
    c.context[i] = static_cast<std::int8_t>(static_cast<int>(i * 13 % 61) - 30);
  c.plan = bev_pool_plan(plan_case.options, plan_case.calibration);
  // SimaBEVPool admits a v1 plan tensor that holds the worst case.
  const auto cells = std::size_t(c.options.cells[0]) * c.options.cells[1] * c.options.cells[2];
  c.plan.resize(64 + 4 * (cells + 1) + 4 * static_cast<std::size_t>(frustum), 0);
  return c;
}

// The same frustum and plan with BF16 depth and context and zero points of 0.
// Products are generally inexact, so the plan-order binary32 sum is observable;
// one context element is NaN and one is -0.0. A BF16 bev uses the identity
// quantization; an INT8 bev quantizes the sums.
inline BEVPoolCase bev_pool_case_bf16(SimaKernelTarget target = SimaKernelTarget::A65,
                                      TensorDType output = TensorDType::BFloat16) {
  auto c = bev_pool_case(target);
  c.bf16 = true;
  c.options.depth_zero_point = 0;
  c.options.context_zero_point = 0;
  c.options.output_dtype = output;
  c.options.quantization = output == TensorDType::BFloat16
                               ? SimaKernelQuantization{}
                               : SimaKernelQuantization{4.0f, -2, SimaQuantRounding::NearestEven};
  c.depth_bf16.resize(c.depth.size());
  for (std::size_t p = 0; p < c.depth_bf16.size(); ++p)
    c.depth_bf16[p] = to_bf16(0.0371f * static_cast<float>(p % 11) + 0.0123f);
  c.context_bf16.resize(c.context.size());
  for (std::size_t i = 0; i < c.context_bf16.size(); ++i)
    c.context_bf16[i] = to_bf16(static_cast<float>(static_cast<int>(i * 13 % 61) - 30) / 7.3f);
  c.context_bf16[0] = 0x7FC0; // NaN at (n0, c0, h0, w0): kept points sum to NaN
  c.context_bf16[1] = 0x8000; // -0.0 at (n0, c0, h0, w1)
  c.depth.clear();
  c.context.clear();
  return c;
}

// A hand-built v1 plan that puts the 12 depth-bin-0 points in cell 5, so one
// sum has 12 terms. In channel 0 the products are 1, 2^-8 and ten of 2^-30:
// binary32 addition in plan order holds 1 + 2^-8, a BF16 rounding tie, and
// rounds to 0x3F80; any wider accumulator would round up to 0x3F81.
inline BEVPoolCase
bev_pool_case_bf16_accumulation(SimaKernelTarget target = SimaKernelTarget::A65) {
  auto c = bev_pool_case_bf16(target, TensorDType::BFloat16);
  const auto cells = std::size_t(c.options.cells[0]) * c.options.cells[1] * c.options.cells[2];
  constexpr std::size_t cell = 5;
  std::vector<std::uint32_t> points;
  for (std::int64_t n = 0; n < c.cameras; ++n)
    for (std::int64_t h = 0; h < c.rows; ++h)
      for (std::int64_t w = 0; w < c.columns; ++w)
        points.push_back(
            static_cast<std::uint32_t>(((n * c.depth_bins + 0) * c.rows + h) * c.columns + w));
  std::fill(c.plan.begin(), c.plan.end(), 0);
  const std::array<std::uint32_t, 11> header{0x4C505642U,
                                             1,
                                             static_cast<std::uint32_t>(c.cameras),
                                             static_cast<std::uint32_t>(c.depth_bins),
                                             static_cast<std::uint32_t>(c.rows),
                                             static_cast<std::uint32_t>(c.columns),
                                             static_cast<std::uint32_t>(c.options.cells[0]),
                                             static_cast<std::uint32_t>(c.options.cells[1]),
                                             static_cast<std::uint32_t>(c.options.cells[2]),
                                             static_cast<std::uint32_t>(points.size()),
                                             static_cast<std::uint32_t>(points.size())};
  for (std::size_t i = 0; i < header.size(); ++i)
    detail::put_u32(c.plan, 4 * i, header[i]);
  for (std::size_t i = 0; i <= cells; ++i)
    detail::put_u32(c.plan, 64 + 4 * i, i <= cell ? 0 : static_cast<std::uint32_t>(points.size()));
  for (std::size_t i = 0; i < points.size(); ++i)
    detail::put_u32(c.plan, 64 + 4 * (cells + 1) + 4 * i, points[i]);
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto p = points[i];
    const auto w = p % c.columns, h = p / c.columns % c.rows;
    const auto n = p / (c.columns * c.rows * c.depth_bins);
    const std::uint16_t value = i == 0 ? 0x3F80 : i == 1 ? 0x3D80 : 0x3800; // 1, 2^-4, 2^-15
    c.depth_bf16[p] = value;
    c.context_bf16[((n * c.channels + 0) * c.rows + h) * c.columns + w] = value;
  }
  return c;
}

// ----------------------------------------------------------------------------
// Scenario fixtures shared by the hardware suites and host cross-checks
// ----------------------------------------------------------------------------
// The sweep translated by whole cells (x -= dx, y -= dy): still exact; points
// may leave the grid.
inline std::vector<float> voxelize_shifted(const VoxelizeCase& c, float dx, float dy) {
  auto points = c.points;
  for (std::size_t i = 0; i + 3 < points.size(); i += 4) {
    points[i] -= dx;
    points[i + 1] -= dy;
  }
  return points;
}
// Sixteen points in cell (1, 2); means divide by 8 or 16, so stay exact.
inline std::vector<float> voxelize_dense_cell() {
  std::vector<float> points;
  for (int i = 0; i < 16; ++i) {
    const std::array<float, 4> p{1.25f + 0.03125f * float(i), 2.5f, 0.125f * float(i - 8),
                                 0.03125f * float(i)};
    points.insert(points.end(), p.begin(), p.end());
  }
  return points;
}

// Every canvas cell occupied once, in a scrambled order, with distinct values.
inline PillarScatterCase pillar_scatter_full(const PillarScatterCase& base) {
  auto c = base;
  const auto cells = static_cast<std::size_t>(c.options.rows * c.options.columns);
  std::vector<std::size_t> order(cells);
  for (std::size_t i = 0; i < cells; ++i)
    order[i] = i;
  for (std::size_t i = 0; i < cells; ++i)
    std::swap(order[i], order[(i * 7 + 3) % cells]);
  c.coordinates.clear();
  for (const auto cell : order) {
    c.coordinates.push_back(static_cast<std::int32_t>(cell / c.options.columns));
    c.coordinates.push_back(static_cast<std::int32_t>(cell % c.options.columns));
  }
  const bool wide = !base.features_bf16.empty();
  c.features.clear();
  c.features_bf16.clear();
  for (std::size_t i = 0; i < cells * c.channels; ++i) {
    if (wide)
      c.features_bf16.push_back(to_bf16(static_cast<float>(int(i % 97) - 48) / 4.0f));
    else
      c.features.push_back(static_cast<std::int8_t>(int(i * 11 % 255) - 127));
  }
  c.pillars = cells;
  c.count = static_cast<std::int32_t>(cells);
  return c;
}
// The fixture's pillars with `channels` channels of fresh values.
inline PillarScatterCase pillar_scatter_channels(const PillarScatterCase& base,
                                                 std::size_t channels) {
  auto c = base;
  const bool wide = !base.features_bf16.empty();
  c.channels = channels;
  c.features.clear();
  c.features_bf16.clear();
  for (std::size_t i = 0; i < c.pillars * channels; ++i) {
    if (wide)
      c.features_bf16.push_back(to_bf16(static_cast<float>(int(i % 61) - 30) * 0.375f));
    else
      c.features.push_back(static_cast<std::int8_t>(int(i * 13 % 251) - 125));
  }
  return c;
}
// Push `generation` of a reuse sequence: fewer pillars at moved cells.
inline PillarScatterCase pillar_scatter_generation(const PillarScatterCase& base, int generation) {
  auto c = base;
  c.count = std::min<std::int32_t>(base.count, generation + 1);
  for (std::size_t j = 0; j < 3; ++j) {
    c.coordinates[2 * j] = static_cast<std::int32_t>((j + generation) % c.options.rows);
    c.coordinates[2 * j + 1] = static_cast<std::int32_t>((2 * j + generation) % c.options.columns);
  }
  return c;
}

// Camera 1 moved `dx` metres along x.
inline Calibration calibration_moved(const Calibration& base, float dx) {
  auto result = base;
  result.camera2lidar[1][3] = dx;
  return result;
}
// Camera 0's image augmentation resized by 1/4 instead of 1/2.
inline Calibration calibration_zoomed(const Calibration& base) {
  auto result = base;
  result.img_aug[0][0] = result.img_aug[0][5] = 0.25f;
  return result;
}

// nZ = 2 over the same frustum: bev channel z*C + c.
inline BEVPoolCase bev_pool_case_two_heights(SimaKernelTarget target = SimaKernelTarget::A65) {
  auto c = bev_pool_case(target);
  auto p = bev_pool_plan_case();
  p.options.nx = {4, 4, 2};
  p.options.dx = {1.0f, 1.0f, 4.0f};
  c.options.cells = p.options.nx;
  c.plan = bev_pool_plan(p.options, p.calibration);
  c.plan.resize(64 + 4 * (32 + 1) + 4 * 24, 0);
  return c;
}
// Push `generation` of a reuse sequence: new depth and context, same plan.
inline BEVPoolCase bev_pool_generation(const BEVPoolCase& base, int generation) {
  auto c = base;
  for (auto& v : c.depth)
    v = static_cast<std::int8_t>(int(v) * (generation + 1) / 3);
  for (auto& v : c.context)
    v = static_cast<std::int8_t>(int(v) - generation * 7);
  for (auto& v : c.depth_bf16)
    v = to_bf16(from_bf16(v) * float(generation + 1));
  for (auto& v : c.context_bf16)
    v = to_bf16(from_bf16(v) - 0.5f * float(generation));
  return c;
}

} // namespace bevfusion_test::reference

// Independent ResizeArgMax oracle for the ResizeArgMax suites, written from the
// normative rule in the Kernels operation contract: PyTorch upsample_bilinear2d
// source coordinates, the interpolation
//   r = hy * (hx * v00 + lx * v01) + ly * (hx * v10 + lx * v11)
// in uncontracted FP32, and the first class with the largest r (for floating
// scores, the first NaN wins). Build users with -ffp-contract=off.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace resize_argmax_test::reference {

enum class Scores { Int8, UInt8, BFloat16, Float32 };

inline std::size_t scalar_bytes(Scores scores) {
  switch (scores) {
  case Scores::Int8:
  case Scores::UInt8:
    return 1;
  case Scores::BFloat16:
    return 2;
  case Scores::Float32:
    return 4;
  }
  throw std::invalid_argument("resize_argmax reference: unknown score type");
}

// One output row or column: the two source indices and their FP32 weights.
struct Tap {
  std::size_t i0 = 0, i1 = 0;
  float lambda = 0.0f, complement = 1.0f;
};

inline Tap tap(std::size_t index, std::size_t input, std::size_t output, bool align_corners) {
  const float i = static_cast<float>(index);
  float source = 0.0f;
  if (align_corners) {
    if (output > 1)
      source = i * (static_cast<float>(input - 1) / static_cast<float>(output - 1));
  } else {
    const float scale = static_cast<float>(input) / static_cast<float>(output);
    const float scaled = (i + 0.5f) * scale;
    source = std::max(scaled - 0.5f, 0.0f);
  }
  Tap result;
  result.i0 = std::min(static_cast<std::size_t>(std::floor(source)), input - 1);
  result.i1 = std::min(result.i0 + 1, input - 1);
  result.lambda = source - static_cast<float>(result.i0);
  result.complement = 1.0f - result.lambda;
  return result;
}

// Channels-last scores [N, h, w, C] stored at the given byte strides (pixel
// pitch may exceed C; padding is never read).
struct Input {
  const std::uint8_t* bytes = nullptr;
  Scores scores = Scores::Int8;
  std::size_t n = 1, h = 0, w = 0, c = 0;
  std::size_t stride_n = 0, stride_h = 0, stride_w = 0; ///< Bytes; channels are packed.
};

inline float value(const Input& in, std::size_t n, std::size_t y, std::size_t x, std::size_t c) {
  const auto* at =
      in.bytes + n * in.stride_n + y * in.stride_h + x * in.stride_w + c * scalar_bytes(in.scores);
  switch (in.scores) {
  case Scores::Int8:
    return static_cast<float>(static_cast<std::int8_t>(*at));
  case Scores::UInt8:
    return static_cast<float>(*at);
  case Scores::BFloat16: {
    std::uint16_t bits;
    std::memcpy(&bits, at, 2);
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float result;
    std::memcpy(&result, &wide, 4);
    return result;
  }
  case Scores::Float32: {
    float result;
    std::memcpy(&result, at, 4);
    return result;
  }
  }
  return 0.0f;
}

// Class indices [N, H, W], row-major, as int32 (callers narrow to UINT8 when C <= 256).
inline std::vector<std::int32_t> resize_argmax(const Input& in, std::size_t out_h,
                                               std::size_t out_w, bool align_corners) {
  if (!in.bytes || !in.n || !in.h || !in.w || !in.c || !out_h || !out_w)
    throw std::invalid_argument("resize_argmax reference: empty extent");
  std::vector<std::int32_t> result(in.n * out_h * out_w);
  for (std::size_t n = 0; n < in.n; ++n)
    for (std::size_t y = 0; y < out_h; ++y) {
      const auto ty = tap(y, in.h, out_h, align_corners);
      for (std::size_t x = 0; x < out_w; ++x) {
        const auto tx = tap(x, in.w, out_w, align_corners);
        std::size_t best = 0;
        float best_r = 0.0f;
        for (std::size_t c = 0; c < in.c; ++c) {
          const float top = tx.complement * value(in, n, ty.i0, tx.i0, c) +
                            tx.lambda * value(in, n, ty.i0, tx.i1, c);
          const float bottom = tx.complement * value(in, n, ty.i1, tx.i0, c) +
                               tx.lambda * value(in, n, ty.i1, tx.i1, c);
          const float r = ty.complement * top + ty.lambda * bottom;
          if (c == 0 || (!std::isnan(best_r) && (std::isnan(r) || r > best_r))) {
            best = c;
            best_r = r;
          }
        }
        result[(n * out_h + y) * out_w + x] = static_cast<std::int32_t>(best);
      }
    }
  return result;
}

} // namespace resize_argmax_test::reference

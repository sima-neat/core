#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "gst/GstHelpers.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/Tess.h"
#include "pipeline/Graph.h"
#include "pipeline/TensorCore.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/runtime/RunInternal.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapterInternal.h"
#include "test_utils.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct TessBenchCase {
  std::string name;
  int input_width = 0;
  int input_height = 0;
  int input_depth = 1;
  int input_channels = 1;
  int tile_width = 0;
  int tile_height = 0;
  int tile_depth = 1;
  int tile_channels = 1;
  int elem_bytes = 1;
  bool byte_align = true;
  std::string dtype = "INT8";
  std::string format = "INT8";
};

struct BenchOptions {
  std::string case_name = "all";
  int iterations = 200;
  int warmup = 25;
  int timeout_ms = 20000;
  std::uint32_t seed = 0x12345678u;
};

struct TileGrid {
  std::vector<int> widths;
  std::vector<int> heights;
  std::vector<int> depths;
  std::vector<int> channels;
  std::vector<std::size_t> tile_sizes;
  std::vector<std::size_t> tile_offsets;
  std::size_t payload_bytes = 0;
  std::size_t aligned_bytes = 0;
};

struct CompareStats {
  std::size_t payload_bytes = 0;
  std::size_t payload_mismatches = 0;
  std::size_t padding_bytes = 0;
  std::size_t padding_mismatches = 0;
  std::size_t first_diff_absolute = std::numeric_limits<std::size_t>::max();
  std::size_t first_diff_tile = std::numeric_limits<std::size_t>::max();
  bool exact_payload_match = true;
  bool exact_full_match = true;
};

struct TimingSummary {
  double avg_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double approx_gbps = 0.0;
};

struct Ev74RunSummary {
  TimingSummary wall;
  TimingSummary element;
  std::vector<std::uint64_t> payload_hashes;
  std::vector<std::uint64_t> full_hashes;
  std::vector<std::uint8_t> first_output;
};

std::string upper_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

std::size_t align16(std::size_t value) {
  return (value + 15U) & ~std::size_t(15U);
}

std::vector<int> make_tile_sizes(int total, int tile) {
  if (total <= 0 || tile <= 0) {
    throw std::runtime_error("invalid tess geometry");
  }
  std::vector<int> out;
  while (total > 0) {
    const int chunk = std::min(total, tile);
    out.push_back(chunk);
    total -= chunk;
  }
  return out;
}

std::uint64_t fnv1a64(const std::uint8_t* data, std::size_t size_bytes) {
  constexpr std::uint64_t kOffset = 1469598103934665603ULL;
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  std::uint64_t h = kOffset;
  for (std::size_t i = 0; i < size_bytes; ++i) {
    h ^= static_cast<std::uint64_t>(data[i]);
    h *= kPrime;
  }
  return h;
}

std::string hex64(std::uint64_t value) {
  std::ostringstream oss;
  oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
  return oss.str();
}

std::uint16_t fp32_to_bf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t lsb = (bits >> 16U) & 1U;
  bits += 0x7FFFU + lsb;
  return static_cast<std::uint16_t>(bits >> 16U);
}

TileGrid build_tile_grid(const TessBenchCase& c) {
  TileGrid out;
  out.widths = make_tile_sizes(c.input_width, c.tile_width);
  out.heights = make_tile_sizes(c.input_height, c.tile_height);
  out.depths = make_tile_sizes(c.input_depth, c.tile_depth);
  out.channels = make_tile_sizes(c.input_channels, c.tile_channels);

  std::size_t offset = 0;
  for (int tc : out.channels) {
    for (int td : out.depths) {
      for (int th : out.heights) {
        for (int tw : out.widths) {
          const std::size_t payload = static_cast<std::size_t>(tc) * static_cast<std::size_t>(td) *
                                      static_cast<std::size_t>(th) * static_cast<std::size_t>(tw) *
                                      static_cast<std::size_t>(c.elem_bytes);
          const std::size_t aligned = c.byte_align ? align16(payload) : payload;
          out.tile_offsets.push_back(offset);
          out.tile_sizes.push_back(aligned);
          out.payload_bytes += payload;
          out.aligned_bytes += aligned;
          offset += aligned;
        }
      }
    }
  }
  return out;
}

void store_lane_major_pixel_generic(std::uint8_t* dst, const std::uint8_t* src, int channels,
                                    int elem_bytes) {
  for (int byte_lane = 0; byte_lane < elem_bytes; ++byte_lane) {
    for (int ch = 0; ch < channels; ++ch) {
      dst[byte_lane * channels + ch] = src[ch * elem_bytes + byte_lane];
    }
  }
}

#if defined(__aarch64__) || defined(__ARM_NEON)
bool can_use_neon_fast_path(const TessBenchCase& c) {
  return c.input_depth == 1 && c.tile_depth == 1 && c.byte_align &&
         c.input_channels == c.tile_channels && c.elem_bytes == 2 &&
         (c.input_channels == 1 || c.input_channels == 2);
}

void tessellate_neon_bf16_row_c2(std::uint8_t* dst, const std::uint8_t* src, int pixels) {
  int x = 0;
  for (; x + 8 <= pixels; x += 8) {
    const auto deinterleaved = vld2q_u8(src + static_cast<std::size_t>(x) * 4U);
    const auto zipped = vzipq_u16(vreinterpretq_u16_u8(deinterleaved.val[0]),
                                  vreinterpretq_u16_u8(deinterleaved.val[1]));
    vst1q_u8(dst + static_cast<std::size_t>(x) * 4U, vreinterpretq_u8_u16(zipped.val[0]));
    vst1q_u8(dst + static_cast<std::size_t>(x) * 4U + 16U, vreinterpretq_u8_u16(zipped.val[1]));
  }
  for (; x < pixels; ++x) {
    dst[0] = src[0];
    dst[1] = src[2];
    dst[2] = src[1];
    dst[3] = src[3];
    src += 4;
    dst += 4;
  }
}

void tessellate_neon_tile_row(const TessBenchCase& c, std::uint8_t* dst, const std::uint8_t* src,
                              int pixels) {
  if (c.elem_bytes == 2 && c.input_channels == 1) {
    std::memcpy(dst, src, static_cast<std::size_t>(pixels) * 2U);
    return;
  }
  if (c.elem_bytes == 2 && c.input_channels == 2) {
    tessellate_neon_bf16_row_c2(dst, src, pixels);
    return;
  }
  for (int x = 0; x < pixels; ++x) {
    store_lane_major_pixel_generic(
        dst + static_cast<std::size_t>(x) * static_cast<std::size_t>(c.input_channels) *
                  static_cast<std::size_t>(c.elem_bytes),
        src + static_cast<std::size_t>(x) * static_cast<std::size_t>(c.input_channels) *
                  static_cast<std::size_t>(c.elem_bytes),
        c.input_channels, c.elem_bytes);
  }
}
#else
bool can_use_neon_fast_path(const TessBenchCase&) {
  return false;
}
#endif

std::vector<std::uint8_t> tessellate_host_generic(const TessBenchCase& c,
                                                  const std::vector<std::uint8_t>& input) {
  const TileGrid grid = build_tile_grid(c);
  const std::size_t expected_input_bytes =
      static_cast<std::size_t>(c.input_width) * static_cast<std::size_t>(c.input_height) *
      static_cast<std::size_t>(c.input_depth) * static_cast<std::size_t>(c.input_channels) *
      static_cast<std::size_t>(c.elem_bytes);
  require(input.size() == expected_input_bytes, "host tess input size mismatch");

  std::vector<std::uint8_t> output(grid.aligned_bytes, 0);
  std::size_t tile_index = 0;
  int channel_start = 0;
  for (int tc : grid.channels) {
    int depth_start = 0;
    for (int td : grid.depths) {
      int y_start = 0;
      for (int th : grid.heights) {
        int x_start = 0;
        for (int tw : grid.widths) {
          std::uint8_t* tile_dst = output.data() + grid.tile_offsets[tile_index];
          for (int d = 0; d < td; ++d) {
            for (int y = 0; y < th; ++y) {
              const std::size_t tile_row_offset =
                  (static_cast<std::size_t>(d) * static_cast<std::size_t>(th) +
                   static_cast<std::size_t>(y)) *
                  static_cast<std::size_t>(tw) * static_cast<std::size_t>(tc) *
                  static_cast<std::size_t>(c.elem_bytes);
              std::uint8_t* dst_row = tile_dst + tile_row_offset;
              const std::size_t src_row_offset = ((((static_cast<std::size_t>(depth_start + d) *
                                                     static_cast<std::size_t>(c.input_height)) +
                                                    static_cast<std::size_t>(y_start + y)) *
                                                       static_cast<std::size_t>(c.input_width) +
                                                   static_cast<std::size_t>(x_start)) *
                                                      static_cast<std::size_t>(c.input_channels) +
                                                  static_cast<std::size_t>(channel_start)) *
                                                 static_cast<std::size_t>(c.elem_bytes);
              const std::uint8_t* src_row = input.data() + src_row_offset;

              for (int x = 0; x < tw; ++x) {
                store_lane_major_pixel_generic(
                    dst_row + static_cast<std::size_t>(x) * static_cast<std::size_t>(tc) *
                                  static_cast<std::size_t>(c.elem_bytes),
                    src_row + static_cast<std::size_t>(x) *
                                  static_cast<std::size_t>(c.input_channels) *
                                  static_cast<std::size_t>(c.elem_bytes),
                    tc, c.elem_bytes);
              }
            }
          }
          x_start += tw;
          ++tile_index;
        }
        y_start += th;
      }
      depth_start += td;
    }
    channel_start += tc;
  }
  return output;
}

std::vector<std::uint8_t> tessellate_host_neon(const TessBenchCase& c,
                                               const std::vector<std::uint8_t>& input) {
  if (!can_use_neon_fast_path(c)) {
    return tessellate_host_generic(c, input);
  }

  const TileGrid grid = build_tile_grid(c);
  std::vector<std::uint8_t> output(grid.aligned_bytes, 0);
  std::size_t tile_index = 0;
  int y_start = 0;
  for (int th : grid.heights) {
    int x_start = 0;
    for (int tw : grid.widths) {
      std::uint8_t* tile_dst = output.data() + grid.tile_offsets[tile_index];
      for (int y = 0; y < th; ++y) {
        const std::size_t src_row_offset =
            (static_cast<std::size_t>(y_start + y) * static_cast<std::size_t>(c.input_width) +
             static_cast<std::size_t>(x_start)) *
            static_cast<std::size_t>(c.input_channels) * static_cast<std::size_t>(c.elem_bytes);
        const std::uint8_t* src_row = input.data() + src_row_offset;
        std::uint8_t* dst_row = tile_dst + static_cast<std::size_t>(y) *
                                               static_cast<std::size_t>(tw) *
                                               static_cast<std::size_t>(c.input_channels) *
                                               static_cast<std::size_t>(c.elem_bytes);
#if defined(__aarch64__) || defined(__ARM_NEON)
        tessellate_neon_tile_row(c, dst_row, src_row, tw);
#else
        (void)c;
        (void)dst_row;
        (void)src_row;
        (void)tw;
#endif
      }
      x_start += tw;
      ++tile_index;
    }
    y_start += th;
  }
  return output;
}

CompareStats compare_outputs(const TessBenchCase& c, const std::vector<std::uint8_t>& lhs,
                             const std::vector<std::uint8_t>& rhs) {
  const TileGrid grid = build_tile_grid(c);
  require(lhs.size() == grid.aligned_bytes, "lhs size mismatch");
  require(rhs.size() == grid.aligned_bytes, "rhs size mismatch");

  CompareStats stats;
  std::size_t tile_index = 0;
  for (int tc : grid.channels) {
    for (int td : grid.depths) {
      for (int th : grid.heights) {
        for (int tw : grid.widths) {
          const std::size_t payload = static_cast<std::size_t>(tc) * static_cast<std::size_t>(td) *
                                      static_cast<std::size_t>(th) * static_cast<std::size_t>(tw) *
                                      static_cast<std::size_t>(c.elem_bytes);
          const std::size_t aligned = grid.tile_sizes[tile_index];
          const std::size_t base = grid.tile_offsets[tile_index];
          stats.payload_bytes += payload;
          stats.padding_bytes += (aligned - payload);
          for (std::size_t i = 0; i < payload; ++i) {
            if (lhs[base + i] != rhs[base + i]) {
              stats.payload_mismatches += 1U;
              stats.exact_payload_match = false;
              stats.exact_full_match = false;
              if (stats.first_diff_absolute == std::numeric_limits<std::size_t>::max()) {
                stats.first_diff_absolute = base + i;
                stats.first_diff_tile = tile_index;
              }
            }
          }
          for (std::size_t i = payload; i < aligned; ++i) {
            if (lhs[base + i] != rhs[base + i]) {
              stats.padding_mismatches += 1U;
              stats.exact_full_match = false;
              if (stats.first_diff_absolute == std::numeric_limits<std::size_t>::max()) {
                stats.first_diff_absolute = base + i;
                stats.first_diff_tile = tile_index;
              }
            }
          }
          ++tile_index;
        }
      }
    }
  }
  return stats;
}

std::uint64_t payload_hash(const TessBenchCase& c, const std::vector<std::uint8_t>& bytes) {
  const TileGrid grid = build_tile_grid(c);
  std::uint64_t h = 1469598103934665603ULL;
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  std::size_t tile_index = 0;
  for (int tc : grid.channels) {
    for (int td : grid.depths) {
      for (int th : grid.heights) {
        for (int tw : grid.widths) {
          const std::size_t payload = static_cast<std::size_t>(tc) * static_cast<std::size_t>(td) *
                                      static_cast<std::size_t>(th) * static_cast<std::size_t>(tw) *
                                      static_cast<std::size_t>(c.elem_bytes);
          const std::size_t base = grid.tile_offsets[tile_index];
          for (std::size_t i = 0; i < payload; ++i) {
            h ^= static_cast<std::uint64_t>(bytes[base + i]);
            h *= kPrime;
          }
          ++tile_index;
        }
      }
    }
  }
  return h;
}

TimingSummary summarize_timings(const std::vector<double>& ms, std::size_t moved_bytes) {
  TimingSummary out;
  if (ms.empty()) {
    return out;
  }
  out.min_ms = *std::min_element(ms.begin(), ms.end());
  out.max_ms = *std::max_element(ms.begin(), ms.end());
  out.avg_ms = std::accumulate(ms.begin(), ms.end(), 0.0) / static_cast<double>(ms.size());
  if (out.avg_ms > 0.0) {
    out.approx_gbps = (static_cast<double>(moved_bytes) / (out.avg_ms / 1000.0)) / 1.0e9;
  }
  return out;
}

std::vector<std::uint8_t> make_input_bytes(const TessBenchCase& c, std::uint32_t seed) {
  const std::size_t elem_count =
      static_cast<std::size_t>(c.input_width) * static_cast<std::size_t>(c.input_height) *
      static_cast<std::size_t>(c.input_depth) * static_cast<std::size_t>(c.input_channels);
  std::vector<std::uint8_t> out(elem_count * static_cast<std::size_t>(c.elem_bytes), 0);

  std::uint32_t state = seed;
  auto next_u32 = [&]() {
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    return state;
  };

  const std::string dtype = upper_copy(c.dtype);
  if (dtype == "BF16" || dtype == "BFLOAT16") {
    auto* ptr = reinterpret_cast<std::uint16_t*>(out.data());
    for (std::size_t i = 0; i < elem_count; ++i) {
      const float value =
          static_cast<float>((next_u32() % 8192U) - 4096U) / 97.0f + static_cast<float>(i % 17U);
      ptr[i] = fp32_to_bf16(value);
    }
    return out;
  }

  if (dtype.find("INT8") != std::string::npos || dtype.find("UINT8") != std::string::npos) {
    for (std::size_t i = 0; i < out.size(); ++i) {
      out[i] = static_cast<std::uint8_t>(next_u32() & 0xFFU);
    }
    return out;
  }

  if (dtype.find("INT32") != std::string::npos || dtype.find("UINT32") != std::string::npos) {
    auto* ptr = reinterpret_cast<std::uint32_t*>(out.data());
    for (std::size_t i = 0; i < elem_count; ++i) {
      ptr[i] = next_u32();
    }
    return out;
  }

  throw std::runtime_error("unsupported dtype for input generation: " + c.dtype);
}

simaai::neat::TensorDType tensor_dtype_for(std::string dtype) {
  dtype = upper_copy(std::move(dtype));
  if (dtype == "BF16" || dtype == "BFLOAT16") {
    return simaai::neat::TensorDType::BFloat16;
  }
  if (dtype.find("INT8") != std::string::npos) {
    return simaai::neat::TensorDType::Int8;
  }
  if (dtype.find("UINT8") != std::string::npos) {
    return simaai::neat::TensorDType::UInt8;
  }
  if (dtype.find("INT32") != std::string::npos) {
    return simaai::neat::TensorDType::Int32;
  }
  throw std::runtime_error("unsupported tensor dtype: " + dtype);
}

std::string input_caps_format_for_dtype(std::string dtype) {
  dtype = upper_copy(std::move(dtype));
  if (dtype == "BF16" || dtype == "BFLOAT16" || dtype == "EVXX_BFLOAT16") {
    return "EVXX_BFLOAT16";
  }
  if (dtype == "FP32" || dtype == "FLOAT32" || dtype == "EVXX_FLOAT32") {
    return "EVXX_FLOAT32";
  }
  if (dtype.find("UINT8") != std::string::npos || dtype.find("INT8") != std::string::npos ||
      dtype == "EVXX_INT8") {
    return "EVXX_INT8";
  }
  if (dtype.find("INT32") != std::string::npos || dtype == "EVXX_INT32") {
    return "EVXX_INT32";
  }
  return dtype;
}

simaai::neat::Tensor make_dense_tensor(const TessBenchCase& c,
                                       const std::vector<std::uint8_t>& bytes) {
  using namespace simaai::neat;

  auto storage = make_cpu_owned_storage(bytes.size());
  auto mapping = storage->map(MapMode::Write);
  require(mapping.data && mapping.size_bytes >= bytes.size(), "failed to allocate input tensor");
  std::memcpy(mapping.data, bytes.data(), bytes.size());

  Tensor tensor;
  tensor.storage = std::move(storage);
  tensor.dtype = tensor_dtype_for(c.dtype);
  tensor.layout = TensorLayout::HWC;
  tensor.shape = {c.input_height, c.input_width, c.input_channels};
  tensor.strides_bytes = {static_cast<std::int64_t>(c.input_width) * c.input_channels *
                              c.elem_bytes,
                          static_cast<std::int64_t>(c.input_channels) * c.elem_bytes, c.elem_bytes};
  tensor.device = {DeviceType::CPU, 0};
  tensor.read_only = true;
  tensor.byte_offset = 0;
  return tensor;
}

std::vector<std::uint8_t> tensor_bytes(const simaai::neat::Tensor& tensor) {
  const auto cpu = tensor.cpu().contiguous();
  return cpu.copy_dense_bytes_tight();
}

namespace pss = simaai::neat::pipeline_internal::sima::plugin_contracts;
namespace pssm = simaai::neat::pipeline_internal::sima::stagesemantics;

void normalize_packed_tess_compiled_contract(simaai::neat::CompiledProcessCvuContract* compiled,
                                             std::uint64_t packed_size_bytes) {
  if (!compiled) {
    throw std::runtime_error("missing compiled tess contract");
  }
  require(packed_size_bytes > 0U, "packed tess contract requires non-zero byte size");
  require(compiled->runtime_contract.logical_outputs.size() == 1U,
          "packed tess benchmark expects exactly one logical output");
  require(compiled->runtime_contract.physical_outputs.size() == 1U,
          "packed tess benchmark expects exactly one physical output");

  auto& logical = compiled->runtime_contract.logical_outputs.front();
  logical.shape = {static_cast<std::int64_t>(packed_size_bytes)};
  logical.stride_bytes = {1};
  logical.size_bytes = packed_size_bytes;
  logical.dtype = "UINT8";
  logical.layout = "HW";
  logical.byte_offset = 0;

  auto& physical = compiled->runtime_contract.physical_outputs.front();
  physical.size_bytes = packed_size_bytes;
}

simaai::neat::TessOptions make_tess_options_for_case(const TessBenchCase& c) {
  pss::CastContractSubset cast_subset;
  cast_subset.input_shape = {1, c.input_height, c.input_width, c.input_channels};
  cast_subset.input_layout = "HWC";
  cast_subset.input_dtype = upper_copy(c.dtype);
  cast_subset.output_dtype = upper_copy(c.dtype);

  pss::TessellateContractSubset tess_subset;
  tess_subset.input_shape = cast_subset.input_shape;
  tess_subset.input_layout = "HWC";
  tess_subset.frame_type = upper_copy(c.dtype);
  tess_subset.slice_shape = {c.tile_height, c.tile_width, c.tile_channels};
  tess_subset.align_c16 = false;
  tess_subset.cblock = false;
  tess_subset.output_size_bytes = build_tile_grid(c).aligned_bytes;

  const auto runtime = pss::build_tessellate_runtime_config_from_subsets(
      cast_subset, tess_subset, "output_tensor", "output_tensor");
  auto compiled = pssm::build_processcvu_compiled_contract_from_runtime_config(runtime);
  normalize_packed_tess_compiled_contract(&compiled, build_tile_grid(c).aligned_bytes);

  simaai::neat::TessOptions opt;
  opt.element_name = "bench_tess_" + c.name;
  opt.compiled_contract =
      std::make_shared<const simaai::neat::CompiledProcessCvuContract>(compiled);
  return opt;
}

std::optional<simaai::neat::RunElementTimingStats>
find_element_timing(const simaai::neat::RunDiagSnapshot& snap, const std::string& name) {
  for (const auto& stat : snap.element_timings) {
    if (stat.element_name == name) {
      return stat;
    }
  }
  return std::nullopt;
}

Ev74RunSummary run_ev74_benchmark(const TessBenchCase& c, const BenchOptions& opt,
                                  const simaai::neat::Tensor& input) {
  using namespace simaai::neat;

  Graph graph;
  InputOptions in;
  in.payload_type = simaai::neat::PayloadType::Tensor;
  in.format = input_caps_format_for_dtype(c.dtype);
  in.width = c.input_width;
  in.height = c.input_height;
  in.depth = c.input_channels;
  in.use_simaai_pool = true;
  in.pool_min_buffers = 2;
  in.pool_max_buffers = 2;
  in.buffer_name = "input_tensor";

  OutputOptions out;
  out.sync = true;
  out.drop = false;
  out.max_buffers = 1;

  graph.add(nodes::Input(in));
  graph.add(nodes::Tess(make_tess_options_for_case(c)));
  graph.add(nodes::Output(out));

  RunOptions run_opt;
  run_opt.output_memory = OutputMemory::Owned;
  run_opt.queue_depth = 1;

  auto warm_run = graph.build_seeded_internal(TensorList{input}, RunMode::Sync, run_opt);
  for (int i = 0; i < opt.warmup; ++i) {
    (void)warm_run.run(TensorList{input}, opt.timeout_ms);
  }
  warm_run.close();

  auto run = graph.build_seeded_internal(TensorList{input}, RunMode::Sync, run_opt);
  std::vector<double> wall_ms;
  wall_ms.reserve(static_cast<std::size_t>(opt.iterations));

  Ev74RunSummary summary;
  summary.payload_hashes.reserve(static_cast<std::size_t>(opt.iterations));
  summary.full_hashes.reserve(static_cast<std::size_t>(opt.iterations));

  for (int i = 0; i < opt.iterations; ++i) {
    const auto t0 = Clock::now();
    TensorList outs = run.run(TensorList{input}, opt.timeout_ms);
    const auto t1 = Clock::now();
    wall_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    require(outs.size() == 1U, "EV74 tess benchmark expected exactly one output tensor");
    std::vector<std::uint8_t> bytes = tensor_bytes(outs.front());
    if (summary.first_output.empty()) {
      summary.first_output = bytes;
    }
    summary.payload_hashes.push_back(payload_hash(c, bytes));
    summary.full_hashes.push_back(fnv1a64(bytes.data(), bytes.size()));
  }

  const auto snap = run_internal::diag_snapshot(run);
  const auto timing = find_element_timing(snap, "bench_tess_" + c.name);
  run.close();

  const TileGrid grid = build_tile_grid(c);
  summary.wall = summarize_timings(wall_ms, grid.payload_bytes * 2U);
  if (timing.has_value() && timing->samples > 0U) {
    summary.element.avg_ms =
        static_cast<double>(timing->total_us) / static_cast<double>(timing->samples) / 1000.0;
    summary.element.min_ms = static_cast<double>(timing->min_us) / 1000.0;
    summary.element.max_ms = static_cast<double>(timing->max_us) / 1000.0;
    if (summary.element.avg_ms > 0.0) {
      summary.element.approx_gbps =
          (static_cast<double>(grid.payload_bytes * 2U) / (summary.element.avg_ms / 1000.0)) /
          1.0e9;
    }
  }
  return summary;
}

TimingSummary time_host_impl(const TessBenchCase& c, const BenchOptions& opt,
                             const std::vector<std::uint8_t>& input, bool use_neon,
                             std::vector<std::uint8_t>* last_output) {
  std::vector<double> samples_ms;
  samples_ms.reserve(static_cast<std::size_t>(opt.iterations));

  for (int i = 0; i < opt.warmup; ++i) {
    (void)(use_neon ? tessellate_host_neon(c, input) : tessellate_host_generic(c, input));
  }

  std::vector<std::uint8_t> last;
  for (int i = 0; i < opt.iterations; ++i) {
    const auto t0 = Clock::now();
    last = use_neon ? tessellate_host_neon(c, input) : tessellate_host_generic(c, input);
    const auto t1 = Clock::now();
    samples_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  if (last_output) {
    *last_output = std::move(last);
  }
  const TileGrid grid = build_tile_grid(c);
  return summarize_timings(samples_ms, grid.payload_bytes * 2U);
}

std::string preview_bytes(const std::vector<std::uint8_t>& bytes, std::size_t count = 16U) {
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  const std::size_t n = std::min(count, bytes.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (i) {
      oss << ' ';
    }
    oss << std::setw(2) << static_cast<unsigned int>(bytes[i]);
  }
  return oss.str();
}

void print_timing_line(std::string_view label, const TimingSummary& summary) {
  std::cout << "  " << label << ": avg_ms=" << std::fixed << std::setprecision(4) << summary.avg_ms
            << " min_ms=" << summary.min_ms << " max_ms=" << summary.max_ms
            << " approx_gbps=" << summary.approx_gbps << "\n";
}

void run_case(const TessBenchCase& c, const BenchOptions& opt) {
  const std::uint64_t name_hash =
      fnv1a64(reinterpret_cast<const std::uint8_t*>(c.name.data()), c.name.size());
  const auto input_bytes =
      make_input_bytes(c, opt.seed ^ static_cast<std::uint32_t>(name_hash & 0xFFFFFFFFU));
  const auto input_tensor = make_dense_tensor(c, input_bytes);
  const TileGrid grid = build_tile_grid(c);

  std::cout << "[case] " << c.name << " input=" << c.input_height << "x" << c.input_width << "x"
            << c.input_channels << " tile=" << c.tile_height << "x" << c.tile_width << "x"
            << c.tile_channels << " dtype=" << c.dtype << " elem_bytes=" << c.elem_bytes
            << " payload_bytes=" << grid.payload_bytes << " aligned_bytes=" << grid.aligned_bytes
            << "\n";

  std::vector<std::uint8_t> generic_out;
  std::vector<std::uint8_t> neon_out;
  const auto host_generic = time_host_impl(c, opt, input_bytes, false, &generic_out);
  const auto host_neon = time_host_impl(c, opt, input_bytes, true, &neon_out);
  const auto host_compare = compare_outputs(c, generic_out, neon_out);
  require(host_compare.exact_full_match, "host generic and host NEON outputs differ for " + c.name);

  const auto ev74 = run_ev74_benchmark(c, opt, input_tensor);
  require(!ev74.first_output.empty(), "EV74 tess benchmark returned no output for " + c.name);
  const auto host_vs_ev74 = compare_outputs(c, generic_out, ev74.first_output);

  const bool ev74_payload_stable =
      std::adjacent_find(ev74.payload_hashes.begin(), ev74.payload_hashes.end(),
                         std::not_equal_to<>()) == ev74.payload_hashes.end();
  const bool ev74_full_stable = std::adjacent_find(ev74.full_hashes.begin(), ev74.full_hashes.end(),
                                                   std::not_equal_to<>()) == ev74.full_hashes.end();

  print_timing_line("host_generic", host_generic);
  print_timing_line("host_neon", host_neon);
  print_timing_line("ev74_wall", ev74.wall);
  print_timing_line("ev74_element", ev74.element);
  std::cout << "  hashes: host_payload=" << hex64(payload_hash(c, generic_out))
            << " host_full=" << hex64(fnv1a64(generic_out.data(), generic_out.size()))
            << " ev74_payload_first=" << hex64(ev74.payload_hashes.front())
            << " ev74_full_first=" << hex64(ev74.full_hashes.front()) << "\n";
  std::cout << "  ev74_stability: payload=" << (ev74_payload_stable ? "stable" : "unstable")
            << " full=" << (ev74_full_stable ? "stable" : "unstable") << "\n";
  std::cout << "  compare_host_vs_ev74: payload_match="
            << (host_vs_ev74.exact_payload_match ? "yes" : "no")
            << " full_match=" << (host_vs_ev74.exact_full_match ? "yes" : "no")
            << " payload_mismatches=" << host_vs_ev74.payload_mismatches
            << " padding_mismatches=" << host_vs_ev74.padding_mismatches << "\n";
  if (!host_vs_ev74.exact_full_match) {
    std::cout << "  first_diff: absolute=" << host_vs_ev74.first_diff_absolute
              << " tile=" << host_vs_ev74.first_diff_tile << "\n";
  }
  std::cout << "  previews: host=" << preview_bytes(generic_out)
            << " ev74=" << preview_bytes(ev74.first_output) << "\n";
}

BenchOptions parse_args(int argc, char** argv) {
  BenchOptions opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char* label) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + label);
      }
      return argv[++i];
    };
    if (arg == "--case") {
      opt.case_name = require_value("--case");
    } else if (arg == "--iterations") {
      opt.iterations = std::stoi(require_value("--iterations"));
    } else if (arg == "--warmup") {
      opt.warmup = std::stoi(require_value("--warmup"));
    } else if (arg == "--timeout-ms") {
      opt.timeout_ms = std::stoi(require_value("--timeout-ms"));
    } else if (arg == "--seed") {
      opt.seed = static_cast<std::uint32_t>(std::stoul(require_value("--seed")));
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: tess_a65_vs_ev74_bench [--case all|evo-bf16-luma|evo-bf16-uv]"
                << " [--iterations N] [--warmup N] [--timeout-ms N] [--seed N]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (opt.iterations <= 0 || opt.warmup < 0 || opt.timeout_ms <= 0) {
    throw std::runtime_error("invalid benchmark arguments");
  }
  return opt;
}

TessBenchCase make_case(std::string name, int input_width, int input_height, int input_channels,
                        int tile_width, int tile_height, int tile_channels, int elem_bytes,
                        std::string dtype, std::string format) {
  TessBenchCase c;
  c.name = std::move(name);
  c.input_width = input_width;
  c.input_height = input_height;
  c.input_channels = input_channels;
  c.tile_width = tile_width;
  c.tile_height = tile_height;
  c.tile_channels = tile_channels;
  c.elem_bytes = elem_bytes;
  c.dtype = std::move(dtype);
  c.format = std::move(format);
  return c;
}

std::vector<TessBenchCase> selected_cases(const BenchOptions& opt) {
  const TessBenchCase evo_luma =
      make_case("evo-bf16-luma", 1664, 384, 1, 17, 384, 1, 2, "BF16", "BF16");
  const TessBenchCase evo_uv = make_case("evo-bf16-uv", 832, 192, 2, 42, 39, 2, 2, "BF16", "BF16");
  if (opt.case_name == "all") {
    return {evo_luma, evo_uv};
  }
  if (opt.case_name == evo_luma.name) {
    return {evo_luma};
  }
  if (opt.case_name == evo_uv.name) {
    return {evo_uv};
  }
  throw std::runtime_error("unknown case: " + opt.case_name);
}

} // namespace

int main(int argc, char** argv) {
  try {
    using namespace simaai::neat;

    require(element_exists("neatprocesscvu"), "missing NEAT CVU plugin (neatprocesscvu)");
    const BenchOptions opt = parse_args(argc, argv);
    for (const auto& c : selected_cases(opt)) {
      run_case(c, opt);
    }
    std::cout << "[OK] tess_a65_vs_ev74_bench completed\n";
    return 0;
  } catch (const SkipTest& e) {
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (is_dispatcher_unavailable(msg)) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "[FAIL] " << msg << "\n";
    return 1;
  }
}

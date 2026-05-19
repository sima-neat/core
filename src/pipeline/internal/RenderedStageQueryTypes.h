#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/TensorTypes.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::stages {

struct TensorDims {
  // Boundary/query projection only. Never authoritative tensor semantics.
  int width = 0;
  int height = 0;
  int depth = 0;
};

enum class PreprocOutputTransportKind : std::uint8_t {
  Unknown = 0,
  Dense = 1,
  Packed = 2,
};

enum class PreprocOutputSemanticKind : std::uint8_t {
  Unknown = 0,
  Image = 1,
  TessellatedImage = 2,
  QuantizedTensor = 3,
  QuantTessTensor = 4,
  Tensor = 5,
};

struct PreprocOutputInfo {
  PreprocOutputTransportKind transport_kind = PreprocOutputTransportKind::Unknown;
  PreprocOutputSemanticKind semantic_kind = PreprocOutputSemanticKind::Unknown;
  std::string output_dtype;
  TensorDims logical_dims;
  TensorLayout logical_layout = TensorLayout::Unknown;
  std::string primary_output_name;
  int primary_route_slot = -1;
  std::vector<std::string> output_memory_order;
};

struct MlaOutputInfo {
  std::string data_type;
  std::string logical_data_type;
  std::vector<int64_t> logical_shape;
  TensorDims dims;
  int64_t size_bytes = 0;
  TensorLayout layout = TensorLayout::Unknown;
  std::string output_format;
};

struct MlaOutputTensorInfo {
  std::vector<int64_t> shape;
  std::vector<int64_t> stride_bytes;
  int64_t byte_offset = 0;
  int memory_index = 0;
  std::string data_type;
  TensorLayout layout = TensorLayout::Unknown;
  std::string output_format;
  std::string name;
  std::string segment_name;
  int64_t size_bytes = 0;
};

struct MlaInputTensorInfo {
  std::vector<int64_t> logical_shape;
  TensorLayout logical_layout = TensorLayout::Unknown;
  std::string logical_dtype;
  std::string logical_format;
  std::string segment_name;
  int64_t span_byte_offset = 0;
  int64_t span_size_bytes = 0;
  std::string media_type;
  std::optional<std::vector<int64_t>> physical_shape;
};

struct MlaInputInfo {
  TensorDims dims;
  TensorLayout layout = TensorLayout::Unknown;
  std::string input_dtype;
  std::string input_format;
  std::string input_media_type;
  int64_t size_bytes = 0;
  TensorDims logical_dims;
  TensorLayout logical_layout = TensorLayout::Unknown;
  std::string logical_input_dtype;
  std::string logical_input_format;
};

struct PackedTensorOutput {
  std::vector<int64_t> shape;
  std::vector<int64_t> stride_bytes;
  int64_t byte_offset = 0;
};

struct PackedTensorOutputInfo {
  std::vector<PackedTensorOutput> outputs;
  TensorDType dtype = TensorDType::Float32;
  TensorLayout layout = TensorLayout::Unknown;
  std::string output_format;
};

using DetessDequantTensorInfo = PackedTensorOutput;
using DetessDequantOutputInfo = PackedTensorOutputInfo;
using DequantOutputInfo = PackedTensorOutputInfo;

} // namespace simaai::neat::stages

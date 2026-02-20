#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "builder/NodeGroup.h"
#include "pipeline/TensorTypes.h"

#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat::stages {

struct TensorDims {
  int width = 0;
  int height = 0;
  int depth = 0;
};

struct PreprocOutputInfo {
  bool tessellate = true;
  std::string output_dtype;
  TensorDims dims;
  std::vector<std::string> output_memory_order;
};

struct MlaOutputInfo {
  std::string data_type;
  TensorDims dims;
  int64_t size_bytes = 0;
  TensorLayout layout = TensorLayout::Unknown;
  std::string output_format;
};

struct MlaInputInfo {
  TensorDims dims;
  TensorLayout layout = TensorLayout::Unknown;
  std::string input_format;
};

struct BoxDecodeInputInfo {
  TensorDims dims;
};

struct BoxDecodeExpectedInfo {
  int64_t buffer_size = 0;
  int64_t total_elems = 0;
  int elem_size = 1;
  int64_t total_bytes = 0;
};

struct DetessDequantTensorInfo {
  std::vector<int64_t> shape;
  int64_t byte_offset = 0;
};

struct DetessDequantOutputInfo {
  std::vector<DetessDequantTensorInfo> outputs;
  TensorDType dtype = TensorDType::Float32;
  TensorLayout layout = TensorLayout::Unknown;
  std::string output_format;
};

PreprocOutputInfo read_preproc_output_info(const NodeGroup& group);
MlaOutputInfo read_mla_output_info(const NodeGroup& group);
MlaInputInfo read_mla_input_info(const NodeGroup& group);
BoxDecodeInputInfo read_boxdecode_input_info(const NodeGroup& group);
BoxDecodeExpectedInfo read_boxdecode_expected_info(const NodeGroup& group);
std::string build_boxdecode_caps_override(const NodeGroup& group);
DetessDequantOutputInfo read_detessdequant_output_info(const NodeGroup& group,
                                                       bool include_batch_axis = true);

} // namespace simaai::neat::stages

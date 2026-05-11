#pragma once

#include <gst/gst.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ProcessMlaOutputDesc {
  std::string name;
  std::size_t size = 0;
  std::size_t source_output_index = 0;
  std::uint64_t source_byte_offset = 0;
};

struct ProcessMlaLogicalOutputDesc {
  int logical_index = -1;
  int backend_output_index = -1;
  int physical_index = -1;
  int route_slot = -1;
  int memory_index = -1;
  int logical_name_id = -1;
  int backend_name_id = -1;
  int segment_name_id = -1;
  std::string logical_name;
  std::string backend_name;
  std::string segment_name;
  std::int64_t byte_offset = 0;
  std::size_t size_bytes = 0;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> stride_bytes;
  std::string dtype;
  std::string layout;
  bool has_quant = false;
  int quant_granularity = 0;
  int quant_axis = -1;
  std::vector<double> quant_scales;
  std::vector<std::int64_t> quant_zero_points;
};

struct ProcessMlaRuntimeConfig {
  std::string stage_key;
  std::string model_path;
  gint32 batch_size = 0;
  gint32 batch_model = 0;
  std::vector<std::string> dispatcher_input_names;
  bool has_input_segment_name = false;
  std::string input_segment_name;
  bool has_input_contract = false;
  std::size_t input_expected_bytes = 0;
  std::size_t input_physical_size_bytes = 0;
  std::uint64_t input_physical_byte_offset = 0;
  std::string input_expected_dtype;
  int input_tensor_index = -1;
  std::vector<ProcessMlaOutputDesc> outputs;
  std::vector<ProcessMlaOutputDesc> dispatcher_outputs;
  std::vector<ProcessMlaLogicalOutputDesc> logical_outputs;
  std::vector<double> output_q_scales;
  std::vector<double> output_q_zero_points;
  bool ofm_unpack_enabled = false;
};

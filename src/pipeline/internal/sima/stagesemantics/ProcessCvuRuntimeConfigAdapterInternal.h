#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"

#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::stagesemantics {

struct CompiledProcessCvuRuntimeConfig {
  std::string graph_family;
  std::string graph_name;
  int graph_id = -1;

  std::string default_input_name;
  std::vector<std::string> runtime_input_names;
  std::vector<std::string> runtime_output_names;
  std::vector<std::string> published_output_names;
  std::vector<std::string> physical_input_names;
  std::vector<std::string> physical_output_names;
  std::string primary_output_name;
  bool single_output_handoff = false;
  ProcessCvuOutputTransportKind primary_output_transport_kind =
      ProcessCvuOutputTransportKind::Unknown;
  ProcessCvuOutputSemanticKind primary_output_semantic_kind = ProcessCvuOutputSemanticKind::Unknown;

  int scaled_width = 0;
  int scaled_height = 0;
  int input_stride = 0;
  int output_stride = 0;
  int input_offset = 0;
  int batch_size = 1;
  int round_off = 0;
  int byte_align = 1;
  std::uint32_t opt_flags = 0;
  int pad_value = 0;

  int aspect_ratio = -1;
  int normalize = -1;
  int tessellate = -1;

  bool has_q_scale = false;
  double q_scale = 0.0;
  bool has_q_zp = false;
  std::int64_t q_zp = 0;

  std::vector<sima_ev_tensor_desc> input_tensors;
  std::vector<sima_ev_tensor_desc> output_tensors;
  std::vector<double> q_scale_list;
  std::vector<std::int64_t> q_zp_list;
  std::vector<double> dq_scale_list;
  std::vector<std::int64_t> dq_zp_list;

  std::vector<std::vector<int>> input_shapes;
  std::vector<std::vector<int>> slice_shapes;
  std::vector<std::vector<int>> output_shapes;
  std::vector<int> runtime_output_logical_index_list;
  std::vector<int> runtime_output_output_slot_list;
  std::vector<int> runtime_output_physical_index_list;
  std::vector<std::string> runtime_output_dtype_list;
  std::vector<ProcessCvuOutputTransportKind> runtime_output_transport_kind_list;
  std::vector<ProcessCvuOutputSemanticKind> runtime_output_semantic_kind_list;
  std::vector<std::vector<int>> runtime_output_logical_shapes;
  std::vector<std::string> runtime_output_logical_layout_list;

  std::vector<float> channel_mean = {0.0f, 0.0f, 0.0f};
  std::vector<float> channel_stddev = {1.0f, 1.0f, 1.0f};

  std::string input_img_type;
  std::string output_img_type;
  std::string input_dtype;
  std::string output_dtype;
  std::string out_dtype;
  std::string scaling_type;
  std::string padding_type;
};

ProcessCvuStagePayload build_processcvu_payload_from_runtime_config_internal(
    const CompiledProcessCvuRuntimeConfig& config);

ProcessCvuStagePayload build_processcvu_payload_from_runtime_config_unchecked_internal(
    const CompiledProcessCvuRuntimeConfig& config);

ProcessCvuCanonicalFacts
build_processcvu_facts_from_runtime_config_internal(const CompiledProcessCvuRuntimeConfig& config);

ProcessCvuCanonicalCompileInputs
build_processcvu_compile_inputs_from_runtime_config(const CompiledProcessCvuRuntimeConfig& config);

CompiledProcessCvuContract build_processcvu_compiled_contract_from_runtime_config(
    const CompiledProcessCvuRuntimeConfig& config);

} // namespace simaai::neat::pipeline_internal::sima::stagesemantics

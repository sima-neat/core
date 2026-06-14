#pragma once

#include <gst/gst.h>

#include <gst/SimaPreparedRuntimeAbi.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {

struct PreparedRuntimeDescriptor {
  std::string session_id;
  std::string model_id;
  std::vector<simaai::gst::PreparedStageSpec> stages;
};

// Small C-compatible ABI probe exported by libneatpreparedruntimebridge.so.
// The prepared-runtime context still carries rich C++ objects, so the caller
// must verify that both DSOs were built against the same layout before passing
// PreparedRuntimeDescriptor across the shared-library boundary.
struct PreparedRuntimeBridgeAbiSizes {
  std::uint32_t abi_version = SIMA_PREPARED_RUNTIME_ABI_VERSION;
  std::uint64_t prepared_stage_spec_size = 0U;
  std::uint64_t processmla_prepared_stage_size = 0U;
  std::uint64_t processcvu_prepared_stage_size = 0U;
  std::uint64_t prepared_processcvu_typed_config_size = 0U;
};

extern "C" const PreparedRuntimeBridgeAbiSizes* sima_neat_prepared_runtime_bridge_abi_sizes();

enum class GraphTensorMaterializationKind : std::uint8_t {
  Unknown = 0,
  Direct = 1,
  OffsetView = 2,
  Bf16LaneSplitRepack = 3,
};

struct GraphTensorContract {
  int tensor_index = -1;
  int physical_index = -1;
  int source_physical_index = -1;
  std::string name;
  std::string segment_name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::uint64_t size_bytes = 0U;
  std::int64_t byte_offset = 0;
  std::int64_t source_byte_offset = 0;
  std::vector<std::int64_t> stride_bytes;
  GraphTensorMaterializationKind materialization_kind = GraphTensorMaterializationKind::Direct;
};

struct GraphQuantContract {
  std::vector<double> scales;
  std::vector<std::int64_t> zero_points;
  int axis = -1;
};

struct GraphProcessMlaStageRequest {
  std::string stage_key;
  std::string model_path;
  gint32 batch_size = 0;
  gint32 batch_model = 0;
  std::vector<GraphTensorContract> dispatcher_inputs;
  std::vector<GraphTensorContract> logical_inputs;
  std::vector<GraphTensorContract> physical_inputs;
  std::vector<GraphTensorContract> stage_outputs;
  std::vector<GraphTensorContract> logical_outputs;
  std::optional<GraphQuantContract> output_quant;
};

struct GraphProcessCvuStageRequest {
  std::string stage_key;
  std::string graph_name;
  std::string requested_run_target = "AUTO";
  std::string run_target = "AUTO";
  std::string resolved_exec_backend = "EVXX";
  std::string run_target_resolution_reason;
  int graph_id = 0;
  // Native visual-frontend scalar fields for EV74 graphs 235-238.
  // Used by Neat graph/processcvu builders when the graph has a native
  // config ABI instead of the generic tensor-pair ABI.
  int width = -1;
  int height = -1;
  int threshold = -1;
  int max_features = -1;
  int grid_x = -1;
  int grid_y = -1;
  int min_px_dist = -1;
  int descriptor_words = -1;
  int num_points = -1;
  int win_half = -1;
  int max_iters = -1;
  int max_level = -1;
  int detect_new_features = -1;
  int fast_threshold = -1;
  int debug = 0;
  int batch_size = 0;
  int round_off = 0;
  int byte_align = 0;
  int aspect_ratio = -1;
  int normalize = -1;
  int tessellate = -1;
  int scaled_width = 0;
  int scaled_height = 0;
  int input_stride = 0;
  int output_stride = 0;
  int input_offset = 0;
  int pad_value = 0;
  std::string input_img_type;
  std::string output_img_type;
  std::string scaling_type;
  std::string padding_type;
  std::string input_dtype;
  std::string output_dtype;
  std::vector<GraphTensorContract> input_tensors;
  std::vector<GraphTensorContract> output_tensors;
  std::vector<std::vector<int>> slice_shapes;
  std::string canonical_input_dtype;
  std::string canonical_output_dtype;
  std::string input_slot_name;
  std::vector<std::string> runtime_output_slot_names;
  std::vector<std::string> runtime_output_logical_layout_list;
  std::vector<float> dq_scale_array;
  std::vector<int32_t> dq_zp_array;
  // True when the tiled side of the op uses c16-padded channel storage
  // (MPK contract sets align_c16 or cblock). Tells the EV tile descriptor
  // builder to clear SIMA_EV_TILED_FLAG_COMPACT_CHANNELS so the kernel
  // strides through the per-row C-padding instead of treating the buffer as
  // dense. Required for e.g. resnet50 (1000 logical → 1008 stored channels).
  bool c16_packed_io = false;
};

bool prepare_processmla_runtime_config(ProcessMlaRuntimeConfig* runtime_cfg,
                                       std::string* error_message = nullptr);

bool build_graph_processmla_prepared_stage(const GraphProcessMlaStageRequest& request,
                                           simaai::gst::ProcessMlaPreparedStage* out,
                                           std::string* error_message = nullptr);

bool build_graph_processcvu_prepared_stage(const GraphProcessCvuStageRequest& request,
                                           simaai::gst::ProcessCvuPreparedStage* out,
                                           std::string* error_message = nullptr);

bool build_prepared_stage_from_manifest_context(const GstContext* static_manifest_context,
                                                const char* stage_id_or_name,
                                                const char* element_name_fallback,
                                                simaai::gst::PreparedStageSpec* out,
                                                std::string* error_message = nullptr);

bool attach_prepared_runtime_context(GstElement* pipeline,
                                     PreparedRuntimeDescriptor prepared_runtime,
                                     std::string* error_message = nullptr);

} // namespace simaai::neat

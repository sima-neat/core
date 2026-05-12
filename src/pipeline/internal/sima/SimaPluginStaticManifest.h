/**
 * @file
 * @ingroup internal_sima
 * @brief **Framework internal — reach-through tier.** In-memory representation of a SiMa
 *        GStreamer plugin's static manifest.
 *
 * The static manifest is the runtime-facing artifact the framework produces from a parsed
 * MPK + planned route. It enumerates every stage with the full set of static specs the SiMa
 * GStreamer plugins need: logical I/O specs, physical buffer specs, output routes, quant params,
 * and per-family payloads (`ProcessCvuStagePayload`, `ProcessMlaStagePayload`,
 * `BoxDecodeStagePayload`, etc.). The header also exposes JSON serialization helpers and a
 * `parse_pipeline_elements` parser for GStreamer launch strings.
 *
 * Heavy header — many small types, all of which downstream code (rendering / runtime bridge /
 * stage-transform passes) reaches into. The reach-through tier owns this header.
 *
 * @see ProcessCvuStagePayload
 * @see StageStaticSpec
 * @see ContractRender.h (the renderer that builds these manifests)
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <gst/gst.h>
#include <nlohmann/json.hpp>

#include "pipeline/BoxDecodeType.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"
#include <ev/ev_tensor_abi.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima {

/// Quantization granularity carried by a quant spec.
enum class QuantGranularity : std::uint8_t {
  PerTensor = 0, ///< One scale / zero-point for the whole tensor.
  PerAxis = 1,   ///< One scale / zero-point per slice along an axis.
};

/// What activation the box-decoder applies to class scores before NMS.
enum class BoxDecodeScoreActivation : std::uint8_t {
  Unknown = 0,
  Identity = 1, ///< Pass-through (scores already activated upstream).
  Sigmoid = 2,  ///< Apply sigmoid to scores.
};

/// Quantization parameters for one tensor.
struct QuantStaticSpec {
  QuantGranularity granularity = QuantGranularity::PerTensor;
  int axis = -1;                         ///< Quant axis (PerAxis only); -1 if PerTensor.
  std::vector<double> scales;            ///< Quant scales (length 1 if PerTensor).
  std::vector<std::int64_t> zero_points; ///< Quant zero-points (length 1 if PerTensor).
};

/// One tensor's shape / dtype / layout / envelope geometry.
struct TensorStaticSpec {
  int tensor_index = -1;           ///< Tensor index inside its parent stage.
  std::vector<std::int64_t> shape; ///< Tensor shape.
  std::string dtype;               ///< Dtype token.
  std::string layout;              ///< Layout token (CHW / HWC / HW).
  int max_w = 0;                   ///< Envelope max width.
  int max_h = 0;                   ///< Envelope max height.
  int max_stride = 0;              ///< Envelope max row stride.
  std::string semantic_tag;        ///< Semantic tag (e.g., `"image"`, `"tensor"`).
};

/// Provenance trace capturing how one resolved field was chosen.
struct ResolutionTrace {
  std::string field;                       ///< Name of the resolved field.
  std::string source_used;                 ///< Source eventually selected.
  std::vector<std::string> fallback_chain; ///< Sources tried in order before `source_used`.
  bool conflict = false;                   ///< True if multiple sources disagreed.
};

/// One logical-output route entry on a stage.
struct StageOutputRoute {
  int output_slot = 0;           ///< Output slot index on the stage.
  int logical_output_index = -1; ///< Logical-output index this route exposes.
  int tensor_index = -1;         ///< Tensor index in the parent contract.
  std::string cm_output_name;    ///< Contract-manifest output name.
  std::string segment_name;      ///< Backing segment name.
};

/// Where a physical buffer lives.
enum class DeviceKind : std::uint8_t {
  Unknown = 0,
  Cpu = 1,  ///< Host (DDR) memory.
  Mla = 2,  ///< MLA-attached memory.
  Evxx = 3, ///< EVxx-attached memory.
};

/// How a tensor's physical bytes materialize from its parent buffer.
enum class TensorMaterializationKind : std::uint8_t {
  Unknown = 0,
  Direct = 1,              ///< Direct buffer (1:1).
  OffsetView = 2,          ///< Byte-offset view into parent.
  Bf16LaneSplitRepack = 3, ///< Requires BF16 lane-split repack at runtime.
};

/// Static spec for one physical buffer (input or output) on a stage.
struct PhysicalBufferStaticSpec {
  int physical_index = -1;                      ///< Stage-local physical index.
  int allocator_index = -1;                     ///< Allocator pool index.
  int source_physical_index = -1;               ///< Source buffer's physical index (when sharing).
  std::uint64_t size_bytes = 0;                 ///< Buffer size.
  std::int64_t source_byte_offset = 0;          ///< Byte offset within the source buffer.
  DeviceKind device_kind = DeviceKind::Unknown; ///< Device the buffer sits on.
  std::uint64_t memory_flags = 0;               ///< Allocator-specific memory flags.
  int segment_name_id = -1; ///< Index into the stage's name table for `segment_name`.
  std::string segment_name; ///< Segment name.
};

/// Static spec for one logical (publishable) output tensor on a stage.
struct LogicalTensorStaticSpec {
  int logical_index = -1;                 ///< Stage-local logical-output index.
  int backend_output_index = -1;          ///< Backend (compiled) output index.
  int physical_index = -1;                ///< Backing physical-output index.
  int output_slot = -1;                   ///< Output slot on the parent stage.
  int tensor_index = -1;                  ///< Tensor index in the contract.
  std::int64_t byte_offset = 0;           ///< Byte offset within the physical buffer.
  std::uint64_t size_bytes = 0;           ///< Tensor byte size.
  std::vector<std::int64_t> shape;        ///< Tensor shape.
  std::vector<std::int64_t> stride_bytes; ///< Per-dim byte strides.
  std::string dtype;                      ///< Dtype token.
  std::string layout;                     ///< Layout token.
  int logical_name_id = -1;               ///< Name-table index for `logical_name`.
  std::string logical_name;               ///< Logical (model-facing) tensor name.
  int backend_name_id = -1;               ///< Name-table index for `backend_name`.
  std::string backend_name;               ///< Backend (compiled artifact) tensor name.
  int segment_name_id = -1;               ///< Name-table index for `segment_name`.
  std::string segment_name;               ///< Backing segment name.
  std::optional<QuantStaticSpec> quant;   ///< Quant params, if quantized.
};

/// Static spec for one logical input tensor on a stage (the consumer-side counterpart).
struct LogicalInputStaticSpec {
  int logical_index = -1;                 ///< Stage-local logical-input index.
  int backend_input_index = -1;           ///< Backend input index.
  int physical_index = -1;                ///< Backing physical-input index.
  std::vector<std::int64_t> shape;        ///< Tensor shape.
  std::vector<std::int64_t> stride_bytes; ///< Per-dim byte strides.
  std::int64_t byte_offset = 0;           ///< Byte offset within the physical buffer.
  std::uint64_t size_bytes = 0;           ///< Tensor byte size.
  std::string dtype;                      ///< Dtype token.
  std::string layout;                     ///< Layout token.
  int logical_name_id = -1;               ///< Name-table index for `logical_name`.
  std::string logical_name;               ///< Logical (model-facing) input name.
  int backend_name_id = -1;               ///< Name-table index for `backend_name`.
  std::string backend_name;               ///< Backend input name.
  int segment_name_id = -1;               ///< Name-table index for `segment_name`.
  std::string segment_name;               ///< Source segment name.
  TensorMaterializationKind materialization_kind =
      TensorMaterializationKind::Direct; ///< Materialization kind.
  std::optional<QuantStaticSpec> quant;  ///< Quant params, if quantized.
};

/// Binding from one of this stage's inputs to a specific upstream output.
struct InputBindingStaticSpec {
  int sink_pad_index = -1;                   ///< Sink-pad index on the consumer.
  int local_logical_input_index = -1;        ///< Consumer's logical-input index this binding fills.
  int src_stage_index = -1;                  ///< Producer stage index (in manifest order).
  std::string src_stage_id;                  ///< Producer stage id.
  int src_logical_output_index = -1;         ///< Producer's logical-output index.
  int src_output_slot = -1;                  ///< Producer's output slot.
  int src_physical_output_index = -1;        ///< Producer's physical-output index.
  std::uint64_t src_physical_size_bytes = 0; ///< Producer physical buffer size.
  std::int64_t src_physical_byte_offset = 0; ///< Byte offset within producer buffer.
  bool required = true;            ///< False = optional binding (e.g., conditional input).
  int cm_input_name_id = -1;       ///< Name-table index for `cm_input_name`.
  std::string cm_input_name;       ///< Contract-manifest input name on the consumer.
  int source_segment_name_id = -1; ///< Name-table index for `source_segment_name`.
  std::string source_segment_name; ///< Producer segment name.
};

/// Discriminator for which payload variant a `StageStaticSpec` carries.
enum class StagePayloadKind : std::uint8_t {
  None = 0,
  ProcessCvu = 1,
  ProcessMla = 2,
  BoxDecode = 3,
  DetessDequant = 4,
  Quant = 5,
  Tess = 6,
  Dequant = 7,
  QuantTess = 8,
};

/// Compiled graph family for a process-CVU stage.
enum class ProcessCvuGraphFamily : std::uint8_t {
  Unknown = 0,
  Preproc = 1,
  Quant = 2,
  Tess = 3,
  QuantTess = 4,
  CastTess = 5,
  Detess = 6,
  Dequant = 7,
  DetessCast = 8,
  DetessDequant = 9,
  Cast = 10,
};

/// Whether a process-CVU output is dense or packed on the wire.
enum class ProcessCvuOutputTransportKind : std::uint8_t {
  Unknown = 0,
  Dense = 1,  ///< Plain row-major / strided buffer.
  Packed = 2, ///< Packed (e.g., tessellated tile-stream) buffer.
};

/// What a process-CVU output represents semantically.
enum class ProcessCvuOutputSemanticKind : std::uint8_t {
  Unknown = 0,
  Image = 1,
  TessellatedImage = 2,
  QuantizedTensor = 3,
  QuantTessTensor = 4,
  Tensor = 5,
};

/**
 * @brief Process-CVU stage payload.
 *
 * Carries the canonical input/output geometry, the graph family, run-target settings, and the
 * full set of compiled per-output runtime descriptors. Members that begin with `runtime_` are
 * parallel arrays indexed by output number.
 */
struct ProcessCvuStagePayload {
  bool canonical_contract = false;
  std::vector<std::int64_t> slice_shape_raw;
  std::vector<std::int64_t> out_shape_raw;
  bool has_align_c16 = false;
  bool align_c16 = false;
  bool has_cblock = false;
  bool cblock = false;

  std::string graph_family;
  ProcessCvuGraphFamily graph_family_enum = ProcessCvuGraphFamily::Unknown;
  std::string graph_name;
  std::string requested_run_target = "AUTO";
  std::string run_target = "AUTO";
  std::string resolved_exec_backend = "EVXX";
  std::string run_target_resolution_reason;
  // When a processcvu stage is synthesized from an MPK route, preserve the
  // exact MPK stage identity that supplied the canonical facts. Runtime graph
  // overlay uses this to select the correct raw/fused graph node for multi-
  // ingress models instead of falling back to the first compatible node.
  std::string exact_stage_name_or_id;
  std::string default_input_name;
  std::vector<std::string> default_output_names;
  std::string primary_output_name;
  ProcessCvuOutputTransportKind primary_output_transport_kind =
      ProcessCvuOutputTransportKind::Unknown;
  ProcessCvuOutputSemanticKind primary_output_semantic_kind = ProcessCvuOutputSemanticKind::Unknown;
  std::string input_img_type;
  std::string output_img_type;
  std::string scaling_type;
  std::string padding_type;
  std::string input_dtype;
  std::string output_dtype;
  std::string out_dtype;

  int scaled_width = 0;
  int scaled_height = 0;
  int input_stride = 0;
  int output_stride = 0;
  int input_offset = 0;
  int batch_size = 0;
  int round_off = 0;
  int byte_align = 0;
  int graph_id = 0;
  std::uint32_t opt_flags = 0;

  int aspect_ratio = -1;
  int normalize = -1;
  int tessellate = -1;
  bool preproc_single_output_handoff = false;

  double q_scale = 0.0;
  int q_zp = 0;
  bool has_q_scale = false;
  bool has_q_zp = false;
  std::vector<double> q_scale_list;
  std::vector<int> q_zp_list;
  int num_in_tensor = 0;
  std::vector<sima_ev_tensor_desc> input_tensors;
  std::vector<sima_ev_tensor_desc> output_tensors;
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
  std::vector<double> dq_scale_list;
  std::vector<int> dq_zp_list;

  std::vector<double> channel_mean;
  std::vector<double> channel_stddev;

  /// Layout token of the stage's `index`-th typed input (falls back to first input).
  std::string typed_input_layout_token(std::size_t index = 0U) const {
    if (index < input_tensors.size()) {
      return tensorsemantics::layout_token_from_ev_tensor_desc(input_tensors[index]);
    }
    if (!input_tensors.empty()) {
      return tensorsemantics::layout_token_from_ev_tensor_desc(input_tensors.front());
    }
    return {};
  }

  /// Layout token of the stage's `index`-th typed output (falls back to first output).
  std::string typed_output_layout_token(std::size_t index = 0U) const {
    if (index < output_tensors.size()) {
      return tensorsemantics::layout_token_from_ev_tensor_desc(output_tensors[index]);
    }
    if (!output_tensors.empty()) {
      return tensorsemantics::layout_token_from_ev_tensor_desc(output_tensors.front());
    }
    return {};
  }

  /// Logical-output layout token (with normalization), falling back to typed output layout.
  std::string logical_output_layout_token(std::size_t index = 0U) const {
    if (index < runtime_output_logical_layout_list.size() &&
        !runtime_output_logical_layout_list[index].empty()) {
      return tensorsemantics::normalize_layout_token(runtime_output_logical_layout_list[index]);
    }
    if (!runtime_output_logical_layout_list.empty() &&
        !runtime_output_logical_layout_list.front().empty()) {
      return tensorsemantics::normalize_layout_token(runtime_output_logical_layout_list.front());
    }
    return typed_output_layout_token(index);
  }
};

/// ProcessMla stage payload — model path + dispatcher fan-out.
struct ProcessMlaStagePayload {
  std::string model_path;                             ///< Path to the MLA model artifact.
  int batch_size = 0;                                 ///< Effective batch size at this stage.
  int batch_sz_model = 0;                             ///< Batch size baked into the model.
  std::vector<std::string> dispatcher_output_names;   ///< Per-dispatcher-output names.
  std::vector<std::uint64_t> dispatcher_output_sizes; ///< Per-dispatcher-output byte sizes.
};

/// BoxDecode stage payload — decode flavor, NMS / topK params, slice geometry.
struct BoxDecodeStagePayload {
  BoxDecodeType decode_type = BoxDecodeType::Unspecified;
  std::optional<BoxDecodeTypeOption> decode_type_option;
  BoxDecodeScoreActivation score_activation = BoxDecodeScoreActivation::Unknown;
  std::string input_dtype;
  bool tess_needed = false;
  bool quant_needed = false;
  bool model_owned_flags = false;
  bool quant_contract_required = false;
  double detection_threshold = 0.0;
  double nms_iou_threshold = 0.0;
  int topk = 0;
  int num_classes = 0;
  std::vector<sima_ev_shape_desc> slice_shapes;
};

/// Placeholder payload for quant / dequant / tess / quanttess stages (carries only a reserved field
/// today).
struct QuantLikeStagePayload {
  int reserved = 0;
};

/**
 * @brief Static spec for one stage in a `SimaPluginStaticManifest`.
 *
 * Aggregates the stage's identity, all I/O specs (logical / physical), the output route table,
 * the active payload variant (selected by `payload_kind`), and a resolution trace recording
 * how each non-trivial field was decided.
 */
struct StageStaticSpec {
  std::string element_name;
  std::string logical_stage_id;
  bool model_managed_stage = false;
  std::string plugin_kind;
  std::string kernel_kind;
  std::vector<std::string> name_table;
  std::vector<LogicalInputStaticSpec> logical_inputs;
  std::vector<InputBindingStaticSpec> input_bindings;
  std::vector<PhysicalBufferStaticSpec> physical_inputs;
  std::vector<PhysicalBufferStaticSpec> physical_outputs;
  std::vector<LogicalTensorStaticSpec> logical_outputs;
  std::vector<StageOutputRoute> output_order;
  std::vector<QuantStaticSpec> output_quant;
  std::vector<std::string> required_preprocess_meta_fields;
  // Mirrors CompiledRuntimeContract::consumer_keeps_distinct_physical_inputs.
  // Plumbed from the upstream MLA contract through ContractRender so that
  // publish-contract construction can stamp TensorBufferPublishContract::
  // preserve_physical_segments when this stage's outputs feed a multi-IFM MLA.
  bool consumer_keeps_distinct_physical_inputs = false;
  // Optional .elf-derived I/O symbol names. See MlaStaticContractExtractor.h.
  // When populated, downstream binding/dispatch can use the actual placeholder
  // symbol names instead of synthesized "ifmN" / "ofmN" strings.
  std::vector<std::string> elf_ifm_symbol_names;
  std::vector<std::string> elf_ofm_symbol_names;
  StagePayloadKind payload_kind = StagePayloadKind::None;
  ProcessCvuStagePayload processcvu;
  ProcessMlaStagePayload processmla;
  BoxDecodeStagePayload boxdecode;
  QuantLikeStagePayload detessdequant;
  QuantLikeStagePayload quant;
  QuantLikeStagePayload tess;
  QuantLikeStagePayload dequant;
  QuantLikeStagePayload quanttess;
  std::vector<ResolutionTrace> resolution_trace;
};

/**
 * @brief Top-level static manifest passed to the SiMa GStreamer plugins.
 *
 * Carries the session identity, the model id, and the ordered list of stage specs.
 */
struct SimaPluginStaticManifest {
  std::string session_id;              ///< Per-run session identifier.
  std::string model_id;                ///< Model identity (typically derived from MPK).
  std::vector<StageStaticSpec> stages; ///< Stages in execution order.
};

/// One element parsed from a GStreamer launch-string fragment, with optional property overrides.
struct PipelineElementSpec {
  std::size_t element_index = 0; ///< Position in the pipeline element list.
  std::string plugin;            ///< Plugin (factory) name.
  std::string element_name;      ///< Element instance name.
  std::string stage_id;          ///< Stage identifier this element corresponds to.
  std::string config_path;       ///< Optional config-JSON path attached to the element.
  std::optional<BoxDecodeType> decode_type_property;  ///< Optional `decode-type` property override.
  std::optional<double> detection_threshold_property; ///< Optional `detection-threshold` override.
  std::optional<double> nms_iou_threshold_property;   ///< Optional `nms-iou-threshold` override.
  std::optional<int> topk_property;                   ///< Optional `topk` override.
  std::optional<std::string> model_path_property;     ///< Optional `model-path` override.
  std::optional<int> batch_size_property;             ///< Optional `batch-size` override.
  std::optional<int> batch_sz_model_property;         ///< Optional `batch-sz-model` override.
  std::string fragment; ///< Original launch-string fragment for this element.
};

/// Diagnostics captured during manifest assembly.
struct ManifestBuildDiagnostics {
  std::vector<std::string> warnings;
  std::vector<std::string> errors;
};

/// Parse a GStreamer launch string into a list of `PipelineElementSpec`s.
std::vector<PipelineElementSpec> parse_pipeline_elements(const std::string& pipeline_string);

/// JSON serialization for a `QuantStaticSpec`.
nlohmann::json to_json(const QuantStaticSpec& spec);
/// JSON serialization for a `TensorStaticSpec`.
nlohmann::json to_json(const TensorStaticSpec& spec);
/// JSON serialization for a `ResolutionTrace`.
nlohmann::json to_json(const ResolutionTrace& trace);
/// JSON serialization for a `StageOutputRoute`.
nlohmann::json to_json(const StageOutputRoute& route);
/// JSON serialization for a `PhysicalBufferStaticSpec`.
nlohmann::json to_json(const PhysicalBufferStaticSpec& spec);
/// JSON serialization for a `LogicalTensorStaticSpec`.
nlohmann::json to_json(const LogicalTensorStaticSpec& spec);
/// JSON serialization for a `LogicalInputStaticSpec`.
nlohmann::json to_json(const LogicalInputStaticSpec& spec);
/// JSON serialization for an `InputBindingStaticSpec`.
nlohmann::json to_json(const InputBindingStaticSpec& spec);
/// JSON serialization for a `StageStaticSpec`.
nlohmann::json to_json(const StageStaticSpec& spec);
/// JSON serialization for a full `SimaPluginStaticManifest`.
nlohmann::json to_json(const SimaPluginStaticManifest& manifest);

/// Serialize a manifest to a JSON string.
std::string serialize_manifest_json(const SimaPluginStaticManifest& manifest);

/// Parse a manifest from a JSON string; returns nullopt on failure (see `error_message`).
std::optional<SimaPluginStaticManifest> parse_manifest_json(const std::string& manifest_json,
                                                            std::string* error_message = nullptr);

/// Attach a manifest as a GstContext to `pipeline` so SiMa elements can retrieve it.
bool attach_manifest_context(GstElement* pipeline, const SimaPluginStaticManifest& manifest,
                             std::string* error_message = nullptr);

} // namespace simaai::neat::pipeline_internal::sima

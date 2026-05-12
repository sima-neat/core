/**
 * @file
 * @ingroup internal_sima
 * @brief **Framework internal — reach-through tier.** Internal representation of an MPK
 *        manifest after the loader gates run.
 *
 * `MpkContract` is the parsed-and-validated form of `mpk.json`. It captures the plugin chain,
 * tensor contracts at each plugin boundary, ingress tensor contracts, raw MPK edges, and the
 * fused `MpkGraph` view used by the route planner. Only the MPK manifest JSON should feed
 * this — see the project-level guidance not to read other JSON files in a model pack.
 *
 * The header also exposes a small set of convenience accessors (e.g.,
 * `get_mla_stage_io_contract`, `mla_consumer_keeps_distinct_physical_inputs`) used heavily by
 * the planner's MLA handoff logic.
 *
 * @see SimaPluginStaticManifest (the rendered, runtime-facing form)
 * @see RouteGraph (the planner's reduced graph view)
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima {

/// Whether an MPK shape describes raw geometry or a packed transport extent.
enum class MpkShapeSemantics {
  Unknown = 0,  ///< Not specified in the MPK.
  Geometry,     ///< Logical geometry (raw rank/dims).
  PackedExtent, ///< Packed-transport byte extent expressed as a shape.
};

/// How an MPK tensor materializes into runtime memory.
enum class MpkTensorMaterializationKind {
  Unknown = 0,
  Direct = 1,              ///< Tensor occupies its own contiguous physical buffer.
  OffsetView = 2,          ///< Tensor is a byte-offset view into a parent buffer.
  Bf16LaneSplitRepack = 3, ///< Tensor requires BF16 lane-split repack at runtime.
};

/// One tensor contract at an MPK plugin boundary.
struct MpkTensorContract {
  int tensor_index = -1;
  int physical_index = -1;
  int source_physical_index = -1;
  std::string name;
  std::string segment_name;
  std::string kind;
  std::string dtype;
  std::vector<std::int64_t> mpk_shape;
  MpkShapeSemantics shape_semantics = MpkShapeSemantics::Unknown;
  std::size_t size_bytes = 0;
  std::int64_t byte_offset = 0;
  std::int64_t source_byte_offset = 0;
  std::vector<std::int64_t> stride_bytes;
  std::vector<std::int64_t> logical_shape;
  std::string logical_dtype;
  std::vector<double> input_range;
  std::string logical_source_plugin;
  std::string logical_source_kernel;
  int logical_source_sequence = -1;
  MpkTensorMaterializationKind materialization_kind = MpkTensorMaterializationKind::Direct;
};

/// Quantization parameters captured from the MPK (per-tensor or per-axis).
struct MpkQuantContract {
  std::vector<double> scales;            ///< Per-channel scales (length 1 if per-tensor).
  std::vector<std::int64_t> zero_points; ///< Per-channel zero-points (length 1 if per-tensor).
  int axis = -1;                         ///< Quantization axis; -1 if per-tensor.
};

/// Reasons MPK contract loading can fail.
enum class MpkContractErrorCode {
  None = 0,
  InvalidPackageRoot,         ///< Package root path is missing or invalid.
  MissingMpkJson,             ///< `mpk.json` was not found in the package root.
  ParseFailed,                ///< JSON parse failure.
  MissingPluginsArray,        ///< `mpk.json` lacks the `plugins` array.
  MissingProducerForTensor,   ///< A tensor consumer references a producer that doesn't exist.
  AmbiguousProducerForTensor, ///< Multiple producers claim the same tensor name.
  InvalidMlaTopology,         ///< MLA stage topology violates expected invariants.
  GraphFillFailed,            ///< Fused graph construction failed.
};

/// One error captured during MPK contract loading.
struct MpkContractError {
  MpkContractErrorCode code = MpkContractErrorCode::None;
  std::string message;
};

/// One raw producer-to-consumer edge in the MPK plugin chain.
struct MpkContractEdge {
  std::size_t src_plugin_index = 0U; ///< Producer plugin index.
  int src_output_index = -1;         ///< Producer output slot.
  std::size_t dst_plugin_index = 0U; ///< Consumer plugin index.
  int dst_input_index = -1;          ///< Consumer input slot.
  std::string src_plugin;            ///< Producer plugin name.
  std::string dst_plugin;            ///< Consumer plugin name.
  std::string tensor_name;           ///< Tensor name on the wire.
};

/// Kind of node in the fused MPK graph.
enum class MpkGraphNodeKind {
  Unknown = 0,
  IngressTensor,      ///< External input tensor (graph source).
  Plugin,             ///< A single MPK plugin.
  FusedPreproc,       ///< Fused preprocess group.
  FusedBoxDecode,     ///< Fused box-decode terminal group.
  FusedQuantTess,     ///< Fused quantize+tessellate group.
  FusedDetessDequant, ///< Fused detess+dequantize group.
};

/// Kind of edge in the fused MPK graph.
enum class MpkGraphEdgeKind {
  Unknown = 0,
  CandidateTensorMatch, ///< Tentative edge based on tensor-name matching.
  FusedRoute,           ///< Confirmed routing edge after fusion.
};

/// Origin (in MPK terms) of a kernel-contract field.
enum class MpkGraphKernelFieldKind {
  Unknown = 0,
  Argument,  ///< Kernel argument from the MPK spec.
  Value,     ///< Inline literal value.
  Parameter, ///< Bound parameter (e.g., from a parent template).
};

/// One field in a kernel contract carried by an MPK graph node.
struct MpkGraphKernelField {
  std::string name;
  std::string value;
  MpkGraphKernelFieldKind kind = MpkGraphKernelFieldKind::Unknown;
  bool known = false;
};

/// Kernel contract attached to an MPK graph node.
struct MpkGraphKernelContract {
  std::string kernel_name;                 ///< Name of the kernel (e.g., `"preproc_v2"`).
  std::string contract_type;               ///< Contract type tag.
  std::vector<MpkGraphKernelField> fields; ///< Field bindings.
};

/// Which fused-stage requirements the route around this node must satisfy.
struct MpkGraphFusionRequirements {
  bool preproc = false;        ///< Requires a preprocess fusion target.
  bool boxdecode = false;      ///< Requires a box-decode fusion target.
  bool quantization = false;   ///< Requires quant.
  bool tessellation = false;   ///< Requires tess.
  bool detessellation = false; ///< Requires detess.
  bool dequantization = false; ///< Requires dequant.
  bool cast = false;           ///< Requires cast.
};

/// MLA unpack metadata: where N output tensors are sliced from a packed MLA OFM.
struct MpkGraphMlaUnpackMetadata {
  bool present = false;                  ///< Set when unpack metadata exists.
  bool explicit_from_mpk = false;        ///< True when MPK specified unpack explicitly.
  std::string source_stage;              ///< Source MLA stage name.
  std::vector<std::int64_t> input_shape; ///< Packed input shape.
  std::vector<std::string> output_names; ///< Per-slice output names.
  std::vector<std::vector<std::int64_t>> output_shapes;       ///< Per-slice shapes.
  std::vector<std::vector<std::int64_t>> output_slice_begins; ///< Per-slice begin offsets.
  std::vector<std::size_t> output_sizes;                      ///< Per-slice byte sizes.
  int output_count = 0;                                       ///< Total number of slices.
};

/// MLA pack metadata (placeholder; current presence flag only).
struct MpkGraphMlaPackMetadata {
  bool present = false;
};

/// All MLA-related metadata captured from the MPK for one MLA stage.
struct MpkGraphMlaMetadata {
  bool present = false;                            ///< Set when MLA metadata exists.
  std::string input_transport_dtype;               ///< Wire-level input dtype.
  std::string output_transport_dtype;              ///< Wire-level output dtype.
  std::vector<std::string> input_semantic_dtypes;  ///< Per-input semantic (model) dtypes.
  std::vector<std::size_t> input_sizes;            ///< Per-input byte sizes.
  MpkGraphMlaPackMetadata pack;                    ///< Pack metadata.
  std::vector<std::string> output_semantic_dtypes; ///< Per-output semantic dtypes.
  MpkGraphMlaUnpackMetadata unpack;                ///< Unpack metadata.
};

/// One node in the fused MPK graph (after `graph_mpk_creation` / `graph_fuser`).
struct MpkGraphNode {
  std::string node_id;
  std::string label;
  std::string name;
  std::string plugin_id;
  std::string processor;
  std::string kernel;
  std::string canonical_op;
  MpkGraphKernelContract kernel_contract;
  std::string tensor_kind;
  std::string dtype;
  std::vector<std::int64_t> mpk_shape;
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::vector<std::string> member_node_ids;
  MpkGraphNodeKind kind = MpkGraphNodeKind::Unknown;
  MpkGraphFusionRequirements requirements;
  std::size_t plugin_index = static_cast<std::size_t>(-1);
  int tensor_index = -1;
  int sequence = -1;
  int branch_count = 0;
  std::size_t size_bytes = 0U;
  bool synthetic = false;
  MpkGraphMlaMetadata mla_metadata;
};

/// One edge in the fused MPK graph.
struct MpkGraphEdge {
  std::string edge_id;                               ///< Stable edge identifier.
  std::string src_node_id;                           ///< Source node id.
  std::string dst_node_id;                           ///< Destination node id.
  std::string tensor_name;                           ///< Tensor flowing along the edge.
  MpkGraphEdgeKind kind = MpkGraphEdgeKind::Unknown; ///< Candidate match vs. confirmed routing.
  std::size_t src_plugin_index =
      static_cast<std::size_t>(-1); ///< Source plugin index (when present).
  std::size_t dst_plugin_index = static_cast<std::size_t>(-1); ///< Destination plugin index.
  int src_tensor_index = -1; ///< Source tensor index within the producer.
  int dst_tensor_index = -1; ///< Destination tensor index within the consumer.
};

/// Full graph view of an MPK: raw plugin/tensor nodes plus the fused, route-ready view.
struct MpkGraph {
  std::string mpk_json_path;           ///< Path the MPK was loaded from.
  std::string model_name;              ///< Model name from the manifest.
  std::string model_path;              ///< Model artifact path.
  std::vector<MpkGraphNode> raw_nodes; ///< Pre-fusion node list.
  std::vector<MpkGraphEdge> raw_edges; ///< Pre-fusion edge list (candidate matches).
  std::vector<MpkGraphNode> nodes;     ///< Post-fusion node list.
  std::vector<MpkGraphEdge> edges;     ///< Post-fusion edge list (confirmed routes).
};

/**
 * @brief One plugin's I/O contract from the MPK.
 *
 * Captures the plugin's identity (name, id, processor, kernel, executable), batch geometry,
 * slice/frame shape, the canonical process-CVU contract decoded from MPK params, and the
 * per-tensor input/output contracts. Quant params are attached when the plugin exposes them.
 */
struct MpkPluginIoContract {
  std::string name;
  std::string plugin_id;
  std::string processor;
  std::string kernel;
  std::string executable;
  int batch_size = 0;
  int batch_sz_model = 0;
  int sequence = -1;
  std::vector<std::int64_t> slice_shape;
  std::vector<std::int64_t> slice_begin;
  std::vector<std::int64_t> slice_end;
  std::vector<std::int64_t> frame_shape;
  std::string frame_type;
  std::string round_off;
  // Canonical processcvu contract decoded from MPK params and tensor contracts.
  // These fields are model-managed source-of-truth and keep raw output rank.
  bool has_canonical_processcvu_contract = false;
  std::vector<std::int64_t> out_shape_raw;
  std::string canonical_input_dtype;
  std::string canonical_output_dtype;
  bool has_align_c16 = false;
  bool align_c16 = false;
  bool has_cblock = false;
  bool cblock = false;
  std::vector<MpkTensorContract> input_tensors;
  std::vector<MpkTensorContract> output_tensors;
  std::optional<MpkQuantContract> quant;
};

/**
 * @brief Top-level parsed-and-validated form of an MPK manifest.
 *
 * Aggregates the manifest-level identity (path, model name, model path), all ingress tensors,
 * the plugin chain with per-plugin I/O contracts, the raw tensor edges, the fused graph, and
 * any errors collected during loading.
 */
struct MpkContract {
  std::string mpk_json_path;                      ///< Source `mpk.json` path.
  std::string model_name;                         ///< Model name from the manifest.
  std::string model_path;                         ///< Model artifact path.
  std::vector<MpkTensorContract> ingress_tensors; ///< External input tensors.
  std::vector<MpkPluginIoContract> plugins;       ///< Plugin chain.
  std::vector<MpkContractEdge> edges;             ///< Raw producer-consumer edges.
  MpkGraph graph;                       ///< Fused graph (filled by `fill_graph_from_mpk`).
  std::vector<MpkContractError> errors; ///< Errors collected during load.
};

/**
 * @brief Load and validate an MPK contract from a model-pack root directory.
 *
 * Reads the MPK manifest JSON only (per project policy), validates the plugin chain, fills
 * tensor contracts, and returns the parsed `MpkContract`.
 *
 * @param package_root   Path to the model-pack root.
 * @param error_message  Optional out-parameter populated on failure.
 * @return Parsed contract on success; `std::nullopt` on failure (see `error_message`).
 */
std::optional<MpkContract> load_mpk_contract_from_pack_root(const std::string& package_root,
                                                            std::string* error_message = nullptr);

/// Initial pass: build the raw graph view from the parsed plugin/tensor list.
void graph_mpk_creation(MpkContract* contract);

/// Second pass: fuse adjacent plugin nodes into composite stages (preproc / quanttess / etc.).
void graph_fuser(MpkContract* contract);

/// Convenience: run both `graph_mpk_creation` and `graph_fuser`.
void fill_graph_from_mpk(MpkContract* contract);

/// Render an `MpkGraph` as a Markdown report (for diagnostics).
std::string render_mpk_graph_markdown(const MpkGraph& graph,
                                      const std::string& title = "MPK Graph");

/// Look up a plugin's I/O contract by name or id; returns null if not found.
const MpkPluginIoContract* get_stage_io_contract(const MpkContract& contract,
                                                 const std::string& plugin_name_or_id);

/// Returns the MLA stage's I/O contract, or null if no MLA stage is present.
const MpkPluginIoContract* get_mla_stage_io_contract(const MpkContract& contract);

/// Returns the MLA-unpack stage's I/O contract, or null if absent.
const MpkPluginIoContract* get_mla_unpack_stage_io_contract(const MpkContract& contract);

// True when the MLA stage in `contract` consumes N>1 input tensors and none of
// those inputs is produced by a stage with canonical_op == "pack". This signals
// a native multi-IFM .elf compilation path (e.g. data.ifm.persistent.input_NN
// placeholders) where downstream coalescers must NOT collapse the cast/quant
// branches into a single packed parent. Returns false for monolithic-IFM models
// (single MLA input) and for models where the MPK includes an explicit pack
// producer (already in the right shape for collapse).
bool mla_consumer_keeps_distinct_physical_inputs(const MpkContract& contract);

/// Returns a pointer to the MLA stage's input tensor contracts, or null if no MLA stage.
const std::vector<MpkTensorContract>* get_mla_input_contract(const MpkContract& contract);

/// Returns the logical-input contracts at the MLA boundary (post-pre-stage rewrites).
std::vector<MpkTensorContract>
get_mla_boundary_logical_inputs_contract(const MpkContract& contract);

/// Returns the physical-input contracts at the MLA boundary.
std::vector<MpkTensorContract>
get_mla_boundary_physical_inputs_contract(const MpkContract& contract);

/// Returns a pointer to the MLA stage's output tensor contracts, or null if no MLA stage.
const std::vector<MpkTensorContract>* get_mla_outputs_contract(const MpkContract& contract);

/// Returns the physical-output contracts at the MLA boundary.
std::vector<MpkTensorContract>
get_mla_boundary_physical_outputs_contract(const MpkContract& contract);

/// Returns the published-output contracts at the MLA boundary (post-unpack/dequant).
std::vector<MpkTensorContract> get_mla_published_outputs_contract(const MpkContract& contract);

/// Returns the logical-output contracts at the MLA boundary.
std::vector<MpkTensorContract> get_mla_logical_outputs_contract(const MpkContract& contract);

/// Returns the quant params for a named plugin if present.
std::optional<MpkQuantContract> get_quant_params_contract(const MpkContract& contract,
                                                          const std::string& plugin_name_or_id);

/// Find a plugin's index by name or id.
std::optional<std::size_t> find_plugin_index_by_name_or_id(const MpkContract& contract,
                                                           const std::string& plugin_name_or_id);

/// Return the plugin indices in execution order (topologically sorted).
std::vector<std::size_t> plugins_in_execution_order(const MpkContract& contract);

} // namespace simaai::neat::pipeline_internal::sima

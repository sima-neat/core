/**
 * @file
 * @ingroup internal_sima
 * @brief **Framework internal — reach-through tier.** Internal route-plan graph.
 *
 * `RouteGraph` is the planner's internal representation of an MPK route after fusion: a node
 * per plugin (each labeled with its canonical `RouteGraphKernelKind`), edges describing tensor
 * flow between plugins, an execution order, and the index of the MLA stage. It is consumed by
 * the contract render and stage-transform passes to position pre/post stages relative to the
 * MLA core.
 *
 * @see MpkContract
 * @see RouteGraphKernelKind
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/MpkContract.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima {

/**
 * @brief Canonical kind of a stage in the route graph.
 *
 * Collapses the many concrete kernel names emitted by upstream tooling into a small fixed set
 * of categories the planner reasons over (Preproc / Quant / Tess / Cast / Detess / Dequant /
 * BoxDecode / etc.).
 */
enum class RouteGraphKernelKind {
  Unknown = 0,
  Preproc,        ///< Preprocess (resize / normalize / layout convert).
  Quant,          ///< Standalone quantize.
  Tess,           ///< Standalone tessellate.
  QuantTess,      ///< Fused quantize + tessellate.
  Cast,           ///< Standalone dtype cast.
  CastTess,       ///< Fused cast + tessellate.
  Detess,         ///< Standalone detessellate.
  DetessCast,     ///< Fused detess + cast.
  DetessDequant,  ///< Fused detess + dequantize.
  Dequantize,     ///< Standalone dequantize.
  BoxDecode,      ///< Postprocess box-decode (YOLO/SSD/etc.).
  Unpack,         ///< MLA-output unpack.
  Slice,          ///< Tensor slice / split.
  PassThrough,    ///< No-op identity stage.
  Mla,            ///< The MLA inference core.
};

/**
 * @brief A single stage in the route graph.
 *
 * Carries the plugin index in the source MPK, identity (name/id/processor/kernel), the canonical
 * kind, the execution sequence number, and whether the stage sits upstream or downstream of the
 * MLA in the route.
 */
struct RouteGraphNode {
  std::size_t plugin_index = 0U;  ///< Index into `MpkContract::plugins`.
  std::string plugin_name;        ///< MPK plugin display name.
  std::string plugin_id;          ///< MPK plugin id.
  std::string processor;          ///< Processor token (e.g., `"cvu"`, `"mla"`).
  std::string kernel;             ///< Raw kernel name from the MPK.
  RouteGraphKernelKind kind = RouteGraphKernelKind::Unknown;  ///< Canonical kind.
  int sequence = -1;       ///< Execution sequence number; -1 if unspecified.
  bool before_mla = false; ///< True if this stage runs upstream of the MLA core.
  bool after_mla = false;  ///< True if this stage runs downstream of the MLA core.
};

/**
 * @brief A directed tensor edge between two route-graph nodes.
 *
 * Carries source/destination plugin indices and slot indices, plus the tensor name(s) at each
 * end (separate `src_tensor_name`/`dst_tensor_name` are kept in case fusion renamed them).
 */
struct RouteGraphEdge {
  std::size_t src_plugin_index = 0U;  ///< Source plugin index.
  int src_output_index = -1;          ///< Source plugin's output slot.
  std::size_t dst_plugin_index = 0U;  ///< Destination plugin index.
  int dst_input_index = -1;           ///< Destination plugin's input slot.
  std::string src_plugin;             ///< Source plugin name.
  std::string dst_plugin;             ///< Destination plugin name.
  std::string tensor_name;            ///< Common tensor name (when source and dest agree).
  std::string src_tensor_name;        ///< Tensor name as seen at the source.
  std::string dst_tensor_name;        ///< Tensor name as seen at the destination.
};

/**
 * @brief Full route graph for one model.
 *
 * Contains nodes, edges, a topologically-ordered execution sequence, and the position of the
 * MLA stage in the plugin list (or -1 if no MLA present).
 */
struct RouteGraph {
  std::string model_name;                       ///< Model name (from MPK).
  std::vector<RouteGraphNode> nodes;            ///< All stages.
  std::vector<RouteGraphEdge> edges;            ///< Tensor edges between stages.
  std::vector<std::size_t> execution_order;     ///< Plugin indices in execution order.
  int mla_plugin_index = -1;                    ///< Index of the MLA stage; -1 if none.
};

/// Map a raw kernel-name string to its canonical `RouteGraphKernelKind`.
RouteGraphKernelKind canonical_route_graph_kernel_kind(const std::string& raw_kernel);

/// Stable string token for a `RouteGraphKernelKind` (for diagnostics).
const char* route_graph_kernel_name(RouteGraphKernelKind kind);

/// Build a `RouteGraph` from a parsed MPK contract.
RouteGraph build_route_graph(const MpkContract& contract);

/// Returns the node at `plugin_index` (by plugin-index lookup), or null if missing.
const RouteGraphNode* route_graph_node(const RouteGraph& graph, std::size_t plugin_index);

/// Returns all edges whose destination is `plugin_index`.
std::vector<const RouteGraphEdge*> route_graph_incoming_edges(const RouteGraph& graph,
                                                              std::size_t plugin_index);

/// Returns all edges whose source is `plugin_index`.
std::vector<const RouteGraphEdge*> route_graph_outgoing_edges(const RouteGraph& graph,
                                                              std::size_t plugin_index);

/// Returns the rank of `plugin_index` in `execution_order`, or `nullopt` if not present.
std::optional<std::size_t> route_graph_execution_rank(const RouteGraph& graph,
                                                      std::size_t plugin_index);

} // namespace simaai::neat::pipeline_internal::sima

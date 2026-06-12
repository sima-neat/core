#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/Model.h"
#include "model/internal/ModelPack.h"

#include "pipeline/internal/RenderedStageQueryTypes.h"
#include "pipeline/internal/sima/RouteGraph.h"

#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::internal {

struct PreprocessPlannerResult;

enum class PreRouteStageKind {
  None = 0,
  Preproc = 1,
  Quant = 2,
  Tess = 3,
  QuantTess = 4,
  Cast = 5,
  CastTess = 6,
  Unknown = 7,
};

enum class PostRouteStageKind {
  None = 0,
  Detess = 1,
  DetessDequant = 2,
  Dequantize = 3,
  BoxDecode = 4,
  Cast = 5,
  DetessCast = 6,
  Unknown = 7,
};

struct OrderedRouteOp {
  enum class Kind {
    Unknown = 0,
    Preproc = 1,
    Quant = 2,
    Tess = 3,
    QuantTess = 4,
    Cast = 5,
    CastTess = 6,
    Detess = 7,
    DetessCast = 8,
    DetessDequant = 9,
    Dequantize = 10,
    BoxDecode = 11,
    Unpack = 12,
  };

  Kind kind = Kind::Unknown;
  std::string plugin_name;
  std::string plugin_id;
  std::string kernel;
  int sequence = -1;
  bool before_mla = false;
  bool after_mla = false;
};

enum class SessionPreStageOp {
  Preproc = 0,
  Quant = 1,
  Tess = 2,
  QuantTess = 3,
  Cast = 4,
  CastTess = 5,
};

enum class SessionPreAdapterKind {
  None = 0,
  Quant = 1,
  Tess = 2,
  QuantTess = 3,
  CastTess = 4,
};

enum class SessionPostStageOp {
  Detess = 0,
  DetessDequant = 1,
  Dequantize = 2,
  BoxDecode = 3,
  Cast = 4,
  DetessCast = 5,
};

enum class SessionPostAdapterKind {
  None = 0,
  Detess = 1,
  DetessDequant = 2,
  Dequant = 3,
  BoxDecode = 4,
  Cast = 5,
  DetessCast = 6,
};

enum class TessellationLocation { Unknown = 0, External = 1, MLA = 2 };

struct IngressTensorContract {
  bool valid = false;
  int ingress_index = -1;
  std::string media_type;
  std::string dtype;
  std::string layout; // routing/boundary projection only
  int rank = 0;
  int batch = 1;
  int width = 0;  // routing/boundary projection only
  int height = 0; // routing/boundary projection only
  int depth = 0;  // routing/boundary projection only
  // Canonical tensor shape from the MPK contract (e.g. [10,1,1024,2] for a
  // batched MLA input). This is the source of truth for tensor-domain
  // consumers. The batch/height/width/depth scalars above are derived
  // convenience for image-domain consumers and may not faithfully represent
  // ranks > 4 or non-image axis labellings — read logical_shape instead when
  // building TensorSpec or pushable Tensor shapes.
  std::vector<std::int64_t> logical_shape;
  std::string source_tensor_name;
  std::string source_stage;
  std::size_t dst_plugin_index = 0U;
  int dst_input_index = -1;
  std::optional<std::size_t> join_plugin_index;
  std::vector<OrderedRouteOp> branch_ops;
};

struct EgressTensorContract {
  bool valid = false;
  std::string media_type;
  std::string dtype;
  std::string layout; // routing/boundary projection only
  int rank = 0;
  int batch = 1;
  int width = 0;  // routing/boundary projection only
  int height = 0; // routing/boundary projection only
  int depth = 0;  // routing/boundary projection only
  // Canonical tensor shape from the MPK contract. Source of truth for
  // tensor-domain consumers (see comment on IngressTensorContract).
  std::vector<std::int64_t> logical_shape;
  std::string source_tensor_name;
  std::string source_stage;
};

struct RouteTensorBinding {
  std::size_t src_plugin_index = 0U;
  std::size_t dst_plugin_index = 0U;
  std::string src_tensor_name;
  std::string dst_tensor_name;
  int logical_index = -1;
  int physical_index = -1;
  std::string segment_name;
};

enum class RouteRegionKind {
  Linear = 0,
  FanoutMap = 1,
  FaninJoin = 2,
  BoxDecodeTerminal = 3,
};

struct RouteRegion {
  RouteRegionKind kind = RouteRegionKind::Linear;
  pipeline_internal::sima::RouteGraphKernelKind op_kind =
      pipeline_internal::sima::RouteGraphKernelKind::Unknown;
  std::size_t producer_plugin_index = 0U;
  std::vector<std::size_t> member_plugin_indices;
  std::optional<std::size_t> join_plugin_index;
  std::vector<RouteTensorBinding> inputs;
  std::vector<RouteTensorBinding> outputs;
  std::vector<EgressTensorContract> egress_contracts;
};

struct RouteIntrinsicNeeds {
  bool pre_quantization = false;
  bool pre_tessellation = false;
  bool pre_cast = false;
  bool post_detessellation = false;
  bool post_dequantization = false;
  bool post_cast = false;
};

struct RouteAdapterCapabilities {
  bool has_pre_quantization = false;
  bool has_pre_tessellation = false;
  bool has_pre_cast = false;
  bool has_post_detessellation = false;
  bool has_post_dequantization = false;
  bool has_post_cast = false;
  bool has_post_boxdecode = false;
};

struct RouteUserIntent {
  bool pre_auto = true;
  bool post_auto = true;
  bool requested_boxdecode = false;
};

struct RouteEffectiveRoute {
  PipelineType pipeline_type = PipelineType::Preproc;
  bool include_preprocess_stage = true;
  bool include_postprocess_stage = true;
  PostRouteStageKind selected_post_kind = PostRouteStageKind::None;
  bool cast_symmetry_ok = true;
  bool infer_only = false;
  bool mla_tessellation = false;
};

struct RouteCapability {
  PreRouteStageKind pre_kind = PreRouteStageKind::None;
  PostRouteStageKind post_kind = PostRouteStageKind::None;

  // Canonical post-route needs derived from MPK topology + tensor dtypes.
  bool tess_needed = false;
  bool quant_needed = false;

  bool has_external_pre = false;
  bool has_external_post = false;
  bool has_external_tess = false;
  bool has_external_pre_cast = false;
  bool has_external_detess = false;
  bool has_external_dequant = false;
  bool has_external_post_cast = false;
  bool has_external_boxdecode = false;
  bool has_strict_boxdecode_route = false;

  TessellationLocation tessellation_location = TessellationLocation::Unknown;

  std::string mla_input_dtype_raw;
  std::string mla_output_dtype_raw;
  std::string mla_input_media_type;
  stages::TensorDims mla_input_dims;
  stages::TensorDims mla_output_dims;
  bool mla_input_bf16 = false;
  bool mla_input_quantized = false;
  bool mla_output_bf16 = false;
  bool mla_output_quantized = false;

  RouteIntrinsicNeeds needs;
  RouteAdapterCapabilities adapter_capabilities;

  std::vector<IngressTensorContract> ingress_contracts;
  std::vector<RouteRegion> ingress_regions;
  EgressTensorContract egress_contract;
  std::vector<EgressTensorContract> egress_contracts;
  std::vector<OrderedRouteOp> ordered_pre_ops;
  std::vector<OrderedRouteOp> ordered_post_ops;

  std::vector<std::string> evidence;
};

struct RouteSelection {
  RouteUserIntent user_intent;
  RouteEffectiveRoute effective;

  std::string modelpack_media_type;
  std::string modelpack_format;
  int modelpack_input_depth = 0;
  int modelpack_max_width = 0;
  int modelpack_max_height = 0;
  int modelpack_max_depth = 0;

  bool ambiguous = false;
  std::string ambiguity_reason;
  std::vector<std::string> diagnostics;
};

struct ModelSemantics {
  bool quant_needed = false;
  bool tess_needed = false;
  bool pre_quant_needed = false;
  bool pre_tess_needed = false;
  bool has_pre_adapter = false;
  bool has_post_adapter = false;
  bool has_post_boxdecode = false;
  bool has_pre_quant_adapter = false;
  bool has_pre_tess_adapter = false;
  bool has_pre_quanttess_adapter = false;
  bool has_post_dequant_adapter = false;
  bool has_post_detess_adapter = false;
  bool has_post_cast_adapter = false;
  bool pre_cast_needed = false;
  bool post_cast_needed = false;
  bool cast_symmetry_ok = true;
  std::string mla_input_dtype_raw;
  std::string mla_output_dtype_raw;
  stages::TensorDims mla_input_dims;
  stages::TensorDims mla_output_dims;
  std::size_t output_physical_count = 0U;
  std::size_t output_logical_count = 0U;
};

struct RouteMaterializationPlan {
  struct ModelManagedRouteFlags {
    bool quant_needed = false;
    bool tess_needed = false;
    bool pre_cast_needed = false;
    bool quant_contract_required = false;
    bool include_pre_stage = false;
    bool boxdecode_selected = false;
  };

  bool use_preproc = false;
  std::vector<RouteRegion> pre_regions;
  std::vector<RouteRegion> post_regions;
  bool boxdecode_selected = false;
  bool pre_cast_fp32_to_bf16 = false;
  bool post_cast_bf16_to_fp32 = false;
  struct PreprocContext {
    bool quant_needed = false;
    bool tess_needed = false;
    bool pre_quant_needed = false;
    bool pre_tess_needed = false;
    bool pre_cast_needed = false;
  } preproc_context;
  bool include_pre_stage = false;
  bool include_post_stage = false;
  SessionPreAdapterKind pre_adapter = SessionPreAdapterKind::None;
  std::vector<SessionPreStageOp> pre_chain;
  std::vector<SessionPostStageOp> post_chain;
  SessionPostAdapterKind post_adapter = SessionPostAdapterKind::None;
  ModelManagedRouteFlags model_managed_route_flags;
  std::vector<IngressTensorContract> ingress_contracts;
  std::vector<RouteRegion> ingress_regions;
  EgressTensorContract egress_contract;
  std::vector<EgressTensorContract> egress_contracts;
  std::size_t output_physical_count = 0U;
  std::size_t output_logical_count = 0U;
  PipelineType pipeline_type = PipelineType::Preproc;
  PostRouteStageKind selected_post_kind = PostRouteStageKind::None;
  bool infer_only = false;
  bool cast_symmetry_ok = true;
  std::vector<std::string> diagnostics;
};

using SessionRoutePlan = RouteMaterializationPlan;

ModelSemantics build_model_semantics(const ModelPack& pack);
RouteMaterializationPlan build_route_plan(const Model::Options& options,
                                          const ModelSemantics& semantics,
                                          const RouteCapability* capability = nullptr,
                                          const ModelPack* pack = nullptr);

RouteCapability extract_route_capability(const ModelPack& pack,
                                         const PreprocessPlannerResult& preprocess_plan);
RouteSelection plan_route_selection(const Model::Options& options,
                                    const PreprocessPlannerResult& preprocess_plan,
                                    const RouteCapability& capability);

std::string route_capability_debug_string(const RouteCapability& capability);
std::string route_selection_debug_string(const RouteSelection& selection);

} // namespace simaai::neat::internal

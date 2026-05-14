/**
 * @file
 * @brief Internal helpers for Model (not part of the public API).
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/Model.h"
#include "model/internal/RoutePlanner.h"
#include "nodes/io/Input.h"

namespace simaai::neat {
class NodeGroup;
class Session;
struct CompiledBoxDecodeContract;
struct PreprocOptions;
struct QuantOptions;
struct TessOptions;
struct QuantTessOptions;
struct CastTessOptions;
struct DetessOptions;
struct DetessCastOptions;
struct DetessDequantOptions;
} // namespace simaai::neat

namespace simaai::neat::pipeline_internal::sima {
struct ModelManagedRouteFlags;
} // namespace simaai::neat::pipeline_internal::sima

namespace simaai::neat::internal {

class ModelPack;

struct PreprocessContractFlags {
  bool quant_needed = false;
  bool tess_needed = false;
};

struct IngressConsumerTensorIdentity {
  int logical_index = -1;
  int physical_index = -1;
  int route_slot = -1;
  int memory_index = -1;
};

enum class StageNodeKind {
  Preproc = 0,
  Quant = 1,
  Tess = 2,
  QuantTess = 3,
  CastTess = 4,
  Detess = 5,
  DetessCast = 6,
  DetessDequant = 7,
  Dequant = 8,
  BoxDecode = 9,
};

struct ModelAccess {
  static const class ModelPack& pack(const Model& model);
  static const class ModelPack& pack_for_sync(const Model& model);
  static std::string model_id(const Model& model);
  static std::string source_path(const Model& model);
  static Model::Options options(const Model& model);
  static Model clone_with_options(const Model& model, const Model::Options& opt);
  static PostRouteStageKind resolved_post_kind(const Model& model);
  static PreprocessContractFlags preprocess_contract_flags(const Model& model);
  static pipeline_internal::sima::ModelManagedRouteFlags
  model_managed_route_flags(const Model& model);
  static bool has_model_managed_stage(const Model& model, StageNodeKind kind);
  static void require_model_managed_stage(const Model& model, StageNodeKind kind,
                                          const char* caller);
  static PreprocOptions build_preprocess_stage_options(const Model& model, bool sync);
  static QuantOptions build_quant_stage_options(const Model& model, bool sync);
  static TessOptions build_tess_stage_options(const Model& model, bool sync);
  static QuantTessOptions build_quanttess_stage_options(const Model& model, bool sync);
  static CastTessOptions build_casttess_stage_options(const Model& model, bool sync);
  static DetessOptions build_detess_stage_options(const Model& model, bool sync);
  static DetessCastOptions build_detesscast_stage_options(const Model& model, bool sync);
  static DetessDequantOptions build_detessdequant_stage_options(const Model& model, bool sync);
  static CompiledBoxDecodeContract build_boxdecode_stage_contract(const Model& model, bool sync);
  static simaai::neat::NodeGroup build_preprocess_group(const Model& model, bool sync);
  static simaai::neat::NodeGroup
  build_preprocess_group_for_input(const Model& model, const InputOptions& input, bool sync);
  static simaai::neat::NodeGroup build_infer_group(const Model& model, bool sync);
  static simaai::neat::NodeGroup build_postprocess_group(const Model& model, bool sync);
  static void configure_session_input_route(simaai::neat::Session& session, const Model& model,
                                            const Model::SessionOptions& opt);
};

Tensor remap_tensor_to_consumer_identity(Tensor tensor,
                                         const IngressConsumerTensorIdentity& identity);
Sample remap_sample_to_consumer_identity(Sample sample,
                                         const IngressConsumerTensorIdentity& identity);
TensorList materialize_joined_ingress_tensors(
    const TensorList& tensors,
    const std::vector<IngressConsumerTensorIdentity>& consumer_identities,
    const std::string& segment_name, const char* where);

} // namespace simaai::neat::internal

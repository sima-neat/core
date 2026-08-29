#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "contracts/NodeContractDefinition.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/PluginCompiledContracts.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"

#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::plugin_contracts {
struct BoxDecodeContractSubset;
} // namespace simaai::neat::pipeline_internal::sima::plugin_contracts

namespace simaai::neat::pipeline_internal::sima::stagesemantics {

struct BoxDecodeCompiledContractOptions {
  BoxDecodeType decode_type = BoxDecodeType::Unspecified;
  std::optional<BoxDecodeTypeOption> decode_type_option;
  BoxDecodeScoreActivation score_activation = BoxDecodeScoreActivation::Unknown;
  double detection_threshold = 0.0;
  double nms_iou_threshold = 0.0;
  int topk = 0;
  int num_classes = 0;
  /// Explicit SuperPoint overlay. Unset preserves all MPK subset fields verbatim.
  std::optional<SuperPointStaticContract> superpoint;
  bool model_owned_flags = false;
  std::optional<bool> quant_contract_required;
  std::vector<std::string> required_preprocess_meta_fields;
};

/// Resolve a grouped DFL tensor layout without conflating layout with the
/// class-score domain. Explicit probability/logit options win; otherwise the
/// already inferred score activation selects the matching grouped option.
void resolve_grouped_yolo_dfl_score_domain(BoxDecodeStaticContract* contract);

// SSD class count is fixed by the confidence-head depth. The resolved profile decides whether an
// explicit value must match exactly or may select a contiguous prefix. No-op for non-SSD.
void validate_ssd_num_classes(BoxDecodeType decode_type, SsdRecipeId recipe_id, int requested,
                              int encoded, const char* context);

/// Return the confidence-head encoded depth for SSD, even if a prior contract already selected a
/// narrower prefix. Non-SSD and incomplete legacy contracts use fallback_num_classes.
int ssd_encoded_num_classes(BoxDecodeType decode_type, const SsdClassSelection& selection,
                            int fallback_num_classes);

struct SsdModelFrame {
  int width = 0;
  int height = 0;
};

// Recipe-required SSD frame for this contract's exact ordered head geometry. Returns {0,0} for
// non-SSD and throws for an unsupported SSD signature.
SsdModelFrame ssd_expected_model_frame(const BoxDecodeStaticContract& contract);

// Apply SSD defaults (recipe-specific activation, grouped-by-role layout, class-count) to a
// model-managed contract before lowering. No-op for non-SSD decode types.
void apply_ssd_model_managed_contract_defaults(BoxDecodeStaticContract* contract);

// Apply the raw P3/P4/P5 layout, sigmoid score domain, and inferred class count
// before lowering a model-managed YOLOv5 contract.
void apply_yolov5_model_managed_contract_defaults(BoxDecodeStaticContract* contract);

/// Resolve an explicit class-count override against an inferred decoder contract. YOLO26 family
/// layouts reject contradictory positive values; other families preserve their existing override
/// behavior. A non-positive override selects the inferred value.
int resolve_boxdecode_num_classes_override(BoxDecodeType decode_type, int inferred_num_classes,
                                           int requested_num_classes, const char* context);

BoxDecodeStaticContract finalize_boxdecode_static_contract(
    const BoxDecodeStaticContract& contract, BoxDecodeType decode_type,
    const std::optional<ModelBoxdecodeSemantics>& model_semantics,
    const std::optional<ModelManagedRouteFlags>& model_route_flags,
    BoxDecodeTypeOption decode_type_option, double detection_threshold, double nms_iou_threshold,
    int topk, int num_classes, const std::vector<std::string>& required_preprocess_meta_fields);

CompiledBoxDecodeContract build_boxdecode_compiled_contract_from_subset(
    const plugin_contracts::BoxDecodeContractSubset& subset,
    const BoxDecodeCompiledContractOptions& options = {});

CompiledBoxDecodeContract
build_boxdecode_compiled_contract(const BoxDecodeStaticContract& contract);

bool build_boxdecode_node_contract(const std::string& node_kind, const std::string& plugin_kind,
                                   const std::string& element_name,
                                   const std::string& logical_stage_id,
                                   const NodeContractDefinition& definition,
                                   const CompiledBoxDecodeContract& compiled,
                                   CompiledNodeContract* out, std::string* error_message = nullptr);

} // namespace simaai::neat::pipeline_internal::sima::stagesemantics

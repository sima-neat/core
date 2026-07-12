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
  bool model_owned_flags = false;
  std::optional<bool> quant_contract_required;
  std::vector<std::string> required_preprocess_meta_fields;
};

/// Resolve a grouped DFL tensor layout without conflating layout with the
/// class-score domain. Explicit probability/logit options win; otherwise the
/// already inferred score activation selects the matching grouped option.
void resolve_grouped_yolo_dfl_score_domain(BoxDecodeStaticContract* contract);

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

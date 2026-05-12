#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "contracts/NodeContractDefinition.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/PluginCompiledContracts.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"

#include <optional>

namespace simaai::neat::pipeline_internal::sima::stagesemantics {

struct DequantCanonicalFacts {
  std::string input_name = "input_tensor";
  std::string output_name = "output_tensor";
  std::string input_dtype = "INT8";
  std::string output_dtype = "FLOAT32";
  std::string layout;
  ProcessCvuOutputRepresentation output_representation =
      ProcessCvuOutputRepresentation::DenseTensor;
  std::optional<QuantStaticSpec> input_quant;
};

CompiledDequantContract
build_dequant_compiled_contract_from_facts(const DequantCanonicalFacts& facts);
CompiledDequantContract
build_dequant_compiled_contract_from_upstream(const CompiledRuntimeContract& upstream,
                                              const DequantCanonicalFacts& facts, std::string* err);

bool build_dequant_node_contract(const std::string& node_kind, const std::string& plugin_kind,
                                 const std::string& element_name,
                                 const std::string& logical_stage_id,
                                 const NodeContractDefinition& definition,
                                 const CompiledDequantContract& compiled, CompiledNodeContract* out,
                                 std::string* error_message = nullptr);

} // namespace simaai::neat::pipeline_internal::sima::stagesemantics

#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "contracts/NodeContractDefinition.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/PluginCompiledContracts.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/MlaStaticContractExtractor.h"

#include <nlohmann/json.hpp>

namespace simaai::neat::pipeline_internal::sima::stagesemantics {

CompiledMlaContract build_mla_compiled_contract(const MlaStaticContract& contract);

CompiledMlaContract
build_mla_compiled_contract_from_subset(const plugin_contracts::ProcessMlaContractSubset& subset,
                                        const MlaStaticContract& contract);

bool build_mla_node_contract(const std::string& node_kind, const std::string& element_name,
                             const std::string& logical_stage_id,
                             const NodeContractDefinition& definition,
                             const CompiledMlaContract& compiled, CompiledNodeContract* out,
                             std::string* error_message = nullptr);

} // namespace simaai::neat::pipeline_internal::sima::stagesemantics

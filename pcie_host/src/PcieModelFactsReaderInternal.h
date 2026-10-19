#pragma once

#include "PcieModelFactsReader.h"

#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/static_contract/ModelExecutionPlan.h"

#include <vector>

namespace simaai::neat::pcie::internal::detail {

std::vector<pipeline_internal::sima::MpkTensorContract>
application_input_contracts(const pipeline_internal::sima::MpkContract& contract);

void validate_supported_input_dtype(const pipeline_internal::sima::MpkTensorContract& input);

PcieModelFacts read_mla_only_facts(const pipeline_internal::sima::MpkContract& contract);

// Manifest-only contract derivation; archive loading remains the public entry path.
PcieModelFacts read_model_facts(const pipeline_internal::sima::MpkContract& contract,
                                const ModelOptions& options = {});

// Rejects public inputs/outputs that differ from the strict plan in count, name,
// order, or size. When every public output is one MLA port of a single stage,
// the outputs take that stage's executable port order, which the card uses for
// carriers.
void apply_execution_plan(const pipeline_internal::sima::static_contract::ModelExecutionPlan& plan,
                          PcieModelFacts* facts);

} // namespace simaai::neat::pcie::internal::detail

#pragma once

#include "PcieModelFactsReader.h"

#include "pipeline/internal/sima/MpkContract.h"

#include <vector>

namespace simaai::neat::pcie::internal::detail {

std::vector<pipeline_internal::sima::MpkTensorContract>
application_input_contracts(const pipeline_internal::sima::MpkContract& contract);

void validate_supported_input_dtype(const pipeline_internal::sima::MpkTensorContract& input);

PcieModelFacts read_mla_only_facts(const pipeline_internal::sima::MpkContract& contract);

// Manifest-only contract derivation; archive loading remains the public entry path.
PcieModelFacts read_model_facts(const pipeline_internal::sima::MpkContract& contract,
                                const ModelOptions& options = {});

} // namespace simaai::neat::pcie::internal::detail

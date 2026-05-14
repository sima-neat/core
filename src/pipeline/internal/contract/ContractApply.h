#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "builder/Node.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

void apply_compiled_contracts(std::vector<std::shared_ptr<Node>>* nodes,
                              const CompiledPipelineContracts& compiled,
                              std::string* error_message);

} // namespace simaai::neat

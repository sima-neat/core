#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "builder/Node.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include <memory>
#include <vector>

namespace simaai::neat {

CompiledPipelineContracts
compile_node_contracts(const std::vector<std::shared_ptr<Node>>& nodes,
                       const ContractCompileInput& input,
                       pipeline_internal::sima::ManifestBuildDiagnostics* diagnostics);

} // namespace simaai::neat

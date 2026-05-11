#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "contracts/NodeContractDefinition.h"
#include "pipeline/internal/contract/PluginCompiledContracts.h"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {

struct CompiledNodeContract {
  std::string node_kind;
  std::string plugin_kind;
  std::string element_name;
  std::string logical_stage_id;
  NodeContractDefinition definition;
  std::optional<CompiledProcessCvuContract> processcvu;
  std::optional<CompiledMlaContract> processmla;
  std::optional<CompiledBoxDecodeContract> boxdecode;
  std::optional<CompiledDequantContract> dequant;
  std::optional<CompiledTransportContract> transport;
  std::vector<CompiledNodeContract> child_stages;
  bool renderable = false;
};

struct CompiledPipelineContracts {
  std::vector<CompiledNodeContract> stages;
  bool fully_renderable = false;
};

} // namespace simaai::neat

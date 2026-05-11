#include "pipeline/internal/contract/ContractApply.h"

#include "builder/NodeContractConfigurable.h"

namespace simaai::neat {

void apply_compiled_contracts(std::vector<std::shared_ptr<Node>>* nodes,
                              const CompiledPipelineContracts& compiled,
                              std::string* error_message) {
  if (error_message) {
    error_message->clear();
  }
  if (!nodes) {
    if (error_message) {
      *error_message = "contract apply: nodes is null";
    }
    return;
  }

  std::size_t stage_index = 0U;
  for (auto& node : *nodes) {
    if (!node || node->kind() == "Input" || node->kind() == "Output") {
      continue;
    }
    if (stage_index >= compiled.stages.size()) {
      break;
    }
    auto* configurable = dynamic_cast<NodeContractConfigurable*>(node.get());
    if (!configurable) {
      ++stage_index;
      continue;
    }
    std::string err;
    configurable->apply_compiled_contract(compiled.stages[stage_index], &err);
    if (!err.empty() && error_message && error_message->empty()) {
      *error_message = err;
    }
    ++stage_index;
  }
}

} // namespace simaai::neat

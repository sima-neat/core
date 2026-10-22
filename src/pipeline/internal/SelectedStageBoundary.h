#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/internal/ModelPack.h"
#include "pipeline/internal/RenderedMlaContractQuery.h"

#include <optional>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

namespace simaai::neat::pipeline_internal {

// Only a true ingress MLA can constrain the application boundary by its IFM.
// A fragment may instead begin with a native transform. Inspect the exact
// selected schedule without compiling/preparing any of its executable stages.
inline std::optional<stages::MlaInputTensorInfo>
direct_model_mla_input_info(const std::vector<std::shared_ptr<Node>>& nodes) {
  std::size_t first = 0U;
  while (first < nodes.size() && (!nodes[first] || nodes[first]->kind() == "Input")) {
    ++first;
  }
  if (first == nodes.size() || nodes[first]->kind() != "ModelFragment") {
    return std::nullopt;
  }
  std::string error;
  const auto selected = internal::selected_model_fragment_stage_facts(nodes, &error);
  if (!selected) {
    throw std::runtime_error("Model stage boundary projection failed: " + error);
  }
  const auto facts = selected->stage_facts[first];
  if (facts.empty() || !facts.front().mla_compiled) {
    return std::nullopt;
  }
  return rendered_stage_query::mla_input_tensor_info_from_compiled_contract(
      *facts.front().mla_compiled);
}

} // namespace simaai::neat::pipeline_internal

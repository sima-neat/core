#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "contracts/NodeContractDefinition.h"

#include <span>
#include <string>

namespace simaai::neat::nodes_internal::kernel_requests {

/// `neatkernelrunner name=n<index>_<suffix>`, after checking the element exists.
std::string kernel_runner_fragment(int node_index, const std::string& suffix, const char* node);
/// `n<index>_<suffix>`.
std::string kernel_runner_element_name(int node_index, const std::string& suffix);

/// Describe tensor ports without resolving input storage or preparing execution.
NodeContractDefinition kernel_contract_definition(const std::string& node_kind,
                                                  std::span<const char* const> inputs,
                                                  std::span<const char* const> outputs);

} // namespace simaai::neat::nodes_internal::kernel_requests

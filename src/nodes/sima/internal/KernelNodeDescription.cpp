#include "nodes/sima/internal/KernelNodeDescription.h"

#include "gst/GstHelpers.h"

#include <utility>

namespace simaai::neat::nodes_internal::kernel_requests {

std::string kernel_runner_element_name(int node_index, const std::string& suffix) {
  return "n" + std::to_string(node_index) + "_" + suffix;
}

std::string kernel_runner_fragment(int node_index, const std::string& suffix, const char* node) {
  const auto context = std::string(node) + "::backend_fragment";
  require_element("neatkernelrunner", context.c_str());
  return "neatkernelrunner name=" + kernel_runner_element_name(node_index, suffix);
}

NodeContractDefinition kernel_contract_definition(const std::string& node_kind,
                                                  std::span<const char* const> inputs,
                                                  std::span<const char* const> outputs) {
  NodeContractDefinition result;
  result.node_kind = node_kind;
  result.plugin_kind = "kernel";
  for (const auto* id : inputs) {
    ContractPortSpec port;
    port.port_id = id;
    port.media_type = "application/vnd.simaai.tensor";
    result.inputs.push_back(std::move(port));
  }
  for (const auto* id : outputs) {
    ContractPortSpec port;
    port.port_id = id;
    port.media_type = "application/vnd.simaai.tensor";
    result.outputs.push_back(std::move(port));
  }
  return result;
}

} // namespace simaai::neat::nodes_internal::kernel_requests

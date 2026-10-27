#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// Shared contract compilation for Nodes that run exactly one named
// kernels-library request on a neatkernelrunner stage. The Node supplies the
// request builder; this helper resolves the upstream tensors (a compiled
// producer stage or the graph's ingress sample), forwards the request to the
// Scheduler's canonical check_kernel(), and authors the stage's runtime
// contract from the checked inspection.

#include "nodes/sima/internal/LidarBevKernelRequests.h"

#include <functional>
#include <span>
#include <string>

namespace simaai::neat {
struct ContractCompileInput;
struct CompiledNodeContract;
struct NodeContractDefinition;
} // namespace simaai::neat

namespace simaai::neat::nodes_internal::kernel_requests {

using RequestBuilder = std::function<KernelRequestPlan(std::span<const TensorFact>)>;

/// Compile one single-member kernel stage. Returns false with `error` set on any
/// structural, admission or inspection failure; there is no fallback target.
bool compile_kernel_request_stage(const std::string& node_kind, const std::string& element_name,
                                  const NodeContractDefinition& definition,
                                  const ContractCompileInput& input, const RequestBuilder& build,
                                  CompiledNodeContract* output, std::string* error);

/// `neatkernelrunner name=n<index>_<suffix>`, after checking the element exists.
std::string kernel_runner_fragment(int node_index, const std::string& suffix, const char* node);
/// `n<index>_<suffix>`.
std::string kernel_runner_element_name(int node_index, const std::string& suffix);

/// Tensor-port contract definition with the given input and output port ids.
NodeContractDefinition kernel_contract_definition(const std::string& node_kind,
                                                  std::span<const char* const> inputs,
                                                  std::span<const char* const> outputs);

} // namespace simaai::neat::nodes_internal::kernel_requests

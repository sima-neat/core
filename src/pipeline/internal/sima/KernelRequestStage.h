#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// Shared contract compilation for Nodes that run exactly one named
// kernels-library request on a neatkernelrunner stage. The Node supplies the
// request description; this pipeline helper resolves the upstream tensors (a compiled
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

namespace simaai::neat::pipeline_internal::sima {

using RequestBuilder = std::function<nodes_internal::kernel_requests::KernelRequestPlan(
    std::span<const nodes_internal::kernel_requests::TensorFact>)>;

/// Compile one single-member kernel stage. Returns false with `error` set on any
/// structural, admission or inspection failure; there is no fallback target.
bool compile_kernel_request_stage(const std::string& node_kind, const std::string& element_name,
                                  const NodeContractDefinition& definition,
                                  const ContractCompileInput& input, const RequestBuilder& build,
                                  CompiledNodeContract* output, std::string* error);

} // namespace simaai::neat::pipeline_internal::sima

// Preserve the NodeContractProvider hooks while keeping cold compilation in
// the pipeline layer. Nodes supply immutable operation/options descriptions;
// only this layer resolves carriers, admits requests and constructs recipes.
#include "pipeline/internal/sima/KernelRequestStage.h"

#include "pipeline/internal/contract/ContractFacts.h"

namespace simaai::neat {
namespace kr = nodes_internal::kernel_requests;
namespace si = pipeline_internal::sima;

bool SimaVoxelize::compile_node_contract(const ContractCompileInput& input,
                                         CompiledNodeContract* out, std::string* error) const {
  const auto build = [this](std::span<const kr::TensorFact> facts) {
    return kr::voxelize_request(opt_, facts);
  };
  return si::compile_kernel_request_stage(kind(), element_names(input.node_index).front(),
                                          contract_definition(), input, build, out, error);
}

bool SimaPillarScatter::compile_node_contract(const ContractCompileInput& input,
                                              CompiledNodeContract* out, std::string* error) const {
  const auto build = [this](std::span<const kr::TensorFact> facts) {
    return kr::pillar_scatter_request(opt_, facts);
  };
  return si::compile_kernel_request_stage(kind(), element_names(input.node_index).front(),
                                          contract_definition(), input, build, out, error);
}

bool SimaBEVPoolPlan::compile_node_contract(const ContractCompileInput& input,
                                            CompiledNodeContract* out, std::string* error) const {
  const auto build = [this](std::span<const kr::TensorFact> facts) {
    return kr::bev_pool_plan_request(opt_, facts);
  };
  return si::compile_kernel_request_stage(kind(), element_names(input.node_index).front(),
                                          contract_definition(), input, build, out, error);
}

bool SimaBEVPool::compile_node_contract(const ContractCompileInput& input,
                                        CompiledNodeContract* out, std::string* error) const {
  const auto build = [this](std::span<const kr::TensorFact> facts) {
    return kr::bev_pool_request(opt_, facts);
  };
  return si::compile_kernel_request_stage(kind(), element_names(input.node_index).front(),
                                          contract_definition(), input, build, out, error);
}

} // namespace simaai::neat

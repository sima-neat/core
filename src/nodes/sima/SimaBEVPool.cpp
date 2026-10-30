#include "nodes/sima/SimaBEVPool.h"

#include "nodes/sima/internal/KernelNodeDescription.h"
#include "nodes/sima/internal/LidarBevKernelRequests.h"

#include <array>
#include <utility>

namespace simaai::neat {
namespace {
namespace kr = nodes_internal::kernel_requests;
constexpr std::array<const char*, 3> kInputs{"depth", "context", "plan"};
constexpr std::array<const char*, 1> kOutputs{"bev"};
} // namespace

SimaBEVPool::SimaBEVPool(SimaBEVPoolOptions opt) : opt_(std::move(opt)) {
  kr::validate(opt_);
}

std::string SimaBEVPool::backend_fragment(int node_index) const {
  return kr::kernel_runner_fragment(node_index, "bev_pool", "SimaBEVPool");
}

std::vector<std::string> SimaBEVPool::element_names(int node_index) const {
  return {kr::kernel_runner_element_name(node_index, "bev_pool")};
}

OutputSpec SimaBEVPool::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = "BEV_POOL";
  out.memory = input.memory;
  out.certainty = SpecCertainty::Hint;
  out.note = "neatkernelrunner";
  return out;
}

NodeContractDefinition SimaBEVPool::contract_definition() const {
  return kr::kernel_contract_definition(kind(), kInputs, kOutputs);
}

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> SimaBEVPool(SimaBEVPoolOptions opt) {
  return std::make_shared<simaai::neat::SimaBEVPool>(std::move(opt));
}
} // namespace simaai::neat::nodes

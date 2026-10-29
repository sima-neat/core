#include "nodes/sima/SimaBEVPoolPlan.h"

#include "nodes/sima/internal/KernelNodeDescription.h"
#include "nodes/sima/internal/LidarBevKernelRequests.h"

#include <array>
#include <utility>

namespace simaai::neat {
namespace {
namespace kr = nodes_internal::kernel_requests;
constexpr std::array<const char*, 4> kInputs{"camera2lidar", "camera_intrinsics", "img_aug_matrix",
                                             "lidar_aug_matrix"};
constexpr std::array<const char*, 1> kOutputs{"plan"};
} // namespace

SimaBEVPoolPlan::SimaBEVPoolPlan(SimaBEVPoolPlanOptions opt) : opt_(std::move(opt)) {
  kr::validate(opt_);
}

std::string SimaBEVPoolPlan::backend_fragment(int node_index) const {
  return kr::kernel_runner_fragment(node_index, "bev_pool_plan", "SimaBEVPoolPlan");
}

std::vector<std::string> SimaBEVPoolPlan::element_names(int node_index) const {
  return {kr::kernel_runner_element_name(node_index, "bev_pool_plan")};
}

OutputSpec SimaBEVPoolPlan::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = "BEV_POOL_PLAN";
  out.memory = input.memory;
  out.certainty = SpecCertainty::Hint;
  out.note = "neatkernelrunner";
  return out;
}

NodeContractDefinition SimaBEVPoolPlan::contract_definition() const {
  return kr::kernel_contract_definition(kind(), kInputs, kOutputs);
}

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> SimaBEVPoolPlan(SimaBEVPoolPlanOptions opt) {
  return std::make_shared<simaai::neat::SimaBEVPoolPlan>(std::move(opt));
}
} // namespace simaai::neat::nodes

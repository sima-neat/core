#include "nodes/sima/SimaVoxelize.h"

#include "nodes/sima/internal/KernelNodeDescription.h"
#include "nodes/sima/internal/LidarBevKernelRequests.h"

#include <array>
#include <utility>

namespace simaai::neat {
namespace {
namespace kr = nodes_internal::kernel_requests;
constexpr std::array<const char*, 2> kInputs{"points", "point_count"};
constexpr std::array<const char*, 3> kOutputs{"decorated", "coordinates", "count"};
} // namespace

SimaVoxelize::SimaVoxelize(SimaVoxelizeOptions opt) : opt_(std::move(opt)) {
  kr::validate(opt_);
}

std::string SimaVoxelize::backend_fragment(int node_index) const {
  return kr::kernel_runner_fragment(node_index, "voxelize", "SimaVoxelize");
}

std::vector<std::string> SimaVoxelize::element_names(int node_index) const {
  return {kr::kernel_runner_element_name(node_index, "voxelize")};
}

OutputSpec SimaVoxelize::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = "VOXELIZE";
  out.memory = input.memory;
  out.certainty = SpecCertainty::Hint;
  out.note = "neatkernelrunner";
  return out;
}

NodeContractDefinition SimaVoxelize::contract_definition() const {
  return kr::kernel_contract_definition(kind(), kInputs, kOutputs);
}

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> SimaVoxelize(SimaVoxelizeOptions opt) {
  return std::make_shared<simaai::neat::SimaVoxelize>(std::move(opt));
}
} // namespace simaai::neat::nodes

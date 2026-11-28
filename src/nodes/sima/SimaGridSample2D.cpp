#include "nodes/sima/SimaGridSample2D.h"

#include "nodes/sima/internal/KernelNodeDescription.h"
#include "nodes/sima/internal/LidarBevKernelRequests.h"

#include <array>
#include <utility>

namespace simaai::neat {
namespace {
namespace kr = nodes_internal::kernel_requests;
constexpr std::array<const char*, 2> kInputs{"features", "grid"};
constexpr std::array<const char*, 1> kOutputs{"sampled"};
} // namespace

SimaGridSample2D::SimaGridSample2D(SimaGridSample2DOptions opt) : opt_(std::move(opt)) {
  kr::validate(opt_);
}

std::string SimaGridSample2D::backend_fragment(int node_index) const {
  return kr::kernel_runner_fragment(node_index, "grid_sample_2d", "SimaGridSample2D");
}

std::vector<std::string> SimaGridSample2D::element_names(int node_index) const {
  return {kr::kernel_runner_element_name(node_index, "grid_sample_2d")};
}

OutputSpec SimaGridSample2D::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = "GRID_SAMPLE_2D";
  out.memory = input.memory;
  out.certainty = SpecCertainty::Hint;
  out.note = "neatkernelrunner";
  return out;
}

NodeContractDefinition SimaGridSample2D::contract_definition() const {
  return kr::kernel_contract_definition(kind(), kInputs, kOutputs);
}

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> SimaGridSample2D(SimaGridSample2DOptions opt) {
  return std::make_shared<simaai::neat::SimaGridSample2D>(std::move(opt));
}
} // namespace simaai::neat::nodes

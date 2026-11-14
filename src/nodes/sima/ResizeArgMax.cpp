#include "nodes/sima/ResizeArgMax.h"

#include "nodes/sima/internal/KernelNodeDescription.h"
#include "nodes/sima/internal/LidarBevKernelRequests.h"

#include <array>
#include <utility>

namespace simaai::neat {
namespace {
namespace kr = nodes_internal::kernel_requests;
constexpr std::array<const char*, 1> kInputs{"scores"};
constexpr std::array<const char*, 1> kOutputs{"class_map"};
} // namespace

ResizeArgMax::ResizeArgMax(ResizeArgMaxOptions opt) : opt_(std::move(opt)) {
  kr::validate(opt_);
}

std::string ResizeArgMax::backend_fragment(int node_index) const {
  return kr::kernel_runner_fragment(node_index, "resize_argmax", "ResizeArgMax");
}

std::vector<std::string> ResizeArgMax::element_names(int node_index) const {
  return {kr::kernel_runner_element_name(node_index, "resize_argmax")};
}

OutputSpec ResizeArgMax::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = "RESIZE_ARGMAX";
  out.memory = input.memory;
  out.certainty = SpecCertainty::Hint;
  out.note = "neatkernelrunner";
  return out;
}

NodeContractDefinition ResizeArgMax::contract_definition() const {
  return kr::kernel_contract_definition(kind(), kInputs, kOutputs);
}

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> ResizeArgMax(ResizeArgMaxOptions opt) {
  return std::make_shared<simaai::neat::ResizeArgMax>(std::move(opt));
}
} // namespace simaai::neat::nodes

#include "nodes/sima/SimaResizeArgmax.h"

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

SimaResizeArgmax::SimaResizeArgmax(SimaResizeArgmaxOptions opt) : opt_(std::move(opt)) {
  kr::validate(opt_);
}

std::string SimaResizeArgmax::backend_fragment(int node_index) const {
  return kr::kernel_runner_fragment(node_index, "resize_argmax", "SimaResizeArgmax");
}

std::vector<std::string> SimaResizeArgmax::element_names(int node_index) const {
  return {kr::kernel_runner_element_name(node_index, "resize_argmax")};
}

OutputSpec SimaResizeArgmax::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = "RESIZE_ARGMAX";
  out.memory = input.memory;
  out.certainty = SpecCertainty::Hint;
  out.note = "neatkernelrunner";
  return out;
}

NodeContractDefinition SimaResizeArgmax::contract_definition() const {
  return kr::kernel_contract_definition(kind(), kInputs, kOutputs);
}

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> SimaResizeArgmax(SimaResizeArgmaxOptions opt) {
  return std::make_shared<simaai::neat::SimaResizeArgmax>(std::move(opt));
}
} // namespace simaai::neat::nodes

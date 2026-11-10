#include "nodes/sima/SimaCenterHeadDecode.h"

#include "nodes/sima/internal/KernelNodeDescription.h"
#include "nodes/sima/internal/LidarBevKernelRequests.h"

#include <array>
#include <utility>

namespace simaai::neat {
namespace {
namespace kr = nodes_internal::kernel_requests;
constexpr std::array<const char*, 1> kInputs{"heads"};
constexpr std::array<const char*, 4> kOutputs{"boxes", "scores", "labels", "count"};
} // namespace

SimaCenterHeadDecode::SimaCenterHeadDecode(SimaCenterHeadDecodeOptions opt) : opt_(std::move(opt)) {
  kr::validate(opt_);
}

std::string SimaCenterHeadDecode::backend_fragment(int node_index) const {
  return kr::kernel_runner_fragment(node_index, "centerhead_decode", "SimaCenterHeadDecode");
}

std::vector<std::string> SimaCenterHeadDecode::element_names(int node_index) const {
  return {kr::kernel_runner_element_name(node_index, "centerhead_decode")};
}

OutputSpec SimaCenterHeadDecode::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = "CENTERHEAD_DECODE";
  out.memory = input.memory;
  out.certainty = SpecCertainty::Hint;
  out.note = "neatkernelrunner";
  return out;
}

NodeContractDefinition SimaCenterHeadDecode::contract_definition() const {
  return kr::kernel_contract_definition(kind(), kInputs, kOutputs);
}

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> SimaCenterHeadDecode(SimaCenterHeadDecodeOptions opt) {
  return std::make_shared<simaai::neat::SimaCenterHeadDecode>(std::move(opt));
}
} // namespace simaai::neat::nodes

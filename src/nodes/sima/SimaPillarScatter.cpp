#include "nodes/sima/SimaPillarScatter.h"

#include "nodes/sima/internal/KernelRequestStage.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/ContractFacts.h"

#include <array>
#include <utility>

namespace simaai::neat {
namespace {
namespace kr = nodes_internal::kernel_requests;
constexpr std::array<const char*, 3> kInputs{"features", "coordinates", "count"};
constexpr std::array<const char*, 1> kOutputs{"canvas"};
} // namespace

SimaPillarScatter::SimaPillarScatter(SimaPillarScatterOptions opt) : opt_(std::move(opt)) {
  kr::validate(opt_);
}

std::string SimaPillarScatter::backend_fragment(int node_index) const {
  return kr::kernel_runner_fragment(node_index, "pillar_scatter", "SimaPillarScatter");
}

std::vector<std::string> SimaPillarScatter::element_names(int node_index) const {
  return {kr::kernel_runner_element_name(node_index, "pillar_scatter")};
}

OutputSpec SimaPillarScatter::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = "PILLAR_SCATTER";
  out.memory = input.memory;
  out.certainty = SpecCertainty::Hint;
  out.note = "neatkernelrunner";
  return out;
}

NodeContractDefinition SimaPillarScatter::contract_definition() const {
  return kr::kernel_contract_definition(kind(), kInputs, kOutputs);
}

bool SimaPillarScatter::compile_node_contract(const ContractCompileInput& input,
                                              CompiledNodeContract* out, std::string* error) const {
  const auto build = [this](std::span<const kr::TensorFact> facts) {
    return kr::pillar_scatter_request(opt_, facts);
  };
  return kr::compile_kernel_request_stage(kind(), element_names(input.node_index).front(),
                                          contract_definition(), input, build, out, error);
}

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> SimaPillarScatter(SimaPillarScatterOptions opt) {
  return std::make_shared<simaai::neat::SimaPillarScatter>(std::move(opt));
}
} // namespace simaai::neat::nodes

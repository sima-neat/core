#include "nodes/sima/Kernel.h"

#include "nodes/sima/internal/KernelNodeDescription.h"
#include "nodes/sima/internal/LidarBevKernelRequests.h"

#include <simaai/scheduler/client/kernel.h>

#include <stdexcept>
#include <utility>
#include <variant>

namespace simaai::neat {
namespace {
namespace kr = nodes_internal::kernel_requests;
namespace rt = ::simaai::scheduler::client;
constexpr const char* kElementSuffix = "kernel";
} // namespace

Kernel::Kernel(KernelOptions opt) : opt_(std::move(opt)) {
  kr::validate(opt_);
}

std::string Kernel::backend_fragment(int node_index) const {
  return kr::kernel_runner_fragment(node_index, kElementSuffix, "Kernel");
}

std::vector<std::string> Kernel::element_names(int node_index) const {
  return {kr::kernel_runner_element_name(node_index, kElementSuffix)};
}

OutputSpec Kernel::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = "KERNEL";
  out.memory = input.memory;
  out.certainty = SpecCertainty::Hint;
  out.note = "neatkernelrunner";
  return out;
}

NodeContractDefinition Kernel::contract_definition() const {
  return kr::kernel_contract_definition(kind(), {}, {});
}

std::vector<std::string> load_kernel_library(const std::string& path) {
  auto loaded = rt::load_kernel_library(path);
  if (const auto* failure = std::get_if<rt::Error>(&loaded))
    throw std::runtime_error("load_kernel_library: " + failure->message);
  return std::get<std::vector<std::string>>(std::move(loaded));
}

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> Kernel(KernelOptions opt) {
  return std::make_shared<simaai::neat::Kernel>(std::move(opt));
}
} // namespace simaai::neat::nodes

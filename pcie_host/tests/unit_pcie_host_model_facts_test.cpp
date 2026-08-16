#include "PcieModelFactsReaderInternal.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pcie_internal = simaai::neat::pcie::internal;
namespace mpk = simaai::neat::pipeline_internal::sima;

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

mpk::MpkTensorContract tensor(std::string name, std::string dtype, std::vector<std::int64_t> shape,
                              const std::size_t size_bytes) {
  mpk::MpkTensorContract out;
  out.name = std::move(name);
  out.dtype = std::move(dtype);
  out.mpk_shape = std::move(shape);
  out.size_bytes = size_bytes;
  return out;
}

void test_ingress_uses_root_consumer() {
  mpk::MpkContract contract;
  contract.ingress_tensors.push_back(tensor("shared", "", {}, 0));

  mpk::MpkPluginIoContract downstream;
  downstream.name = "downstream";
  downstream.sequence = 1;
  downstream.input_tensors.push_back(tensor("shared", "FP32", {8, 8}, 256));

  mpk::MpkPluginIoContract root;
  root.name = "root";
  root.sequence = 0;
  root.input_tensors.push_back(tensor("shared", "INT8", {2, 3}, 6));
  root.output_tensors.push_back(tensor("shared", "INT8", {2, 3}, 6));

  contract.plugins = {downstream, root};
  contract.edges.push_back(mpk::MpkContractEdge{
      .src_plugin_index = 1,
      .src_output_index = 0,
      .dst_plugin_index = 0,
      .dst_input_index = 0,
      .src_plugin = "root",
      .dst_plugin = "downstream",
      .tensor_name = "shared",
  });

  const auto inputs = pcie_internal::detail::application_input_contracts(contract);
  require(inputs.size() == 1U, "expected one application input");
  require(inputs.front().logical_dtype == "INT8", "must select root-consumer dtype");
  require(inputs.front().logical_shape == std::vector<std::int64_t>({2, 3}),
          "must select root-consumer shape");
  require(inputs.front().size_bytes == 6U, "must select root-consumer size");
}

void test_unsupported_input_dtypes_fail_early() {
  for (const std::string dtype : {std::string("UINT16"), std::string("FP64")}) {
    bool rejected = false;
    try {
      pcie_internal::detail::validate_supported_input_dtype(
          tensor("input", dtype, {1}, dtype == "UINT16" ? 2U : 8U));
    } catch (const std::runtime_error& error) {
      rejected = std::string(error.what()).find("unsupported dtype") != std::string::npos;
    }
    require(rejected, dtype + " input must be rejected during model inspection");
  }

  pcie_internal::detail::validate_supported_input_dtype(tensor("input", "INT8", {1}, 1));
  pcie_internal::detail::validate_supported_input_dtype(tensor("input", "FP32", {1}, 4));
}

} // namespace

int main() {
  try {
    test_ingress_uses_root_consumer();
    test_unsupported_input_dtypes_fail_early();
    std::cout << "[PASS] model facts\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}

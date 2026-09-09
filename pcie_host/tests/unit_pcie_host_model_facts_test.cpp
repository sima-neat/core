#include "PcieModelFactsReaderInternal.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
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

void test_public_model_info_carries_quant() {
  pcie_internal::PcieModelFacts facts;
  pcie_internal::PcieTensorFact output;
  output.name = "head";
  output.quant = simaai::neat::pcie::QuantParams{.axis = -1, .scales = {0.5f}, .zero_points = {3}};
  facts.inputs.push_back(pcie_internal::PcieTensorFact{});
  facts.outputs.push_back(output);

  const auto info = pcie_internal::to_public_model_info(facts);
  require(!info.inputs.front().quant.has_value(), "input without quant must publish none");
  require(info.outputs.front().quant.has_value(), "output quant must reach ModelInfo");
  require(info.outputs.front().quant->scales == std::vector<float>{0.5f}, "quant scales mismatch");
  require(info.outputs.front().quant->zero_points == std::vector<std::int32_t>{3},
          "quant zero points mismatch");
}

mpk::MpkPluginIoContract stage(std::string name, std::string kernel,
                               std::vector<mpk::MpkTensorContract> inputs,
                               std::vector<mpk::MpkTensorContract> outputs) {
  mpk::MpkPluginIoContract out;
  out.name = std::move(name);
  out.kernel = std::move(kernel);
  out.input_tensors = std::move(inputs);
  out.output_tensors = std::move(outputs);
  return out;
}

mpk::MpkContract mla_only_contract() {
  mpk::MpkContract contract;
  contract.ingress_tensors.push_back(tensor("input_luv", "FP32", {2, 3, 4}, 96));

  auto mla_input = tensor("quantize_0", "INT8", {1, 2, 3, 4}, 24);
  mla_input.logical_shape = {2, 3, 4};
  mla_input.logical_dtype = "INT8";

  contract.plugins = {
      stage("quantize_0", "quantization_transform", {tensor("input_luv", "FP32", {2, 3, 4}, 96)},
            {mla_input}),
      stage("MLA_0", "mla", {mla_input}, {tensor("MLA_0", "", {1, 160}, 160)}),
  };
  for (std::size_t i = 0; i < contract.plugins.size(); ++i) {
    contract.plugins[i].sequence = static_cast<int>(i);
  }
  return contract;
}

template <typename Fn>
void require_rejected(Fn&& fn, const std::string& needle, const std::string& message) {
  bool rejected = false;
  try {
    fn();
  } catch (const std::runtime_error& error) {
    rejected = std::string(error.what()).find(needle) != std::string::npos;
  }
  require(rejected, message);
}

void test_mla_only_rejects_unsupported_stages() {
  require_rejected([] { (void)pcie_internal::detail::read_mla_only_facts(mpk::MpkContract{}); },
                   "MLA stage", "a contract without an MLA stage must be rejected");

  auto tessellated = mla_only_contract();
  tessellated.plugins.push_back(stage("tessellate_0", "tessellation_transform", {}, {}));
  require_rejected([&] { (void)pcie_internal::detail::read_mla_only_facts(tessellated); },
                   "tessellate_0", "a tessellation stage must be rejected");

  auto fused = mla_only_contract();
  fused.plugins.front().kernel = "quantize_tessellate_transform";
  require_rejected([&] { (void)pcie_internal::detail::read_mla_only_facts(fused); }, "quantize_0",
                   "a fused quantize+tessellate stage must be rejected");
}

void test_mla_only_rejects_non_dense_int8_inputs() {
  auto bf16 = mla_only_contract();
  bf16.plugins[1].input_tensors.front().logical_dtype = "BF16";
  require_rejected([&] { (void)pcie_internal::detail::read_mla_only_facts(bf16); }, "must be INT8",
                   "a BF16 MLA input must be rejected");

  auto padded = mla_only_contract();
  padded.plugins[1].input_tensors.front().size_bytes = 32;
  require_rejected([&] { (void)pcie_internal::detail::read_mla_only_facts(padded); },
                   "not a dense INT8 tensor", "a padded MLA input must be rejected");
}

void test_mla_only_facts_reject_until_outputs_exist() {
  require_rejected([] { (void)pcie_internal::detail::read_mla_only_facts(mla_only_contract()); },
                   "output facts are not implemented",
                   "mla_only facts must fail loudly rather than publish an empty output list");
}

} // namespace

int main() {
  try {
    test_ingress_uses_root_consumer();
    test_unsupported_input_dtypes_fail_early();
    test_public_model_info_carries_quant();
    test_mla_only_rejects_unsupported_stages();
    test_mla_only_rejects_non_dense_int8_inputs();
    test_mla_only_facts_reject_until_outputs_exist();
    std::cout << "[PASS] model facts\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}

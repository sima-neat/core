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

mpk::MpkTensorContract head(std::string name, std::vector<std::int64_t> mpk_shape,
                            std::vector<std::int64_t> logical_shape, const std::size_t size_bytes) {
  auto out = tensor(std::move(name), "INT8", std::move(mpk_shape), size_bytes);
  out.logical_shape = std::move(logical_shape);
  out.logical_dtype = "INT8";
  return out;
}

void link(mpk::MpkContract& contract, const std::size_t src, const int src_output,
          const std::size_t dst, const int dst_input) {
  contract.edges.push_back(mpk::MpkContractEdge{
      .src_plugin_index = src,
      .src_output_index = src_output,
      .dst_plugin_index = dst,
      .dst_input_index = dst_input,
      .src_plugin = contract.plugins[src].name,
      .dst_plugin = contract.plugins[dst].name,
      .tensor_name =
          contract.plugins[src].output_tensors[static_cast<std::size_t>(src_output)].name,
  });
}

mpk::MpkContract mla_only_contract() {
  mpk::MpkContract contract;
  contract.ingress_tensors.push_back(tensor("input_luv", "FP32", {2, 3, 4}, 96));

  const auto mla_input = head("quantize_0", {1, 2, 3, 4}, {2, 3, 4}, 24);
  const auto carrier = tensor("MLA_0", "", {1, 160}, 160);
  const auto unpack_0 = head("MLA_0_ofm_unpack_transform_0", {1, 2, 3, 16}, {2, 3, 16}, 96);
  const auto unpack_1 = head("MLA_0_ofm_unpack_transform_1", {1, 1, 4, 16}, {1, 4, 16}, 64);
  const auto slice_0 =
      head("slice_MLA_0/tuple_get_item_0_slice_transform", {1, 2, 3, 2}, {2, 3, 2}, 12);
  const auto out_0 = tensor("dequantize_2/head_0", "FP32", {2, 3, 2}, 48);
  const auto out_1 = tensor("dequantize_3/head_1", "FP32", {1, 4, 16}, 256);

  contract.plugins = {
      stage("quantize_0", "quantization_transform", {tensor("input_luv", "FP32", {2, 3, 4}, 96)},
            {mla_input}),
      stage("MLA_0", "mla", {mla_input}, {carrier}),
      stage("MLA_0_ofm_unpack_transform", "unpack_transform", {carrier}, {unpack_0, unpack_1}),
      stage(slice_0.name, "slice_transform", {unpack_0}, {slice_0}),
      stage("dequantize_2", "dequantization_transform", {slice_0}, {out_0}),
      stage("dequantize_3", "dequantization_transform", {unpack_1}, {out_1}),
      stage("PassThrough", "pass_through", {out_0, out_1}, {out_0, out_1}),
  };
  contract.plugins[0].quant = mpk::MpkQuantContract{.scales = {4.0}, .zero_points = {-128}};
  contract.plugins[3].slice_begin = {0, 0, 0, 0};
  contract.plugins[4].quant = mpk::MpkQuantContract{.scales = {0.5}, .zero_points = {3}};
  contract.plugins[5].quant = mpk::MpkQuantContract{.scales = {2.0}, .zero_points = {-7}};
  for (std::size_t i = 0; i < contract.plugins.size(); ++i) {
    contract.plugins[i].sequence = static_cast<int>(i);
  }
  link(contract, 0, 0, 1, 0);
  link(contract, 1, 0, 2, 0);
  link(contract, 2, 0, 3, 0);
  link(contract, 3, 0, 4, 0);
  link(contract, 2, 1, 5, 0);
  link(contract, 4, 0, 6, 0);
  link(contract, 5, 0, 6, 1);
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

void test_mla_only_facts_describe_ingress_and_heads() {
  const auto facts = pcie_internal::detail::read_mla_only_facts(mla_only_contract());

  require(facts.inputs.size() == 1U, "expected one mla_only input");
  require(facts.inputs.front().name == "input_luv", "input must carry the public name");
  require(facts.inputs.front().dtype == "INT8", "input must be INT8");
  require(facts.inputs.front().shape == std::vector<std::int64_t>({2, 3, 4}),
          "input must use the MLA logical shape");
  require(facts.inputs.front().size_bytes == 24U && facts.packed_input_bytes == 24U,
          "input must use the MLA byte size");
  require(facts.inputs.front().quant.has_value() &&
              facts.inputs.front().quant->scales == std::vector<float>{0.25f} &&
              facts.inputs.front().quant->zero_points == std::vector<std::int32_t>{-128},
          "input quant must publish the inverted quantize scale");

  require(facts.outputs.size() == 2U, "expected two mla_only heads");
  const auto& sliced = facts.outputs[0];
  require(sliced.name == "head_0", "sliced head must carry the dequantized output name");
  require(sliced.quant.has_value() && sliced.quant->scales == std::vector<float>{0.5f} &&
              sliced.quant->zero_points == std::vector<std::int32_t>{3},
          "sliced head must carry its dequantize parameters");
  require(sliced.dtype == "INT8" && sliced.shape == std::vector<std::int64_t>({2, 3, 2}) &&
              sliced.size_bytes == 12U,
          "sliced head must publish its logical INT8 geometry");
  require(sliced.transport_strides_bytes == std::vector<std::int64_t>({48, 16, 1}),
          "sliced head must carry the padded unpack strides");
  require(sliced.payload_offset == 0U && sliced.transport_size_bytes == 96U,
          "sliced head must span its unpack region");
  require(sliced.dense_offset == 0U, "first head starts the dense block");

  const auto& direct = facts.outputs[1];
  require(direct.name == "head_1" && direct.quant.has_value() &&
              direct.quant->scales == std::vector<float>{2.0f} &&
              direct.quant->zero_points == std::vector<std::int32_t>{-7},
          "direct head must carry its dequantized name and parameters");
  require(direct.shape == std::vector<std::int64_t>({1, 4, 16}) && direct.size_bytes == 64U,
          "direct head must publish its logical INT8 geometry");
  require(direct.transport_strides_bytes == std::vector<std::int64_t>({64, 16, 1}),
          "direct head must carry contiguous strides");
  require(direct.payload_offset == 96U && direct.transport_size_bytes == 64U,
          "direct head must follow the sliced head in the carrier");
  require(direct.dense_offset == 12U, "second head follows the first in the dense block");

  require(facts.packed_output_bytes == 160U, "packed output must be the raw carrier");
  require(facts.dense_output_bytes == 76U, "dense output must be the logical sum");
  require(!facts.has_preprocess && !facts.has_boxdecode, "mla_only publishes no CVU stages");
}

void test_mla_only_rejects_unusable_output_geometry() {
  auto gap = mla_only_contract();
  gap.plugins[1].output_tensors.front().size_bytes = 200;
  gap.plugins[2].input_tensors.front().size_bytes = 200;
  require_rejected([&] { (void)pcie_internal::detail::read_mla_only_facts(gap); }, "do not tile",
                   "heads that leave carrier bytes unclaimed must be rejected");

  auto lane_split = mla_only_contract();
  lane_split.plugins[1].has_align_c16 = true;
  lane_split.plugins[1].align_c16 = true;
  require_rejected([&] { (void)pcie_internal::detail::read_mla_only_facts(lane_split); },
                   "lane-split", "a lane-split MLA boundary must be rejected");

  auto orphan = mla_only_contract();
  orphan.edges.erase(orphan.edges.begin() + 4);
  require_rejected([&] { (void)pcie_internal::detail::read_mla_only_facts(orphan); },
                   "no dequantize consumer", "a head without a dequantize stage must be rejected");

  auto reshaped = mla_only_contract();
  reshaped.plugins[5].output_tensors.front().mpk_shape = {1, 4, 15};
  require_rejected([&] { (void)pcie_internal::detail::read_mla_only_facts(reshaped); },
                   "shape of its dequantized output",
                   "a head whose dequantized output has another shape must be rejected");
}

} // namespace

int main() {
  try {
    test_ingress_uses_root_consumer();
    test_unsupported_input_dtypes_fail_early();
    test_public_model_info_carries_quant();
    test_mla_only_rejects_unsupported_stages();
    test_mla_only_rejects_non_dense_int8_inputs();
    test_mla_only_facts_describe_ingress_and_heads();
    test_mla_only_rejects_unusable_output_geometry();
    std::cout << "[PASS] model facts\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}

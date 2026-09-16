#include "PcieModelFactsReaderInternal.h"

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

// An MLA-only capable contract: one quantize stage per input, optionally joined by an ifm pack
// (the MLA always ingests one buffer), the MLA, an unpack into two heads, a slice on the first
// head, and one dequantize per head. The pack consumes the inputs in reverse order so that
// publishing in pack order - the layout of the staged payload - is load-bearing.
mpk::MpkContract mla_only_contract(const std::size_t input_count = 1, const bool pack = false) {
  mpk::MpkContract contract;
  const mpk::MpkTensorContract ingress[] = {tensor("input_0", "FP32", {2, 3, 4}, 96),
                                            tensor("input_1", "FP32", {1, 4, 4}, 64)};
  const mpk::MpkTensorContract quantized[] = {head("quantize_0", {1, 2, 3, 4}, {2, 3, 4}, 24),
                                              head("quantize_1", {1, 1, 4, 4}, {1, 4, 4}, 16)};
  const mpk::MpkQuantContract input_quant[] = {{.scales = {4.0}, .zero_points = {-128}},
                                               {.scales = {8.0}, .zero_points = {5}}};
  std::vector<mpk::MpkTensorContract> packed_parts;
  std::size_t packed_bytes = 0;
  for (std::size_t i = 0; i < input_count; ++i) {
    contract.ingress_tensors.push_back(ingress[i]);
    contract.plugins.push_back(
        stage(quantized[i].name, "quantization_transform", {ingress[i]}, {quantized[i]}));
    contract.plugins.back().quant = input_quant[i];
    packed_parts.insert(packed_parts.begin(), quantized[i]);
    packed_bytes += quantized[i].size_bytes;
  }
  const auto packed = tensor("MLA_0_ifm_pack_transform", "",
                             {1, static_cast<std::int64_t>(packed_bytes)}, packed_bytes);
  const auto carrier = tensor("MLA_0", "", {1, 160}, 160);
  const auto unpack_0 = head("MLA_0_ofm_unpack_transform_0", {1, 2, 3, 16}, {2, 3, 16}, 96);
  const auto unpack_1 = head("MLA_0_ofm_unpack_transform_1", {1, 1, 4, 16}, {1, 4, 16}, 64);
  const auto slice_0 =
      head("slice_MLA_0/tuple_get_item_0_slice_transform", {1, 2, 3, 2}, {2, 3, 2}, 12);
  const auto out_0 = tensor("dequantize_2/head_0", "FP32", {2, 3, 2}, 48);
  const auto out_1 = tensor("dequantize_3/head_1", "FP32", {1, 4, 16}, 256);

  if (pack) {
    contract.plugins.push_back(stage(packed.name, "pack_transform", packed_parts, {packed}));
  }
  const std::size_t mla = contract.plugins.size();
  contract.plugins.push_back(stage("MLA_0", "mla", {pack ? packed : quantized[0]}, {carrier}));
  contract.plugins.push_back(
      stage("MLA_0_ofm_unpack_transform", "unpack_transform", {carrier}, {unpack_0, unpack_1}));
  contract.plugins.push_back(stage(slice_0.name, "slice_transform", {unpack_0}, {slice_0}));
  contract.plugins.back().slice_begin = {0, 0, 0, 0};
  contract.plugins.push_back(stage("dequantize_2", "dequantization_transform", {slice_0}, {out_0}));
  contract.plugins.back().quant = mpk::MpkQuantContract{.scales = {0.5}, .zero_points = {3}};
  contract.plugins.push_back(
      stage("dequantize_3", "dequantization_transform", {unpack_1}, {out_1}));
  contract.plugins.back().quant = mpk::MpkQuantContract{.scales = {2.0}, .zero_points = {-7}};
  contract.plugins.push_back(stage("PassThrough", "pass_through", {out_0, out_1}, {out_0, out_1}));
  for (std::size_t i = 0; i < contract.plugins.size(); ++i) {
    contract.plugins[i].sequence = static_cast<int>(i);
  }

  for (std::size_t i = 0; i < input_count; ++i) {
    if (pack) {
      link(contract, i, 0, mla - 1, static_cast<int>(input_count - 1 - i));
    } else {
      link(contract, i, 0, mla, 0);
    }
  }
  if (pack) {
    link(contract, mla - 1, 0, mla, 0);
  }
  link(contract, mla, 0, mla + 1, 0);
  link(contract, mla + 1, 0, mla + 2, 0);
  link(contract, mla + 2, 0, mla + 3, 0);
  link(contract, mla + 1, 1, mla + 4, 0);
  link(contract, mla + 3, 0, mla + 5, 0);
  link(contract, mla + 4, 0, mla + 5, 1);
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
  bf16.plugins[0].output_tensors.front().logical_dtype = "BF16";
  require_rejected([&] { (void)pcie_internal::detail::read_mla_only_facts(bf16); }, "must be INT8",
                   "a BF16 MLA input must be rejected");

  auto padded = mla_only_contract();
  padded.plugins[0].output_tensors.front().size_bytes = 32;
  require_rejected([&] { (void)pcie_internal::detail::read_mla_only_facts(padded); },
                   "not a dense INT8 tensor", "a padded MLA input must be rejected");
}

void test_mla_only_facts_describe_ingress_and_heads() {
  const auto facts = pcie_internal::detail::read_mla_only_facts(mla_only_contract());

  require(facts.inputs.size() == 1U, "expected one mla_only input");
  require(facts.inputs.front().name == "input_0", "input must carry the public name");
  require(facts.inputs.front().dtype == "INT8", "input must be INT8");
  require(facts.inputs.front().shape == std::vector<std::int64_t>({2, 3, 4}),
          "input must use the MLA logical shape");
  require(facts.inputs.front().size_bytes == 24U && facts.packed_input_bytes == 24U,
          "input must use the MLA byte size");
  require(facts.inputs.front().quant.has_value() &&
              facts.inputs.front().quant->scales == std::vector<float>{0.25f} &&
              facts.inputs.front().quant->zero_points == std::vector<std::int32_t>{-128},
          "input quant must publish the inverted quantize scale");
  require(!facts.packed_input.has_value(), "a direct MLA input needs no packed carrier");

  require(facts.outputs.size() == 2U, "expected two mla_only heads");
  const auto& sliced = facts.outputs[0];
  require(sliced.name == "head_0", "sliced head must carry the dequantized output name");
  require(sliced.quant.has_value() && sliced.quant->scales == std::vector<float>{2.0f} &&
              sliced.quant->zero_points == std::vector<std::int32_t>{3},
          "sliced head must publish the inverted dequantize scale");
  require(sliced.dtype == "INT8" && sliced.shape == std::vector<std::int64_t>({2, 3, 2}) &&
              sliced.size_bytes == 12U,
          "sliced head must publish its logical INT8 geometry");
  require(sliced.transport_strides_bytes == std::vector<std::int64_t>({48, 16, 1}),
          "sliced head must carry the padded unpack strides");
  require(sliced.payload_offset == 0U && sliced.dense_offset == 0U,
          "sliced head must start the carrier and the dense block");

  const auto& direct = facts.outputs[1];
  require(direct.name == "head_1" && direct.quant.has_value() &&
              direct.quant->scales == std::vector<float>{0.5f} &&
              direct.quant->zero_points == std::vector<std::int32_t>{-7},
          "direct head must carry its dequantized name and parameters");
  require(direct.shape == std::vector<std::int64_t>({1, 4, 16}) && direct.size_bytes == 64U,
          "direct head must publish its logical INT8 geometry");
  require(direct.transport_strides_bytes == std::vector<std::int64_t>({64, 16, 1}),
          "direct head must carry contiguous strides");
  require(direct.payload_offset == 96U && direct.dense_offset == 12U,
          "direct head must follow the sliced head in the carrier and the dense block");

  require(facts.packed_output_bytes == 160U, "packed output must be the raw carrier");
  require(facts.dense_output_bytes == 76U, "dense output must be the logical sum");
  require(!facts.has_preprocess && !facts.has_boxdecode, "mla_only publishes no CVU stages");
}

void test_mla_only_supports_multiple_inputs() {
  const auto facts = pcie_internal::detail::read_mla_only_facts(mla_only_contract(2, true));

  require(facts.inputs.size() == 2U, "expected two mla_only inputs");
  require(facts.inputs[0].name == "input_1" && facts.inputs[1].name == "input_0",
          "inputs must be published in pack order, which is the staged payload layout");
  require(facts.inputs[0].shape == std::vector<std::int64_t>({1, 4, 4}) &&
              facts.inputs[1].shape == std::vector<std::int64_t>({2, 3, 4}),
          "each input must carry the geometry of its own quantize stage");
  require(facts.inputs[0].size_bytes == 16U && facts.inputs[1].size_bytes == 24U,
          "each input must carry the byte size of its own quantize stage");
  require(facts.packed_input_bytes == 40U, "packed input bytes must sum every input");
  require(facts.inputs[0].quant.has_value() &&
              facts.inputs[0].quant->scales == std::vector<float>{0.125f} &&
              facts.inputs[0].quant->zero_points == std::vector<std::int32_t>{5},
          "input_1 must publish its own quantize parameters");
  require(facts.inputs[1].quant.has_value() &&
              facts.inputs[1].quant->scales == std::vector<float>{0.25f} &&
              facts.inputs[1].quant->zero_points == std::vector<std::int32_t>{-128},
          "input_0 must publish its own quantize parameters");
}

// One input can reach the MLA through a pack stage too. The MLA then ingests the packed segment,
// not the quantized tensor the host submits, so the facts have to carry the carrier descriptor.
void test_mla_only_packs_a_single_input() {
  const auto facts = pcie_internal::detail::read_mla_only_facts(mla_only_contract(1, true));
  require(facts.inputs.size() == 1U, "the packed single-input model exposes one input");
  require(facts.packed_input.has_value() && facts.packed_input->name == "MLA_0_ifm_pack_transform",
          "one input reaching the MLA through a pack still needs the packed carrier");
  require(facts.packed_input->size_bytes == facts.packed_input_bytes, "packed input size mismatch");
}

void test_mla_only_rejects_hybrid_quantization() {
  auto not_quantized = mla_only_contract(2, true);
  not_quantized.plugins[1].kernel = "pass_through";
  require_rejected([&] { (void)pcie_internal::detail::read_mla_only_facts(not_quantized); },
                   "not a quantize stage",
                   "a packed input not produced by a quantize stage must be rejected");

  auto stranded = mla_only_contract(2, true);
  stranded.ingress_tensors.push_back(tensor("input_2", "FP32", {1, 2}, 8));
  require_rejected([&] { (void)pcie_internal::detail::read_mla_only_facts(stranded); },
                   "hybrid host/card quantization",
                   "a model input without its own quantize stage must be rejected");
}

void test_mla_only_rejects_unusable_output_geometry() {
  auto lane_split = mla_only_contract();
  lane_split.plugins[1].has_align_c16 = true;
  lane_split.plugins[1].align_c16 = true;
  require_rejected([&] { (void)pcie_internal::detail::read_mla_only_facts(lane_split); },
                   "lane-split", "a lane-split MLA boundary must be rejected");

  auto orphan = mla_only_contract();
  std::erase_if(orphan.edges, [](const mpk::MpkContractEdge& edge) {
    return edge.tensor_name == "MLA_0_ofm_unpack_transform_1";
  });
  require_rejected([&] { (void)pcie_internal::detail::read_mla_only_facts(orphan); },
                   "no dequantize consumer", "a head without a dequantize stage must be rejected");
}

} // namespace

int main() {
  try {
    test_ingress_uses_root_consumer();
    test_unsupported_input_dtypes_fail_early();
    test_mla_only_rejects_unsupported_stages();
    test_mla_only_rejects_non_dense_int8_inputs();
    test_mla_only_facts_describe_ingress_and_heads();
    test_mla_only_supports_multiple_inputs();
    test_mla_only_packs_a_single_input();
    test_mla_only_rejects_hybrid_quantization();
    test_mla_only_rejects_unusable_output_geometry();
    std::cout << "[PASS] model facts\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}

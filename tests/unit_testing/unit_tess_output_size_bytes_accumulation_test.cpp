#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "test_main.h"

#include <cstdint>
#include <vector>

namespace {

using simaai::neat::pipeline_internal::sima::MpkContract;
using simaai::neat::pipeline_internal::sima::MpkPluginIoContract;
using simaai::neat::pipeline_internal::sima::MpkQuantContract;
using simaai::neat::pipeline_internal::sima::MpkShapeSemantics;
using simaai::neat::pipeline_internal::sima::MpkTensorContract;
namespace pcs = simaai::neat::pipeline_internal::sima::plugin_contracts;

MpkTensorContract make_tensor_with_size(const std::string& name, const std::string& dtype,
                                        const std::vector<std::int64_t>& shape,
                                        std::size_t size_bytes) {
  MpkTensorContract t;
  t.name = name;
  t.dtype = dtype;
  t.logical_dtype = dtype;
  t.logical_shape = shape;
  t.mpk_shape = shape;
  t.shape_semantics = MpkShapeSemantics::Geometry;
  t.size_bytes = size_bytes;
  return t;
}

MpkPluginIoContract make_tess_stage(std::vector<MpkTensorContract> output_tensors) {
  MpkPluginIoContract stage;
  stage.name = "tess_0";
  stage.kernel = "tess";
  stage.sequence = 0;
  stage.frame_type = "BF16";
  stage.slice_shape = {640, 32, 3};
  stage.input_tensors = {make_tensor_with_size("ifm0", "BF16", {640, 640, 3}, 640 * 640 * 3 * 2)};
  stage.output_tensors = std::move(output_tensors);
  return stage;
}

MpkPluginIoContract make_quanttess_stage(std::vector<MpkTensorContract> output_tensors) {
  MpkPluginIoContract stage;
  stage.name = "quanttess_0";
  stage.kernel = "quanttess";
  stage.sequence = 0;
  stage.frame_type = "INT8";
  stage.slice_shape = {32, 128, 3};
  stage.canonical_input_dtype = "FP32";
  stage.canonical_output_dtype = "INT8";
  stage.round_off = "RT_ZERO";
  stage.quant = MpkQuantContract{{1.0}, {0}, -1};
  stage.input_tensors = {make_tensor_with_size("ifm0", "FP32", {640, 640, 3}, 640 * 640 * 3 * 4)};
  stage.output_tensors = std::move(output_tensors);
  return stage;
}

// Build a minimal MPK contract with a separate quant stage and tess stage.
MpkContract make_quant_plus_tess_contract(std::vector<MpkTensorContract> tess_output_tensors) {
  MpkPluginIoContract quant;
  quant.name = "quant_0";
  quant.kernel = "quant_transform";
  quant.sequence = 0;
  quant.canonical_input_dtype = "FP32";
  quant.canonical_output_dtype = "INT8";
  quant.round_off = "RT_ZERO";
  quant.quant = MpkQuantContract{{1.0}, {0}, -1};
  quant.input_tensors = {make_tensor_with_size("ifm0", "FP32", {640, 640, 3}, 640 * 640 * 3 * 4)};
  quant.output_tensors = {make_tensor_with_size("qfm0", "INT8", {640, 640, 3}, 640 * 640 * 3)};

  MpkPluginIoContract tess;
  tess.name = "tess_0";
  tess.kernel = "tess";
  tess.sequence = 1;
  tess.frame_type = "INT8";
  tess.slice_shape = {32, 128, 3};
  tess.input_tensors = {make_tensor_with_size("qfm0", "INT8", {640, 640, 3}, 640 * 640 * 3)};
  tess.output_tensors = std::move(tess_output_tensors);

  MpkContract contract;
  contract.plugins = {quant, tess};
  return contract;
}

void verify_tess_stage_single_output_tensor() {
  const auto stage = make_tess_stage({make_tensor_with_size("ofm0", "BF16", {640, 32, 3}, 1000U)});
  const auto subset = pcs::extract_tessellate_contract_subset_from_stage(stage);
  require(subset.output_size_bytes == 1000U,
          "tessellate single-output: output_size_bytes must equal the sole tensor's size_bytes");
}

void verify_tess_stage_no_output_tensors() {
  const auto stage = make_tess_stage({});
  const auto subset = pcs::extract_tessellate_contract_subset_from_stage(stage);
  require(subset.output_size_bytes == 0U,
          "tessellate no-output: output_size_bytes must remain zero when output_tensors is empty");
}

void verify_tess_stage_multiple_output_tensors_are_summed() {
  const auto stage = make_tess_stage({
      make_tensor_with_size("ofm0", "BF16", {640, 32, 3}, 400U),
      make_tensor_with_size("ofm1", "BF16", {640, 32, 3}, 600U),
      make_tensor_with_size("ofm2", "BF16", {640, 32, 3}, 200U),
  });
  const auto subset = pcs::extract_tessellate_contract_subset_from_stage(stage);
  require(subset.output_size_bytes == 1200U,
          "tessellate multi-output: output_size_bytes must be the sum across all output tensors, "
          "not just the first");
}

void verify_tess_stage_first_tensor_zero_but_others_nonzero() {
  // Verifies that a leading tensor with size_bytes=0 does not suppress accumulation of later ones.
  const auto stage = make_tess_stage({
      make_tensor_with_size("ofm0", "BF16", {640, 32, 3}, 0U),
      make_tensor_with_size("ofm1", "BF16", {640, 32, 3}, 500U),
  });
  const auto subset = pcs::extract_tessellate_contract_subset_from_stage(stage);
  require(subset.output_size_bytes == 500U,
          "tessellate: leading zero-size tensor must not prevent accumulation from subsequent "
          "tensors");
}

void verify_quanttess_stage_single_output_tensor() {
  const auto stage =
      make_quanttess_stage({make_tensor_with_size("ofm0", "INT8", {640, 640, 3}, 2000U)});
  const auto subset = pcs::extract_quanttess_contract_subset_from_stage(stage);
  require(subset.output_size_bytes == 2000U,
          "quanttess single-output: output_size_bytes must equal the sole tensor's size_bytes");
}

void verify_quanttess_stage_multiple_output_tensors_are_summed() {
  const auto stage = make_quanttess_stage({
      make_tensor_with_size("ofm0", "INT8", {320, 640, 3}, 300U),
      make_tensor_with_size("ofm1", "INT8", {320, 640, 3}, 700U),
  });
  const auto subset = pcs::extract_quanttess_contract_subset_from_stage(stage);
  require(subset.output_size_bytes == 1000U,
          "quanttess multi-output: output_size_bytes must be the sum across all output tensors");
}

void verify_quanttess_mpk_separate_stages_multiple_tess_outputs_are_summed() {
  const auto contract = make_quant_plus_tess_contract({
      make_tensor_with_size("tess_out0", "INT8", {32, 128, 3}, 800U),
      make_tensor_with_size("tess_out1", "INT8", {32, 128, 3}, 1200U),
  });
  const auto subset = pcs::extract_quanttess_contract_subset_from_mpk(contract);
  require(subset.output_size_bytes == 2000U,
          "quanttess (separate quant+tess stages): output_size_bytes must sum all tess-stage "
          "output tensors, not just the first");
}

void verify_quanttess_mpk_separate_stages_single_tess_output() {
  const auto contract = make_quant_plus_tess_contract(
      {make_tensor_with_size("tess_out0", "INT8", {32, 128, 3}, 1500U)});
  const auto subset = pcs::extract_quanttess_contract_subset_from_mpk(contract);
  require(subset.output_size_bytes == 1500U,
          "quanttess (separate quant+tess stages): single tess output must be preserved exactly");
}

void verify_quanttess_mpk_separate_stages_no_tess_outputs() {
  const auto contract = make_quant_plus_tess_contract({});
  const auto subset = pcs::extract_quanttess_contract_subset_from_mpk(contract);
  require(subset.output_size_bytes == 0U,
          "quanttess (separate quant+tess stages): zero tess outputs must leave output_size_bytes "
          "at zero");
}

} // namespace

RUN_TEST("unit_tess_output_size_bytes_accumulation_test", ([] {
           verify_tess_stage_single_output_tensor();
           verify_tess_stage_no_output_tensors();
           verify_tess_stage_multiple_output_tensors_are_summed();
           verify_tess_stage_first_tensor_zero_but_others_nonzero();
           verify_quanttess_stage_single_output_tensor();
           verify_quanttess_stage_multiple_output_tensors_are_summed();
           verify_quanttess_mpk_separate_stages_multiple_tess_outputs_are_summed();
           verify_quanttess_mpk_separate_stages_single_tess_output();
           verify_quanttess_mpk_separate_stages_no_tess_outputs();
         }));

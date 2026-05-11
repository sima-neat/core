#include "pipeline/internal/sima/MlaStaticContractExtractor.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "test_main.h"

#include <sstream>
#include <string>
#include <vector>

namespace {

using simaai::neat::pipeline_internal::sima::MlaStaticContract;
using simaai::neat::pipeline_internal::sima::MpkContract;
using simaai::neat::pipeline_internal::sima::MpkContractEdge;
using simaai::neat::pipeline_internal::sima::MpkPluginIoContract;
using simaai::neat::pipeline_internal::sima::MpkShapeSemantics;
using simaai::neat::pipeline_internal::sima::MpkTensorContract;

MpkTensorContract make_tensor(const std::string& name,
                              const std::string& dtype,
                              std::vector<std::int64_t> shape,
                              const std::size_t size_bytes) {
  MpkTensorContract tensor;
  tensor.name = name;
  tensor.dtype = dtype;
  tensor.logical_shape = shape;
  tensor.mpk_shape = std::move(shape);
  tensor.shape_semantics = MpkShapeSemantics::Geometry;
  tensor.size_bytes = size_bytes;
  return tensor;
}

MpkContract make_packed_bf16_mla_boundary_fixture() {
  MpkContract mpk;

  MpkPluginIoContract mla;
  mla.name = "MLA_0";
  mla.processor = "MLA";
  mla.kernel = "mla";
  mla.canonical_output_dtype = "BF16";
  auto mla_ofm = make_tensor("MLA_0", "BF16", {1, 1, 1, 640016}, 1280032U);
  mla_ofm.physical_index = 0;
  mla.output_tensors.push_back(std::move(mla_ofm));
  mpk.plugins.push_back(std::move(mla));

  MpkPluginIoContract unpack;
  unpack.name = "MLA_0_ofm_unpack_transform";
  unpack.kernel = "unpack_transform";
  for (int i = 0; i < 22; ++i) {
    if (i == 0) {
      unpack.output_tensors.push_back(make_tensor(
          "MLA_0_ofm_unpack_transform_0", "BF16", {1, 12, 52, 48}, 59904U));
      continue;
    }
    if (i == 1) {
      unpack.output_tensors.push_back(make_tensor(
          "MLA_0_ofm_unpack_transform_1", "BF16", {1, 3, 13, 16}, 1248U));
      continue;
    }
    if (i == 21) {
      unpack.output_tensors.push_back(make_tensor(
          "MLA_0_ofm_unpack_transform_21", "BF16", {1, 96, 416, 8}, 638976U));
      continue;
    }
    const std::size_t size_bytes = (i == 2) ? 30268U : 30258U;
    unpack.output_tensors.push_back(
        make_tensor("MLA_0_ofm_unpack_transform_" + std::to_string(i), "BF16",
                    {1, 1, 1, static_cast<std::int64_t>(size_bytes / 2U)}, size_bytes));
  }
  mpk.plugins.push_back(std::move(unpack));

  mpk.edges.push_back(MpkContractEdge{
      .src_plugin_index = 0U,
      .src_output_index = 0,
      .dst_plugin_index = 1U,
      .dst_input_index = 0,
      .src_plugin = "MLA_0",
      .dst_plugin = "MLA_0_ofm_unpack_transform",
      .tensor_name = "MLA_0",
  });

  MpkPluginIoContract slice0;
  slice0.name = "slice_MLA_0/tuple_get_item_0_slice_transform";
  slice0.kernel = "slice_transform";
  slice0.slice_begin = {0, 0, 0, 0};
  slice0.slice_end = {1, 12, 52, 45};
  slice0.input_tensors.push_back(
      make_tensor("MLA_0_ofm_unpack_transform_0", "BF16", {1, 12, 52, 48}, 59904U));
  slice0.output_tensors.push_back(
      make_tensor("slice_MLA_0/tuple_get_item_0_slice_transform", "BF16", {1, 12, 52, 45}, 56160U));
  mpk.plugins.push_back(std::move(slice0));
  mpk.edges.push_back(MpkContractEdge{
      .src_plugin_index = 1U,
      .src_output_index = 0,
      .dst_plugin_index = 2U,
      .dst_input_index = 0,
      .src_plugin = "MLA_0_ofm_unpack_transform",
      .dst_plugin = "slice_MLA_0/tuple_get_item_0_slice_transform",
      .tensor_name = "MLA_0_ofm_unpack_transform_0",
  });

  MpkPluginIoContract cast0;
  cast0.name = "cast_0_post";
  cast0.kernel = "cast_transform";
  cast0.input_tensors.push_back(
      make_tensor("slice_MLA_0/tuple_get_item_0_slice_transform", "BF16", {1, 12, 52, 45}, 56160U));
  cast0.output_tensors.push_back(make_tensor("cast_0_post_out", "FP32", {1, 12, 52, 45}, 112320U));
  mpk.plugins.push_back(std::move(cast0));
  mpk.edges.push_back(MpkContractEdge{
      .src_plugin_index = 2U,
      .src_output_index = 0,
      .dst_plugin_index = 3U,
      .dst_input_index = 0,
      .src_plugin = "slice_MLA_0/tuple_get_item_0_slice_transform",
      .dst_plugin = "cast_0_post",
      .tensor_name = "slice_MLA_0/tuple_get_item_0_slice_transform",
  });

  MpkPluginIoContract cast1;
  cast1.name = "cast_1_post";
  cast1.kernel = "cast_transform";
  cast1.input_tensors.push_back(
      make_tensor("MLA_0_ofm_unpack_transform_1", "BF16", {1, 3, 13, 16}, 1248U));
  cast1.output_tensors.push_back(make_tensor("cast_1_post_out", "FP32", {1, 3, 13, 16}, 2496U));
  mpk.plugins.push_back(std::move(cast1));
  mpk.edges.push_back(MpkContractEdge{
      .src_plugin_index = 1U,
      .src_output_index = 1,
      .dst_plugin_index = 4U,
      .dst_input_index = 0,
      .src_plugin = "MLA_0_ofm_unpack_transform",
      .dst_plugin = "cast_1_post",
      .tensor_name = "MLA_0_ofm_unpack_transform_1",
  });

  MpkPluginIoContract cast21;
  cast21.name = "cast_21_post";
  cast21.kernel = "cast_transform";
  cast21.input_tensors.push_back(
      make_tensor("MLA_0_ofm_unpack_transform_21", "BF16", {1, 96, 416, 8}, 638976U));
  cast21.output_tensors.push_back(
      make_tensor("cast_21_post_out", "FP32", {1, 96, 416, 8}, 1277952U));
  mpk.plugins.push_back(std::move(cast21));
  mpk.edges.push_back(MpkContractEdge{
      .src_plugin_index = 1U,
      .src_output_index = 21,
      .dst_plugin_index = 5U,
      .dst_input_index = 0,
      .src_plugin = "MLA_0_ofm_unpack_transform",
      .dst_plugin = "cast_21_post",
      .tensor_name = "MLA_0_ofm_unpack_transform_21",
  });

  return mpk;
}

MpkPluginIoContract make_declared_size_mla_stage_fixture() {
  MpkPluginIoContract mla;
  mla.name = "MLA_declared_size";
  mla.processor = "MLA";
  mla.kernel = "mla";
  mla.canonical_output_dtype = "BF16";
  auto ofm = make_tensor("ofm0", "BF16", {1, 24, 104, 9}, 79872U);
  ofm.physical_index = 0;
  ofm.tensor_index = 0;
  mla.output_tensors.push_back(std::move(ofm));
  return mla;
}

} // namespace

RUN_TEST("unit_mla_boundary_output_contract_test", ([] {
           using namespace simaai::neat::pipeline_internal::sima;

           const MpkContract mpk = make_packed_bf16_mla_boundary_fixture();

           const auto logical_outputs = get_mla_logical_outputs_contract(mpk);
           require(logical_outputs.size() == 22U,
                   "MLA logical outputs should preserve the unpack fanout width");
           require(logical_outputs[0].size_bytes == 56160U,
                   "sliced MLA head should use slice boundary size instead of post-cast size");
           require(logical_outputs[0].dtype == "BF16",
                   "sliced MLA head should preserve BF16 boundary dtype");
           require(logical_outputs[0].name == "slice_MLA_0/tuple_get_item_0_slice_transform",
                   "sliced MLA head should expose the slice transform name");
           require(logical_outputs[1].size_bytes == 1248U,
                   "unsliced MLA head should use unpack boundary size instead of post-cast size");
           require(logical_outputs[1].name == "MLA_0_ofm_unpack_transform_1",
                   "unsliced MLA head should keep the unpack boundary name");
           require(logical_outputs[21].size_bytes == 638976U,
                   "semseg MLA head should preserve the BF16 unpack size instead of FP32 cast size");
           require(logical_outputs[21].dtype == "BF16",
                   "semseg MLA head should preserve BF16 boundary dtype");
           // Per-head byte_offset is no longer recorded on the logical output
           // (slices/unpacks track view geometry through other fields now).
           require(logical_outputs[0].byte_offset == 0,
                   "logical output byte_offset should default to zero");
           require(logical_outputs[1].byte_offset == 0,
                   "logical output byte_offset should default to zero");
           require(logical_outputs[21].byte_offset == 0,
                   "logical output byte_offset should default to zero");
           const std::uint64_t logical_end =
               static_cast<std::uint64_t>(logical_outputs.back().byte_offset) +
               logical_outputs.back().size_bytes;
           require(logical_end <= 1280032U,
                   "packed MLA logical outputs should stay within the single dispatcher buffer");

           const auto published_outputs = get_mla_published_outputs_contract(mpk);
           require(published_outputs.size() == 22U,
                   "MLA published outputs should preserve the unpack fanout width");
           require(published_outputs[0].size_bytes == 56160U,
                   "published sliced head should use the slice boundary size");
           require(published_outputs[1].size_bytes == 1248U,
                   "published unsliced head should use the unpack boundary size");
           require(published_outputs[21].size_bytes == 638976U,
                   "published semseg head should use the BF16 unpack size");
           // Published outputs now expose the per-head transform name as their
           // segment_name (slice/unpack child) and reset byte_offset to 0; the
           // parent MLA segment is tracked separately on the dispatcher contract.
           require(published_outputs[0].segment_name ==
                       "slice_MLA_0/tuple_get_item_0_slice_transform",
                   "packed MLA sliced head should expose the slice transform segment");
           require(published_outputs[1].segment_name == "MLA_0_ofm_unpack_transform_1",
                   "packed MLA unsliced head should expose the unpack boundary segment");
           require(published_outputs[0].byte_offset == 0,
                   "published head byte_offset should default to zero");
           require(published_outputs[1].byte_offset == 0,
                   "published head byte_offset should default to zero");
           require(published_outputs[21].byte_offset == 0,
                   "published head byte_offset should default to zero");
           const std::uint64_t published_end =
               static_cast<std::uint64_t>(published_outputs.back().byte_offset) +
               published_outputs.back().size_bytes;
           require(published_end <= 1280032U,
                   "published MLA transport spans should stay within the dispatcher buffer");
           const auto* mla_stage = get_mla_stage_io_contract(mpk);
           require(mla_stage != nullptr, "fixture should expose one MLA stage");

           const MlaStaticContract valid_contract =
               build_mla_static_contract_from_mpk_stage(*mla_stage, logical_outputs,
                                                        mla_stage->output_tensors, "MLA_0_test");
           require(valid_contract.dispatcher_physical_outputs.size() == 1U,
                   "dispatcher outputs should preserve the single raw MLA parent buffer");
           require(valid_contract.dispatcher_physical_outputs[0].size_bytes == 1280032U,
                   "dispatcher output should preserve the raw MLA parent byte size");
           require(valid_contract.physical_outputs.size() == 1U,
                   "published outputs should keep one raw parent transport segment");
           require(valid_contract.physical_outputs[0].size_bytes == 1280032U,
                   "published physical output should preserve the raw parent byte size");
           require(valid_contract.logical_outputs.size() == 22U,
                   "valid MLA static contract should preserve all logical outputs");
           require(valid_contract.logical_outputs[0].byte_offset == 0,
                   "static contract logical output byte_offset should default to zero");
           require(valid_contract.logical_outputs[1].byte_offset == 0,
                   "static contract logical output byte_offset should default to zero");
           require(valid_contract.logical_outputs[21].byte_offset == 0,
                   "static contract logical output byte_offset should default to zero");
           require(valid_contract.logical_outputs[0].segment_name ==
                       "slice_MLA_0/tuple_get_item_0_slice_transform",
                   "static contract should publish sliced heads against the slice transform segment");

           auto normalized_logical_mpk = mpk;
           normalized_logical_mpk.plugins[2].output_tensors[0].logical_shape = {12, 52, 45};
           normalized_logical_mpk.plugins[2].output_tensors[0].logical_dtype = "BF16";

           const auto normalized_logical_outputs =
               get_mla_logical_outputs_contract(normalized_logical_mpk);
           require(normalized_logical_outputs.size() == 22U,
                   "normalized logical MLA boundary outputs should still publish all heads");
           require(normalized_logical_outputs[0].mpk_shape ==
                       std::vector<std::int64_t>({12, 52, 45}),
                   "normalized logical sliced head should use the batch-stripped logical shape");
           require(normalized_logical_outputs[0].stride_bytes ==
                       std::vector<std::int64_t>({4992, 96, 2}),
                   "normalized logical sliced head should drop the leading unit-batch stride");

           const auto normalized_published_outputs =
               get_mla_published_outputs_contract(normalized_logical_mpk);
           require(normalized_published_outputs.size() == 22U,
                   "normalized logical MLA published outputs should still publish all heads");
           require(normalized_published_outputs[0].mpk_shape ==
                       std::vector<std::int64_t>({1, 12, 52, 45}),
                   "normalized logical sliced published head should keep the batched shape");
           require(normalized_published_outputs[0].stride_bytes ==
                       std::vector<std::int64_t>({59904, 4992, 96, 2}),
                   "normalized logical sliced published head should preserve batched strides");

           // Note: dispatcher-overflow rejection moved out of
           // build_mla_static_contract_from_mpk_stage; per-head byte_offset is
           // no longer tracked on logical outputs, so the previous size+offset
           // overflow check no longer applies here.

           const auto declared_size_stage = make_declared_size_mla_stage_fixture();
           std::vector<MpkTensorContract> declared_size_logical_outputs = {
               make_tensor("declared_ofm0", "BF16", {1, 24, 104, 9}, 79872U)};
           declared_size_logical_outputs[0].physical_index = 0;
           declared_size_logical_outputs[0].tensor_index = 0;

           const MlaStaticContract declared_size_contract =
               build_mla_static_contract_from_mpk_stage(declared_size_stage,
                                                        declared_size_logical_outputs,
                                                        declared_size_stage.output_tensors,
                                                        "MLA_declared_size_test");
           require(declared_size_contract.logical_outputs.size() == 1U,
                   "declared-size fixture should build one logical output");
           require(declared_size_contract.logical_outputs[0].size_bytes == 44928U,
                   "declared-size fixture should normalize logical output bytes from shape and dtype");
           require(declared_size_contract.physical_outputs.size() == 1U,
                   "declared-size fixture should build one published physical output");
           require(declared_size_contract.physical_outputs[0].size_bytes == 79872U,
                   "published physical output should preserve declared packed bytes when they exceed dense span");

         }));

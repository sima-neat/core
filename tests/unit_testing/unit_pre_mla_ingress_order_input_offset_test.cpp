#include "model_archive_test_utils.h"
#include "model/internal/ModelPack.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"
#include "test_main.h"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>

// Regression for issue #538 (RF-DETR transformer_after_gather): a multi-input
// MLA whose declared ingress order differs from the IFM-pack (MLA-boundary)
// order.
//
// The framework assembles the user's pushed TensorList into one packed
// "input_tensor" buffer in model ingress order. The pre-MLA fan-in renderer
// sequences its branches in IFM-pack order. When the two orders differ, each
// branch's read offset into the packed buffer must still be taken from where
// its own ingress tensor lands in ingress order.

namespace {

enum class BoundaryOrder {
  IngressOrder,
  Reordered,
};

enum class BranchInputBytes {
  Present,
  Missing,
};

void replace_all(std::string* text, const std::string& from, const std::string& to) {
  std::size_t pos = 0;
  while ((pos = text->find(from, pos)) != std::string::npos) {
    text->replace(pos, from.size(), to);
    pos += to.size();
  }
}

std::filesystem::path build_pack_root(BoundaryOrder boundary_order,
                                      BranchInputBytes branch_input_bytes,
                                      const std::string& suffix) {
  namespace fs = std::filesystem;
  const fs::path root =
      fs::path(sima_test::make_temp_dir("unit_pre_mla_ingress_order_" + suffix)) / "pack_root";
  std::error_code ec;
  fs::create_directories(root / "etc", ec);
  require(!ec, "failed to create synthetic pack root");

  const bool input_sizes_present = branch_input_bytes == BranchInputBytes::Present;
  const std::string big_input_size = input_sizes_present ? "64" : "0";
  const std::string small_input_size = input_sizes_present ? "16" : "0";

  std::string json = R"JSON(
{
  "name": "pre_mla_ingress_order_fixture",
  "model_path": "pre_mla_ingress_order_fixture.elf",
  "model_sdk_version": "2.0.0",
  "sequence": 1,
  "input_nodes": [
    { "name": "big",   "type": "buffer", "size": 64, "logical_shape": [1, 4, 4, 1], "logical_dtype": "FP32" },
    { "name": "small", "type": "buffer", "size": 16, "logical_shape": [1, 2, 2, 1], "logical_dtype": "FP32" }
  ],
  "plugins": [
    { "name": "quantize_0", "sequence": 1, "processor": "EV74",
      "config_params": { "kernel": "quantization_transform",
                         "params": { "input_shapes": [[1,4,4,1]], "output_shapes": [[1,4,4,1]],
                                     "input_dtype": ["FP32"], "output_dtype": "INT8",
                                     "q_scale": [255.0], "q_zp": [-128],
                                     "rounding": "TONEAREST" } },
      "input_nodes":  [{ "name": "big",        "size": __BIG_INPUT_SIZE__, "logical_shape": [1,4,4,1], "logical_dtype": "FP32" }],
      "output_nodes": [{ "name": "quantize_0", "size": 16, "logical_shape": [1,4,4,1], "logical_dtype": "INT8", "type": "buffer" }],
      "type": "sgpProcess", "resources": { "executable": "q0.elf" } },

    { "name": "tessellate_0", "sequence": 2, "processor": "EV74",
      "config_params": { "kernel": "tessellation_transform",
                         "params": { "input_shapes": [[1,4,4,1]], "output_shapes": [[1,4,4,1]],
                                     "input_dtype": ["INT8"], "output_dtype": "INT8",
                                     "slice_shape": [4,4,1], "frame_type": "INT8",
                                     "align_c16": false } },
      "input_nodes":  [{ "name": "quantize_0",   "size": 16, "logical_shape": [1,4,4,1], "logical_dtype": "INT8" }],
      "output_nodes": [{ "name": "tessellate_0", "size": 16, "logical_shape": [1,4,4,1], "logical_dtype": "INT8", "type": "buffer" }],
      "type": "sgpProcess", "resources": { "executable": "t0.elf" } },

    { "name": "quantize_1", "sequence": 3, "processor": "EV74",
      "config_params": { "kernel": "quantization_transform",
                         "params": { "input_shapes": [[1,2,2,1]], "output_shapes": [[1,2,2,1]],
                                     "input_dtype": ["FP32"], "output_dtype": "INT8",
                                     "q_scale": [255.0], "q_zp": [-128],
                                     "rounding": "TONEAREST" } },
      "input_nodes":  [{ "name": "small",      "size": __SMALL_INPUT_SIZE__, "logical_shape": [1,2,2,1], "logical_dtype": "FP32" }],
      "output_nodes": [{ "name": "quantize_1", "size":  4, "logical_shape": [1,2,2,1], "logical_dtype": "INT8", "type": "buffer" }],
      "type": "sgpProcess", "resources": { "executable": "q1.elf" } },

    { "name": "tessellate_1", "sequence": 4, "processor": "EV74",
      "config_params": { "kernel": "tessellation_transform",
                         "params": { "input_shapes": [[1,2,2,1]], "output_shapes": [[1,2,2,1]],
                                     "input_dtype": ["INT8"], "output_dtype": "INT8",
                                     "slice_shape": [2,2,1], "frame_type": "INT8",
                                     "align_c16": false } },
      "input_nodes":  [{ "name": "quantize_1",   "size": 4, "logical_shape": [1,2,2,1], "logical_dtype": "INT8" }],
      "output_nodes": [{ "name": "tessellate_1", "size": 4, "logical_shape": [1,2,2,1], "logical_dtype": "INT8", "type": "buffer" }],
      "type": "sgpProcess", "resources": { "executable": "t1.elf" } },

    { "name": "ifm_pack", "sequence": 5, "processor": "EV74",
      "config_params": { "kernel": "pack_transform",
                         "params": { "input_shapes": [__PACK_INPUT_SHAPES__],
                                     "output_shapes": [[1,2,2,5]],
                                     "input_dtype": ["INT8","INT8"], "output_dtype": "INT8" } },
      "input_nodes": [
__PACK_INPUT_NODES__
      ],
      "output_nodes": [{ "name": "ifm_pack", "size": 20, "logical_shape": [1,2,2,5], "logical_dtype": "INT8", "type": "buffer" }],
      "type": "sgpProcess", "resources": { "executable": "pack.elf" } },

    { "name": "mla_0", "sequence": 6, "processor": "MLA",
      "config_params": { "desired_batch_size": 1, "actual_batch_size": 1,
                         "number_of_quads_to_user": 1,
                         "input_shapes": [[1,2,2,5]], "input_data_type": ["INT8"],
                         "output_shapes": [[1,2,2,5]], "data_type": ["INT8"] },
      "input_nodes":  [{ "name": "ifm_pack", "size": 20, "logical_shape": [1,2,2,5], "logical_dtype": "INT8" }],
      "output_nodes": [{ "name": "mla_0",    "size": 20, "logical_shape": [1,2,2,5], "logical_dtype": "INT8", "type": "buffer" }],
      "type": "sgpProcess", "resources": { "executable": "mla.elf" } }
  ]
}
)JSON";

  replace_all(&json, "__BIG_INPUT_SIZE__", big_input_size);
  replace_all(&json, "__SMALL_INPUT_SIZE__", small_input_size);

  if (boundary_order == BoundaryOrder::Reordered) {
    replace_all(&json, "__PACK_INPUT_SHAPES__", "[1,2,2,1], [1,4,4,1]");
    replace_all(&json, "__PACK_INPUT_NODES__", R"JSON(
        { "name": "tessellate_1", "size":  4, "logical_shape": [1,2,2,1], "logical_dtype": "INT8" },
        { "name": "tessellate_0", "size": 16, "logical_shape": [1,4,4,1], "logical_dtype": "INT8" }
)JSON");
  } else {
    replace_all(&json, "__PACK_INPUT_SHAPES__", "[1,4,4,1], [1,2,2,1]");
    replace_all(&json, "__PACK_INPUT_NODES__", R"JSON(
        { "name": "tessellate_0", "size": 16, "logical_shape": [1,4,4,1], "logical_dtype": "INT8" },
        { "name": "tessellate_1", "size":  4, "logical_shape": [1,2,2,1], "logical_dtype": "INT8" }
)JSON");
  }

  const fs::path json_path = root / "etc" / "pre_mla_ingress_order_fixture_mpk.json";
  std::ofstream out(json_path);
  require(out.is_open(), "failed to open synthetic mpk json for write");
  out << json;
  out.close();
  require(out.good(), "failed to finalize synthetic mpk json");
  return fs::absolute(root);
}

auto build_quanttess_contract_from_root(const std::filesystem::path& root,
                                        const std::string& label) {
  namespace fs = std::filesystem;
  using simaai::neat::pipeline_internal::sima::load_mpk_contract_from_pack_root;
  using simaai::neat::pipeline_internal::sima::stagesemantics::
      build_processcvu_mpk_compiled_contract_for_stage_kind;
  using ExecutionStageKind = ::simaai::neat::internal::ExecutionStageKind;

  require(fs::exists(root), label + ": pack root does not exist: " + root.string());

  std::string error;
  const auto contract = load_mpk_contract_from_pack_root(root.string(), &error);
  require(contract.has_value(), label + ": failed to load mpk contract: " + error);

  return build_processcvu_mpk_compiled_contract_for_stage_kind(*contract,
                                                               ExecutionStageKind::QuantTess);
}

void require_input_offsets(const std::filesystem::path& root, std::uint64_t first_offset,
                           std::uint64_t second_offset, const std::string& label) {
  const auto compiled = build_quanttess_contract_from_root(root, label);

  require(compiled.payload.num_in_tensor == 2,
          label + ": expected num_in_tensor=2 from 2-branch fan-in render, got " +
              std::to_string(compiled.payload.num_in_tensor));
  require(compiled.payload.input_tensors.size() == 2,
          label + ": expected 2 packed input descriptors");

  const std::uint64_t actual_first_offset = compiled.payload.input_tensors[0].storage.addr;
  const std::uint64_t actual_second_offset = compiled.payload.input_tensors[1].storage.addr;
  require(actual_first_offset == first_offset, label + ": first branch offset mismatch (expected " +
                                                   std::to_string(first_offset) + ", got " +
                                                   std::to_string(actual_first_offset) + ")");
  require(actual_second_offset == second_offset,
          label + ": second branch offset mismatch (expected " + std::to_string(second_offset) +
              ", got " + std::to_string(actual_second_offset) + ")");
}

void require_compile_error_contains(const std::filesystem::path& root,
                                    const std::string& expected_fragment,
                                    const std::string& label) {
  bool threw = false;
  std::string what;
  try {
    (void)build_quanttess_contract_from_root(root, label);
  } catch (const std::exception& e) {
    threw = true;
    what = e.what();
  }
  require(threw, label + ": expected graph build to fail");
  require_contains(what, expected_fragment, label + ": unexpected graph build error");
}

} // namespace

RUN_TEST("unit_pre_mla_ingress_order_input_offset_test", ([] {
           const auto reordered_resolved = build_pack_root(
               BoundaryOrder::Reordered, BranchInputBytes::Present, "reordered_resolved");
           require_input_offsets(reordered_resolved, 64U, 0U, "reordered resolved route");

           const auto same_order_unresolved = build_pack_root(
               BoundaryOrder::IngressOrder, BranchInputBytes::Missing, "same_order_unresolved");
           require_input_offsets(same_order_unresolved, 0U, 64U, "same-order unresolved route");

           const auto reordered_unresolved = build_pack_root(
               BoundaryOrder::Reordered, BranchInputBytes::Missing, "reordered_unresolved");
           require_compile_error_contains(reordered_unresolved,
                                          "could not resolve every branch to public ingress order",
                                          "reordered unresolved route");
         }));

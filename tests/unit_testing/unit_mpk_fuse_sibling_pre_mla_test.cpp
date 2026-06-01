#include "model_archive_test_utils.h"
#include "model/internal/ModelPack.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"
#include "pipeline/internal/contract/PluginCompiledContracts.h"
#include "test_main.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

// Pre-MLA multi-IO fan-in verification.
//
// Synthesizes a minimal 3-branch pre-MLA topology that mirrors
// rpn_head_640_640_concat_4d:
//
//   p1 -> quantize_0 -> tessellate_0 \
//   p2 -> quantize_1 -> tessellate_1  -> ifm_pack -> MLA
//   p3 -> quantize_2 -> tessellate_2 /
//
// Asserts that the QuantTess pre-adapter renderer, called for the whole
// contract without an exact_stage_name_or_id, produces a multi-IO payload
// with num_in_tensor == 3 — i.e. a single fused multi-input quanttess
// stage, the fan-in analog of the post-MLA detessdequant fan-out.

namespace {

std::filesystem::path build_synthetic_pack_root() {
  namespace fs = std::filesystem;
  const fs::path root =
      fs::path(sima_test::make_temp_dir("unit_mpk_fuse_sibling_pre_mla")) / "pack_root";
  std::error_code ec;
  fs::create_directories(root / "etc", ec);
  require(!ec, "failed to create synthetic pack root");

  const fs::path json_path = root / "etc" / "fuse_sibling_pre_mla_fixture_mpk.json";
  std::ofstream out(json_path);
  require(out.is_open(), "failed to open synthetic mpk json for write");
  out << R"JSON(
{
  "name": "fuse_sibling_pre_mla_fixture",
  "model_path": "fuse_sibling_pre_mla_fixture.elf",
  "model_sdk_version": "2.0.0",
  "sequence": 1,
  "input_nodes": [
    { "name": "p1", "type": "buffer", "size": 16, "logical_shape": [1, 2, 2, 1], "logical_dtype": "FP32" },
    { "name": "p2", "type": "buffer", "size": 16, "logical_shape": [1, 2, 2, 1], "logical_dtype": "FP32" },
    { "name": "p3", "type": "buffer", "size": 16, "logical_shape": [1, 2, 2, 1], "logical_dtype": "FP32" }
  ],
  "plugins": [
    { "name": "quantize_0", "sequence": 1, "processor": "EV74",
      "config_params": { "kernel": "quantization_transform",
                         "params": { "input_shapes": [[1,2,2,1]], "output_shapes": [[1,2,2,1]],
                                     "input_dtype": ["FP32"], "output_dtype": "INT8",
                                     "q_scale": [255.0], "q_zp": [-128],
                                     "rounding": "TONEAREST" } },
      "input_nodes":  [{ "name": "p1",         "size": 16, "logical_shape": [1,2,2,1], "logical_dtype": "FP32" }],
      "output_nodes": [{ "name": "quantize_0", "size":  4, "logical_shape": [1,2,2,1], "logical_dtype": "INT8", "type": "buffer" }],
      "type": "sgpProcess", "resources": { "executable": "q0.elf" } },

    { "name": "tessellate_0", "sequence": 2, "processor": "EV74",
      "config_params": { "kernel": "tessellation_transform",
                         "params": { "input_shapes": [[1,2,2,1]], "output_shapes": [[1,2,2,1]],
                                     "input_dtype": ["INT8"], "output_dtype": "INT8",
                                     "slice_shape": [2,2,1], "frame_type": "INT8",
                                     "align_c16": false } },
      "input_nodes":  [{ "name": "quantize_0",   "size": 4, "logical_shape": [1,2,2,1], "logical_dtype": "INT8" }],
      "output_nodes": [{ "name": "tessellate_0", "size": 4, "logical_shape": [1,2,2,1], "logical_dtype": "INT8", "type": "buffer" }],
      "type": "sgpProcess", "resources": { "executable": "t0.elf" } },

    { "name": "quantize_1", "sequence": 3, "processor": "EV74",
      "config_params": { "kernel": "quantization_transform",
                         "params": { "input_shapes": [[1,2,2,1]], "output_shapes": [[1,2,2,1]],
                                     "input_dtype": ["FP32"], "output_dtype": "INT8",
                                     "q_scale": [255.0], "q_zp": [-128],
                                     "rounding": "TONEAREST" } },
      "input_nodes":  [{ "name": "p2",         "size": 16, "logical_shape": [1,2,2,1], "logical_dtype": "FP32" }],
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

    { "name": "quantize_2", "sequence": 5, "processor": "EV74",
      "config_params": { "kernel": "quantization_transform",
                         "params": { "input_shapes": [[1,2,2,1]], "output_shapes": [[1,2,2,1]],
                                     "input_dtype": ["FP32"], "output_dtype": "INT8",
                                     "q_scale": [255.0], "q_zp": [-128],
                                     "rounding": "TONEAREST" } },
      "input_nodes":  [{ "name": "p3",         "size": 16, "logical_shape": [1,2,2,1], "logical_dtype": "FP32" }],
      "output_nodes": [{ "name": "quantize_2", "size":  4, "logical_shape": [1,2,2,1], "logical_dtype": "INT8", "type": "buffer" }],
      "type": "sgpProcess", "resources": { "executable": "q2.elf" } },

    { "name": "tessellate_2", "sequence": 6, "processor": "EV74",
      "config_params": { "kernel": "tessellation_transform",
                         "params": { "input_shapes": [[1,2,2,1]], "output_shapes": [[1,2,2,1]],
                                     "input_dtype": ["INT8"], "output_dtype": "INT8",
                                     "slice_shape": [2,2,1], "frame_type": "INT8",
                                     "align_c16": false } },
      "input_nodes":  [{ "name": "quantize_2",   "size": 4, "logical_shape": [1,2,2,1], "logical_dtype": "INT8" }],
      "output_nodes": [{ "name": "tessellate_2", "size": 4, "logical_shape": [1,2,2,1], "logical_dtype": "INT8", "type": "buffer" }],
      "type": "sgpProcess", "resources": { "executable": "t2.elf" } },

    { "name": "ifm_pack", "sequence": 7, "processor": "EV74",
      "config_params": { "kernel": "pack_transform",
                         "params": { "input_shapes": [[1,2,2,1],[1,2,2,1],[1,2,2,1]],
                                     "output_shapes": [[1,2,2,3]],
                                     "input_dtype": ["INT8","INT8","INT8"], "output_dtype": "INT8" } },
      "input_nodes": [
        { "name": "tessellate_0", "size": 4, "logical_shape": [1,2,2,1], "logical_dtype": "INT8" },
        { "name": "tessellate_1", "size": 4, "logical_shape": [1,2,2,1], "logical_dtype": "INT8" },
        { "name": "tessellate_2", "size": 4, "logical_shape": [1,2,2,1], "logical_dtype": "INT8" }
      ],
      "output_nodes": [{ "name": "ifm_pack", "size": 12, "logical_shape": [1,2,2,3], "logical_dtype": "INT8", "type": "buffer" }],
      "type": "sgpProcess", "resources": { "executable": "pack.elf" } },

    { "name": "mla_0", "sequence": 8, "processor": "MLA",
      "config_params": { "desired_batch_size": 1, "actual_batch_size": 1,
                         "number_of_quads_to_user": 1,
                         "input_shapes": [[1,2,2,3]], "input_data_type": ["INT8"],
                         "output_shapes": [[1,2,2,3]], "data_type": ["INT8"] },
      "input_nodes":  [{ "name": "ifm_pack", "size": 12, "logical_shape": [1,2,2,3], "logical_dtype": "INT8" }],
      "output_nodes": [{ "name": "mla_0",    "size": 12, "logical_shape": [1,2,2,3], "logical_dtype": "INT8", "type": "buffer" }],
      "type": "sgpProcess", "resources": { "executable": "mla.elf" } }
  ]
}
)JSON";
  out.close();
  require(out.good(), "failed to finalize synthetic mpk json");
  return std::filesystem::absolute(root);
}

} // namespace

RUN_TEST("unit_mpk_fuse_sibling_pre_mla_test", ([] {
           namespace fs = std::filesystem;
           using simaai::neat::pipeline_internal::sima::load_mpk_contract_from_pack_root;
           using simaai::neat::pipeline_internal::sima::stagesemantics::
               build_processcvu_mpk_compiled_contract_for_stage_kind;
           using ExecutionStageKind = ::simaai::neat::internal::ExecutionStageKind;

           const fs::path root = build_synthetic_pack_root();
           require(fs::exists(root), "pack root does not exist: " + root.string());

           std::string error;
           const auto contract = load_mpk_contract_from_pack_root(root.string(), &error);
           require(contract.has_value(), "failed to load mpk contract: " + error);

           const auto compiled = build_processcvu_mpk_compiled_contract_for_stage_kind(
               *contract, ExecutionStageKind::QuantTess);

           require(compiled.payload.num_in_tensor == 3,
                   "expected num_in_tensor=3 from 3-branch fan-in render, got " +
                       std::to_string(compiled.payload.num_in_tensor));
         }));

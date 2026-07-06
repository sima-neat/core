#include "model_archive_test_utils.h"
#include "model/internal/ModelPack.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"
#include "test_main.h"

#include <cstdint>
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
// its *own* ingress tensor lands in ingress order — otherwise every branch
// reads the wrong region (feature bytes cast/tessellated as gather and vice
// versa) and the MLA receives a scrambled IFM.
//
// Topology (ingress order [big, small], but pack consumes [small, big]):
//
//   big(64B)   -> quantize_0 -> tessellate_0 \
//                                             +-> ifm_pack[small, big] -> MLA
//   small(16B) -> quantize_1 -> tessellate_1 /
//
// After the fix, the packed-input read offset of the branch that consumes
// `small` must be 64 (it sits *after* `big` in ingress order) and the branch
// that consumes `big` must be 0 — not the sequential 0 / 16 the boundary-order
// accumulation would produce.

namespace {

std::filesystem::path build_ingress_reordered_pack_root() {
  namespace fs = std::filesystem;
  const fs::path root =
      fs::path(sima_test::make_temp_dir("unit_pre_mla_ingress_order")) / "pack_root";
  std::error_code ec;
  fs::create_directories(root / "etc", ec);
  require(!ec, "failed to create synthetic pack root");

  const fs::path json_path = root / "etc" / "pre_mla_ingress_order_fixture_mpk.json";
  std::ofstream out(json_path);
  require(out.is_open(), "failed to open synthetic mpk json for write");
  out << R"JSON(
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
      "input_nodes":  [{ "name": "big",        "size": 64, "logical_shape": [1,4,4,1], "logical_dtype": "FP32" }],
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
      "input_nodes":  [{ "name": "small",      "size": 16, "logical_shape": [1,2,2,1], "logical_dtype": "FP32" }],
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
                         "params": { "input_shapes": [[1,2,2,1],[1,4,4,1]],
                                     "output_shapes": [[1,2,2,5]],
                                     "input_dtype": ["INT8","INT8"], "output_dtype": "INT8" } },
      "input_nodes": [
        { "name": "tessellate_1", "size":  4, "logical_shape": [1,2,2,1], "logical_dtype": "INT8" },
        { "name": "tessellate_0", "size": 16, "logical_shape": [1,4,4,1], "logical_dtype": "INT8" }
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
  out.close();
  require(out.good(), "failed to finalize synthetic mpk json");
  return std::filesystem::absolute(root);
}

} // namespace

RUN_TEST("unit_pre_mla_ingress_order_input_offset_test", ([] {
           namespace fs = std::filesystem;
           using simaai::neat::pipeline_internal::sima::load_mpk_contract_from_pack_root;
           using simaai::neat::pipeline_internal::sima::stagesemantics::
               build_processcvu_mpk_compiled_contract_for_stage_kind;
           using ExecutionStageKind = ::simaai::neat::internal::ExecutionStageKind;

           const fs::path root = build_ingress_reordered_pack_root();
           require(fs::exists(root), "pack root does not exist: " + root.string());

           std::string error;
           const auto contract = load_mpk_contract_from_pack_root(root.string(), &error);
           require(contract.has_value(), "failed to load mpk contract: " + error);

           const auto compiled = build_processcvu_mpk_compiled_contract_for_stage_kind(
               *contract, ExecutionStageKind::QuantTess);

           require(compiled.payload.num_in_tensor == 2,
                   "expected num_in_tensor=2 from 2-branch fan-in render, got " +
                       std::to_string(compiled.payload.num_in_tensor));
           require(compiled.payload.input_tensors.size() == 2,
                   "expected 2 packed input descriptors");

           // Branches are sequenced in IFM-pack order [small, big]. Each branch
           // must read from where its *own* ingress tensor lands in the ingress
           // buffer laid out in declared order [big(64B), small(16B)]:
           //   branch 0 (small) -> offset 64  (small sits after big)
           //   branch 1 (big)   -> offset 0
           const std::uint64_t small_branch_offset = compiled.payload.input_tensors[0].storage.addr;
           const std::uint64_t big_branch_offset = compiled.payload.input_tensors[1].storage.addr;
           require(small_branch_offset == 64U,
                   "packed-input read offset for the 'small' branch must follow ingress order "
                   "(expected 64, got " +
                       std::to_string(small_branch_offset) + ")");
           require(big_branch_offset == 0U,
                   "packed-input read offset for the 'big' branch must follow ingress order "
                   "(expected 0, got " +
                       std::to_string(big_branch_offset) + ")");
         }));

#include "model_archive_test_utils.h"
#include "model/internal/ModelPack.h"
#include "pipeline/internal/contract/PluginCompiledContracts.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"
#include "test_main.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

// Regression for compiler-batched multi-input models.  The compiler's IFM
// pack boundary is sample-major:
//
//   a[0], b[0], a[1], b[1]
//
// The fused cast adapter must therefore use per-frame descriptors with the
// complete packed frame as the output batch stride.  Concatenating complete
// branch tensors instead produces a[0], a[1], b[0], b[1] and causes MLA batch
// rows to consume data from other samples.

namespace {

std::filesystem::path build_batch2_cast_pack_root() {
  namespace fs = std::filesystem;
  const fs::path root =
      fs::path(sima_test::make_temp_dir("unit_batch2_multi_input_cast_pack")) / "pack_root";
  std::error_code ec;
  fs::create_directories(root / "etc", ec);
  require(!ec, "failed to create synthetic pack root");

  const fs::path json_path = root / "etc" / "batch2_multi_input_cast_pack_mpk.json";
  std::ofstream out(json_path);
  require(out.is_open(), "failed to open synthetic mpk json for write");
  out << R"JSON(
{
  "name": "batch2_multi_input_cast_pack",
  "model_path": "batch2_multi_input_cast_pack.elf",
  "model_sdk_version": "2.1.3",
  "sequence": 1,
  "input_nodes": [
    { "name": "a", "type": "buffer", "size": 1536,
      "logical_shape": [2,1,192,1], "logical_dtype": "FP32" },
    { "name": "b", "type": "buffer", "size": 1536,
      "logical_shape": [2,1,192,1], "logical_dtype": "FP32" }
  ],
  "plugins": [
    { "name": "cast_0", "sequence": 1, "processor": "EV74",
      "config_params": { "desired_batch_size": 2, "actual_batch_size": 2,
        "kernel": "cast_transform", "params": {
          "in_dtype": "float32", "out_dtype": "bfloat16",
          "input_shapes": [[2,1,192,1]], "output_shapes": [[2,1,192,1]] } },
      "input_nodes":  [{ "name": "a", "size": 1536,
                          "logical_shape": [2,1,192,1], "logical_dtype": "FP32" }],
      "output_nodes": [{ "name": "cast_0", "type": "buffer", "size": 768,
                          "logical_shape": [2,1,192,1], "logical_dtype": "BF16" }],
      "type": "sgpProcess", "resources": { "executable": "kernel_name_tbd" } },

    { "name": "cast_1", "sequence": 2, "processor": "EV74",
      "config_params": { "desired_batch_size": 2, "actual_batch_size": 2,
        "kernel": "cast_transform", "params": {
          "in_dtype": "float32", "out_dtype": "bfloat16",
          "input_shapes": [[2,1,192,1]], "output_shapes": [[2,1,192,1]] } },
      "input_nodes":  [{ "name": "b", "size": 1536,
                          "logical_shape": [2,1,192,1], "logical_dtype": "FP32" }],
      "output_nodes": [{ "name": "cast_1", "type": "buffer", "size": 768,
                          "logical_shape": [2,1,192,1], "logical_dtype": "BF16" }],
      "type": "sgpProcess", "resources": { "executable": "kernel_name_tbd" } },

    { "name": "MLA_0_ifm_pack_transform", "sequence": 3, "processor": "EV74",
      "config_params": { "desired_batch_size": 2, "actual_batch_size": 2,
        "kernel": "pack_transform", "params": {
          "input_shapes": [[2,1,192,1],[2,1,192,1]],
          "output_shapes": [[2,1536]] } },
      "input_nodes": [
        { "name": "cast_0",
          "logical_shape": [2,1,192,1], "logical_dtype": "BF16" },
        { "name": "cast_1",
          "logical_shape": [2,1,192,1], "logical_dtype": "BF16" }
      ],
      "output_nodes": [{ "name": "MLA_0_ifm_pack_transform", "type": "buffer", "size": 3072,
                          "logical_shape": [2,1536], "logical_dtype": "BF16" }],
      "type": "sgpProcess", "resources": { "executable": "kernel_name_tbd" } },

    { "name": "MLA_0", "sequence": 4, "processor": "MLA",
      "config_params": { "desired_batch_size": 2, "actual_batch_size": 2,
                          "number_of_quads_to_user": 4 },
      "input_nodes": [{ "name": "MLA_0_ifm_pack_transform", "size": 3072,
                         "logical_shape": [2,1536], "logical_dtype": "BF16" }],
      "output_nodes": [{ "name": "MLA_0", "type": "buffer", "size": 12288,
                          "logical_shape": [2,1,192,16], "logical_dtype": "BF16" }],
      "type": "sgpProcess", "resources": { "executable": "model.elf" } }
  ]
}
)JSON";
  out.close();
  require(out.good(), "failed to finalize synthetic mpk json");
  return fs::absolute(root);
}

} // namespace

RUN_TEST("unit_batch2_multi_input_cast_pack_test", ([] {
           namespace fs = std::filesystem;
           using simaai::neat::pipeline_internal::sima::load_mpk_contract_from_pack_root;
           using simaai::neat::pipeline_internal::sima::stagesemantics::
               build_processcvu_mpk_compiled_contract_for_stage_kind;
           using ExecutionStageKind = ::simaai::neat::internal::ExecutionStageKind;

           const fs::path root = build_batch2_cast_pack_root();
           std::string error;
           const auto contract = load_mpk_contract_from_pack_root(root.string(), &error);
           require(contract.has_value(), "failed to load batch-2 cast-pack contract: " + error);

           const auto compiled = build_processcvu_mpk_compiled_contract_for_stage_kind(
               *contract, ExecutionStageKind::Cast);
           const auto& payload = compiled.payload;

           require(payload.batch_size == 2, "cast payload did not preserve compiler batch size");
           require(payload.input_tensors.size() == 2U, "expected two cast input descriptors");
           require(payload.output_tensors.size() == 2U, "expected two cast output descriptors");
           require(compiled.runtime_contract.physical_outputs.size() == 1U,
                   "expected one packed physical output");
           require(compiled.runtime_contract.physical_outputs.front().size_bytes == 1536U,
                   "packed physical output must span every batch row");

           for (std::size_t i = 0; i < 2U; ++i) {
             require(payload.input_tensors[i].shape.rank == 3U,
                     "cast input descriptor must describe one batch frame");
             require(payload.output_tensors[i].shape.rank == 3U,
                     "cast output descriptor must describe one batch frame");
             require(payload.input_tensors[i].storage.nbytes == 768U,
                     "cast input batch stride must equal one FP32 frame, got " +
                         std::to_string(payload.input_tensors[i].storage.nbytes));
             require(payload.output_tensors[i].storage.nbytes == 768U,
                     "cast output batch stride must equal one complete packed BF16 frame, got " +
                         std::to_string(payload.output_tensors[i].storage.nbytes));
           }

           require(payload.input_tensors[0].storage.addr == 0U,
                   "first cast input must start at the first public tensor");
           require(payload.input_tensors[1].storage.addr == 1536U,
                   "second cast input must start after the complete first public tensor");
           require(payload.output_tensors[0].storage.addr == 0U,
                   "first cast output must start at the beginning of each packed frame");
           require(payload.output_tensors[1].storage.addr == 384U,
                   "second cast output must follow the first output inside each packed frame");
         }));

#include "mpk_test_utils.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "test_main.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::filesystem::path resolve_pack_root() {
  namespace fs = std::filesystem;

  const char* raw = std::getenv("SIMA_MPK_SINGLE_PACK_ROOT");
  if (raw && *raw) {
    return fs::absolute(fs::path(raw));
  }

  const fs::path root = fs::path(sima_test::make_temp_dir("unit_mpk_graph_dump")) / "pack_root";
  std::error_code ec;
  fs::create_directories(root / "etc", ec);
  require(!ec, "failed to create synthetic pack root");

  const fs::path json_path = root / "etc" / "graph_dump_fixture_mpk.json";
  std::ofstream out(json_path);
  require(out.is_open(), "failed to open synthetic mpk json for write");
  out << R"JSON(
{
  "name": "graph_dump_fixture",
  "model_path": "graph_dump_fixture.elf",
  "model_sdk_version": "2.0.0",
  "sequence": 1,
  "input_nodes": [
    {
      "name": "images",
      "type": "buffer",
      "size": 1228800,
      "logical_shape": [1, 640, 640, 3],
      "logical_dtype": "UINT8"
    }
  ],
  "plugins": [
    {
      "name": "preproc_0",
      "sequence": 1,
      "processor": "EV74",
      "config_params": {
        "kernel": "preproc",
        "params": {
          "input_shapes": [[1, 640, 640, 3]],
          "output_shapes": [[1, 640, 640, 3]],
          "input_dtype": ["UINT8"],
          "output_dtype": "BF16"
        }
      },
      "input_nodes": [
        {
          "name": "images",
          "size": 1228800,
          "logical_shape": [1, 640, 640, 3],
          "logical_dtype": "UINT8"
        }
      ],
      "output_nodes": [
        {
          "name": "preproc_0",
          "type": "buffer",
          "size": 2457600,
          "logical_shape": [1, 640, 640, 3],
          "logical_dtype": "BF16"
        }
      ],
      "type": "sgpProcess",
      "resources": { "executable": "preproc.elf" }
    },
    {
      "name": "mla_0",
      "sequence": 2,
      "processor": "MLA",
      "config_params": {
        "desired_batch_size": 1,
        "actual_batch_size": 1,
        "number_of_quads_to_user": 1,
        "input_shapes": [[1, 640, 640, 3]],
        "input_data_type": ["BF16"],
        "output_shapes": [[1, 80, 80, 6]],
        "data_type": ["BF16"]
      },
      "input_nodes": [
        {
          "name": "preproc_0",
          "size": 2457600,
          "logical_shape": [1, 640, 640, 3],
          "logical_dtype": "BF16"
        }
      ],
      "output_nodes": [
        {
          "name": "mla_0",
          "type": "buffer",
          "size": 153600,
          "logical_shape": [1, 80, 80, 6],
          "logical_dtype": "BF16"
        }
      ],
      "type": "sgpProcess",
      "resources": { "executable": "stage0.elf" }
    }
  ]
}
)JSON";
  out.close();
  require(out.good(), "failed to finalize synthetic mpk json");
  return fs::absolute(root);
}

} // namespace

RUN_TEST("unit_mpk_graph_dump_test", ([] {
           namespace fs = std::filesystem;
           using simaai::neat::pipeline_internal::sima::load_mpk_contract_from_pack_root;

           const fs::path root = resolve_pack_root();
           require(fs::exists(root), "pack root does not exist: " + root.string());

           std::string error;
           const auto contract = load_mpk_contract_from_pack_root(root.string(), &error);
           require(contract.has_value(), "failed to load mpk contract: " + error);
           require(!contract->graph.nodes.empty(), "graph should contain nodes");
         }));

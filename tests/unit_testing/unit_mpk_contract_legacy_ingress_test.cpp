#include "pipeline/internal/sima/MpkContract.h"
#include "test_main.h"

#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::filesystem::path make_temp_pack_root(const std::string& name) {
  const auto root = std::filesystem::temp_directory_path() / ("sima_mpk_legacy_ingress_" + name);
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  require(!ec, "failed to create temp pack root");
  return root;
}

} // namespace

RUN_TEST("unit_mpk_contract_legacy_ingress_test", ([] {
           namespace fs = std::filesystem;
           using simaai::neat::pipeline_internal::sima::load_mpk_contract_from_pack_root;

           const fs::path root = make_temp_pack_root("image_uv");
           const fs::path json_path = root / "legacy_mpk.json";

           std::ofstream out(json_path);
           require(out.is_open(), "failed to open temp mpk json for write");
           out << R"JSON(
{
  "name": "legacy_multi_input",
  "model_path": "legacy.elf",
  "input_nodes": [
    { "name": "image_l", "type": "buffer", "size": 16 },
    { "name": "image_uv", "type": "buffer", "size": 8 }
  ],
  "plugins": [
    {
      "name": "cast_0",
      "sequence": 1,
      "processor": "EV74",
      "config_params": {
        "kernel": "cast_transform",
        "params": {
          "input_shapes": [[1, 4, 4, 1]],
          "output_shapes": [[1, 4, 4, 1]],
          "out_dtype": "bfloat16"
        }
      },
      "input_nodes": [{ "name": "image_l", "size": 16 }],
      "output_nodes": [{ "name": "cast_0", "type": "buffer", "size": 32 }]
    },
    {
      "name": "cast_1",
      "sequence": 2,
      "processor": "EV74",
      "config_params": {
        "kernel": "cast_transform",
        "params": {
          "input_shapes": [[1, 2, 2, 2]],
          "output_shapes": [[1, 2, 2, 2]],
          "out_dtype": "bfloat16"
        }
      },
      "input_nodes": [{ "name": "image_uv", "size": 8 }],
      "output_nodes": [{ "name": "cast_1", "type": "buffer", "size": 16 }]
    },
    {
      "name": "pack_0",
      "sequence": 3,
      "processor": "EV74",
      "config_params": {
        "kernel": "pack_transform",
        "params": {
          "input_shapes": [[1, 32], [1, 16]],
          "output_shapes": [[1, 48]]
        }
      },
      "input_nodes": [
        { "name": "cast_0", "size": 32 },
        { "name": "cast_1", "size": 16 }
      ],
      "output_nodes": [{ "name": "pack_0", "type": "buffer", "size": 48 }]
    }
  ]
}
)JSON";
           out.close();

           std::string error;
           const auto contract = load_mpk_contract_from_pack_root(root.string(), &error);
           require(contract.has_value(),
                   "legacy multi-ingress mpk should parse successfully: " + error);
           require(contract->ingress_tensors.size() == 2U,
                   "legacy mpk should preserve both declared ingress tensors");
           require(contract->edges.size() == 2U,
                   "legacy mpk should resolve only intra-graph edges");
           require(contract->edges[0].tensor_name == "cast_0",
                   "first resolved edge should connect cast_0 to pack");
           require(contract->edges[1].tensor_name == "cast_1",
                   "second resolved edge should connect cast_1 to pack");
         }));

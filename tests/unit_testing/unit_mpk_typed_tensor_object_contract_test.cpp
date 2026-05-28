#include "pipeline/internal/sima/MpkContract.h"
#include "test_main.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::filesystem::path make_temp_pack_root(const std::string& name) {
  const auto root = std::filesystem::temp_directory_path() / ("sima_mpk_typed_tensor_" + name);
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  require(!ec, "failed to create temp pack root");
  return root;
}

} // namespace

RUN_TEST("unit_mpk_typed_tensor_object_contract_test", ([] {
           namespace fs = std::filesystem;
           using simaai::neat::pipeline_internal::sima::load_mpk_contract_from_pack_root;

           const fs::path root = make_temp_pack_root("apu_output_types");
           const fs::path json_path = root / "typed_object_mpk.json";

           std::ofstream out(json_path);
           require(out.is_open(), "failed to open temp mpk json for write");
           out << R"JSON(
{
  "name": "typed_object_mpk",
  "model_path": "typed.elf",
  "input_nodes": [
    { "name": "MLA_0", "type": "buffer", "size": 3145728 }
  ],
  "plugins": [
    {
      "name": "APU_1",
      "sequence": 1,
      "processor": "A65",
      "config_params": {
        "kernel": "processtvm",
        "params": {
          "input_types": [
            { "scalar": "int32", "shape": [1, 768, 1024, 1] }
          ],
          "output_types": [
            { "scalar": "int64", "shape": [1, 768, 1024, 1] }
          ]
        }
      },
      "input_nodes": [{ "name": "MLA_0", "type": "buffer", "size": 3145728 }],
      "output_nodes": [{ "name": "APU_1/output_", "type": "buffer", "size": 6291456 }]
    }
  ]
}
)JSON";
           out.close();

           std::string error;
           const auto contract = load_mpk_contract_from_pack_root(root.string(), &error);
           require(contract.has_value(),
                   "typed-object mpk should parse successfully: " + error);
           require(contract->plugins.size() == 1U, "expected one parsed plugin");
           const auto& stage = contract->plugins.front();
           require(stage.input_tensors.size() == 1U, "expected one parsed input tensor");
           require(stage.output_tensors.size() == 1U, "expected one parsed output tensor");

           const std::vector<std::int64_t> expected_shape{1, 768, 1024, 1};
           require(stage.input_tensors.front().dtype == "INT32",
                   "typed input dtype should come from scalar=int32 object");
           require(stage.input_tensors.front().mpk_shape == expected_shape,
                   "typed input shape should come from input_types object");
           require(stage.output_tensors.front().dtype == "INT64",
                   "typed output dtype should come from scalar=int64 object, not FP32 size inference");
           require(stage.output_tensors.front().mpk_shape == expected_shape,
                   "typed output shape should come from output_types object");
           require(stage.output_tensors.front().size_bytes == 6291456U,
                   "typed output should preserve physical byte size from output node");
         }));

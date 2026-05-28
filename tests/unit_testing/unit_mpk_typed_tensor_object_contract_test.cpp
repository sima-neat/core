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
           using simaai::neat::pipeline_internal::sima::DTypeSource;
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
           require(stage.input_tensors.front().dtype_source == DTypeSource::TypedObject,
                   "typed input dtype source should be tracked as typed object");
           require(stage.input_tensors.front().mpk_shape == expected_shape,
                   "typed input shape should come from input_types object");
           require(stage.output_tensors.front().dtype == "INT64",
                   "typed output dtype should come from scalar=int64 object, not FP32 size inference");
           require(stage.output_tensors.front().dtype_source == DTypeSource::TypedObject,
                   "typed output dtype source should be tracked as typed object");
           require(stage.output_tensors.front().mpk_shape == expected_shape,
                   "typed output shape should come from output_types object");
           require(stage.output_tensors.front().size_bytes == 6291456U,
                   "typed output should preserve physical byte size from output node");

           const fs::path inferred_root = make_temp_pack_root("inferred_dtype");
           const fs::path inferred_json_path = inferred_root / "inferred_mpk.json";
           std::ofstream inferred_out(inferred_json_path);
           require(inferred_out.is_open(), "failed to open inferred temp mpk json for write");
           inferred_out << R"JSON(
{
  "name": "inferred_mpk",
  "model_path": "typed.elf",
  "plugins": [
    {
      "name": "APU_1",
      "sequence": 1,
      "processor": "A65",
      "config_params": {
        "kernel": "slice",
        "params": {
          "input_shape": [[1, 2, 2, 1]],
          "output_shape": [[1, 2, 2, 1]],
          "begin": [0, 0, 0, 0],
          "end": [1, 2, 2, 1]
        }
      },
      "input_nodes": [{ "name": "APU_1/input_", "type": "buffer", "size": 16 }],
      "output_nodes": [{ "name": "APU_1/output_", "type": "buffer", "size": 16 }]
    }
  ]
}
)JSON";
           inferred_out.close();

           std::string inferred_error;
           const auto inferred_contract =
               load_mpk_contract_from_pack_root(inferred_root.string(), &inferred_error);
           require(inferred_contract.has_value(),
                   "inferred-dtype mpk should parse successfully: " + inferred_error);
           require(inferred_contract->plugins.size() == 1U, "expected one inferred plugin");
           const auto& inferred_stage = inferred_contract->plugins.front();
           require(inferred_stage.output_tensors.size() == 1U,
                   "expected one inferred output tensor");
           require(inferred_stage.output_tensors.front().dtype == "FP32",
                   "legacy inference still infers FP32 internally for now");
           require(inferred_stage.output_tensors.front().dtype_source ==
                       DTypeSource::InferredFromSize,
                   "element-size-only FP32 inference must be marked as inferred");
         }));

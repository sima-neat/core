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

RUN_TEST(
    "unit_mpk_typed_tensor_object_contract_test", ([] {
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
      require(contract.has_value(), "typed-object mpk should parse successfully: " + error);
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
      require(inferred_stage.output_tensors.size() == 1U, "expected one inferred output tensor");
      require(inferred_stage.output_tensors.front().dtype == "FP32",
              "legacy inference still infers FP32 internally for now");
      require(inferred_stage.output_tensors.front().dtype_source == DTypeSource::InferredFromSize,
              "element-size-only FP32 inference must be marked as inferred");

      const fs::path semantic_alias_root = make_temp_pack_root("semantic_transport_alias");
      std::ofstream semantic_alias_out(semantic_alias_root / "semantic_alias_mpk.json");
      require(semantic_alias_out.is_open(), "failed to open semantic-alias mpk json");
      semantic_alias_out << R"JSON(
{
  "name": "semantic_alias_mpk",
  "model_path": "semantic_alias.elf",
  "plugins": [
    {
      "name": "MLA_0_ofm_unpack_transform",
      "sequence": 1,
      "processor": "EV74",
      "config_params": {
        "kernel": "unpack_transform",
        "params": {
          "input_shapes": [[1, 16]],
          "output_shapes": [[1, 1, 1, 4]],
          "tensor_types": ["float32"]
        }
      },
      "input_nodes": [{ "name": "image", "type": "buffer", "size": 16 }],
      "output_nodes": [{ "name": "unpack_out_0", "type": "buffer", "size": 16 }]
    },
    {
      "name": "PassThrough",
      "sequence": 2,
      "processor": "EV74",
      "config_params": { "kernel": "pass_through", "params": {} },
      "input_nodes": [
        { "name": "unpack_out_0//convDb/Conv_output_0", "type": "buffer", "size": 16,
          "dtype": "float32", "shape": [1, 1, 1, 4] }
      ],
      "output_nodes": [
        { "name": "pass_through_out_0", "type": "buffer", "size": 16,
          "dtype": "float32", "shape": [1, 1, 1, 4] }
      ]
    }
  ]
}
)JSON";
      semantic_alias_out.close();
      std::string semantic_alias_error;
      const auto semantic_alias_contract =
          load_mpk_contract_from_pack_root(semantic_alias_root.string(), &semantic_alias_error);
      require(semantic_alias_contract.has_value(),
              "AFE semantic tensor decoration should resolve to its exact transport producer: " +
                  semantic_alias_error);
      require(semantic_alias_contract->edges.size() == 1U,
              "semantic transport alias should produce exactly one strict edge");
      require(semantic_alias_contract->edges.front().src_plugin == "MLA_0_ofm_unpack_transform" &&
                  semantic_alias_contract->edges.front().dst_plugin == "PassThrough" &&
                  semantic_alias_contract->edges.front().src_output_index == 0 &&
                  semantic_alias_contract->edges.front().dst_input_index == 0,
              "semantic transport alias resolved to the wrong producer or binding");
      require(semantic_alias_contract->plugins.front().input_tensors.front().shape_semantics ==
                      simaai::neat::pipeline_internal::sima::MpkShapeSemantics::PackedExtent &&
                  semantic_alias_contract->plugins.front().output_tensors.front().shape_semantics ==
                      simaai::neat::pipeline_internal::sima::MpkShapeSemantics::Geometry,
              "unpack transforms must classify packed inputs and dense geometric outputs");

      const fs::path superpoint_root = make_temp_pack_root("superpoint_schema_v1");
      const fs::path superpoint_json_path = superpoint_root / "superpoint_mpk.json";
      std::ofstream superpoint_out(superpoint_json_path);
      require(superpoint_out.is_open(), "failed to open SuperPoint temp mpk json for write");
      superpoint_out << R"JSON(
{
  "name": "superpoint_mpk",
  "model_path": "superpoint.elf",
  "plugins": [
    {
      "name": "boxdecode_0",
      "sequence": 1,
      "processor": "A65",
      "config_params": {
        "kernel": "boxdecode",
        "params": {
          "decode_type": "superpoint",
          "superpoint": {
            "schema_version": 1,
            "profile": "lightglue-v1",
            "profile_fingerprint": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "output_format": "feature-points-v1",
            "detector_tensor_id": "semi",
            "descriptor_tensor_id": "desc",
            "detector_representation": "raw-logits-65",
            "descriptor_representation": "coarse-pre-l2",
            "descriptor_output_dtype": "BF16",
            "nms_radius": 0,
            "border_margin": 4,
            "cell_stride": 8,
            "descriptor_stride": 8,
            "descriptor_dim": 256
          }
        }
      },
      "input_nodes": [
        { "name": "semi", "type": "buffer", "size": 1248000 },
        { "name": "desc", "type": "buffer", "size": 4915200 }
      ],
      "output_nodes": [{ "name": "features", "type": "buffer", "size": 1024 }]
    }
  ]
}
)JSON";
      superpoint_out.close();
      std::string superpoint_error;
      const auto superpoint_contract =
          load_mpk_contract_from_pack_root(superpoint_root.string(), &superpoint_error);
      require(superpoint_contract.has_value(),
              "SuperPoint schema-v1 mpk should parse successfully: " + superpoint_error);
      require(superpoint_contract->plugins.size() == 1U, "expected one parsed SuperPoint plugin");
      const auto& superpoint = superpoint_contract->plugins.front().superpoint;
      require(superpoint.schema_version == 1 && superpoint.profile == "lightglue-v1",
              "SuperPoint schema/profile parsing mismatch");
      require(superpoint.profile_fingerprint.starts_with("sha256:") &&
                  superpoint.profile_fingerprint.size() == 71U,
              "SuperPoint fingerprint parsing mismatch");
      require(superpoint.detector_tensor_id == "semi" && superpoint.descriptor_tensor_id == "desc",
              "SuperPoint tensor-id parsing mismatch");
      require(superpoint.detector_representation == "raw-logits-65" &&
                  superpoint.descriptor_representation == "coarse-pre-l2",
              "SuperPoint representation parsing mismatch");
      require(superpoint.descriptor_output_dtype == "BF16" && superpoint.nms_radius == 0 &&
                  superpoint.border_margin == 4 && superpoint.cell_stride == 8 &&
                  superpoint.descriptor_stride == 8 && superpoint.descriptor_dim == 256,
              "SuperPoint numeric field parsing mismatch or radius zero was lost");
      require(superpoint.validation_error.empty(),
              "valid SuperPoint integer metadata must not report a parser error");

      const fs::path malformed_sp_root = make_temp_pack_root("superpoint_bad_integer");
      std::ofstream malformed_sp_out(malformed_sp_root / "malformed_superpoint_mpk.json");
      require(malformed_sp_out.is_open(), "failed to open malformed SuperPoint mpk json");
      malformed_sp_out << R"JSON(
{
  "name": "malformed_superpoint_mpk",
  "model_path": "superpoint.elf",
  "plugins": [{
    "name": "boxdecode_0",
    "sequence": 1,
    "processor": "A65",
    "config_params": {
      "kernel": "boxdecode",
      "params": {
        "decode_type": "superpoint",
        "superpoint": {
          "schema_version": 1.5,
          "profile": "lightglue-v1"
        }
      }
    },
    "input_nodes": [{ "name": "semi", "type": "buffer", "size": 1 }],
    "output_nodes": [{ "name": "features", "type": "buffer", "size": 1 }]
  }]
}
)JSON";
      malformed_sp_out.close();
      std::string malformed_sp_error;
      const auto malformed_sp_contract =
          load_mpk_contract_from_pack_root(malformed_sp_root.string(), &malformed_sp_error);
      require(malformed_sp_contract.has_value(),
              "generic MPK loading should retain malformed SuperPoint metadata for the "
              "BoxDecode validator: " +
                  malformed_sp_error);
      require(malformed_sp_contract->plugins.front().superpoint.validation_error.find(
                  "schema_version must be an integer") != std::string::npos,
              "fractional SuperPoint integer metadata must fail closed downstream");
    }));

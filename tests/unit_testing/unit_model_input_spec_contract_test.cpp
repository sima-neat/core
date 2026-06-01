#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "model_archive_fixture_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstring>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;

std::string require_yolov9_tar() {
  const std::string tar =
      sima_test::resolve_modelzoo_tar("yolo_v9c_seg", sima_test::repo_root_for_modelzoo());
  require(!tar.empty(), "missing yolo_v9c_seg tar; run sima-cli modelzoo get yolo_v9c_seg");
  require(std::filesystem::exists(tar), "resolved yolo_v9c_seg tar does not exist: " + tar);
  return tar;
}

sima_test::ModelArchiveFixture make_evo_style_multi_ingress_fixture(const std::string& tag) {
  return sima_test::make_model_archive_fixture(tag, {
                                                        {"etc/evo_style_multi_ingress_mpk.json",
                                                         R"json({
  "name": "evo_style_multi_ingress",
  "model_path": "evo_style_multi_ingress.onnx",
  "model_sdk_version": "2.0.0",
  "sequence": 1,
  "input_nodes": [
    { "name": "image_l",  "type": "buffer", "size": 64 },
    { "name": "image_uv", "type": "buffer", "size": 32 }
  ],
  "plugins": [
    {
      "name": "quantize_0",
      "sequence": 1,
      "processor": "EV74",
      "config_params": {
        "kernel": "quantization_transform",
        "params": {
          "channel_params": [[255.0, -128]],
          "num_bits": 8,
          "rounding": "TONEAREST",
          "input_shapes": [[1, 4, 4, 1]],
          "input_data_type": ["FP32"],
          "output_shapes": [[1, 4, 4, 1]],
          "output_data_type": "INT8"
        }
      },
      "input_nodes": [
        { "name": "image_l", "size": 64 }
      ],
      "output_nodes": [
        { "name": "quantize_0", "type": "buffer", "size": 16 }
      ],
      "type": "sgpProcess"
    },
    {
      "name": "quantize_1",
      "sequence": 2,
      "processor": "EV74",
      "config_params": {
        "kernel": "quantization_transform",
        "params": {
          "channel_params": [[255.0, -128]],
          "num_bits": 8,
          "rounding": "TONEAREST",
          "input_shapes": [[1, 2, 4, 1]],
          "input_data_type": ["FP32"],
          "output_shapes": [[1, 2, 4, 1]],
          "output_data_type": "INT8"
        }
      },
      "input_nodes": [
        { "name": "image_uv", "size": 32 }
      ],
      "output_nodes": [
        { "name": "quantize_1", "type": "buffer", "size": 8 }
      ],
      "type": "sgpProcess"
    },
    {
      "name": "MLA_0_ifm_pack_transform",
      "sequence": 3,
      "processor": "EV74",
      "config_params": {
        "kernel": "pack_transform",
        "params": {
          "input_shapes": [[1, 4, 4, 1], [1, 2, 4, 1]],
          "input_data_type": ["INT8", "INT8"],
          "output_shapes": [[1, 24]],
          "data_type": ["INT8"]
        }
      },
      "input_nodes": [
        { "name": "quantize_0", "size": 16 },
        { "name": "quantize_1", "size": 8 }
      ],
      "output_nodes": [
        { "name": "MLA_0_ifm_pack_transform", "type": "buffer", "size": 24 }
      ],
      "type": "sgpProcess"
    },
    {
      "name": "MLA_0",
      "sequence": 4,
      "processor": "MLA",
      "config_params": {
        "desired_batch_size": 1,
        "actual_batch_size": 1,
        "number_of_quads_to_user": 1
      },
      "input_nodes": [
        { "name": "MLA_0_ifm_pack_transform", "size": 24 }
      ],
      "output_nodes": [
        { "name": "MLA_0", "type": "buffer", "size": 24 }
      ],
      "type": "sgpProcess",
      "resources": { "executable": "stage0.elf" }
    },
    {
      "name": "MLA_0_ofm_unpack_transform",
      "sequence": 5,
      "processor": "EV74",
      "config_params": {
        "kernel": "unpack_transform",
        "params": {
          "tensor_types": ["INT8", "INT8"],
          "tensor_shapes": [[1, 4, 4, 1], [1, 2, 4, 1]],
          "input_shapes": [[1, 24]],
          "output_shapes": [[1, 4, 4, 1], [1, 2, 4, 1]]
        }
      },
      "input_nodes": [
        { "name": "MLA_0", "size": 24 }
      ],
      "output_nodes": [
        { "name": "MLA_0_ofm_unpack_transform_0", "type": "buffer", "size": 16 },
        { "name": "MLA_0_ofm_unpack_transform_1", "type": "buffer", "size": 8 }
      ],
      "type": "sgpProcess"
    },
    {
      "name": "dequantize_0",
      "sequence": 6,
      "processor": "EV74",
      "config_params": {
        "kernel": "dequantization_transform",
        "params": {
          "channel_params": [[1.0, -128]],
          "input_data_type": "INT8",
          "output_data_type": "FP32",
          "input_shapes": [[1, 4, 4, 1]],
          "output_shapes": [[1, 4, 4, 1]]
        }
      },
      "input_nodes": [
        { "name": "MLA_0_ofm_unpack_transform_0", "size": 16 }
      ],
      "output_nodes": [
        { "name": "dequantize_0/output_l", "type": "buffer", "size": 64 }
      ],
      "type": "sgpProcess"
    },
    {
      "name": "dequantize_1",
      "sequence": 7,
      "processor": "EV74",
      "config_params": {
        "kernel": "dequantization_transform",
        "params": {
          "channel_params": [[1.0, -128]],
          "input_data_type": "INT8",
          "output_data_type": "FP32",
          "input_shapes": [[1, 2, 4, 1]],
          "output_shapes": [[1, 2, 4, 1]]
        }
      },
      "input_nodes": [
        { "name": "MLA_0_ofm_unpack_transform_1", "size": 8 }
      ],
      "output_nodes": [
        { "name": "dequantize_1/output_uv", "type": "buffer", "size": 32 }
      ],
      "type": "sgpProcess"
    },
    {
      "name": "PassThrough",
      "sequence": 8,
      "processor": "EV74",
      "config_params": {
        "kernel": "pass_through",
        "params": {
          "input_shapes": [[1, 4, 4, 1], [1, 2, 4, 1]],
          "input_data_type": ["FP32", "FP32"],
          "output_shapes": [[1, 4, 4, 1], [1, 2, 4, 1]],
          "data_type": ["FP32", "FP32"]
        }
      },
      "input_nodes": [
        { "name": "dequantize_0/output_l", "size": 64 },
        { "name": "dequantize_1/output_uv", "size": 32 }
      ],
      "output_nodes": [
        { "name": "pass_through_out_0", "type": "buffer", "size": 64 },
        { "name": "pass_through_out_1", "type": "buffer", "size": 32 }
      ],
      "type": "sgpProcess"
    }
  ]
})json"},
                                                        {"etc/pipeline_sequence.json",
                                                         R"json({
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
                "name": "MLA_0",
                "pluginId": "processmla",
                "configPath": "0_process_mla.json",
                "processor": "MLA",
                "kernel": "infer",
        "input": "decoder"
      }
    ]
  }]
})json"},
                                                        {"etc/0_process_mla.json",
                                                         R"json({
  "node_name": "MLA_0",
  "input_buffers": [{"name": "MLA_0_ifm_pack_transform"}],
  "data_type": ["INT8"],
  "output_width": [24],
  "output_height": [1],
  "output_depth": [1]
})json"},
                                                    });
}

Tensor make_route_tensor(int logical_index, int physical_index, int route_slot, int memory_index,
                         std::size_t bytes, std::uint8_t fill, const std::string& name,
                         const std::string& backend_name, const std::string& segment_name) {
  Tensor tensor;
  tensor.dtype = TensorDType::Float32;
  tensor.layout = TensorLayout::HW;
  tensor.shape = {static_cast<std::int64_t>(bytes / sizeof(float)), 1};
  tensor.strides_bytes = {static_cast<std::int64_t>(sizeof(float)),
                          static_cast<std::int64_t>(sizeof(float))};
  tensor.storage = make_cpu_owned_storage(bytes);
  tensor.byte_offset = 0;
  tensor.route.logical_index = logical_index;
  tensor.route.physical_index = physical_index;
  tensor.route.route_slot = route_slot;
  tensor.route.memory_index = memory_index;
  tensor.route.name = name;
  tensor.route.backend_name = backend_name;
  tensor.route.segment_name = segment_name;

  Mapping map = tensor.storage->map(MapMode::Write);
  require(map.data != nullptr && map.size_bytes >= bytes, "route tensor storage map failed");
  std::memset(map.data, fill, bytes);
  return tensor;
}

} // namespace

RUN_TEST(
    "unit_model_input_spec_contract_test", ([] {
      using namespace simaai::neat;

      {
        const std::string tar = require_yolov9_tar();
        Model model(tar);
        const TensorSpec spec = model.input_spec();

        require(spec.rank == 3, "Model::input_spec rank should be 3 for yolov9 tensor ingress");
        require(spec.shape == std::vector<int64_t>({640, 640, 3}),
                "Model::input_spec should expose yolov9 Model boundary tensor ingress shape");
        require(!spec.dtypes.empty() && spec.dtypes[0] == TensorDType::Float32,
                "Model::input_spec should expose yolov9 Model boundary ingress dtype");
        require(!spec.image_format.has_value(),
                "Tensor ingress Model::input_spec should not expose image_format");
      }

      {
        const auto root = sima_test::repo_root_for_modelzoo();
        const std::vector<std::filesystem::path> bf16_candidates = {
            root / "tmp" / "yolov8n_drive" / "yolov8n_A_W_BF16_mpk.tar.gz",
            root / "tmp" / "yolov8n_drive" / "yolov8n_A_BF16_W_INT8_mpk.tar.gz",
        };
        for (const auto& bf16_model : bf16_candidates) {
          if (!std::filesystem::exists(bf16_model)) {
            continue;
          }
          Model model(bf16_model.string());
          const TensorSpec spec = model.input_spec();
          require(!spec.dtypes.empty() && spec.dtypes[0] == TensorDType::Float32,
                  "BF16 cast ingress must expose Float32 Model boundary input_spec");
          break;
        }
      }

      {
        const auto legacy = sima_test::make_model_archive_fixture(
            "model_input_spec_legacy_missing_mpk", {
                                                       {"etc/pipeline_sequence.json",
                                                        R"json({
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
        "name": "mla_0",
        "pluginId": "processmla",
        "configPath": "0_process_mla.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "decoder"
      }
    ]
  }]
})json"},
                                                       {"etc/0_process_mla.json",
                                                        R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "decoder"}]
})json"},
                                                   });
        bool threw = false;
        try {
          Model legacy_model(legacy.tar_path);
          (void)legacy_model.input_spec();
        } catch (const std::exception& e) {
          threw = true;
          require_contains(std::string(e.what()), "strict MPK contract required",
                           "legacy missing-mpk fixture should fail with strict contract error");
        }
        require(threw, "legacy missing-mpk fixture must fail under strict contract");
      }

      {
        const auto fixture = make_evo_style_multi_ingress_fixture("model_input_spec_multi_ingress");
        Model model(fixture.tar_path);

        const auto specs = model.input_specs();
        require(specs.size() == 2U,
                "Model::input_specs should expose both ordered ingress tensors");
        require(specs[0].shape == std::vector<int64_t>({4, 4, 1}),
                "first ingress spec should expose image_l geometry");
        require(specs[1].shape == std::vector<int64_t>({2, 4, 1}),
                "second ingress spec should expose image_uv geometry");
        require(!specs[0].dtypes.empty() && specs[0].dtypes[0] == TensorDType::Float32,
                "multi-ingress input_specs should preserve ingress dtype");

        const auto appsrc_opts = model.input_appsrc_options_list(true);
        require(appsrc_opts.size() == 2U,
                "Model::input_appsrc_options_list should expose one option per ingress");
        require(appsrc_opts[0].buffer_name == "image_l",
                "first multi-ingress appsrc option should preserve ingress tensor name");
        require(appsrc_opts[1].buffer_name == "image_uv",
                "second multi-ingress appsrc option should preserve ingress tensor name");

        bool singular_spec_threw = false;
        try {
          (void)model.input_spec();
        } catch (const std::exception& e) {
          singular_spec_threw = true;
          require_contains(std::string(e.what()), "plural ingress API",
                           "singular input_spec should fail clearly for multi-ingress models");
        }
        require(singular_spec_threw, "singular input_spec must reject multi-ingress models");

        bool singular_opt_threw = false;
        try {
          (void)model.input_appsrc_options(true);
        } catch (const std::exception& e) {
          singular_opt_threw = true;
          require_contains(
              std::string(e.what()), "plural ingress API",
              "singular input_appsrc_options should fail clearly for multi-ingress models");
        }
        require(singular_opt_threw,
                "singular input_appsrc_options must reject multi-ingress models");

        Tensor image_l;
        image_l.dtype = TensorDType::Float32;
        image_l.layout = TensorLayout::HWC;
        image_l.shape = {4, 4, 1};
        image_l.strides_bytes = {16, 4, 4};
        image_l.storage = make_cpu_owned_storage(64);

        bool bad_count_threw = false;
        try {
          (void)model.build(TensorList{image_l});
        } catch (const std::exception& e) {
          bad_count_threw = true;
          require_contains(std::string(e.what()), "expected exactly 2 ingress tensor inputs",
                           "multi-ingress build should enforce exact tensor input count");
        }
        require(bad_count_threw, "multi-ingress build must reject wrong tensor input count");
      }

      {
        Tensor global_branch_tensor =
            make_route_tensor(1, 4, 1, 9, 64U, 0x5A, "image_uv", "cast_1", "cast_1");
        const auto original_storage = global_branch_tensor.storage;
        const auto original_shape = global_branch_tensor.shape;
        const auto original_strides = global_branch_tensor.strides_bytes;

        const Tensor localized = internal::remap_tensor_to_consumer_identity(
            std::move(global_branch_tensor), internal::IngressConsumerTensorIdentity{
                                                 .logical_index = 0,
                                                 .physical_index = 0,
                                                 .route_slot = 0,
                                             });

        require(localized.storage == original_storage,
                "branch-local remap must preserve storage ownership");
        require(localized.shape == original_shape, "branch-local remap must preserve tensor shape");
        require(localized.strides_bytes == original_strides,
                "branch-local remap must preserve tensor strides");
        require(localized.route.logical_index == 0,
                "branch-local remap must relabel logical index to consumer-local zero");
        require(localized.route.physical_index == 0,
                "branch-local remap must relabel physical index to consumer-local zero");
        require(localized.route.route_slot == 0,
                "branch-local remap must relabel route slot to consumer-local zero");
        require(localized.route.memory_index == 9,
                "branch-local remap must preserve storage memory index");
        require(localized.route.name == "image_uv",
                "branch-local remap must not rewrite logical tensor name");
        require(localized.route.backend_name == "cast_1",
                "branch-local remap must not rewrite backend tensor name");
        require(localized.route.segment_name == "cast_1",
                "branch-local remap must not rewrite segment name");
      }

      {
        Tensor first = make_route_tensor(0, 0, 0, 0, 64U, 0x11, "image_l", "cast_0", "cast_0");
        Tensor second = make_route_tensor(1, 0, 1, 0, 32U, 0x22, "image_uv", "cast_1", "cast_1");
        const std::size_t first_bytes = first.dense_bytes_tight();
        const std::size_t second_bytes = second.dense_bytes_tight();
        const std::vector<internal::IngressConsumerTensorIdentity> consumer_identities = {
            {.logical_index = 0, .physical_index = 0, .route_slot = 0, .memory_index = 0},
            {.logical_index = 1, .physical_index = 1, .route_slot = 1, .memory_index = 0},
        };

        const TensorList joined = internal::materialize_joined_ingress_tensors(
            TensorList{first, second}, consumer_identities, "MLA_0",
            "unit_model_input_spec_contract_test");

        require(joined.size() == 2U, "joined-ingress materialization must preserve tensor count");
        require(joined[0].storage != nullptr && joined[0].storage == joined[1].storage,
                "joined-ingress materialization must compact tensors into one shared storage");
        require(joined[0].byte_offset == 0, "first joined tensor must start at byte offset zero");
        require(joined[1].byte_offset == static_cast<std::int64_t>(first_bytes),
                "second joined tensor must follow the first tensor payload");
        require(joined[0].route.logical_index == 0 && joined[0].route.physical_index == 0 &&
                    joined[0].route.route_slot == 0 && joined[0].route.memory_index == 0,
                "first joined tensor must preserve consumer-visible input identity");
        require(joined[1].route.logical_index == 1 && joined[1].route.physical_index == 1 &&
                    joined[1].route.route_slot == 1 && joined[1].route.memory_index == 0,
                "second joined tensor must preserve distinct consumer-visible input identity");
        require(joined[0].route.segment_name == "MLA_0" && joined[1].route.segment_name == "MLA_0",
                "joined-ingress materialization must stamp the shared join segment");
        require(joined[0].route.backend_name == "cast_0" &&
                    joined[1].route.backend_name == "cast_1",
                "joined-ingress materialization must preserve backend names");

        Mapping map = joined[0].storage->map(MapMode::Read);
        require(map.data != nullptr &&
                    map.size_bytes >= static_cast<std::size_t>(first_bytes + second_bytes),
                "joined-ingress shared storage map failed");
        const auto* bytes = static_cast<const std::uint8_t*>(map.data);
        require(bytes[0] == 0x11,
                "joined-ingress materialization must copy the first tensor payload");
        require(bytes[first_bytes] == 0x22,
                "joined-ingress materialization must copy the second tensor payload");
      }

      {
        Tensor first = make_route_tensor(0, 0, 0, 0, 64U, 0x33, "image_l", "cast_0", "cast_0");
        Tensor second = make_route_tensor(0, 0, 0, 0, 32U, 0x44, "image_uv", "cast_1", "cast_1");
        const std::size_t first_bytes = first.dense_bytes_tight();
        const std::vector<internal::IngressConsumerTensorIdentity> consumer_identities = {
            {.logical_index = 0, .physical_index = 0, .route_slot = 0, .memory_index = 0},
            {.logical_index = 1, .physical_index = 0, .route_slot = 1, .memory_index = 0},
        };

        const TensorList joined = internal::materialize_joined_ingress_tensors(
            TensorList{first, second}, consumer_identities, "MLA_0_ifm_pack_transform",
            "unit_model_input_spec_contract_test");

        require(joined.size() == 2U, "joined-ingress materialization must preserve tensor count "
                                     "for shared-physical contracts");
        require(joined[0].storage != nullptr && joined[0].storage == joined[1].storage,
                "joined-ingress materialization must still compact shared-physical inputs into one "
                "storage");
        require(joined[0].route.logical_index == 0 && joined[0].route.physical_index == 0 &&
                    joined[0].route.route_slot == 0 && joined[0].route.memory_index == 0,
                "shared-physical joined ingress must preserve the first consumer input identity");
        require(joined[1].route.logical_index == 1 && joined[1].route.physical_index == 0 &&
                    joined[1].route.route_slot == 1 && joined[1].route.memory_index == 0,
                "shared-physical joined ingress must preserve the second consumer input identity");
        require(joined[1].byte_offset == static_cast<std::int64_t>(first_bytes),
                "shared-physical joined ingress must still compact payloads sequentially");
        require(joined[0].route.segment_name == "MLA_0_ifm_pack_transform" &&
                    joined[1].route.segment_name == "MLA_0_ifm_pack_transform",
                "shared-physical joined ingress must stamp the shared consumer segment");
      }
    }));

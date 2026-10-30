#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "model_archive_fixture_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <cstring>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;

std::string require_yolov9_tar() {
  // Metadata-only unit coverage: reuse the checked exact modelzoo manifest
  // and existing topology fixture builder; no inference binary is executed.
  static const auto fixture = sima_test::make_strict_model_archive_fixture(
      "unit_model_input_spec_contract_test", {}, true, "yolo_v9c_seg", true);
  return fixture.tar_path;
}

sima_test::ModelArchiveFixture make_cast_ingress_fixture(const std::vector<int64_t>& shape) {
  auto mpk = nlohmann::json::parse(R"json({
  "name": "cast_ingress", "model_path": "cast.onnx", "model_sdk_version": "3.0.0",
  "input_nodes": [{"name": "input", "type": "buffer", "size": 192}],
  "plugins": [
    {"name": "cast_0", "processor": "EV74", "sequence": 1, "type": "sgpProcess",
     "config_params": {"kernel": "cast_transform", "params": {
       "in_dtype": "float32", "out_dtype": "bfloat16",
       "input_shapes": [[1,4,4,3]], "output_shapes": [[1,4,4,3]]}},
     "input_nodes": [{"name": "input", "size": 192}],
     "output_nodes": [{"name": "cast_0", "type": "buffer", "size": 96}]},
    {"name": "MLA_0", "processor": "MLA", "sequence": 2, "type": "sgpProcess",
     "config_params": {"desired_batch_size": 1, "actual_batch_size": 1},
     "input_nodes": [{"name": "cast_0", "size": 96}],
     "output_nodes": [{"name": "MLA_0", "type": "buffer", "size": 32}],
     "resources": {"executable": "model.elf"}},
    {"name": "cast_1", "processor": "EV74", "sequence": 3, "type": "sgpProcess",
     "config_params": {"kernel": "cast_transform", "params": {
       "in_dtype": "bfloat16", "out_dtype": "float32",
       "input_shapes": [[1,16]], "output_shapes": [[1,16]]}},
     "input_nodes": [{"name": "MLA_0", "size": 32}],
     "output_nodes": [{"name": "output", "type": "buffer", "size": 64}]}
  ]
})json");
  const int64_t batch = shape.size() == 4U ? shape.front() : 1;
  auto& plugins = mpk["plugins"];
  plugins[0]["config_params"]["params"]["input_shapes"] = {shape};
  plugins[0]["config_params"]["params"]["output_shapes"] = {shape};
  plugins[1]["config_params"] = {{"desired_batch_size", batch}, {"actual_batch_size", batch}};
  plugins[2]["config_params"]["params"]["input_shapes"] = {{batch, 16}};
  plugins[2]["config_params"]["params"]["output_shapes"] = {{batch, 16}};
  mpk["input_nodes"][0]["size"] = 192 * batch;
  for (auto& plugin : plugins) {
    for (const auto* side : {"input_nodes", "output_nodes"})
      for (auto& tensor : plugin[side])
        tensor["size"] = tensor["size"].get<int64_t>() * batch;
  }
  return sima_test::make_model_archive_fixture(
      "model_input_cast_rank" + std::to_string(shape.size()) + "_batch" + std::to_string(batch),
      {{"etc/cast_mpk.json", mpk.dump()}});
}

void check_tensor_shape_metadata() {
  for (const auto& shape : {std::vector<int64_t>{4, 4, 3}, {1, 4, 4, 3}, {2, 4, 4, 3}}) {
    const auto fixture = make_cast_ingress_fixture(shape);
    Model::Options options;
    options.preprocess.kind = InputKind::Tensor;
    Model model(fixture.tar_path, options);
    const auto specs = model.input_specs();
    require(specs.size() == 1U && specs[0].shape == shape &&
                specs[0].rank == static_cast<int>(shape.size()),
            "tensor input metadata must retain the authored rank, including batch one");
    const std::vector<int64_t> hwc_shape{4, 4, 3};
    InputGeometry hwc;
    hwc.shape = hwc_shape;
    hwc.layout = TensorLayout::HWC;
    const bool batched = shape.size() == 4U && shape.front() > 1;
    require(input_geometry_matches_logical_shape(hwc, specs[0].shape, TensorLayout::HWC) ==
                !batched,
            "HWC shorthand must remain valid only for a batch-one tensor contract");
    require(model.compiled_batch_size() == (batched ? 2 : 1),
            "metadata rank must not change the compiled batch size");
  }
}

void check_ingress_memory_policy() {
  const auto cast_fixture = make_cast_ingress_fixture({1, 4, 4, 3});
  for (const auto& path : {cast_fixture.tar_path, require_yolov9_tar()}) {
    for (const auto* target : {"AUTO", "EV74", "A65"}) {
      const auto expected =
          std::string(target) == "A65" ? InputMemoryPolicy::Dms0 : InputMemoryPolicy::Ev74;
      for (const bool advanced : {false, true}) {
        Model::Options options;
        options.preprocess.kind = InputKind::Tensor;
        if (advanced)
          options.advanced_execution.preprocess_target = target;
        else
          options.processcvu.pre_run_target = target;
        Model model(path, options);
        require(model.input_appsrc_options(true).memory_policy == expected,
                "ingress memory must follow the selected Cast/Quant executor");
        require(model.input_appsrc_options_list(true).front().memory_policy == expected,
                "ordered ingress memory must follow the selected executor");
      }
    }
  }
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

void check_synthetic_image_inputs() {
  // This exercises the production seed builder directly for every planar and packed
  // public Model preprocessing format, including NV16.
  for (const auto* format : {"NV12", "NV16", "I420", "RGB", "BGR", "GRAY8"}) {
    InputOptions options;
    options.payload_type = PayloadType::Image;
    options.format = format;
    options.width = 8;
    options.height = 6;
    const auto tensor = internal::ModelAccess::make_dummy_input(options);
    const bool nv16 = std::string(format) == "NV16";
    const bool nv12 = std::string(format) == "NV12";
    const bool i420 = std::string(format) == "I420";
    const bool planar = nv16 || nv12 || i420;
    const bool gray = std::string(format) == "GRAY8";
    const std::size_t bytes = nv16 ? 96U : (nv12 || i420) ? 72U : gray ? 48U : 144U;
    require(tensor.storage && tensor.storage->size_bytes == bytes,
            std::string(format) + " synthetic input byte count");
    require(tensor.device.type == DeviceType::CPU && tensor.read_only &&
                tensor.dtype == TensorDType::UInt8,
            "synthetic image ownership/type changed");
    require(tensor.shape ==
                (planar || gray ? std::vector<int64_t>{6, 8} : std::vector<int64_t>{6, 8, 3}),
            std::string(format) + " synthetic image shape");
    std::string error;
    require(tensor.validate(&error), std::string(format) + " synthetic input: " + error);
    if (nv12 || nv16) {
      require(tensor.planes.size() == 2 && tensor.planes[0].role == PlaneRole::Y &&
                  tensor.planes[1].role == PlaneRole::UV &&
                  tensor.planes[0].shape == std::vector<int64_t>{6, 8} &&
                  tensor.planes[1].shape == std::vector<int64_t>{nv16 ? 6 : 3, 8} &&
                  tensor.planes[0].strides_bytes == std::vector<int64_t>{8, 1} &&
                  tensor.planes[1].strides_bytes == std::vector<int64_t>{8, 1} &&
                  tensor.planes[0].byte_offset == 0 && tensor.planes[1].byte_offset == 48,
              std::string(format) + " synthetic Y/UV geometry");
    }
  }
  InputOptions options;
  options.payload_type = PayloadType::Image;
  options.format = FormatTag::NV16;
  options.max_width = 8;
  options.max_height = 5;
  const auto odd_height = internal::ModelAccess::make_dummy_input(options);
  std::string error;
  require(odd_height.validate(&error), "odd-height NV16 synthetic input: " + error);
  require(odd_height.storage->size_bytes == 80 &&
              odd_height.axis_semantics ==
                  std::vector<TensorAxisSemantic>{TensorAxisSemantic::H, TensorAxisSemantic::W} &&
              odd_height.semantic.image &&
              odd_height.semantic.image->format == ImageSpec::PixelFormat::NV16,
          "NV16 synthetic semantics or fallback capacity changed");
  const auto spec = describe_tensor_spec_or_throw(odd_height, options, "NV16 synthetic input");
  require(spec.format == "NV16" && spec.width == 8 && spec.height == 5 && spec.planes.size() == 2 &&
              spec.planes[0].height == 5 && spec.planes[1].height == 5 &&
              spec.planes[1].offset_bytes == 40,
          "NV16 synthetic ingress description lost full-height UV");
  options.max_width = 7;
  const auto odd_width = internal::ModelAccess::make_dummy_input(options);
  require(!odd_width.validate(&error), "odd-width NV16 synthetic input must remain invalid");

  options.payload_type = PayloadType::Tensor;
  options.format = FormatTag::FP32;
  options.max_width = 8;
  options.max_depth = 3;
  const auto dense = internal::ModelAccess::make_dummy_input(options);
  require(dense.dtype == TensorDType::Float32 && dense.planes.empty() &&
              dense.shape == std::vector<int64_t>{5, 8, 3} &&
              dense.storage->size_bytes == 5U * 8U * 3U * sizeof(float),
          "tensor-mode synthetic input changed");
}

void check_synthetic_tensor_contract_shapes() {
  InputOptions options;
  options.payload_type = PayloadType::Tensor;
  options.format = FormatTag::FP32;
  options.width = 192;
  options.height = 1;
  options.depth = 144;
  internal::IngressTensorContract contract;
  contract.valid = true;
  contract.ingress_index = 0;
  contract.media_type = "application/vnd.simaai.tensor";
  contract.height = 1;
  contract.width = 192;
  contract.depth = 144;
  for (const auto& shape :
       {std::vector<int64_t>{2, 1, 1, 192, 144}, {2, 1, 192, 144}, {1, 1, 192, 144}}) {
    contract.rank = static_cast<int>(shape.size());
    // RoutePlanner projects batch from the fourth-last axis, losing leading axes at rank > 4.
    contract.batch = static_cast<int>(shape[shape.size() - 4]);
    contract.logical_shape = shape;
    const auto tensor = internal::ModelAccess::make_dummy_input(options, &contract);
    std::string error;
    require(tensor.validate(&error), "synthetic tensor input: " + error);
    require(
        tensor.dtype == TensorDType::Float32 && tensor.planes.empty() && tensor.shape == shape &&
            tensor.storage->size_bytes == std::size_t(shape.front()) * 192U * 144U * sizeof(float),
        "synthetic tensor must retain every canonical axis and its full storage");
    require(tensor.layout == (shape.size() == 4U ? TensorLayout::HWC : TensorLayout::Unknown),
            "synthetic tensor layout must match its canonical rank");
    const auto geometry =
        describe_tensor_geometry_or_throw(tensor, options, "synthetic tensor input");
    require(input_geometry_matches_logical_shape(geometry, shape, TensorLayout::Unknown),
            "synthetic tensor must satisfy canonical input geometry");
  }
}

} // namespace

RUN_TEST(
    "unit_model_input_spec_contract_test", ([] {
      using namespace simaai::neat;
      check_synthetic_image_inputs();
      check_synthetic_tensor_contract_shapes();
      check_ingress_memory_policy();
      check_tensor_shape_metadata();

      {
        const std::string tar = require_yolov9_tar();
        Model model(tar);
        const std::vector<TensorSpec> specs = model.input_specs();
        require(specs.size() == 1U, "Model::input_specs should expose one yolov9 input");
        const TensorSpec& spec = specs.front();

        require(spec.rank == 4, "Model::input_specs must retain the authored yolov9 batch axis");
        require(spec.shape == std::vector<int64_t>({1, 640, 640, 3}),
                "Model::input_specs should expose yolov9 Model boundary tensor ingress shape");
        require(!spec.dtypes.empty() && spec.dtypes[0] == TensorDType::Float32,
                "Model::input_specs should expose yolov9 Model boundary ingress dtype");
        require(!spec.image_format.has_value(),
                "Tensor ingress Model::input_specs should not expose image_format");
      }

      {
        Model::Options options;
        options.preprocess.kind = InputKind::Image;
        Model model(require_yolov9_tar(), options);
        const auto specs = model.input_specs();
        require(specs.size() == 1U && specs[0].rank == 3 &&
                    specs[0].shape == std::vector<int64_t>({-1, -1, 3}) &&
                    specs[0].image_format.has_value(),
                "image preprocessing must retain its dynamic HWC input contract");
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
          const std::vector<TensorSpec> specs = model.input_specs();
          require(!specs.empty(), "BF16 model input_specs should not be empty");
          const TensorSpec& spec = specs.front();
          require(!spec.dtypes.empty() && spec.dtypes[0] == TensorDType::Float32,
                  "BF16 cast ingress must expose Float32 Model boundary input_specs");
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
          (void)legacy_model.input_specs();
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
        require(specs[0].shape == std::vector<int64_t>({1, 4, 4, 1}),
                "first ingress spec should expose image_l geometry");
        require(specs[1].shape == std::vector<int64_t>({1, 2, 4, 1}),
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

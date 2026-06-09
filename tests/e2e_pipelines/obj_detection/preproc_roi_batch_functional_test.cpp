#include "model/Model.h"
#include "pipeline/StageRun.h"
#include "pipeline/TensorCore.h"

#include "model_archive_fixture_utils.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef SIMA_ROI_USER_SMOKE_DEFAULT
#define SIMA_ROI_USER_SMOKE_DEFAULT 0
#endif

namespace {

sima_test::ModelArchiveFixture
make_roi_batch_preproc_fixture_from_config(const std::string& tag,
                                           const std::string& preproc_config_json) {
  return sima_test::make_model_archive_fixture(tag,
                                               {
                                                   {"etc/preproc_roi_batch_functional_mpk.json",
                                                    R"json({
  "name": "preproc_roi_batch_functional",
  "model_path": "preproc_roi_batch_functional.onnx",
  "model_sdk_version": "2.0.0",
  "sequence": 1,
  "input_nodes": [
    {
      "name": "images",
      "type": "buffer",
      "size": 9216,
      "input_range": [0.0, 255.0],
      "logical_shape": [1, 48, 64, 3],
      "logical_dtype": "UINT8"
    }
  ],
  "plugins": [
    {
      "name": "preproc_0",
      "sequence": 1,
      "processor": "EV74",
      "config_params": {
        "desired_batch_size": 1,
        "actual_batch_size": 1,
        "kernel": "preproc",
        "params": {
          "input_shapes": [[1, 48, 64, 3]],
          "output_shapes": [[1, 48, 64, 3]],
          "input_dtype": ["UINT8"],
          "output_dtype": "BF16"
        }
      },
      "input_nodes": [{"name": "images", "size": 9216}],
      "output_nodes": [{
        "name": "preproc_0",
        "type": "buffer",
        "size": 18432,
        "logical_shape": [1, 48, 64, 3],
        "logical_dtype": "BF16"
      }],
      "type": "sgpProcess",
      "resources": {"executable": "kernel_name_tbd"}
    },
    {
      "name": "mla_0",
      "sequence": 2,
      "processor": "MLA",
      "config_params": {
        "desired_batch_size": 1,
        "actual_batch_size": 1,
        "number_of_quads_to_user": 1,
        "input_shapes": [[1, 48, 64, 3]],
        "input_data_type": ["BF16"],
        "output_shapes": [[1, 1, 1, 1]],
        "data_type": ["BF16"]
      },
      "input_nodes": [{
        "name": "preproc_0",
        "size": 18432,
        "logical_shape": [1, 48, 64, 3],
        "logical_dtype": "BF16"
      }],
      "output_nodes": [{
        "name": "mla_0",
        "type": "buffer",
        "size": 2,
        "logical_shape": [1, 1, 1, 1],
        "logical_dtype": "BF16"
      }],
      "type": "sgpProcess",
      "resources": {"executable": "stage0.elf"}
    }
  ]
})json"},
                                                   {"etc/pipeline_sequence.json",
                                                    R"json({
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
        "name": "preproc_0",
        "pluginId": "processcvu",
        "configPath": "0_preproc.json",
        "processor": "CVU",
        "kernel": "preproc",
        "input": "decoder"
      },
      {
        "sequence_id": 2,
        "name": "mla_0",
        "pluginId": "processmla",
        "configPath": "0_process_mla.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "preproc_0"
      }
    ]
  }]
})json"},
                                                   {"etc/0_preproc.json", preproc_config_json},
                                                   {"etc/0_process_mla.json",
                                                    R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "preproc_0"}],
  "input_format": ["EV81_BFLOAT16"],
  "data_type": ["EV81_BFLOAT16"],
  "input_width": [64],
  "input_height": [48],
  "input_depth": [3],
  "output_width": [1],
  "output_height": [1],
  "output_depth": [1]
})json"},
                                               },
                                               true);
}

sima_test::ModelArchiveFixture make_roi_batch_preproc_fixture() {
  return make_roi_batch_preproc_fixture_from_config("preproc_roi_batch_functional",
                                                    R"json({
  "node_name": "preproc_0",
  "graph_name": "preproc",
  "input_width": 64,
  "input_height": 48,
  "input_img_type": "RGB",
  "output_width": 64,
  "output_height": 48,
  "output_img_type": "RGB",
  "normalize": true,
  "channel_mean": [0.0, 0.0, 0.0],
  "channel_stddev": [1.0, 1.0, 1.0],
  "output_dtype": "BF16",
  "tessellate": false,
  "aspect_ratio": false,
  "scaling_type": "BILINEAR"
})json");
}

sima_test::ModelArchiveFixture
make_roi_batch_preproc_resize_fixture(const std::string& tag, const std::string& scaling_type,
                                      bool aspect_ratio) {
  std::ostringstream config;
  config << R"json({
  "node_name": "preproc_0",
  "graph_name": "preproc",
  "input_width": 64,
  "input_height": 48,
  "input_img_type": "RGB",
  "output_width": 64,
  "output_height": 48,
  "output_img_type": "RGB",
  "normalize": true,
  "channel_mean": [0.0, 0.0, 0.0],
  "channel_stddev": [1.0, 1.0, 1.0],
  "output_dtype": "BF16",
  "tessellate": false,
  "aspect_ratio": )json"
         << (aspect_ratio ? "true" : "false") << R"json(,
  "scaling_type": ")json"
         << scaling_type << R"json("
})json";
  return make_roi_batch_preproc_fixture_from_config(tag, config.str());
}

sima_test::ModelArchiveFixture
make_roi_batch_preproc_letterbox_fixture(const std::string& scaling_type = "BILINEAR") {
  return make_roi_batch_preproc_resize_fixture("preproc_roi_batch_letterbox_" + scaling_type,
                                               scaling_type,
                                               /*aspect_ratio=*/true);
}

sima_test::ModelArchiveFixture make_roi_batch_preproc_tess_route_fixture(
    const std::string& tag, const std::string& adapter_name, const std::string& adapter_kernel,
    const std::string& mla_input_dtype, std::size_t adapter_output_bytes,
    const std::string& adapter_params_extra = "") {
  std::ostringstream mpk;
  mpk << R"json({
  "name": ")json"
      << tag << R"json(",
  "model_path": "preproc_roi_batch_functional.onnx",
  "model_sdk_version": "2.0.0",
  "sequence": 1,
  "input_nodes": [
    {
      "name": "images",
      "type": "buffer",
      "size": 9216,
      "input_range": [0.0, 255.0],
      "logical_shape": [1, 48, 64, 3],
      "logical_dtype": "UINT8"
    }
  ],
  "plugins": [
    {
      "name": "preproc_0",
      "sequence": 1,
      "processor": "EV74",
      "config_params": {
        "desired_batch_size": 1,
        "actual_batch_size": 1,
        "kernel": "preproc",
        "params": {
          "input_shapes": [[1, 48, 64, 3]],
          "output_shapes": [[1, 48, 64, 3]],
          "input_dtype": ["UINT8"],
          "output_dtype": "BF16"
        }
      },
      "input_nodes": [{"name": "images", "size": 9216}],
      "output_nodes": [{
        "name": "preproc_0",
        "type": "buffer",
        "size": 18432,
        "logical_shape": [1, 48, 64, 3],
        "logical_dtype": "BF16"
      }],
      "type": "sgpProcess",
      "resources": {"executable": "kernel_name_tbd"}
    },
    {
      "name": ")json"
      << adapter_name << R"json(",
      "sequence": 2,
      "processor": "EV74",
      "config_params": {
        "desired_batch_size": 1,
        "actual_batch_size": 1,
        "kernel": ")json"
      << adapter_kernel << R"json(",
        "params": {
          "input_shapes": [[1, 48, 64, 3]],
          "output_shapes": [[1, 48, 64, 3]],
          "input_dtype": ["BF16"],
          "output_dtype": ")json"
      << mla_input_dtype << R"json(",
          "frame_type": ")json"
      << mla_input_dtype << R"json(",
          "align_c16": true,
          "slice_shape": [16, 16, 3])json";
  if (!adapter_params_extra.empty()) {
    mpk << ",\n" << adapter_params_extra;
  }
  mpk << R"json(
        }
      },
      "input_nodes": [{"name": "preproc_0", "size": 18432}],
      "output_nodes": [{
        "name": ")json"
      << adapter_name << R"json(",
        "type": "buffer",
        "size": )json"
      << adapter_output_bytes << R"json(,
        "logical_shape": [1, 48, 64, 3],
        "logical_dtype": ")json"
      << mla_input_dtype << R"json("
      }],
      "type": "sgpProcess",
      "resources": {"executable": "kernel_name_tbd"}
    },
    {
      "name": "mla_0",
      "sequence": 3,
      "processor": "MLA",
      "config_params": {
        "desired_batch_size": 1,
        "actual_batch_size": 1,
        "number_of_quads_to_user": 1,
        "input_shapes": [[1, 48, 64, 3]],
        "input_data_type": [")json"
      << mla_input_dtype << R"json("],
        "output_shapes": [[1, 1, 1, 1]],
        "data_type": ["BF16"]
      },
      "input_nodes": [{
        "name": ")json"
      << adapter_name << R"json(",
        "size": )json"
      << adapter_output_bytes << R"json(,
        "logical_shape": [1, 48, 64, 3],
        "logical_dtype": ")json"
      << mla_input_dtype << R"json("
      }],
      "output_nodes": [{
        "name": "mla_0",
        "type": "buffer",
        "size": 2,
        "logical_shape": [1, 1, 1, 1],
        "logical_dtype": "BF16"
      }],
      "type": "sgpProcess",
      "resources": {"executable": "stage0.elf"}
    }
  ]
})json";

  std::ostringstream pipeline;
  pipeline << R"json({
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
        "name": "preproc_0",
        "pluginId": "processcvu",
        "configPath": "0_preproc.json",
        "processor": "CVU",
        "kernel": "preproc",
        "input": "decoder"
      },
      {
        "sequence_id": 2,
        "name": ")json"
           << adapter_name << R"json(",
        "pluginId": "processcvu",
        "configPath": "1_tess.json",
        "processor": "CVU",
        "kernel": ")json"
           << adapter_kernel << R"json(",
        "input": "preproc_0"
      },
      {
        "sequence_id": 3,
        "name": "mla_0",
        "pluginId": "processmla",
        "configPath": "0_process_mla.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": ")json"
           << adapter_name << R"json("
      }
    ]
  }]
})json";

  std::ostringstream adapter_config;
  adapter_config << R"json({
  "node_name": ")json"
                 << adapter_name << R"json(",
  "graph_name": ")json"
                 << adapter_kernel << R"json(",
  "num_in_tensor": 1,
  "input_shapes": [[1, 48, 64, 3]],
  "slice_shape": [16, 16, 3],
  "input_dtype": "BF16",
  "output_dtype": ")json"
                 << mla_input_dtype << R"json("
})json";

  std::ostringstream mla_config;
  mla_config << R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": ")json"
             << adapter_name << R"json("}],
  "input_format": [")json"
             << (mla_input_dtype == "INT8" ? "EV81_INT8" : "EV81_BFLOAT16") << R"json("],
  "data_type": [")json"
             << (mla_input_dtype == "INT8" ? "EV81_INT8" : "EV81_BFLOAT16") << R"json("],
  "input_width": [64],
  "input_height": [48],
  "input_depth": [3],
  "output_width": [1],
  "output_height": [1],
  "output_depth": [1]
})json";

  return sima_test::make_model_archive_fixture(
      tag,
      {
          {"etc/preproc_roi_batch_functional_mpk.json", mpk.str()},
          {"etc/pipeline_sequence.json", pipeline.str()},
          {"etc/0_preproc.json",
           R"json({
  "node_name": "preproc_0",
  "graph_name": "preproc",
  "input_width": 64,
  "input_height": 48,
  "input_img_type": "RGB",
  "output_width": 64,
  "output_height": 48,
  "output_img_type": "RGB",
  "normalize": true,
  "channel_mean": [0.0, 0.0, 0.0],
  "channel_stddev": [1.0, 1.0, 1.0],
  "output_dtype": "BF16",
  "tessellate": false,
  "aspect_ratio": false,
  "scaling_type": "BILINEAR"
})json"},
          {"etc/1_tess.json", adapter_config.str()},
          {"etc/0_process_mla.json", mla_config.str()},
      },
      true);
}

sima_test::ModelArchiveFixture make_roi_batch_preproc_tess_bf16_fixture() {
  return make_roi_batch_preproc_tess_route_fixture("preproc_roi_batch_tess_bf16_functional",
                                                   "tessellate_1", "tessellate", "BF16", 18432);
}

sima_test::ModelArchiveFixture make_roi_batch_preproc_quanttess_int8_fixture() {
  return make_roi_batch_preproc_tess_route_fixture("preproc_roi_batch_quanttess_int8_functional",
                                                   "quanttess_1", "quanttess", "INT8", 9216,
                                                   R"json(          "q_scale": 32.0,
          "q_zp": -3)json");
}

cv::Mat make_test_image(int width, int height, int seed) {
  cv::Mat image(height, width, CV_8UC3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      image.at<cv::Vec3b>(y, x) = cv::Vec3b{
          static_cast<std::uint8_t>((seed + x * 3 + y * 5) & 0xff),
          static_cast<std::uint8_t>((seed * 7 + x * 11 + y * 13) & 0xff),
          static_cast<std::uint8_t>((seed * 17 + x * 19 + y * 23) & 0xff),
      };
    }
  }
  return image;
}

simaai::neat::PreprocessRoi single_image_roi(simaai::neat::PreprocessRoi roi) {
  roi.batch_index = 0;
  return roi;
}

simaai::neat::Model::Options make_roi_batch_model_options() {
  simaai::neat::Model::Options model_opt;
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.resize.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.resize.width = 64;
  model_opt.preprocess.resize.height = 48;
  model_opt.preprocess.resize.mode = simaai::neat::ResizeMode::Stretch;
  model_opt.preprocess.resize.scaling_type = "BILINEAR";
  model_opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.normalize.mean = {0.0f, 0.0f, 0.0f};
  model_opt.preprocess.normalize.stddev = {1.0f, 1.0f, 1.0f};
  model_opt.preprocess.color_convert.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::RGB;
  model_opt.preprocess.color_convert.output_format = simaai::neat::PreprocessColorFormat::RGB;
  model_opt.upstream_name = "decoder";
  return model_opt;
}

simaai::neat::Model::Options make_roi_batch_normalized_preproc_model_options() {
  simaai::neat::Model::Options model_opt = make_roi_batch_model_options();
  model_opt.preprocess.normalize.mean = {0.25f, 0.50f, 0.125f};
  model_opt.preprocess.normalize.stddev = {0.50f, 0.25f, 0.75f};
  return model_opt;
}

simaai::neat::Model::Options make_roi_batch_letterbox_model_options() {
  simaai::neat::Model::Options model_opt = make_roi_batch_model_options();
  model_opt.preprocess.resize.mode = simaai::neat::ResizeMode::Letterbox;
  model_opt.preprocess.resize.pad_value = 114;
  return model_opt;
}

simaai::neat::Model::Options make_roi_batch_resize_model_options(const std::string& scaling_type,
                                                                 simaai::neat::ResizeMode mode) {
  simaai::neat::Model::Options model_opt = make_roi_batch_model_options();
  model_opt.preprocess.resize.mode = mode;
  model_opt.preprocess.resize.scaling_type = scaling_type;
  model_opt.preprocess.resize.pad_value = 114;
  return model_opt;
}

struct ExpectedOutputTraits {
  bool enabled = false;
  bool normalize = false;
  bool tessellate = false;
  bool quantize = false;
  simaai::neat::TensorDType dtype = simaai::neat::TensorDType::UInt8;
};

void require_output_traits(const simaai::neat::Tensor& tensor, const ExpectedOutputTraits& expected,
                           const std::string& label) {
  if (!expected.enabled) {
    return;
  }
  require(tensor.dtype == expected.dtype, label + ": output dtype mismatch");
  require(tensor.semantic.preprocess.has_value(), label + ": missing preprocess metadata");
  const auto& meta = *tensor.semantic.preprocess;
  require(meta.normalize == expected.normalize, label + ": normalize metadata mismatch");
  require(meta.tessellate == expected.tessellate, label + ": tessellate metadata mismatch");
  require(meta.quantize == expected.quantize, label + ": quantize metadata mismatch");
  if (expected.tessellate) {
    require(tensor.route.segment_name == "output_tessellated_image",
            label + ": expected output_tessellated_image handoff");
  }
}

std::vector<std::uint8_t> tensor_payload(const simaai::neat::Tensor& tensor,
                                         const std::string& label) {
  try {
    return tensor.copy_payload_bytes();
  } catch (const std::exception& e) {
    throw std::runtime_error(label + ": failed to copy tensor payload: " + e.what());
  }
}

std::uint64_t payload_hash(const std::vector<std::uint8_t>& bytes) {
  std::uint64_t h = 1469598103934665603ULL;
  for (const std::uint8_t b : bytes) {
    h ^= static_cast<std::uint64_t>(b);
    h *= 1099511628211ULL;
  }
  return h;
}

struct TensorPayloadSnapshot {
  simaai::neat::TensorDType dtype = simaai::neat::TensorDType::UInt8;
  std::vector<int64_t> shape;
  std::vector<std::uint8_t> bytes;
};

TensorPayloadSnapshot snapshot_tensor_payload(const simaai::neat::Tensor& tensor,
                                              const std::string& label) {
  TensorPayloadSnapshot snap;
  snap.dtype = tensor.dtype;
  snap.shape = tensor.shape;
  snap.bytes = tensor_payload(tensor, label);
  return snap;
}

std::vector<TensorPayloadSnapshot> snapshot_tensor_payloads(const simaai::neat::TensorList& tensors,
                                                            const std::string& label) {
  std::vector<TensorPayloadSnapshot> out;
  out.reserve(tensors.size());
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    out.push_back(snapshot_tensor_payload(tensors[i], label + " snapshot " + std::to_string(i)));
  }
  return out;
}

bool roi_diag_pre_reference_enabled() {
  const char* env = std::getenv("SIMA_ROI_DIAG_PRE_REF");
  return env && *env && std::string(env) != "0";
}

bool roi_diag_hashes_enabled() {
  const char* env = std::getenv("SIMA_ROI_DIAG_HASHES");
  return env && *env && std::string(env) != "0";
}

void require_same_tensor_payload(const simaai::neat::Tensor& actual,
                                 const simaai::neat::Tensor& expected, const std::string& label) {
  require(actual.dtype == expected.dtype, label + ": dtype mismatch");
  require(actual.shape == expected.shape, label + ": shape mismatch");
  const std::vector<std::uint8_t> actual_bytes = tensor_payload(actual, label + " actual");
  const std::vector<std::uint8_t> expected_bytes = tensor_payload(expected, label + " expected");
  require(actual_bytes.size() == expected_bytes.size(), label + ": payload size mismatch");
  if (actual_bytes != expected_bytes) {
    std::size_t first = 0;
    while (first < actual_bytes.size() && actual_bytes[first] == expected_bytes[first]) {
      ++first;
    }
    throw std::runtime_error(
        label + ": payload bytes mismatch at offset " + std::to_string(first) +
        " actual=" + std::to_string(first < actual_bytes.size() ? actual_bytes[first] : 0) +
        " expected=" + std::to_string(first < expected_bytes.size() ? expected_bytes[first] : 0) +
        " actual_hash=" + std::to_string(payload_hash(actual_bytes)) +
        " expected_hash=" + std::to_string(payload_hash(expected_bytes)));
  }
}

void require_same_tensor_payload(const TensorPayloadSnapshot& actual,
                                 const TensorPayloadSnapshot& expected, const std::string& label,
                                 const TensorPayloadSnapshot* pre_reference = nullptr) {
  require(actual.dtype == expected.dtype, label + ": dtype mismatch");
  require(actual.shape == expected.shape, label + ": shape mismatch");
  require(actual.bytes.size() == expected.bytes.size(), label + ": payload size mismatch");
  if (actual.bytes != expected.bytes) {
    std::size_t first = 0;
    while (first < actual.bytes.size() && actual.bytes[first] == expected.bytes[first]) {
      ++first;
    }
    std::string msg =
        label + ": payload bytes mismatch at offset " + std::to_string(first) +
        " actual=" + std::to_string(first < actual.bytes.size() ? actual.bytes[first] : 0) +
        " expected=" + std::to_string(first < expected.bytes.size() ? expected.bytes[first] : 0) +
        " actual_hash=" + std::to_string(payload_hash(actual.bytes)) +
        " expected_hash=" + std::to_string(payload_hash(expected.bytes));
    if (pre_reference) {
      msg += " pre_ref_hash=" + std::to_string(payload_hash(pre_reference->bytes));
      msg += " actual_matches_pre=" +
             std::string(actual.bytes == pre_reference->bytes ? "true" : "false");
      msg += " post_matches_pre=" +
             std::string(expected.bytes == pre_reference->bytes ? "true" : "false");
    }
    throw std::runtime_error(msg);
  }
}

void require_same_tensor_payload(const TensorPayloadSnapshot& actual,
                                 const simaai::neat::Tensor& expected, const std::string& label) {
  require(actual.dtype == expected.dtype, label + ": dtype mismatch");
  require(actual.shape == expected.shape, label + ": shape mismatch");
  const std::vector<std::uint8_t> expected_bytes = tensor_payload(expected, label + " expected");
  require(actual.bytes.size() == expected_bytes.size(), label + ": payload size mismatch");
  if (actual.bytes != expected_bytes) {
    std::size_t first = 0;
    while (first < actual.bytes.size() && actual.bytes[first] == expected_bytes[first]) {
      ++first;
    }
    throw std::runtime_error(
        label + ": payload bytes mismatch at offset " + std::to_string(first) +
        " actual=" + std::to_string(first < actual.bytes.size() ? actual.bytes[first] : 0) +
        " expected=" + std::to_string(first < expected_bytes.size() ? expected_bytes[first] : 0) +
        " actual_hash=" + std::to_string(payload_hash(actual.bytes)) +
        " expected_hash=" + std::to_string(payload_hash(expected_bytes)));
  }
}

void require_roi_meta(const simaai::neat::Tensor& tensor,
                      const simaai::neat::PreprocessRoi& expected_roi,
                      int expected_source_batch_size, const std::string& label) {
  require(tensor.semantic.preprocess.has_value(), label + ": missing preprocess metadata");
  const auto& meta = *tensor.semantic.preprocess;
  require(meta.roi_list_enabled, label + ": ROI list metadata is not enabled");
  require(meta.rois.size() == 1U, label + ": expected scalar ROI metadata after split");
  const auto& roi = meta.rois.front();
  require(roi.batch_index == expected_roi.batch_index, label + ": ROI batch_index mismatch");
  require(roi.x == expected_roi.x && roi.y == expected_roi.y && roi.width == expected_roi.width &&
              roi.height == expected_roi.height,
          label + ": ROI rectangle mismatch");
  require(meta.roi_input_batch_size == expected_source_batch_size,
          label + ": source batch size metadata mismatch");
  require(meta.roi_source_width > 0 && meta.roi_source_height > 0,
          label + ": source geometry metadata missing");
  require(meta.roi_source_stride_bytes > 0, label + ": source stride metadata missing");
}

void require_center_letterbox_roi_geometry(const simaai::neat::Tensor& tensor,
                                           const simaai::neat::PreprocessRoi& roi, int target_w,
                                           int target_h, const std::string& label) {
  require(tensor.semantic.preprocess.has_value(), label + ": missing preprocess metadata");
  const auto& meta = *tensor.semantic.preprocess;
  require(meta.resize_mode == "letterbox", label + ": resize mode should be letterbox");
  require(roi.width > 0 && roi.height > 0 && target_w > 0 && target_h > 0,
          label + ": invalid geometry test inputs");
  int expected_scaled_w = target_w;
  int expected_scaled_h = target_h;
  const int64_t d =
      static_cast<int64_t>(roi.height) * target_w - static_cast<int64_t>(roi.width) * target_h;
  const auto ceil_div = [](int64_t n, int64_t den) -> int {
    return static_cast<int>((n + den - 1) / den);
  };
  if (d < 0) {
    expected_scaled_w = target_w;
    expected_scaled_h = ceil_div(static_cast<int64_t>(roi.height) * target_w, roi.width);
  } else {
    expected_scaled_w = ceil_div(static_cast<int64_t>(roi.width) * target_h, roi.height);
    expected_scaled_h = target_h;
  }
  const int expected_pad_left = std::max(0, target_w - expected_scaled_w) / 2;
  const int expected_pad_top = std::max(0, target_h - expected_scaled_h) / 2;
  require(meta.scaled_width == expected_scaled_w, label + ": scaled width metadata mismatch");
  require(meta.scaled_height == expected_scaled_h, label + ": scaled height metadata mismatch");
  require(meta.pad_left == expected_pad_left, label + ": pad_left metadata mismatch");
  require(meta.pad_top == expected_pad_top, label + ": pad_top metadata mismatch");
}

void require_pairwise_distinct_payloads(const std::vector<TensorPayloadSnapshot>& tensors,
                                        const std::string& label) {
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    for (std::size_t j = i + 1U; j < tensors.size(); ++j) {
      if (tensors[i].bytes == tensors[j].bytes) {
        throw std::runtime_error(label +
                                 ": synthetic ROI outputs should be pairwise distinct; hashes=" +
                                 std::to_string(payload_hash(tensors[i].bytes)) + "," +
                                 std::to_string(payload_hash(tensors[j].bytes)));
      }
    }
  }
}

simaai::neat::TensorList
run_roi_batch_and_compare(const std::vector<cv::Mat>& images, const simaai::neat::Model& model,
                          const std::vector<simaai::neat::PreprocessRoi>& rois,
                          const std::string& label, bool require_distinct = true,
                          const ExpectedOutputTraits& expected = {}) {
  std::vector<TensorPayloadSnapshot> pre_reference_payloads;
  if (roi_diag_pre_reference_enabled()) {
    pre_reference_payloads.reserve(rois.size());
    for (std::size_t i = 0; i < rois.size(); ++i) {
      const simaai::neat::PreprocessRoi ref_roi = single_image_roi(rois[i]);
      const cv::Mat& ref_image = images[static_cast<std::size_t>(rois[i].batch_index)];
      simaai::neat::TensorList reference =
          simaai::neat::stages::Preproc(std::vector<cv::Mat>{ref_image}, model,
                                        std::vector<simaai::neat::PreprocessRoi>{ref_roi});
      require(reference.size() == 1U,
              label + " pre-reference " + std::to_string(i) + ": output count mismatch");
      pre_reference_payloads.push_back(snapshot_tensor_payload(
          reference.front(), label + " pre-reference " + std::to_string(i)));
    }
  }

  simaai::neat::TensorList batched = simaai::neat::stages::Preproc(images, model, rois);
  require(batched.size() == rois.size(), label + ": output count mismatch");
  const std::vector<TensorPayloadSnapshot> batched_payloads =
      snapshot_tensor_payloads(batched, label + " batched");

  for (std::size_t i = 0; i < batched.size(); ++i) {
    require_roi_meta(batched[i], rois[i], static_cast<int>(images.size()),
                     label + " output " + std::to_string(i));
    require_output_traits(batched[i], expected, label + " output " + std::to_string(i));

    const simaai::neat::PreprocessRoi ref_roi = single_image_roi(rois[i]);
    const cv::Mat& ref_image = images[static_cast<std::size_t>(rois[i].batch_index)];
    simaai::neat::TensorList reference = simaai::neat::stages::Preproc(
        std::vector<cv::Mat>{ref_image}, model, std::vector<simaai::neat::PreprocessRoi>{ref_roi});
    require(reference.size() == 1U,
            label + " reference " + std::to_string(i) + ": output count mismatch");
    require_roi_meta(reference.front(), ref_roi, 1, label + " reference " + std::to_string(i));
    require_output_traits(reference.front(), expected, label + " reference " + std::to_string(i));
    TensorPayloadSnapshot post_reference =
        snapshot_tensor_payload(reference.front(), label + " reference " + std::to_string(i));
    const TensorPayloadSnapshot* pre_reference =
        (i < pre_reference_payloads.size()) ? &pre_reference_payloads[i] : nullptr;
    if (roi_diag_hashes_enabled()) {
      std::cerr << "[ROI_DIAG][HASH] " << label << " roi=" << i
                << " batched=" << payload_hash(batched_payloads[i].bytes)
                << " post_ref=" << payload_hash(post_reference.bytes);
      if (pre_reference) {
        std::cerr << " pre_ref=" << payload_hash(pre_reference->bytes);
      }
      std::cerr << "\n";
    }
    require_same_tensor_payload(batched_payloads[i], post_reference,
                                label + " output " + std::to_string(i), pre_reference);
  }

  if (require_distinct) {
    require_pairwise_distinct_payloads(batched_payloads, label);
  }
  return batched;
}

void run_roi_batch_repeat_consistency(const std::vector<cv::Mat>& images,
                                      const simaai::neat::Model& model,
                                      const std::vector<simaai::neat::PreprocessRoi>& rois,
                                      const std::string& label, int iterations,
                                      const ExpectedOutputTraits& expected = {}) {
  require(iterations > 1, label + ": repeat consistency requires at least two iterations");
  std::vector<TensorPayloadSnapshot> baseline;
  for (int iter = 0; iter < iterations; ++iter) {
    simaai::neat::TensorList out = simaai::neat::stages::Preproc(images, model, rois);
    require(out.size() == rois.size(), label + ": output count mismatch");
    for (std::size_t i = 0; i < out.size(); ++i) {
      require_roi_meta(out[i], rois[i], static_cast<int>(images.size()),
                       label + " iter " + std::to_string(iter) + " output " + std::to_string(i));
      require_output_traits(out[i], expected,
                            label + " iter " + std::to_string(iter) + " output " +
                                std::to_string(i));
    }
    const std::vector<TensorPayloadSnapshot> payloads =
        snapshot_tensor_payloads(out, label + " iter " + std::to_string(iter));
    if (roi_diag_hashes_enabled()) {
      for (std::size_t i = 0; i < payloads.size(); ++i) {
        std::cerr << "[ROI_DIAG][HASH] " << label << " iter=" << iter << " roi=" << i
                  << " hash=" << payload_hash(payloads[i].bytes) << "\n";
      }
    }
    if (iter == 0) {
      baseline = payloads;
      continue;
    }
    require(payloads.size() == baseline.size(), label + ": repeated output count changed");
    for (std::size_t i = 0; i < payloads.size(); ++i) {
      require_same_tensor_payload(payloads[i], baseline[i],
                                  label + " repeat iter " + std::to_string(iter) + " output " +
                                      std::to_string(i));
    }
  }
}

void run_roi_batch_dynamic_sequence_compare(
    const std::vector<cv::Mat>& images, const simaai::neat::Model& model,
    const std::vector<std::vector<simaai::neat::PreprocessRoi>>& sequences,
    const std::string& label, const ExpectedOutputTraits& expected = {}) {
  for (std::size_t i = 0; i < sequences.size(); ++i) {
    (void)run_roi_batch_and_compare(images, model, sequences[i],
                                    label + " sequence " + std::to_string(i),
                                    sequences[i].size() > 1U, expected);
  }
}

void run_full_frame_roi_equals_preproc(const std::vector<cv::Mat>& images,
                                       const simaai::neat::Model& model, const std::string& label,
                                       const ExpectedOutputTraits& expected = {}) {
  require(!images.empty(), label + ": expected at least one source image");
  const cv::Mat& image = images.front();
  const simaai::neat::PreprocessRoi roi{0, 0, 0, image.cols, image.rows};
  const std::vector<cv::Mat> single_image{image};
  simaai::neat::TensorList roi_out = simaai::neat::stages::Preproc(
      single_image, model, std::vector<simaai::neat::PreprocessRoi>{roi});
  simaai::neat::TensorList direct_out = simaai::neat::stages::Preproc(single_image, model);

  require(roi_out.size() == 1U, label + ": full-frame ROI output count mismatch");
  require(direct_out.size() == 1U, label + ": direct preproc output count mismatch");
  require_roi_meta(roi_out.front(), roi, 1, label + " full-frame ROI");
  require_output_traits(roi_out.front(), expected, label + " full-frame ROI");
  require_output_traits(direct_out.front(), expected, label + " direct preproc");
  require_same_tensor_payload(roi_out.front(), direct_out.front(), label);
}

} // namespace

int main(int argc, char** argv) {
  try {
#if !SIMA_HAS_SIMAAI_POOL
    skip_long_test_exception("preproc ROI batch functional test requires simaai pool/meta");
#endif

    const fs::path root = (argc > 1) ? fs::path(argv[1]) : sima_test::test_source_root();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const auto fixture = make_roi_batch_preproc_fixture();

    simaai::neat::Model model(fixture.tar_path, make_roi_batch_model_options());

    std::vector<cv::Mat> images;
    images.push_back(make_test_image(64, 48, 3));
    images.push_back(make_test_image(64, 48, 29));

    const ExpectedOutputTraits dense_bf16_traits{.enabled = true,
                                                 .normalize = true,
                                                 .tessellate = false,
                                                 .quantize = false,
                                                 .dtype = simaai::neat::TensorDType::BFloat16};
    const ExpectedOutputTraits tess_bf16_traits{.enabled = true,
                                                .normalize = true,
                                                .tessellate = true,
                                                .quantize = false,
                                                .dtype = simaai::neat::TensorDType::BFloat16};
    const ExpectedOutputTraits tess_int8_traits{.enabled = true,
                                                .normalize = true,
                                                .tessellate = true,
                                                .quantize = true,
                                                .dtype = simaai::neat::TensorDType::Int8};

    bool any_case_failed = false;
    const bool smoke_mode = SIMA_ROI_USER_SMOKE_DEFAULT || env_flag("SIMA_ROI_USER_SMOKE");
    if (smoke_mode) {
      std::cerr << "[ROI_DIAG][MODE] user smoke\n";
    }
    const char* filter_env = std::getenv("SIMA_ROI_DIAG_CASE");
    const std::string case_filter = filter_env ? filter_env : "";
    auto case_enabled = [&](const std::string& label) {
      return case_filter.empty() || label.find(case_filter) != std::string::npos;
    };
    auto run_checked = [&](const std::string& label, auto&& fn) {
      if (!case_enabled(label)) {
        return;
      }
      try {
        fn();
        std::cerr << "[ROI_DIAG][PASS] " << label << "\n";
      } catch (const std::exception& e) {
        any_case_failed = true;
        std::cerr << "[ROI_DIAG][FAIL] " << label << ": " << e.what() << "\n";
      }
    };
    auto run_case = [&](const std::string& label,
                        const std::vector<simaai::neat::PreprocessRoi>& case_rois,
                        bool require_distinct = true, const ExpectedOutputTraits& expected = {}) {
      run_checked(label, [&] {
        (void)run_roi_batch_and_compare(images, model, case_rois, label, require_distinct,
                                        expected);
      });
    };
    auto run_repeat_case = [&](const std::string& label,
                               const std::vector<simaai::neat::PreprocessRoi>& case_rois,
                               int iterations = 4, const ExpectedOutputTraits& expected = {}) {
      run_checked(label, [&] {
        run_roi_batch_repeat_consistency(images, model, case_rois, label, iterations, expected);
      });
    };
    auto run_full_frame_case = [&](const std::string& label, const ExpectedOutputTraits& expected) {
      run_checked(label,
                  [&] { run_full_frame_roi_equals_preproc(images, model, label, expected); });
    };
    auto run_letterbox_case = [&](const std::string& label,
                                  const std::vector<simaai::neat::PreprocessRoi>& case_rois) {
      run_checked(label, [&] {
        const auto letterbox_fixture = make_roi_batch_preproc_letterbox_fixture();
        simaai::neat::Model letterbox_model(letterbox_fixture.tar_path,
                                            make_roi_batch_letterbox_model_options());
        simaai::neat::TensorList out =
            run_roi_batch_and_compare(images, letterbox_model, case_rois, label,
                                      /*require_distinct=*/true, dense_bf16_traits);
        require(out.size() == case_rois.size(), label + ": output count mismatch");
        for (std::size_t i = 0; i < out.size(); ++i) {
          require_center_letterbox_roi_geometry(out[i], case_rois[i], 64, 48,
                                                label + " output " + std::to_string(i));
        }
      });
    };
    auto run_resize_type_case = [&](const std::string& label, const std::string& scaling_type) {
      run_checked(label, [&] {
        const auto resize_fixture = make_roi_batch_preproc_resize_fixture(
            "preproc_roi_batch_resize_" + scaling_type, scaling_type,
            /*aspect_ratio=*/false);
        simaai::neat::Model resize_model(
            resize_fixture.tar_path,
            make_roi_batch_resize_model_options(scaling_type, simaai::neat::ResizeMode::Stretch));
        (void)run_roi_batch_and_compare(images, resize_model,
                                        {{0, 0, 0, 32, 24}, {1, 4, 3, 48, 36}, {0, -4, 8, 40, 32}},
                                        label, /*require_distinct=*/true, dense_bf16_traits);
      });
    };

    if (!smoke_mode) {
      run_case("ROI isolate single ROI image0", {{0, 0, 0, 32, 24}});
      run_case("ROI isolate single ROI image1", {{1, 16, 10, 32, 24}});
      run_repeat_case("ROI isolate single ROI image0 repeated [repeat_image0]",
                      {{0, 0, 0, 32, 24}});
      run_repeat_case("ROI isolate same-image mixed-size repeated [repeat_same_mixed]",
                      {{0, 0, 0, 32, 24}, {0, 8, 8, 40, 32}});
      run_case("ROI isolate same-image same-size two ROI",
               {{0, 0, 0, 32, 24}, {0, 16, 10, 32, 24}});
      run_case("ROI isolate cross-image same-size two ROI",
               {{0, 0, 0, 32, 24}, {1, 16, 10, 32, 24}});
      run_case("ROI isolate same-image mixed-size two ROI [same_mixed_fwd]",
               {{0, 0, 0, 32, 24}, {0, 8, 8, 40, 32}});
      run_case("ROI isolate same-image mixed-size two ROI reversed [same_mixed_rev]",
               {{0, 8, 8, 40, 32}, {0, 0, 0, 32, 24}});
      run_case("ROI isolate cross-image mixed-size two ROI [cross_mixed_fwd]",
               {{0, 0, 0, 32, 24}, {1, 8, 8, 40, 32}});
    }

    const std::vector<simaai::neat::PreprocessRoi> rois = {
        {0, 0, 0, 32, 24},
        {1, 16, 10, 32, 24},
        {0, 8, 8, 40, 32},
    };

    run_case("ROI batch functional", rois, /*require_distinct=*/true, dense_bf16_traits);

    const std::vector<simaai::neat::PreprocessRoi> padded_rois = {
        {0, -8, -6, 40, 32},
        {1, 54, 36, 28, 20},
        {0, -4, 30, 72, 28},
    };
    run_case("ROI batch padded/out-of-frame functional", padded_rois,
             /*require_distinct=*/false, dense_bf16_traits);

    const std::vector<std::vector<simaai::neat::PreprocessRoi>> dynamic_sequences = {
        rois,
        {{1, 16, 10, 32, 24}},
        {{0, 0, 0, 32, 24}, {1, 20, 12, 32, 24}},
        rois,
    };
    if (smoke_mode) {
      run_case("ROI batch dynamic shrink [smoke_dynamic]", dynamic_sequences[1],
               /*require_distinct=*/false, dense_bf16_traits);
    } else {
      for (std::size_t i = 0; i < dynamic_sequences.size(); ++i) {
        run_case("ROI batch dynamic sequence " + std::to_string(i), dynamic_sequences[i],
                 dynamic_sequences[i].size() > 1U);
      }
      run_full_frame_case("ROI full-frame equals direct preproc dense BF16 [full_frame_dense]",
                          dense_bf16_traits);
    }
    run_letterbox_case("ROI batch letterbox/aspect-ratio functional [letterbox]",
                       {{0, 0, 0, 48, 24}, {1, 8, 0, 32, 48}});
    run_resize_type_case("ROI batch resize NEAREST functional [resize_nearest]",
                         "NEAREST_NEIGHBOUR");
    run_resize_type_case("ROI batch resize INTERAREA functional [resize_interarea]", "INTERAREA");
    run_resize_type_case("ROI batch resize BICUBIC functional [resize_bicubic]", "BICUBIC");
    run_resize_type_case("ROI batch resize NO_SCALING functional [resize_no_scaling]",
                         "NO_SCALING");

    auto run_model_case = [&](const std::string& label,
                              const sima_test::ModelArchiveFixture& route_fixture,
                              const simaai::neat::Model::Options& route_options,
                              const ExpectedOutputTraits& expected) {
      run_checked(label, [&] {
        simaai::neat::Model route_model(route_fixture.tar_path, route_options);
        (void)run_roi_batch_and_compare(images, route_model, rois, label,
                                        /*require_distinct=*/true, expected);
      });
    };
    auto run_model_repeat_case = [&](const std::string& label,
                                     const sima_test::ModelArchiveFixture& route_fixture,
                                     const simaai::neat::Model::Options& route_options,
                                     const ExpectedOutputTraits& expected) {
      run_checked(label, [&] {
        simaai::neat::Model route_model(route_fixture.tar_path, route_options);
        run_roi_batch_repeat_consistency(images, route_model, rois, label,
                                         /*iterations=*/4, expected);
      });
    };
    auto run_model_dynamic_case = [&](const std::string& label,
                                      const sima_test::ModelArchiveFixture& route_fixture,
                                      const simaai::neat::Model::Options& route_options,
                                      const ExpectedOutputTraits& expected) {
      run_checked(label, [&] {
        simaai::neat::Model route_model(route_fixture.tar_path, route_options);
        run_roi_batch_dynamic_sequence_compare(images, route_model, dynamic_sequences, label,
                                               expected);
      });
    };
    auto run_model_full_frame_case = [&](const std::string& label,
                                         const sima_test::ModelArchiveFixture& route_fixture,
                                         const simaai::neat::Model::Options& route_options,
                                         const ExpectedOutputTraits& expected) {
      run_checked(label, [&] {
        simaai::neat::Model route_model(route_fixture.tar_path, route_options);
        run_full_frame_roi_equals_preproc(images, route_model, label, expected);
      });
    };

    const auto normalized_route_options = make_roi_batch_normalized_preproc_model_options();
    const auto tess_bf16_fixture = make_roi_batch_preproc_tess_bf16_fixture();
    const auto tess_int8_fixture = make_roi_batch_preproc_quanttess_int8_fixture();

    run_model_case("ROI batch normalized tessellated BF16 functional [tess_bf16]",
                   tess_bf16_fixture, normalized_route_options, tess_bf16_traits);
    if (!smoke_mode) {
      run_model_repeat_case("ROI batch normalized tessellated BF16 repeated [repeat_tess_bf16]",
                            tess_bf16_fixture, normalized_route_options, tess_bf16_traits);
      run_model_dynamic_case("ROI batch normalized tessellated BF16 dynamic [dynamic_tess_bf16]",
                             tess_bf16_fixture, normalized_route_options, tess_bf16_traits);
      run_model_full_frame_case(
          "ROI full-frame equals direct preproc tessellated BF16 [full_frame_tess_bf16]",
          tess_bf16_fixture, normalized_route_options, tess_bf16_traits);
    }

    run_model_case("ROI batch normalized tessellated INT8 functional [tess_int8]",
                   tess_int8_fixture, normalized_route_options, tess_int8_traits);
    if (!smoke_mode) {
      run_model_repeat_case("ROI batch normalized tessellated INT8 repeated [repeat_tess_int8]",
                            tess_int8_fixture, normalized_route_options, tess_int8_traits);
      run_model_dynamic_case("ROI batch normalized tessellated INT8 dynamic [dynamic_tess_int8]",
                             tess_int8_fixture, normalized_route_options, tess_int8_traits);
      run_model_full_frame_case(
          "ROI full-frame equals direct preproc tessellated INT8 [full_frame_tess_int8]",
          tess_int8_fixture, normalized_route_options, tess_int8_traits);
    }

    if (any_case_failed) {
      std::cerr << "[ERR] preproc_roi_batch_functional_test: one or more ROI cases failed\n";
      return 1;
    }

    std::cout << "[OK] "
              << (smoke_mode ? "preproc_roi_user_smoke_test" : "preproc_roi_batch_functional_test")
              << " passed\n";
    return 0;
  } catch (const SkipTest& e) {
    std::cout << "[SKIP] " << e.what() << "\n";
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    if (is_dispatcher_unavailable(e.what())) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}

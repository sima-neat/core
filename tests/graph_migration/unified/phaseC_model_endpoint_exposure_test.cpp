#include "model/Model.h"
#include "model_archive_fixture_utils.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <opencv2/core/mat.hpp>

namespace {

sima_test::ModelArchiveFixture make_multi_ingress_model_fixture(const std::string& tag) {
  return sima_test::make_model_archive_fixture(tag, {{"etc/phaseC_multi_ingress_mpk.json", R"json({
  "name": "phaseC_multi_ingress",
  "model_path": "phaseC_multi_ingress.onnx",
  "model_sdk_version": "2.0.0",
  "sequence": 1,
  "input_nodes": [
    { "name": "image_l",  "type": "buffer", "size": 64 },
    { "name": "image_uv", "type": "buffer", "size": 32 }
  ],
  "plugins": [
    {
      "name": "quantize_l",
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
        { "name": "image_l", "size": 64, "logical_shape": [1, 4, 4, 1], "logical_dtype": "FP32" }
      ],
      "output_nodes": [
        { "name": "quantize_l", "type": "buffer", "size": 16,
          "logical_shape": [1, 4, 4, 1], "logical_dtype": "INT8" }
      ],
      "type": "sgpProcess"
    },
    {
      "name": "quantize_uv",
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
        { "name": "image_uv", "size": 32, "logical_shape": [1, 2, 4, 1], "logical_dtype": "FP32" }
      ],
      "output_nodes": [
        { "name": "quantize_uv", "type": "buffer", "size": 8,
          "logical_shape": [1, 2, 4, 1], "logical_dtype": "INT8" }
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
        { "name": "quantize_l",  "size": 16,
          "logical_shape": [1, 4, 4, 1], "logical_dtype": "INT8" },
        { "name": "quantize_uv", "size": 8,
          "logical_shape": [1, 2, 4, 1], "logical_dtype": "INT8" }
      ],
      "output_nodes": [
        { "name": "MLA_0_ifm_pack_transform", "type": "buffer", "size": 24,
          "logical_shape": [1, 24], "logical_dtype": "INT8" }
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
        "number_of_quads_to_user": 1,
        "input_shapes": [[1, 24]],
        "input_data_type": ["INT8"],
        "output_shapes": [[1, 4, 4, 1]],
        "data_type": ["INT8"]
      },
      "input_nodes": [
        { "name": "MLA_0_ifm_pack_transform", "size": 24,
          "logical_shape": [1, 24], "logical_dtype": "INT8" }
      ],
      "output_nodes": [
        { "name": "classes", "type": "buffer", "size": 16,
          "logical_shape": [1, 4, 4, 1], "logical_dtype": "INT8" }
      ],
      "type": "sgpProcess",
      "resources": { "executable": "stage0.elf" }
    }
  ]
})json"},
                                                     {"etc/pipeline_sequence.json", R"json({
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
        "name": "MLA_0",
        "pluginId": "processmla",
        "configPath": "0_process_mla.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "MLA_0_ifm_pack_transform"
      }
    ]
  }]
})json"},
                                                     {"etc/0_process_mla.json", R"json({
  "node_name": "MLA_0",
  "input_buffers": [{ "name": "MLA_0_ifm_pack_transform" }],
  "input_format": ["EV81_INT8"],
  "data_type": ["EV81_INT8"],
  "input_width": [24],
  "input_height": [1],
  "input_depth": [1],
  "output_width": [4],
  "output_height": [4],
  "output_depth": [1]
})json"}});
}

void require_contains_local(const std::string& haystack, const std::string& needle,
                            const std::string& label) {
  if (haystack.find(needle) == std::string::npos) {
    throw std::runtime_error(label + ": expected '" + needle + "' in:\n" + haystack);
  }
}

std::filesystem::path tmp_path(const std::string& name) {
  const char* tmp = std::getenv("TMPDIR");
  return std::filesystem::path(tmp ? tmp : "/tmp") / name;
}

void require_file_contains(const std::filesystem::path& path, const std::string& needle,
                           const std::string& label) {
  std::ifstream in(path);
  const std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (body.find(needle) == std::string::npos) {
    throw std::runtime_error(label + ": missing '" + needle + "' in\n" + body);
  }
}

} // namespace

RUN_TEST("graph_migration_phaseC_model_endpoint_exposure_test", [] {
  using namespace simaai::neat;

  const auto fixture = make_multi_ingress_model_fixture("phaseC_model_endpoint_exposure");
  Model::Options model_opt;
  model_opt.name_suffix = "_phasec";
  model_opt.processcvu.async = false;
  model_opt.async_queue_depth = 3;
  Model model(fixture.tar_path, model_opt);

  const Graph model_fragment_for_names = model.graph();
  const auto model_inputs = model_fragment_for_names.inputs();
  require(std::find(model_inputs.begin(), model_inputs.end(), "image_l") != model_inputs.end(),
          "model.graph().inputs() should expose image_l");
  require(std::find(model_inputs.begin(), model_inputs.end(), "image_uv") != model_inputs.end(),
          "model.graph().inputs() should expose image_uv");

  Graph left;
  left.add(nodes::Input("image_l"));

  Graph right;
  right.add(nodes::Input("image_uv"));

  Graph classes;
  classes.add(nodes::Output("classes"));

  Graph app("model_endpoints");
  app.connect(left, model);
  app.connect(right, model);
  app.connect(model, classes);

  const auto app_inputs = app.inputs();
  require(app_inputs.size() == 2U &&
              std::find(app_inputs.begin(), app_inputs.end(), "image_l") != app_inputs.end() &&
              std::find(app_inputs.begin(), app_inputs.end(), "image_uv") != app_inputs.end(),
          "composed multi-ingress app should expose only external image_l/image_uv inputs");
  const auto app_outputs = app.outputs();
  require(app_outputs.size() == 1U && app_outputs.front() == "classes",
          "composed multi-ingress app should expose only external classes output");

  const std::string public_description = app.describe();
  require_contains_local(public_description, "endpoint image_l -> image_l",
                         "left ingress should resolve by model endpoint name");
  require_contains_local(public_description, "endpoint image_uv -> image_uv",
                         "right ingress should resolve by model endpoint name");

  const std::string backend = app.describe_backend(false);
  require_contains_local(backend, "JoinBundle",
                         "multi-ingress model edges should lower through JoinBundle");
  require_contains_local(backend, "image_l", "backend plan should preserve image_l endpoint");
  require_contains_local(backend, "image_uv", "backend plan should preserve image_uv endpoint");

  const GraphReport report = app.validate();
  require(report.error_code.empty(), "multi-ingress model endpoint validate should pass, got " +
                                         report.error_code + ": " + report.repro_note);

  const std::filesystem::path saved_path = tmp_path("phaseC_model_endpoint_roundtrip.neat.json");
  std::filesystem::remove(saved_path);
  app.save(saved_path.string());
  require_file_contains(saved_path, "\"from_endpoint\":\"image_l\"",
                        "model endpoint save should preserve left endpoint edge");
  require_file_contains(saved_path, "\"from_endpoint\":\"image_uv\"",
                        "model endpoint save should preserve right endpoint edge");
  require_file_contains(saved_path, "\"model_fragments\"",
                        "model endpoint save should preserve model fragment provenance");
  require_file_contains(saved_path, "\"source_path\"",
                        "model endpoint save should preserve model source path");
  require_file_contains(saved_path, "\"model_options\"",
                        "model endpoint save should preserve model construction options");
  require_file_contains(saved_path, "\"route_options\"",
                        "model endpoint save should preserve model route options");
  require_file_contains(saved_path, "\"name_suffix\":\"_phasec\"",
                        "model endpoint save should preserve model name suffix option");
  require_file_contains(saved_path, "\"async_queue_depth\":3",
                        "model endpoint save should preserve model async queue depth option");

  Graph loaded = Graph::load(saved_path.string());
  const std::string loaded_description = loaded.describe();
  require_contains_local(loaded_description, "endpoint image_l -> image_l",
                         "loaded graph should preserve left model endpoint edge");
  require_contains_local(loaded_description, "endpoint image_uv -> image_uv",
                         "loaded graph should preserve right model endpoint edge");
  const GraphReport loaded_report = loaded.validate();
  require(loaded_report.error_code.empty(),
          "loaded graph should rehydrate model ingress provenance, got " +
              loaded_report.error_code + ": " + loaded_report.repro_note);
  const std::filesystem::path resaved_path = tmp_path("phaseC_model_endpoint_resaved.neat.json");
  std::filesystem::remove(resaved_path);
  loaded.save(resaved_path.string());
  require_file_contains(resaved_path, "\"model_options\"",
                        "loaded graph should preserve model options when saved again");
  require_file_contains(resaved_path, "\"name_suffix\":\"_phasec\"",
                        "loaded graph should preserve model option name suffix when resaved");
  require_file_contains(resaved_path, "\"route_options\"",
                        "loaded graph should preserve route options when saved again");

  Model::RouteOptions route_opt;
  route_opt.name_suffix = "_route";
  route_opt.buffer_name = "route_pool";
  route_opt.processmla.output_pool_buffers = 6;
  route_opt.async_queue_depth = 5;
  Graph left_route;
  left_route.add(nodes::Input("image_l"));
  Graph right_route;
  right_route.add(nodes::Input("image_uv"));
  Graph route = model.graph(route_opt);
  Graph route_classes;
  route_classes.add(nodes::Output("classes"));
  Graph route_app("model_route_options");
  route_app.connect(left_route, route);
  route_app.connect(right_route, route);
  route_app.connect(route, route_classes);

  const std::filesystem::path route_saved_path =
      tmp_path("phaseC_model_route_options_roundtrip.neat.json");
  std::filesystem::remove(route_saved_path);
  route_app.save(route_saved_path.string());
  require_file_contains(route_saved_path, "\"route_options\"",
                        "explicit model Graph save should preserve route options");
  require_file_contains(route_saved_path, "\"name_suffix\":\"_route\"",
                        "explicit model Graph save should preserve route name suffix");
  require_file_contains(route_saved_path, "\"buffer_name\":\"route_pool\"",
                        "explicit model Graph save should preserve route buffer name");
  require_file_contains(route_saved_path, "\"output_pool_buffers\":6",
                        "explicit model Graph save should preserve MLA route options");
  require_file_contains(route_saved_path, "\"async_queue_depth\":5",
                        "explicit model Graph save should preserve route async queue depth");

  Graph loaded_route_app = Graph::load(route_saved_path.string());
  const GraphReport loaded_route_report = loaded_route_app.validate();
  require(loaded_route_report.error_code.empty(),
          "loaded explicit route Graph should rehydrate model route options, got " +
              loaded_route_report.error_code + ": " + loaded_route_report.repro_note);
  const std::filesystem::path route_resaved_path =
      tmp_path("phaseC_model_route_options_resaved.neat.json");
  std::filesystem::remove(route_resaved_path);
  loaded_route_app.save(route_resaved_path.string());
  require_file_contains(route_resaved_path, "\"model_options\"",
                        "loaded explicit route Graph should preserve model options when resaved");
  require_file_contains(route_resaved_path, "\"route_options\"",
                        "loaded explicit route Graph should preserve route options when resaved");
  require_file_contains(
      route_resaved_path, "\"name_suffix\":\"_route\"",
      "loaded explicit route Graph should preserve route name suffix when resaved");
  require_file_contains(
      route_resaved_path, "\"async_queue_depth\":5",
      "loaded explicit route Graph should preserve route async depth when resaved");

  Model::RouteOptions linear_route_opt;
  linear_route_opt.include_input = true;
  linear_route_opt.include_output = true;
  Graph linear_model = model.graph(linear_route_opt);
  linear_model.set_name("linear_model");
  ValidateOptions linear_validate_opt;
  linear_validate_opt.parse_launch = false;
  const cv::Mat real_validate_input = cv::Mat::zeros(1, 96, CV_8UC1);
  const GraphReport linear_report = linear_model.validate(linear_validate_opt, real_validate_input);
  require(linear_report.error_code.empty(),
          "linear model Graph with explicit Input/Output should validate, got " +
              linear_report.error_code + ": " + linear_report.repro_note);
  const std::filesystem::path linear_saved_path = tmp_path("phaseC_add_model_roundtrip.neat.json");
  std::filesystem::remove(linear_saved_path);
  linear_model.save(linear_saved_path.string());
  require_file_contains(linear_saved_path, "\"model_fragments\"",
                        "linear model Graph save should preserve model fragment provenance");
  require_file_contains(linear_saved_path, "\"source_path\"",
                        "linear model Graph save should preserve model source path");
  require_file_contains(linear_saved_path, "\"model_options\"",
                        "linear model Graph save should preserve model construction options");
  Graph loaded_linear_model = Graph::load(linear_saved_path.string());
  const GraphReport loaded_linear_report =
      loaded_linear_model.validate(linear_validate_opt, real_validate_input);
  require(loaded_linear_report.error_code.empty(),
          "loaded linear model Graph should validate, got " + loaded_linear_report.error_code +
              ": " + loaded_linear_report.repro_note);

  Graph preprocess_fragment = model.preprocess();
  const std::filesystem::path pre_saved_path =
      tmp_path("phaseC_model_preprocess_fragment.neat.json");
  std::filesystem::remove(pre_saved_path);
  preprocess_fragment.save(pre_saved_path.string());
  require_file_contains(pre_saved_path, "\"model_fragments\"",
                        "Model::preprocess Graph save should preserve model fragment provenance");
  require_file_contains(pre_saved_path, "\"stage_role\":\"preprocess\"",
                        "Model::preprocess Graph save should preserve stage role");
  require_file_contains(pre_saved_path, "\"source_path\"",
                        "Model::preprocess Graph save should preserve source path");
  require_file_contains(pre_saved_path, "\"model_options\"",
                        "Model::preprocess Graph save should preserve model options");
  Graph loaded_preprocess_fragment = Graph::load(pre_saved_path.string());
  const std::filesystem::path pre_resaved_path =
      tmp_path("phaseC_model_preprocess_fragment_resaved.neat.json");
  std::filesystem::remove(pre_resaved_path);
  loaded_preprocess_fragment.save(pre_resaved_path.string());
  require_file_contains(pre_resaved_path, "\"stage_role\":\"preprocess\"",
                        "loaded Model::preprocess Graph should preserve stage role when resaved");
  require_file_contains(
      pre_resaved_path, "\"model_options\"",
      "loaded Model::preprocess Graph should preserve model options when resaved");

  Graph infer_fragment = model.fragment(Model::Stage::Inference);
  const std::filesystem::path infer_saved_path =
      tmp_path("phaseC_model_inference_fragment.neat.json");
  std::filesystem::remove(infer_saved_path);
  infer_fragment.save(infer_saved_path.string());
  require_file_contains(infer_saved_path, "\"stage_role\":\"inference\"",
                        "Model::fragment(Inference) save should preserve stage role");
  Graph loaded_infer_fragment = Graph::load(infer_saved_path.string());
  const std::filesystem::path infer_resaved_path =
      tmp_path("phaseC_model_inference_fragment_resaved.neat.json");
  std::filesystem::remove(infer_resaved_path);
  loaded_infer_fragment.save(infer_resaved_path.string());
  require_file_contains(
      infer_resaved_path, "\"stage_role\":\"inference\"",
      "loaded Model::fragment(Inference) should preserve stage role when resaved");

  Graph wrong;
  wrong.add(nodes::Input("wrong"));
  Graph bad_app;
  bool threw = false;
  try {
    bad_app.connect(wrong, model);
  } catch (const std::exception& e) {
    threw = true;
    require_contains(std::string(e.what()), "image_l",
                     "bad model endpoint diagnostic should list image_l");
    require_contains(std::string(e.what()), "image_uv",
                     "bad model endpoint diagnostic should list image_uv");
  }
  require(threw, "wrongly named source should not silently bind to a multi-ingress model");

  Graph renamed_left;
  renamed_left.add(nodes::Input("a_new_name_image_l"));
  Graph renamed_bad_app;
  threw = false;
  try {
    renamed_bad_app.connect(renamed_left, model);
  } catch (const std::exception& e) {
    threw = true;
    require_contains(std::string(e.what()), "image_l",
                     "renamed model endpoint diagnostic should list image_l");
    require_contains(std::string(e.what()), "image_uv",
                     "renamed model endpoint diagnostic should list image_uv");
  }
  require(threw, "renamed source endpoint should not implicitly bind to model input image_l");

  Graph duplicate_left_a;
  duplicate_left_a.add(nodes::Input("image_l"));
  Graph duplicate_left_b;
  duplicate_left_b.add(nodes::Input("image_l"));
  Graph duplicate_app;
  duplicate_app.connect(duplicate_left_a, model);
  threw = false;
  try {
    duplicate_app.connect(duplicate_left_b, model);
  } catch (const std::exception& e) {
    threw = true;
    require_contains(std::string(e.what()), "already connected",
                     "duplicate model input diagnostic should mention endpoint occupancy");
  }
  require(threw, "same model input endpoint should not accept two direct sources");
});

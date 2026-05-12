#include "model/Model.h"
#include "model/internal/InputPlanner.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/RoutePlanner.h"
#include "mpk_fixture_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <vector>

namespace {

sima_test::MpkFixture make_bf16_mla_tess_fixture(const std::string& tag) {
  return sima_test::make_strict_mpk_tar_fixture(tag,
                                                {
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
                                                    {"etc/0_preproc.json",
                                                     R"json({
  "node_name": "preproc_0",
  "input_width": 1280,
  "input_height": 720,
  "input_img_type": "RGB",
  "output_width": 640,
  "output_height": 640,
  "output_img_type": "RGB",
  "dynamic_input_dims": true
})json"},
                                                    {"etc/0_process_mla.json",
                                                     R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "preproc_0"}],
  "input_format": ["EV81_BFLOAT16"],
  "data_type": ["EV81_BFLOAT16"],
  "input_width": [640],
  "input_height": [640],
  "input_depth": [3],
  "output_width": [80],
  "output_height": [80],
  "output_depth": [6]
})json"},
                                                },
                                                true);
}

sima_test::MpkFixture make_quanttess_post_fixture(const std::string& tag) {
  return sima_test::make_strict_mpk_tar_fixture(tag,
                                                {
                                                    {"etc/pipeline_sequence.json",
                                                     R"json({
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
        "name": "quanttess_0",
        "pluginId": "processcvu",
        "configPath": "0_quanttess.json",
        "processor": "CVU",
        "kernel": "quanttess",
        "input": "decoder"
      },
      {
        "sequence_id": 2,
        "name": "mla_0",
        "pluginId": "processmla",
        "configPath": "0_process_mla.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "quanttess_0"
      },
      {
        "sequence_id": 3,
        "name": "detessdequant_0",
        "pluginId": "processcvu",
        "configPath": "0_postproc.json",
        "processor": "CVU",
        "kernel": "detessdequant",
        "input": "mla_0"
      }
    ]
  }]
})json"},
                                                    {"etc/0_preproc.json",
                                                     R"json({
  "node_name": "preproc_0",
  "input_width": 1280,
  "input_height": 720,
  "input_img_type": "RGB",
  "output_width": 640,
  "output_height": 640,
  "output_img_type": "RGB",
  "dynamic_input_dims": true
})json"},
                                                    {"etc/0_quanttess.json",
                                                     R"json({
  "node_name": "quanttess_0",
  "input_width": 640,
  "input_height": 640,
  "input_depth": 3
})json"},
                                                    {"etc/0_process_mla.json",
                                                     R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "quanttess_0"}],
  "input_format": ["EV81_INT8"],
  "data_type": ["EV81_INT8"],
  "input_width": [640],
  "input_height": [640],
  "input_depth": [3],
  "output_width": [80],
  "output_height": [80],
  "output_depth": [6]
})json"},
                                                    {"etc/0_postproc.json",
                                                     R"json({
  "node_name": "detessdequant_0",
  "num_in_tensor": 1,
  "out_data_type": "FP32",
  "input_width": [80],
  "input_height": [80],
  "input_depth": [6]
})json"},
                                                },
                                                true);
}

sima_test::MpkFixture make_quant_no_post_fixture(const std::string& tag) {
  return sima_test::make_strict_mpk_tar_fixture(tag,
                                                {
                                                    {"etc/pipeline_sequence.json",
                                                     R"json({
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
        "name": "quanttess_0",
        "pluginId": "processcvu",
        "configPath": "0_quanttess.json",
        "processor": "CVU",
        "kernel": "quanttess",
        "input": "decoder"
      },
      {
        "sequence_id": 2,
        "name": "mla_0",
        "pluginId": "processmla",
        "configPath": "0_process_mla.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "quanttess_0"
      },
      {
        "sequence_id": 3,
        "name": "detessdequant_0",
        "pluginId": "processcvu",
        "configPath": "0_postproc.json",
        "processor": "CVU",
        "kernel": "detessdequant",
        "input": "mla_0"
      }
    ]
  }]
})json"},
                                                    {"etc/0_preproc.json",
                                                     R"json({
  "node_name": "preproc_0",
  "input_width": 1280,
  "input_height": 720,
  "input_img_type": "RGB",
  "output_width": 640,
  "output_height": 640,
  "output_img_type": "RGB",
  "dynamic_input_dims": true
})json"},
                                                    {"etc/0_quanttess.json",
                                                     R"json({
  "node_name": "quanttess_0",
  "input_width": 640,
  "input_height": 640,
  "input_depth": 3
})json"},
                                                    {"etc/0_process_mla.json",
                                                     R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "quanttess_0"}],
  "input_format": ["EV81_INT8"],
  "data_type": ["EV81_INT8"],
  "input_width": [640],
  "input_height": [640],
  "input_depth": [3],
  "output_width": [80],
  "output_height": [80],
  "output_depth": [6]
})json"},
                                                    {"etc/0_postproc.json",
                                                     R"json({
  "node_name": "detessdequant_0",
  "num_in_tensor": 1,
  "out_data_type": "FP32",
  "input_width": [80],
  "input_height": [80],
  "input_depth": [6]
})json"},
                                                },
                                                true);
}

sima_test::MpkFixture make_ambiguous_tess_fixture(const std::string& tag) {
  return sima_test::make_strict_mpk_tar_fixture(tag,
                                                {
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
      },
      {
        "sequence_id": 3,
        "name": "detessdequant_0",
        "pluginId": "processcvu",
        "configPath": "0_detessdequant.json",
        "processor": "CVU",
        "kernel": "detessdequant",
        "input": "mla_0"
      }
    ]
  }]
})json"},
                                                    {"etc/0_preproc.json",
                                                     R"json({
  "node_name": "preproc_0",
  "input_width": 1280,
  "input_height": 720,
  "input_img_type": "RGB",
  "output_width": 640,
  "output_height": 640,
  "output_img_type": "RGB",
  "dynamic_input_dims": true
})json"},
                                                    {"etc/0_process_mla.json",
                                                     R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "preproc_0"}],
  "input_format": ["EV81_BFLOAT16"],
  "data_type": ["EV81_BFLOAT16"],
  "input_width": [640],
  "input_height": [640],
  "input_depth": [3],
  "output_width": [80],
  "output_height": [80],
  "output_depth": [6]
})json"},
                                                    {"etc/0_detessdequant.json",
                                                     R"json({
  "node_name": "detessdequant_0",
  "num_in_tensor": 1,
  "out_data_type": "FP32",
  "input_width": [80],
  "input_height": [80],
  "input_depth": [6]
})json"},
                                                },
                                                true);
}

sima_test::MpkFixture make_multi_ingress_cast_join_fixture(const std::string& tag) {
  return sima_test::make_mpk_tar_fixture(tag, {
                                                  {"etc/multi_ingress_cast_join_mpk.json",
                                                   R"json({
  "name": "multi_ingress_cast_join",
  "model_path": "multi_ingress_cast_join.onnx",
  "model_sdk_version": "2.0.0",
  "sequence": 1,
  "input_nodes": [
    { "name": "image_l",  "type": "buffer", "size": 64 },
    { "name": "image_uv", "type": "buffer", "size": 32 }
  ],
  "plugins": [
    {
      "name": "cast_0",
      "sequence": 1,
      "processor": "EV74",
      "config_params": {
        "kernel": "cast_transform",
        "input_shapes": [[1, 4, 4, 1]],
        "input_data_type": ["FP32"],
        "output_shapes": [[1, 4, 4, 1]],
        "data_type": ["BF16"]
      },
      "input_nodes": [
        { "name": "image_l", "size": 64, "logical_shape": [1, 4, 4, 1], "logical_dtype": "FP32" }
      ],
      "output_nodes": [
        { "name": "cast_0", "type": "buffer", "size": 32, "logical_shape": [1, 4, 4, 1], "logical_dtype": "BF16" }
      ],
      "type": "sgpProcess"
    },
    {
      "name": "cast_1",
      "sequence": 2,
      "processor": "EV74",
      "config_params": {
        "kernel": "cast_transform",
        "input_shapes": [[1, 2, 4, 1]],
        "input_data_type": ["FP32"],
        "output_shapes": [[1, 2, 4, 1]],
        "data_type": ["BF16"]
      },
      "input_nodes": [
        { "name": "image_uv", "size": 32, "logical_shape": [1, 2, 4, 1], "logical_dtype": "FP32" }
      ],
      "output_nodes": [
        { "name": "cast_1", "type": "buffer", "size": 16, "logical_shape": [1, 2, 4, 1], "logical_dtype": "BF16" }
      ],
      "type": "sgpProcess"
    },
    {
      "name": "pack_0",
      "sequence": 3,
      "processor": "EV74",
      "config_params": {
        "kernel": "pack_transform",
        "input_shapes": [[1, 4, 4, 1], [1, 2, 4, 1]],
        "input_data_type": ["BF16", "BF16"],
        "output_shapes": [[1, 48]],
        "data_type": ["BF16"]
      },
      "input_nodes": [
        { "name": "cast_0", "size": 32 },
        { "name": "cast_1", "size": 16 }
      ],
      "output_nodes": [
        { "name": "pack_0", "type": "buffer", "size": 48, "logical_shape": [1, 48], "logical_dtype": "BF16" }
      ],
      "type": "sgpProcess"
    },
    {
      "name": "mla_0",
      "sequence": 4,
      "processor": "MLA",
      "config_params": {
        "desired_batch_size": 1,
        "actual_batch_size": 1,
        "number_of_quads_to_user": 1,
        "input_shapes": [[1, 48]],
        "input_data_type": ["BF16"],
        "output_shapes": [[1, 2, 2, 1]],
        "data_type": ["FP32"]
      },
      "input_nodes": [
        { "name": "pack_0", "size": 48, "logical_shape": [1, 48], "logical_dtype": "BF16" }
      ],
      "output_nodes": [
        { "name": "mla_0", "type": "buffer", "size": 16, "logical_shape": [1, 2, 2, 1], "logical_dtype": "FP32" }
      ],
      "type": "sgpProcess",
      "resources": { "executable": "stage0.elf" }
    }
  ]
})json"},
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
  "input_buffers": [{"name": "pack_0"}],
  "data_type": ["BF16"],
  "output_width": [2],
  "output_height": [2],
  "output_depth": [1]
})json"},
                                              });
}

std::filesystem::path core_root() {
  return sima_test::test_source_root();
}

std::filesystem::path repo_root() {
  return core_root().parent_path();
}

std::filesystem::path bf16_ev74_model_path() {
  return core_root() / "tmp" / "yolov8n_drive" / "yolov8n_A_BF16_W_INT8_mpk.tar.gz";
}

std::filesystem::path int8_ev74_model_path() {
  return core_root() / "tmp" / "yolov8n_drive" / "yolov8n_A_W_int8_mpk.tar.gz";
}

std::filesystem::path bf16_mlatess_model_path() {
  return core_root() / "tmp" / "yolov8n_drive" / "yolov8n_A_W_BF16_MLATess.tar.gz";
}

std::filesystem::path int8_mlatess_model_path() {
  return core_root() / "tmp" / "yolov8n_drive" / "yolov8n_A_W_INT8_MLATess.tar.gz";
}

std::filesystem::path evo_bf16_mlatess_model_path() {
  return repo_root() / "tmp" / "evo_models" / "evo_testing" / "evo_testing" / "models" /
         "evo50_v2_mlatess_bf16_mpk.tar.gz";
}

std::filesystem::path evo_int8_ev74tess_model_path() {
  return repo_root() / "tmp" / "evo_models" / "evo_testing" / "evo_testing" / "models" /
         "evo50_v2_ev74tess_int8_mpk.tar.gz";
}

} // namespace

RUN_TEST(
    "unit_model_route_planner_test", ([] {
      using namespace simaai::neat;
      using namespace simaai::neat::internal;

      {
        const auto fixture = make_bf16_mla_tess_fixture("model_route_bf16_mla_tess");
        Model model(fixture.tar_path);

        const NodeGroup pre = model.preprocess();
        const NodeGroup post = model.postprocess();
        require(!pre.empty(), "strict MPK route should produce preprocess stage");
        require(!post.empty(), "strict MPK route should produce postprocess stage");

        const std::string pre_kind = pre.nodes().front() ? pre.nodes().front()->kind() : "";
        const std::string post_kind = post.nodes().front() ? post.nodes().front()->kind() : "";
        require(pre_kind == "QuantTess",
                "strict MPK route should select QuantTess preprocess stage");
        require(post_kind == "DetessDequant",
                "strict MPK route should select DetessDequant postprocess stage");

        const TensorSpec in_spec = model.input_spec();
        require(!in_spec.dtypes.empty() && in_spec.dtypes[0] == TensorDType::Float32,
                "strict MPK route input spec should expose FP32 host ingress dtype");
      }

      {
        const auto fixture = make_quanttess_post_fixture("model_route_quanttess_post");
        Model model(fixture.tar_path);

        const NodeGroup pre = model.preprocess();
        const NodeGroup post = model.postprocess();
        require(!pre.empty(), "strict MPK route should keep preprocess stage");
        require(!post.empty(), "strict MPK route should keep postprocess stage");

        const std::string pre_kind = pre.nodes().front() ? pre.nodes().front()->kind() : "";
        const std::string post_kind = post.nodes().front() ? post.nodes().front()->kind() : "";
        require(pre_kind == "QuantTess", "strict MPK route should select QuantTess pre stage");
        require(post_kind == "DetessDequant",
                "strict MPK route should keep DetessDequant post stage");
      }

      {
        const auto fixture = make_quant_no_post_fixture("model_route_quant_no_post");
        Model model(fixture.tar_path);

        const NodeGroup pre = model.preprocess();
        const NodeGroup post = model.postprocess();
        require(!pre.empty(), "strict MPK route should keep quant/tess preprocess");
        require(!post.empty(), "strict MPK route should keep strict postprocess stage");

        const std::string pre_kind = pre.nodes().front() ? pre.nodes().front()->kind() : "";
        require(pre_kind == "QuantTess", "strict MPK route should select QuantTess pre stage");

        const TensorSpec out_spec = model.output_spec();
        require(!out_spec.dtypes.empty() && out_spec.dtypes[0] == TensorDType::Float32,
                "strict MPK route output spec should expose Model boundary post dtype");
        std::ostringstream out_shape_msg;
        out_shape_msg << "strict MPK route output spec should expose Model boundary post shape"
                      << " actual=[";
        for (std::size_t i = 0; i < out_spec.shape.size(); ++i) {
          if (i) {
            out_shape_msg << ",";
          }
          out_shape_msg << out_spec.shape[i];
        }
        out_shape_msg << "]";
        require(out_spec.shape == std::vector<int64_t>({1, 80, 80, 64}), out_shape_msg.str());
      }

      {
        const auto fixture = make_ambiguous_tess_fixture("model_route_ambiguous_tess");
        Model model(fixture.tar_path);
        const NodeGroup pre = model.preprocess();
        const NodeGroup post = model.postprocess();
        require(!pre.empty(), "strict MPK route should keep preprocess stage");
        require(!post.empty(), "strict MPK route should keep postprocess stage");
      }

      {
        const auto legacy = sima_test::make_mpk_tar_fixture("model_route_legacy_missing_mpk",
                                                            {
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
          (void)legacy_model.preprocess();
        } catch (const std::exception& e) {
          threw = true;
          require_contains(std::string(e.what()), "strict MPK contract required",
                           "legacy missing-mpk fixture should fail with strict contract error");
        }
        require(threw, "legacy missing-mpk fixture must fail under strict contract");
      }

      {
        const auto fixture =
            make_multi_ingress_cast_join_fixture("model_route_multi_ingress_cast_join");
        Model::Options opt;
        opt.preprocess.kind = InputKind::Tensor;
        opt.preprocess.enable = AutoFlag::On;
        Model model(fixture.tar_path, opt);

        const auto& pack = ModelAccess::pack(model);
        const PreprocessCapabilities capabilities = inspect_preprocess_capabilities(pack);
        const PreprocessPlannerResult preprocess_plan = plan_preprocess(opt, capabilities);
        const RouteCapability capability = extract_route_capability(pack, preprocess_plan);
        const ModelSemantics semantics = build_model_semantics(pack);
        const SessionRoutePlan route_plan = build_route_plan(opt, semantics, &capability, &pack);

        require(capability.ingress_contracts.size() == 2U,
                "multi-ingress route should preserve ordered ingress contracts");
        require(capability.ingress_contracts[0].branch_ops.size() == 1U,
                "first ingress branch should preserve cast branch op");
        require(capability.ingress_contracts[1].branch_ops.size() == 1U,
                "second ingress branch should preserve cast branch op");
        require(capability.ingress_contracts[0].branch_ops.front().kind ==
                    OrderedRouteOp::Kind::Cast,
                "first ingress branch should classify cast op");
        require(capability.ingress_contracts[1].branch_ops.front().kind ==
                    OrderedRouteOp::Kind::Cast,
                "second ingress branch should classify cast op");
        require(capability.ingress_contracts[0].join_plugin_index.has_value() &&
                    capability.ingress_contracts[1].join_plugin_index.has_value() &&
                    capability.ingress_contracts[0].join_plugin_index ==
                        capability.ingress_contracts[1].join_plugin_index,
                "multi-ingress route should preserve common join plugin");

        require(route_plan.ingress_regions.size() == 2U,
                "multi-ingress route should expose fanout+fanin ingress regions");
        require(route_plan.ingress_contracts.size() == 2U,
                "multi-ingress route plan should preserve both ingress contracts");
        require(route_plan.ingress_contracts[0].branch_ops.size() == 1U,
                "first route-plan ingress branch should remain a single cast op");
        require(route_plan.ingress_contracts[1].branch_ops.size() == 1U,
                "second route-plan ingress branch should remain a single cast op");
        require(route_plan.ingress_contracts[0].branch_ops.front().kind ==
                    OrderedRouteOp::Kind::Cast,
                "first route-plan ingress branch should preserve cast op");
        require(route_plan.ingress_contracts[1].branch_ops.front().kind ==
                    OrderedRouteOp::Kind::Cast,
                "second route-plan ingress branch should preserve cast op");
        require(route_plan.ingress_regions[0].kind == RouteRegionKind::FanoutMap,
                "first ingress region should be branch fanout");
        require(route_plan.ingress_regions[0].op_kind ==
                    pipeline_internal::sima::RouteGraphKernelKind::Cast,
                "first ingress region should preserve cast op family");
        require(route_plan.ingress_regions[1].kind == RouteRegionKind::FaninJoin,
                "second ingress region should preserve explicit fanin join");
        require(route_plan.ingress_regions[1].op_kind ==
                    pipeline_internal::sima::RouteGraphKernelKind::PassThrough,
                "fanin join region should preserve pass-through join kind");
      }

      const auto check_real_model_route =
          [](const std::filesystem::path& model_path,
             const std::vector<
                 std::pair<RouteRegionKind, pipeline_internal::sima::RouteGraphKernelKind>>&
                 expected_regions,
             const std::vector<std::size_t>& expected_member_counts,
             const std::vector<SessionPostStageOp>& expected_post_chain) {
            if (!std::filesystem::exists(model_path)) {
              return;
            }

            Model::Options opt;
            opt.preprocess.kind = InputKind::Tensor;
            opt.preprocess.enable = AutoFlag::On;
            Model model(model_path.string(), opt);

            const auto& pack = ModelAccess::pack(model);
            const PreprocessCapabilities capabilities = inspect_preprocess_capabilities(pack);
            const PreprocessPlannerResult preprocess_plan = plan_preprocess(opt, capabilities);
            const RouteCapability capability = extract_route_capability(pack, preprocess_plan);
            const ModelSemantics semantics = build_model_semantics(pack);
            const SessionRoutePlan route_plan =
                build_route_plan(opt, semantics, &capability, &pack);

            require(route_plan.post_regions.size() == expected_regions.size(),
                    "unexpected post region count for " + model_path.filename().string());
            require(route_plan.post_chain.size() == expected_post_chain.size(),
                    "unexpected post chain summary size for " + model_path.filename().string());
            require(route_plan.include_post_stage == !expected_post_chain.empty(),
                    "include_post_stage should follow final post regions for " +
                        model_path.filename().string());

            for (std::size_t i = 0; i < expected_regions.size(); ++i) {
              require(route_plan.post_regions[i].kind == expected_regions[i].first,
                      "unexpected post region kind at index " + std::to_string(i) + " for " +
                          model_path.filename().string());
              require(route_plan.post_regions[i].op_kind == expected_regions[i].second,
                      "unexpected post region kernel at index " + std::to_string(i) + " for " +
                          model_path.filename().string());
              require(route_plan.post_regions[i].member_plugin_indices.size() ==
                          expected_member_counts[i],
                      "unexpected post region member count at index " + std::to_string(i) +
                          " for " + model_path.filename().string());
              require(route_plan.post_chain[i] == expected_post_chain[i],
                      "unexpected post chain summary op at index " + std::to_string(i) + " for " +
                          model_path.filename().string());
            }

            const bool expects_post_cast =
                std::find(expected_post_chain.begin(), expected_post_chain.end(),
                          SessionPostStageOp::Cast) != expected_post_chain.end();
            require(route_plan.post_cast_bf16_to_fp32 == expects_post_cast,
                    "post_cast_bf16_to_fp32 should be derived from final post regions for " +
                        model_path.filename().string());
          };

      check_real_model_route(bf16_ev74_model_path(),
                             {
                                 {RouteRegionKind::FanoutMap,
                                  pipeline_internal::sima::RouteGraphKernelKind::DetessCast},
                             },
                             {6U}, {SessionPostStageOp::DetessCast});

      check_real_model_route(int8_ev74_model_path(),
                             {
                                 {RouteRegionKind::FanoutMap,
                                  pipeline_internal::sima::RouteGraphKernelKind::DetessDequant},
                             },
                             {6U}, {SessionPostStageOp::DetessDequant});

      check_real_model_route(
          bf16_mlatess_model_path(),
          {
              {RouteRegionKind::FanoutMap, pipeline_internal::sima::RouteGraphKernelKind::Cast},
          },
          {6U}, {SessionPostStageOp::Cast});

      check_real_model_route(int8_mlatess_model_path(),
                             {
                                 {RouteRegionKind::FanoutMap,
                                  pipeline_internal::sima::RouteGraphKernelKind::Dequantize},
                             },
                             {6U}, {SessionPostStageOp::Dequantize});

      if (std::filesystem::exists(evo_bf16_mlatess_model_path())) {
        Model::Options opt;
        opt.preprocess.kind = InputKind::Tensor;
        opt.preprocess.enable = AutoFlag::On;
        Model model(evo_bf16_mlatess_model_path().string(), opt);

        const auto& pack = ModelAccess::pack(model);
        const PreprocessCapabilities capabilities = inspect_preprocess_capabilities(pack);
        const PreprocessPlannerResult preprocess_plan = plan_preprocess(opt, capabilities);
        const RouteCapability capability = extract_route_capability(pack, preprocess_plan);
        const ModelSemantics semantics = build_model_semantics(pack);
        const SessionRoutePlan route_plan = build_route_plan(opt, semantics, &capability, &pack);

        const auto graph_chain_it =
            std::find_if(route_plan.diagnostics.begin(), route_plan.diagnostics.end(),
                         [](const std::string& diag) {
                           return diag.find("graph_post_chain_raw=") != std::string::npos;
                         });
        require(graph_chain_it != route_plan.diagnostics.end(),
                "EVO BF16 MLATess route should emit raw graph post-chain diagnostics");
        require_contains(*graph_chain_it, "graph_post_chain_raw=slice,cast",
                         "EVO BF16 MLATess raw graph chain should preserve slice before cast");

        const auto graph_regions_it =
            std::find_if(route_plan.diagnostics.begin(), route_plan.diagnostics.end(),
                         [](const std::string& diag) {
                           return diag.find("graph_post_regions_raw=") != std::string::npos;
                         });
        require(graph_regions_it != route_plan.diagnostics.end(),
                "EVO BF16 MLATess route should emit raw graph post-region diagnostics");
        require_contains(*graph_regions_it, "slice",
                         "EVO BF16 MLATess raw post regions should preserve slice");

        require(route_plan.post_chain.size() == 1U &&
                    route_plan.post_chain.front() == SessionPostStageOp::Cast,
                "EVO BF16 MLATess final materialized post chain should remain cast-only");
        require(route_plan.post_regions.size() == 1U,
                "EVO BF16 MLATess final materialized post regions should exclude slice");
        require(route_plan.post_regions.front().op_kind ==
                    pipeline_internal::sima::RouteGraphKernelKind::Cast,
                "EVO BF16 MLATess final materialized post region should be cast");
        require(std::none_of(route_plan.post_regions.begin(), route_plan.post_regions.end(),
                             [](const RouteRegion& region) {
                               return region.op_kind ==
                                      pipeline_internal::sima::RouteGraphKernelKind::Slice;
                             }),
                "EVO BF16 MLATess final materialized post regions must not include slice");
      }

      if (std::filesystem::exists(evo_int8_ev74tess_model_path())) {
        Model::Options opt;
        opt.preprocess.kind = InputKind::Tensor;
        opt.preprocess.enable = AutoFlag::On;
        Model model(evo_int8_ev74tess_model_path().string(), opt);

        const auto& pack = ModelAccess::pack(model);
        const PreprocessCapabilities capabilities = inspect_preprocess_capabilities(pack);
        const PreprocessPlannerResult preprocess_plan = plan_preprocess(opt, capabilities);
        const RouteCapability capability = extract_route_capability(pack, preprocess_plan);
        const ModelSemantics semantics = build_model_semantics(pack);
        const SessionRoutePlan route_plan = build_route_plan(opt, semantics, &capability, &pack);

        require(route_plan.pre_chain.size() == 1U &&
                    route_plan.pre_chain.front() == SessionPreStageOp::QuantTess,
                "EVO INT8 EV74Tess final pre chain should fuse to QuantTess");
        require(route_plan.ingress_contracts.size() == 1U,
                "EVO INT8 EV74Tess should expose exactly one ingress contract");
        require(route_plan.ingress_contracts.front().branch_ops.size() == 1U,
                "EVO INT8 EV74Tess ingress branch should materialize one fused pre op");
        require(route_plan.ingress_contracts.front().branch_ops.front().kind ==
                    OrderedRouteOp::Kind::QuantTess,
                "EVO INT8 EV74Tess ingress branch should normalize to QuantTess");
        require(route_plan.ingress_regions.size() == 1U &&
                    route_plan.ingress_regions.front().op_kind ==
                        pipeline_internal::sima::RouteGraphKernelKind::QuantTess,
                "EVO INT8 EV74Tess ingress regions should follow the fused QuantTess pre chain");
      }
    }));

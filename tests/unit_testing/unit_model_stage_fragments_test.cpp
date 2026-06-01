#include "model/Model.h"
#include "model_archive_fixture_utils.h"
#include "pipeline/Graph.h"
#include "test_main.h"
#include "test_utils.h"

namespace {

sima_test::ModelArchiveFixture make_stage_fixture(const std::string& tag) {
  return sima_test::make_strict_model_archive_fixture(tag,
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
  "output_img_type": "RGB"
})json"},
                                                          {"etc/0_process_mla.json",
                                                           R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "preproc_0"}],
  "data_type": ["INT8"],
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

} // namespace

RUN_TEST("unit_model_stage_fragments_test", ([] {
           using namespace simaai::neat;

           const auto fixture = make_stage_fixture("model_stage_fragments");
           Model model(fixture.tar_path);

           Graph pre = model.preprocess();
           Graph infer = model.inference();
           Graph post = model.postprocess();
           Model::RouteOptions runnable_opt;
           runnable_opt.include_input = true;
           runnable_opt.include_output = true;
           Graph sess = model.graph(runnable_opt);
           Graph full_graph = model.graph();
           Graph direct_model_graph;
           direct_model_graph.add(model);

           require_contains(pre.describe_backend(false), "neatprocesscvu",
                            "Model::preprocess should produce a non-empty Graph fragment");
           require_contains(infer.describe_backend(false), "neatprocessmla",
                            "Model::inference should produce a non-empty Graph fragment");
           require_contains(post.describe_backend(false), "neatprocesscvu",
                            "Model::postprocess should expose Model boundary post stage");
           const std::string route_backend = sess.describe_backend(false);
           require_contains(route_backend, "appsrc",
                            "Model::graph({include_input=true}) should include appsrc");
           require_contains(route_backend, "appsink",
                            "Model::graph({include_output=true}) should include appsink");
           const std::string graph_backend = full_graph.describe_backend(false);
           require_contains(graph_backend, "neatprocessmla",
                            "Model::graph should expose the model route as a Graph fragment");
           require(graph_backend.find("appsrc") == std::string::npos,
                   "Model::graph should not bake in a default appsrc boundary");
           require(graph_backend.find("appsink") == std::string::npos,
                   "Model::graph should not bake in a default appsink boundary");
           const std::string direct_model_backend = direct_model_graph.describe_backend(false);
           require_contains(direct_model_backend, "neatprocessmla",
                            "Graph::add(Model) should splice the model route directly");
           require(direct_model_backend.find("appsrc") == std::string::npos,
                   "Graph::add(Model) should not bake in a default appsrc boundary");
           require(direct_model_backend.find("appsink") == std::string::npos,
                   "Graph::add(Model) should not bake in a default appsink boundary");

           const std::string infer_fragment = model.backend_fragment(Model::Stage::Inference);
           require_contains(infer_fragment, "neatprocessmla",
                            "Model::backend_fragment(inference) should include MLA plugin");
           require_contains(infer_fragment, "stage-id=",
                            "Model::backend_fragment(inference) should include stage metadata");

           const std::string full_fragment = model.backend_fragment(Model::Stage::Full);
           require_contains(full_fragment, "neatprocesscvu",
                            "Model::backend_fragment(full) should include CVU preproc plugin");
           require_contains(full_fragment, "neatprocessmla",
                            "Model::backend_fragment(full) should include MLA plugin");
           require_contains(full_fragment, "stage-id=",
                            "Model::backend_fragment(full) should include stage metadata");

           Graph infer_only = model.fragment(Model::Stage::Inference);
           require_contains(infer_only.describe_backend(false), "neatprocessmla",
                            "Model::fragment(inference) should produce non-empty Graph");

           const auto legacy = sima_test::make_model_archive_fixture(
               "model_stage_fragments_legacy_missing_mpk", {
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
             (void)legacy_model.graph();
           } catch (const std::exception& e) {
             threw = true;
             require_contains(std::string(e.what()), "strict MPK contract required",
                              "legacy missing-mpk fixture should fail with strict contract error");
           }
           require(threw, "legacy missing-mpk fixture must fail under strict contract");
         }));

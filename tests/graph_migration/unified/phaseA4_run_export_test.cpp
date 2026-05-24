#include "graph_migration/common/phase3_graph_test_utils.h"
#include "nodes/io/Input.h"
#include "nodes/common/Output.h"
#include "pipeline/Graph.h"
#include "pipeline/RunExport.h"
#include "test_main.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::filesystem::path tmp_path(const std::string& name) {
  const char* tmp = std::getenv("TMPDIR");
  return std::filesystem::path(tmp ? tmp : "/tmp") / name;
}

bool array_has_name(const nlohmann::json& arr, const std::string& name) {
  for (const auto& item : arr) {
    if (item.contains("name") && item.at("name").get<std::string>() == name) {
      return true;
    }
  }
  return false;
}

bool public_view_has_edge(const nlohmann::json& arr, const std::string& from,
                          const std::string& to) {
  for (const auto& item : arr) {
    if (item.value("from_endpoint", "") == from && item.value("to_endpoint", "") == to) {
      return true;
    }
  }
  return false;
}

bool public_view_has_node_block(const nlohmann::json& arr, const std::string& endpoint,
                                const std::string& block, const std::string& kind) {
  for (const auto& item : arr) {
    if (item.value("endpoint_name", "") == endpoint && item.contains(block) &&
        item.at(block).value("kind", "") == kind) {
      return true;
    }
  }
  return false;
}

} // namespace

RUN_TEST("graph_migration_phaseA4_run_export_test", [] {
  using namespace simaai::neat;

  Graph input("image");
  input.add(nodes::Input("image"));
  Graph output("classes");
  output.add(nodes::Output("classes"));

  Graph app("exportable");
  app.connect(input, output);

  Run run = app.build();
  require(run.push("image", TensorList{graph_phase3_test::make_rgb_tensor(16, 16, 0x42)}),
          "named push failed: " + run.last_error());
  TensorList tensors = run.pull_tensors("classes", 5000);
  graph_phase3_test::require_nonempty_tensor_output(tensors, "graph run export preflight pull");

  RunExportOptions opt;
  opt.label = "unit_export";
  opt.metadata.push_back(std::make_pair(std::string("test_name"),
                                        std::string("graph_migration_phaseA4_run_export_test")));

  std::string err;
  const std::string body = run_to_json(run, opt, &err);
  require(err.empty(), "run_to_json error: " + err);
  require(!body.empty(), "run_to_json returned empty body");

  const nlohmann::json json = nlohmann::json::parse(body);
  require(json.at("schema").get<std::string>() == "sima.neat.graph_run",
          "wrong graph run export schema");
  require(json.at("schema_version").get<int>() == 1, "wrong graph run schema version");
  require(json.at("label").get<std::string>() == "unit_export", "wrong export label");
  require(json.at("graph").at("mode").get<std::string>() == "connected",
          "connected graph should export connected mode");
  require(array_has_name(json.at("graph").at("named_inputs"), "image"),
          "export should list named input image");
  require(array_has_name(json.at("graph").at("named_outputs"), "classes"),
          "export should list named output classes");
  require(json.at("graph").at("edges").size() >= 1U, "export should include runtime edges");
  require(json.at("graph").at("public_view").at("nodes").size() >= 2U,
          "export should include public graph nodes");
  require(public_view_has_node_block(json.at("graph").at("public_view").at("nodes"), "image",
                                     "source", "app_push"),
          "export should annotate public Input as an app_push source");
  require(public_view_has_node_block(json.at("graph").at("public_view").at("nodes"), "classes",
                                     "sink", "appsink"),
          "export should annotate public Output as an appsink sink");
  require(public_view_has_edge(json.at("graph").at("public_view").at("edges"), "image", "classes"),
          "export should include public endpoint edge image -> classes");
  require(json.at("graph").at("lowered_view").at("nodes").size() >= 1U,
          "export should include lowered runtime nodes");
  require(json.at("run").at("stats").at("outputs_pulled").get<std::uint64_t>() >= 1U,
          "export should include run output counters");
  require(!json.at("run").at("identity").at("uuid").get<std::string>().empty(),
          "export should include run identity uuid");
  require(!json.at("run").at("identity").at("hostname").get<std::string>().empty(),
          "export should include run identity hostname");

  const std::filesystem::path path = tmp_path("run_export.neat.graph_run.json");
  std::filesystem::remove(path);
  require(save_run_json(run, path.string(), opt, &err), "save_run_json failed: " + err);
  require(std::filesystem::exists(path), "save_run_json did not create output file");
  std::ifstream in(path);
  const nlohmann::json saved = nlohmann::json::parse(in);
  require(saved.at("schema").get<std::string>() == "sima.neat.graph_run",
          "saved export schema mismatch");
  run.close();

  Graph auto_input("auto_image");
  auto_input.add(nodes::Input("auto_image"));
  Graph auto_output("auto_classes");
  auto_output.add(nodes::Output("auto_classes"));
  Graph auto_app("auto_exportable");
  auto_app.connect(auto_input, auto_output);

  const std::filesystem::path auto_path = tmp_path("graph_run_auto_export.neat.graph_run.json");
  std::filesystem::remove(auto_path);
  std::filesystem::remove(std::filesystem::path(auto_path.string() + ".tmp"));

  RunOptions build_opt;
  build_opt.run_export.path = auto_path.string();
  build_opt.run_export.label = "auto_export";
  Run auto_run = auto_app.build(build_opt);
  require(std::filesystem::exists(auto_path), "Graph::build auto export did not create JSON");
  std::ifstream auto_in(auto_path);
  const nlohmann::json auto_json = nlohmann::json::parse(auto_in);
  require(auto_json.at("schema").get<std::string>() == "sima.neat.graph_run",
          "auto export schema mismatch");
  require(auto_json.at("label").get<std::string>() == "auto_export", "auto export label mismatch");
  require(array_has_name(auto_json.at("graph").at("named_inputs"), "auto_image"),
          "auto export should list named input auto_image");
  require(array_has_name(auto_json.at("graph").at("named_outputs"), "auto_classes"),
          "auto export should list named output auto_classes");
  require(public_view_has_edge(auto_json.at("graph").at("public_view").at("edges"), "auto_image",
                               "auto_classes"),
          "auto export should include public endpoint edge auto_image -> auto_classes");
  auto_run.close();
});

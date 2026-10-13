#include "model/Model.h"
#include "pipeline/ExecutionPlan.h"

#include "asset_utils.h"
#include "cli_utils.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

json parse_plan(const simaai::neat::ExecutionPlan& plan, const std::string& where) {
  require(plan.schema_version() == 1, where + ": schema_version should be 1");
  json doc = json::parse(plan.to_json());
  require(doc.at("schema").get<std::string>() == "sima.neat.execution_plan",
          where + ": wrong schema name");
  require(doc.at("schema_version").get<int>() == 1, where + ": wrong schema_version");
  return doc;
}

// Every node says where it runs, accelerator nodes name their kernel, and every edge between
// two compiled nodes carries batch-first tensors.
void require_plan_shape(const json& doc, const std::string& where) {
  const json& model = doc.at("model");
  require(!model.at("name").get<std::string>().empty(), where + ": model name is empty");
  const int batch = model.at("batch").get<int>();
  require(batch >= 1, where + ": batch should be at least 1");

  const std::set<std::string> engines = {"A65", "EV74", "MLA"};
  bool has_mla = false;
  for (const auto& node : doc.at("nodes")) {
    const std::string kind = node.at("kind").get<std::string>();
    const std::string runs_on = node.at("runs_on").get<std::string>();
    require(engines.count(runs_on) == 1U, where + ": node " + kind + " runs on " + runs_on);
    has_mla = has_mla || runs_on == "MLA";
    const bool endpoint = kind == "Input" || kind == "Output";
    if (!endpoint && runs_on != "MLA") {
      require(node.contains("kernel"), where + ": node " + kind + " names no kernel");
    }
  }
  require(has_mla, where + ": no node runs on the MLA");

  const auto& nodes = doc.at("nodes");
  const auto& edges = doc.at("edges");
  require(edges.size() + 1U == nodes.size(), where + ": a linear plan has one edge per node pair");
  for (const auto& edge : edges) {
    if (!edge.contains("tensors")) {
      continue;
    }
    for (const auto& tensor : edge.at("tensors")) {
      const auto& shape = tensor.at("shape");
      require(!shape.empty() && shape.front().get<int>() == batch,
              where + ": tensor " + tensor.at("name").get<std::string>() +
                  " shape is not batch-first");
    }
  }
}

// A zero tensor in EV74 memory shaped like the Model input, as applications seed build.
simaai::neat::TensorList ev74_seed(const simaai::neat::Model& model) {
  const simaai::neat::InputOptions opt = model.input_appsrc_options(true);
  const int height = opt.height > 0 ? opt.height : opt.max_height;
  const int width = opt.width > 0 ? opt.width : opt.max_width;
  const int depth = opt.depth > 0 ? opt.depth : opt.max_depth;
  const std::vector<float> zeros(static_cast<std::size_t>(height) * width * depth, 0.0F);
  return {simaai::neat::Tensor::from_vector(zeros, {height, width, depth},
                                            simaai::neat::TensorMemory::EV74)};
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  std::string tar_gz;
  sima_test::get_arg(argc, argv, "--model", tar_gz);
  if (tar_gz.empty()) {
    tar_gz = sima_test::resolve_yolov8s_tar();
  }
  if (tar_gz.empty() || !fs::exists(tar_gz)) {
    std::cerr << "Failed to resolve the yolov8s model tar.gz. Set SIMA_YOLO_TAR or pass "
                 "--model <path>.\n";
    return 3;
  }

  try {
    simaai::neat::Model model(tar_gz);

    const json planned = parse_plan(model.execution_plan(), "Model::execution_plan");
    require_plan_shape(planned, "Model::execution_plan");
    require(parse_plan(model.execution_plan(), "second Model::execution_plan") == planned,
            "planning twice should give the same plan");

    const simaai::neat::TensorList seed = ev74_seed(model);
    require(parse_plan(model.execution_plan(seed), "seeded Model::execution_plan") == planned,
            "a seed shaped like the Model input should not change the plan");

    require(simaai::neat::ExecutionPlan{}.to_json(-1) == "{}",
            "a default ExecutionPlan should be empty");
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] model_execution_plan_test: " << e.what() << "\n";
    return 1;
  }
  std::cout << "[OK] model_execution_plan_test passed\n";
  return 0;
}

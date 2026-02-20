#include "builder/ConfigJsonProvider.h"
#include "builder/NodeGroup.h"
#include "pipeline/internal/StageConfig.h"
#include "test_main.h"
#include "test_utils.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

class JsonConfigNode final : public simaai::neat::Node, public simaai::neat::ConfigJsonProvider {
public:
  explicit JsonConfigNode(nlohmann::json cfg) : cfg_(std::move(cfg)) {}

  std::string kind() const override {
    return "JsonConfigNode";
  }
  std::string user_label() const override {
    return "json_cfg";
  }
  std::string backend_fragment(int) const override {
    return "identity";
  }
  std::vector<std::string> element_names(int) const override {
    return {};
  }
  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Static;
  }

  const nlohmann::json* config_json() const override {
    return &cfg_;
  }

private:
  nlohmann::json cfg_;
};

simaai::neat::NodeGroup one_node_group(nlohmann::json cfg) {
  return simaai::neat::NodeGroup({std::make_shared<JsonConfigNode>(std::move(cfg))});
}

} // namespace

RUN_TEST("unit_stageconfig_mla_info_test", ([] {
           using namespace simaai::neat;

           {
             nlohmann::json cfg = {
                 {"data_type", nlohmann::json::array({"INT8"})},
                 {"output_width", nlohmann::json::array({80})},
                 {"output_height", nlohmann::json::array({40})},
                 {"output_depth", nlohmann::json::array({6})},
                 {"output_format", "NCHW"},
                 {"input_width", 224},
                 {"input_height", 112},
                 {"input_depth", 3},
                 {"input_format", "HWC"},
             };

             const auto out = stages::read_mla_output_info(one_node_group(cfg));
             require(out.data_type == "INT8", "read_mla_output_info: top-level dtype mismatch");
             require(out.dims.width == 80 && out.dims.height == 40 && out.dims.depth == 6,
                     "read_mla_output_info: top-level dims mismatch");
             require(out.layout == TensorLayout::CHW,
                     "read_mla_output_info: output_format NCHW should map to CHW layout");

             const auto in = stages::read_mla_input_info(one_node_group(cfg));
             require(in.dims.width == 224 && in.dims.height == 112 && in.dims.depth == 3,
                     "read_mla_input_info: top-level dims mismatch");
             require(in.layout == TensorLayout::HWC,
                     "read_mla_input_info: input_format HWC should map to HWC layout");
           }

           {
             nlohmann::json cfg = {
                 {"data_type", nlohmann::json::array({"INT8"})},
                 {"simaai__params",
                  {
                      {"data_type", nlohmann::json::array({"INT16"})},
                      {"output_width", nlohmann::json::array({64})},
                      {"output_height", nlohmann::json::array({48})},
                      {"output_depth", nlohmann::json::array({5})},
                      {"output_format", "HWC"},
                      {"input_width", nlohmann::json::array({320})},
                      {"input_height", nlohmann::json::array({240})},
                      {"input_channels", nlohmann::json::array({4})},
                      {"input_format", "CHW"},
                      {"outputs", nlohmann::json::array({{{"size", 1024}}})},
                  }},
             };

             const auto out = stages::read_mla_output_info(one_node_group(cfg));
             require(out.data_type == "INT16",
                     "read_mla_output_info: params dtype should override top-level value");
             require(out.dims.width == 64 && out.dims.height == 48 && out.dims.depth == 5,
                     "read_mla_output_info: params dims mismatch");
             require(out.layout == TensorLayout::HWC,
                     "read_mla_output_info: params output_format HWC should map to HWC layout");
             require(out.size_bytes == 1024, "read_mla_output_info: params output size mismatch");

             const auto in = stages::read_mla_input_info(one_node_group(cfg));
             require(in.dims.width == 320 && in.dims.height == 240 && in.dims.depth == 4,
                     "read_mla_input_info: params dims mismatch");
             require(in.layout == TensorLayout::CHW,
                     "read_mla_input_info: params input_format CHW should map to CHW layout");
           }
         }));

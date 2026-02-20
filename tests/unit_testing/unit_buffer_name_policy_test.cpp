#include "builder/ConfigJsonOverride.h"
#include "builder/ConfigJsonOverrideMulti.h"
#include "builder/ConfigJsonProvider.h"
#include "builder/ConfigJsonWire.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Session.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

std::string extract_element_name(const std::string& pipeline, const std::string& factory) {
  const std::string needle = factory + " name=";
  size_t pos = pipeline.find(needle);
  if (pos == std::string::npos)
    return "";
  pos += needle.size();
  size_t end = pos;
  while (end < pipeline.size()) {
    const char c = pipeline[end];
    if (std::isspace(static_cast<unsigned char>(c)) || c == '!' || c == ';')
      break;
    ++end;
  }
  return pipeline.substr(pos, end - pos);
}

class FakeConfigNode final : public simaai::neat::Node,
                             public simaai::neat::ConfigJsonProvider,
                             public simaai::neat::ConfigJsonOverride {
public:
  explicit FakeConfigNode(std::string node_name)
      : config_({{"node_name", std::move(node_name)},
                 {"input_buffers", json::array({{{"name", "decoder"}}})}}) {}

  std::string kind() const override {
    return "FakeConfig";
  }
  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Static;
  }
  bool has_config_json() const override {
    return true;
  }

  std::string backend_fragment(int node_index) const override {
    return "identity name=n" + std::to_string(node_index) + "_fakecfg";
  }

  std::vector<std::string> element_names(int node_index) const override {
    return {"n" + std::to_string(node_index) + "_fakecfg"};
  }

  bool wire_input_names(const std::vector<std::string>& upstream_names,
                        const std::string& tag) override {
    if (upstream_names.empty() || upstream_names[0].empty())
      return false;
    return override_config_json(
        [&](json& j) { (void)simaai::neat::set_input_buffer_name_if_exists(j, upstream_names[0]); },
        tag);
  }

  const json* config_json() const override {
    return &config_;
  }

  bool override_config_json(const std::function<void(json&)>& edit, const std::string&) override {
    json cfg = config_;
    edit(cfg);
    if (cfg == config_)
      return false;
    config_ = std::move(cfg);
    return true;
  }

  const json& config() const {
    return config_;
  }

private:
  json config_;
};

class FakeMultiNode final : public simaai::neat::Node,
                            public simaai::neat::ConfigJsonProvider,
                            public simaai::neat::ConfigJsonOverrideMulti {
public:
  FakeMultiNode() {
    configs_.push_back(
        {{"node_name", "node_a"}, {"input_buffers", json::array({{{"name", "decoder"}}})}});
    configs_.push_back(
        {{"node_name", "node_b"}, {"input_buffers", json::array({{{"name", "node_a"}}})}});
  }

  std::string kind() const override {
    return "FakeMulti";
  }
  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Static;
  }
  bool has_config_json() const override {
    return true;
  }

  std::string backend_fragment(int node_index) const override {
    return "identity name=n" + std::to_string(node_index) +
           "_a ! "
           "identity name=n" +
           std::to_string(node_index) + "_b";
  }

  std::vector<std::string> element_names(int node_index) const override {
    return {"n" + std::to_string(node_index) + "_a", "n" + std::to_string(node_index) + "_b"};
  }

  bool wire_input_names(const std::vector<std::string>& upstream_names,
                        const std::string& tag) override {
    if (upstream_names.empty())
      return false;
    return override_config_json_multi(
        [&](json& j, std::size_t idx, std::size_t) {
          const std::string& name =
              (idx < upstream_names.size()) ? upstream_names[idx] : upstream_names.back();
          (void)simaai::neat::set_input_buffer_name_if_exists(j, name);
        },
        tag);
  }

  const json* config_json() const override {
    if (configs_.empty())
      return nullptr;
    return &configs_.front();
  }

  bool override_config_json_multi(const std::function<void(json&, std::size_t, std::size_t)>& edit,
                                  const std::string&) override {
    if (configs_.empty())
      return false;
    bool changed = false;
    const std::size_t count = configs_.size();
    for (std::size_t i = 0; i < configs_.size(); ++i) {
      json cfg = configs_[i];
      edit(cfg, i, count);
      if (cfg == configs_[i])
        continue;
      configs_[i] = std::move(cfg);
      changed = true;
    }
    return changed;
  }

  const std::vector<json>& configs() const {
    return configs_;
  }

private:
  std::vector<json> configs_;
};

} // namespace

int main() {
  try {
    auto multi = std::make_shared<FakeMultiNode>();
    auto single = std::make_shared<FakeConfigNode>("custom");

    simaai::neat::Session p;
    p.add(simaai::neat::nodes::Input());
    p.add(multi);
    p.add(single);
    p.add(simaai::neat::nodes::Output());

    const std::string pipeline = p.describe_backend(false);
    require(!pipeline.empty(), "Pipeline string empty");

    const std::string appsrc_name = extract_element_name(pipeline, "appsrc");
    require(!appsrc_name.empty(), "Appsrc name missing");
    require_contains(appsrc_name, "mysrc", "Appsrc name missing base");

    std::string suffix;
    if (appsrc_name.rfind("mysrc", 0) == 0) {
      suffix = appsrc_name.substr(std::string("mysrc").size());
    }
    require(!suffix.empty(), "Expected auto suffix on element names");

    const json& single_cfg = single->config();
    require(single_cfg.contains("node_name"), "single node_name missing");
    const std::string single_name = single_cfg["node_name"].get<std::string>();
    require(single_name == "custom", "single node_name should remain unchanged");
    require(single_cfg.contains("input_buffers"), "single input_buffers missing");
    const std::string single_in = single_cfg["input_buffers"][0]["name"].get<std::string>();
    require(single_in == "decoder",
            "single input_buffers name should remain unchanged");

    const auto& cfgs = multi->configs();
    require(cfgs.size() == 2, "multi configs size mismatch");
    const std::string node_a = cfgs[0]["node_name"].get<std::string>();
    const std::string node_b = cfgs[1]["node_name"].get<std::string>();
    require(node_a == "node_a", "multi node_a should remain unchanged");
    require(node_b == "node_b", "multi node_b should remain unchanged");
    const std::string multi_in0 = cfgs[0]["input_buffers"][0]["name"].get<std::string>();
    require(multi_in0 == "decoder",
            "multi input_buffers[0] name should remain unchanged");
    const std::string multi_in1 = cfgs[1]["input_buffers"][0]["name"].get<std::string>();
    require(multi_in1 == "node_a",
            "multi input_buffers[1] name should remain unchanged");

    std::cout << "[OK] unit_buffer_name_policy_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

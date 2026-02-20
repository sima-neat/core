/**
 * @file
 * @ingroup nodes_sima
 * @brief SimaAI detessellation + dequantization node wrapper.
 */
#pragma once

#include "builder/ConfigJsonConsumer.h"
#include "builder/ConfigJsonOverride.h"
#include "builder/ConfigJsonProvider.h"
#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {
class Model;
} // namespace simaai::neat

namespace simaai::neat {

struct DetessDequantOptions {
  DetessDequantOptions() = default;
  explicit DetessDequantOptions(const simaai::neat::Model& model);

  std::string config_path;
  std::string config_dir;
  bool keep_config = false;
  std::optional<nlohmann::json> config_json;
  std::string upstream_name;
  std::string element_name;
  int num_buffers = 0;
  int num_buffers_model = 0;
  bool num_buffers_locked = false;
};

class DetessDequant final : public Node,
                            public OutputSpecProvider,
                            public ConfigJsonProvider,
                            public ConfigJsonOverride,
                            public ConfigJsonConsumer {
public:
  explicit DetessDequant(DetessDequantOptions opt = {});
  struct ConfigHolder;

  std::string kind() const override {
    return "DetessDequant";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  bool has_config_json() const override {
    return true;
  }
  bool wire_input_names(const std::vector<std::string>& upstream_names,
                        const std::string& tag) override;
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  const nlohmann::json* config_json() const override;
  bool override_config_json(const std::function<void(nlohmann::json&)>& edit,
                            const std::string& tag) override;
  void apply_upstream_config(const nlohmann::json& upstream,
                             const std::string& upstream_kind) override;

  const DetessDequantOptions& options() const {
    return opt_;
  }
  const std::string& config_path() const {
    return config_path_;
  }

private:
  DetessDequantOptions opt_;
  std::shared_ptr<ConfigHolder> config_holder_;
  std::string config_path_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> DetessDequant(DetessDequantOptions opt = {});
} // namespace simaai::neat::nodes

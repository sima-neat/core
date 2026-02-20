/**
 * @file
 * @ingroup nodes_sima
 * @brief SimaAI box decode node wrapper.
 */
#pragma once

#include "builder/ConfigJsonConsumer.h"
#include "builder/ConfigJsonOverride.h"
#include "builder/ConfigJsonProvider.h"
#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {
class Model;
} // namespace simaai::neat

namespace simaai::neat {

struct BoxDecodeOptionsInternal;

class SimaBoxDecode final : public Node,
                            public OutputSpecProvider,
                            public ConfigJsonProvider,
                            public ConfigJsonOverride,
                            public ConfigJsonConsumer {
public:
  explicit SimaBoxDecode(const simaai::neat::Model& model, const std::string& decode_type = "",
                         int original_width = 0, int original_height = 0,
                         double detection_threshold = 0.0, double nms_iou_threshold = 0.0,
                         int top_k = 0);

  std::string kind() const override {
    return "SimaBoxDecode";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  bool has_config_json() const override;
  bool wire_input_names(const std::vector<std::string>& upstream_names,
                        const std::string& tag) override;
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  const std::string& config_path() const {
    return config_path_;
  }
  const nlohmann::json* config_json() const override;
  bool override_config_json(const std::function<void(nlohmann::json&)>& edit,
                            const std::string& tag) override;
  void apply_upstream_config(const nlohmann::json& upstream,
                             const std::string& upstream_kind) override;

private:
  struct BoxDecodeConfigHolder;

  std::unique_ptr<BoxDecodeOptionsInternal> opt_;
  std::shared_ptr<BoxDecodeConfigHolder> config_holder_;
  std::string config_path_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> SimaBoxDecode(const simaai::neat::Model& model,
                                                  const std::string& decode_type = "",
                                                  int original_width = 0, int original_height = 0,
                                                  double detection_threshold = 0.0,
                                                  double nms_iou_threshold = 0.0, int top_k = 0);
} // namespace simaai::neat::nodes

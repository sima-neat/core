/**
 * @file
 * @ingroup nodes_sima
 * @brief SimaAI ArgMax node wrapper.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

struct SimaArgMaxOptions {
  std::string config_path;
  int sima_allocator_type = 2;
  bool silent = true;
  bool emit_signals = false;
  bool transmit = false;
};

class SimaArgMax final : public Node, public OutputSpecProvider {
public:
  explicit SimaArgMax(SimaArgMaxOptions opt = {});

  std::string kind() const override {
    return "SimaArgMax";
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

  const SimaArgMaxOptions& options() const {
    return opt_;
  }

private:
  struct ConfigHolder;

  SimaArgMaxOptions opt_;
  std::shared_ptr<ConfigHolder> config_holder_;
  std::string config_path_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> SimaArgMax(SimaArgMaxOptions opt = {});
} // namespace simaai::neat::nodes

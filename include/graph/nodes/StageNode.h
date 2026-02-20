/**
 * @file
 * @ingroup graph_nodes
 * @brief Stage-backed graph node wrapper.
 */
#pragma once

#include "graph/GraphTypes.h"
#include "graph/Node.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::graph {
class StageExecutor;
} // namespace simaai::neat::graph

namespace simaai::neat::graph::nodes {

enum class StageKeyBy {
  None = 0,
  StreamId,
};

struct StageNodeOptions {
  int instances = 1; // number of executor instances
  StageKeyBy key_by = StageKeyBy::None;
  std::size_t max_inflight = 0; // 0 => use GraphRunOptions.edge_queue
};

class StageNode final : public simaai::neat::graph::Node {
public:
  using StageExecutorFactory = std::function<std::unique_ptr<simaai::neat::graph::StageExecutor>()>;
  using OutputSpecFn = std::function<OutputSpec(const std::vector<OutputSpec>&, PortId)>;

  StageNode(std::string kind, StageExecutorFactory factory, std::vector<PortDesc> inputs,
            std::vector<PortDesc> outputs, std::string label = {}, OutputSpecFn out_fn = {},
            StageNodeOptions options = {})
      : kind_(std::move(kind)), label_(std::move(label)), inputs_(std::move(inputs)),
        outputs_(std::move(outputs)), factory_(std::move(factory)),
        output_spec_fn_(std::move(out_fn)), options_(std::move(options)) {}

  Backend backend() const override {
    return Backend::Stage;
  }

  std::string kind() const override {
    return kind_;
  }

  std::string user_label() const override {
    return label_;
  }

  std::vector<PortDesc> input_ports() const override {
    return inputs_;
  }

  std::vector<PortDesc> output_ports() const override {
    return outputs_;
  }

  OutputSpec output_spec(const std::vector<OutputSpec>& inputs, PortId out_port) const override {
    if (output_spec_fn_)
      return output_spec_fn_(inputs, out_port);
    if (outputs_.size() == 1)
      return outputs_.front().spec;
    return OutputSpec{};
  }

  const StageExecutorFactory& factory() const {
    return factory_;
  }
  const StageNodeOptions& options() const {
    return options_;
  }

private:
  std::string kind_;
  std::string label_;
  std::vector<PortDesc> inputs_;
  std::vector<PortDesc> outputs_;
  StageExecutorFactory factory_;
  OutputSpecFn output_spec_fn_;
  StageNodeOptions options_;
};

} // namespace simaai::neat::graph::nodes

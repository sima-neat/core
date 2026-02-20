/**
 * @file
 * @ingroup graph_nodes
 * @brief Stage executor for Model inference using stage APIs.
 */
#pragma once

#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"
#include "pipeline/StageRun.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
class Model;
} // namespace simaai::neat

namespace simaai::neat::graph::nodes {

struct StageModelExecutorOptions {
  std::shared_ptr<const simaai::neat::Model> model;
  bool do_preproc = true;
  bool do_mla = true;
  bool do_boxdecode = false;
  simaai::neat::stages::BoxDecodeOptions box_opt;
};

class StageModelExecutor final : public simaai::neat::graph::StageExecutor {
public:
  explicit StageModelExecutor(StageModelExecutorOptions opt);

  void set_ports(const StagePorts& ports) override;
  void on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) override;

private:
  StageModelExecutorOptions opt_;
  PortId out_port_ = kInvalidPort;
};

// Convenience: wrap StageModelExecutor in a StageNode.
std::shared_ptr<simaai::neat::graph::Node>
StageModelExecutorNode(const StageModelExecutorOptions& opt, std::string label = {},
                       StageNodeOptions node_opt = {});

} // namespace simaai::neat::graph::nodes

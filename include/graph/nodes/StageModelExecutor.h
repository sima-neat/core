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

/**
 * @brief Configuration for a `StageModelExecutor` stage.
 *
 * Selects the model to run and which inference sub-stages (preprocess, MLA, optional
 * box-decode) the executor should perform.
 *
 * @ingroup graph
 */
struct StageModelExecutorOptions {
  std::shared_ptr<const simaai::neat::Model> model; ///< Model to run inference against.
  bool do_preproc = true;                           ///< Run the preprocess stage.
  bool do_mla = true;                               ///< Run the MLA inference stage.
  bool do_boxdecode = false;                        ///< Run the box-decode post stage.
  simaai::neat::stages::BoxDecodeOptions box_opt{
      simaai::neat::BoxDecodeType::Unspecified}; ///< Box-decode options when `do_boxdecode` is
                                                 ///< true.
};

/**
 * @brief Stage executor that runs a `Model`'s inference pipeline as a runtime-graph stage.
 *
 * Wraps the staged inference APIs (preprocess / MLA / box-decode) so a `Model` can be
 * embedded directly in a runtime graph alongside other stages.
 *
 * @see StageModelExecutorNode
 * @see StageExecutor
 * @ingroup graph
 */
class StageModelExecutor final : public simaai::neat::graph::StageExecutor {
public:
  /// Construct a StageModelExecutor from the given options.
  explicit StageModelExecutor(StageModelExecutorOptions opt);

  /// Bind the executor to its runtime ports.
  void set_ports(const StagePorts& ports) override;
  /// Run the configured inference sub-stages on the incoming sample.
  void on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) override;

private:
  StageModelExecutorOptions opt_;
  PortId out_port_ = kInvalidPort;
};

/**
 * @brief Convenience helper that wraps a `StageModelExecutor` in a `StageNode`.
 *
 * @param opt      Inference configuration (model + which stages to run).
 * @param label    Optional human-readable label.
 * @param node_opt Optional `StageNode` options.
 * @return Shared pointer to a `graph::Node` ready for insertion in a runtime graph.
 */
std::shared_ptr<simaai::neat::graph::Node>
StageModelExecutorNode(const StageModelExecutorOptions& opt, std::string label = {},
                       StageNodeOptions node_opt = {});

} // namespace simaai::neat::graph::nodes

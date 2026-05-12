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

/**
 * @brief Selects how the runtime keys messages for a stage node's instance assignment.
 *
 * Used by `StageNodeOptions::key_by` to pick a deterministic assignment policy when a node
 * has multiple executor instances.
 *
 * @ingroup graph
 */
enum class StageKeyBy {
  None = 0, ///< No keying — round-robin or unconstrained dispatch.
  StreamId, ///< Key on `Sample::stream_id` so all messages for a stream go to the same instance.
};

/**
 * @brief Options controlling how a `StageNode` is instantiated and scheduled by the runtime.
 *
 * @ingroup graph
 */
struct StageNodeOptions {
  int instances = 1;                    ///< Number of executor instances to spawn.
  StageKeyBy key_by = StageKeyBy::None; ///< Keying policy for multi-instance dispatch.
  std::size_t max_inflight =
      0; ///< Max in-flight messages per node; 0 defers to `GraphRunOptions::edge_queue`.
};

/**
 * @brief Generic runtime-graph node that adapts a `StageExecutor` factory into a `Node`.
 *
 * The most generic stage adapter — given a `StageExecutorFactory`, an input/output port
 * description, and an `Options` bundle, exposes the stage as a runtime graph `Node`. Most
 * concrete stages (`FanOut`, `JoinBundle`, `Map`, ...) have helpers that build a `StageNode`
 * around their executor.
 *
 * @see StageExecutor
 * @see StageNodeOptions
 * @ingroup graph
 */
class StageNode final : public simaai::neat::graph::Node {
public:
  /// Factory producing a fresh `StageExecutor` for each instance the runtime needs.
  using StageExecutorFactory = std::function<std::unique_ptr<simaai::neat::graph::StageExecutor>()>;
  /// Resolver that returns the `OutputSpec` for a given output port from the per-input specs.
  using OutputSpecFn = std::function<OutputSpec(const std::vector<OutputSpec>&, PortId)>;

  /// Construct a StageNode bundling kind, executor factory, port descriptors, and options.
  StageNode(std::string kind, StageExecutorFactory factory, std::vector<PortDesc> inputs,
            std::vector<PortDesc> outputs, std::string label = {}, OutputSpecFn out_fn = {},
            StageNodeOptions options = {})
      : kind_(std::move(kind)), label_(std::move(label)), inputs_(std::move(inputs)),
        outputs_(std::move(outputs)), factory_(std::move(factory)),
        output_spec_fn_(std::move(out_fn)), options_(std::move(options)) {}

  /// Always returns `Backend::Stage`.
  Backend backend() const override {
    return Backend::Stage;
  }

  /// Returns the stage's kind label.
  std::string kind() const override {
    return kind_;
  }

  /// Returns the user-supplied label (or empty).
  std::string user_label() const override {
    return label_;
  }

  /// Returns the configured input ports.
  std::vector<PortDesc> input_ports() const override {
    return inputs_;
  }

  /// Returns the configured output ports.
  std::vector<PortDesc> output_ports() const override {
    return outputs_;
  }

  /// Resolves an output port's spec via the user-supplied resolver, falling back to the single
  /// output's static spec.
  OutputSpec output_spec(const std::vector<OutputSpec>& inputs, PortId out_port) const override {
    if (output_spec_fn_)
      return output_spec_fn_(inputs, out_port);
    if (outputs_.size() == 1)
      return outputs_.front().spec;
    return OutputSpec{};
  }

  /// Access the executor factory.
  const StageExecutorFactory& factory() const {
    return factory_;
  }
  /// Access the stage options.
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

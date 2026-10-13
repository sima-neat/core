/**
 * @file
 * @ingroup diagnostics
 * @brief What a Model executes: its nodes, where each one runs, and the tensors between them.
 *
 * `Model::execution_plan(...)` takes the same arguments as `Model::build(...)` and returns the
 * plan that build would execute, without starting a pipeline. `Model::Runner::execution_plan()`
 * returns the plan of a built Runner. Both come from one build path, so they are equal.
 *
 * The plan is read through its versioned JSON document (schema `sima.neat.execution_plan`,
 * documented in `schemas/execution_plan_v1.schema.json`).
 */
#pragma once

#include <memory>
#include <string>

namespace simaai::neat {

namespace internal {
struct ExecutionPlanAccess;
} // namespace internal

/**
 * @brief Execution plan of one Model build, laid out as the graph it runs.
 *
 * `nodes` lists the Model's graph nodes in pipeline order. Each node reports where it runs
 * (`A65`, `EV74`, or `MLA`), the kernel an EV74 or A65 node executes, and the resolved settings
 * of Preproc and SimaBoxDecode nodes. `edges` connect consecutive nodes and list the tensors that
 * flow between them, with batch-first shapes, dtypes, and quantization parameters of quantized
 * tensors. `model` holds the model name, its batch size, and its per-frame device memory.
 * @ingroup diagnostics
 */
class ExecutionPlan {
public:
  /// Construct an empty plan; `to_json()` returns `{}`.
  ExecutionPlan();
  ~ExecutionPlan();
  ExecutionPlan(const ExecutionPlan&);
  ExecutionPlan& operator=(const ExecutionPlan&);
  ExecutionPlan(ExecutionPlan&&) noexcept;
  ExecutionPlan& operator=(ExecutionPlan&&) noexcept;

  /// Version of the JSON schema `to_json()` emits.
  int schema_version() const;
  /// Serialize the plan as JSON; `indent < 0` emits one line.
  std::string to_json(int indent = 2) const;

private:
  friend struct internal::ExecutionPlanAccess;
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

} // namespace simaai::neat

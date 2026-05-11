/**
 * @file
 * @ingroup nodes_sima
 * @brief `Dequant` Node — CVU kernel that dequantizes INT8 to FP32 (scale + zero-point).
 *
 * Reads INT8 tensors and emits FP32 using the per-tensor scale and zero-point bound to
 * the model. Inserted after MLA stages on the INT8 path so downstream postprocess
 * (argmax, NMS, etc.) sees float values.
 *
 * @see "The dtype contract" page in /concepts/dtype_contract
 */
#pragma once

#include "builder/Node.h"
#include "builder/NodeContractConfigurable.h"
#include "builder/NodeContractProvider.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {
class Model;
struct CompiledDequantContract;
struct CompiledProcessCvuContract;

/**
 * @brief Construction options for a `Dequant` Node.
 *
 * @ingroup nodes_sima
 */
struct DequantOptions {
  /// Default-construct with framework-default values; tune fields after construction.
  DequantOptions() = default;
  /// Initialize options from a loaded `Model` (pulls scale/zp from the model's quant params).
  explicit DequantOptions(const simaai::neat::Model& model);

  std::string element_name;        ///< Optional GStreamer element name (default: auto-generated).
  std::string stage_id;            ///< Logical stage identifier used for routing/diagnostics.
  bool model_managed = false;      ///< If true, the bound model owns scale/zp resolution.

  std::optional<double> q_scale;        ///< Per-tensor dequantization scale; required unless `model_managed`.
  std::optional<std::int64_t> q_zp;     ///< Per-tensor dequantization zero-point; required unless `model_managed`.
  std::shared_ptr<const CompiledDequantContract> compiled_contract; ///< Pre-compiled neatdequant contract; bypasses re-compilation.
  std::shared_ptr<const CompiledProcessCvuContract> processcvu_compiled_contract; ///< Pre-compiled processcvu dequantize contract for model-managed A65/CVU routes.

  int num_buffers = 0;             ///< Override for the element's buffer pool size; 0 = use default/model.
  int num_buffers_model = 0;       ///< Buffer count derived from the bound model.
  bool num_buffers_locked = false; ///< If true, planner won't override `num_buffers`.
};

/**
 * @brief CVU kernel Node that dequantizes INT8 tensors to FP32 using scale + zero-point.
 *
 * Inserted by the route planner after MLA stages on the INT8 path. Application code
 * rarely adds this directly — it appears via planner-generated routing or via a
 * `DetessDequant` fusion.
 *
 * @see "The dtype contract" page in /concepts/dtype_contract
 *
 * @ingroup nodes_sima
 */
class Dequant final : public Node,
                      public NodeContractProvider,
                      public NodeContractConfigurable {
public:
  /// Construct with optional `DequantOptions`.
  explicit Dequant(DequantOptions opt = {});
  struct ConfigHolder;

  /// Type label for this Node kind.
  std::string kind() const override {
    return "Dequant";
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  /// Structural contract definition for this Node.
  NodeContractDefinition contract_definition() const override;
  /// Compile this Node's contract from the given input.
  bool compile_node_contract(const ContractCompileInput& input,
                             CompiledNodeContract* out,
                             std::string* err) const override;
  /// Apply a compiled contract back into this Node.
  void apply_compiled_contract(const CompiledNodeContract& contract,
                               std::string* err) override;

  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;

  /// Inspect the Node's options.
  const DequantOptions& options() const {
    return opt_;
  }

#ifdef SIMA_NEAT_INTERNAL
  const std::optional<CompiledDequantContract>& compiled_contract_internal() const;
#endif

private:
  DequantOptions opt_;
  std::shared_ptr<ConfigHolder> config_holder_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `Dequant` Node with optional `DequantOptions`.
std::shared_ptr<simaai::neat::Node> Dequant(DequantOptions opt = {});
} // namespace simaai::neat::nodes

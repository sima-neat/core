/**
 * @file
 * @ingroup nodes_sima
 * @brief `Cast` Node — CVU kernel that converts between FP32 and BF16 element-wise.
 *
 * Pure dtype conversion with no scale/zero-point — distinct from `Quant`/`Dequant`.
 * Inserted at the MLA boundary on the BF16 path: before the MLA when the model expects
 * BF16 input but the upstream stage emits FP32, or after the MLA on the BF16 output path.
 *
 * @see "The dtype contract" page in /concepts/dtype_contract
 */
#pragma once

#include "builder/Node.h"
#include "builder/NodeContractConfigurable.h"
#include "builder/NodeContractProvider.h"
#include "builder/OutputSpec.h"
#ifdef SIMA_NEAT_INTERNAL
#include "model/internal/ModelRouteRetarget.h"
#endif

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {
struct CompiledProcessCvuContract;

/// Direction of the BF16/FP32 conversion performed by a `Cast` Node.
enum class CastDirection {
  Bf16ToFp32 = 0, ///< Read BF16, emit FP32 (typical post-MLA path).
  Fp32ToBf16 = 1, ///< Read FP32, emit BF16 (typical pre-MLA path).
};

/**
 * @brief Construction options for a `Cast` Node.
 *
 * @ingroup nodes_sima
 */
struct CastOptions {
  CastDirection direction = CastDirection::Bf16ToFp32; ///< Conversion direction.
  std::string element_name; ///< Optional GStreamer element name (default: auto-generated).
  bool silent = true;       ///< If true, suppresses element-level diagnostic logging.
  std::shared_ptr<const CompiledProcessCvuContract>
      compiled_contract; ///< Optional processcvu contract for model-managed cast.
  int num_buffers = 0;   ///< processcvu buffer pool size when compiled_contract is set.
#ifdef SIMA_NEAT_INTERNAL
  std::shared_ptr<const simaai::neat::internal::ModelLineageBinding> model_lineage;
#endif
};

/**
 * @brief CVU kernel Node that casts a tensor between FP32 and BF16 (no scale/zero-point).
 *
 * Typically inserted by the route planner at the MLA boundary on the BF16 path. Application
 * code rarely adds `Cast` directly — it appears via `DetessCast`/`CastTess` fusions or via
 * planner-generated routing — but the standalone form exists for graphs that need a bare
 * dtype conversion stage.
 *
 * @see "The dtype contract" page in /concepts/dtype_contract
 *
 * @ingroup nodes_sima
 */
class Cast final : public Node,
                   public OutputSpecProvider,
                   public NodeContractProvider,
                   public NodeContractConfigurable {
public:
  /// Construct with optional `CastOptions`.
  explicit Cast(CastOptions opt = {});

  /// Type label for this Node kind.
  std::string kind() const override {
    return "Cast";
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;
  /// Negotiated downstream caps produced by this Node.
  OutputSpec output_spec(const OutputSpec& input) const override;
  /// Structural contract definition for this Node.
  NodeContractDefinition contract_definition() const override;
  /// Compile this Node's contract from the given input.
  bool compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                             std::string* err) const override;
  /// Apply a compiled contract back into this Node.
  void apply_compiled_contract(const CompiledNodeContract& contract, std::string* err) override;

  /// Inspect the Node's options.
  const CastOptions& options() const {
    return opt_;
  }

private:
  CastOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `Cast` Node with optional `CastOptions`.
std::shared_ptr<simaai::neat::Node> Cast(CastOptions opt = {});
} // namespace simaai::neat::nodes

/**
 * @file
 * @ingroup nodes_sima
 * @brief `DetessDequant` Node — fused CVU kernel that detessellates and dequantizes INT8→FP32.
 *
 * Combines an MLA-layout untessellate with an INT8→FP32 dequantize in a single CVU pass.
 * Inserted after MLA stages on the INT8 output path when downstream wants natural HWC/CHW
 * float tensors — fusing saves a DDR round-trip versus running `Detess` then `Dequant`.
 *
 * @see "The dtype contract" page in /concepts/dtype_contract
 */
#pragma once

#include "builder/Node.h"
#include "builder/NodeContractConfigurable.h"
#include "builder/NodeContractProvider.h"
#include "builder/OutputSpec.h"

#include <nlohmann/json.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {
class Model;
struct CompiledProcessCvuContract;
} // namespace simaai::neat

namespace simaai::neat {

/**
 * @brief Construction options for a `DetessDequant` Node.
 *
 * @ingroup nodes_sima
 */
struct DetessDequantOptions {
  /// Default-construct with framework-default values; tune fields after construction.
  DetessDequantOptions() = default;
  /// Initialize options from a loaded `Model` (pulls tile geometry, scale/zp, etc.).
  explicit DetessDequantOptions(const simaai::neat::Model& model);

  std::string config_path; ///< Path to the kernel config JSON, if loaded from disk.
  std::optional<nlohmann::json>
      config_json;           ///< Inline kernel config; takes precedence over `config_path`.
  std::string upstream_name; ///< Name of the upstream MLA element (used for tag wiring).
  std::string element_name;  ///< Optional GStreamer element name (default: auto-generated).
  std::shared_ptr<const CompiledProcessCvuContract>
      compiled_contract; ///< Pre-compiled CVU contract; bypasses re-compilation.
  int num_buffers = 0;   ///< Override for the element's buffer pool size; 0 = use default/model.
  int num_buffers_model = 0;       ///< Buffer count derived from the bound model.
  bool num_buffers_locked = false; ///< If true, planner won't override `num_buffers`.
};

/**
 * @brief Fused CVU kernel Node: `Detess` followed by `Dequant` (INT8→FP32).
 *
 * The route planner picks `DetessDequant` after MLA stages whenever the INT8 path needs
 * to surface FP32 tensors in natural layout downstream. Application code rarely adds
 * this directly.
 *
 * @see "The dtype contract" page in /concepts/dtype_contract
 *
 * @ingroup nodes_sima
 */
class DetessDequant final : public Node,
                            public NodeContractProvider,
                            public NodeContractConfigurable,
                            public OutputSpecProvider {
public:
  /// Construct with optional `DetessDequantOptions`.
  explicit DetessDequant(DetessDequantOptions opt = {});
  struct ConfigHolder;

  /// Type label for this Node kind.
  std::string kind() const override {
    return "DetessDequant";
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  /// Structural contract definition for this Node.
  NodeContractDefinition contract_definition() const override;
  /// Compile this Node's contract from the given input.
  bool compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                             std::string* err) const override;
  /// Apply a compiled contract back into this Node.
  void apply_compiled_contract(const CompiledNodeContract& contract, std::string* err) override;
  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;
  /// Negotiated downstream caps produced by this Node.
  OutputSpec output_spec(const OutputSpec& input) const override;

  /// Resolved kernel config JSON, or null if no config was supplied/loaded.
  const nlohmann::json* config_json() const;

  /// Inspect the Node's options.
  const DetessDequantOptions& options() const {
    return opt_;
  }
  /// Path to the kernel config JSON, if one was loaded from disk.
  const std::string& config_path() const {
    return config_path_;
  }

#ifdef SIMA_NEAT_INTERNAL
  const std::optional<CompiledProcessCvuContract>& compiled_contract_internal() const;
#endif

private:
  DetessDequantOptions opt_;
  std::shared_ptr<ConfigHolder> config_holder_;
  std::string config_path_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `DetessDequant` Node with optional `DetessDequantOptions`.
std::shared_ptr<simaai::neat::Node> DetessDequant(DetessDequantOptions opt = {});
} // namespace simaai::neat::nodes

/**
 * @file
 * @ingroup nodes_sima
 * @brief `QuantTess` Node — fused CVU kernel that quantizes FP32→INT8 then tessellates.
 *
 * Combines an INT8 quantize and a tessellate layout shuffle into a single CVU pass.
 * Inserted by the route planner before the MLA when the model expects INT8 input *and*
 * MLA-side tessellation is not part of the compiled MLA kernel — fusing the two stages
 * saves a DDR round-trip versus running `Quant` and `Tess` back to back.
 *
 * @see "The dtype contract" page in /concepts/dtype_contract
 */
#pragma once

#include "builder/NodeContractConfigurable.h"
#include "builder/NodeContractProvider.h"
#include "builder/Node.h"
#ifdef SIMA_NEAT_INTERNAL
#include "model/internal/ModelRouteRetarget.h"
#endif

#include <nlohmann/json.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {
class Model;
struct CompiledProcessCvuContract;

/**
 * @brief Construction options for a `QuantTess` Node.
 *
 * @ingroup nodes_sima
 */
struct QuantTessOptions {
  /// Default-construct with framework-default values; tune fields after construction.
  QuantTessOptions() = default;
  /// Initialize options from a loaded `Model` (pulls scale/zp and tile geometry).
  explicit QuantTessOptions(const simaai::neat::Model& model);

  std::string config_path;                                          ///< Path to the kernel config JSON, if loaded from disk.
  std::optional<nlohmann::json> config_json;                        ///< Inline kernel config; takes precedence over `config_path`.
  std::string element_name;                                         ///< Optional GStreamer element name (default: auto-generated).
  std::shared_ptr<const CompiledProcessCvuContract> compiled_contract; ///< Pre-compiled CVU contract; bypasses re-compilation.
  int num_buffers = 0;                                              ///< Override for the element's buffer pool size; 0 = use default/model.
  int num_buffers_model = 0;                                        ///< Buffer count derived from the bound model.
  bool num_buffers_locked = false;                                  ///< If true, planner won't override `num_buffers`.
#ifdef SIMA_NEAT_INTERNAL
  std::shared_ptr<const simaai::neat::internal::ModelLineageBinding> model_lineage;
#endif
};

/**
 * @brief Fused CVU kernel Node: `Quant` (FP32→INT8) followed by `Tess`.
 *
 * The route planner picks `QuantTess` when the MLA wants INT8 input *and* MLA-side
 * tessellation is not present in the compiled kernel. Application code rarely adds
 * this directly.
 *
 * @see "The dtype contract" page in /concepts/dtype_contract
 *
 * @ingroup nodes_sima
 */
class QuantTess final : public Node,
                        public NodeContractProvider,
                        public NodeContractConfigurable {
public:
  /// Construct with optional `QuantTessOptions`.
  explicit QuantTess(QuantTessOptions opt = {});
  struct ConfigHolder;

  /// Type label for this Node kind.
  std::string kind() const override {
    return "QuantTess";
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
  void apply_compiled_contract(const CompiledNodeContract& contract, std::string* err) override;
  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;

  /// Resolved kernel config JSON, or null if no config was supplied/loaded.
  const nlohmann::json* config_json() const;

  /// Inspect the Node's options.
  const QuantTessOptions& options() const {
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
  QuantTessOptions opt_;
  std::shared_ptr<ConfigHolder> config_holder_;
  std::string config_path_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `QuantTess` Node with optional `QuantTessOptions`.
std::shared_ptr<simaai::neat::Node> QuantTess(QuantTessOptions opt = {});
} // namespace simaai::neat::nodes

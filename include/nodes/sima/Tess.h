/**
 * @file
 * @ingroup nodes_sima
 * @brief `Tess` Node — CVU kernel that tessellates a tensor into MLA tile geometry.
 *
 * Pure layout shuffle — rearranges a natural HWC/CHW tensor into the tile-and-stripe
 * arrangement the MLA expects on input. Inserted by the route planner just before MLA
 * stages whenever the upstream stage emits natural layout (i.e. MLA-side tess is not
 * folded into the compiled MLA kernel).
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
 * @brief Construction options for a `Tess` Node.
 *
 * @ingroup nodes_sima
 */
struct TessOptions {
  /// Default-construct with framework-default values; tune fields after construction.
  TessOptions() = default;
  /// Initialize options from a loaded `Model` (pulls tile geometry, tensor shape, etc.).
  explicit TessOptions(const simaai::neat::Model& model);

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
 * @brief CVU kernel Node that tessellates a natural-layout tensor into MLA tile geometry.
 *
 * Inserted by the route planner before MLA stages whenever upstream emits natural
 * HWC/CHW order. Application code rarely adds this directly — it appears via
 * planner-generated routing or via `CastTess`/`QuantTess` fusions.
 *
 * @see "The dtype contract" page in /concepts/dtype_contract
 *
 * @ingroup nodes_sima
 */
class Tess final : public Node,
                   public NodeContractProvider,
                   public NodeContractConfigurable {
public:
  /// Construct with optional `TessOptions`.
  explicit Tess(TessOptions opt = {});
  struct ConfigHolder;

  /// Type label for this Node kind.
  std::string kind() const override {
    return "Tess";
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
  const TessOptions& options() const {
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
  TessOptions opt_;
  std::shared_ptr<ConfigHolder> config_holder_;
  std::string config_path_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `Tess` Node with optional `TessOptions`.
std::shared_ptr<simaai::neat::Node> Tess(TessOptions opt = {});
} // namespace simaai::neat::nodes

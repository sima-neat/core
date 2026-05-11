/**
 * @file
 * @ingroup nodes_sima
 * @brief `Detess` Node — CVU kernel that untessellates an MLA-layout tensor to natural HWC/CHW.
 *
 * Pure layout shuffle — undoes the tile-and-stripe arrangement that the MLA emits, returning
 * tensors in the natural memory order downstream consumers expect. Inserted by the route
 * planner immediately after MLA stages whenever the next consumer is a host-style layout reader.
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
 * @brief Construction options for a `Detess` Node.
 *
 * @ingroup nodes_sima
 */
struct DetessOptions {
  /// Default-construct with framework-default values; tune fields after construction.
  DetessOptions() = default;
  /// Initialize options from a loaded `Model` (pulls tile geometry, tensor shape, etc.).
  explicit DetessOptions(const simaai::neat::Model& model);

  std::string config_path;                                          ///< Path to the kernel config JSON, if loaded from disk.
  std::string config_dir;                                           ///< Directory used when materializing config snapshots to disk.
  bool keep_config = false;                                         ///< If true, retain the materialized config file after run.
  bool no_json_path = false;                                        ///< If true, pass config to the element inline (no file path).
  std::optional<nlohmann::json> config_json;                        ///< Inline kernel config; takes precedence over `config_path`.
  std::string upstream_name;                                        ///< Name of the upstream MLA element (used for tag wiring).
  std::string element_name;                                         ///< Optional GStreamer element name (default: auto-generated).
  std::shared_ptr<const CompiledProcessCvuContract> compiled_contract; ///< Pre-compiled CVU contract; bypasses re-compilation.
  int num_buffers = 0;                                              ///< Override for the element's buffer pool size; 0 = use default/model.
  int num_buffers_model = 0;                                        ///< Buffer count derived from the bound model.
  bool num_buffers_locked = false;                                  ///< If true, planner won't override `num_buffers`.
};

/**
 * @brief CVU kernel Node that detessellates MLA-layout tensors back to natural HWC/CHW order.
 *
 * Inserted by the route planner whenever an MLA stage feeds a host-style layout consumer
 * and the dtype path does not justify a fused `DetessCast`/`DetessDequant`. Application
 * code rarely adds this directly.
 *
 * @see "The dtype contract" page in /concepts/dtype_contract
 *
 * @ingroup nodes_sima
 */
class Detess final : public Node,
                     public OutputSpecProvider,
                     public NodeContractProvider,
                     public NodeContractConfigurable {
public:
  /// Construct with optional `DetessOptions`.
  explicit Detess(DetessOptions opt = {});
  struct ConfigHolder;

  /// Type label for this Node kind.
  std::string kind() const override {
    return "Detess";
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
  bool compile_node_contract(const ContractCompileInput& input,
                             CompiledNodeContract* out,
                             std::string* err) const override;
  /// Apply a compiled contract back into this Node.
  void apply_compiled_contract(const CompiledNodeContract& contract, std::string* err) override;

  /// Resolved kernel config JSON, or null if no config was supplied/loaded.
  const nlohmann::json* config_json() const;

  /// Inspect the Node's options.
  const DetessOptions& options() const {
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
  DetessOptions opt_;
  std::shared_ptr<ConfigHolder> config_holder_;
  std::string config_path_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `Detess` Node with optional `DetessOptions`.
std::shared_ptr<simaai::neat::Node> Detess(DetessOptions opt = {});
} // namespace simaai::neat::nodes

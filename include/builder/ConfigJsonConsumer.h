/**
 * @file
 * @ingroup builder
 * @brief Mixin for Nodes that consume an upstream Node's JSON config.
 *
 * Some Nodes need to know about their upstream neighbour's config — e.g., a
 * postprocess Node may need the model's preprocess JSON to invert a
 * normalization, or a quant Node may need the upstream cast's dtype. Nodes
 * implementing `ConfigJsonConsumer` get called by the Builder's wiring pass
 * with the upstream config and its kind so they can pull whatever they need.
 */
#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace simaai::neat {

/**
 * @brief Mixin interface implemented by Nodes that read upstream config JSON.
 *
 * The Builder invokes `apply_upstream_config()` once per relevant upstream
 * Node during the wiring pass. The Node should be tolerant: not every upstream
 * kind will be relevant to it, and the JSON layout depends on the upstream's
 * own schema.
 *
 * @ingroup builder
 * @see ConfigJsonProvider
 * @see ConfigJsonOverride
 * @see ConfigJsonWire
 */
class ConfigJsonConsumer {
public:
  virtual ~ConfigJsonConsumer() = default;

  /**
   * @brief Apply config from an upstream Node.
   *
   * @param upstream The upstream Node's JSON config (read-only view).
   * @param upstream_kind The upstream Node's `kind()` string (used to dispatch).
   */
  virtual void apply_upstream_config(const nlohmann::json& upstream,
                                     const std::string& upstream_kind) = 0;
};

} // namespace simaai::neat

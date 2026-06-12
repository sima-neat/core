/**
 * @file
 * @ingroup builder
 * @brief Mixin for Nodes that expose a read-only view of their JSON config.
 *
 * Nodes that own a JSON config (e.g., loaded from a model pack manifest)
 * implement `ConfigJsonProvider` so downstream Nodes — via `ConfigJsonConsumer`
 * — and diagnostic tooling can introspect it without mutating it.
 */
#pragma once

#include <nlohmann/json.hpp>

namespace simaai::neat {

/**
 * @brief Mixin interface implemented by Nodes that hold a JSON config.
 *
 * The Builder reads the returned pointer (which may be null if no config
 * exists). Lifetime: the pointed-to JSON must remain valid for as long as
 * the Node is referenced.
 *
 * @ingroup builder
 * @see ConfigJsonConsumer
 * @see ConfigJsonOverride
 */
class ConfigJsonProvider {
public:
  virtual ~ConfigJsonProvider() = default;

  /// @brief Return a non-owning pointer to the Node's JSON config (or null if absent).
  virtual const nlohmann::json* config_json() const = 0;
};

} // namespace simaai::neat

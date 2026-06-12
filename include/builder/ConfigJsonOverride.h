/**
 * @file
 * @ingroup builder
 * @brief Mixin for Nodes that accept a programmatic JSON override.
 *
 * `ConfigJsonOverride` lets the Builder (or app code) patch a Node's JSON
 * config in-place via a callback. The override is identified by a `tag` so
 * Nodes can decide whether to honor it (some overrides are scoped to specific
 * pipeline phases).
 */
#pragma once

#include <functional>

#include <nlohmann/json.hpp>

namespace simaai::neat {

/**
 * @brief Mixin interface implemented by Nodes whose JSON config can be patched.
 *
 * The framework or test code calls `override_config_json()` with an editor
 * callback that mutates the JSON in place. The Node returns `true` if the
 * override applied (i.e., it owns config JSON and the tag was recognised).
 *
 * @ingroup builder
 * @see ConfigJsonOverrideMulti
 * @see ConfigJsonProvider
 */
class ConfigJsonOverride {
public:
  virtual ~ConfigJsonOverride() = default;

  /**
   * @brief Apply an in-place JSON edit to this Node's config.
   *
   * @param edit Callback invoked with the Node's mutable JSON.
   * @param tag  Caller-supplied tag identifying the override (used by Nodes to gate behavior).
   * @return True if the override was applied.
   */
  virtual bool override_config_json(const std::function<void(nlohmann::json&)>& edit,
                                    const std::string& tag) = 0;
};

} // namespace simaai::neat

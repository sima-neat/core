/**
 * @file
 * @ingroup builder
 * @brief Mixin for Nodes that accept a multi-instance JSON override.
 *
 * Variant of `ConfigJsonOverride` for Nodes that internally hold an array of
 * configs (one per child stage / channel / output). The editor callback is
 * passed `(json, index, count)` so it can patch the right element.
 */
#pragma once

#include <functional>
#include <cstddef>

#include <nlohmann/json.hpp>

namespace simaai::neat {

/**
 * @brief Mixin interface implemented by Nodes whose JSON config is a list/array.
 *
 * The Node calls the editor once per element it holds, providing the index
 * and total count so the editor can target a specific entry. Returns `true`
 * if any element accepted the override.
 *
 * @ingroup builder
 * @see ConfigJsonOverride
 */
class ConfigJsonOverrideMulti {
public:
  virtual ~ConfigJsonOverrideMulti() = default;

  /**
   * @brief Apply an in-place JSON edit to each config element this Node holds.
   *
   * @param edit Callback invoked as `edit(element_json, index, count)`.
   * @param tag  Caller-supplied tag identifying the override.
   * @return True if the override was applied to at least one element.
   */
  virtual bool override_config_json_multi(
      const std::function<void(nlohmann::json&, std::size_t, std::size_t)>& edit,
      const std::string& tag) = 0;
};

} // namespace simaai::neat

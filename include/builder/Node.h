/**
 * @file
 * @ingroup builder
 * @brief Builder-level Node interface and caps behavior metadata.
 */
#pragma once

#include "contracts/ContractTypes.h"

#include <string>
#include <vector>

namespace simaai::neat {

enum class InputRole {
  None,
  Push,
  Source,
};

enum class NodeCapsBehavior {
  Static,
  Dynamic,
};

inline const char* node_caps_behavior_name(NodeCapsBehavior behavior) {
  switch (behavior) {
  case NodeCapsBehavior::Static:
    return "static";
  case NodeCapsBehavior::Dynamic:
    return "dynamic";
  }
  return "unknown";
}

// =============================
// Builder Node API
// =============================
class Node {
public:
  virtual ~Node() = default;

  // Deterministic type label (used in reports).
  virtual std::string kind() const = 0;

  // Optional human label (user-supplied).
  virtual std::string user_label() const {
    return "";
  }

  // Node fragment with deterministic element names (namespace = n<idx>_...).
  virtual std::string backend_fragment(int node_index) const = 0;

  // Deterministic list of element names this node creates.
  virtual std::vector<std::string> element_names(int node_index) const = 0;

  // Optional buffer-name hint used for wiring (e.g., config node_name).
  virtual std::string buffer_name_hint(int /*node_index*/) const {
    return "";
  }

  // Caps behavior contract (dynamic vs static).
  virtual NodeCapsBehavior caps_behavior() const = 0;

  // Optional memory contract for this node (runner may still override).
  virtual MemoryContract memory_contract() const {
    return MemoryContract::AllowEitherButReport;
  }

  // Input role metadata used to validate run() vs run(input) usage.
  virtual InputRole input_role() const {
    return InputRole::None;
  }

  // Whether this node is backed by a config JSON.
  // Legacy note: framework build no longer rewrites JSON wiring fields.
  virtual bool has_config_json() const {
    return false;
  }

  // Legacy hook for node-local JSON rewrites.
  // Framework build does not call this for automatic wiring anymore.
  virtual bool wire_input_names(const std::vector<std::string>& upstream_names,
                                const std::string& tag) {
    (void)upstream_names;
    (void)tag;
    return false;
  }
};

} // namespace simaai::neat

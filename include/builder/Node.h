/**
 * @file
 * @ingroup builder
 * @brief Builder-level `Node` interface — the smallest building block of a pipeline.
 *
 * Every step in a Neat-framework pipeline is a `Node`: the file reader, the H.264 decoder,
 * the resizer, the model's preprocess, the model's MLA inference, the postprocess, the
 * output sink. Nodes snap together in a `Session` to form a pipeline. This header defines
 * the abstract `Node` base class that every concrete node implements, plus the small
 * vocabulary of metadata enums (`InputRole`, `NodeCapsBehavior`).
 *
 * Writing a custom Node is the framework's primary extension point — see "Building a custom
 * Node" (§0.10) of the design deep dive. The contract is small: implement `kind()`,
 * `caps_behavior()`, `backend_fragment()`, `element_names()`, and you're done.
 *
 * @see NodeGroup for bundles of Nodes
 * @see Session for how Nodes get composed into a pipeline
 * @see "Nodes: the building-block philosophy" (§0.10 of the design deep dive)
 */
#pragma once

#include "contracts/ContractTypes.h"

#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Whether a Node accepts external input and how.
 *
 * Used by the framework to validate that the right run-mode is chosen — e.g., a Session
 * starting with a `Source`-role Node uses `run()` (no input), while a Session starting with
 * a `Push`-role Node uses `run(inputs)` or `build(inputs)`.
 * @ingroup builder
 */
enum class InputRole {
  None,   ///< Internal node; doesn't accept external input directly.
  Push,   ///< Node receives input via `Run::push()` (appsrc-style).
  Source, ///< Node generates its own input (e.g., file source, RTSP source, image freeze).
};

/**
 * @brief Whether a Node's GStreamer caps are fixed at build time or adapt at runtime.
 *
 * Nodes that always emit the same caps regardless of input (e.g., a constant generator)
 * are `Static`; nodes whose output caps depend on input caps (e.g., format conversion,
 * resize) are `Dynamic`. The Session uses this to validate caps negotiation paths.
 * @ingroup builder
 */
enum class NodeCapsBehavior {
  Static,  ///< Output caps are fixed; don't depend on upstream caps.
  Dynamic, ///< Output caps are derived from upstream caps at build/runtime.
};

/// Returns a stable string for a `NodeCapsBehavior` (for diagnostics and reports).
inline const char* node_caps_behavior_name(NodeCapsBehavior behavior) {
  switch (behavior) {
  case NodeCapsBehavior::Static:
    return "static";
  case NodeCapsBehavior::Dynamic:
    return "dynamic";
  }
  return "unknown";
}

/**
 * @brief Abstract base class every concrete Node implements.
 *
 * A Node is a typed wrapper over a GStreamer fragment. It exposes a small set of pure-virtual
 * methods that the Session uses to stitch fragments together into a deterministic pipeline.
 *
 * **Minimum implementation**: override `kind()`, `caps_behavior()`, `backend_fragment()`, and
 * `element_names()`. The other methods have sensible defaults.
 *
 * @code
 * class MyNode : public sima::Node {
 *  public:
 *   std::string kind() const override { return "MyNode"; }
 *   NodeCapsBehavior caps_behavior() const override { return NodeCapsBehavior::Dynamic; }
 *   std::string backend_fragment(int i) const override {
 *     return "my_element name=n" + std::to_string(i) + "_my";
 *   }
 *   std::vector<std::string> element_names(int i) const override {
 *     return { "n" + std::to_string(i) + "_my" };
 *   }
 * };
 * @endcode
 *
 * @see "Building a custom Node" (§0.10 of the design deep dive)
 * @ingroup builder
 */
class Node {
public:
  virtual ~Node() = default;

  /// Deterministic type label used in reports and diagnostics (e.g., `"FileInput"`,
  /// `"H264Decode"`).
  virtual std::string kind() const = 0;

  /// Optional human-readable label set by the user; default is empty.
  virtual std::string user_label() const {
    return "";
  }

  /**
   * @brief Returns the GStreamer launch-string fragment this Node emits.
   *
   * The fragment must use deterministic element names of the form `n<node_index>_<role>`. The
   * Session concatenates fragments from all Nodes (with `!` separators) to form the full
   * pipeline string.
   *
   * @param node_index The Node's position in the Session's ordered Node list.
   * @return GStreamer fragment for this Node.
   */
  virtual std::string backend_fragment(int node_index) const = 0;

  /**
   * @brief Lists the deterministic element names this Node will create.
   *
   * Used by `Session::describe()`, the `repro_gst_launch` reproducer, and anywhere the
   * framework needs a stable handle to specific elements.
   */
  virtual std::vector<std::string> element_names(int node_index) const = 0;

  /// Optional buffer-name hint used by config-JSON-driven wiring; default empty.
  virtual std::string buffer_name_hint(int /*node_index*/) const {
    return "";
  }

  /// Whether this Node's caps are static or dynamic. Required.
  virtual NodeCapsBehavior caps_behavior() const = 0;

  /// Memory contract this Node prefers for its buffers. Default lets the runner decide.
  virtual MemoryContract memory_contract() const {
    return MemoryContract::AllowEitherButReport;
  }

  /// Whether this Node is a source, a push input, or neither. Default: `None`.
  virtual InputRole input_role() const {
    return InputRole::None;
  }

  /// Whether this Node is configured by a JSON file. Used by legacy wiring paths; new framework
  /// builds don't rewrite JSON.
  virtual bool has_config_json() const {
    return false;
  }

  /**
   * @brief Legacy hook for node-local JSON rewrites — no longer called by the framework's build.
   *
   * Kept for backward compatibility. New Nodes don't need to override this.
   */
  virtual bool wire_input_names(const std::vector<std::string>& upstream_names,
                                const std::string& tag) {
    (void)upstream_names;
    (void)tag;
    return false;
  }
};

} // namespace simaai::neat

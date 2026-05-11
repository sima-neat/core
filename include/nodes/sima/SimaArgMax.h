/**
 * @file
 * @ingroup nodes_sima
 * @brief `SimaArgMax` Node — postprocess argmax over a tensor (classification).
 *
 * Runs on the EV74. Reads a classification tensor (after dequant/cast) and emits the
 * top class index per sample. Place after `Detess`/`DetessDequant`/`DetessCast` at the
 * tail of a classification pipeline.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Construction options for a `SimaArgMax` Node.
 *
 * @ingroup nodes_sima
 */
struct SimaArgMaxOptions {
  std::string config_path;        ///< Optional path to an external argmax config JSON.
  int sima_allocator_type = 2;    ///< SiMa DMA allocator selector (board-specific; default `2`).
  bool silent = true;             ///< If true, suppress element-level diagnostic logging.
  bool emit_signals = false;      ///< Enable GStreamer signal emission for buffer hand-off.
  bool transmit = false;          ///< If true, transmit results downstream via the element's transport.
};

/**
 * @brief EV74 postprocess Node that emits the top-class index from a classification tensor.
 *
 * @ingroup nodes_sima
 */
class SimaArgMax final : public Node, public OutputSpecProvider {
public:
  /// Construct with optional `SimaArgMaxOptions`.
  explicit SimaArgMax(SimaArgMaxOptions opt = {});

  /// Type label for this Node kind.
  std::string kind() const override {
    return "SimaArgMax";
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  /// Whether this Node is driven by a JSON config.
  bool has_config_json() const override;
  /// Legacy hook for node-local JSON wiring.
  bool wire_input_names(const std::vector<std::string>& upstream_names,
                        const std::string& tag) override;
  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;
  /// Negotiated downstream caps produced by this Node.
  OutputSpec output_spec(const OutputSpec& input) const override;

  /// Inspect the Node's options.
  const SimaArgMaxOptions& options() const {
    return opt_;
  }

private:
  struct ConfigHolder;

  SimaArgMaxOptions opt_;
  std::shared_ptr<ConfigHolder> config_holder_;
  std::string config_path_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `SimaArgMax` Node with optional `SimaArgMaxOptions`.
std::shared_ptr<simaai::neat::Node> SimaArgMax(SimaArgMaxOptions opt = {});
} // namespace simaai::neat::nodes

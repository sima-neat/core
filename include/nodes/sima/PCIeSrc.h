/**
 * @file
 * @ingroup nodes_sima
 * @brief PCIe source node for receiving data from host via PCIe.
 *
 * Wraps the legacy `simaaipciesrc` GStreamer element which uses standard
 * GStreamer buffer allocation (no SiMa DMA allocator), keeping memory
 * usage low and compatible with all board configurations.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

struct PCIeSrcOptions {
  /// Size in bytes of each incoming frame buffer.
  /// Default 4 MB matches the simaaipciesrc default.
  int buffer_size = 4194304;

  /// Optional caps enforcement — when format, width, and height are all
  /// set, a capsfilter is appended to lock the negotiated format.
  std::string format;
  int width = -1;
  int height = -1;
  int fps_n = 0;
  int fps_d = 1;
};

class PCIeSrc final : public Node, public OutputSpecProvider {
public:
  explicit PCIeSrc(PCIeSrcOptions opt = {});

  std::string kind() const override {
    return "PCIeSrc";
  }
  std::string user_label() const override {
    return "pciesrc";
  }
  InputRole input_role() const override {
    return InputRole::Source;
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  const PCIeSrcOptions& options() const {
    return opt_;
  }

private:
  PCIeSrcOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> PCIeSrc(PCIeSrcOptions opt = {});
} // namespace simaai::neat::nodes

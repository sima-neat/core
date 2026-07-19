/**
 * @file
 * @ingroup nodes_rtp
 * @brief `H265Depacketize` Node — extracts and parses H.265 access units from RTP packets.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <vector>

namespace simaai::neat {

/**
 * @brief Depayloads RTP-encapsulated H.265 into parsed Annex-B access units.
 *
 * @ingroup nodes_rtp
 */
class H265Depacketize final : public Node, public OutputSpecProvider {
public:
  /// Construct with an RTP payload type. Values `<= 0` disable payload filtering.
  explicit H265Depacketize(int payload_type = 96);

  std::string kind() const override {
    return "H265Depacketize";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  int payload_type() const {
    return payload_type_;
  }

private:
  int payload_type_ = 96;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for an `H265Depacketize` Node.
std::shared_ptr<simaai::neat::Node> H265Depacketize(int payload_type = 96);
} // namespace simaai::neat::nodes

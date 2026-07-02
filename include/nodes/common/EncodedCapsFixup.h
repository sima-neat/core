/**
 * @file
 * @ingroup nodes_common
 * @brief `EncodedCapsFixup` Node - repairs missing encoded stream caps fields.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/// Configuration for `EncodedCapsFixup`.
struct EncodedCapsFixupOptions {
  std::string media_type; ///< Encoded media type to patch, e.g. `"image/jpeg"`.
  int fallback_fps = -1;  ///< Framerate inserted when caps omit a valid rate; `-1` = no patch.
};

/**
 * @brief Identity node whose runtime probe fills invalid encoded caps fields.
 *
 * Use before native decode when an encoded source can negotiate caps such as
 * `image/jpeg,framerate=0/1` but the downstream decoder requires a positive
 * stream framerate.
 *
 * @ingroup nodes_common
 */
class EncodedCapsFixup final : public Node {
public:
  explicit EncodedCapsFixup(EncodedCapsFixupOptions opt);

  std::string kind() const override {
    return "EncodedCapsFixup";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  const EncodedCapsFixupOptions& options() const {
    return opt_;
  }

private:
  EncodedCapsFixupOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {

/// Convenience factory for an `EncodedCapsFixup` Node.
std::shared_ptr<simaai::neat::Node> EncodedCapsFixup(EncodedCapsFixupOptions opt);

} // namespace simaai::neat::nodes

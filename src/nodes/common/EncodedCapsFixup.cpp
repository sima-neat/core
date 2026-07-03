#include "nodes/common/EncodedCapsFixup.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

EncodedCapsFixup::EncodedCapsFixup(EncodedCapsFixupOptions opt) : opt_(std::move(opt)) {
  if (opt_.media_type.empty()) {
    throw std::invalid_argument("EncodedCapsFixup: media_type must be set");
  }
}

std::string EncodedCapsFixup::backend_fragment(int node_index) const {
  return "identity name=n" + std::to_string(node_index) + "_encoded_capsfix silent=true";
}

std::vector<std::string> EncodedCapsFixup::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_encoded_capsfix"};
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> EncodedCapsFixup(EncodedCapsFixupOptions opt) {
  return std::make_shared<simaai::neat::EncodedCapsFixup>(std::move(opt));
}

} // namespace simaai::neat::nodes

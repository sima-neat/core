#include "nodes/sima/PCIeSrc.h"

#include "gst/GstHelpers.h"

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
PCIeSrc::PCIeSrc(PCIeSrcOptions opt) : opt_(std::move(opt)) {}

std::string PCIeSrc::backend_fragment(int node_index) const {
  require_element("neatpciesrc", "PCIeSrc::backend_fragment");

  std::ostringstream ss;
  ss << "neatpciesrc name=n" << node_index << "_pciesrc" << " queue=" << opt_.queue
     << " buffer-size=" << opt_.buffer_size;
  if (opt_.pool_size > 0) {
    ss << " pool-size=" << opt_.pool_size;
  }
  return ss.str();
}

std::vector<std::string> PCIeSrc::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_pciesrc"};
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> PCIeSrc(PCIeSrcOptions opt) {
  return std::make_shared<simaai::neat::PCIeSrc>(std::move(opt));
}

} // namespace simaai::neat::nodes

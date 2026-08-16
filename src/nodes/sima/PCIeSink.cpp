#include "nodes/sima/PCIeSink.h"

#include "gst/GstHelpers.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
namespace {

void require_no_quotes(const std::string& value, const char* field) {
  if (value.find('"') != std::string::npos) {
    throw std::invalid_argument(std::string(field) + " must not contain '\"': " + value);
  }
}

} // namespace

PCIeSink::PCIeSink(PCIeSinkOptions opt) : opt_(std::move(opt)) {}

std::string PCIeSink::backend_fragment(int node_index) const {
  require_element("neatpciesink", "PCIeSink::backend_fragment");
  require_no_quotes(opt_.config_file, "config_file");

  std::ostringstream ss;
  ss << "neatpciesink name=n" << node_index << "_pciesink queue=" << opt_.queue;

  if (!opt_.config_file.empty()) {
    ss << " config=\"" << opt_.config_file << "\"";
  }
  ss << " transmit=" << (opt_.transmit_kpi ? "true" : "false");

  return ss.str();
}

std::vector<std::string> PCIeSink::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_pciesink"};
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> PCIeSink(PCIeSinkOptions opt) {
  return std::make_shared<simaai::neat::PCIeSink>(std::move(opt));
}

} // namespace simaai::neat::nodes

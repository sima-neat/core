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
  require_no_quotes(opt_.data_buf_name, "data_buf_name");
  require_no_quotes(opt_.config_file, "config_file");
  require_no_quotes(opt_.param_buf_name, "param_buf_name");

  std::ostringstream ss;
  ss << "neatpciesink name=n" << node_index << "_pciesink" << " data-buf-name=\""
     << opt_.data_buf_name << "\"" << " data-buffer-size=" << opt_.data_buffer_size
     << " num-buffers=" << opt_.num_buffers << " queue=" << opt_.queue
     << " sync=" << (opt_.sync ? "true" : "false")
     << " async=" << (opt_.async_state ? "true" : "false");

  if (!opt_.config_file.empty()) {
    ss << " config=\"" << opt_.config_file << "\"";
  }
  if (opt_.use_multi_buffers) {
    ss << " use-multi-buffers=true" << " param-buf-name=\"" << opt_.param_buf_name << "\""
       << " param-buffer-size=" << opt_.param_buffer_size;
  }
  if (opt_.max_lateness_ns >= 0) {
    ss << " max-lateness=" << opt_.max_lateness_ns;
  }

  ss << " processing-deadline=" << opt_.processing_deadline_ns
     << " transmit=" << (opt_.transmit_kpi ? "true" : "false")
     << " qos=" << (opt_.qos ? "true" : "false");

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

#include "nodes/common/Output.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/ErrorUtil.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace simaai::neat {

OutputOptions OutputOptions::Latest() {
  OutputOptions opt;
  opt.max_buffers = 1;
  opt.drop = true;
  opt.sync = false;
  return opt;
}

OutputOptions OutputOptions::EveryFrame(int max_buffers) {
  OutputOptions opt;
  opt.drop = false;
  opt.sync = false;
  opt.max_buffers = max_buffers;
  return opt;
}

OutputOptions OutputOptions::Clocked(int max_buffers) {
  OutputOptions opt;
  opt.drop = true;
  opt.sync = true;
  opt.max_buffers = max_buffers;
  return opt;
}

Output::Output(std::string name, OutputOptions opt)
    : opt_(std::move(opt)), endpoint_name_(std::move(name)) {
  if (opt_.max_buffers < 0) {
    pipeline_internal::error_util::throw_session_error(error_codes::kOptionOutOfRange,
                                                       "OutputOptions.max_buffers must be >= 0");
  }
}

std::string Output::backend_fragment(int /*node_index*/) const {
  std::ostringstream ss;
  ss << "appsink name=mysink emit-signals=false " << "sync=" << (opt_.sync ? "true" : "false")
     << " " << "max-buffers=" << opt_.max_buffers << " "
     << "drop=" << (opt_.drop ? "true" : "false") << " " << "enable-last-sample=false";
  return ss.str();
}

std::vector<std::string> Output::element_names(int /*node_index*/) const {
  return {"mysink"};
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> Output(OutputOptions opt) {
  return std::make_shared<simaai::neat::Output>(std::move(opt));
}

std::shared_ptr<simaai::neat::Node> Output(std::string name, OutputOptions opt) {
  return std::make_shared<simaai::neat::Output>(std::move(name), std::move(opt));
}

} // namespace simaai::neat::nodes

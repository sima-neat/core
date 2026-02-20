#include "nodes/common/FileInput.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

FileInput::FileInput(std::string path) : path_(std::move(path)) {}

std::string FileInput::backend_fragment(int node_index) const {
  const std::string el = "n" + std::to_string(node_index) + "_filesrc";
  std::ostringstream ss;
  ss << "filesrc name=" << el << " location=\"" << path_ << "\"";
  return ss.str();
}

std::vector<std::string> FileInput::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_filesrc"};
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> FileInput(std::string path) {
  return std::make_shared<simaai::neat::FileInput>(std::move(path));
}

} // namespace simaai::neat::nodes

#include "nodes/common/Queue.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

std::string Queue::backend_fragment(int node_index) const {
  return "queue name=n" + std::to_string(node_index) + "_queue";
}

std::vector<std::string> Queue::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_queue"};
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> Queue() {
  return std::make_shared<simaai::neat::Queue>();
}

} // namespace simaai::neat::nodes

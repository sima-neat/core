#include "nodes/common/Queue.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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

namespace {

class ConfiguredQueue final : public simaai::neat::Queue {
public:
  explicit ConfiguredQueue(simaai::neat::QueueOptions options) : options_(std::move(options)) {
    if (options_.max_buffers.has_value() && *options_.max_buffers <= 0) {
      throw std::invalid_argument("QueueOptions.max_buffers must be positive when set");
    }
  }

  std::string backend_fragment(int node_index) const override {
    std::string fragment = simaai::neat::Queue::backend_fragment(node_index);
    if (options_.max_buffers.has_value()) {
      fragment += " max-size-buffers=" + std::to_string(*options_.max_buffers) +
                  " max-size-bytes=0 max-size-time=0";
    }
    if (options_.overflow_policy == simaai::neat::OverflowPolicy::KeepLatest) {
      fragment += " leaky=downstream";
    } else if (options_.overflow_policy == simaai::neat::OverflowPolicy::DropIncoming) {
      fragment += " leaky=upstream";
    }
    return fragment;
  }

private:
  simaai::neat::QueueOptions options_;
};

} // namespace

std::shared_ptr<simaai::neat::Node> Queue() {
  return std::make_shared<simaai::neat::Queue>();
}

std::shared_ptr<simaai::neat::Node> Queue(simaai::neat::QueueOptions options) {
  return std::make_shared<ConfiguredQueue>(std::move(options));
}

} // namespace simaai::neat::nodes

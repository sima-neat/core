#include "nodes/common/ImageFreeze.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "builder/Node.h"

namespace {

class ImageFreezeNode final : public simaai::neat::Node {
public:
  explicit ImageFreezeNode(int num_buffers) : num_buffers_(num_buffers) {}

  std::string kind() const override {
    return "ImageFreeze";
  }
  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override {
    std::ostringstream ss;
    ss << "imagefreeze name=n" << node_index << "_imagefreeze";
    if (num_buffers_ > 0)
      ss << " num-buffers=" << num_buffers_;
    return ss.str();
  }

  std::vector<std::string> element_names(int node_index) const override {
    return {std::string("n") + std::to_string(node_index) + "_imagefreeze"};
  }

private:
  int num_buffers_ = -1;
};

} // namespace

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> ImageFreeze(int num_buffers) {
  return std::make_shared<ImageFreezeNode>(num_buffers);
}

} // namespace simaai::neat::nodes

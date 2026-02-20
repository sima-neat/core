#include "nodes/common/VideoRate.h"

#include <memory>
#include <string>
#include <vector>

#include "builder/Node.h"

namespace {

class VideoRateNode final : public simaai::neat::Node {
public:
  std::string kind() const override {
    return "VideoRate";
  }
  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override {
    return "videorate name=n" + std::to_string(node_index) + "_videorate";
  }

  std::vector<std::string> element_names(int node_index) const override {
    return {"n" + std::to_string(node_index) + "_videorate"};
  }
};

} // namespace

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> VideoRate() {
  return std::make_shared<VideoRateNode>();
}

} // namespace simaai::neat::nodes

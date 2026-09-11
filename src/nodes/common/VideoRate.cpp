#include "nodes/common/VideoRate.h"

#include <memory>
#include <string>
#include <vector>

#include "builder/Node.h"

namespace {

class VideoRateNode final : public simaai::neat::Node {
public:
  explicit VideoRateNode(bool drop_only) : drop_only_(drop_only) {}

  std::string kind() const override {
    return "VideoRate";
  }
  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override {
    return "videorate name=n" + std::to_string(node_index) +
           "_videorate drop-only=" + (drop_only_ ? "true" : "false");
  }

  std::vector<std::string> element_names(int node_index) const override {
    return {"n" + std::to_string(node_index) + "_videorate"};
  }

private:
  bool drop_only_;
};

} // namespace

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> VideoRate() {
  return VideoRate(true);
}

std::shared_ptr<simaai::neat::Node> VideoRate(bool drop_only) {
  return std::make_shared<VideoRateNode>(drop_only);
}

} // namespace simaai::neat::nodes

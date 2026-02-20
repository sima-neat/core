/**
 * @file
 * @ingroup nodes_sima
 * @brief SimaAI H264 encode node wrapper.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

class H264EncodeSima final : public Node, public OutputSpecProvider {
public:
  H264EncodeSima(int w, int h, int fps, int bitrate_kbps = 4000, std::string profile = "baseline",
                 std::string level = "4.0");

  std::string kind() const override {
    return "H264EncodeSima";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  int width() const {
    return w_;
  }
  int height() const {
    return h_;
  }
  int fps() const {
    return fps_;
  }
  int bitrate_kbps() const {
    return bitrate_kbps_;
  }
  const std::string& profile() const {
    return profile_;
  }
  const std::string& level() const {
    return level_;
  }

private:
  int w_ = 0;
  int h_ = 0;
  int fps_ = 30;

  int bitrate_kbps_ = 4000;
  std::string profile_ = "baseline";
  std::string level_ = "4.0";
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> H264EncodeSima(int w, int h, int fps, int bitrate_kbps = 4000,
                                                   std::string profile = "baseline",
                                                   std::string level = "4.0");

// picks x264enc/openh264enc/avenc_h264
std::shared_ptr<simaai::neat::Node> H264EncodeSW(int bitrate_kbps = 4000);
} // namespace simaai::neat::nodes

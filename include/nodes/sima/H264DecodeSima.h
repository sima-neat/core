/**
 * @file
 * @ingroup nodes_sima
 * @brief SimaAI H264 decode node wrapper.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

class H264Decode final : public Node, public OutputSpecProvider {
public:
  H264Decode(int sima_allocator_type = 2, std::string out_format = "NV12",
             std::string decoder_name = {}, bool raw_output = false, std::string next_element = {},
             int dec_width = -1, int dec_height = -1, int dec_fps = -1, int num_buffers = -1);
  std::string kind() const override {
    return "H264Decode";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  int sima_allocator_type() const {
    return sima_allocator_type_;
  }
  const std::string& out_format() const {
    return out_format_;
  }
  const std::string& decoder_name() const {
    return decoder_name_;
  }
  bool raw_output() const {
    return raw_output_;
  }
  const std::string& next_element() const {
    return next_element_;
  }
  int dec_width() const {
    return dec_width_;
  }
  int dec_height() const {
    return dec_height_;
  }
  int dec_fps() const {
    return dec_fps_;
  }
  int num_buffers() const {
    return num_buffers_;
  }

private:
  int sima_allocator_type_ = 2;
  std::string out_format_ = "NV12";
  std::string decoder_name_;
  bool raw_output_ = false;
  std::string next_element_;
  int dec_width_ = -1;
  int dec_height_ = -1;
  int dec_fps_ = -1;
  int num_buffers_ = -1;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node>
H264Decode(int sima_allocator_type = 2, std::string out_format = "NV12",
           std::string decoder_name = {}, bool raw_output = false, std::string next_element = {},
           int dec_width = -1, int dec_height = -1, int dec_fps = -1, int num_buffers = -1);
} // namespace simaai::neat::nodes

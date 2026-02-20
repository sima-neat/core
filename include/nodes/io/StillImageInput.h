/**
 * @file
 * @ingroup nodes_io
 * @brief StillImageInput node for RTSP server mode.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

class StillImageInput final : public Node, public OutputSpecProvider {
public:
  struct ContentWidth {
    int value = 0;
    constexpr ContentWidth() = default;
    constexpr ContentWidth(int v) : value(v) {}
  };

  struct ContentHeight {
    int value = 0;
    constexpr ContentHeight() = default;
    constexpr ContentHeight(int v) : value(v) {}
  };

  struct EncodeWidth {
    int value = 0;
    constexpr EncodeWidth() = default;
    constexpr EncodeWidth(int v) : value(v) {}
  };

  struct EncodeHeight {
    int value = 0;
    constexpr EncodeHeight() = default;
    constexpr EncodeHeight(int v) : value(v) {}
  };

  struct FramesPerSecond {
    int value = 30;
    constexpr FramesPerSecond() = default;
    constexpr FramesPerSecond(int v) : value(v) {}
  };

  StillImageInput(std::string image_path, ContentWidth content_w, ContentHeight content_h,
                  EncodeWidth enc_w, EncodeHeight enc_h, FramesPerSecond fps);

  std::string kind() const override {
    return "StillImageInput";
  }
  std::string user_label() const override {
    return image_path_;
  }
  InputRole input_role() const override {
    return InputRole::Source;
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  const std::string& image_path() const {
    return image_path_;
  }
  int content_w() const {
    return content_w_;
  }
  int content_h() const {
    return content_h_;
  }
  int enc_w() const {
    return enc_w_;
  }
  int enc_h() const {
    return enc_h_;
  }
  int fps() const {
    return fps_;
  }
  const std::shared_ptr<std::vector<uint8_t>>& nv12_enc() const {
    return nv12_enc_;
  }

private:
  std::string image_path_;
  int content_w_ = 0;
  int content_h_ = 0;
  int enc_w_ = 0;
  int enc_h_ = 0;
  int fps_ = 30;

  std::shared_ptr<std::vector<uint8_t>> nv12_enc_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node>
StillImageInput(std::string image_path, simaai::neat::StillImageInput::ContentWidth content_w,
                simaai::neat::StillImageInput::ContentHeight content_h,
                simaai::neat::StillImageInput::EncodeWidth enc_w,
                simaai::neat::StillImageInput::EncodeHeight enc_h,
                simaai::neat::StillImageInput::FramesPerSecond fps);
} // namespace simaai::neat::nodes

/**
 * @file
 * @ingroup nodes_io
 * @brief `StillImageInput` Node — source that emits a single image, typically as a freeze frame.
 *
 * Reads one image file from disk and presents it as a video source at a configured
 * framerate. Useful as a stand-in for a live camera (e.g. powering an RTSP server
 * with a fixed test pattern, or driving a Session that expects continuous video
 * input).
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Source Node that emits a single image as a continuous video stream.
 *
 * The image is loaded once at construction and re-emitted at the configured
 * framerate. Carries `InputRole::Source`, so the Session is driven with
 * `Run::run()`.
 *
 * @ingroup nodes_io
 */
class StillImageInput final : public Node, public OutputSpecProvider {
public:
  /// Strong-typed wrapper for the original content width, in pixels.
  struct ContentWidth {
    int value = 0; ///< Underlying integer value, in pixels.
    /// Default-construct with value 0.
    constexpr ContentWidth() = default;
    /// Construct from raw integer value.
    constexpr ContentWidth(int v) : value(v) {}
  };

  /// Strong-typed wrapper for the original content height, in pixels.
  struct ContentHeight {
    int value = 0; ///< Underlying integer value, in pixels.
    /// Default-construct with value 0.
    constexpr ContentHeight() = default;
    /// Construct from raw integer value.
    constexpr ContentHeight(int v) : value(v) {}
  };

  /// Strong-typed wrapper for the encoded output width, in pixels.
  struct EncodeWidth {
    int value = 0; ///< Underlying integer value, in pixels.
    /// Default-construct with value 0.
    constexpr EncodeWidth() = default;
    /// Construct from raw integer value.
    constexpr EncodeWidth(int v) : value(v) {}
  };

  /// Strong-typed wrapper for the encoded output height, in pixels.
  struct EncodeHeight {
    int value = 0; ///< Underlying integer value, in pixels.
    /// Default-construct with value 0.
    constexpr EncodeHeight() = default;
    /// Construct from raw integer value.
    constexpr EncodeHeight(int v) : value(v) {}
  };

  /// Strong-typed wrapper for the emit framerate, in frames per second.
  struct FramesPerSecond {
    int value = 30; ///< Underlying integer value, in fps.
    /// Default-construct with value 30 fps.
    constexpr FramesPerSecond() = default;
    /// Construct from raw integer value.
    constexpr FramesPerSecond(int v) : value(v) {}
  };

  /// Construct from an image path plus content / encode geometry and emit framerate.
  StillImageInput(std::string image_path, ContentWidth content_w, ContentHeight content_h,
                  EncodeWidth enc_w, EncodeHeight enc_h, FramesPerSecond fps);

  /// Type label for this Node kind.
  std::string kind() const override {
    return "StillImageInput";
  }
  /// User-facing label for this Node.
  std::string user_label() const override {
    return image_path_;
  }
  /// Role this Node plays as a stream source.
  InputRole input_role() const override {
    return InputRole::Source;
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }

  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;
  /// Negotiated downstream caps produced by this Node.
  OutputSpec output_spec(const OutputSpec& input) const override;

  /// Path to the still image file loaded at construction.
  const std::string& image_path() const {
    return image_path_;
  }
  /// Original content width, in pixels.
  int content_w() const {
    return content_w_;
  }
  /// Original content height, in pixels.
  int content_h() const {
    return content_h_;
  }
  /// Encoded output width, in pixels.
  int enc_w() const {
    return enc_w_;
  }
  /// Encoded output height, in pixels.
  int enc_h() const {
    return enc_h_;
  }
  /// Emit framerate, in frames per second.
  int fps() const {
    return fps_;
  }
  /// Pre-encoded NV12 buffer of the loaded image, sized to the encode geometry.
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
/// Convenience factory for a `StillImageInput` Node.
std::shared_ptr<simaai::neat::Node>
StillImageInput(std::string image_path, simaai::neat::StillImageInput::ContentWidth content_w,
                simaai::neat::StillImageInput::ContentHeight content_h,
                simaai::neat::StillImageInput::EncodeWidth enc_w,
                simaai::neat::StillImageInput::EncodeHeight enc_h,
                simaai::neat::StillImageInput::FramesPerSecond fps);
} // namespace simaai::neat::nodes

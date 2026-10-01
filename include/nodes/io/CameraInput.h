/**
 * @file
 * @ingroup nodes_io
 * @brief MIPI/libcamera camera source node wrapper.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Options for CameraInput, a live libcamera/MIPI source.
 *
 * The public contract is deliberately camera/frame oriented. Neat's private
 * camera memory bridge negotiates its allocator with libcamerasrc and passes
 * EV74 SiMaAI buffers through. When fallback is enabled, OS/libcamera buffers
 * may instead be copied into pooled EV74 SiMaAI memory and stamped with
 * GstSimaMeta. Users do not expose an OsToSima node.
 */
struct CameraInputOptions {
  // Optional libcamera camera-name, e.g. "imx477 5-001a" from `cam -l`.
  // Leave unset to let libcamera select its default camera.
  std::optional<std::string> camera_name;

  std::uint32_t width = 1920;
  std::uint32_t height = 1080;
  std::uint32_t framerate_num = 30;
  std::uint32_t framerate_den = 1;
  std::string format = "NV12";

  // Name used by downstream CVU/MLA configs through GstSimaMeta.
  std::string buffer_name = "camera";

  // Insert a small live-source queue by default to avoid unbounded camera backpressure.
  bool insert_queue = true;
  bool leaky_queue = true;
  std::uint32_t queue_depth = 2;

  // False (default): require strict camera/device zero-copy and fail if unavailable.
  // True: permit Neat's private adaptive bridge to copy non-SiMaAI buffers into
  // EV74 SiMaAI memory. This is an explicit compatibility escape hatch for
  // camera stacks without DMA-BUF export support.
  bool allow_cpu_fallback = false;
};

/** Explicit raw V4L2 capture backend. Use CameraInputOptions.format="RAW8"
 * and wire dimensions, not decoded image dimensions. Captures progressive,
 * single-plane GREY/BA81/GBRG/GRBG/RGGB bytes without ISP conversion.
 * The resulting UInt8 tensor is flat [bytesused], including row padding and
 * valid trailers. Negotiated geometry/stride/sizeimage are in Sample.caps_string.
 * Camera framerate options do not reconfigure this backend's device cadence.
 * Platform drivers require stop/release qualification before streaming.
 */
struct CameraV4L2Options {
  std::string device;
  std::string fourcc = "GREY";
  // Only false is supported: copy before requeue, with no DMA-BUF export/import.
  // True fails explicitly; it must never silently fall back to a copy.
  bool zero_copy = false;
  std::uint32_t capture_buffer_count = 8;
  // A retained output consumes one pooled buffer. Exhaustion applies cancellable
  // backpressure; it never overwrites tensors held by an application.
  std::uint32_t output_buffer_count = 8;
  std::uint32_t frame_timeout_ms = 2000;
};

class CameraInput final : public Node, public OutputSpecProvider {
public:
  explicit CameraInput(CameraInputOptions opt = {});

  std::string kind() const override {
    return "CameraInput";
  }
  std::string user_label() const override;
  InputRole input_role() const override {
    return InputRole::Source;
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  MemoryContract memory_contract() const override {
    return MemoryContract::PreferDeviceZeroCopy;
  }

  std::string buffer_name_hint(int node_index) const override;
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  const CameraInputOptions& options() const {
    return opt_;
  }
  std::string caps_string() const;

private:
  CameraInputOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> CameraInput(simaai::neat::CameraInputOptions opt = {});

/** Capture owned raw byte tensors through the explicit V4L2 copy backend.
 * Existing one-argument CameraInput behavior and options layout are unchanged.
 * SIMOR decoding, calibration and ROS publication remain application concerns.
 */
std::shared_ptr<simaai::neat::Node> CameraInput(simaai::neat::CameraInputOptions opt,
                                                simaai::neat::CameraV4L2Options backend);

/**
 * @brief Create a camera input with an application-owned capture queue minimum.
 *
 * @param opt Camera format and delivery options.
 * @param capture_buffer_count Minimum number of ISP-output buffers. Zero keeps
 *        the camera pipeline default. Neat's provider supports at most 128.
 */
std::shared_ptr<simaai::neat::Node>
CameraInputWithCaptureBuffers(simaai::neat::CameraInputOptions opt,
                              std::uint32_t capture_buffer_count);
} // namespace simaai::neat::nodes

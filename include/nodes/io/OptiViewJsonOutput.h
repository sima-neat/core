/**
 * @file
 * @ingroup nodes_io
 * @brief `OptiViewJsonOutput` — UDP sender for SiMa OptiView dashboard JSON payloads.
 *
 * Ships structured detection results to the SiMa OptiView dashboard over UDP. Each
 * channel binds a video port and a JSON port; the JSON port carries detection
 * metadata (bounding boxes, scores, labels) that the OptiView viewer overlays on
 * the video stream.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief A single detected object as the OptiView dashboard expects it.
 *
 * @ingroup nodes_io
 */
struct OptiViewObject {
  int x = 0;          ///< Bounding-box top-left x, in pixels.
  int y = 0;          ///< Bounding-box top-left y, in pixels.
  int w = 0;          ///< Bounding-box width, in pixels.
  int h = 0;          ///< Bounding-box height, in pixels.
  float score = 0.0f; ///< Detection confidence in `[0, 1]`.
  int class_id = -1;  ///< Class index into the labels table. `-1` = unknown.
};

/**
 * @brief Channel binding for an OptiView UDP destination.
 *
 * The viewer expects the video and JSON ports to share a base offset per channel
 * (e.g. channel `0` uses `9000` / `9100`, channel `1` uses `9001` / `9101`).
 *
 * @ingroup nodes_io
 */
struct OptiViewChannelOptions {
  std::string host = "127.0.0.1"; ///< OptiView dashboard host.
  int channel = 0;                ///< Channel index added to each port base.
  int video_port_base = 9000;     ///< First video UDP port; actual port is `base + channel`.
  int json_port_base = 9100;      ///< First JSON UDP port; actual port is `base + channel`.
};

/// Default class-label table used when the application doesn't provide one.
std::vector<std::string> OptiViewDefaultLabels();

/// Render a detection result as an OptiView-shaped JSON string.
std::string OptiViewMakeJson(int64_t timestamp_ms, const std::string& frame_id,
                             const std::vector<OptiViewObject>& objects,
                             const std::vector<std::string>& labels);

/**
 * @brief UDP JSON sender for the SiMa OptiView dashboard.
 *
 * Construct one per output channel; use `send_detection()` per inference result.
 * Cross-reference the OptiView dashboard for the expected payload shape and
 * port-pairing convention.
 *
 * @ingroup nodes_io
 */
class OptiViewJsonOutput {
public:
  /// Construct a sender bound to the channel; `err` receives a setup error if any.
  explicit OptiViewJsonOutput(const OptiViewChannelOptions& opt, std::string* err = nullptr);
  /// Destructor; closes the underlying UDP socket.
  ~OptiViewJsonOutput();
  /// Deleted copy constructor; the sender owns a socket and is non-copyable.
  OptiViewJsonOutput(const OptiViewJsonOutput&) = delete;
  /// Deleted copy assignment; the sender owns a socket and is non-copyable.
  OptiViewJsonOutput& operator=(const OptiViewJsonOutput&) = delete;
  /// Move constructor; transfers ownership of the underlying socket.
  OptiViewJsonOutput(OptiViewJsonOutput&&) noexcept;
  /// Move assignment; transfers ownership of the underlying socket.
  OptiViewJsonOutput& operator=(OptiViewJsonOutput&&) noexcept;

  /// True if the underlying UDP socket was set up successfully.
  bool ok() const;
  /// Destination host this sender is bound to.
  const std::string& host() const;
  /// Resolved JSON UDP port (`json_port_base + channel`).
  int json_port() const;
  /// Resolved video UDP port (`video_port_base + channel`).
  int video_port() const;

  /// Send a pre-rendered JSON payload to the JSON port.
  bool send_json(const std::string& payload, std::string* err = nullptr) const;

  /// Render a detection set with `OptiViewMakeJson` and ship it to the JSON port.
  bool send_detection(int64_t timestamp_ms, const std::string& frame_id,
                      const std::vector<OptiViewObject>& objects,
                      const std::vector<std::string>& labels, std::string* err = nullptr) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat

/**
 * @file
 * @ingroup nodes_groups
 * @brief OptiView fan-out helpers — multi-stream UDP H.264 video plus per-frame JSON results.
 *
 * Hosts two cooperating NodeGroup-style classes used by demos and OptiView dashboards:
 * `UdpOutputNodeGroup` opens N parallel UDP H.264 video sinks (one per camera), and
 * `OptiViewOutputNodeGroup` adds a matching set of OptiView-formatted JSON sinks that
 * carry the detection results alongside the video. Typical placement: tail of a
 * multi-stream detection Session that needs to ship video and metadata to an external
 * OptiView viewer.
 *
 * @see OptiViewJsonOutput
 * @see UdpH264OutputGroup
 */
#pragma once

#include "nodes/io/OptiViewJsonOutput.h"
#include "pipeline/SessionOptions.h"
#include "pipeline/Run.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat::nodes::groups {

/**
 * @brief Configuration for `UdpOutputNodeGroup` — N parallel H.264-over-UDP video sinks.
 *
 * Specifies the H.264 caps, RTP packetization, and the UDP host plus port-base from
 * which per-stream destination ports are derived.
 *
 * @ingroup nodes_groups
 */
struct UdpOutputNodeGroupOptions {
  std::string h264_caps;          ///< Optional caps string applied to the H.264 elementary stream.
  int payload_type = 96;          ///< RTP payload type number for H.264.
  int config_interval = 1;        ///< SPS/PPS repeat interval (seconds).
  bool enable_timings = false;    ///< Attach timing metadata to outgoing buffers.
  std::string host = "127.0.0.1"; ///< Destination UDP host (shared across streams).
  int video_port_base = 9000;     ///< First UDP port; stream `i` is sent to `video_port_base + i`.
  bool udp_sync = false;          ///< Pass `sync` to the underlying `udpsink` element.
  bool udp_async = false;         ///< Pass `async` to the underlying `udpsink` element.
};

/**
 * @brief Manages N parallel H.264-over-UDP video output streams.
 *
 * Each call to `init()` constructs `streams` independent `Run`s, one per UDP video
 * sink. Callers `push_video()` rendered samples per stream index; `stop()` tears
 * them all down. Used as the video-side companion to `OptiViewOutputNodeGroup`.
 *
 * @see OptiViewOutputNodeGroup
 * @ingroup nodes_groups
 */
class UdpOutputNodeGroup {
public:
  /**
   * @brief Initialize and start `streams` parallel UDP H.264 output runs.
   * @param opt Shared output configuration.
   * @param streams Number of parallel video sinks to create.
   * @param err Optional out-parameter that receives a diagnostic message on failure.
   * @return True on success.
   */
  bool init(const UdpOutputNodeGroupOptions& opt, size_t streams, std::string* err = nullptr);

  /**
   * @brief Blocking push of a video sample into the indexed stream.
   * @param idx Stream index (must be `< size()`).
   * @param sample The video sample to send.
   */
  bool push_video(size_t idx, const simaai::neat::Sample& sample) const;

  /**
   * @brief Non-blocking push of a video sample; drops on full queue.
   * @param idx Stream index (must be `< size()`).
   * @param sample The video sample to send.
   */
  bool try_push_video(size_t idx, const simaai::neat::Sample& sample) const;

  /// Stop all underlying runs and release their resources.
  void stop();

  /// Number of active output streams.
  size_t size() const {
    return runs_.size();
  }

  /// Direct access to the underlying per-stream `Run` handles.
  const std::vector<std::shared_ptr<simaai::neat::Run>>& runs() const {
    return runs_;
  }

private:
  UdpOutputNodeGroupOptions opt_{};
  std::vector<std::shared_ptr<simaai::neat::Run>> runs_;
};

/**
 * @brief Configuration for `OptiViewOutputNodeGroup`.
 *
 * Combines the underlying UDP-video options with OptiView-specific JSON tunables
 * (port base, frame geometry, top-K cap, optional class label table).
 *
 * @ingroup nodes_groups
 */
struct OptiViewOutputNodeGroupOptions {
  UdpOutputNodeGroupOptions udp{}; ///< Embedded options for the H.264 video side.
  bool send_json = true;           ///< If false, the JSON side is not initialized.
  int json_port_base = 9100;       ///< First JSON UDP port; stream `i` uses `json_port_base + i`.
  int frame_w = 0;                 ///< Frame width reported in the JSON (0 = unknown).
  int frame_h = 0;                 ///< Frame height reported in the JSON (0 = unknown).
  int topk = 100;                  ///< Maximum number of detections emitted per frame.
  bool parse_debug = false;        ///< Enable verbose parse-side diagnostics.
  int json_delay_ms = 0;           ///< Optional delay applied to JSON emission.
  int video_delay_ms = 0;          ///< Optional delay applied to video emission.
  std::vector<std::string> labels; ///< Class label table consulted when emitting JSON.
};

/**
 * @brief Per-frame inputs needed to emit one OptiView JSON record.
 *
 * Carries the per-stream identity, frame and timestamp metadata, plus pointers to
 * the YOLO-output sample (detections to be encoded) and the decoded image sample
 * (for shape/timestamp cross-reference).
 *
 * @ingroup nodes_groups
 */
struct OptiViewJsonInput {
  size_t stream_idx = 0;    ///< Index into the OptiView group's parallel streams.
  std::string stream_id;    ///< Logical stream identifier reported in the JSON.
  int64_t frame_id = -1;    ///< Source-side frame counter.
  int64_t capture_ms = -1;  ///< Capture timestamp in ms (-1 if unknown).
  int64_t yolo_ms = -1;     ///< Detector completion timestamp in ms (-1 if unknown).
  int output_frame_id = -1; ///< Output-side frame counter (-1 if unknown).
  const simaai::neat::Sample* yolo_sample = nullptr; ///< Detector output sample (boxes/scores).
  const simaai::neat::Sample* decoded_sample =
      nullptr; ///< Decoded image sample paired with the result.
};

/**
 * @brief Result of a single `emit_json()` call.
 *
 * @ingroup nodes_groups
 */
struct OptiViewJsonResult {
  bool ok = false;       ///< True if the JSON was sent without error.
  bool nonempty = false; ///< True if at least one detection was emitted.
  int boxes = 0;         ///< Number of detection boxes included in the message.
  std::string error;     ///< Diagnostic string when `ok == false`.
};

/**
 * @brief OptiView fan-out: parallel UDP H.264 video sinks plus matching JSON senders.
 *
 * Composes a `UdpOutputNodeGroup` for the video streams with one
 * `OptiViewJsonOutput` per stream. Use as the tail of a multi-stream detection
 * Session to broadcast both video and per-frame detection results to an external
 * OptiView viewer.
 *
 * @see OptiViewJsonOutput
 * @see UdpOutputNodeGroup
 * @ingroup nodes_groups
 */
class OptiViewOutputNodeGroup {
public:
  OptiViewOutputNodeGroup() = default;
  OptiViewOutputNodeGroup(const OptiViewOutputNodeGroup&) = delete;
  OptiViewOutputNodeGroup& operator=(const OptiViewOutputNodeGroup&) = delete;

  /**
   * @brief Initialize the video runs and the per-stream JSON senders.
   * @param opt OptiView fan-out configuration (video + JSON).
   * @param streams Number of parallel streams to provision.
   * @param err Optional out-parameter that receives a diagnostic message on failure.
   * @return True on success.
   */
  bool init(const OptiViewOutputNodeGroupOptions& opt, size_t streams, std::string* err = nullptr);

  /**
   * @brief Blocking push of a video sample into the indexed stream's H.264 sink.
   * @param idx Stream index.
   * @param sample The video sample to send.
   */
  bool push_video(size_t idx, const simaai::neat::Sample& sample) const;

  /**
   * @brief Non-blocking push of a video sample; drops on full queue.
   * @param idx Stream index.
   * @param sample The video sample to send.
   */
  bool try_push_video(size_t idx, const simaai::neat::Sample& sample) const;

  /**
   * @brief Build and send a single OptiView JSON record.
   * @param in Per-frame inputs (samples, ids, timestamps).
   * @param out Optional out-parameter receiving send status and statistics.
   * @return True on success.
   */
  bool emit_json(const OptiViewJsonInput& in, OptiViewJsonResult* out = nullptr) const;

  /// Stop all video and JSON senders and release their resources.
  void stop();

  /// Direct access to the underlying per-stream video `Run` handles.
  const std::vector<std::shared_ptr<simaai::neat::Run>>& video_runs() const {
    return udp_.runs();
  }

private:
  int64_t pick_timestamp_ms_(const OptiViewJsonInput& in) const;

  OptiViewOutputNodeGroupOptions opt_{};
  UdpOutputNodeGroup udp_;
  std::vector<std::unique_ptr<simaai::neat::OptiViewJsonOutput>> senders_;
};

} // namespace simaai::neat::nodes::groups

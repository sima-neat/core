/**
 * @file
 * @ingroup pipeline
 * @brief Session entry point and RTSP server handle.
 */
#pragma once

#include "builder/Node.h"
#include "builder/NodeGroup.h"
#include "pipeline/SessionOptions.h"
#include "pipeline/Run.h"
#include "builder/GraphPrinter.h"
#include "nodes/common/Output.h"
#include "nodes/common/Caps.h"
#include "nodes/common/FileInput.h"
#include "nodes/common/JpegDecode.h"
#include "nodes/common/VideoTrackSelect.h"
#include "nodes/common/Queue.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/common/VideoScale.h"
#include "nodes/io/StillImageInput.h"
#include "nodes/io/Input.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264Depacketize.h"
#include "nodes/sima/H264DecodeSima.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/H264Packetize.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <gst/gst.h>

namespace simaai::neat {

class RtspServerHandle {
public:
  RtspServerHandle() = default;
  ~RtspServerHandle();

  RtspServerHandle(const RtspServerHandle&) = delete;
  RtspServerHandle& operator=(const RtspServerHandle&) = delete;

  RtspServerHandle(RtspServerHandle&&) noexcept;
  RtspServerHandle& operator=(RtspServerHandle&&) noexcept;

  const std::string& url() const {
    return url_;
  }
  void stop();
  void kill() {
    stop();
  }
  bool running() const;

private:
  friend class Session;

  std::string url_;
  void* impl_ = nullptr;
  std::shared_ptr<void> guard_;
};

/**
 * @brief Build, validate, and run deterministic GStreamer pipelines.
 *
 * Session assembles Nodes/NodeGroups into a reproducible pipeline
 * string and returns Run handles for push/pull execution.
 */
class Session {
public:
  using TensorCallback = std::function<bool(const simaai::neat::Tensor&)>;

  explicit Session(const SessionOptions& opt = {});
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  Session(Session&&) noexcept;

  // Core: add a node (factory functions return std::shared_ptr<Node>)
  Session& add(std::shared_ptr<Node> node);
  Session& add(const NodeGroup& group);
  Session& add(NodeGroup&& group);

  // Explicit raw-string escape hatch (keeps "power user" obvious)
  Session& custom(std::string fragment);
  Session& custom(std::string fragment, InputRole role);

  // Typed runner: last node must be Output().
  void run();
  Sample run(const cv::Mat& input, const RunOptions& opt = {});
  Sample run(const simaai::neat::Tensor& input, const RunOptions& opt = {});
  Run build(const cv::Mat& input, RunMode mode = RunMode::Async, const RunOptions& opt = {});
  Run build(const simaai::neat::Tensor& input, RunMode mode = RunMode::Async,
            const RunOptions& opt = {});
  Run build(const Sample& input, RunMode mode = RunMode::Async, const RunOptions& opt = {});
  SessionReport validate(const ValidateOptions& opt, const cv::Mat& input) const;

  // Server-style run
  RtspServerHandle run_rtsp(const RtspServerOptions& opt);

  // Build + validate pipeline (no PLAYING). Returns machine-readable report.
  SessionReport validate(const ValidateOptions& opt = {}) const;

  // Tensor-friendly output helper (adds convert/scale/caps + Output).
  Session& add_output_tensor(const OutputTensorOptions& opt = {});

  // UX helpers: builder-only view and gst-launch string.
  std::string describe(const GraphPrinter::Options& opt = {}) const;
  std::string describe_backend(bool insert_boundaries = false) const;

  // Optional external guard (for externally managed runtimes).
  void set_guard(std::shared_ptr<void> guard);

  // Output callback for source pipelines (used by run()).
  void set_tensor_callback(TensorCallback cb);

  // Save/load pipeline config for reproducible runs.
  void save(const std::string& path) const;
  static Session load(const std::string& path);

  // Build pipeline once (no frames processed) and return an async runner.
  // Push pipelines use build(input) to configure appsrc caps.
  Run build(const RunOptions& opt = {});

  const std::string& last_pipeline() const {
    return last_pipeline_;
  }

private:
  struct BuiltState {
    GstElement* pipeline = nullptr;
    GstElement* sink = nullptr;
    std::shared_ptr<void> diag;
  };
  struct RunCache;
  struct GroupMeta {
    std::size_t start = 0;
    std::size_t end = 0; // exclusive
    NodeCapsBehavior caps_behavior = NodeCapsBehavior::Dynamic;
    std::string label;
  };

  std::vector<std::shared_ptr<Node>> nodes_;
  std::vector<GroupMeta> groups_;
  std::string last_pipeline_;
  std::shared_ptr<void> guard_;
  SessionOptions opt_{};
  TensorCallback tensor_cb_;
  std::string last_sima_manifest_json_;
  std::atomic<uint64_t> nodes_version_{0};
  std::unique_ptr<BuiltState> built_;
  std::unique_ptr<RunCache> run_cache_;
  uint64_t built_version_ = 0;

  void build_cached_source();
};

} // namespace simaai::neat

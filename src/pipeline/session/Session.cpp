// src/pipeline/Session.cpp

#include "pipeline/Session.h"
#include "SessionDetail.h"

#include "gst/GstInit.h"
#include "gst/GstParseLaunch.h"
#include "gst/GstBusWatch.h"
#include "gst/GstHelpers.h"

#include "pipeline/SessionError.h"
#include "pipeline/SessionReport.h"
#include "internal/InputStream.h"
#include "internal/InputStreamUtil.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/SimaaiGuard.h"
#include "pipeline/internal/SyncBuild.h"
#include "pipeline/internal/TensorUtil.h"
#include "builder/Node.h"
#include "builder/NodeGroup.h"
#include "builder/ConfigJsonProvider.h"
#include "builder/ConfigJsonConsumer.h"
#include "builder/NextCpuConfigurable.h"
#include "builder/OutputSpec.h"
#include "builder/GraphPrinter.h"
#include "contracts/ContractRegistry.h"
#include "contracts/Validators.h"
#include "pipeline/Tensor.h"
#include "pipeline/TensorAdapters.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264CapsFixup.h"

#include <gst/gst.h>
#include <gst/gstdebugutils.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/sdp/sdp.h>
#include <gst/video/video.h>
#include <glib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simaai::neat {

using simaai::neat::pipeline_internal::DiagCtx;

namespace {
std::atomic<std::uint64_t> g_pipeline_instance{0};

bool stop_trace_enabled() {
  const char* v = std::getenv("SIMA_STOP_TRACE");
  return v && *v && std::strcmp(v, "0") != 0;
}
} // namespace

Session::Session(const SessionOptions& opt) : opt_(opt) {
  if (opt_.element_name_suffix.empty()) {
    const std::uint64_t id = g_pipeline_instance.fetch_add(1) + 1;
    opt_.element_name_suffix = "_" + std::to_string(id);
  }
}

Session::~Session() {
  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] Session::~Session begin\n");
  }
  if (built_ && built_->sink) {
    gst_object_unref(built_->sink);
    built_->sink = nullptr;
  }
  if (built_ && built_->pipeline) {
    stop_and_unref(built_->pipeline);
    built_->pipeline = nullptr;
  }
  built_.reset();
  run_cache_.reset();
  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] Session::~Session end\n");
  }
}

Session::Session(Session&& other) noexcept {
  nodes_ = std::move(other.nodes_);
  groups_ = std::move(other.groups_);
  last_pipeline_ = std::move(other.last_pipeline_);
  guard_ = std::move(other.guard_);
  opt_ = other.opt_;
  tensor_cb_ = std::move(other.tensor_cb_);
  nodes_version_.store(other.nodes_version_.load(std::memory_order_relaxed),
                       std::memory_order_relaxed);
  built_ = std::move(other.built_);
  run_cache_ = std::move(other.run_cache_);
  built_version_ = other.built_version_;
}

Session& Session::add(std::shared_ptr<Node> node) {
  const std::size_t start = nodes_.size();
  nodes_.push_back(std::move(node));
  const std::size_t end = nodes_.size();
  if (end > start) {
    NodeCapsBehavior behavior = NodeCapsBehavior::Dynamic;
    if (nodes_.back())
      behavior = nodes_.back()->caps_behavior();
    groups_.push_back({start, end, behavior, ""});
  }
  nodes_version_.fetch_add(1, std::memory_order_relaxed);
  if (built_ && built_->sink) {
    gst_object_unref(built_->sink);
  }
  if (built_ && built_->pipeline) {
    stop_and_unref(built_->pipeline);
  }
  built_.reset();
  run_cache_.reset();
  return *this;
}

Session& Session::add(const NodeGroup& group) {
  const auto& gnodes = group.nodes();
  const std::size_t start = nodes_.size();
  nodes_.insert(nodes_.end(), gnodes.begin(), gnodes.end());
  const std::size_t end = nodes_.size();
  if (end > start) {
    groups_.push_back({start, end, group.caps_behavior(), ""});
  }
  nodes_version_.fetch_add(1, std::memory_order_relaxed);
  if (built_ && built_->sink) {
    gst_object_unref(built_->sink);
  }
  if (built_ && built_->pipeline) {
    stop_and_unref(built_->pipeline);
  }
  built_.reset();
  run_cache_.reset();
  return *this;
}

Session& Session::add(NodeGroup&& group) {
  const NodeCapsBehavior behavior = group.caps_behavior();
  auto& gnodes = group.nodes_mut();
  const std::size_t start = nodes_.size();
  nodes_.insert(nodes_.end(), std::make_move_iterator(gnodes.begin()),
                std::make_move_iterator(gnodes.end()));
  const std::size_t end = nodes_.size();
  gnodes.clear();
  if (end > start) {
    groups_.push_back({start, end, behavior, ""});
  }
  nodes_version_.fetch_add(1, std::memory_order_relaxed);
  if (built_ && built_->sink) {
    gst_object_unref(built_->sink);
  }
  if (built_ && built_->pipeline) {
    stop_and_unref(built_->pipeline);
  }
  built_.reset();
  run_cache_.reset();
  return *this;
}

Session& Session::custom(std::string fragment) {
  return add(nodes::Custom(std::move(fragment)));
}

Session& Session::custom(std::string fragment, InputRole role) {
  return add(nodes::Custom(std::move(fragment), role));
}

void Session::set_guard(std::shared_ptr<void> guard) {
  guard_ = std::move(guard);
  if (built_ && built_->sink) {
    gst_object_unref(built_->sink);
  }
  if (built_ && built_->pipeline) {
    stop_and_unref(built_->pipeline);
  }
  built_.reset();
  run_cache_.reset();
}

void Session::set_tensor_callback(TensorCallback cb) {
  tensor_cb_ = std::move(cb);
}

Session& Session::add_output_tensor(const OutputTensorOptions& opt) {
  OutputTensorOptions o = opt;
  if (o.format.empty())
    o.format = "RGB";
  if (o.dtype != TensorDType::UInt8) {
    throw std::runtime_error("add_output_tensor: only UInt8 is supported for now");
  }

  // Normalize to a CPU-friendly raw-video tensor path.
  add(nodes::VideoConvert());
  add(nodes::VideoScale());

  // Force SystemMemory to keep CPU-accessible tensors for future bindings.
  add(nodes::CapsRaw(o.format, o.target_width, o.target_height, o.target_fps,
                     simaai::neat::CapsMemory::SystemMemory));
  add(nodes::Output());
  return *this;
}

} // namespace simaai::neat

#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/Run.h"

#include <deque>
#include <functional>
#include <mutex>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct _GstElement;
typedef struct _GstElement GstElement;

namespace simaai::neat::pipeline_internal {

class SourceEvents {
public:
  void push(SourceEvent event) {
    std::lock_guard lock(mutex_);
    if (events_.size() == 256)
      events_.pop_front();
    events_.push_back(std::move(event));
  }
  std::optional<SourceEvent> pop() {
    std::lock_guard lock(mutex_);
    if (events_.empty())
      return {};
    auto event = std::move(events_.front());
    events_.pop_front();
    return event;
  }

private:
  std::mutex mutex_;
  std::deque<SourceEvent> events_;
};

struct SourceMonitor {
  std::shared_ptr<SourceEvents> events;
  std::string label;
  std::vector<std::string> branch_labels;
};

struct RtspReconnectSource {
  std::shared_ptr<SourceEvents> events;
  std::string label;
  std::string bin_name;
  std::string source_name;
  std::string boundary_name;
  int interval_sec = 0;
  int attempts = -1;
  std::function<void(const PullError&)> on_exhausted;
};

void attach_rtsp_reconnect(GstElement* pipeline, const std::vector<RtspReconnectSource>& sources);
bool start_rtsp_reconnect(GstElement* pipeline);
void stop_rtsp_reconnect(GstElement* pipeline);
std::optional<PullError> rtsp_reconnect_close_detail(GstElement* pipeline);
} // namespace simaai::neat::pipeline_internal

#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/Run.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct _GstElement;
typedef struct _GstElement GstElement;

namespace simaai::neat::pipeline_internal {
struct DiagCtx;

struct RtspReconnectSource {
  std::string bin_name;
  std::string source_name;
  std::string boundary_name;
  int timeout_ms = 0;
  int attempts = -1;
  std::function<void(const PullError&)> on_exhausted;
};

void attach_rtsp_reconnect(GstElement* pipeline, const std::vector<RtspReconnectSource>& sources,
                           const std::shared_ptr<DiagCtx>& diag);
bool start_rtsp_reconnect(GstElement* pipeline);
void stop_rtsp_reconnect(GstElement* pipeline);
std::optional<PullError> rtsp_reconnect_close_detail(GstElement* pipeline);
} // namespace simaai::neat::pipeline_internal

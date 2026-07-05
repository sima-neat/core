// include/pipeline/internal/GstDiagnosticsUtil.h
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/Diagnostics.h" // provides simaai::neat::pipeline_internal::DiagCtx
#include "pipeline/GraphReport.h"

#include <gst/gst.h>
#include <gst/video/video.h>

#include <optional>
#include <string>

namespace simaai::neat::pipeline_internal {

struct SampleHolder {
  GstSample* sample = nullptr;
  GstVideoInfo vinfo{};
  GstVideoFrame frame{};
  bool mapped = false;

  explicit SampleHolder(GstSample* s) : sample(s) {}
  ~SampleHolder() {
    if (mapped)
      gst_video_frame_unmap(&frame);
    if (sample)
      gst_sample_unref(sample);
  }

  SampleHolder(const SampleHolder&) = delete;
  SampleHolder& operator=(const SampleHolder&) = delete;
};

bool map_video_frame_read(SampleHolder& h, std::string& err);

// -----------------------------
// GStreamer stringify helpers
// -----------------------------
std::string gst_caps_to_string_safe(GstCaps* caps);
std::string gst_structure_to_string_safe(const GstStructure* st);
std::string gst_message_to_string(GstMessage* msg);

// Optional convenience for caps formatting
std::string caps_features_string(GstCaps* caps);

// -----------------------------
// DOT dump helper
// -----------------------------
void maybe_dump_dot(GstElement* pipeline, const std::string& tag);

// -----------------------------
// Diagnostics helpers
// -----------------------------
std::string boundary_summary(const std::shared_ptr<DiagCtx>& diag);
std::string stage_timing_summary(const std::shared_ptr<DiagCtx>& diag);
std::string element_timing_summary(const std::shared_ptr<DiagCtx>& diag);
std::string element_flow_summary(const std::shared_ptr<DiagCtx>& diag);

void drain_bus(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag,
               const char* where = nullptr, bool* eos_seen = nullptr);

void throw_if_bus_error(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag,
                        const char* where);

// -----------------------------
// Appsink polling helper
// Returns a GstSample* that the caller must gst_sample_unref().
// -----------------------------
std::optional<GstSample*> try_pull_sample_sliced(GstElement* pipeline, GstElement* appsink,
                                                 int timeout_ms,
                                                 const std::shared_ptr<DiagCtx>& diag,
                                                 const char* where, bool* eos_seen = nullptr);

// -----------------------------
// Teardown helper (best-effort, avoids deadlocks)
// SIMA_GST_TEARDOWN_TIMEOUT_MS: wait for NULL (ms)
// SIMA_GST_TEARDOWN_ASYNC: skip wait, defer to reaper
// SIMA_GST_TEARDOWN_REAPER_MS: retry interval (ms)
// SIMA_GST_TEARDOWN_DEFER_NO_FLUSH: defer no-flush teardowns to reaper
// -----------------------------
void stop_and_unref(GstElement*& e);
// Skip flush events during teardown (avoid gst_element_send_event deadlocks).
void stop_and_unref_no_flush(GstElement*& e, bool prefer_synchronous = false);

} // namespace simaai::neat::pipeline_internal

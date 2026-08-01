#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/InputStreamUtil.h"

#include <gst/gst.h>

#include <cstdio>

namespace simaai::neat::pipeline_internal::sample_timing_gst_detail {

// These helpers intentionally have internal linkage. They split the existing
// timing writer for terminal paths which already own a writable GstSimaMeta,
// without adding symbols to the shared-library ABI.
static inline bool apply_to_buffer_header(GstBuffer* buffer, const SampleTimingOverrides& timing) {
  if (!buffer) {
    return false;
  }

  const bool debug_timing = pipeline_internal::env_bool("SIMA_SAMPLE_TIMING_DEBUG", false);
  const GstClockTime before_pts = GST_BUFFER_PTS(buffer);
  const GstClockTime before_dts = GST_BUFFER_DTS(buffer);
  const GstClockTime before_dur = GST_BUFFER_DURATION(buffer);

  if (timing.pts_ns.has_value()) {
    GST_BUFFER_PTS(buffer) = static_cast<GstClockTime>(*timing.pts_ns);
  }
  if (timing.dts_ns.has_value()) {
    GST_BUFFER_DTS(buffer) = static_cast<GstClockTime>(*timing.dts_ns);
  }
  if (timing.duration_ns.has_value()) {
    GST_BUFFER_DURATION(buffer) = static_cast<GstClockTime>(*timing.duration_ns);
  }

  if (debug_timing) {
    std::fprintf(stderr,
                 "[SAMPLE_TIMING] write buffer=%p pts_valid=%d pts=%llu dts_valid=%d dts=%llu "
                 "dur_valid=%d dur=%llu before_pts=%llu after_pts=%llu before_dts=%llu "
                 "after_dts=%llu before_dur=%llu after_dur=%llu\n",
                 static_cast<void*>(buffer), timing.pts_ns.has_value() ? 1 : 0,
                 static_cast<unsigned long long>(timing.pts_ns.value_or(0)),
                 timing.dts_ns.has_value() ? 1 : 0,
                 static_cast<unsigned long long>(timing.dts_ns.value_or(0)),
                 timing.duration_ns.has_value() ? 1 : 0,
                 static_cast<unsigned long long>(timing.duration_ns.value_or(0)),
                 static_cast<unsigned long long>(before_pts),
                 static_cast<unsigned long long>(GST_BUFFER_PTS(buffer)),
                 static_cast<unsigned long long>(before_dts),
                 static_cast<unsigned long long>(GST_BUFFER_DTS(buffer)),
                 static_cast<unsigned long long>(before_dur),
                 static_cast<unsigned long long>(GST_BUFFER_DURATION(buffer)));
  }

  return true;
}

static inline bool write_to_structure(GstStructure* structure,
                                      const SampleTimingOverrides& timing) {
  if (!structure) {
    return false;
  }

  const gboolean pts_valid = timing.pts_ns.has_value() ? TRUE : FALSE;
  const gboolean dts_valid = timing.dts_ns.has_value() ? TRUE : FALSE;
  const gboolean duration_valid = timing.duration_ns.has_value() ? TRUE : FALSE;
  const gboolean frame_id_valid = timing.frame_id.has_value() ? TRUE : FALSE;
  gst_structure_set(
      structure, "sample-frame-id-valid", G_TYPE_BOOLEAN, frame_id_valid, "sample-frame-id",
      G_TYPE_INT64, static_cast<gint64>(timing.frame_id.value_or(0)), "sample-pts-valid",
      G_TYPE_BOOLEAN, pts_valid, "sample-pts-ns", G_TYPE_UINT64,
      static_cast<guint64>(timing.pts_ns.value_or(0)), "sample-dts-valid", G_TYPE_BOOLEAN,
      dts_valid, "sample-dts-ns", G_TYPE_UINT64, static_cast<guint64>(timing.dts_ns.value_or(0)),
      "sample-duration-valid", G_TYPE_BOOLEAN, duration_valid, "sample-duration-ns", G_TYPE_UINT64,
      static_cast<guint64>(timing.duration_ns.value_or(0)), nullptr);

  // Preserve the legacy SimaMeta timestamp field for existing plugins/tools,
  // while Sample reconstruction uses the explicit sample timing fields above.
  if (timing.pts_ns.has_value()) {
    gst_structure_set(structure, "timestamp", G_TYPE_UINT64, static_cast<guint64>(*timing.pts_ns),
                      nullptr);
  }
  return true;
}

} // namespace simaai::neat::pipeline_internal::sample_timing_gst_detail

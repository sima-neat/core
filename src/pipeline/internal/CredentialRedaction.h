#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <string>

namespace simaai::neat::pipeline_internal {

/// Remove credentials without truncating report-facing pipeline or diagnostic text: a URI's
/// `user:password@` becomes `<redacted>@`, and secret-bearing fields are masked. Plain string
/// work (GstErrorNormalizer.cpp), shared by both backends' texts; it calls nothing of GStreamer.
std::string redact_gst_credentials(std::string text);

} // namespace simaai::neat::pipeline_internal

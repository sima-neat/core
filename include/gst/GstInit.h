// include/gst/GstInit.h
/**
 * @file
 * @ingroup gst
 * @brief GStreamer initialization helpers.
 */
#pragma once

namespace simaai::neat {

// Thread-safe one-time gst_init().
void gst_init_once();

} // namespace simaai::neat

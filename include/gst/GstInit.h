// include/gst/GstInit.h
/**
 * @file
 * @ingroup gst
 * @brief Thread-safe, idempotent GStreamer initialization.
 *
 * Wraps the underlying `gst_init()` call with a `std::call_once` so that the framework
 * (and any nested user code) can ask for initialization unconditionally without risking
 * double-init or races. Every framework code path that touches GStreamer goes through
 * this helper.
 */
#pragma once

namespace simaai::neat {

/**
 * @brief Initialize GStreamer exactly once across the process.
 *
 * Safe to call from multiple threads and multiple times — only the first call performs
 * the actual `gst_init()`; subsequent calls return immediately.
 */
void gst_init_once();

} // namespace simaai::neat

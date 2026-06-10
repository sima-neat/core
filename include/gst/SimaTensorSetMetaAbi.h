/**
 * @file
 * @ingroup gst
 * @brief Shim for the tensor-set GstMeta ABI that SiMa attaches to GStreamer buffers
 * carrying tensor data.
 *
 * The real `GstSimaMeta` struct definition lives in the installed internals SDK. This
 * header `#include_next`s it so the framework's GStreamer plugins and consumers can
 * read tensor-set metadata through a stable include path. Treat as a pinned ABI.
 */
#pragma once

#if defined(__has_include)
#if __has_include(<simaai/gst/SimaTensorSetMetaAbi.h>)
#include <simaai/gst/SimaTensorSetMetaAbi.h>
#elif defined(__GNUC__) || defined(__clang__)
#include_next <gst/SimaTensorSetMetaAbi.h>
#else
#error "SimaTensorSetMetaAbi.h must be provided by the installed internals SDK."
#endif
#elif defined(__GNUC__) || defined(__clang__)
#include_next <gst/SimaTensorSetMetaAbi.h>
#else
#error "SimaTensorSetMetaAbi.h must be provided by the installed internals SDK."
#endif

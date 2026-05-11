/**
 * @file
 * @ingroup gst
 * @brief Helpers for watching a GStreamer pipeline's message bus.
 *
 * Provides a small set of free functions used by Run/Session to drain bus messages and
 * surface them into the framework's reporting layer (SessionReport entries, SessionError
 * exceptions). Forward-declares the GStreamer C types so this header doesn't drag the
 * full `<gst/gst.h>` into framework public headers.
 *
 * @see SessionReport
 * @see SessionError
 */
#pragma once

#include <string>

struct _GstElement;
struct _GstMessage;
/// Forward declaration of GStreamer's GstElement struct.
typedef struct _GstElement GstElement;
/// Forward declaration of GStreamer's GstMessage struct.
typedef struct _GstMessage GstMessage;

namespace simaai::neat {

/// Callback invoked for every bus message drained.
/// @param type GStreamer message type as a stable string (e.g., `"warning"`, `"info"`).
/// @param src Name of the element that posted the message, or empty.
/// @param line Pre-formatted single-line summary of the message.
/// @param user_data Opaque pointer the caller registered.
using BusMessageFn = void (*)(const char* type, const char* src, const std::string& line,
                              void* user_data);

/// Callback invoked when a bus error message is encountered.
using BusErrorFn = void (*)(const std::string& line, void* user_data);

/**
 * @brief Format a single GStreamer bus message into a stable, one-line string.
 * @param msg The bus message to format. Must be non-null.
 * @return Human-readable summary suitable for logs and SessionReport.
 */
std::string gst_message_to_string(GstMessage* msg);

/**
 * @brief Drain pending messages from the pipeline's bus, invoking `on_message` for each.
 *
 * Non-blocking: pulls all messages currently posted, then returns. Safe to call from
 * the framework's polling loop while the pipeline is running.
 *
 * @param pipeline Pipeline whose bus to drain.
 * @param on_message Callback invoked for each message (may be null).
 * @param user_data Opaque pointer forwarded to `on_message`.
 */
void drain_bus(GstElement* pipeline, BusMessageFn on_message, void* user_data);

/**
 * @brief Drain the bus and throw a SessionError if any error message is found.
 *
 * Drains exactly like `drain_bus` but additionally invokes `on_error` and raises a
 * SessionError if any `GST_MESSAGE_ERROR` is observed.
 *
 * @param pipeline Pipeline whose bus to drain.
 * @param on_message Callback invoked for every message (may be null).
 * @param user_data Opaque pointer forwarded to `on_message`.
 * @param on_error Callback invoked once for the error line before throwing (may be null).
 * @param error_user_data Opaque pointer forwarded to `on_error`.
 */
void throw_if_bus_error(GstElement* pipeline, BusMessageFn on_message, void* user_data,
                        BusErrorFn on_error, void* error_user_data);

} // namespace simaai::neat
